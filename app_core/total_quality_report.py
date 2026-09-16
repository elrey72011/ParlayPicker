"""Read-only prospective diagnostics. Never relabel, train, or change gates."""
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from io import StringIO
from pathlib import Path
import pandas as pd
from core.exposure_ledger import digest
from core.wager_decisions import aware, finite, decimal_price
from app_core.total_signal_quality import VERSION


def read_candidates(database):
    from app_core.candidate_recap import _grade_candidate
    output, closes = [], []
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        for sid, candidates, decisions, inputs, expected in db.execute("SELECT snapshot_id,candidates,decisions,inputs,payload_hash FROM snapshots ORDER BY generated_at"):
            if hashlib.sha256("\0".join([candidates,decisions,inputs]).encode()).hexdigest() != expected:
                raise ValueError("Prediction payload changed")
            audit = pd.read_csv(StringIO(candidates))
            if "total_input_version" not in audit:
                continue  # Never retrospectively label legacy predictions.
            audit = audit[audit.total_input_version.eq(VERSION)].copy()
            if audit.empty:
                continue
            revisions = db.execute("SELECT scores,recorded_at,evidence_hash FROM score_revisions WHERE snapshot_id=? ORDER BY recorded_at,evidence_hash", (sid,)).fetchall()
            if any(hashlib.sha256(raw.encode()).hexdigest() != h for raw,_,h in revisions):
                raise ValueError("Score revision changed")
            if revisions:
                scores = pd.concat([pd.read_csv(StringIO(raw)) for raw,_,_ in revisions], ignore_index=True).drop_duplicates("matchup_id", keep="last")
                audit = audit.merge(scores, on="matchup_id", how="left", validate="many_to_one")
            else:
                audit["actual_home_score"] = None
                audit["actual_away_score"] = None
            audit["candidate_outcome"] = audit.apply(_grade_candidate, axis=1)
            output.extend(audit.to_dict("records"))
        for identity, raw in db.execute("SELECT observation_id,payload FROM closing_observations"):
            value = json.loads(raw)
            if digest(value) != identity:
                raise ValueError("Closing payload changed")
            closes.append(value)
    return output, closes


def summarize(rows, closes=()):
    eligible = []
    for row in rows:
        generated, start = aware(row.get("prediction_generated_at")), aware(row.get("game_start_utc"))
        if (row.get("total_input_version") == VERSION and row.get("total_input_status") in {"COMPLETE","DEGRADED","INCOMPLETE"}
            and row.get("market_type") in {"total_over","total_under"} and generated and start and generated < start):
            eligible.append(row)
    latest_close = {}
    for close in closes:
        if close.get("quote_verified") is not True or aware(close.get("closing_capture_at")) is None:
            continue
        key = (close["snapshot_id"],close["candidate_id"])
        if key not in latest_close or aware(close["closing_capture_at"]) > aware(latest_close[key]["closing_capture_at"]):
            latest_close[key] = close
    groups = []
    for status in ("COMPLETE","DEGRADED","INCOMPLETE"):
        for direction in ("all","total_over","total_under"):
            cohort = [r for r in eligible if r["total_input_status"] == status and (direction == "all" or r["market_type"] == direction)]
            count = lambda o: sum(r.get("candidate_outcome") == o for r in cohort)
            w,l,push = count("WIN"),count("LOSS"),count("PUSH")
            brier, logs, returns, line_clv, price_clv = [],[],[],[],[]
            for row in cohort:
                outcome = row.get("candidate_outcome")
                p = finite(row.get("calibrated_probability"))
                if p is not None and 0 <= p <= 1 and outcome in {"WIN","LOSS"}:
                    y = int(outcome == "WIN")
                    brier.append((p-y)**2)
                    bounded = min(1-1e-15,max(1e-15,p))
                    logs.append(-math.log(bounded if y else 1-bounded))
                price = decimal_price(row.get("odds_american"))
                if price is not None and outcome in {"WIN","LOSS","PUSH"}:
                    returns.append(price-1 if outcome == "WIN" else -1 if outcome == "LOSS" else 0)
                close = latest_close.get((row.get("snapshot_id"),row.get("candidate_id")))
                if close and valid_close(row, close):
                    for field, values in (("line_clv",line_clv),("price_clv",price_clv)):
                        value = finite(close.get(field))
                        if value is not None:
                            values.append(value)
            mean = lambda values: sum(values)/len(values) if values else None
            groups.append(dict(status=status,direction=direction,sample_size=len(cohort),wins=w,losses=l,pushes=push,
                               pending=len(cohort)-w-l-push,win_rate=w/(w+l) if w+l else None,
                               probability_sample=len(brier),brier=mean(brier),log_loss=mean(logs),
                               priced_settled=len(returns),roi=mean(returns),
                               line_clv_sample=len(line_clv),line_clv=mean(line_clv),price_clv_sample=len(price_clv),price_clv=mean(price_clv)))
    return dict(version=VERSION,cohorts=groups,
                interpretation="Descriptive candidate snapshots, not independent wagers. Repeated games and opposite directions overlap. No superiority, validation, or promotion claim; no automatic gate. Missing probabilities and genuine closes remain unavailable.")


def valid_close(row, close):
    """Recheck recorded close provenance; never substitute a proxy closing line."""
    from app_core.public_quote_policy import canonical_book_label
    from core.clv import line_clv, price_clv
    quote = close.get("quote", {})
    start, at, capture = aware(row.get("game_start_utc")), aware(quote.get("quote_recorded_at")), aware(close.get("closing_capture_at"))
    if None in (start, at, capture) or not at <= capture < start or not 0 < (start-at).total_seconds() <= 1800 or (capture-at).total_seconds() > 1800:
        return False
    if row.get("provider_namespace") not in {"mlb","espn","odds_api"} or row.get("provider_namespace") != quote.get("provider_namespace"):
        return False
    for key in ("game_id","sport","market_type","provider_event_id"):
        if not row.get(key) or row[key] != quote.get(key):
            return False
    book = canonical_book_label(row.get("quote_bookmaker"))
    if not book or book != canonical_book_label(quote.get("sportsbook")):
        return False
    opening, closing = finite(row.get("market_line_used")), finite(quote.get("line"))
    if opening is None or closing is None or decimal_price(quote.get("price")) is None:
        return False
    return (close.get("line_clv") == line_clv(row["market_type"],opening,closing)
            and close.get("price_clv") == (price_clv(row.get("odds_american"),quote["price"]) if opening == closing else None))
