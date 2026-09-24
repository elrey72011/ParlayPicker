"""Read-only canonical views of the three pre-existing research stores.

These projections retain source identity and missing facts.  Reading them never
copies research rows into the new validation store or grants wager authority.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

from app_core.mlb_spread_total_model import digest, identity, label, receipt_features


CANONICAL_FIELDS = (
    "observation_id", "sport", "market_family", "game_id", "provider_namespace",
    "provider_event_id", "home_team", "away_team", "home_team_id", "away_team_id",
    "scheduled_start", "selection", "line", "american_odds", "decimal_odds",
    "sportsbook", "quote_timestamp", "quote_source", "quote_verified", "model_id",
    "model_version", "model_trained_through", "model_available_at", "feature_version",
    "feature_snapshot_id", "feature_frozen_at", "calibration_id",
    "calibration_version", "calibration_available_at", "policy_version",
    "prediction_timestamp", "mean_probability", "conservative_probability",
    "push_probability", "loss_probability", "probability_semantics",
    "evidence_snapshot_id", "evidence_hash", "runtime_hash", "source_commit",
)


def _json(raw):
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite legacy evidence")))


def _records(path, table):
    """Read only existing SQLite bytes; a missing file stays missing."""
    path = Path(path)
    if not path.is_file():
        return []
    if table not in {"records", "receipts", "outcomes", "observations"}:
        raise ValueError("unsupported legacy table")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        try:
            if table == "records":
                rows = db.execute("SELECT id,payload FROM records ORDER BY rowid").fetchall()
            else:
                rows = db.execute(f"SELECT id,sha256,payload FROM {table} ORDER BY rowid").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                raise ValueError("legacy evidence table missing") from None
            raise
    result = []
    for row in rows:
        key, raw = row[0], row[-1]
        payload = _json(raw)
        expected = hashlib.sha256(raw.encode()).hexdigest() if table == "records" else digest(payload)
        if expected != (key if table == "records" else row[1]):
            raise ValueError("legacy evidence hash mismatch")
        result.append((key, payload))
    return result


def _time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _family(sport, market):
    if not isinstance(market, str):
        return None
    if market.startswith("total") or market == "totals":
        return "TOTAL"
    if market.startswith("spread") or market == "spreads":
        return {"MLB": "RUN_LINE", "NHL": "PUCK_LINE"}.get(sport, "SPREAD")
    return None


def _row(sport, family, source_id, selection, **facts):
    row = {field: None for field in CANONICAL_FIELDS}
    row.update(sport=sport, market_family=family, selection=selection,
               observation_id=digest({"source": source_id, "selection": selection,
                                      "model": facts.get("model_id"),
                                      "model_name": facts.get("legacy_model_name"),
                                      "market": family, "book": facts.get("sportsbook"),
                                      "line": facts.get("line"),
                                      "quote_at": facts.get("quote_timestamp")}),
               source_record_id=source_id, source_store="legacy", legacy_research_only=True,
               production_eligible=False, recommended_stake=0.0,
               result_outcome=None, result_available_at=None, result_source=None,
               result_home_score=None, result_away_score=None)
    row.update(facts)
    blockers = ["LEGACY_RESEARCH_NOT_VALIDATION_EVIDENCE"]
    for key, code in (("home_team_id", "MISSING_STABLE_TEAM_IDS"),
                      ("model_id", "MISSING_MARKET_MODEL"),
                      ("model_version", "MISSING_MODEL_VERSION"),
                      ("calibration_id", "MISSING_CALIBRATION"),
                      ("conservative_probability", "MISSING_CONSERVATIVE_PROBABILITY")):
        if row[key] is None:
            blockers.append(code)
    if row["quote_verified"] is not True:
        blockers.append("QUOTE_SOURCE_NOT_REPLAYABLY_VERIFIED")
    if row["result_available_at"] is None:
        blockers.append("MISSING_AVAILABLE_RESULT")
    if row["prediction_timestamp"] is None:
        blockers.append("MISSING_PREGAME_PREDICTION")
    elif (_time(row["prediction_timestamp"]) is None or _time(row["scheduled_start"]) is None
          or _time(row["prediction_timestamp"]) >= _time(row["scheduled_start"])):
        blockers.append("PREDICTION_CHRONOLOGY_INVALID")
    blockers.extend(("MISSING_SPORT_MARKET_VALIDATION_ARTIFACT", "MISSING_OWNER_ACTIVATION"))
    row["blockers"] = tuple(blockers)
    return row


def _mlb_quote_bound(source, quote):
    """Check the retained Odds API response against the selected exact offer."""
    if (source.get("source") != "odds_api"
            or source.get("payload", {}).get("id") != quote.get("provider_event_id")
            or source.get("observed_at") != quote.get("observed_at")):
        return False
    from app_core.prediction_evidence import provider_quotes
    from app_core.public_quote_policy import canonical_book_label
    try:
        offered = _json(provider_quotes(source["payload"]))
        return any(
            candidate.get("provider_event_id") == quote["provider_event_id"]
            and candidate.get("market_type") == quote["market_type"]
            and candidate.get("point") == quote["line"]
            and candidate.get("recorded_at") == quote.get("provider_updated_at")
            and canonical_book_label(candidate.get("book")) == quote["sportsbook"]
            and abs(_decimal(candidate.get("price")) - float(quote["decimal_odds"])) < 1e-9
            for candidate in offered
        )
    except (TypeError, ValueError, KeyError):
        return False


def _decimal(american):
    value = float(american)
    if not value.is_integer() or abs(value) < 100:
        raise ValueError("invalid american price")
    return 1 + (value / 100 if value > 0 else 100 / abs(value))


def _mlb(path):
    outcomes = {key: value for key, value in _records(path, "outcomes")}
    observations = {key: value for key, value in _records(path, "observations")}
    result = []
    for source_id, snapshot in _records(path, "receipts"):
        payload, _ = receipt_features(snapshot)
        q = payload["quote"]
        family = _family("MLB", q["market_type"])
        if family is None:
            continue
        schedule = observations.get(payload.get("source_observations", {}).get("schedule"), {})
        names = {}
        for day in schedule.get("payload", {}).get("dates", []):
            for game in day.get("games", []):
                if str(game.get("gamePk")) == str(payload["provider_event_id"]):
                    names = {side: game["teams"][side]["team"].get("name") for side in ("home", "away")}
        quote_source = observations.get(payload.get("source_observations", {}).get("quotes"), {})
        quote_bound = _mlb_quote_bound(quote_source, q)
        event_key = digest({"provider_namespace": "mlb", "provider_event_id": payload["provider_event_id"]})
        outcome = outcomes.get(event_key)
        try:
            if outcome and (identity(outcome) != identity(payload)
                            or _time(outcome.get("available_at")) < _time(payload["game_start_utc"])):
                raise ValueError("outcome identity or chronology mismatch")
            settled = label(q["market_type"], q["line"], outcome["home_score"],
                            outcome["away_score"], outcome["status"]) if outcome else None
        except (KeyError, TypeError, ValueError):
            settled = "NEEDS_REVIEW"
        baseline = payload.get("baselines", {}).get("deterministic", {})
        row = _row("MLB", family, source_id, q["market_type"],
                   game_id=payload["provider_event_id"], provider_namespace="mlb",
                   provider_event_id=payload["provider_event_id"],
                   home_team=names.get("home"), away_team=names.get("away"),
                   home_team_id=payload["home_team_id"], away_team_id=payload["away_team_id"],
                   scheduled_start=payload["game_start_utc"], line=q["line"],
                   decimal_odds=q["decimal_odds"], sportsbook=q["sportsbook"],
                   quote_timestamp=q["observed_at"], quote_source="odds_api",
                   quote_verified=quote_bound, policy_version=payload["schema_version"],
                   prediction_timestamp=baseline.get("generated_at"),
                   mean_probability=baseline.get("probability"),
                   probability_semantics=baseline.get("probability_semantics"),
                   evidence_snapshot_id=snapshot["sha256"], evidence_hash=snapshot["sha256"],
                   feature_version="mlb-prior-scoring-line-v1",
                   feature_snapshot_id=source_id, feature_frozen_at=payload["captured_at"],
                   result_outcome=settled,
                   result_available_at=outcome.get("available_at") if outcome else None,
                   result_source="mlb_statsapi" if outcome else None,
                   result_home_score=outcome.get("home_score") if outcome else None,
                   result_away_score=outcome.get("away_score") if outcome else None)
        result.append(row)
    return result


def _ncaaf(path):
    records = _records(path, "records")
    models = {key: value for key, value in records if value.get("kind") == "model"}
    scores = {}
    for key, record in records:
        if record.get("kind") == "scores":
            for score in record.get("data", {}).get("scores", []):
                scores.setdefault(str(score["cfbd_id"]), (record["created_at"], score))
    result = []
    for source_id, record in records:
        if record.get("kind") != "capture":
            continue
        data = record["data"]
        model = models.get(data.get("model_id"))
        for event in data.get("events", []):
            score_pair = scores.get(str(event.get("cfbd_id")))
            for model_name, prediction in event.get("models", {}).items():
                quote = prediction.get("selected", {})
                market = quote.get("market_type")
                family = _family("NCAAF", market)
                if family is None:
                    continue
                score = score_pair[1] if score_pair else None
                settled = None
                if score and (score.get("home_id"), score.get("away_id")) == (event.get("home_id"), event.get("away_id")):
                    margin = score["home_score"] - score["away_score"]
                    delta = (score["home_score"] + score["away_score"] - quote["point"]
                             if market == "total_over" else quote["point"] - score["home_score"] - score["away_score"]
                             if market == "total_under" else (margin if market == "spread_home" else -margin) + quote["point"])
                    settled = "WIN" if delta > 0 else "LOSS" if delta < 0 else "PUSH"
                result.append(_row("NCAAF", family, source_id, market,
                    game_id=str(event.get("cfbd_id")), provider_namespace="odds_api",
                    provider_event_id=event.get("event_id"),
                    home_team=event.get("home"), away_team=event.get("away"),
                    home_team_id=event.get("home_id"), away_team_id=event.get("away_id"),
                    scheduled_start=event.get("start"), line=quote.get("point"),
                    american_odds=quote.get("price"), decimal_odds=quote.get("decimal_odds"),
                    sportsbook=quote.get("book"), quote_timestamp=quote.get("recorded_at"),
                    quote_source="legacy_odds_api_derived", quote_verified=False,
                    model_id=data.get("model_id") if model else None,
                    model_available_at=model.get("created_at") if model else None,
                    feature_snapshot_id=digest(event.get("features")),
                    feature_frozen_at=data.get("captured_at"),
                    prediction_timestamp=data.get("captured_at"),
                    mean_probability=quote.get("win"), push_probability=quote.get("push"),
                    loss_probability=quote.get("loss"),
                    evidence_snapshot_id=source_id, evidence_hash=source_id,
                    runtime_hash=model.get("data", {}).get("runtime_hash") if model else None,
                    legacy_model_name=model_name,
                    result_outcome=settled,
                    result_available_at=score_pair[0] if settled else None,
                    result_source="legacy_cfbd_score" if settled else None,
                    result_home_score=score.get("home_score") if settled else None,
                    result_away_score=score.get("away_score") if settled else None))
    if any(record.get("kind") == "closing" for _, record in records):
        from app_core.ncaaf_closing import report as closing_report
        try:
            proxies = {(str(item["model_id"]), str(item["game_id"]), item["model"],
                        item["market"], item["book"]): item
                       for item in closing_report(path)["rows"]}
        except (KeyError, TypeError, ValueError):
            proxies = {}
        for row in result:
            key = (str(row["model_id"]), str(row["game_id"]), row["legacy_model_name"],
                   row["selection"], row["sportsbook"])
            proxy = proxies.get(key)
            row["close_status"] = proxy.get("status") if proxy else "NO_COMPARABLE_CLOSING_PROXY"
            row["closing_proxy_price_clv"] = proxy.get("price_clv") if proxy else None
            row["verified_clv"] = None
    else:
        for row in result:
            row.update(close_status="NO_COMPARABLE_CLOSING_PROXY",
                       closing_proxy_price_clv=None, verified_clv=None)
    return result


def _nfl(path):
    records = _records(path, "records")
    scores = {str(value["data"]["event_id"]): (value["created_at"], value["data"])
              for _, value in records if value.get("kind") == "scores"}
    result = []
    for source_id, record in records:
        if record.get("kind") != "capture" or record.get("data", {}).get("protocol") != "nfl-market-v1":
            continue
        for event in record["data"].get("events", []):
            score_pair = scores.get(str(event.get("event_id")))
            for quote in event.get("quotes", []):
                family = _family("NFL", quote.get("market"))
                if family is None:
                    continue
                result.append(_row("NFL", family, source_id, quote.get("selection"),
                    game_id=event.get("event_id"), provider_namespace="odds_api",
                    provider_event_id=event.get("event_id"),
                    home_team=event.get("home"), away_team=event.get("away"),
                    scheduled_start=event.get("start"), line=quote.get("point"),
                    american_odds=quote.get("odds_american"), sportsbook=quote.get("book"),
                    quote_timestamp=quote.get("recorded_at"),
                    quote_source="legacy_odds_api_derived", quote_verified=False,
                    evidence_snapshot_id=source_id, evidence_hash=source_id,
                    result_outcome="NEEDS_REVIEW" if score_pair else None,
                    result_available_at=score_pair[0] if score_pair else None,
                    result_source="legacy_odds_api_score" if score_pair else None,
                    result_home_score=score_pair[1].get("home_score") if score_pair else None,
                    result_away_score=score_pair[1].get("away_score") if score_pair else None))
                result[-1]["blockers"] += (
                    "MISSING_NFL_MODEL_RUNTIME_HASH", "MISSING_NFL_INJURY_CONTEXT",
                    "MISSING_NFL_RECENT_RESULT_CONTEXT", "NO_VALID_CLOSE_QUOTES",
                    "MISSING_NFL_MARKET_SETTLEMENT_RULES")
    return result


def legacy_evidence(sport, path):
    """Project only verified on-disk source records; never create or migrate rows."""
    if sport not in {"MLB", "NCAAF", "NFL"}:
        raise ValueError("unsupported legacy sport")
    return {"MLB": _mlb, "NCAAF": _ncaaf, "NFL": _nfl}[sport](path)
