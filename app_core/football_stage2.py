"""Fail-closed, read-only Stage 2 football training audit and research models.

Only Stage 1's selected training manifest can supply training observations.
This module has no deployment, stake, parlay, or wager integration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3

from app_core import football_stage1 as stage1
from app_core import prospective_evidence as evidence

VERSION = "football-stage2-v1"
FEATURE_VERSION = "football-market-schedule-asof-v1"
SCOPES = tuple((sport, market) for sport in ("NFL", "NCAAF") for market in ("SPREAD", "TOTAL"))
CLASSES = {"SPREAD": ("COVER", "PUSH", "NO_COVER"),
           "TOTAL": ("OVER", "PUSH", "UNDER")}
MINIMUM = {"development": 200, "selection_validation": 60,
           "calibration_candidate": 60, "research_holdout": 60}
PARTITIONS = tuple(MINIMUM)
ARTIFACT_NAMES = {
    "training_audit": "football-stage2-training-audit.json",
    "feature_contract": "football-stage2-feature-contract.json",
    "split_plan": "football-stage2-split-plan.json",
    "baselines": "football-stage2-baselines.json",
    "model_comparison": "football-stage2-model-comparison.json",
    "model_inventory": "football-stage2-model-inventory.json",
    "calibration_readiness": "football-stage2-calibration-readiness.json",
    "research_prediction_readiness": "football-stage2-research-prediction-readiness.json",
}
BLOCKERS = ("NO_VERIFIED_PREGAME_PRICE", "QUOTE_TIMESTAMP_UNVERIFIED",
            "EVENT_IDENTITY_AMBIGUOUS", "LINE_UNVERIFIED", "RESULT_UNVERIFIED",
            "RESULT_LEAKAGE_RISK", "FEATURE_ASOF_UNAVAILABLE",
            "SETTLEMENT_UNREPRODUCIBLE", "SOURCE_LINEAGE_UNVERIFIED")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256((value if isinstance(value, bytes) else
                           (value if isinstance(value, str) else canonical(value)).encode())).hexdigest()


def _iso(value):
    return stage1.iso(value)


def _odds_probability(american):
    if type(american) is not int or american == 0 or -100 < american < 100:
        raise ValueError("NO_VERIFIED_PREGAME_PRICE")
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def exact_label(market, selection, line, home_score, away_score, home_team, away_team,
                *, provider_home_team=None, provider_away_team=None):
    """The target is the stored selection's exact betting result, including push."""
    if market not in CLASSES or not all(type(score) is int and score >= 0
                                         for score in (home_score, away_score)):
        raise ValueError("SETTLEMENT_UNREPRODUCIBLE")
    if not isinstance(line, (float, int)) or not math.isfinite(line):
        raise ValueError("LINE_UNVERIFIED")
    if market == "SPREAD":
        if (provider_home_team and provider_away_team and provider_home_team != provider_away_team):
            names = (provider_home_team, provider_away_team)
        else:
            names = (home_team, away_team)
        if selection not in names or names[0] == names[1]:
            raise ValueError("EVENT_IDENTITY_AMBIGUOUS")
        margin = (home_score-away_score if selection == names[0] else away_score-home_score) + line
        return "PUSH" if margin == 0 else "COVER" if margin > 0 else "NO_COVER"
    if selection not in ("Over", "Under"):
        raise ValueError("LINE_UNVERIFIED")
    delta = home_score + away_score - line
    return "PUSH" if delta == 0 else ("OVER" if delta > 0 else "UNDER")


def _spread_side(event, quote):
    raw = json.loads(quote["raw_source"]) if quote.get("raw_source") else {}
    if not isinstance(raw, dict):
        raise ValueError("SOURCE_LINEAGE_UNVERIFIED")
    return stage1._spread_side(dict(quote, sport=event["sport"],
                                    home_team=event["home_team"], away_team=event["away_team"],
                                    provider_home_team=raw.get("home_team"),
                                    provider_away_team=raw.get("away_team")))


def validate_feature_snapshot(snapshot, *, quote_at, kickoff, event_discovered_at, result_available_at):
    """Only quote and schedule fields frozen by prediction time are allowed."""
    quote_time, start = stage1.at(quote_at), stage1.at(kickoff)
    if not stage1.at(event_discovered_at) <= quote_time < start:
        raise ValueError("FEATURE_ASOF_UNAVAILABLE")
    if stage1.at(result_available_at) <= quote_time:
        raise ValueError("RESULT_LEAKAGE_RISK")
    allowed = {"line", "price_implied_probability", "neutral_site", "selection_is_home_or_over"}
    if set(snapshot) != allowed:
        raise ValueError("FEATURE_ASOF_UNAVAILABLE")
    for item in snapshot.values():
        if stage1.at(item["available_at"]) > quote_time or item["source"] not in ("verified_quote", "pregame_schedule"):
            raise ValueError("FEATURE_ASOF_UNAVAILABLE")
        if item.get("same_event_result") or item.get("season_end") or item.get("current_roster") or item.get("future_rating") or item.get("closing_price"):
            raise ValueError("RESULT_LEAKAGE_RISK")
        if not isinstance(item.get("value"), (int, float)) or not math.isfinite(item["value"]):
            raise ValueError("FEATURE_ASOF_UNAVAILABLE")


def _snapshot(event, quote, result):
    qtime = quote["observed_at"]
    schedule_time = event["discovered_at"]
    spread_side = _spread_side(event, quote) if quote["market_family"] == "SPREAD" else None
    if quote["market_family"] == "SPREAD" and spread_side is None:
        raise ValueError("EVENT_IDENTITY_AMBIGUOUS")
    snapshot = {
        "line": {"value": quote["line"], "source": "verified_quote", "available_at": qtime},
        "price_implied_probability": {"value": _odds_probability(quote["american_odds"]),
                                      "source": "verified_quote", "available_at": qtime},
        "neutral_site": {"value": int(event["neutral_site"]), "source": "pregame_schedule",
                         "available_at": schedule_time},
        "selection_is_home_or_over": {"value": int(spread_side == "home"
                                                    if quote["market_family"] == "SPREAD" else
                                                    quote["selection"] == "Over"),
                                      "source": "verified_quote", "available_at": qtime},
    }
    validate_feature_snapshot(snapshot, quote_at=qtime, kickoff=event["scheduled_start"],
                              event_discovered_at=schedule_time,
                              result_available_at=result["available_at"])
    return snapshot


def _selected_rows(db):
    return [dict(row) for row in db.execute("""
        SELECT m.*, e.version_id, e.season, e.season_type, e.home_team,
               e.away_team, e.neutral_site, e.discovered_at, e.scheduled_start,
               q.provider_last_update, q.quote_verified, q.identity_mapping_hash,
               q.capture_run_id, r.home_score, r.away_score, r.available_at AS result_available_at,
               r.result_status, s.outcome AS settlement_outcome,
               t.training_row_status, t.blockers AS training_blockers
        FROM prospective_football_training_manifest m
        JOIN prospective_football_quote q ON q.quote_id=m.quote_id
        JOIN prospective_football_event e ON e.version_id=q.event_version_id
        JOIN prospective_football_result r ON r.result_id=m.result_id
        JOIN prospective_football_settlement s ON s.settlement_id=m.settlement_id
        JOIN prospective_football_training_row t ON t.training_row_id=m.training_row_id
        ORDER BY m.sport,m.market_family,e.scheduled_start,m.game_id
    """)]


def _verify_selected(db, row):
    """Replay all selected evidence; never trust a view row by itself."""
    reasons = []
    sport, market = row["sport"], row["market_family"]
    if (sport, market) not in SCOPES or not row["game_id"].startswith(sport.lower() + ":"):
        reasons.append("EVENT_IDENTITY_AMBIGUOUS")
    if row["season_type"].lower() != "regular" or not row["home_team"] or not row["away_team"]:
        reasons.append("EVENT_IDENTITY_AMBIGUOUS")
    if row["quote_verified"] != 1 or not row["sportsbook"] or not row["capture_run_id"]:
        reasons.append("NO_VERIFIED_PREGAME_PRICE")
    try:
        if not (stage1.at(row["provider_last_update"]) <= stage1.at(row["quote_observed_at"]) <
                stage1.at(row["scheduled_start"])):
            reasons.append("QUOTE_TIMESTAMP_UNVERIFIED")
    except ValueError:
        reasons.append("QUOTE_TIMESTAMP_UNVERIFIED")
    if not row["event_source_hash"] or not row["quote_source_hash"] or not row["result_source_hash"]:
        reasons.append("SOURCE_LINEAGE_UNVERIFIED")
    try:
        _odds_probability(row["american_odds"])
        if not isinstance(row["line"], (int, float)) or not math.isfinite(row["line"]):
            raise ValueError("LINE_UNVERIFIED")
    except ValueError as exc:
        reasons.append(str(exc))
    if (row["result_status"] != "FINAL" or row["home_score"] is None or row["away_score"] is None or
            stage1.at(row["result_available_at"]) < stage1.at(row["scheduled_start"])):
        reasons.append("RESULT_UNVERIFIED")
    if stage1.at(row["available_for_training_at"]) != stage1.at(row["result_available_at"]):
        reasons.append("RESULT_LEAKAGE_RISK")
    if row["training_row_status"] != "TRAINING_READY" or json.loads(row["training_blockers"]):
        reasons.append("SETTLEMENT_UNREPRODUCIBLE")
    try:
        event = stage1._read(db, "prospective_football_event", "version_id", row["version_id"])
        quote = stage1._read(db, "prospective_football_quote", "quote_id", row["quote_id"])
        result = stage1._read(db, "prospective_football_result", "result_id", row["result_id"])
        settlement = stage1._read(db, "prospective_football_settlement", "settlement_id", row["settlement_id"])
        training = stage1._read(db, "prospective_football_training_row", "training_row_id", row["training_row_id"])
        if (event["source_hash"] != row["event_source_hash"] or
                quote["source_hash"] != row["quote_source_hash"] or
                result["source_hash"] != row["result_source_hash"] or
                settlement["payload_hash"] != row["settlement_payload_hash"] or
                training["source_manifest_hash"] != stage1.digest(
                    [event["source_hash"], quote["source_hash"], result["source_hash"], settlement["settlement_id"]])):
            reasons.append("SOURCE_LINEAGE_UNVERIFIED")
        if (quote["game_id"] != event["game_id"] or result["game_id"] != event["game_id"] or
                settlement["quote_id"] != quote["quote_id"] or settlement["result_id"] != result["result_id"]):
            reasons.append("EVENT_IDENTITY_AMBIGUOUS")
        raw_quote = json.loads(quote["raw_source"])
        label = exact_label(market, quote["selection"], quote["line"], result["home_score"],
                            result["away_score"], event["home_team"], event["away_team"],
                            provider_home_team=raw_quote.get("home_team"),
                            provider_away_team=raw_quote.get("away_team"))
        expected_stage1_label = ("PUSH" if label == "PUSH" else
                                 "WIN" if label in ("COVER", "OVER") and
                                 (market == "SPREAD" or quote["selection"] == "Over") else
                                 "WIN" if label == "UNDER" and quote["selection"] == "Under" else "LOSS")
        if training["label"] != expected_stage1_label or settlement["outcome"] != stage1.outcome(
                dict(quote, home_team=event["home_team"], away_team=event["away_team"],
                     provider_home_team=raw_quote.get("home_team"),
                     provider_away_team=raw_quote.get("away_team")), result):
            reasons.append("SETTLEMENT_UNREPRODUCIBLE")
        snapshot = _snapshot(event, quote, result)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, sqlite3.Error) as exc:
        code = str(exc)
        reasons.append(code if code in BLOCKERS else "SOURCE_LINEAGE_UNVERIFIED")
        snapshot = None
        label = None
    if reasons:
        return None, sorted(set(reasons))
    return {"manifest_id": row["manifest_id"], "game_id": row["game_id"], "sport": sport,
            "market_family": market, "season": row["season"], "kickoff": row["scheduled_start"],
            "quote_id": row["quote_id"], "selection": row["selection"], "line": row["line"],
            "label": label, "features": snapshot,
            "feature_snapshot_hash": digest(snapshot), "quote_observed_at": row["quote_observed_at"],
            "available_for_training_at": row["available_for_training_at"],
            "source_manifest_hash": row["source_manifest_hash"]}, []


def _scope_key(sport, market):
    return sport + "/" + market


def _canonical_source_audit(db, selected):
    audits = {}
    for sport, market in SCOPES:
        key = _scope_key(sport, market)
        quotes = [dict(r) for r in db.execute("""SELECT q.game_id,q.quote_id,q.quote_verified,
            q.observed_at,q.line,q.american_odds,q.sportsbook,e.season,e.scheduled_start
            FROM prospective_football_quote q JOIN prospective_football_event e
            ON e.version_id=q.event_version_id WHERE q.sport=? AND q.market_family=?""", (sport, market))]
        ready = db.execute("""SELECT count(*) FROM prospective_football_active_training_row
            WHERE sport=? AND market_family=?""", (sport, market)).fetchone()[0]
        selected_ids = {row["quote_id"] for row in selected if row["sport"] == sport and row["market_family"] == market}
        blockers = Counter()
        for q in quotes:
            if q["quote_id"] in selected_ids:
                continue
            if q["quote_verified"] != 1 or q["american_odds"] is None:
                blockers["NO_VERIFIED_PREGAME_PRICE"] += 1
            elif stage1.at(q["observed_at"]) >= stage1.at(q["scheduled_start"]):
                blockers["QUOTE_TIMESTAMP_UNVERIFIED"] += 1
            elif not db.execute("SELECT 1 FROM prospective_football_result WHERE game_id=? LIMIT 1", (q["game_id"],)).fetchone():
                blockers["RESULT_UNVERIFIED"] += 1
            elif not db.execute("SELECT 1 FROM prospective_football_settlement WHERE quote_id=? LIMIT 1", (q["quote_id"],)).fetchone():
                blockers["SETTLEMENT_UNREPRODUCIBLE"] += 1
            else:
                blockers["NONSELECTED_BOOK_OR_REPRICE"] += 1
        audits[key] = {
            "source": "canonical_stage1_quote_and_manifest", "source_rows": len(quotes),
            "unique_games": len({q["game_id"] for q in quotes}),
            "seasons": dict(sorted(Counter(q["season"] for q in quotes).items())),
            "stable_identity": bool(quotes), "exact_line": sum(q["line"] is not None for q in quotes),
            "verified_pregame_price_timestamp": sum(q["quote_verified"] == 1 and
                stage1.at(q["observed_at"]) < stage1.at(q["scheduled_start"]) for q in quotes),
            "final_result_games": db.execute("""SELECT count(DISTINCT game_id) FROM prospective_football_result
                WHERE sport=? AND result_status='FINAL'""", (sport,)).fetchone()[0],
            "reproducible_settlement_rows": db.execute("""SELECT count(*) FROM prospective_football_settlement
                WHERE sport=? AND market_family=?""", (sport, market)).fetchone()[0],
            "asof_feature_contract": FEATURE_VERSION, "training_ready_raw_rows": ready,
            "selected_manifest_rows": len(selected_ids), "nonselected_or_blocked_raw_rows": len(quotes)-len(selected_ids),
            "blocked_reasons": dict(sorted(blockers.items()))}
    return audits


def _legacy_source_audit(db, root):
    """Inventory candidates without promoting prior projections or CSVs."""
    output = []
    for table in ("prospective_quote", "prospective_reconciled_fact"):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not exists:
            continue
        for sport, market in SCOPES:
            rows = [dict(x) for x in db.execute(
                f"SELECT * FROM {table} WHERE sport=? AND market_family=?", (sport, market))]
            if not rows:
                continue
            output.append({"source": table, "scope": _scope_key(sport, market), "source_rows": len(rows),
                           "unique_games": len({x.get("game_id") or x.get("canonical_identity") for x in rows}),
                           "stable_identity": bool(all(x.get("event_id") or x.get("canonical_identity") for x in rows)),
                           "exact_line": sum(x.get("line") is not None for x in rows),
                           "verified_pregame_price_timestamp": 0,
                           "final_result": 0, "reproducible_settlement": 0,
                           "asof_features": 0, "training_ready": 0, "seasons": {},
                           "blocked_counts": {"SETTLEMENT_UNREPRODUCIBLE": len(rows),
                                              "FEATURE_ASOF_UNAVAILABLE": len(rows)},
                           "note": "Not joined to and certified by the Stage 1 football training manifest"})
    paths = [root / "data/master_all_sports.csv", root / "data/theover_spreads.csv",
             root / "data/theover_totals.csv", *sorted((root / "data/backtest_exports").glob("*.csv"))]
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            reader = csv.DictReader(stream)
            selected = [x for x in reader if (x.get("sport") or x.get("league") or x.get("League") or "").upper()
                        in ("NFL", "NCAAF")]
        if not selected:
            continue
        for sport in ("NFL", "NCAAF"):
            rows = [x for x in selected if (x.get("sport") or x.get("league") or x.get("League") or "").upper() == sport]
            if not rows:
                continue
            market = ("SPREAD" if path.name == "theover_spreads.csv" else
                      "TOTAL" if path.name == "theover_totals.csv" else "UNASSIGNED")
            ids = {x.get("game_id") or (x.get("Home") or x.get("Home Team"),
                                       x.get("Away") or x.get("Away Team"), x.get("commence_time")) for x in rows}
            seasons = Counter((x.get("commence_time") or x.get("Game Date") or "")[:4] or "UNKNOWN" for x in rows)
            output.append({"source": str(path.relative_to(root)), "scope": _scope_key(sport, market),
                           "source_rows": len(rows), "unique_games": len(ids), "seasons": dict(sorted(seasons.items())),
                           "stable_identity": False, "exact_line": 0,
                           "verified_pregame_price_timestamp": 0, "final_result": 0,
                           "reproducible_settlement": 0, "asof_features": 0,
                           "training_ready": 0,
                           "blocked_counts": {"NO_VERIFIED_PREGAME_PRICE": len(rows),
                                              "QUOTE_TIMESTAMP_UNVERIFIED": len(rows),
                                              "SOURCE_LINEAGE_UNVERIFIED": len(rows),
                                              "FEATURE_ASOF_UNAVAILABLE": len(rows)},
                           "note": "CSV has no certified pregame quote/result/as-of lineage; unassigned market rows do not count in either exact market"})
    local_inventory = root / "docs/audits/football-stage2-local-candidate-sources.json"
    if local_inventory.is_file():
        snapshot = json.loads(local_inventory.read_text())
        if snapshot.get("schema") != "football-stage2-local-candidate-inventory-v1":
            raise ValueError("LOCAL_CANDIDATE_AUDIT_SCHEMA_INVALID")
        output.extend(dict(item, source_availability="OWNER_ATTACHED_OFFLINE_SNAPSHOT_NOT_CI_INPUT")
                      for item in snapshot["sources"])
    return output


def feature_contract():
    features = [
        ("line", "Exact stored selection spread or game total", "verified_quote", "quote observed_at", "required"),
        ("price_implied_probability", "American-odds implied chance, before vig adjustment; benchmark input only", "verified_quote", "quote observed_at", "required"),
        ("neutral_site", "Pregame scheduled neutral-site flag", "pregame_schedule", "schedule discovered_at", "required"),
        ("selection_is_home_or_over", "Home selection for spread, Over selection for total", "verified_quote", "quote observed_at", "required"),
    ]
    return {"schema": VERSION, "feature_version": FEATURE_VERSION,
            "scope_versions": {_scope_key(*scope): FEATURE_VERSION for scope in SCOPES},
            "features": [dict(name=n, definition=d, source=s, availability=a, missing_policy=m)
                         for n,d,s,a,m in features],
            "forbidden": ["same-event result/score", "season-end statistics used before season end",
                          "current roster or injury substituted for historical snapshot", "future ratings/rankings",
                          "postgame market data", "unavailable closing price", "TheOver ModelHitRate as game probability"],
            "target_classes": CLASSES, "raw_team_stats_status": "EXCLUDED_UNTIL_VERSIONED_ASOF_SNAPSHOTS_EXIST"}


def chronological_partitions(rows):
    """One deterministic time ordered partition per sport; ties stay together."""
    by_sport = {}
    for sport in ("NFL", "NCAAF"):
        games = {}
        for row in rows:
            if row["sport"] != sport:
                continue
            prior = games.setdefault(row["game_id"], row["kickoff"])
            if prior != row["kickoff"]:
                raise ValueError("EVENT_IDENTITY_AMBIGUOUS")
        groups = defaultdict(list)
        for game, kickoff in games.items():
            groups[kickoff].append(game)
        ordered = sorted(groups.items(), key=lambda item: stage1.at(item[0]))
        n = len(games)
        targets = [round(n * 0.55), round(n * 0.70), round(n * 0.85)]
        assignment = {}
        count = 0
        for kickoff, game_ids in ordered:
            index = sum(count >= target for target in targets)
            part = PARTITIONS[min(index, 3)]
            assignment.update({game: part for game in game_ids})
            count += len(game_ids)
        by_sport[sport] = assignment
    return by_sport


def validate_partitions(rows, partitions):
    """Reject random/overlapping/future-leaking final evaluation designs."""
    if any(part not in PARTITIONS for part in partitions.values()):
        raise ValueError("RANDOM_FINAL_HOLDOUT_PROHIBITED")
    seen = {}
    chronology = defaultdict(list)
    for row in rows:
        game = (row["sport"], row["game_id"])
        part = partitions.get(game)
        if part is None or row.get("assigned_partition", part) != part or (game in seen and seen[game] != part):
            raise ValueError("SAME_GAME_CROSSES_PARTITIONS")
        seen[game] = part
        chronology[row["sport"]].append((stage1.at(row["kickoff"]), PARTITIONS.index(part)))
    for pairs in chronology.values():
        indices = [i for _, i in sorted(pairs)]
        if indices != sorted(indices):
            raise ValueError("RANDOM_FINAL_HOLDOUT_PROHIBITED")


def validate_probability_vector(probabilities, market):
    if (market not in CLASSES or set(probabilities) != set(CLASSES[market]) or
            any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x > 1
                for x in probabilities.values()) or
            abs(sum(probabilities.values()) - 1) > 1e-8):
        raise ValueError("INVALID_EXACT_TARGET_PROBABILITY_VECTOR")
    return probabilities


def _metric_rows(rows, predictions, market):
    if not rows:
        return None
    classes = CLASSES[market]
    for p in predictions:
        validate_probability_vector(p, market)
    brier = sum(sum((p[c] - int(r["label"] == c)) ** 2 for c in classes)
                for r,p in zip(rows,predictions)) / len(rows)
    logloss = -sum(math.log(max(p[r["label"]], 1e-15)) for r,p in zip(rows,predictions)) / len(rows)
    hit = sum(max(p, key=p.get) == r["label"] for r,p in zip(rows,predictions)) / len(rows)
    ece = sum(abs(sum(p[c] for p in predictions)/len(rows) -
                  sum(r["label"] == c for r in rows)/len(rows)) for c in classes)/len(classes)
    result = {"n": len(rows), "brier": brier, "log_loss": logloss,
              "calibration_error": ece, "coverage": len(predictions)/len(rows),
              "hit_rate": hit, "push_rate": sum(r["label"] == "PUSH" for r in rows)/len(rows),
              "class_balance": dict(Counter(r["label"] for r in rows)),
              "uncertainty": "Small-sample scores are descriptive; no confidence interval claimed"}
    nonpush = [(r,p) for r,p in zip(rows,predictions) if r["label"] != "PUSH"]
    if len({r["label"] for r,_ in nonpush}) == 2:
        from sklearn.metrics import roc_auc_score
        positive = classes[0]
        result["nonpush_auc"] = float(roc_auc_score(
            [int(r["label"] == positive) for r,_ in nonpush],
            [p[positive]/max(p[positive]+p[classes[2]], 1e-15) for _,p in nonpush]))
    else:
        result["nonpush_auc"] = None
    return result


def _baselines(training, validation, market):
    classes = CLASSES[market]
    counts = Counter(x["label"] for x in training)
    base = {c: (counts[c]+1)/(len(training)+len(classes)) for c in classes}
    base_predictions = [base.copy() for _ in validation]
    market_predictions = []
    for row in validation:
        implied = row["features"]["price_implied_probability"]["value"]
        push = base["PUSH"]
        # Vig is not removed: this is an explicit sportsbook-price benchmark.
        win = (1-push)*implied
        if market == "SPREAD":
            market_predictions.append({"COVER": win, "PUSH": push, "NO_COVER": 1-push-win})
        else:
            over = win if row["selection"] == "Over" else 1-push-win
            market_predictions.append({"OVER": over, "PUSH": push, "UNDER": 1-push-over})
    return {"base_rate": {"status": "BASELINE_ONLY", "probabilities": base,
                          "metrics": _metric_rows(validation, base_predictions, market)},
            "market_implied": {"status": "BASELINE_ONLY", "method": "stored American odds with empirical push share; no-vig not asserted",
                               "metrics": _metric_rows(validation, market_predictions, market)}}


def _scope_split(rows, assignment):
    partition_rows = {name: [] for name in PARTITIONS}
    for row in rows:
        partition_rows[assignment[row["sport"]][row["game_id"]]].append(row)
    return partition_rows


def _training_gate(parts, rows):
    problems = []
    for name, required in MINIMUM.items():
        if len(parts[name]) < required:
            problems.append(f"{name.upper()}_INDEPENDENT_N_BELOW_{required}")
    if len({x["season"] for x in rows}) < 2:
        problems.append("TWO_DISTINCT_SEASONS_REQUIRED")
    for left, right in zip(PARTITIONS, PARTITIONS[1:]):
        if parts[left] and parts[right] and max(stage1.at(x["available_for_training_at"]) for x in parts[left]) >= \
                min(stage1.at(x.get("quote_observed_at", x["kickoff"])) for x in parts[right]):
            problems.append("RESULT_AVAILABILITY_CROSSES_" + right.upper())
    counts = Counter(x["label"] for x in parts["development"])
    if not rows or any(counts[c] < 5 for c in CLASSES[rows[0]["market_family"]]):
        problems.append("DEVELOPMENT_CLASS_SUPPORT_INSUFFICIENT")
    return problems


def validate_calibration_cohort(training, validation, calibration, holdout):
    sets = [{(r["sport"], r["game_id"]) for r in group} for group in
            (training, validation, calibration, holdout)]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i+1, 4)):
        raise ValueError("CALIBRATION_COHORT_OVERLAP")
    times = [[stage1.at(r["kickoff"]) for r in group] for group in
             (training, validation, calibration, holdout)]
    if any(not group for group in times) or any(max(times[i]) >= min(times[i+1]) for i in range(3)):
        raise ValueError("CALIBRATION_CHRONOLOGY_INVALID")
    for left, right in zip((training, validation, calibration), (validation, calibration, holdout)):
        if max(stage1.at(x["available_for_training_at"]) for x in left) >= \
                min(stage1.at(x.get("quote_observed_at", x["kickoff"])) for x in right):
            raise ValueError("CALIBRATION_RESULT_AVAILABILITY_LEAKAGE")


def _temperature_vector(probabilities, classes, temperature):
    weights = {c: max(probabilities[c], 1e-15) ** (1/temperature) for c in classes}
    total = sum(weights.values())
    return {c: weights[c]/total for c in classes}


def _artifacts_for_scope(rows, parts, blockers, source_commit, now):
    """Train only when the prespecified gate and exact-scope checks pass."""
    sport, market = rows[0]["sport"], rows[0]["market_family"]
    key = _scope_key(sport, market)
    counts = {name: len(parts[name]) for name in PARTITIONS}
    reasons = _training_gate(parts, rows)
    split = {"scope": key, "strategy": "CHRONOLOGICAL_WHOLE_GAME_NO_RANDOM_HOLDOUT",
             "partition_n": counts, "minimum_independent_n": MINIMUM,
             "boundaries": {name: {"first_kickoff": min((x["kickoff"] for x in parts[name]), default=None),
                                  "last_kickoff": max((x["kickoff"] for x in parts[name]), default=None)}
                            for name in PARTITIONS},
             "assignment_hash": digest(sorted((x["game_id"], name) for name in PARTITIONS for x in parts[name])),
             "status": "FEASIBLE" if not reasons else "INSUFFICIENT_TRAINING_EVIDENCE", "blockers": reasons}
    if reasons:
        return split, {"status": "NOT_EVALUATED", "reason": "INSUFFICIENT_TRAINING_EVIDENCE"}, \
            {"status": "NOT_TRAINED", "candidates": []}, \
            {"model_status": "INSUFFICIENT_TRAINING_EVIDENCE", "model_id": None, "artifact_hash": None,
             "blockers": reasons, "independent_n": len(rows)}, \
            {"calibration_status": "INSUFFICIENT_EVIDENCE", "calibration_id": None,
             "blockers": ["NO_SELECTED_EXACT_SCOPE_MODEL"]}, None
    # The only admitted features are quote/schedule facts. Fit a transparent
    # three-class regularized logistic candidate after base-rate benchmarks.
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    def matrix(items):
        return np.asarray([[row["features"][name]["value"] for name in
                            ("line", "price_implied_probability", "neutral_site", "selection_is_home_or_over")]
                           for row in items], dtype=float)
    training, validation = parts["development"], parts["selection_validation"]
    baseline = _baselines(training, validation, market)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(matrix(training), [r["label"] for r in training])
    probs = clf.predict_proba(matrix(validation))
    candidate_predictions = [{label: float(p[list(clf.classes_).index(label)]) for label in CLASSES[market]}
                             for p in probs]
    candidate_metrics = _metric_rows(validation, candidate_predictions, market)
    baseline_scores = [baseline[name]["metrics"] for name in ("base_rate", "market_implied")]
    improves = all(candidate_metrics["brier"] < score["brier"] and
                   candidate_metrics["log_loss"] < score["log_loss"] for score in baseline_scores)
    comparison = {"status": "SELECTED" if improves else "NO_STABLE_IMPROVEMENT_OVER_BASELINES",
                  "candidates": [{"algorithm": "regularized_multinomial_logistic", "metrics": candidate_metrics,
                                  "selection_rule": "lower Brier and log loss than both baselines"}]}
    if not improves:
        return split, baseline, comparison, \
            {"model_status": "INSUFFICIENT_TRAINING_EVIDENCE", "model_id": None, "artifact_hash": None,
             "blockers": ["NO_STABLE_IMPROVEMENT_OVER_BASELINES"], "independent_n": len(rows)}, \
            {"calibration_status": "INSUFFICIENT_EVIDENCE", "calibration_id": None,
             "blockers": ["NO_SELECTED_EXACT_SCOPE_MODEL"]}, None
    train_manifest = digest(sorted((r["manifest_id"], r["source_manifest_hash"]) for r in training))
    config = {"feature_version": FEATURE_VERSION, "algorithm": "regularized_multinomial_logistic",
              "C": 1.0, "max_iter": 1000, "class_order": list(clf.classes_), "split_assignment_hash": split["assignment_hash"]}
    artifact = {"schema": VERSION, "sport": sport, "market_family": market, "target_classes": CLASSES[market],
                "feature_version": FEATURE_VERSION, "training_start": min(r["kickoff"] for r in training),
                "training_cutoff": max(r["available_for_training_at"] for r in training),
                "training_manifest_hash": train_manifest, "independent_training_n": len(training),
                "validation_window": split["boundaries"]["selection_validation"],
                "validation_n": len(validation), "algorithm": config["algorithm"],
                "hyperparameters": {"C": 1.0, "max_iter": 1000}, "classes": list(clf.classes_),
                "coefficients": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                "created_at": now, "available_at": now, "source_commit": source_commit,
                "runtime_environment_hash": digest({"python": platform.python_version(), "sklearn": sklearn.__version__}),
                "training_config_hash": digest(config), "metrics_artifact_hash": digest(comparison),
                "deployment_state": "UNVALIDATED", "production_eligible": False, "stake": 0}
    artifact_hash = digest(artifact)
    artifact["model_id"] = "football-stage2-" + artifact_hash
    inventory = {"model_status": "RESEARCH_MODEL_SELECTED", "model_id": artifact["model_id"],
                 "artifact_hash": artifact_hash, "independent_n": len(rows), "blockers": [], "model_artifact": artifact}
    fit_rows, holdout = parts["calibration_candidate"], parts["research_holdout"]
    try:
        validate_calibration_cohort(training, validation, fit_rows, holdout)
        if any(Counter(x["label"] for x in fit_rows)[c] < 2 for c in CLASSES[market]):
            raise ValueError("CALIBRATION_CLASS_SUPPORT_INSUFFICIENT")
        from scipy.optimize import minimize_scalar
        def predict(items):
            values = clf.predict_proba(matrix(items))
            return [{label: float(p[list(clf.classes_).index(label)]) for label in CLASSES[market]}
                    for p in values]
        raw_fit = predict(fit_rows)
        def objective(temperature):
            adjusted = [_temperature_vector(p, CLASSES[market], temperature) for p in raw_fit]
            return _metric_rows(fit_rows, adjusted, market)["log_loss"]
        fitted = minimize_scalar(objective, bounds=(0.5, 2.0), method="bounded",
                                 options={"xatol": 1e-5})
        temperature = float(fitted.x)
        adjusted_fit = [_temperature_vector(p, CLASSES[market], temperature) for p in raw_fit]
        raw_metrics = _metric_rows(fit_rows, raw_fit, market)
        adjusted_metrics = _metric_rows(fit_rows, adjusted_fit, market)
        if (not fitted.success or abs(temperature-1) < 1e-3 or
                adjusted_metrics["log_loss"] >= raw_metrics["log_loss"] - 1e-4 or
                adjusted_metrics["brier"] > raw_metrics["brier"] + 1e-4):
            raise ValueError("CALIBRATION_NO_STABLE_GAIN")
        raw_holdout = predict(holdout)
        calibrated_holdout = [_temperature_vector(p, CLASSES[market], temperature) for p in raw_holdout]
        holdout_diagnostics = {
            "uncalibrated": _metric_rows(holdout, raw_holdout, market),
            "calibrated": _metric_rows(holdout, calibrated_holdout, market)}
        calibration_artifact = {
            "schema": VERSION, "sport": sport, "market_family": market,
            "model_id": artifact["model_id"], "method": "multiclass_temperature_scaling",
            "temperature": temperature,
            "fit_window": split["boundaries"]["calibration_candidate"],
            "fit_cutoff": max(r["available_for_training_at"] for r in fit_rows),
            "fit_manifest_hash": digest(sorted((r["manifest_id"], r["source_manifest_hash"]) for r in fit_rows)),
            "independent_fit_n": len(fit_rows), "created_at": now, "available_at": now,
            "source_commit": source_commit,
            "diagnostics": {"raw_fit": raw_metrics, "calibrated_fit": adjusted_metrics,
                            "holdout": holdout_diagnostics},
            "deployment_state": "UNVALIDATED", "production_eligible": False, "stake": 0}
        calibration_hash = digest(calibration_artifact)
        calibration_artifact["calibration_id"] = "football-stage2-calibration-" + calibration_hash
        calibration = {"calibration_status": "RESEARCH_CALIBRATION_FIT", "calibration_id": calibration_artifact["calibration_id"],
                       "artifact_hash": calibration_hash, "calibration_artifact": calibration_artifact,
                       "blockers": []}
    except ValueError as exc:
        calibration = {"calibration_status": "INSUFFICIENT_EVIDENCE", "calibration_id": None,
                       "blockers": [str(exc)]}
    return split, baseline, comparison, inventory, calibration, artifact


def validate_research_prediction(model, event, quote, snapshot, timestamp):
    """Pure validation only; this does not create a production prediction."""
    if model is None or model.get("model_status") != "RESEARCH_MODEL_SELECTED":
        raise ValueError("NO_VALID_EXACT_SCOPE_MODEL")
    artifact = model.get("model_artifact") or {}
    spread_side = _spread_side(event, quote) if quote["market_family"] == "SPREAD" else None
    if (model.get("artifact_hash") != digest({k:v for k,v in artifact.items() if k != "model_id"}) or
            model.get("model_id") != artifact.get("model_id") or
            event["sport"] != artifact.get("sport") or quote["market_family"] != artifact.get("market_family") or
            tuple(artifact.get("target_classes") or ()) != CLASSES.get(quote["market_family"]) or
            artifact.get("feature_version") != FEATURE_VERSION or
            artifact.get("deployment_state") != "UNVALIDATED" or
            artifact.get("production_eligible") is not False or artifact.get("stake") != 0 or
            event["game_id"] != quote["game_id"] or quote["event_version_id"] != event["version_id"] or
            quote.get("quote_verified") != 1 or not quote.get("quote_id") or
            (spread_side is None if quote["market_family"] == "SPREAD" else
             quote["selection"] not in ("Over", "Under")) or
            not math.isclose(quote["line"], snapshot["line"]["value"], abs_tol=1e-9) or
            not math.isclose(_odds_probability(quote["american_odds"]),
                             snapshot["price_implied_probability"]["value"], abs_tol=1e-9) or
            snapshot["neutral_site"]["value"] != int(event["neutral_site"]) or
            snapshot["selection_is_home_or_over"]["value"] != int(
                spread_side == "home" if quote["market_family"] == "SPREAD" else
                quote["selection"] == "Over") or
            not stage1.at(artifact["training_cutoff"]) <= stage1.at(artifact["available_at"]) <= stage1.at(timestamp) < stage1.at(event["scheduled_start"]) or
            not stage1.at(quote["provider_last_update"]) <= stage1.at(quote["observed_at"]) <= stage1.at(timestamp)):
        raise ValueError("RESEARCH_PREDICTION_EVIDENCE_MISMATCH")
    validate_feature_snapshot(snapshot, quote_at=quote["observed_at"], kickoff=event["scheduled_start"],
                              event_discovered_at=event["discovered_at"],
                              result_available_at=event["scheduled_start"])
    return {"deployment_state": "UNVALIDATED", "production_eligible": False, "stake": 0,
            "model_id": model["model_id"], "game_id": event["game_id"], "quote_id": quote["quote_id"],
            "selection": quote["selection"], "line": quote["line"],
            "feature_snapshot_hash": digest(snapshot), "prediction_timestamp": _iso(timestamp)}


def research_prediction(model, calibration, event, quote, snapshot, timestamp):
    """Return an unvalidated research record with exact-target probabilities."""
    import numpy as np
    bound = validate_research_prediction(model, event, quote, snapshot, timestamp)
    artifact = model["model_artifact"]
    values = np.asarray([snapshot[name]["value"] for name in
                         ("line", "price_implied_probability", "neutral_site", "selection_is_home_or_over")], dtype=float)
    logits = np.asarray(artifact["coefficients"]) @ values + np.asarray(artifact["intercept"])
    logits -= max(logits)
    weights = np.exp(logits)
    raw = {label: float(value/sum(weights)) for label,value in zip(artifact["classes"], weights)}
    vector = validate_probability_vector(raw, quote["market_family"])
    calibrated = False
    if calibration and calibration.get("calibration_status") == "RESEARCH_CALIBRATION_FIT":
        item = calibration.get("calibration_artifact") or {}
        if (calibration.get("artifact_hash") != digest({k:v for k,v in item.items() if k != "calibration_id"}) or
                item.get("model_id") != model["model_id"] or
                item.get("sport") != event["sport"] or item.get("market_family") != quote["market_family"] or
                stage1.at(item["available_at"]) > stage1.at(timestamp)):
            raise ValueError("CALIBRATION_EVIDENCE_MISMATCH")
        vector = validate_probability_vector(
            _temperature_vector(vector, CLASSES[quote["market_family"]], item["temperature"]),
            quote["market_family"])
        calibrated = True
    # This is a versioned descriptive uncertainty bound, not a validated
    # frequentist confidence interval or an executable betting probability.
    target = "COVER" if quote["market_family"] == "SPREAD" else quote["selection"].upper()
    mean = vector[target]
    n = (calibration.get("calibration_artifact") or {}).get("independent_fit_n") if calibrated else None
    interval = [max(0.0, mean-1.96*math.sqrt(mean*(1-mean)/n)),
                min(1.0, mean+1.96*math.sqrt(mean*(1-mean)/n))] if n else None
    return dict(bound, target_classes=CLASSES[quote["market_family"]], probabilities=vector,
                probability_semantics="EXACT_SPREAD_COVER" if quote["market_family"] == "SPREAD" else "EXACT_TOTAL_OVER_UNDER",
                calibration_id=calibration["calibration_id"] if calibrated else None,
                mean_probability=mean, uncertainty_interval=interval,
                conservative_probability=interval[0] if interval else None,
                uncertainty_method="normal_binomial_descriptive_v1" if interval else None)


def build_reports(path, source_root, *, stage1_report, source_commit, now=None):
    """Return eight sanitized reports; require authenticated, read-back Stage 1."""
    remote = stage1_report.get("remote") or {}
    if (not remote.get("pre_mutation_verified") or not remote.get("backup_readback_verified") or
            stage1_report.get("execution_state") not in ("COMPLETE", "PARTIAL_FAILURE") or
            not stage1_report.get("run_id")):
        raise ValueError("STAGE1_AUTHENTICATED_REFRESH_NOT_VERIFIED")
    now = _iso(now or datetime.now(timezone.utc))
    with closing(evidence.connect(path)) as db:
        selected = _selected_rows(db)
        source_audit = _canonical_source_audit(db, selected)
        legacy_sources = _legacy_source_audit(db, Path(source_root))
        valid, quarantine = [], []
        seen = set()
        for row in selected:
            key = (row["sport"], row["market_family"], row["game_id"])
            if key in seen:
                quarantine.append({"scope": _scope_key(*key[:2]), "manifest_id": row["manifest_id"],
                                   "blockers": ["DUPLICATE_INDEPENDENT_GAME"]})
                continue
            seen.add(key)
            clean, problems = _verify_selected(db, row)
            if problems:
                quarantine.append({"scope": _scope_key(*key[:2]), "manifest_id": row["manifest_id"],
                                   "blockers": problems})
            else:
                valid.append(clean)
    assignments = chronological_partitions(valid)
    validate_partitions(valid, {(s,g): p for s, mapping in assignments.items() for g,p in mapping.items()})
    audit = {"schema": VERSION, "stage1_run_id": stage1_report["run_id"],
             "stage1_execution_state": stage1_report["execution_state"],
             "stage1_backup_readback_verified": True, "selection_rule_version": "football-one-observation-v1",
             "canonical_sources": source_audit, "historical_candidate_sources": legacy_sources,
             "quarantined_selected_rows": quarantine,
             "stage1_capture_status": {sport: {"requested_slate_success":
                 (stage1_report.get("sports") or {}).get(sport, {}).get("requested_slate_success"),
                 "errors": (stage1_report.get("sports") or {}).get(sport, {}).get("errors", [])}
                 for sport in ("NFL", "NCAAF")}, "scopes": {}}
    split_report = {"schema": VERSION, "split_frozen_at": now, "split_policy": "whole games, kickoff ordered; same sport uses one assignment across spread and total", "scopes": {}}
    baseline_report = {"schema": VERSION, "scopes": {}}
    comparison_report = {"schema": VERSION, "scopes": {}}
    inventory_report = {"schema": VERSION, "source_commit": source_commit, "scopes": {}}
    calibration_report = {"schema": VERSION, "scopes": {}}
    prediction_report = {"schema": VERSION, "scopes": {}}
    for sport, market in SCOPES:
        key = _scope_key(sport, market)
        scope_rows = [row for row in valid if (row["sport"], row["market_family"]) == (sport, market)]
        season_counts = dict(sorted(Counter(row["season"] for row in scope_rows).items()))
        quarantine_reasons = Counter(reason for item in quarantine if item["scope"] == key for reason in item["blockers"])
        audit["scopes"][key] = {"raw_quote_rows": source_audit[key]["source_rows"],
            "raw_training_ready_rows": source_audit[key]["training_ready_raw_rows"],
            "manifest_selected_rows": source_audit[key]["selected_manifest_rows"],
            "legal_independent_n": len(scope_rows), "legal_by_season": season_counts,
            "blocked_or_nonselected_raw_reasons": source_audit[key]["blocked_reasons"],
            "quarantined_manifest_reasons": dict(sorted(quarantine_reasons.items())),
            "historical_csv_rows_promoted": 0, "legacy_rows_promoted": 0,
            "verified_fields": ["canonical identity", "exact line", "verified pregame odds/timestamp",
                                "final result", "replayed settlement", "as-of quote/schedule features", "source lineage"]}
        if scope_rows:
            parts = _scope_split(scope_rows, assignments)
            split, baselines, comparison, inventory, calibration, _ = _artifacts_for_scope(
                scope_rows, parts, quarantine_reasons, source_commit, now)
        else:
            blockers = ["NO_LEGAL_INDEPENDENT_TRAINING_ROWS"]
            split = {"scope": key, "strategy": "CHRONOLOGICAL_WHOLE_GAME_NO_RANDOM_HOLDOUT",
                     "partition_n": {name: 0 for name in PARTITIONS}, "minimum_independent_n": MINIMUM,
                     "boundaries": {name: {"first_kickoff": None, "last_kickoff": None} for name in PARTITIONS},
                     "assignment_hash": digest([]), "status": "INSUFFICIENT_TRAINING_EVIDENCE", "blockers": blockers}
            baselines = {"status": "NOT_EVALUATED", "reason": "INSUFFICIENT_TRAINING_EVIDENCE"}
            comparison = {"status": "NOT_TRAINED", "candidates": []}
            inventory = {"model_status": "INSUFFICIENT_TRAINING_EVIDENCE", "model_id": None,
                         "artifact_hash": None, "blockers": blockers, "independent_n": 0}
            calibration = {"calibration_status": "INSUFFICIENT_EVIDENCE", "calibration_id": None,
                           "blockers": ["NO_SELECTED_EXACT_SCOPE_MODEL"]}
        split_report["scopes"][key] = split
        baseline_report["scopes"][key] = baselines
        comparison_report["scopes"][key] = comparison
        inventory_report["scopes"][key] = inventory
        calibration_report["scopes"][key] = calibration
        prediction_report["scopes"][key] = {
            "status": "RESEARCH_ONLY_READY" if inventory["model_id"] else "BLOCKED_NO_VALID_MODEL",
            "model_id": inventory["model_id"], "required_bindings": ["canonical game", "exact quote/line",
                "model availability and training cutoff", "feature snapshot", "prediction timestamp", "exact target vector"],
            "calibration_status": calibration["calibration_status"],
            "calibration_id": calibration["calibration_id"],
            "deployment_state": "UNVALIDATED", "production_eligible": False, "stake": 0,
            "validation_status": "NOT_STARTED", "production_parlay_eligible": False,
            "wager_eligible": False}
    return {"training_audit": audit, "feature_contract": feature_contract(), "split_plan": split_report,
            "baselines": baseline_report, "model_comparison": comparison_report,
            "model_inventory": inventory_report, "calibration_readiness": calibration_report,
            "research_prediction_readiness": prediction_report}


def write_reports(reports, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key, filename in ARTIFACT_NAMES.items():
        (output / filename).write_text(json.dumps(reports[key], sort_keys=True, indent=2, allow_nan=False) + "\n")
    for key, item in reports["model_inventory"]["scopes"].items():
        if item.get("model_artifact"):
            (output / ("football-stage2-model-" + key.lower().replace("/", "-") + "-" + item["artifact_hash"] + ".json")).write_text(
                json.dumps(item["model_artifact"], sort_keys=True, indent=2, allow_nan=False) + "\n")
    for key, item in reports["calibration_readiness"]["scopes"].items():
        if item.get("calibration_artifact"):
            (output / ("football-stage2-calibration-" + key.lower().replace("/", "-") + "-" + item["artifact_hash"] + ".json")).write_text(
                json.dumps(item["calibration_artifact"], sort_keys=True, indent=2, allow_nan=False) + "\n")
