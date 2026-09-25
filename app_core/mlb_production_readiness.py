"""Read-only, exact-market MLB production-readiness audit.

Only restored, immutable MLB pregame receipts can enter the research training
manifest. Other stores are inventoried, never silently joined or promoted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3

from app_core import mlb_pregame_receipts as receipts
from app_core import mlb_spread_total_model as old_model
from app_core.mlb_history import timestamp
from app_core.mlb_event_matcher import scheduled_eastern_date
from app_core.public_quote_policy import canonical_book_label
from core.wager_decisions import decimal_price

SCHEMA = "mlb-production-readiness-v1"
POLICY = "mlb-one-game-market-home-over-v1"
FEATURE_VERSION = "mlb-receipt-asof-exact-market-v1"
SCOPES = ("MLB/RUN_LINE", "MLB/TOTAL")
CLASSES = {"RUN_LINE": ("COVER", "PUSH", "NO_COVER"),
           "TOTAL": ("OVER", "PUSH", "UNDER")}
PARTITIONS = ("development", "selection_validation", "calibration_candidate", "research_holdout")
# Eight pregame features plus an intercept yield about 18 class logits. 240
# development games give roughly 13 events per parameter before class imbalance;
# the three separate 80-game evaluation windows avoid pretending a tiny
# calibration or holdout sample is precise. Two seasons check temporal drift.
MINIMUM = {"development": 240, "selection_validation": 80,
           "calibration_candidate": 80, "research_holdout": 80}
REPORTS = {
    "inventory": "mlb-production-readiness-inventory.json",
    "receipt_audit": "mlb-receipt-reconciliation-audit.json",
    "training": "mlb-training-readiness.json",
    "manifest": "mlb-independent-training-manifest.json",
    "features": "mlb-feature-contract.json",
    "split": "mlb-split-plan.json",
    "baselines": "mlb-baselines.json",
    "comparison": "mlb-model-comparison.json",
    "models": "mlb-model-inventory.json",
    "calibration": "mlb-calibration-readiness.json",
    "validation": "mlb-validation-readiness.json",
    "close": "mlb-close-evidence-audit.json",
}


def digest(value):
    return old_model.digest(value)


def iso(value):
    return timestamp(value).astimezone(timezone.utc).isoformat()


def scope_for(market_type):
    if market_type in ("spread_home", "spread_away"):
        return "MLB/RUN_LINE"
    if market_type in ("total_over", "total_under"):
        return "MLB/TOTAL"
    raise ValueError("WRONG_TARGET")


def exact_label(market_type, line, home, away, status="FINAL"):
    """Preserve pushes and voids; never score a moneyline as a run line."""
    scope = scope_for(market_type)
    try:
        outcome = old_model.label(market_type, line, home, away, status)
    except (ValueError, TypeError, KeyError):
        return "NEEDS_REVIEW"
    if outcome in ("VOID", "PUSH"):
        return outcome
    if scope == "MLB/RUN_LINE":
        return "COVER" if outcome == "WIN" else "NO_COVER"
    if market_type == "total_over":
        return "OVER" if outcome == "WIN" else "UNDER"
    return "UNDER" if outcome == "WIN" else "OVER"


def validate_vector(vector, scope):
    classes = CLASSES[scope.split("/")[1]]
    if (not isinstance(vector, dict) or set(vector) != set(classes) or
            any(type(vector[k]) not in (int, float) or not math.isfinite(vector[k]) or
                vector[k] < 0 or vector[k] > 1 for k in classes) or
            not math.isclose(sum(vector.values()), 1, abs_tol=1e-9)):
        raise ValueError("INVALID_EXACT_TARGET_VECTOR")
    return vector


def exact_feature_values(snapshot):
    """Replay six as-of team summaries and bind the selected line and price."""
    payload, prior_values = old_model.receipt_features(snapshot)
    quote = payload["quote"]
    if len(prior_values) != 7:
        raise ValueError("FEATURE_CONTRACT_MISMATCH")
    values = prior_values[:6] + [float(quote["line"]), 1.0 / float(quote["decimal_odds"])]
    if len(values) != 8 or any(not math.isfinite(value) for value in values):
        raise ValueError("FEATURE_CONTRACT_MISMATCH")
    return payload, values


def _quote_source_check(payload, observations):
    """Replay the selected price from the retained Odds API response."""
    quote = payload["quote"]
    key = (payload.get("source_observations") or {}).get("quotes")
    source = observations.get(key)
    if not source or digest(source) != key or source.get("source") != "odds_api":
        return "SOURCE_LINEAGE_UNVERIFIED"
    raw = source.get("payload") or {}
    if (str(raw.get("id")) != str(quote.get("provider_event_id")) or
            raw.get("sport_key") != "baseball_mlb"):
        return "EVENT_IDENTITY_AMBIGUOUS"
    if iso(source["observed_at"]) != iso(quote["observed_at"]):
        return "QUOTE_TIMESTAMP_UNVERIFIED"
    if iso(raw["commence_time"]) != iso(quote.get("source_game_start_utc", payload["game_start_utc"])):
        return "EVENT_IDENTITY_AMBIGUOUS"
    kind = quote["market_type"]
    family = "spreads" if kind.startswith("spread") else "totals"
    selection = (raw.get("home_team") if kind == "spread_home" else
                 raw.get("away_team") if kind == "spread_away" else
                 "Over" if kind == "total_over" else "Under")
    exact = []
    seen_book = False
    seen_line = False
    for book in raw.get("bookmakers", []):
        if canonical_book_label(book.get("key")) != canonical_book_label(quote["sportsbook"]):
            continue
        seen_book = True
        for market in book.get("markets", []):
            if market.get("key") != family:
                continue
            if quote.get("provider_updated_at") and iso(
                    market.get("last_update") or book.get("last_update")) != iso(quote["provider_updated_at"]):
                continue
            for entry in market.get("outcomes", []):
                if str(entry.get("name", "")).casefold() != str(selection).casefold():
                    continue
                if entry.get("point") is None or float(entry["point"]) != float(quote["line"]):
                    continue
                seen_line = True
                if decimal_price(entry.get("price")) == float(quote["decimal_odds"]):
                    exact.append(entry)
    if len(exact) == 1:
        return None
    if len(exact) > 1:
        return "PRICE_CONFLICT"
    return "PRICE_CONFLICT" if seen_line else "LINE_CONFLICT" if seen_book else "SOURCE_LINEAGE_UNVERIFIED"


def _schedule_check(payload, observations):
    key = (payload.get("source_observations") or {}).get("schedule")
    source = observations.get(key)
    if not source or digest(source) != key or source.get("source") != "mlb_statsapi":
        return "SOURCE_LINEAGE_UNVERIFIED", None
    matches = [g for day in source.get("payload", {}).get("dates", []) for g in day.get("games", [])
               if str(g.get("gamePk")) == str(payload["provider_event_id"])]
    if len(matches) != 1:
        return "EVENT_IDENTITY_AMBIGUOUS", None
    game = matches[0]
    try:
        if (iso(game["gameDate"]) != iso(payload["game_start_utc"]) or
                int(game["season"]) != int(payload["season"]) or
                any(receipts.team_id(game["teams"][side]["team"]["id"]) != payload[side + "_team_id"]
                    for side in ("home", "away"))):
            return "EVENT_IDENTITY_AMBIGUOUS", None
        same_day = [g for day in source["payload"]["dates"] for g in day.get("games", [])
                    if g.get("gamePk") != game.get("gamePk") and
                    {g["teams"][s]["team"]["id"] for s in ("home", "away")} ==
                    {game["teams"][s]["team"]["id"] for s in ("home", "away")} and
                    scheduled_eastern_date({"start": g["gameDate"]}) ==
                    scheduled_eastern_date({"start": game["gameDate"]})]
        number = game.get("gameNumber")
        if same_day and (not isinstance(number, int) or number < 1 or
                         any(g.get("gameNumber") == number for g in same_day)):
            return "DOUBLEHEADER_IDENTITY_AMBIGUOUS", None
        return None, number if isinstance(number, int) else None
    except (KeyError, ValueError, TypeError):
        return "EVENT_IDENTITY_AMBIGUOUS", None


def _prior_check(payload, observations):
    for game in payload.get("prior_games", []):
        key = game.get("observation_hash")
        source = observations.get(key)
        if not source or digest(source) != key:
            return "SOURCE_LINEAGE_UNVERIFIED"
        try:
            replay = receipts.prior_from_observation(source)
        except (KeyError, ValueError, TypeError):
            return "FEATURE_ASOF_UNAVAILABLE"
        if replay != game or timestamp(game["available_at"]) > timestamp(payload["quote"]["observed_at"]):
            return "FEATURE_ASOF_UNAVAILABLE"
    return None


def classify_receipt(snapshot, outcome, observations):
    """Return a sanitized row and explicit blockers for one immutable receipt."""
    payload = snapshot.get("payload", {})
    quote = payload.get("quote", {})
    result = {"game_id": str(payload.get("provider_event_id", "")),
              "receipt_hash": snapshot.get("sha256"), "market_type": quote.get("market_type"),
              "price_status": "NO_VERIFIED_PRICE", "blockers": []}
    problems = []
    try:
        scope = scope_for(quote.get("market_type"))
        result["scope"] = scope
        if not isinstance(quote.get("line"), (int, float)) or isinstance(quote.get("line"), bool):
            problems.append("LINE_UNVERIFIED")
        if quote.get("decimal_odds") is None:
            result["price_status"] = "LINE_PRESENT_PRICE_MISSING"
            problems.append("NO_VERIFIED_PREGAME_PRICE")
        elif not quote.get("observed_at"):
            result["price_status"] = "PRICE_PRESENT_TIMESTAMP_UNVERIFIED"
            problems.append("QUOTE_TIMESTAMP_UNVERIFIED")
        elif not quote.get("sportsbook"):
            problems.append("NO_VERIFIED_PREGAME_PRICE")
        if not problems:
            p, values = exact_feature_values(snapshot)
            if timestamp(p["quote"]["observed_at"]) >= timestamp(p["game_start_utc"]):
                problems.append("QUOTE_TIMESTAMP_UNVERIFIED")
            for check in (_quote_source_check(p, observations), _prior_check(p, observations)):
                if check:
                    problems.append(check)
            schedule_problem, game_number = _schedule_check(p, observations)
            if schedule_problem:
                problems.append(schedule_problem)
            result["game_number"] = game_number
            result["feature_values"] = values
            result["feature_hash"] = digest(values)
            if not problems:
                result["price_status"] = "VERIFIED_PREGAME_PRICE"
        if outcome is None:
            problems.append("RESULT_UNVERIFIED")
        elif not problems or all(x == "RESULT_UNVERIFIED" for x in problems):
            if old_model.identity(outcome) != old_model.identity(payload):
                problems.append("RESULT_CONFLICT")
            else:
                source = observations.get(outcome.get("observation_hash"))
                if not source or digest(source) != outcome["observation_hash"]:
                    problems.append("SOURCE_LINEAGE_UNVERIFIED")
                else:
                    receipts.verify_outcome_source(outcome, source)
                    if timestamp(outcome["available_at"]) <= timestamp(quote["observed_at"]):
                        problems.append("RESULT_LEAKAGE_RISK")
                    result["label"] = exact_label(quote["market_type"], quote["line"],
                        outcome.get("home_score"), outcome.get("away_score"), outcome.get("status"))
                    if result["label"] in ("VOID", "NEEDS_REVIEW"):
                        problems.append("SETTLEMENT_UNREPRODUCIBLE")
        if outcome is not None:
            result["outcome_available_at"] = outcome.get("available_at")
        result.update(season=int(payload["season"]), game_start_utc=iso(payload["game_start_utc"]),
                      quote_observed_at=iso(quote["observed_at"]) if quote.get("observed_at") else None,
                      selection=quote.get("market_type"), line=quote.get("line"),
                      decimal_odds=quote.get("decimal_odds"), sportsbook=quote.get("sportsbook"),
                      home_team_id=payload["home_team_id"], away_team_id=payload["away_team_id"],
                      provider_namespace="mlb", provider_event_id=payload["provider_event_id"],
                      odds_provider_event_id=quote.get("provider_event_id"),
                      outcome_id=receipts.event_key(payload) if outcome else None)
    except (KeyError, ValueError, TypeError, OverflowError):
        problems.append("SOURCE_LINEAGE_UNVERIFIED")
    result["blockers"] = sorted(set(problems))
    result["training_status"] = "TRAINING_READY" if not result["blockers"] else "TRAINING_BLOCKED"
    return result


def select_manifest(rows):
    """One legal independent game per exact market, fixed before labels."""
    by_game = defaultdict(list)
    for row in rows:
        if row["training_status"] == "TRAINING_READY":
            by_game[(row["scope"], row["game_id"])].append(row)
    manifest = []
    for (scope, game_id), candidates in sorted(by_game.items()):
        priority = ("spread_home", "spread_away") if scope.endswith("RUN_LINE") else ("total_over", "total_under")
        candidates.sort(key=lambda r: (priority.index(r["market_type"]), r["receipt_hash"]))
        selected = candidates[0]
        core = {k: selected[k] for k in ("scope", "game_id", "season", "game_start_utc",
            "provider_namespace", "provider_event_id", "odds_provider_event_id", "game_number",
            "home_team_id", "away_team_id", "market_type", "selection", "line", "decimal_odds",
            "sportsbook", "quote_observed_at", "receipt_hash", "outcome_id",
            "outcome_available_at", "feature_hash", "label")}
        core["selection_policy_version"] = POLICY
        core["independent_game_market_id"] = digest([scope, game_id])
        core["manifest_id"] = digest(core)
        manifest.append(core)
    if len(manifest) != len({(r["scope"], r["game_id"]) for r in manifest}):
        raise ValueError("DUPLICATE_INDEPENDENT_GAME")
    return manifest


def receipt_reports(path):
    observations = receipts.read("observations", path)
    snapshots = receipts.read("receipts", path)
    outcomes = receipts.read("outcomes", path)
    rows = []
    for key, snapshot in sorted(snapshots.items()):
        payload = snapshot.get("payload", {})
        row = classify_receipt(snapshot, outcomes.get(receipts.event_key(payload)), observations)
        if key != receipts.receipt_key(payload):
            row["blockers"] = sorted(set(row["blockers"] + ["SOURCE_LINEAGE_UNVERIFIED"]))
            row["training_status"] = "TRAINING_BLOCKED"
        rows.append(row)
    manifest = select_manifest(rows)
    summary = {}
    for scope in SCOPES:
        subset = [x for x in rows if x.get("scope") == scope]
        legal = [x for x in manifest if x["scope"] == scope]
        summary[scope] = {"raw_receipts": len(subset),
            "unique_games": len({x["game_id"] for x in subset}),
            "exact_line_rows": sum(x.get("line") is not None for x in subset),
            "verified_pregame_price_rows": sum(x["price_status"] == "VERIFIED_PREGAME_PRICE" for x in subset),
            "final_result_rows": sum(x.get("outcome_id") is not None for x in subset),
            "reproducible_settlement_rows": sum(x.get("label") in CLASSES[scope.split("/")[1]] for x in subset),
            "asof_feature_rows": sum(x.get("feature_hash") is not None for x in subset),
            "training_ready_raw_rows": sum(x["training_status"] == "TRAINING_READY" for x in subset),
            "legal_independent_n": len(legal),
            "legal_by_season": dict(sorted(Counter(str(x["season"]) for x in legal).items())),
            "class_balance": dict(Counter(x["label"] for x in legal)),
            "blocker_counts": dict(sorted(Counter(b for x in subset for b in x["blockers"]).items())),
            "price_status_counts": dict(sorted(Counter(x["price_status"] for x in subset).items())),
            "first_start": min((x.get("game_start_utc") for x in subset if x.get("game_start_utc")), default=None),
            "last_start": max((x.get("game_start_utc") for x in subset if x.get("game_start_utc")), default=None)}
    state = Counter()
    for row in rows:
        state["matched" if row["training_status"] == "TRAINING_READY" else
              "unmatched" if "RESULT_UNVERIFIED" in row["blockers"] else "ambiguous"] += 1
        for name, reasons in {"price_conflict": ("PRICE_CONFLICT",),
                              "line_conflict": ("LINE_CONFLICT",),
                              "chronology_failure": ("QUOTE_TIMESTAMP_UNVERIFIED", "RESULT_LEAKAGE_RISK"),
                              "result_conflict": ("RESULT_CONFLICT",),
                              "duplicate": ("DUPLICATE_INDEPENDENT_GAME",)}.items():
            if any(reason in row["blockers"] for reason in reasons):
                state[name] += 1
    for name in ("matched", "unmatched", "ambiguous", "duplicate", "price_conflict", "line_conflict",
                 "chronology_failure", "result_conflict"):
        state.setdefault(name, 0)
    return rows, manifest, summary, {"schema": SCHEMA, "receipt_rows": len(rows),
        "observation_rows": len(observations), "outcome_rows": len(outcomes),
        "statuses": dict(state), "scope_counts": {k: summary[k]["blocker_counts"] for k in SCOPES},
        "repairs_performed": 0, "source_proven_defects_only": True}


def _source_summary(name, rows, *, candidate_legal_use=False, note=None):
    dates = [str(r.get("scheduled_start") or r.get("game_start_utc") or
                 r.get("prediction_timestamp") or r.get("quote_timestamp") or "") for r in rows]
    dates = [d for d in dates if len(d) >= 10 and d[:4].isdigit()]
    games = {str(r.get("game_id") or r.get("provider_event_id") or r.get("event_id"))
             for r in rows if r.get("game_id") or r.get("provider_event_id") or r.get("event_id")}
    prices = sum(r.get("price") is not None or r.get("decimal_odds") is not None for r in rows)
    lines = sum(r.get("line") is not None for r in rows)
    return {"source": name, "record_count": len(rows), "unique_games_or_events": len(games),
            "seasons": dict(sorted(Counter(d[:4] for d in dates).items())),
            "date_range": [min(dates, default=None), max(dates, default=None)],
            "line_present_rows": lines, "price_present_rows": prices,
            "identity_quality": "SOURCE_SPECIFIC_NOT_TRAINING_CERTIFIED",
            "candidate_legal_training_use": candidate_legal_use,
            "note": note or "No automatic join to the immutable MLB receipt manifest"}


def local_csv_inventory(root):
    """Inventory repository research exports; never infer quote timestamps."""
    paths = [Path(root) / "data/master_all_sports.csv",
             Path(root) / "data/theover_spreads.csv", Path(root) / "data/theover_totals.csv"]
    paths += sorted((Path(root) / "data/backtest_exports").glob("*.csv"))
    output = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = [r for r in reader if str(r.get("League") or r.get("league") or
                                              r.get("sport") or r.get("Sport") or "").upper() == "MLB"]
        normalized = []
        for row in rows:
            line = next((row[name] for name in ("Spread Line", "Total Line", "market_line_used", "line")
                         if row.get(name) not in (None, "")), None)
            price = next((row[name] for name in ("Odds American", "odds_american", "price")
                          if row.get(name) not in (None, "")), None)
            normalized.append({"game_id": row.get("game_id"),
                "scheduled_start": row.get("Game Date") or row.get("commence_time"),
                "line": line, "price": price})
        statuses = Counter("PRICE_PRESENT_TIMESTAMP_UNVERIFIED" if item["price"] is not None else
                           "RESEARCH_LINE_ONLY" if item["line"] is not None else "NO_VERIFIED_PRICE"
                           for item in normalized)
        output.append({**_source_summary(path.relative_to(root).as_posix(), normalized),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stable_identity_rows": 0, "verified_pregame_price_rows": 0,
            "quote_timestamp_present_rows": 0, "price_status_counts": dict(statuses),
            "reproducible_settlement_rows": 0, "training_ready_rows": 0,
            "blockers": {"SOURCE_LINEAGE_UNVERIFIED": len(rows),
                         "QUOTE_TIMESTAMP_UNVERIFIED": len(rows),
                         "FEATURE_ASOF_UNAVAILABLE": len(rows)},
            "theover_model_hit_rate_is_game_probability": False})
    return output


def receipt_source_inventory(path):
    """Summarize retained provider observations and final feeds without raw data."""
    by_source = defaultdict(list)
    for observation in receipts.read("observations", path).values():
        raw = observation.get("payload") or {}
        provider = observation.get("source") or "UNKNOWN"
        games = [game for day in raw.get("dates", []) for game in day.get("games", [])]
        event_ids = {str(game.get("gamePk")) for game in games if game.get("gamePk") is not None}
        if raw.get("id") is not None:
            event_ids.add(str(raw["id"]))
        if raw.get("gamePk") is not None:
            event_ids.add(str(raw["gamePk"]))
        markets = [market for book in raw.get("bookmakers", [])
                   for market in book.get("markets", [])
                   if market.get("key") in ("spreads", "totals")]
        outcomes = [entry for market in markets for entry in market.get("outcomes", [])]
        by_source[provider].append({
            "observed_at": observation.get("observed_at"),
            "event_ids": event_ids,
            "line": 1 if any(entry.get("point") is not None for entry in outcomes) else None,
            "price": 1 if any(entry.get("price") is not None for entry in outcomes) else None,
            "scheduled_start": raw.get("commence_time") or
                               (games[0].get("gameDate") if games else None),
        })
    sources = []
    for provider, items in sorted(by_source.items()):
        summary = _source_summary("receipt_observations/" + provider, items)
        summary.update(
            unique_games_or_events=len(set().union(*(item["event_ids"] for item in items))),
            timestamp_present_rows=sum(bool(item["observed_at"]) for item in items),
            lineage_hash_verified=True, candidate_legal_training_use=False,
            training_ready_rows=0,
            note="Provider observations support exact receipt replay; they are not independent training rows",
        )
        sources.append(summary)
    outcomes = list(receipts.read("outcomes", path).values())
    result_source = _source_summary("receipt_outcomes/mlb_statsapi", [
        {"game_id": item.get("provider_event_id"),
         "scheduled_start": item.get("game_start_utc")}
        for item in outcomes])
    result_source.update(
        final_result_rows=sum(item.get("status") == "FINAL" for item in outcomes),
        timestamp_present_rows=sum(bool(item.get("available_at")) for item in outcomes),
        lineage_hash_verified=True, candidate_legal_training_use=False,
        training_ready_rows=0,
        note="Final results require exact receipt, quote and settlement replay before training use",
    )
    sources.append(result_source)
    return sources


def read_canonical_remote(client, folder, path):
    """Verify every canonical object without uploading or changing the source."""
    from app_core import prospective_evidence, prospective_reconciliation, prospective_remote
    prospective_reconciliation.ensure_reconciliation_schema(path)
    with closing(prospective_evidence.connect(path)) as db:
        schema = prospective_remote._schema(db)
    tables = defaultdict(list)
    verified = 0
    for key, raw in prospective_remote._remote_objects(client, folder):
        table, row = prospective_remote._decode(key, raw, schema)
        verified += 1
        columns = schema[table][0]
        item = dict(zip(columns, row))
        if item.get("sport") == "MLB":
            tables[table].append(item)
    return dict(tables), verified


def read_native_remote(client):
    from app_core.mlb_prospective_store import PREFIX, encode
    records = []
    for key, raw in client.read_objects(Prefix=PREFIX):
        if len(raw) > 40_000_000 or key != PREFIX + hashlib.sha256(raw).hexdigest() + ".json":
            raise ValueError("MLB_NATIVE_REMOTE_INTEGRITY_FAILURE")
        item = json.loads(raw)
        if encode(item) != raw:
            raise ValueError("MLB_NATIVE_REMOTE_CANONICAL_MISMATCH")
        records.append(item)
    return records


def read_snapshot_remote(client, path):
    from app_core import evidence_remote
    import pandas as pd
    from io import StringIO
    restored = evidence_remote.restore(path, client=client)
    with closing(sqlite3.connect(path)) as db:
        snapshots = db.execute("SELECT snapshot_id,generated_at,candidates FROM snapshots").fetchall()
        closes = [json.loads(raw) for (raw,) in db.execute("SELECT payload FROM closing_observations")]
        plans = [json.loads(raw) for (raw,) in db.execute("SELECT payload FROM validation_plans WHERE sport='MLB'")]
        bundles = db.execute("SELECT version,frozen_at FROM bundles").fetchall()
    mlb_candidates = []
    for snapshot_id, generated_at, raw in snapshots:
        try:
            frame = pd.read_csv(StringIO(raw))
        except pd.errors.EmptyDataError:
            continue
        for item in frame.to_dict("records"):
            if str(item.get("sport") or item.get("league") or "").upper() == "MLB":
                mlb_candidates.append({"snapshot_id": snapshot_id, "prediction_timestamp": generated_at,
                                       "game_id": item.get("game_id"), "market_type": item.get("market_type"),
                                       "line": item.get("market_line_used"), "price": item.get("odds_american"),
                                       "quote_verified": item.get("quote_binding_verified") is True,
                                       "quote_timestamp": item.get("odds_recorded_at")})
    return {"snapshots_restored": restored, "snapshots_total": len(snapshots),
            "bundles": len(bundles), "mlb_candidates": mlb_candidates,
            "mlb_closes": [c for c in closes if c.get("quote", {}).get("sport") == "MLB"],
            "mlb_validation_plans": plans}


def feature_contract():
    fields = [
        ("home_ppg", "mean home-team runs scored in ten prior observed finals", "MLB final-feed observations"),
        ("home_oppg", "mean runs allowed by home team in ten prior observed finals", "MLB final-feed observations"),
        ("home_win_pct", "prior observed final-game wins / ten", "MLB final-feed observations"),
        ("away_ppg", "mean away-team runs scored in ten prior observed finals", "MLB final-feed observations"),
        ("away_oppg", "mean runs allowed by away team in ten prior observed finals", "MLB final-feed observations"),
        ("away_win_pct", "prior observed final-game wins / ten", "MLB final-feed observations"),
        ("exact_line", "selected sportsbook's exact captured run line or total", "Odds API observation"),
        ("price_implied_probability", "1 / selected quote decimal odds; no-vig not asserted", "Odds API observation"),
    ]
    return {"schema": SCHEMA, "feature_version": FEATURE_VERSION,
        "scopes": {scope: {"target_classes": CLASSES[scope.split("/")[1]],
                           "features": [dict(name=n, definition=d, source=s,
                                             availability="no later than selected quote observed_at",
                                             missing_policy="block_row") for n, d, s in fields]}
                   for scope in SCOPES},
        "forbidden": ["same-event result/score", "post-first-pitch features",
            "season-end statistics substituted for earlier snapshots", "future starter/lineup/injury",
            "unavailable closing price", "future standings/ratings", "TheOver ModelHitRate as probability"],
        "other_candidate_features": "Excluded until versioned historical as-of snapshots exist"}


def chronological_plan(manifest):
    """Freeze whole-game assignments before any fitting or outcome scoring."""
    games = {}
    for row in manifest:
        old = games.setdefault(row["game_id"], row["game_start_utc"])
        if old != row["game_start_utc"]:
            raise ValueError("SAME_GAME_START_CONFLICT")
    ordered = sorted(games, key=lambda g: (games[g], g))
    cuts = [int(len(ordered) * share) for share in (0.5, 2/3, 5/6)]
    groups = [ordered[:cuts[0]], ordered[cuts[0]:cuts[1]],
              ordered[cuts[1]:cuts[2]], ordered[cuts[2]:]]
    assignment = {game: part for part, group in zip(PARTITIONS, groups) for game in group}
    if len(assignment) != len(games):
        raise ValueError("SAME_GAME_CROSSES_PARTITIONS")
    plan = {}
    for scope in SCOPES:
        rows = [r for r in manifest if r["scope"] == scope]
        parts = {name: [r for r in rows if assignment[r["game_id"]] == name] for name in PARTITIONS}
        blockers = [f"{name.upper()}_INDEPENDENT_N_BELOW_{MINIMUM[name]}" for name in PARTITIONS
                    if len(parts[name]) < MINIMUM[name]]
        seasons = {r["season"] for r in rows}
        if len(seasons) < 2:
            blockers.append("TWO_DISTINCT_SEASONS_REQUIRED")
        labels = Counter(r["label"] for r in parts["development"])
        target_classes = CLASSES[scope.split("/")[1]]
        if any(labels[k] < 30 for k in (target_classes[0], target_classes[2])):
            blockers.append("DEVELOPMENT_CLASS_SUPPORT_INSUFFICIENT")
        # A half-run/half-total line cannot push with integer MLB scores.
        # Integer-line inference needs observed push support; otherwise the
        # model must be constrained to half lines with P(PUSH)=0.
        integer_lines = [r for r in parts["development"] if float(r["line"]).is_integer()]
        if integer_lines and labels["PUSH"] < 10:
            blockers.append("INTEGER_LINE_PUSH_SUPPORT_INSUFFICIENT")
        for left, right in zip(PARTITIONS, PARTITIONS[1:]):
            if parts[left] and parts[right] and max(timestamp(r["outcome_available_at"]) for r in parts[left]) >= \
                    min(timestamp(r["quote_observed_at"]) for r in parts[right]):
                blockers.append("RESULT_AVAILABILITY_CROSSES_" + right.upper())
        plan[scope] = {"strategy": "CHRONOLOGICAL_WHOLE_GAME_NO_RANDOM_HOLDOUT",
            "partition_n": {name: len(parts[name]) for name in PARTITIONS},
            "minimum_independent_n": MINIMUM, "seasons": sorted(seasons),
            "class_balance_development": dict(labels),
            "line_support": "HALF_LINES_ONLY" if not integer_lines else "INTEGER_AND_HALF_LINES_REQUIRE_PUSH_SUPPORT",
            "boundaries": {name: {"first_pitch": min((r["game_start_utc"] for r in parts[name]), default=None),
                                 "last_pitch": max((r["game_start_utc"] for r in parts[name]), default=None)}
                           for name in PARTITIONS},
            "assignment_hash": digest(sorted((game, assignment[game]) for game in assignment)),
            "blockers": blockers,
            "status": "FEASIBLE" if not blockers else "INSUFFICIENT_TRAINING_EVIDENCE"}
    return {"schema": SCHEMA, "selection_policy_version": POLICY,
            "split_policy": "50/16.7/16.7/16.7 percent; whole games across both markets; declared before fitting",
            "methodology": "Eight pregame inputs plus intercept imply up to 18 three-class logits. Development floor 240 gives about 13 games per parameter before class imbalance; three disjoint 80-game windows and two seasons constrain noisy calibration and drift. Require 30 examples per decided class. Half-point lines have structural P(PUSH)=0 with integer scores; integer-line inference requires at least ten observed development pushes. These are engineering safeguards, not power guarantees.",
            "scopes": plan}


def _canonical_inventory(tables):
    sources = []
    for table, rows in sorted(tables.items()):
        by_scope = defaultdict(list)
        for row in rows:
            market = row.get("market_family")
            scope = "MLB/" + market if market in ("RUN_LINE", "TOTAL") else "MLB/UNASSIGNED"
            by_scope[scope].append(row)
        for scope, subset in sorted(by_scope.items()):
            source = _source_summary("canonical_prospective/" + table + "/" + scope, subset)
            source.update(market_family=scope.split("/")[1],
                lineage_hash_rows=sum(bool(r.get("payload_hash") or r.get("source_hash")) for r in subset),
                verified_price_rows=sum(bool(r.get("quote_verified")) and r.get("line") is not None and
                                        r.get("price") is not None and r.get("quote_timestamp") is not None
                                        for r in subset),
                result_rows=sum(bool(r.get("result_id")) or table == "prospective_result" for r in subset),
                settlement_rows=sum(r.get("settlement_status") in ("WIN", "LOSS", "PUSH", "VOID")
                                    for r in subset),
                training_ready_rows=0, blockers={"NOT_SELECTED_BY_MLB_RECEIPT_MANIFEST": len(subset)})
            sources.append(source)
    return sources


def model_audit(canonical, native=()):
    """Classify actual registered targets, never trust a file's model name."""
    entries = [
        {"source": "app_core/mlb_spread_total_model.py", "actual_target": "WIN_CONDITIONAL_ON_DECISION",
         "features": "six prior-scoring summaries plus exact reference line",
         "algorithm": "separate binary ridge logistic per family, pushes excluded",
         "training_source": "immutable MLB receipt export if operator trained it",
         "current_use": "namespaced research challenger only",
         "classification": "WRONG_TARGET", "exact_scope_reusable": False},
        {"source": "app_core/mlb_prospective.py", "actual_target": "RAW_MARGIN_AND_TOTAL_POINTS",
         "features": "team scoring and probable-pitcher history",
         "algorithm": "ridge regression on 2023 checkpoint",
         "training_source": "historical MLB checkpoint; no exact sportsbook quote",
         "current_use": "research paired-score forecast",
         "classification": "WRONG_TARGET", "exact_scope_reusable": False},
    ]
    for row in canonical.get("prospective_model", []):
        target = row.get("probability_semantics") or row.get("target_semantics")
        entries.append({"source": "canonical_prospective/prospective_model",
            "model_id": row.get("model_id"), "market_family": row.get("market_family"),
            "actual_target": target, "feature_version": row.get("feature_version"),
            "training_cutoff": row.get("training_cutoff"), "available_at": row.get("available_at"),
            "artifact_hash": row.get("artifact_hash") or row.get("payload_hash"),
            "current_use": "canonical research evidence only",
            "classification": "PROVENANCE_INCOMPLETE" if target in ("EXACT_RUN_LINE_COVER", "EXACT_TOTAL_OVER_UNDER")
                              else "WRONG_TARGET", "exact_scope_reusable": False})
    for record in native:
        if record.get("kind") != "model":
            continue
        data = record.get("data") or {}
        entries.append({"source": "mlb_native/model",
            "record_hash": digest(record), "created_at": record.get("created_at"),
            "actual_target": "RAW_MARGIN_AND_TOTAL_POINTS" if data.get("models") else None,
            "features": "team scoring and probable-pitcher history",
            "algorithm": "ridge regression" if data.get("ridge_alpha") is not None else None,
            "training_source": "historical MLB checkpoint" if data.get("history_hash") else None,
            "training_cutoff": data.get("train_year"),
            "training_rows": data.get("train_rows"),
            "runtime_hash": data.get("runtime_hash"),
            "current_use": "native research evidence only",
            "classification": "WRONG_TARGET" if data.get("models") else "PROVENANCE_INCOMPLETE",
            "exact_scope_reusable": False})
    return entries


def validate_research_prediction(model, receipt, feature_snapshot, when):
    """Pure fail-closed gate; this audit never creates a live prediction."""
    if not isinstance(model, dict) or model.get("model_status") != "RESEARCH_MODEL_SELECTED":
        raise ValueError("NO_VALID_EXACT_SCOPE_MODEL")
    artifact = model.get("model_artifact") or {}
    payload, values = exact_feature_values(receipt)
    quote = payload["quote"]
    scope = scope_for(quote["market_type"])
    if (artifact.get("schema") != SCHEMA or artifact.get("scope") != scope or
            tuple(artifact.get("target_classes") or ()) != CLASSES[scope.split("/")[1]] or
            artifact.get("feature_version") != FEATURE_VERSION or
            artifact.get("model_id") != model.get("model_id") or
            artifact.get("artifact_hash") != digest({k: v for k, v in artifact.items()
                                                      if k not in ("model_id", "artifact_hash")}) or
            feature_snapshot.get("receipt_hash") != receipt["sha256"] or
            feature_snapshot.get("feature_values") != values or
            feature_snapshot.get("game_id") != str(payload["provider_event_id"]) or
            feature_snapshot.get("market_type") != quote["market_type"] or
            float(feature_snapshot.get("line", float("nan"))) != float(quote["line"]) or
            float(feature_snapshot.get("decimal_odds", float("nan"))) != float(quote["decimal_odds"]) or
            not timestamp(artifact["training_cutoff"]) <= timestamp(artifact["available_at"]) <=
                timestamp(when) < timestamp(payload["game_start_utc"]) or
            timestamp(quote["observed_at"]) > timestamp(when) or
            artifact.get("deployment_state") != "UNVALIDATED" or
            artifact.get("production_eligible") is not False or artifact.get("stake") != 0):
        raise ValueError("RESEARCH_PREDICTION_EVIDENCE_MISMATCH")
    return {"model_id": model["model_id"], "receipt_hash": receipt["sha256"],
            "game_id": str(payload["provider_event_id"]), "scope": scope,
            "prediction_timestamp": iso(when), "deployment_state": "UNVALIDATED",
            "production_eligible": False, "production_parlay_eligible": False,
            "stake": 0, "wager_eligible": False}


def build_reports(receipt_path, root, *, source_commit, remote_verification,
                  canonical=None, native=None, snapshots=None, now=None):
    """Produce twelve sanitized, deterministic reports from verified sources."""
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("SOURCE_COMMIT_REQUIRED")
    if not isinstance(remote_verification, dict) or any(
            remote_verification.get(k) is not True for k in
            ("receipt_restore_verified", "receipt_readback_verified", "canonical_restore_verified",
             "native_restore_verified", "snapshot_restore_verified")):
        raise ValueError("AUTHENTICATED_REMOTE_READBACK_REQUIRED")
    canonical = canonical or {}
    native = native or []
    snapshots = snapshots or {}
    rows, manifest, readiness, reconciliation = receipt_reports(receipt_path)
    split = chronological_plan(manifest)
    local = local_csv_inventory(Path(root))
    native_by_kind = defaultdict(list)
    for item in native:
        native_by_kind[item.get("kind", "UNKNOWN")].append(item)
    native_sources = []
    for kind, items in sorted(native_by_kind.items()):
        source = _source_summary("mlb_native/" + kind, [dict(i.get("data") or {},
            prediction_timestamp=i.get("created_at")) for i in items])
        source.update(source_hash_verified=True, training_ready_rows=0,
                      blockers={"NOT_EXACT_QUOTE_TARGET": len(items)})
        native_sources.append(source)
    receipt_sources = []
    for scope in SCOPES:
        x = readiness[scope]
        receipt_sources.append({"source": "mlb_immutable_receipts/" + scope,
            "record_count": x["raw_receipts"], "unique_games_or_events": x["unique_games"],
            "seasons": dict(sorted(Counter(str(r["season"]) for r in rows if r.get("scope") == scope and r.get("season")).items())),
            "date_range": [x["first_start"], x["last_start"]],
            "stable_identity_rows": x["raw_receipts"], "exact_line_rows": x["exact_line_rows"],
            "verified_pregame_price_rows": x["verified_pregame_price_rows"],
            "final_result_rows": x["final_result_rows"],
            "reproducible_settlement_rows": x["reproducible_settlement_rows"],
            "asof_feature_rows": x["asof_feature_rows"],
            "training_ready_raw_rows": x["training_ready_raw_rows"],
            "legal_independent_n": x["legal_independent_n"],
            "lineage_hash_verified": True, "blockers": x["blocker_counts"],
            "candidate_legal_training_use": True})
    snapshot_source = _source_summary("prediction_snapshots/MLB",
        snapshots.get("mlb_candidates", []))
    snapshot_source.update(training_ready_rows=0,
        blockers={"NOT_SELECTED_BY_MLB_RECEIPT_MANIFEST": snapshot_source["record_count"]})
    inventory = {"schema": SCHEMA, "source_commit": source_commit,
        "remote_verification": remote_verification,
        "sources": receipt_sources + receipt_source_inventory(receipt_path) +
                   _canonical_inventory(canonical) + native_sources +
                   [snapshot_source] + local,
        "source_categories": ["canonical prospective evidence", "MLB native evidence",
            "receipt/reconciliation stores", "historical odds/price", "The Odds API",
            "schedule/event identity", "results/settlements", "models/calibrations",
            "validation plans", "paper predictions", "close/CLV", "TheOver research"],
        "no_unverified_history_promoted": True}
    manifest_report = {"schema": SCHEMA, "selection_policy_version": POLICY,
        "unit": "ONE_CANONICAL_MLB_GAME_PER_EXACT_MARKET",
        "manifest_hash": digest(manifest), "rows": manifest,
        "raw_books_and_reprices_do_not_increase_n": True}
    features = feature_contract()
    baselines, comparison, models, calibration, validation = {}, {}, {}, {}, {}
    existing = model_audit(canonical, native)
    for scope in SCOPES:
        blocked = split["scopes"][scope]["blockers"]
        # Historical candidate models are deliberately not retrofitted into
        # this exact three-class, selected-manifest contract.
        baselines[scope] = {"status": "NOT_EVALUATED", "reason": "INSUFFICIENT_TRAINING_EVIDENCE",
            "planned": ["Laplace-smoothed three-class base rate", "exact stored-price benchmark"],
            "metrics": ["multiclass Brier", "log loss", "calibration error", "coverage",
                        "hit rate", "nonpush AUC where defined", "class balance", "push rate", "n"],
            "uncertainty": "No interval from the current sample; scores cannot be estimated without legal split cohorts"}
        comparison[scope] = {"status": "NOT_TRAINED", "candidates": [],
            "planned_candidate": "regularized exact-scope three-class multinomial logistic after gates pass"}
        models[scope] = {"model_status": "INSUFFICIENT_TRAINING_EVIDENCE",
            "model_id": None, "artifact_hash": None,
            "independent_n": readiness[scope]["legal_independent_n"], "blockers": blocked}
        calibration[scope] = {"calibration_status": "INSUFFICIENT_EVIDENCE",
            "calibration_id": None, "blockers": ["NO_SELECTED_EXACT_SCOPE_MODEL"],
            "distinct_chronological_cohort_required": True}
        validation[scope] = {"status": "TRAINING_EVIDENCE_BLOCKED" if blocked else "MODEL_PERFORMANCE_BLOCKED",
            "model_id": None, "calibration_id": None,
            "validation_plan_ids": [r.get("validation_plan_id") for r in canonical.get("prospective_validation_plan", [])
                                    if r.get("market_family") == scope.split("/")[1]],
            "frozen_plan_retrospectively_modified": False,
            "blockers": blocked if blocked else ["NO_SELECTED_EXACT_SCOPE_MODEL"],
            "deployment_state": "UNVALIDATED", "production_eligible": False,
            "production_parlay_eligible": False, "stake": 0, "wager_eligible": False}
    closes = canonical.get("prospective_close", [])
    other_closes = snapshots.get("mlb_closes", [])
    close_report = {"schema": SCHEMA,
        "canonical_mlb_close_rows": len(closes),
        "legacy_near_start_observations": len(other_closes),
        "certified_comparable_close_rows": 0,
        "clv_status": "UNAVAILABLE",
        "reason": "No independently certified same-game/market/selection/book/line replayable closing series in this audit",
        "near_start_candidate_is_certified_close": False,
        "frozen_plans_modified": False}
    reconciliation.update(remote_restore_verified=True, remote_readback_verified=True,
                          receipt_backup_ids=remote_verification.get("receipt_backup_ids", []))
    return {"inventory": inventory, "receipt_audit": reconciliation,
        "training": {"schema": SCHEMA, "scopes": readiness,
                     "price_classifications": ("VERIFIED_PREGAME_PRICE", "PRICE_PRESENT_TIMESTAMP_UNVERIFIED",
                        "LINE_PRESENT_PRICE_MISSING", "RESEARCH_LINE_ONLY", "NO_VERIFIED_PRICE")},
        "manifest": manifest_report, "features": features, "split": split,
        "baselines": {"schema": SCHEMA, "scopes": baselines},
        "comparison": {"schema": SCHEMA, "scopes": comparison},
        "models": {"schema": SCHEMA, "source_commit": source_commit,
                   "existing_models": existing, "scopes": models},
        "calibration": {"schema": SCHEMA, "scopes": calibration},
        "validation": {"schema": SCHEMA, "scopes": validation,
                       "research_prediction_status": "BLOCKED_NO_VALID_MODEL",
                       "no_validation_promotion_or_wager": True},
        "close": close_report}


def write_reports(reports, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if set(reports) != set(REPORTS):
        raise ValueError("REQUIRED_REPORT_SET_INCOMPLETE")
    for key, filename in REPORTS.items():
        (output / filename).write_text(json.dumps(reports[key], indent=2, sort_keys=True,
                                                allow_nan=False) + "\n", encoding="utf-8")
    return {key: str(output / filename) for key, filename in REPORTS.items()}
