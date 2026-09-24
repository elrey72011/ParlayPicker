"""Read-only legacy reconciliation into a separate canonical research ledger.

The live ``prospective_prediction`` tables require a prediction made now,
before a future start.  Historical producer rows belong in this immutable
ledger instead.  They remain visible and source-bound, but can never satisfy
the prospective validation or wager gates.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from app_core.prediction_evidence import database_path
from app_core.prospective_evidence import EvidenceConflict, SPORT_MARKETS, connect as canonical_connect
from app_core.prospective_legacy_view import _row, _time
from app_core.prospective_source_view import SOURCE_FILENAMES, source_evidence


LEGACY_SPORTS = frozenset({"NFL", "NCAAF", "MLB"})
SOURCE_TABLES = {"NFL": ("records",), "NCAAF": ("records",),
                 "MLB": ("observations", "receipts", "outcomes")}
_MUTABLE_PROJECTION_FIELDS = frozenset({
    "result_outcome", "result_available_at", "result_source", "result_home_score",
    "result_away_score", "close_status", "closing_proxy_price_clv", "verified_clv",
    "blockers", "model_available_at", "runtime_hash", "model_id",
})
_LINEAGE_FIELDS = frozenset({"observation_id", "source_record_id", "evidence_snapshot_id",
                             "evidence_hash", "feature_snapshot_id", "source_store"})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _key(*parts):
    return _hash(_json(parts))


def _canonical_path(path):
    return Path(path) if path is not None else database_path().with_name("prospective-evidence.sqlite3")


def _source_path(sport, path):
    return Path(path) if path is not None else database_path().with_name(SOURCE_FILENAMES[sport])


def _read_sources(sport, path):
    """Verify exact source bytes through a mode=ro connection; never initialize it."""
    older = path.with_name("mlb-prospective.sqlite3") if sport == "MLB" else None
    if not path.is_file() and (older is None or not older.is_file()):
        return []
    sources = []
    specs = [(path, table, table) for table in SOURCE_TABLES[sport] if path.is_file()]
    if sport == "MLB":
        if older.is_file():
            specs.append((older, "records", "mlb_prospective_records"))
    for file_path, table, source_table in specs:
        uri = file_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            if table == "records":
                rows = db.execute("SELECT id,payload FROM records ORDER BY rowid").fetchall()
            else:
                rows = db.execute(f"SELECT id,sha256,payload FROM {table} ORDER BY rowid").fetchall()
            for item in rows:
                record_id, raw = item[0], item[-1]
                if not isinstance(raw, str) or not isinstance(record_id, str):
                    raise EvidenceConflict("legacy source record encoding invalid")
                try:
                    payload = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except (TypeError, ValueError):
                    raise EvidenceConflict("legacy source JSON invalid") from None
                source_hash = _hash(raw)
                if table == "records":
                    if source_hash != record_id:
                        raise EvidenceConflict("legacy source hash mismatch")
                else:
                    from app_core.mlb_spread_total_model import digest
                    if digest(payload) != item[1]:
                        raise EvidenceConflict("legacy source hash mismatch")
                sources.append({"source_key": _key(sport, source_table, record_id),
                                "sport": sport, "source_table": source_table,
                                "source_record_id": record_id,
                                "source_hash": source_hash, "raw_source": raw.encode(),
                                "record": payload})
    return sources


def _mlb_forecast_rows(sources):
    """Preserve older unpriced score forecasts without calling them wagers."""
    forecasts = [item for item in sources if item["source_table"] == "mlb_prospective_records"]
    models = {item["source_record_id"]: item for item in forecasts
              if item["record"].get("kind") == "model"}
    scores = {str(item["record"].get("data", {}).get("game_id")): item for item in forecasts
              if item["record"].get("kind") == "scores"}
    rows = []
    for source in forecasts:
        record = source["record"]
        if record.get("kind") != "capture":
            continue
        data = record.get("data", {})
        model_ref = data.get("model_id")
        model = models.get(model_ref)
        captured = data.get("observed_at") or record.get("created_at")
        for event in data.get("events", []):
            if not isinstance(event, dict):
                continue
            game_id = event.get("game_id")
            start = _time(event.get("start"))
            observed = _time(captured)
            if (type(game_id) is not int or game_id <= 0 or start is None or
                    observed is None or observed >= start or
                    type(event.get("home_id")) is not int or
                    type(event.get("away_id")) is not int or
                    event["home_id"] <= 0 or event["away_id"] <= 0 or
                    event["home_id"] == event["away_id"]):
                continue
            score_source = scores.get(str(event.get("game_id")))
            score = score_source["record"].get("data", {}) if score_source else {}
            features = event.get("features", {})
            if not isinstance(features, dict):
                continue
            for model_name, predictions in event.get("forecasts", {}).items():
                if not isinstance(predictions, dict):
                    continue
                for target, market in (("margin", "RUN_LINE"), ("total", "TOTAL")):
                    value = predictions.get(target)
                    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                            not math.isfinite(value)):
                        continue
                    selection = f"UNPRICED_{target.upper()}_FORECAST"
                    row = _row("MLB", market, source["source_record_id"], selection,
                        game_id=str(event.get("game_id")), provider_namespace="mlb",
                        provider_event_id=str(event.get("game_id")),
                        home_team_id=f"mlb:{event.get('home_id')}" if event.get("home_id") else None,
                        away_team_id=f"mlb:{event.get('away_id')}" if event.get("away_id") else None,
                        scheduled_start=event.get("start"), quote_verified=False,
                        model_id=model_ref if model else None,
                        model_available_at=model["record"].get("created_at") if model else None,
                        feature_version="mlb-legacy-team-starter-features",
                        feature_snapshot_id=_hash(_json(features)), feature_frozen_at=captured,
                        prediction_timestamp=captured,
                        evidence_snapshot_id=source["source_record_id"],
                        evidence_hash=source["source_hash"],
                        runtime_hash=model["record"].get("data", {}).get("runtime_hash") if model else None,
                        legacy_model_name=model_name, legacy_referenced_model_id=model_ref,
                        legacy_forecast_target=target, legacy_forecast_value=float(value),
                        result_outcome="NEEDS_REVIEW" if score_source else None,
                        result_available_at=score_source["record"].get("created_at") if score_source else None,
                        result_source="legacy_mlb_score_without_raw_feed" if score_source else None,
                        result_home_score=score.get("home_score") if score_source else None,
                        result_away_score=score.get("away_score") if score_source else None)
                    row["source_store"] = "mlb_prospective"
                    row["blockers"] += ("UNPRICED_SCORE_FORECAST", "MISSING_REPLAYABLE_RESULT_SOURCE")
                    rows.append(row)
    return rows


def ensure_reconciliation_schema(path=None):
    """Create the two immutable canonical tables before remote restore."""
    with closing(canonical_connect(_canonical_path(path))) as db, db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS prospective_reconciled_source (
                source_key TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                source_table TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                raw_source BLOB NOT NULL,
                payload TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                UNIQUE(sport,source_table,source_record_id)
            );
            CREATE TABLE IF NOT EXISTS prospective_reconciled_fact (
                fact_id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                market_family TEXT NOT NULL,
                canonical_identity TEXT NOT NULL,
                capture_signature TEXT NOT NULL,
                source_key TEXT NOT NULL REFERENCES prospective_reconciled_source(source_key),
                payload TEXT NOT NULL,
                payload_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reconciled_fact_scope
                ON prospective_reconciled_fact(sport,market_family,canonical_identity);
        """)
        for table in ("prospective_reconciled_source", "prospective_reconciled_fact"):
            for action in ("UPDATE", "DELETE"):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action}_immutable "
                           f"BEFORE {action} ON {table} "
                           "BEGIN SELECT RAISE(ABORT,'append-only evidence'); END")


def _stored_source(db, source):
    old = db.execute("""SELECT sport,source_table,source_record_id,source_hash,raw_source,payload,payload_hash
        FROM prospective_reconciled_source WHERE source_key=?""", (source["source_key"],)).fetchone()
    metadata = {key: source[key] for key in ("source_key", "sport", "source_table",
                                             "source_record_id", "source_hash")}
    payload = _json(metadata)
    values = (source["sport"], source["source_table"], source["source_record_id"],
              source["source_hash"], source["raw_source"], payload, _hash(payload))
    if old is not None:
        if tuple(old) != values:
            raise EvidenceConflict("legacy source immutable identity conflict")
        return False
    try:
        db.execute("""INSERT INTO prospective_reconciled_source
            (source_key,sport,source_table,source_record_id,source_hash,raw_source,payload,payload_hash)
            VALUES (?,?,?,?,?,?,?,?)""", (source["source_key"], *values))
    except sqlite3.IntegrityError:
        raise EvidenceConflict("legacy source immutable identity conflict") from None
    return True


def _identity(row):
    return _key(row["sport"], row["market_family"], row.get("provider_namespace"),
                str(row.get("game_id")), str(row.get("provider_event_id")),
                row.get("selection"), row.get("line"), row.get("sportsbook"),
                row.get("quote_timestamp"), row.get("legacy_referenced_model_id", row.get("model_id")),
                row.get("legacy_model_name"), row.get("prediction_timestamp"))


def _capture_signature(row):
    # A newly appended score, closing proxy, or restored model can enrich a
    # projection.  The original capture facts cannot silently change.
    dynamic = _MUTABLE_PROJECTION_FIELDS | _LINEAGE_FIELDS
    if row["sport"] == "MLB" and row.get("source_store") != "mlb_prospective":
        # Schedule/quote observations are separate immutable source artifacts.
        # An incomplete restore may acquire these verified facts on a later run.
        dynamic |= {"home_team", "away_team", "quote_verified", "american_odds"}
    fixed = {key: value for key, value in row.items() if key not in dynamic}
    return _hash(_json(fixed))


def _enrich_mlb_receipt_quote(row, source_index):
    if row.get("quote_verified") is not True:
        return
    receipt = source_index.get(("receipts", row.get("source_record_id")))
    if receipt is None:
        raise EvidenceConflict("MLB receipt source missing")
    payload = receipt["record"].get("payload", {})
    q = payload.get("quote", {})
    quote_ref = payload.get("source_observations", {}).get("quotes")
    observation = source_index.get(("observations", quote_ref))
    if observation is None:
        raise EvidenceConflict("verified MLB quote source missing")
    from app_core.prediction_evidence import provider_quotes
    from app_core.public_quote_policy import canonical_book_label
    try:
        offered = json.loads(provider_quotes(observation["record"]["payload"]))
        matches = [candidate for candidate in offered if
            candidate.get("provider_event_id") == q.get("provider_event_id") and
            candidate.get("market_type") == q.get("market_type") and
            candidate.get("point") == q.get("line") and
            candidate.get("recorded_at") == q.get("provider_updated_at") and
            canonical_book_label(candidate.get("book")) == q.get("sportsbook") and
            abs((1 + (candidate["price"] / 100 if candidate["price"] > 0
                else 100 / abs(candidate["price"]))) - q["decimal_odds"]) < 1e-9]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        matches = []
    if len(matches) != 1:
        raise EvidenceConflict("verified MLB quote cannot replay exact price")
    row["american_odds"] = matches[0]["price"]


def _source_lineage(sport, row, source_index):
    source_id = row["source_record_id"]
    table = ("mlb_prospective_records" if row.get("source_store") == "mlb_prospective"
             else "receipts" if sport == "MLB" else "records")
    source = source_index.get((table, source_id))
    if source is None:
        raise EvidenceConflict("projected fact is missing its immutable source")
    result = {"capture_source_key": source["source_key"],
              "capture_source_hash": source["source_hash"]}
    # The complete source artifact inventory is imported separately.  These
    # links bind enriched scores and model artifacts to their exact record.
    if sport == "NCAAF":
        model = source_index.get(("records", row.get("model_id")))
        if model is not None:
            result["model_source_key"] = model["source_key"]
        for candidate in source_index.values():
            if candidate["source_table"] != "records" or candidate["record"].get("kind") != "scores":
                continue
            if any(str(score.get("cfbd_id")) == str(row.get("game_id"))
                   for score in candidate["record"].get("data", {}).get("scores", [])):
                result.setdefault("result_source_key", candidate["source_key"])
                break
    elif sport == "NFL":
        for candidate in source_index.values():
            if candidate["source_table"] == "records" and candidate["record"].get("kind") == "scores" \
                    and str(candidate["record"].get("data", {}).get("event_id")) == str(row.get("game_id")):
                result["result_source_key"] = candidate["source_key"]
    elif row.get("source_store") == "mlb_prospective":
        model = source_index.get(("mlb_prospective_records", row.get("legacy_referenced_model_id")))
        if model is not None:
            result["model_source_key"] = model["source_key"]
        for candidate in source_index.values():
            if (candidate["source_table"] == "mlb_prospective_records" and
                    candidate["record"].get("kind") == "scores" and
                    str(candidate["record"].get("data", {}).get("game_id")) == str(row.get("game_id"))):
                result["result_source_key"] = candidate["source_key"]
    else:
        from app_core.mlb_spread_total_model import digest
        event_key = digest({"provider_namespace": "mlb", "provider_event_id": row.get("game_id")})
        outcome = source_index.get(("outcomes", event_key))
        if outcome is not None:
            result["result_source_key"] = outcome["source_key"]
    return result


def _insert_fact(db, sport, row, source_index):
    if row.get("sport") != sport or row.get("market_family") not in SPORT_MARKETS[sport]:
        raise EvidenceConflict("legacy projected fact scope mismatch")
    lineage = _source_lineage(sport, row, source_index)
    canonical_identity = _identity(row)
    capture_signature = _capture_signature(row)
    existing = db.execute("""SELECT DISTINCT capture_signature FROM prospective_reconciled_fact
        WHERE sport=? AND market_family=? AND canonical_identity=?""",
        (sport, row["market_family"], canonical_identity)).fetchall()
    if any(item[0] != capture_signature for item in existing):
        raise EvidenceConflict("changed capture facts under canonical identity")
    content = {"schema": 1, "source_type": "LEGACY_RESEARCH", "research_only": True,
               "production_eligible": False, "recommended_stake": 0.0,
               "canonical_identity": canonical_identity, "lineage": lineage,
               "row": row}
    payload = _json(content)
    fact_id = _hash(payload)
    old = db.execute("""SELECT sport,market_family,canonical_identity,capture_signature,
        source_key,payload,payload_hash FROM prospective_reconciled_fact WHERE fact_id=?""",
        (fact_id,)).fetchone()
    values = (sport, row["market_family"], canonical_identity, capture_signature,
              lineage["capture_source_key"], payload, _hash(payload))
    if old is not None:
        if tuple(old) != values:
            raise EvidenceConflict("reconciled fact immutable hash conflict")
        return False
    db.execute("""INSERT INTO prospective_reconciled_fact
        (fact_id,sport,market_family,canonical_identity,capture_signature,source_key,payload,payload_hash)
        VALUES (?,?,?,?,?,?,?,?)""", (fact_id, *values))
    return True


def _verified_fact(row):
    blockers = set(row.get("blockers", ()))
    start = _time(row.get("scheduled_start"))
    quote_at = _time(row.get("quote_timestamp"))
    prediction_at = _time(row.get("prediction_timestamp"))
    result_at = _time(row.get("result_available_at"))
    return {
        "identity_failure": not all(row.get(key) for key in
            ("provider_namespace", "provider_event_id", "home_team_id", "away_team_id")),
        "chronology_failure": bool((quote_at is not None and start is not None and quote_at >= start)
             or (prediction_at is not None and start is not None and prediction_at >= start)
             or (result_at is not None and start is not None and result_at < start)),
        "model_failure": row.get("model_id") is None or row.get("model_version") is None,
        "price_failure": (row.get("quote_verified") is not True or
                          row.get("decimal_odds") is None or row.get("american_odds") is None),
        "result_failure": (row.get("result_outcome") == "NEEDS_REVIEW" or
                           "MISSING_AVAILABLE_RESULT" in blockers),
        "missing_close": row.get("verified_clv") is None,
    }


def _settled_row(row):
    start = _time(row.get("scheduled_start"))
    available = _time(row.get("result_available_at"))
    return (row.get("result_outcome") in {"WIN", "LOSS", "PUSH", "VOID"} and
            start is not None and available is not None and available >= start)


def _mlb_units(sources):
    from app_core.mlb_spread_total_model import digest, prepare_rows
    from app_core.mlb_receipt_audit import training_inventory
    receipts = [source["record"] for source in sources if source["source_table"] == "receipts"]
    outcomes = {source["source_record_id"]: source["record"] for source in sources
                if source["source_table"] == "outcomes"}
    settled = []
    for receipt in receipts:
        p = receipt.get("payload", {})
        key = digest({"provider_namespace": "mlb", "provider_event_id": p.get("provider_event_id")})
        if key in outcomes:
            settled.append({"snapshot": receipt, "outcome": outcomes[key]})
    if not settled:
        return {"RUN_LINE": 0, "TOTAL": 0}, [], None
    try:
        rows = prepare_rows(settled)
        inventory = training_inventory(rows)
        return {"RUN_LINE": inventory["spread"]["independent_event_line_units"],
                "TOTAL": inventory["total"]["independent_event_line_units"]}, \
               sorted({r["slate"] for r in rows}), None
    except (KeyError, TypeError, ValueError):
        return {"RUN_LINE": 0, "TOTAL": 0}, [], "SETTLED_DATASET_INTEGRITY_FAILED"


def _summarize(sport, rows, sources, *, status, appended_sources=0, appended_facts=0):
    identities = {_identity(row) for row in rows}
    markets = {}
    units, slates, unit_blocker = (_mlb_units(sources) if sport == "MLB" else ({}, [], None))
    for market in SPORT_MARKETS[sport]:
        scoped = [row for row in rows if row["market_family"] == market]
        games = {str(row["game_id"]) for row in scoped if row.get("game_id") is not None}
        settled = {str(row["game_id"]) for row in scoped
                   if row.get("game_id") is not None and _settled_row(row)}
        failures = Counter(name for row in scoped for name, failed in _verified_fact(row).items() if failed)
        markets[market] = {"source_market_rows": len(scoped),
                           "canonical_research_identities": len({_identity(row) for row in scoped}),
                           "canonical_accepted_research_rows": len({_identity(row) for row in scoped}),
                           "research_only_rows": len({_identity(row) for row in scoped}),
                           "captured_games": len(games), "settled_games": len(settled),
                           "pending_games": len(games - settled),
                           "historical_pregame_prediction_facts": sum(bool(row.get("mean_probability") is not None
                               and row.get("prediction_timestamp") is not None) for row in scoped),
                           "canonical_predictions": 0,
                           "historical_rows_promoted_to_canonical_predictions": 0,
                           "independent_settled_event_line_units": units.get(market),
                           "failure_counts": dict(sorted(failures.items())),
                           "research_only": True, "production_eligible": False,
                           "recommended_stake": 0.0}
    blockers = [unit_blocker] if unit_blocker else []
    return {"sport": sport, "source_status": status, "source_records": len(sources),
            "source_receipts": sum(s["source_table"] == "receipts" for s in sources),
            "legacy_forecast_source_records": sum(s["source_table"] == "mlb_prospective_records" for s in sources),
            "legacy_unpriced_forecast_rows": sum(row.get("source_store") == "mlb_prospective" for row in rows),
            "source_games": len({str(row["game_id"]) for row in rows if row.get("game_id") is not None}),
            "source_market_rows": len(rows), "canonical_research_identities": len(identities),
            "canonical_accepted_research_rows": len(identities),
            "research_only_rows": len(identities),
            "duplicates": len(rows) - len(identities), "new_source_artifacts": appended_sources,
            "new_reconciled_facts": appended_facts, "canonical_predictions": 0,
            "historical_rows_promoted_to_canonical_predictions": 0,
            "markets": markets, "mlb_settled_slates": slates,
            "blockers": blockers, "research_only": True, "production_eligible": False,
            "recommended_stake": 0.0}


def reconcile_sport(sport, canonical_path=None, source_path=None):
    """Import verified source artifacts and descriptive facts once, transactionally."""
    if sport not in LEGACY_SPORTS:
        raise ValueError("legacy reconciliation supports NFL, NCAAF, and MLB")
    source_path = _source_path(sport, source_path)
    if not source_path.is_file() and not (sport == "MLB" and
            source_path.with_name("mlb-prospective.sqlite3").is_file()):
        return _summarize(sport, [], [], status="LOCAL_ABSENT_REMOTE_UNKNOWN")
    sources = _read_sources(sport, source_path)
    rows = source_evidence(sport, source_path) if source_path.is_file() else []
    if sport == "MLB":
        rows.extend(_mlb_forecast_rows(sources))
    source_index = {(source["source_table"], source["source_record_id"]): source
                    for source in sources}
    if sport == "MLB":
        for row in rows:
            if row.get("source_store") != "mlb_prospective":
                _enrich_mlb_receipt_quote(row, source_index)
    if sport == "NCAAF":
        for row in rows:
            capture = source_index.get(("records", row["source_record_id"]))
            if capture is None:
                raise EvidenceConflict("NCAAF capture source missing")
            row["legacy_referenced_model_id"] = capture["record"].get("data", {}).get("model_id")
    canonical_path = _canonical_path(canonical_path)
    ensure_reconciliation_schema(canonical_path)
    with closing(canonical_connect(canonical_path)) as db, db:
        source_count = sum(_stored_source(db, source) for source in sources)
        fact_count = sum(_insert_fact(db, sport, row, source_index) for row in rows)
    status = "LOCAL_PRESENT" if source_path.is_file() else "MLB_RECEIPTS_ABSENT_PROSPECTIVE_PRESENT"
    return _summarize(sport, rows, sources, status=status,
                      appended_sources=source_count, appended_facts=fact_count)


def reconcile_all(canonical_path=None, source_directory=None):
    """Reconcile all three legacy sports without inferring remote emptiness."""
    directory = Path(source_directory) if source_directory is not None else database_path().parent
    return {sport: reconcile_sport(sport, canonical_path, directory / SOURCE_FILENAMES[sport])
            for sport in ("NFL", "NCAAF", "MLB")}


def reconciliation_readiness(canonical_path=None):
    """Read-only integrity-checked ledger counts, one row per legacy market."""
    path = _canonical_path(canonical_path)
    empty = [{"sport": sport, "market_family": market,
              "reconciled_research_rows": 0, "reconciled_games": 0,
              "reconciled_settled_games": 0, "historical_pregame_prediction_facts": 0,
              "canonical_predictions": 0, "research_only": True,
              "production_eligible": False, "recommended_stake": 0.0}
             for sport in ("NFL", "NCAAF", "MLB") for market in SPORT_MARKETS[sport]]
    if not path.is_file():
        return empty
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        if db.execute("""SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='prospective_reconciled_fact'""").fetchone() is None:
            return empty
        raw = db.execute("""SELECT f.fact_id,f.sport,f.market_family,f.canonical_identity,
            f.capture_signature,f.source_key,f.payload,f.payload_hash,s.sport,s.source_table,
            s.source_record_id,s.source_hash,s.raw_source,s.payload,s.payload_hash
            FROM prospective_reconciled_fact f JOIN prospective_reconciled_source s
            ON s.source_key=f.source_key ORDER BY f.rowid""").fetchall()
        source_keys = {item[0] for item in db.execute(
            "SELECT source_key FROM prospective_reconciled_source")}
    grouped = {(row["sport"], row["market_family"]): {} for row in empty}
    signatures = {}
    for fact_id, sport, market, identity, signature, source_key, payload, payload_hash, \
            source_sport, source_table, source_record_id, source_hash, raw_source, \
            source_payload, source_payload_hash in raw:
        if (_hash(payload) != fact_id or _hash(payload) != payload_hash or
                _hash(raw_source) != source_hash or source_sport != sport or
                _key(source_sport, source_table, source_record_id) != source_key or
                _hash(source_payload) != source_payload_hash):
            raise EvidenceConflict("reconciled ledger hash mismatch")
        if json.loads(source_payload) != {
                "source_key": source_key, "sport": source_sport,
                "source_table": source_table, "source_record_id": source_record_id,
                "source_hash": source_hash}:
            raise EvidenceConflict("reconciled source metadata mismatch")
        value = json.loads(payload)
        if (value.get("canonical_identity") != identity or
                _identity(value.get("row", {})) != identity or
                value.get("row", {}).get("sport") != sport or
                value["row"].get("market_family") != market or
                value.get("lineage", {}).get("capture_source_key") != source_key or
                value.get("lineage", {}).get("capture_source_hash") != source_hash or
                _capture_signature(value["row"]) != signature):
            raise EvidenceConflict("reconciled ledger lineage mismatch")
        if any(key not in source_keys for name, key in value["lineage"].items()
               if name.endswith("_source_key")):
            raise EvidenceConflict("reconciled dependent source missing")
        scope = (sport, market, identity)
        if scope in signatures and signatures[scope] != signature:
            raise EvidenceConflict("reconciled canonical identity conflict")
        signatures[scope] = signature
        current = grouped[(sport, market)].get(identity)
        stamp = _time(value["row"].get("result_available_at"))
        current_stamp = _time(current.get("result_available_at")) if current else None
        quality = (value["row"].get("quote_verified") is True,
                   value["row"].get("american_odds") is not None)
        current_quality = ((current.get("quote_verified") is True,
                            current.get("american_odds") is not None) if current else (False, False))
        if (current is None or
                (stamp is not None and (current_stamp is None or stamp > current_stamp)) or
                (stamp == current_stamp and quality > current_quality)):
            grouped[(sport, market)][identity] = value["row"]
    for report in empty:
        rows = list(grouped[(report["sport"], report["market_family"])].values())
        report.update(reconciled_research_rows=len(rows),
                      reconciled_games=len({str(row["game_id"]) for row in rows
                          if row.get("game_id") is not None}),
                      reconciled_settled_games=len({str(row.get("game_id")) for row in rows
                          if row.get("game_id") is not None and _settled_row(row)}),
                      historical_pregame_prediction_facts=sum(row.get("mean_probability") is not None
                          and row.get("prediction_timestamp") is not None for row in rows))
    return empty
