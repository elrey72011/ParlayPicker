"""Append-only, sport and market scoped prospective straight-wager evidence.

This database is research evidence, not wager authority. Historical sport stores
remain untouched. Missing source/model/price facts may be represented in research
rows, but can never become validation or deployment evidence by default.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Mapping

from app_core.prediction_evidence import database_path


SPORT_MARKETS = {
    "NFL": ("SPREAD", "TOTAL"),
    "NCAAF": ("SPREAD", "TOTAL"),
    "NBA": ("SPREAD", "TOTAL"),
    "NCAAB": ("SPREAD", "TOTAL"),
    "MLB": ("RUN_LINE", "TOTAL"),
    "NHL": ("PUCK_LINE", "TOTAL"),
}
OUTCOMES = frozenset({"WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW"})
STATES = frozenset({"UNVALIDATED", "PROVISIONAL_VALIDATED", "STANDARD_VALIDATED", "PREMIUM_VALIDATED"})
MAX_QUOTE_AGE = timedelta(minutes=15)
MAX_PREDICTION_CLOCK_SKEW = timedelta(minutes=5)
SCHEMA_VERSION = 1
ODDS_API_SPORT_KEYS = {
    "NFL": "americanfootball_nfl", "NCAAF": "americanfootball_ncaaf",
    "NBA": "basketball_nba", "NCAAB": "basketball_ncaab",
    "MLB": "baseball_mlb", "NHL": "icehockey_nhl",
}
ODDS_API_SIDE_MARKETS = {
    "NFL": "SPREAD", "NCAAF": "SPREAD", "NBA": "SPREAD", "NCAAB": "SPREAD",
    "MLB": "RUN_LINE", "NHL": "PUCK_LINE",
}
TABLE_KEYS = {
    "prospective_event": "event_id", "prospective_quote": "quote_id",
    "prospective_close": "close_id", "prospective_result": "result_id",
    "prospective_model": "model_id", "prospective_calibration": "calibration_id",
    "prospective_prediction": "observation_id", "prospective_validation_plan": "validation_plan_id",
    "prospective_validation_artifact": "artifact_id",
    "prospective_deployment_review": "deployment_id",
}


class EvidenceConflict(ValueError):
    """An immutable ID or source identity already has different evidence."""


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _time(value: object, name: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError:
        raise ValueError(f"{name} requires a timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _number(value: object, name: str, *, optional: bool = False,
            low: float | None = None, high: float | None = None) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result) or (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f"{name} must be finite and within bounds")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _scope(sport: object, market_family: object | None = None) -> tuple[str, str | None]:
    if sport not in SPORT_MARKETS:
        raise ValueError("unsupported sport")
    if market_family is not None and market_family not in SPORT_MARKETS[sport]:
        raise ValueError("unsupported sport/market family")
    return sport, market_family


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("evidence must be finite JSON") from None


def _sha(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source(value: object) -> tuple[bytes, str]:
    if isinstance(value, (Mapping, list)):
        raw = _json(value).encode("utf-8")
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise ValueError("raw_source must be a provider response or retained artifact")
    if not raw or len(raw) > 40_000_000:
        raise ValueError("raw_source is empty or too large")
    return raw, _sha(raw)


def _map(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("evidence must be a mapping")
    return dict(value)


def _path(path: str | Path | None) -> Path:
    return Path(path or database_path().with_name("prospective-evidence.sqlite3"))


def _reader(path: str | Path | None) -> sqlite3.Connection:
    """Open an existing store without creating files, tables, or triggers."""
    target = _path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    db = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _has_schema(path: str | Path | None) -> bool:
    if not _path(path).is_file():
        return False
    with closing(_reader(path)) as db:
        return db.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='prospective_event'""").fetchone() is not None


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open/additive-migrate the canonical research database with FK enforcement."""
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS prospective_event (
            event_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            game_id TEXT NOT NULL,
            provider_namespace TEXT NOT NULL,
            provider_event_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team_id TEXT,
            away_team_id TEXT,
            scheduled_start TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            UNIQUE(sport, provider_namespace, provider_event_id)
        );
        CREATE TABLE IF NOT EXISTS prospective_quote (
            quote_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES prospective_event(event_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL,
            american_odds INTEGER,
            decimal_odds REAL,
            sportsbook TEXT,
            quote_timestamp TEXT NOT NULL,
            quote_source TEXT NOT NULL,
            quote_verified INTEGER NOT NULL CHECK(quote_verified IN (0,1)),
            source_id TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_close (
            close_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES prospective_event(event_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL,
            american_odds INTEGER,
            decimal_odds REAL,
            sportsbook TEXT,
            close_timestamp TEXT NOT NULL,
            close_source TEXT NOT NULL,
            close_verified INTEGER NOT NULL CHECK(close_verified IN (0,1)),
            source_id TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_result (
            result_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES prospective_event(event_id),
            sport TEXT NOT NULL,
            market_family TEXT,
            selection TEXT,
            result_source TEXT NOT NULL,
            result_source_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            outcome TEXT,
            grading_version INTEGER NOT NULL CHECK(grading_version > 0),
            revises_result_id TEXT REFERENCES prospective_result(result_id),
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_model (
            model_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            model_version TEXT NOT NULL,
            training_start TEXT NOT NULL,
            training_cutoff TEXT NOT NULL,
            training_observation_count INTEGER NOT NULL CHECK(training_observation_count >= 0),
            independent_event_count INTEGER NOT NULL CHECK(independent_event_count >= 0),
            feature_version TEXT NOT NULL,
            training_code_commit TEXT NOT NULL,
            created_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_model_training_result (
            model_id TEXT NOT NULL REFERENCES prospective_model(model_id),
            result_id TEXT NOT NULL REFERENCES prospective_result(result_id),
            PRIMARY KEY(model_id,result_id)
        );
        CREATE TABLE IF NOT EXISTS prospective_calibration (
            calibration_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            calibration_version TEXT NOT NULL,
            model_id TEXT NOT NULL REFERENCES prospective_model(model_id),
            fit_start TEXT NOT NULL,
            fit_end TEXT NOT NULL,
            method TEXT NOT NULL,
            created_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_calibration_result (
            calibration_id TEXT NOT NULL REFERENCES prospective_calibration(calibration_id),
            result_id TEXT NOT NULL REFERENCES prospective_result(result_id),
            PRIMARY KEY(calibration_id,result_id)
        );
        CREATE TABLE IF NOT EXISTS prospective_prediction (
            observation_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES prospective_event(event_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            selection TEXT NOT NULL,
            quote_id TEXT REFERENCES prospective_quote(quote_id),
            model_id TEXT REFERENCES prospective_model(model_id),
            model_version TEXT,
            model_trained_through TEXT,
            model_available_at TEXT,
            feature_version TEXT,
            feature_snapshot_id TEXT,
            feature_frozen_at TEXT,
            calibration_id TEXT REFERENCES prospective_calibration(calibration_id),
            calibration_version TEXT,
            calibration_available_at TEXT,
            policy_version TEXT,
            prediction_timestamp TEXT NOT NULL,
            mean_probability REAL,
            conservative_probability REAL,
            push_probability REAL,
            loss_probability REAL,
            probability_semantics TEXT,
            evidence_snapshot_id TEXT,
            evidence_hash TEXT,
            runtime_hash TEXT,
            source_commit TEXT,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_validation_plan (
            validation_plan_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            version INTEGER NOT NULL CHECK(version > 0),
            supersedes_plan_id TEXT REFERENCES prospective_validation_plan(validation_plan_id),
            model_id TEXT REFERENCES prospective_model(model_id),
            calibration_id TEXT REFERENCES prospective_calibration(calibration_id),
            training_cutoff TEXT NOT NULL,
            validation_start TEXT NOT NULL,
            validation_end TEXT NOT NULL,
            holdout_start TEXT NOT NULL,
            holdout_end TEXT NOT NULL,
            minimum_independent_sample INTEGER NOT NULL,
            minimum_effective_sample REAL NOT NULL,
            frozen_at TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            UNIQUE(sport,market_family,version)
        );
        CREATE TABLE IF NOT EXISTS prospective_validation_artifact (
            artifact_id TEXT PRIMARY KEY,
            validation_plan_id TEXT NOT NULL REFERENCES prospective_validation_plan(validation_plan_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('UNVALIDATED','VALIDATION_PASSED')),
            report_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_deployment_review (
            deployment_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES prospective_validation_artifact(artifact_id),
            validation_plan_id TEXT NOT NULL REFERENCES prospective_validation_plan(validation_plan_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            validation_id TEXT NOT NULL,
            deployment_state TEXT NOT NULL CHECK(deployment_state IN
                ('PROVISIONAL_VALIDATED','STANDARD_VALIDATED','PREMIUM_VALIDATED')),
            reviewer_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS prospective_predictions_scope
            ON prospective_prediction(sport,market_family,prediction_timestamp);
        CREATE INDEX IF NOT EXISTS prospective_results_event
            ON prospective_result(event_id,market_family,selection,available_at);
        CREATE INDEX IF NOT EXISTS prospective_quotes_scope
            ON prospective_quote(sport,market_family,event_id);
        CREATE TABLE IF NOT EXISTS prospective_football_event (
            version_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            sport TEXT NOT NULL CHECK(sport IN ('NFL','NCAAF')),
            season INTEGER NOT NULL,
            week INTEGER,
            season_type TEXT NOT NULL,
            provider_namespace TEXT NOT NULL,
            provider_event_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team_id TEXT,
            away_team_id TEXT,
            scheduled_start TEXT NOT NULL,
            neutral_site INTEGER,
            venue TEXT,
            discovered_at TEXT NOT NULL,
            identity_mapping_version TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS prospective_football_event_game
            ON prospective_football_event(sport,game_id);
        CREATE TABLE IF NOT EXISTS prospective_football_quote (
            quote_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            event_version_id TEXT NOT NULL REFERENCES prospective_football_event(version_id),
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL NOT NULL,
            american_odds INTEGER NOT NULL,
            decimal_odds REAL NOT NULL,
            sportsbook TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_event_id TEXT NOT NULL,
            odds_event_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            provider_last_update TEXT NOT NULL,
            capture_run_id TEXT NOT NULL,
            identity_mapping_hash TEXT NOT NULL,
            capture_horizon TEXT NOT NULL,
            minutes_to_start REAL NOT NULL,
            quote_verified INTEGER NOT NULL CHECK(quote_verified IN (0,1)),
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS prospective_football_quote_game
            ON prospective_football_quote(sport,game_id,market_family);
        CREATE TABLE IF NOT EXISTS prospective_football_result (
            result_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            event_version_id TEXT NOT NULL REFERENCES prospective_football_event(version_id),
            sport TEXT NOT NULL,
            result_source TEXT NOT NULL,
            result_source_event_id TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            observed_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            result_status TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_settlement (
            settlement_id TEXT PRIMARY KEY,
            quote_id TEXT NOT NULL REFERENCES prospective_football_quote(quote_id),
            result_id TEXT NOT NULL REFERENCES prospective_football_result(result_id),
            game_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            selection TEXT NOT NULL,
            line REAL NOT NULL,
            outcome TEXT NOT NULL,
            settled_at TEXT NOT NULL,
            settlement_version INTEGER NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_training_row (
            training_row_id TEXT PRIMARY KEY,
            quote_id TEXT NOT NULL REFERENCES prospective_football_quote(quote_id),
            result_id TEXT NOT NULL REFERENCES prospective_football_result(result_id),
            settlement_id TEXT NOT NULL REFERENCES prospective_football_settlement(settlement_id),
            game_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            market_family TEXT NOT NULL,
            label TEXT NOT NULL,
            training_row_status TEXT NOT NULL CHECK(training_row_status IN ('TRAINING_READY','TRAINING_BLOCKED')),
            blockers TEXT NOT NULL,
            available_for_training_at TEXT NOT NULL,
            source_manifest_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_team_identity (
            identity_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            canonical_team_id TEXT NOT NULL,
            provider_namespace TEXT NOT NULL,
            provider_team_id TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            aliases TEXT NOT NULL,
            mapping_version TEXT NOT NULL,
            verified_at TEXT NOT NULL,
            mapping_source TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_theover (
            research_row_id TEXT PRIMARY KEY,
            source_filename TEXT NOT NULL,
            source_file_hash TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            sport TEXT,
            matchup TEXT,
            selection TEXT,
            line REAL,
            win_probability REAL,
            model_hit_rate REAL,
            matched_game_id TEXT,
            match_status TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_coverage (
            coverage_id TEXT PRIMARY KEY,
            capture_run_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            game_id TEXT,
            provider_event_id TEXT,
            regular_season_target INTEGER NOT NULL,
            status TEXT NOT NULL,
            spread_status TEXT NOT NULL,
            total_status TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prospective_football_cycle_coverage (
            coverage_id TEXT PRIMARY KEY,
            capture_run_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            target_games INTEGER NOT NULL,
            requested_slate_success INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            raw_source BLOB NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE VIEW IF NOT EXISTS prospective_football_active_training_row AS
        SELECT t.* FROM prospective_football_training_row t
        JOIN prospective_football_result r ON r.result_id=t.result_id
        WHERE t.training_row_status='TRAINING_READY'
          AND NOT EXISTS (
              SELECT 1 FROM prospective_football_result revision
              WHERE revision.game_id=t.game_id
                AND (revision.home_score<>r.home_score OR revision.away_score<>r.away_score)
          );
    """)
    tables = ("prospective_event", "prospective_quote", "prospective_close", "prospective_result",
              "prospective_model", "prospective_model_training_result", "prospective_calibration",
              "prospective_calibration_result", "prospective_prediction", "prospective_validation_plan",
              "prospective_validation_artifact", "prospective_deployment_review",
              "prospective_football_event", "prospective_football_quote",
              "prospective_football_result", "prospective_football_settlement",
              "prospective_football_training_row", "prospective_football_team_identity",
              "prospective_football_theover", "prospective_football_coverage",
              "prospective_football_cycle_coverage")
    for table in tables:
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action}_immutable "
                       f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only evidence'); END")
    return db


def _stored(db: sqlite3.Connection, table: str, key_column: str, key: str) -> dict | None:
    row = db.execute(f"SELECT * FROM {table} WHERE {key_column}=?", (key,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    if _sha(result["payload"]) != result["payload_hash"]:
        raise EvidenceConflict("stored evidence hash mismatch")
    if "raw_source" in result and _sha(result["raw_source"]) != result["source_hash"]:
        raise EvidenceConflict("stored source hash mismatch")
    return result


def _insert(db: sqlite3.Connection, table: str, key_column: str, values: dict,
            payload: dict, *, raw_source: bytes | None = None) -> str:
    raw = _json(payload)
    key = values[key_column]
    values = dict(values, payload=raw, payload_hash=_sha(raw))
    if raw_source is not None:
        values["raw_source"] = raw_source
    old = _stored(db, table, key_column, key)
    if old is not None:
        if any(old[name] != value for name, value in values.items()):
            raise EvidenceConflict(f"{table} immutable ID conflict")
        return key
    columns = ",".join(values)
    placeholders = ",".join("?" for _ in values)
    try:
        db.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))
    except sqlite3.IntegrityError as exc:
        raise EvidenceConflict(f"{table} source/foreign-key conflict") from exc
    return key


def _source_fields(row: dict) -> tuple[bytes, str, str]:
    source_id = _text(row.get("source_id"), "source_id")
    raw, digest = _source(row.get("raw_source"))
    claimed = row.get("source_hash")
    if claimed is not None and claimed != digest:
        raise EvidenceConflict("source_hash does not match raw_source")
    return raw, digest, source_id


def _event(db: sqlite3.Connection, event_id: object) -> dict:
    key = _text(event_id, "event_id")
    event = _stored(db, "prospective_event", "event_id", key)
    if event is None:
        raise ValueError("unknown prospective event")
    return event


def _same_scope(record: dict, sport: str, market_family: str, *, name: str) -> None:
    if (record["sport"], record["market_family"]) != (sport, market_family):
        raise ValueError(f"{name} sport/market scope mismatch")


def _market_record(db: sqlite3.Connection, table: str, column: str, key: object,
                   sport: str, market_family: str, name: str) -> dict:
    record = _stored(db, table, column, _text(key, name))
    if record is None:
        raise ValueError(f"unknown {name}")
    _same_scope(record, sport, market_family, name=name)
    return record


def _snapshot(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "raw_source"}


def insert_event(path: str | Path | None, event: Mapping) -> str:
    data = _map(event)
    sport, _ = _scope(data.get("sport"))
    observed = _time(data.get("observed_at"), "observed_at")
    start = _time(data.get("scheduled_start"), "scheduled_start")
    now = _clock()
    if not observed <= now < start or observed >= start:
        raise ValueError("event must be discovered before start at ingestion")
    raw, source_hash, source_id = _source_fields(data)
    values = dict(event_id=_text(data.get("event_id"), "event_id"), sport=sport,
                  game_id=_text(data.get("game_id"), "game_id"),
                  provider_namespace=_text(data.get("provider_namespace"), "provider_namespace"),
                  provider_event_id=_text(data.get("provider_event_id"), "provider_event_id"),
                  home_team=_text(data.get("home_team"), "home_team"),
                  away_team=_text(data.get("away_team"), "away_team"),
                  home_team_id=_text(data.get("home_team_id"), "home_team_id", optional=True),
                  away_team_id=_text(data.get("away_team_id"), "away_team_id", optional=True),
                  scheduled_start=_iso(start), observed_at=_iso(observed), ingested_at=_iso(now),
                  source_id=source_id, source_hash=source_hash)
    payload = _snapshot(values)
    with closing(connect(path)) as db, db:
        return _insert(db, "prospective_event", "event_id", values, payload, raw_source=raw)


def _odds(data: dict, *, verified: bool) -> tuple[float | None, int | None, float | None, str | None]:
    line = _number(data.get("line"), "line", optional=True)
    american = data.get("american_odds")
    if american is not None:
        if type(american) is not int or american in (-100, 0) or -100 < american < 100:
            raise ValueError("american_odds must be a valid integer price")
    decimal = _number(data.get("decimal_odds"), "decimal_odds", optional=True, low=1.000001)
    sportsbook = _text(data.get("sportsbook"), "sportsbook", optional=True)
    if american is not None and decimal is not None:
        expected = 1 + (american / 100 if american > 0 else 100 / abs(american))
        if abs(expected - decimal) > 0.02:
            raise ValueError("American and decimal odds disagree")
    if verified and (line is None or american is None or decimal is None or sportsbook is None):
        raise ValueError("verified quote/close requires exact line, price, and sportsbook")
    return line, american, decimal, sportsbook


def _source_json(raw: object) -> dict | None:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (TypeError, ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def verify_provider_offer(event: Mapping, offer: Mapping) -> bool:
    """Replay a verified The Odds API offer against its retained raw response.

    Pregame quotes use the capture event response plus the exact market fragment
    for all six sports. A verified MLB Odds API offer does not establish the
    cross-provider mapping to an MLB Stats API model/game; that requires separate
    identity evidence before prediction or wager eligibility.
    A live near-start odds response is only a close *candidate*: The Odds API
    does not certify that it was the final closing price. No current provider
    adapter can set ``close_verified=True``; such rows need a future dedicated
    closing-source replay verifier. Source consistency alone is not provider
    credential proof or wager authority.
    """
    try:
        sport = event["sport"]
        if sport not in ODDS_API_SPORT_KEYS or event["provider_namespace"] != "THE_ODDS_API":
            return False
        is_close = "close_timestamp" in offer
        if is_close:
            return False
        if offer["close_source" if is_close else "quote_source"] != "THE_ODDS_API":
            return False
        if (event["event_id"] != offer["event_id"] or sport != offer["sport"] or
                not event.get("home_team_id") or not event.get("away_team_id") or
                event["home_team_id"] == event["away_team_id"]):
            return False
        side = ODDS_API_SIDE_MARKETS[sport]
        market_key = "spreads" if offer["market_family"] == side else "totals"
        if offer["market_family"] not in (side, "TOTAL"):
            return False
        raw_offer = _source_json(offer["raw_source"])
        if raw_offer is None:
            return False
        full_event = _source_json(event["raw_source"])
        market = raw_offer
        received = _time(event["observed_at"], "observed_at")
        if not isinstance(full_event, dict) or not isinstance(market, dict):
            return False
        if (full_event.get("id") != event["provider_event_id"] or
                full_event.get("sport_key") != ODDS_API_SPORT_KEYS[sport] or
                full_event.get("home_team") != event["home_team"] or
                full_event.get("away_team") != event["away_team"] or
                _time(full_event.get("commence_time"), "commence_time") !=
                _time(event["scheduled_start"], "scheduled_start")):
            return False
        stamp = _time(offer["close_timestamp" if is_close else "quote_timestamp"], "offer_timestamp")
        if (market.get("key") != market_key or
                _time(market.get("last_update"), "market.last_update") != stamp or
                not stamp <= received < _time(event["scheduled_start"], "scheduled_start") or
                received - stamp > MAX_QUOTE_AGE):
            return False
        books = full_event.get("bookmakers")
        if not isinstance(books, list):
            return False
        matches = [book for book in books if isinstance(book, dict) and
                   book.get("key") == offer["sportsbook"]]
        if len(matches) != 1 or not isinstance(matches[0].get("markets"), list):
            return False
        matched_markets = [item for item in matches[0]["markets"] if
                           isinstance(item, dict) and item.get("key") == market_key]
        if len(matched_markets) != 1 or _json(matched_markets[0]) != _json(market):
            return False
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 2 or any(
                not isinstance(item, dict) for item in outcomes):
            return False
        expected_names = {event["home_team"], event["away_team"]} if market_key == "spreads" else {"Over", "Under"}
        names = [item.get("name") for item in outcomes]
        if len(set(names)) != 2 or set(names) != expected_names:
            return False
        lines = [_number(item.get("point"), "provider.point") for item in outcomes]
        prices = [item.get("price") for item in outcomes]
        if any(type(price) is not int or abs(price) < 100 for price in prices):
            return False
        if ((market_key == "spreads" and abs(sum(lines)) > 1e-9) or
                (market_key == "totals" and (lines[0] != lines[1] or lines[0] <= 0))):
            return False
        indexes = [i for i, name in enumerate(names) if name == offer["selection"]]
        if len(indexes) != 1:
            return False
        index = indexes[0]
        price = prices[index]
        decimal = 1 + (price / 100 if price > 0 else 100 / abs(price))
        if (offer["line"] != lines[index] or offer["american_odds"] != price or
                abs(offer["decimal_odds"] - decimal) > 1e-6):
            return False
        source_id = (f"{event['provider_event_id']}:{offer['sportsbook']}:{market_key}:"
                     f"{offer['selection']}:{_iso(stamp)}")
        return offer["source_id"] == source_id
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _price_record(path: str | Path | None, data: Mapping, *, closing_price: bool) -> str:
    row = _map(data)
    sport, market = _scope(row.get("sport"), row.get("market_family"))
    table = "prospective_close" if closing_price else "prospective_quote"
    key_column = "close_id" if closing_price else "quote_id"
    stamp_column = "close_timestamp" if closing_price else "quote_timestamp"
    origin_column = "close_source" if closing_price else "quote_source"
    verified_column = "close_verified" if closing_price else "quote_verified"
    stamp = _time(row.get(stamp_column), stamp_column)
    now = _clock()
    if stamp > now or now - stamp > MAX_QUOTE_AGE:
        raise ValueError("future or stale price observation")
    verified = row.get(verified_column)
    if type(verified) is not bool:
        raise ValueError(f"{verified_column} must be boolean")
    line, american, decimal, sportsbook = _odds(row, verified=verified)
    raw, source_hash, source_id = _source_fields(row)
    with closing(connect(path)) as db, db:
        event = _event(db, row.get("event_id"))
        if sport != event["sport"] or stamp >= _time(event["scheduled_start"], "scheduled_start"):
            raise ValueError("price event scope or pregame chronology mismatch")
        values = {key_column: _text(row.get(key_column), key_column),
                  "event_id": event["event_id"], "sport": sport, "market_family": market,
                  "selection": _text(row.get("selection"), "selection"), "line": line,
                  "american_odds": american, "decimal_odds": decimal, "sportsbook": sportsbook,
                  stamp_column: _iso(stamp), origin_column: _text(row.get(origin_column), origin_column),
                  verified_column: int(verified), "source_id": source_id, "source_hash": source_hash}
        if verified and not verify_provider_offer(event, dict(values, raw_source=raw)):
            raise ValueError("verified price is not replayable from exact provider source")
        return _insert(db, table, key_column, values, _snapshot(values), raw_source=raw)


def insert_quote(path: str | Path | None, quote: Mapping) -> str:
    return _price_record(path, quote, closing_price=False)


def insert_close(path: str | Path | None, close: Mapping) -> str:
    return _price_record(path, close, closing_price=True)


def insert_result(path: str | Path | None, result: Mapping) -> str:
    data = _map(result)
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    observed = _time(data.get("observed_at"), "observed_at")
    available = _time(data.get("available_at"), "available_at")
    if not observed <= available <= _clock():
        raise ValueError("result availability chronology invalid")
    home = data.get("home_score")
    away = data.get("away_score")
    if home is not None:
        home = _integer(home, "home_score")
    if away is not None:
        away = _integer(away, "away_score")
    outcome = data.get("outcome")
    selection = _text(data.get("selection"), "selection", optional=True)
    if market is None:
        if selection is not None or outcome is not None:
            raise ValueError("event score cannot assert a market settlement")
    elif selection is None or outcome not in OUTCOMES:
        raise ValueError("market result needs selection and explicit outcome")
    if outcome in {"WIN", "LOSS", "PUSH"} and (home is None or away is None):
        raise ValueError("settled market result requires both authentic scores")
    raw, source_hash = _source(data.get("raw_source"))
    version = _integer(data.get("grading_version"), "grading_version", minimum=1)
    with closing(connect(path)) as db, db:
        event = _event(db, data.get("event_id"))
        if sport != event["sport"] or observed < _time(event["scheduled_start"], "scheduled_start"):
            raise ValueError("result event scope or start chronology mismatch")
        previous_id = _text(data.get("revises_result_id"), "revises_result_id", optional=True)
        existing = db.execute("""SELECT result_id,grading_version FROM prospective_result
            WHERE event_id=? AND market_family IS ? AND selection IS ?
            ORDER BY grading_version DESC,result_id DESC LIMIT 1""",
            (event["event_id"], market, selection)).fetchone()
        if previous_id is not None:
            previous = _stored(db, "prospective_result", "result_id", previous_id)
            if previous is None or any(previous[name] != value for name, value in
                                       (("event_id", event["event_id"]), ("market_family", market),
                                        ("selection", selection))):
                raise ValueError("result correction must revise the same event/market/selection")
            if version <= previous["grading_version"] or available <= _time(previous["available_at"], "available_at"):
                raise ValueError("result correction must advance version and availability")
            if existing is None or existing["result_id"] != previous_id:
                raise ValueError("result correction must revise the latest grading")
        elif version != 1:
            raise ValueError("initial grading_version must be 1")
        elif existing is not None:
            raise ValueError("existing result requires an appended correction")
        values = dict(result_id=_text(data.get("result_id"), "result_id"), event_id=event["event_id"],
                      sport=sport, market_family=market, selection=selection,
                      result_source=_text(data.get("result_source"), "result_source"),
                      result_source_id=_text(data.get("result_source_id"), "result_source_id"),
                      observed_at=_iso(observed), available_at=_iso(available), home_score=home,
                      away_score=away, outcome=outcome, grading_version=version,
                      revises_result_id=previous_id, source_hash=source_hash)
        return _insert(db, "prospective_result", "result_id", values,
                       _snapshot(values), raw_source=raw)


def verify_provider_score(event: Mapping, result: Mapping) -> bool:
    """Replay an exact Odds API final score without asserting market settlement."""
    try:
        sport = event["sport"]
        if sport not in {"NBA", "NCAAB", "NHL"} or result["result_source"] != "THE_ODDS_API":
            return False
        if (event["provider_namespace"] != "THE_ODDS_API" or
                result["result_source_id"] != event["provider_event_id"] or
                result["event_id"] != event["event_id"] or
                result["sport"] != sport or result["market_family"] is not None or
                result["selection"] is not None or result["outcome"] is not None):
            return False
        raw = _source_json(result["raw_source"])
        if not isinstance(raw, dict) or raw.get("completed") is not True:
            return False
        if (raw.get("id") != event["provider_event_id"] or
                raw.get("sport_key") != ODDS_API_SPORT_KEYS[sport] or
                raw.get("home_team") != event["home_team"] or
                raw.get("away_team") != event["away_team"]):
            return False
        updated = _time(raw.get("last_update"), "score.last_update")
        observed = _time(result["observed_at"], "observed_at")
        available = _time(result["available_at"], "available_at")
        if not _time(event["scheduled_start"], "scheduled_start") <= updated <= observed <= available:
            return False
        scores = raw.get("scores")
        if not isinstance(scores, list) or len(scores) != 2:
            return False
        expected = {event["home_team"]: result["home_score"],
                    event["away_team"]: result["away_score"]}
        found = {}
        for item in scores:
            if not isinstance(item, dict) or item.get("name") in found:
                return False
            score = item.get("score")
            if isinstance(score, bool) or not str(score).isdigit():
                return False
            found[item.get("name")] = int(score)
        return found == expected
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _training_results(db: sqlite3.Connection, result_ids: object, sport: str,
                      market: str, start: datetime, cutoff: datetime, *,
                      event_score_target: bool = False) -> tuple[list[str], int]:
    if not isinstance(result_ids, (list, tuple)) or len(set(result_ids)) != len(result_ids):
        raise ValueError("training/fit result IDs must be a unique list")
    events = set()
    for result_id in result_ids:
        if event_score_target:
            result = _stored(db, "prospective_result", "result_id", _text(result_id, "training result"))
            if result is None or result["sport"] != sport or not verify_provider_score(
                    _event(db, result["event_id"]), result):
                raise ValueError("training score is not replayable provider evidence")
        else:
            result = _market_record(db, "prospective_result", "result_id", result_id,
                                    sport, market, "training result")
            if result["outcome"] not in {"WIN", "LOSS", "PUSH", "VOID"}:
                raise ValueError("training result is not settled")
        available = _time(result["available_at"], "available_at")
        if not start <= available <= cutoff:
            raise ValueError("outcome was outside the available training/fit window cutoff")
        events.add(result["event_id"])
    return list(result_ids), len(events)


def _digest_text(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def insert_model(path: str | Path | None, model: Mapping) -> str:
    data = _map(model)
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    start = _time(data.get("training_start"), "training_start")
    cutoff = _time(data.get("training_cutoff"), "training_cutoff")
    created = _time(data.get("created_at"), "created_at")
    available = _time(data.get("available_at"), "available_at")
    if not start < cutoff <= created <= available <= _clock():
        raise ValueError("model training/availability chronology invalid")
    target_kind = data.get("training_target_kind", "MARKET_SETTLEMENT_V1")
    if target_kind not in {"MARKET_SETTLEMENT_V1", "FINAL_SCORE_DISTRIBUTION_RESEARCH_V1"}:
        raise ValueError("unsupported model training target")
    event_score_target = target_kind == "FINAL_SCORE_DISTRIBUTION_RESEARCH_V1"
    if event_score_target and (sport not in {"NBA", "NCAAB", "NHL"} or
                               data.get("model_artifact") is None):
        raise ValueError("score-distribution target requires a new-sport bound artifact")
    result_ids = data.get("training_result_ids", [])
    with closing(connect(path)) as db, db:
        result_ids, event_count = _training_results(db, result_ids, sport, market, start, cutoff,
                                                     event_score_target=event_score_target)
        count = _integer(data.get("training_observation_count"), "training_observation_count")
        independent = _integer(data.get("independent_event_count"), "independent_event_count")
        if count != len(result_ids) or independent != event_count:
            raise ValueError("model counts must reconcile to exact-scope available result IDs")
        values = dict(model_id=_text(data.get("model_id"), "model_id"), sport=sport,
                      market_family=market, model_version=_text(data.get("model_version"), "model_version"),
                      training_start=_iso(start), training_cutoff=_iso(cutoff),
                      training_observation_count=count, independent_event_count=independent,
                      feature_version=_text(data.get("feature_version"), "feature_version"),
                      training_code_commit=_text(data.get("training_code_commit"), "training_code_commit"),
                      created_at=_iso(created), available_at=_iso(available),
                      artifact_hash=_digest_text(data.get("artifact_hash"), "artifact_hash"))
        payload = dict(values, training_result_ids=result_ids)
        if event_score_target:
            payload["training_target_kind"] = target_kind
        artifact = data.get("model_artifact")
        if artifact is not None:
            artifact = _map(artifact)
            if _sha(_json(artifact)) != values["artifact_hash"]:
                raise EvidenceConflict("model artifact hash does not match content")
            if event_score_target:
                sources = artifact.get("training_sources")
                if (artifact.get("sport") != sport or artifact.get("market_family") != market or
                        artifact.get("feature_version") != values["feature_version"] or
                        artifact.get("training_target_kind") != target_kind or
                        artifact.get("source_commit") != values["training_code_commit"] or
                        artifact.get("training_independent_events") != independent or
                        artifact.get("production_eligible") is not False or
                        artifact.get("market_settlement_certified") is not False or
                        _time(artifact.get("validation_cutoff"), "validation_cutoff") >=
                        _time(artifact.get("validation_first_observed_at"),
                              "validation_first_observed_at") or
                        not isinstance(sources, list) or
                        [item.get("result_id") for item in sources if isinstance(item, Mapping)] != result_ids or
                        len(sources) != count):
                    raise ValueError("score model artifact scope or training lineage mismatch")
                for item in sources:
                    if not isinstance(item, Mapping):
                        raise ValueError("invalid score model training source")
                    result = _stored(db, "prospective_result", "result_id", item["result_id"])
                    event = _event(db, item["event_id"])
                    quote = _market_record(db, "prospective_quote", "quote_id", item["quote_id"],
                                           sport, market, "training quote")
                    if (result["event_id"] != event["event_id"] or
                            quote["event_id"] != event["event_id"] or
                            quote["quote_verified"] != 1 or
                            not verify_provider_offer(event, quote) or
                            item.get("result_hash") != result["source_hash"] or
                            item.get("event_hash") != event["source_hash"] or
                            item.get("quote_hash") != quote["source_hash"] or
                            item.get("available_at") != result["available_at"] or
                            item.get("start") != event["scheduled_start"] or
                            item.get("observed_at") != event["observed_at"] or
                            item.get("line") != quote["line"] or
                            not isinstance(item.get("features"), list) or
                            not isinstance(item.get("feature_lineage"), list)):
                        raise ValueError("score model training source replay failed")
                    for prior_id, prior_hash in item["feature_lineage"]:
                        prior = _stored(db, "prospective_result", "result_id", prior_id)
                        prior_event = _event(db, prior["event_id"])
                        if (prior["source_hash"] != prior_hash or
                                not verify_provider_score(prior_event, prior) or
                                _time(prior["available_at"], "prior.available_at") >
                                _time(event["observed_at"], "event.observed_at") or
                                _time(prior_event["scheduled_start"], "prior.scheduled_start") >=
                                _time(event["scheduled_start"], "event.scheduled_start")):
                            raise ValueError("score model feature lineage leaked future evidence")
            payload["model_artifact"] = artifact
        key = _insert(db, "prospective_model", "model_id", values, payload)
        for result_id in result_ids:
            db.execute("INSERT OR IGNORE INTO prospective_model_training_result VALUES (?,?)", (key, result_id))
        return key


def insert_calibration(path: str | Path | None, calibration: Mapping) -> str:
    data = _map(calibration)
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    fit_start = _time(data.get("fit_start"), "fit_start")
    fit_end = _time(data.get("fit_end"), "fit_end")
    created = _time(data.get("created_at"), "created_at")
    available = _time(data.get("available_at"), "available_at")
    if not fit_start < fit_end <= created <= available <= _clock():
        raise ValueError("calibration fit/availability chronology invalid")
    with closing(connect(path)) as db, db:
        model = _market_record(db, "prospective_model", "model_id", data.get("model_id"),
                               sport, market, "calibration model")
        if _time(model["available_at"], "model.available_at") > fit_start:
            raise ValueError("calibration model was unavailable at fit start")
        result_ids, _ = _training_results(db, data.get("fit_result_ids", []), sport, market,
                                          fit_start, fit_end)
        if not result_ids:
            raise ValueError("calibration requires exact-scope settled fit evidence")
        training_ids = {r[0] for r in db.execute(
            "SELECT result_id FROM prospective_model_training_result WHERE model_id=?", (model["model_id"],))}
        if training_ids.intersection(result_ids):
            raise ValueError("calibration fit overlaps model training evidence")
        values = dict(calibration_id=_text(data.get("calibration_id"), "calibration_id"),
                      sport=sport, market_family=market,
                      calibration_version=_text(data.get("calibration_version"), "calibration_version"),
                      model_id=model["model_id"], fit_start=_iso(fit_start), fit_end=_iso(fit_end),
                      method=_text(data.get("method"), "method"), created_at=_iso(created),
                      available_at=_iso(available),
                      artifact_hash=_digest_text(data.get("artifact_hash"), "artifact_hash"))
        payload = dict(values, fit_result_ids=result_ids)
        artifact = data.get("calibration_artifact")
        if artifact is not None:
            artifact = _map(artifact)
            if _sha(_json(artifact)) != values["artifact_hash"]:
                raise EvidenceConflict("calibration artifact hash does not match content")
            payload["calibration_artifact"] = artifact
        key = _insert(db, "prospective_calibration", "calibration_id", values, payload)
        for result_id in result_ids:
            db.execute("INSERT OR IGNORE INTO prospective_calibration_result VALUES (?,?)", (key, result_id))
        return key


def insert_prediction(path: str | Path | None, prediction: Mapping) -> str:
    data = _map(prediction)
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    stamp = _time(data.get("prediction_timestamp"), "prediction_timestamp")
    if not timedelta(0) <= _clock() - stamp <= MAX_PREDICTION_CLOCK_SKEW:
        raise ValueError("future or backdated prediction_timestamp")
    mean = _number(data.get("mean_probability"), "mean_probability", optional=True, low=0, high=1)
    conservative = _number(data.get("conservative_probability"), "conservative_probability",
                           optional=True, low=0, high=1)
    push = _number(data.get("push_probability"), "push_probability", optional=True, low=0, high=1)
    loss = _number(data.get("loss_probability"), "loss_probability", optional=True, low=0, high=1)
    if all(v is not None for v in (mean, push, loss)) and abs(mean + push + loss - 1) > 1e-6:
        raise ValueError("win/push/loss probabilities must sum to one")
    if mean is not None and conservative is not None and conservative > mean:
        raise ValueError("conservative probability exceeds mean")
    frozen = _time(data.get("feature_frozen_at"), "feature_frozen_at", optional=True)
    if frozen is not None and frozen > stamp:
        raise ValueError("feature snapshot was frozen after prediction")
    with closing(connect(path)) as db, db:
        event = _event(db, data.get("event_id"))
        if (event["sport"] != sport or
                not stamp <= _clock() < _time(event["scheduled_start"], "scheduled_start")):
            raise ValueError("prediction event scope or pregame chronology mismatch")
        quote_id = _text(data.get("quote_id"), "quote_id", optional=True)
        quote = None
        if quote_id is not None:
            quote = _market_record(db, "prospective_quote", "quote_id", quote_id,
                                   sport, market, "prediction quote")
            if quote["event_id"] != event["event_id"] or quote["selection"] != data.get("selection"):
                raise ValueError("prediction quote event/selection mismatch")
            if _time(quote["quote_timestamp"], "quote_timestamp") > stamp:
                raise ValueError("prediction cannot use a later quote")
        model_id = _text(data.get("model_id"), "model_id", optional=True)
        model = None
        if model_id is not None:
            model = _market_record(db, "prospective_model", "model_id", model_id,
                                   sport, market, "prediction model")
            if _time(model["available_at"], "model.available_at") > stamp:
                raise ValueError("model was unavailable when prediction was frozen")
        calibration_id = _text(data.get("calibration_id"), "calibration_id", optional=True)
        calibration = None
        if calibration_id is not None:
            calibration = _market_record(db, "prospective_calibration", "calibration_id", calibration_id,
                                         sport, market, "prediction calibration")
            if calibration["model_id"] != model_id:
                raise ValueError("calibration is not for the selected model")
            if _time(calibration["available_at"], "calibration.available_at") > stamp:
                raise ValueError("calibration was unavailable when prediction was frozen")
        provided_model_version = _text(data.get("model_version"), "model_version", optional=True)
        provided_calibration_version = _text(data.get("calibration_version"), "calibration_version", optional=True)
        if model is not None and provided_model_version not in (None, model["model_version"]):
            raise ValueError("model version mismatch")
        if calibration is not None and provided_calibration_version not in (None, calibration["calibration_version"]):
            raise ValueError("calibration version mismatch")
        if model is None and provided_model_version is not None:
            raise ValueError("model version without registered model")
        if calibration is None and provided_calibration_version is not None:
            raise ValueError("calibration version without registered calibration")
        values = dict(observation_id=_text(data.get("observation_id"), "observation_id"),
                      event_id=event["event_id"], sport=sport, market_family=market,
                      selection=_text(data.get("selection"), "selection"), quote_id=quote_id,
                      model_id=model_id, model_version=model["model_version"] if model else None,
                      model_trained_through=model["training_cutoff"] if model else None,
                      model_available_at=model["available_at"] if model else None,
                      feature_version=_text(data.get("feature_version"), "feature_version", optional=True),
                      feature_snapshot_id=_text(data.get("feature_snapshot_id"), "feature_snapshot_id", optional=True),
                      feature_frozen_at=_iso(frozen), calibration_id=calibration_id,
                      calibration_version=calibration["calibration_version"] if calibration else None,
                      calibration_available_at=calibration["available_at"] if calibration else None,
                      policy_version=_text(data.get("policy_version"), "policy_version", optional=True),
                      prediction_timestamp=_iso(stamp), mean_probability=mean,
                      conservative_probability=conservative, push_probability=push,
                      loss_probability=loss,
                      probability_semantics=_text(data.get("probability_semantics"), "probability_semantics", optional=True),
                      evidence_snapshot_id=_text(data.get("evidence_snapshot_id"), "evidence_snapshot_id", optional=True),
                      evidence_hash=_text(data.get("evidence_hash"), "evidence_hash", optional=True),
                      runtime_hash=_text(data.get("runtime_hash"), "runtime_hash", optional=True),
                      source_commit=_text(data.get("source_commit"), "source_commit", optional=True))
        payload = _snapshot(values)
        feature_snapshot = data.get("feature_snapshot")
        if feature_snapshot is not None:
            feature_snapshot = _map(feature_snapshot)
            if values["feature_snapshot_id"] != _sha(_json(feature_snapshot)):
                raise EvidenceConflict("feature snapshot ID does not match content")
            payload["feature_snapshot"] = feature_snapshot
        uncertainty = data.get("uncertainty")
        if uncertainty is not None:
            payload["uncertainty"] = _map(uncertainty)
            _json(payload["uncertainty"])
        return _insert(db, "prospective_prediction", "observation_id", values, payload)


def _policy(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be frozen in the validation plan")
    result = dict(value)
    _json(result)
    return result


def freeze_validation_plan(path: str | Path | None, plan: Mapping) -> str:
    """Freeze versioned methodology before its holdout begins.

    The caller cannot supply ``frozen_at``. Every material threshold change
    requires another version and a supersedes link.
    """
    data = _map(plan)
    if "frozen_at" in data:
        raise ValueError("frozen_at is assigned by the database boundary")
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    version = _integer(data.get("version"), "version", minimum=1)
    training = _time(data.get("training_cutoff"), "training_cutoff")
    validation_start = _time(data.get("validation_start"), "validation_start")
    validation_end = _time(data.get("validation_end"), "validation_end")
    holdout_start = _time(data.get("holdout_start"), "holdout_start")
    holdout_end = _time(data.get("holdout_end"), "holdout_end")
    frozen = _clock()
    if not training <= validation_start < validation_end <= holdout_start < holdout_end:
        raise ValueError("validation/holdout windows overlap or precede training")
    if frozen >= holdout_start:
        raise ValueError("validation plan must freeze before holdout starts")
    minimum_independent = _integer(data.get("minimum_independent_sample"),
                                   "minimum_independent_sample", minimum=1)
    minimum_effective = _number(data.get("minimum_effective_sample"),
                                "minimum_effective_sample", low=1)
    thresholds = _policy(data, "probability_thresholds")
    for name in ("max_brier", "max_log_loss", "max_calibration_error"):
        thresholds[name] = _number(thresholds.get(name), name, low=0)
    thresholds["min_coverage"] = _number(thresholds.get("min_coverage"),
                                         "min_coverage", low=0, high=1)
    calibration_requirements = _policy(data, "calibration_requirements")
    if calibration_requirements.get("required") is not True:
        raise ValueError("calibration must be required for validation")
    price_requirements = _policy(data, "price_evidence_requirements")
    price_requirements["min_verified_entry_coverage"] = _number(
        price_requirements.get("min_verified_entry_coverage"),
        "min_verified_entry_coverage", low=0, high=1)
    clv_policy = _policy(data, "clv_policy")
    if type(clv_policy.get("required")) is not bool:
        raise ValueError("clv_policy.required must be boolean")
    clv_policy["min_comparable_close_coverage"] = _number(
        clv_policy.get("min_comparable_close_coverage"),
        "min_comparable_close_coverage", low=0, high=1)
    value_roi_policy = _policy(data, "value_roi_policy")
    value_roi_policy["min_paper_roi"] = _number(
        value_roi_policy.get("min_paper_roi"), "min_paper_roi")
    deployment_criteria = _policy(data, "deployment_criteria")
    if deployment_criteria.get("target_state") not in STATES - {"UNVALIDATED"}:
        raise ValueError("deployment_criteria.target_state invalid")
    if ("validated_policy" in deployment_criteria and
            not isinstance(deployment_criteria["validated_policy"], Mapping)):
        raise ValueError("deployment_criteria.validated_policy must be a mapping")
    if ("maturity_rules" in deployment_criteria and
            not isinstance(deployment_criteria["maturity_rules"], Mapping)):
        raise ValueError("deployment_criteria.maturity_rules must be a mapping")
    method = data.get("independence_method")
    if method != "ONE_EARLIEST_PREDICTION_PER_EVENT_V1":
        raise ValueError("independence_method must predeclare event deduplication")
    # Optional policy provenance is part of the frozen artifact. Older plans
    # remain readable, while current plans can bind their complete methodology
    # without changing the append-only schema.
    policy_metadata = {}
    if "plan_policy_version" in data:
        policy_metadata["plan_policy_version"] = _text(
            data["plan_policy_version"], "plan_policy_version")
    if "policy_source_hash" in data:
        policy_metadata["policy_source_hash"] = _digest_text(
            data["policy_source_hash"], "policy_source_hash")
    if "source_commit" in data:
        policy_metadata["source_commit"] = _text(data["source_commit"], "source_commit")
    for name in ("model_scope", "training_cutoff_policy", "independence_policy",
                 "push_void_policy", "promotion_criteria", "football_v2_methodology"):
        if name in data:
            policy_metadata[name] = _policy(data, name)
    if "model_scope" in policy_metadata:
        scope = policy_metadata["model_scope"]
        if (scope.get("sport"), scope.get("market_family")) != (sport, market):
            raise ValueError("plan model scope must match sport/market family")
    model_id = _text(data.get("model_id"), "model_id", optional=True)
    calibration_id = _text(data.get("calibration_id"), "calibration_id", optional=True)
    supersedes = _text(data.get("supersedes_plan_id"), "supersedes_plan_id", optional=True)
    with closing(connect(path)) as db, db:
        if version == 1 and supersedes is not None:
            raise ValueError("first plan cannot supersede another")
        if version > 1:
            previous = _stored(db, "prospective_validation_plan", "validation_plan_id", supersedes or "")
            if previous is None or (previous["sport"], previous["market_family"], previous["version"]) != (
                    sport, market, version - 1):
                raise ValueError("plan version must supersede the preceding exact-scope plan")
        model = None
        if model_id is not None:
            model = _market_record(db, "prospective_model", "model_id", model_id,
                                   sport, market, "plan model")
            if _time(model["training_cutoff"], "model.training_cutoff") > training:
                raise ValueError("plan training cutoff precedes model training")
            if _time(model["available_at"], "model.available_at") > validation_start:
                raise ValueError("model unavailable at validation start")
        if calibration_id is not None:
            calibration = _market_record(db, "prospective_calibration", "calibration_id",
                                         calibration_id, sport, market, "plan calibration")
            if calibration["model_id"] != model_id:
                raise ValueError("plan calibration does not belong to plan model")
            if _time(calibration["available_at"], "calibration.available_at") > validation_start:
                raise ValueError("calibration unavailable at validation start")
        values = dict(validation_plan_id=_text(data.get("validation_plan_id"), "validation_plan_id"),
                      sport=sport, market_family=market, version=version,
                      supersedes_plan_id=supersedes, model_id=model_id, calibration_id=calibration_id,
                      training_cutoff=_iso(training), validation_start=_iso(validation_start),
                      validation_end=_iso(validation_end), holdout_start=_iso(holdout_start),
                      holdout_end=_iso(holdout_end), minimum_independent_sample=minimum_independent,
                      minimum_effective_sample=minimum_effective, frozen_at=_iso(frozen))
        methodology = dict(values, probability_thresholds=thresholds,
                           calibration_requirements=calibration_requirements,
                           price_evidence_requirements=price_requirements, clv_policy=clv_policy,
                           value_roi_policy=value_roi_policy, deployment_criteria=deployment_criteria,
                           independence_method=method,
                           coverage_policy="WIN_LOSS_SCORED_PUSH_VOID_REPORTED_PENDING_BLOCKS_V1",
                           **policy_metadata)
        values["artifact_hash"] = _sha(_json(methodology))
        return _insert(db, "prospective_validation_plan", "validation_plan_id", values,
                       dict(methodology, artifact_hash=values["artifact_hash"]))


def _latest_result(db: sqlite3.Connection, event_id: str, market: str,
                   selection: str, as_of: datetime) -> dict | None:
    rows = db.execute("""SELECT result_id FROM prospective_result WHERE event_id=?
        AND market_family=? AND selection=? AND available_at<=?
        ORDER BY grading_version ASC, available_at ASC, result_id ASC""",
        (event_id, market, selection, _iso(as_of))).fetchall()
    if not rows:
        return None
    result = None
    for row in rows:
        current = _stored(db, "prospective_result", "result_id", row["result_id"])
        if result is None:
            if current["grading_version"] != 1 or current["revises_result_id"] is not None:
                raise EvidenceConflict("result revision chain has no original")
        elif (current["revises_result_id"] != result["result_id"] or
              current["grading_version"] <= result["grading_version"] or
              _time(current["available_at"], "available_at") <=
              _time(result["available_at"], "available_at")):
            raise EvidenceConflict("ambiguous result revision chain")
        result = current
    return result


def _comparable_close(db: sqlite3.Connection, quote: dict, as_of: datetime) -> dict | None:
    rows = db.execute("""SELECT close_id FROM prospective_close WHERE event_id=?
        AND market_family=? AND selection=? AND sportsbook=? AND line=?
        AND close_verified=1 AND close_timestamp>=? AND close_timestamp<=?
        ORDER BY close_timestamp DESC,close_id DESC""",
        (quote["event_id"], quote["market_family"], quote["selection"], quote["sportsbook"],
         quote["line"], quote["quote_timestamp"], _iso(as_of))).fetchall()
    return _stored(db, "prospective_close", "close_id", rows[0]["close_id"]) if rows else None


def _cohort(db: sqlite3.Connection, plan: dict, name: str, as_of: datetime) -> dict:
    is_v2 = "football_v2_methodology" in json.loads(plan["payload"])
    start = _time(plan[f"{name}_start"], f"{name}_start")
    end = _time(plan[f"{name}_end"], f"{name}_end")
    rows = db.execute("""SELECT p.observation_id,e.scheduled_start FROM prospective_prediction p
        JOIN prospective_event e ON e.event_id=p.event_id
        WHERE p.sport=? AND p.market_family=? AND e.scheduled_start>=?
        AND e.scheduled_start<? ORDER BY e.scheduled_start,p.prediction_timestamp,p.observation_id""",
        (plan["sport"], plan["market_family"], _iso(start), _iso(end))).fetchall()
    unique: dict[str, dict] = {}
    skipped_model_scope = 0
    provenance_missing = 0
    uncertainty_missing = 0
    identity_missing = 0
    unsupported_semantics = 0
    for item in rows:
        prediction = _stored(db, "prospective_prediction", "observation_id", item["observation_id"])
        if is_v2 and _time(item["scheduled_start"], "scheduled_start") <= max(
                _time(plan["frozen_at"], "frozen_at"), start):
            skipped_model_scope += 1
            continue
        if prediction["model_id"] != plan["model_id"] or prediction["calibration_id"] != plan["calibration_id"]:
            skipped_model_scope += 1
            continue
        prediction_at = _time(prediction["prediction_timestamp"], "prediction_timestamp")
        if ((is_v2 and prediction_at <= max(_time(plan["frozen_at"], "frozen_at"), start)) or
                (not is_v2 and name == "holdout" and
                 prediction_at < _time(plan["frozen_at"], "frozen_at"))):
            skipped_model_scope += 1
            continue
        event = _event(db, prediction["event_id"])
        if event["home_team_id"] is None or event["away_team_id"] is None:
            identity_missing += 1
        if any(prediction[field] is None for field in (
                "model_version", "model_trained_through", "model_available_at",
                "feature_version", "feature_snapshot_id", "feature_frozen_at",
                "calibration_version", "calibration_available_at", "policy_version",
                "mean_probability", "conservative_probability", "push_probability",
                "loss_probability", "probability_semantics", "evidence_snapshot_id",
                "evidence_hash", "runtime_hash", "source_commit")):
            provenance_missing += 1
        if str(prediction["probability_semantics"] or "").upper() != "WIN_PUSH_LOSS":
            unsupported_semantics += 1
        if is_v2:
            from app_core.football_validation_v2 import conservative_probability
            try:
                interval = json.loads(prediction["payload"]).get("uncertainty")
                if interval.get("calibration_id") != plan["calibration_id"]:
                    raise ValueError("uncertainty calibration mismatch")
                derived = conservative_probability(prediction["mean_probability"], interval)
                if abs(derived["conservative_probability"] - prediction["conservative_probability"]) > 1e-12:
                    raise ValueError("conservative probability mismatch")
            except (AttributeError, KeyError, TypeError, ValueError):
                uncertainty_missing += 1
        unique.setdefault(prediction["event_id"], prediction)
    outcomes = {key: 0 for key in ("WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW")}
    scored = []
    unconditional_probabilities = []
    predicted_evs = []
    returns = []
    clvs = []
    quote_count = 0
    source_ids = []
    for prediction in unique.values():
        source_ids.append(prediction["observation_id"])
        quote = (_stored(db, "prospective_quote", "quote_id", prediction["quote_id"])
                 if prediction["quote_id"] else None)
        if quote is not None:
            source_ids.append(quote["quote_id"])
        valid_quote = bool(quote is not None and quote["quote_verified"] and
                           quote["line"] is not None and quote["decimal_odds"] is not None and
                           quote["sportsbook"] and
                           _time(quote["quote_timestamp"], "quote_timestamp") <
                           _time(_event(db, prediction["event_id"])["scheduled_start"], "scheduled_start") and
                           (not is_v2 or _time(quote["quote_timestamp"], "quote_timestamp") <=
                            _time(prediction["prediction_timestamp"], "prediction_timestamp")))
        if valid_quote:
            quote_count += 1
            close = _comparable_close(db, quote, as_of)
            if close is not None:
                source_ids.append(close["close_id"])
                clvs.append(quote["decimal_odds"] / close["decimal_odds"] - 1)
            if prediction["mean_probability"] is not None and prediction["loss_probability"] is not None:
                predicted_evs.append(prediction["mean_probability"] * (quote["decimal_odds"] - 1)
                                     - prediction["loss_probability"])
        result = _latest_result(db, prediction["event_id"], plan["market_family"],
                                prediction["selection"], as_of)
        outcome = result["outcome"] if result else "PENDING"
        outcomes[outcome] += 1
        if result is not None:
            source_ids.append(result["result_id"])
        if (outcome in {"WIN", "LOSS"} and (not is_v2 or valid_quote) and
                str(prediction["probability_semantics"] or "").upper() == "WIN_PUSH_LOSS" and
                prediction["mean_probability"] is not None and
                prediction["loss_probability"] is not None and
                prediction["mean_probability"] + prediction["loss_probability"] > 0):
            conditional = prediction["mean_probability"] / (
                prediction["mean_probability"] + prediction["loss_probability"])
            scored.append((conditional, int(outcome == "WIN")))
            unconditional_probabilities.append(prediction["mean_probability"])
        if valid_quote and outcome in {"WIN", "LOSS", "PUSH", "VOID"}:
            returns.append(quote["decimal_odds"] - 1 if outcome == "WIN" else
                           -1.0 if outcome == "LOSS" else 0.0)
    n = len(unique)
    brier = sum((p - y) ** 2 for p, y in scored) / len(scored) if scored else None
    floor = 0.01 if is_v2 else 1e-15
    log_loss = -sum(y * math.log(max(p, floor)) +
                    (1 - y) * math.log(max(1 - p, floor)) for p, y in scored) / len(scored) if scored else None
    buckets = []
    for i in range(10):
        members = [(p, y) for p, y in scored if min(int(p * 10), 9) == i]
        if members:
            buckets.append(dict(bin=i, count=len(members), mean_probability=sum(p for p, _ in members) / len(members),
                                observed_win_rate=sum(y for _, y in members) / len(members)))
    ece = sum(row["count"] / len(scored) * abs(row["mean_probability"] - row["observed_win_rate"])
              for row in buckets) if scored else None
    return dict(name=name, raw_predictions=len(rows), wrong_model_or_calibration_count=skipped_model_scope,
                missing_provenance_count=provenance_missing,
                **({"missing_calibrated_uncertainty_count": uncertainty_missing} if is_v2 else {}),
                missing_stable_identity_count=identity_missing,
                unsupported_probability_semantics_count=unsupported_semantics,
                unique_events=n, effective_observations=len(scored), outcomes=outcomes,
                scored_observations=len(scored), brier=brier, log_loss=log_loss,
                calibration_error=ece, calibration_curve=buckets,
                coverage=len(scored) / n if n else 0.0,
                mean_predicted_probability=(sum(unconditional_probabilities) /
                                            len(unconditional_probabilities)) if unconditional_probabilities else None,
                mean_predicted_decided_win_probability=(sum(p for p, _ in scored) /
                                                       len(scored)) if scored else None,
                observed_win_rate=sum(y for _, y in scored) / len(scored) if scored else None,
                verified_entry_count=quote_count, verified_entry_coverage=quote_count / n if n else 0.0,
                comparable_close_count=len(clvs), comparable_close_coverage=len(clvs) / n if n else 0.0,
                average_predicted_ev=sum(predicted_evs) / len(predicted_evs) if predicted_evs else None,
                paper_return=sum(returns) if returns else None,
                paper_roi=sum(returns) / len(returns) if returns else None,
                accepted_wager_roi=None,
                average_valid_clv=sum(clvs) / len(clvs) if clvs else None,
                predicted_ev_minus_paper_roi=(sum(predicted_evs) / len(predicted_evs) -
                    sum(returns) / len(returns)) if predicted_evs and returns else None,
                evidence_digest=_sha(_json(sorted(source_ids))))


def evaluate_validation_plan(path: str | Path | None, validation_plan_id: str,
                             *, as_of: datetime | str | None = None) -> dict:
    """Recompute exact-scope prospective metrics without changing deployment."""
    cutoff = _time(as_of, "as_of") if as_of is not None else _clock()
    if cutoff > _clock():
        raise ValueError("as_of cannot be in the future")
    if not _has_schema(path):
        raise ValueError("unknown validation plan")
    with closing(_reader(path)) as db:
        plan = _stored(db, "prospective_validation_plan", "validation_plan_id",
                       _text(validation_plan_id, "validation_plan_id"))
        if plan is None:
            raise ValueError("unknown validation plan")
        method = json.loads(plan["payload"])
        validation = _cohort(db, plan, "validation", cutoff)
        holdout = _cohort(db, plan, "holdout", cutoff)
    if "football_v2_methodology" in method:
        from app_core.football_validation_v2 import evaluate as evaluate_football_v2
        decision = evaluate_football_v2(plan, method, validation, holdout, cutoff)
        return dict(validation_plan_id=plan["validation_plan_id"], sport=plan["sport"],
                    market_family=plan["market_family"], model_id=plan["model_id"],
                    calibration_id=plan["calibration_id"], plan_artifact_hash=plan["artifact_hash"],
                    validation=validation, holdout=holdout, **decision)
    blockers = []
    if plan["model_id"] is None:
        blockers.append("MISSING_MODEL")
    if plan["calibration_id"] is None:
        blockers.append("MISSING_CALIBRATION")
    if cutoff < _time(plan["holdout_end"], "holdout_end"):
        blockers.append("HOLDOUT_WINDOW_OPEN")
    thresholds = method["probability_thresholds"]
    for cohort in (validation, holdout):
        label = cohort["name"].upper()
        if cohort["effective_observations"] < plan["minimum_effective_sample"]:
            blockers.append(f"NEED_{math.ceil(plan['minimum_effective_sample'] - cohort['effective_observations'])}_MORE_EFFECTIVE_{label}_EVENTS")
        if cohort["unique_events"] < plan["minimum_independent_sample"]:
            blockers.append(f"NEED_{plan['minimum_independent_sample'] - cohort['unique_events']}_MORE_SETTLED_{label}_EVENTS")
        if cohort["outcomes"]["PENDING"] or cohort["outcomes"]["NEEDS_REVIEW"]:
            blockers.append(f"{label}_OUTCOMES_UNRESOLVED")
        if cohort["wrong_model_or_calibration_count"]:
            blockers.append(f"{label}_MODEL_CALIBRATION_SCOPE_MISMATCH")
        if cohort["missing_provenance_count"]:
            blockers.append(f"{label}_PROVENANCE_INCOMPLETE")
        if cohort["missing_stable_identity_count"]:
            blockers.append(f"{label}_STABLE_TEAM_IDENTITY_MISSING")
        if cohort["unsupported_probability_semantics_count"]:
            blockers.append(f"{label}_PROBABILITY_SEMANTICS_UNSUPPORTED")
        for metric, bound in (("brier", "max_brier"), ("log_loss", "max_log_loss"),
                              ("calibration_error", "max_calibration_error")):
            if cohort[metric] is None or cohort[metric] > thresholds[bound]:
                blockers.append(f"{label}_{metric.upper()}_THRESHOLD_NOT_MET")
        if cohort["coverage"] < thresholds["min_coverage"]:
            blockers.append(f"{label}_COVERAGE_TOO_LOW")
        if (cohort["verified_entry_coverage"] <
                method["price_evidence_requirements"]["min_verified_entry_coverage"]):
            blockers.append(f"{label}_EXACT_ENTRY_PRICE_COVERAGE_TOO_LOW")
        if (method["clv_policy"]["required"] and cohort["comparable_close_coverage"] <
                method["clv_policy"]["min_comparable_close_coverage"]):
            blockers.append(f"{label}_NO_VALID_CLOSE_QUOTES")
        if (cohort["paper_roi"] is None or cohort["paper_roi"] <
                method["value_roi_policy"]["min_paper_roi"]):
            blockers.append(f"{label}_PAPER_ROI_POLICY_NOT_MET")
    return dict(validation_plan_id=plan["validation_plan_id"], sport=plan["sport"],
                market_family=plan["market_family"], model_id=plan["model_id"],
                calibration_id=plan["calibration_id"], plan_artifact_hash=plan["artifact_hash"],
                validation=validation, holdout=holdout, blockers=blockers,
                status="UNVALIDATED" if blockers else "VALIDATION_PASSED")


def create_validation_artifact(path: str | Path | None, validation_plan_id: str,
                               artifact_id: str) -> str:
    """Persist a reproducible report; a pass still grants no stake authority."""
    report = evaluate_validation_plan(path, validation_plan_id)
    created = _clock()
    report_hash = _sha(_json(report))
    values = dict(artifact_id=_text(artifact_id, "artifact_id"),
                  validation_plan_id=report["validation_plan_id"], sport=report["sport"],
                  market_family=report["market_family"], created_at=_iso(created),
                  status=report["status"], report_hash=report_hash)
    with closing(connect(path)) as db, db:
        return _insert(db, "prospective_validation_artifact", "artifact_id", values,
                       dict(values, report=report))


def record_deployment_review(path: str | Path | None, review: Mapping) -> str:
    """Record independent market validation review, never owner activation."""
    data = _map(review)
    sport, market = _scope(data.get("sport"), data.get("market_family"))
    state = data.get("deployment_state")
    if state not in STATES - {"UNVALIDATED"}:
        raise ValueError("review state must be a validated tier")
    reviewed = _clock()
    with closing(connect(path)) as db, db:
        artifact = _market_record(db, "prospective_validation_artifact", "artifact_id",
                                  data.get("artifact_id"), sport, market, "review artifact")
        if artifact["status"] != "VALIDATION_PASSED":
            raise ValueError("deployment review needs a passed validation artifact")
        plan = _market_record(db, "prospective_validation_plan", "validation_plan_id",
                              artifact["validation_plan_id"], sport, market, "review plan")
        latest = db.execute("""SELECT MAX(version) FROM prospective_validation_plan
            WHERE sport=? AND market_family=?""", (sport, market)).fetchone()[0]
        if plan["version"] != latest:
            raise ValueError("deployment review cannot use superseded plan")
        target = json.loads(plan["payload"])["deployment_criteria"]["target_state"]
        if state != target:
            raise ValueError("review state differs from frozen deployment criteria")
    # Recompute outside the write transaction. Corrections or new evidence can
    # make an old pass stale; the check is repeated by deployment_state.
    current = evaluate_validation_plan(path, plan["validation_plan_id"])
    saved_report = json.loads(artifact["payload"])["report"]
    if (current["status"] != "VALIDATION_PASSED" or
            _sha(_json(current)) != artifact["report_hash"] or current != saved_report):
        raise ValueError("validation artifact is stale or no longer passing")
    values = dict(deployment_id=_text(data.get("deployment_id"), "deployment_id"),
                  artifact_id=artifact["artifact_id"], validation_plan_id=plan["validation_plan_id"],
                  sport=sport, market_family=market,
                  validation_id=_text(data.get("validation_id"), "validation_id"),
                  deployment_state=state, reviewer_id=_text(data.get("reviewer_id"), "reviewer_id"),
                  reviewed_at=_iso(reviewed))
    with closing(connect(path)) as db, db:
        return _insert(db, "prospective_deployment_review", "deployment_id", values, values)


def deployment_state(path: str | Path | None, sport: str, market_family: str) -> dict:
    """Exact sport/market status; even reviewed validation grants $0 by itself."""
    _scope(sport, market_family)
    baseline = dict(sport=sport, market_family=market_family,
                    validation_state="UNVALIDATED", deployment_state="UNVALIDATED",
                    validation_id=None, model_id=None, model_version=None,
                    calibration_id=None, calibration_version=None, artifact_id=None,
                    validation_report=None, deployment_criteria=None,
                    validated_policy=None,
                    owner_authorized=False, production_eligible=False, recommended_stake=0.0)
    if not _has_schema(path):
        return baseline
    with closing(_reader(path)) as db:
        row = db.execute("""SELECT deployment_id FROM prospective_deployment_review
            WHERE sport=? AND market_family=? ORDER BY reviewed_at DESC,deployment_id DESC LIMIT 1""",
            (sport, market_family)).fetchone()
        if row is None:
            return baseline
        review = _stored(db, "prospective_deployment_review", "deployment_id", row["deployment_id"])
        artifact = _stored(db, "prospective_validation_artifact", "artifact_id", review["artifact_id"])
        plan = _stored(db, "prospective_validation_plan", "validation_plan_id", review["validation_plan_id"])
        latest_version = db.execute("""SELECT MAX(version) FROM prospective_validation_plan
            WHERE sport=? AND market_family=?""", (sport, market_family)).fetchone()[0]
        if plan["version"] != latest_version or artifact["status"] != "VALIDATION_PASSED":
            return baseline
        plan_payload = json.loads(plan["payload"])
        report = json.loads(artifact["payload"])["report"]
        model = (_stored(db, "prospective_model", "model_id", plan["model_id"])
                 if plan["model_id"] else None)
        calibration = (_stored(db, "prospective_calibration", "calibration_id", plan["calibration_id"])
                       if plan["calibration_id"] else None)
    current = evaluate_validation_plan(path, plan["validation_plan_id"])
    if (current["status"] != "VALIDATION_PASSED" or
            _sha(_json(current)) != artifact["report_hash"] or current != report):
        return baseline
    return dict(baseline, validation_state=review["deployment_state"],
                deployment_state=review["deployment_state"],
                validation_id=review["validation_id"],
                model_id=plan["model_id"], model_version=model["model_version"] if model else None,
                calibration_id=plan["calibration_id"],
                calibration_version=calibration["calibration_version"] if calibration else None,
                artifact_id=artifact["artifact_id"], validation_report=report,
                deployment_criteria=plan_payload["deployment_criteria"],
                validated_policy=plan_payload["deployment_criteria"].get("validated_policy"))


def read_records(path: str | Path | None, table: str, *, sport: str | None = None,
                 market_family: str | None = None) -> list[dict]:
    """Integrity-checked canonical evidence read; callers cannot mutate rows."""
    if table not in TABLE_KEYS:
        raise ValueError("unsupported evidence table")
    if sport is not None:
        _scope(sport, market_family)
    elif market_family is not None:
        raise ValueError("market_family requires sport")
    where = []
    params = []
    if sport is not None:
        where.append("sport=?")
        params.append(sport)
    if market_family is not None:
        if table == "prospective_event":
            raise ValueError("event has no market_family")
        where.append("market_family=?")
        params.append(market_family)
    if not _has_schema(path):
        return []
    sql = f"SELECT {TABLE_KEYS[table]} FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY rowid"
    with closing(_reader(path)) as db:
        return [dict(_stored(db, table, TABLE_KEYS[table], item[0]))
                for item in db.execute(sql, params).fetchall()]


def load_record(path: str | Path | None, table: str, record_id: str) -> dict | None:
    """Integrity-check one immutable source-bound record without writing."""
    if table not in TABLE_KEYS:
        raise ValueError("unsupported evidence table")
    if not _has_schema(path):
        return None
    with closing(_reader(path)) as db:
        return _stored(db, table, TABLE_KEYS[table], _text(record_id, "record_id"))


def market_readiness(path: str | Path | None, sport: str, market_family: str) -> dict:
    """One exact-market readiness row with an evidence-derived next blocker."""
    _scope(sport, market_family)
    if not _has_schema(path):
        return dict(deployment_state(path, sport, market_family),
                    capture_status="NO_EVENTS", discovered_events=0,
                    pregame_observations=0, unique_events=0, settled_events=0,
                    verified_entry_prices=0, verified_closes=0, model_status="MISSING",
                    model_artifact_hash=None, training_count=0,
                    calibration_status="MISSING", validation_plan_status="MISSING",
                    validation_plan_artifact_hash=None, validation_artifact_count=0,
                    validation_count=0, holdout_count=0, effective_sample=0,
                    price_status="MISSING", close_clv_status="UNAVAILABLE",
                    next_blocker="NO_PROSPECTIVE_EVENTS")
    with closing(_reader(path)) as db:
        discovered_events = db.execute("SELECT COUNT(*) FROM prospective_event WHERE sport=?",
                                       (sport,)).fetchone()[0]
        event_ids = {r[0] for r in db.execute("""SELECT DISTINCT event_id FROM prospective_quote
            WHERE sport=? AND market_family=? UNION SELECT DISTINCT event_id
            FROM prospective_prediction WHERE sport=? AND market_family=?""",
            (sport, market_family, sport, market_family))}
        prediction_count = db.execute("""SELECT COUNT(*) FROM prospective_prediction
            WHERE sport=? AND market_family=?""", (sport, market_family)).fetchone()[0]
        quote_count = db.execute("""SELECT COUNT(*) FROM prospective_quote
            WHERE sport=? AND market_family=? AND quote_verified=1""", (sport, market_family)).fetchone()[0]
        close_count = db.execute("""SELECT COUNT(*) FROM prospective_close
            WHERE sport=? AND market_family=? AND close_verified=1""", (sport, market_family)).fetchone()[0]
        settled_ids = {r[0] for r in db.execute("""SELECT DISTINCT event_id FROM prospective_result
            WHERE sport=? AND market_family=? AND outcome IN ('WIN','LOSS','PUSH','VOID')""",
            (sport, market_family))}
        missing_identity = 0
        if event_ids:
            placeholders = ",".join("?" for _ in event_ids)
            missing_identity = db.execute(f"""SELECT COUNT(*) FROM prospective_event
                WHERE event_id IN ({placeholders}) AND
                (home_team_id IS NULL OR away_team_id IS NULL)""", tuple(event_ids)).fetchone()[0]
        model_row = db.execute("""SELECT model_id,training_observation_count,artifact_hash
            FROM prospective_model WHERE sport=? AND market_family=? AND training_observation_count>0
            ORDER BY available_at DESC,model_id DESC LIMIT 1""", (sport, market_family)).fetchone()
        model_count = int(model_row is not None)
        calibration_count = db.execute("""SELECT COUNT(*) FROM prospective_calibration
            WHERE sport=? AND market_family=?""", (sport, market_family)).fetchone()[0]
        plan = db.execute("""SELECT validation_plan_id FROM prospective_validation_plan
            WHERE sport=? AND market_family=? ORDER BY version DESC LIMIT 1""",
            (sport, market_family)).fetchone()
        artifact_count = db.execute("""SELECT COUNT(*) FROM prospective_validation_artifact
            WHERE sport=? AND market_family=? AND status='VALIDATION_PASSED'""",
            (sport, market_family)).fetchone()[0]
    deployment = deployment_state(path, sport, market_family)
    validation = evaluate_validation_plan(path, plan[0]) if plan else None
    if not discovered_events:
        blocker = "NO_PROSPECTIVE_EVENTS"
    elif not event_ids:
        blocker = "NO_MARKET_QUOTES_OR_PREDICTIONS"
    elif missing_identity:
        blocker = "MISSING_STABLE_TEAM_IDS"
    elif not prediction_count:
        blocker = "NO_FROZEN_PREGAME_PREDICTIONS"
    elif not quote_count:
        blocker = "NO_VERIFIED_EXACT_PREGAME_PRICES"
    elif not model_count:
        blocker = "MISSING_SPORT_MARKET_MODEL"
    elif not calibration_count:
        blocker = "MISSING_CALIBRATION"
    elif not plan:
        blocker = "MISSING_FROZEN_VALIDATION_PLAN"
    elif validation["blockers"]:
        blocker = validation["blockers"][0]
    elif not artifact_count:
        blocker = "MISSING_IMMUTABLE_VALIDATION_ARTIFACT"
    elif deployment["deployment_state"] == "UNVALIDATED":
        blocker = "MISSING_INDEPENDENT_DEPLOYMENT_REVIEW"
    else:
        blocker = "MISSING_OWNER_BANKROLL_EXPOSURE_AUTHORITY"
    return dict(deployment, capture_status="CAPTURED" if event_ids else
                "DISCOVERED_NO_MARKET_EVIDENCE" if discovered_events else "NO_EVENTS",
                discovered_events=discovered_events,
                pregame_observations=prediction_count, unique_events=len(event_ids),
                settled_events=len(settled_ids), verified_entry_prices=quote_count,
                verified_closes=close_count, model_status="REGISTERED" if model_count else "MISSING",
                model_artifact_hash=model_row["artifact_hash"] if model_row else None,
                training_count=model_row["training_observation_count"] if model_row else 0,
                calibration_status="REGISTERED" if calibration_count else "MISSING",
                validation_plan_status="FROZEN" if plan else "MISSING",
                validation_plan_artifact_hash=validation["plan_artifact_hash"] if validation else None,
                validation_artifact_count=artifact_count,
                validation_count=validation["validation"]["unique_events"] if validation else 0,
                holdout_count=validation["holdout"]["unique_events"] if validation else 0,
                effective_sample=validation["holdout"]["effective_observations"] if validation else 0,
                price_status="VERIFIED" if quote_count else "MISSING",
                close_clv_status="AVAILABLE" if close_count else "UNAVAILABLE",
                next_blocker=blocker)


def all_market_readiness(path: str | Path | None = None) -> list[dict]:
    return [market_readiness(path, sport, market) for sport, markets in SPORT_MARKETS.items()
            for market in markets]
