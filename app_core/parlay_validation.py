"""Prospective, product-scoped parlay validation evidence.

This is an append-only extension of ``parlay_persistence``. A passed report is
evidence for an independent review, never stake or sportsbook order authority.
No plan or candidate can be backdated through this API: the database records
the wall-clock freeze time, and every underlying event must still be future.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Mapping

from app_core import parlay_persistence as store


PRODUCTS = frozenset(store.PRODUCTS)
OUTCOMES = frozenset({"WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW"})
COHORTS = ("validation", "holdout")
INDEPENDENCE_METHOD = "SHARED_EVENT_CONNECTED_COMPONENTS_V1"


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def _time(value: object, name: str) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError:
        raise ValueError(f"{name} requires a timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


def _optional_time(value: object, name: str) -> datetime | None:
    return _time(value, name) if value is not None else None


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _finite(value: object, name: str, *, low: float | None = None,
            high: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result) or (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f"{name} must be finite and within its policy bounds")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _serialized(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_serialized(value).encode("utf-8")).hexdigest()


def _verified(payload: str, digest: str) -> dict:
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
        raise store.EvidenceConflict("stored prospective evidence hash mismatch")
    return json.loads(payload)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Migrate prospective tables in the same SQLite file as parlay decisions."""
    db = store.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS parlay_validation_plan (
            plan_id TEXT PRIMARY KEY,
            product_type TEXT NOT NULL CHECK (product_type IN
                ('STANDARD_PARLAY','SAME_GAME_PARLAY','CROSS_GAME_PARLAY')),
            version INTEGER NOT NULL CHECK (version > 0),
            supersedes_plan_id TEXT REFERENCES parlay_validation_plan(plan_id),
            frozen_at TEXT NOT NULL,
            training_cutoff TEXT NOT NULL,
            validation_start TEXT NOT NULL,
            validation_end TEXT NOT NULL,
            holdout_start TEXT NOT NULL,
            holdout_end TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            UNIQUE (product_type, version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_candidate (
            decision_id TEXT PRIMARY KEY REFERENCES parlay_ticket(decision_id),
            plan_id TEXT NOT NULL REFERENCES parlay_validation_plan(plan_id),
            parlay_id TEXT NOT NULL REFERENCES parlay_identity(parlay_id),
            product_type TEXT NOT NULL,
            cohort TEXT NOT NULL CHECK (cohort IN ('validation','holdout')),
            frozen_at TEXT NOT NULL,
            first_event_start TEXT NOT NULL,
            last_event_start TEXT NOT NULL,
            ticket_hash TEXT NOT NULL,
            quote_id TEXT,
            probability_mean REAL,
            probability_conservative REAL,
            probability_admissible INTEGER NOT NULL CHECK (probability_admissible IN (0,1)),
            event_keys TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_quote_source (
            decision_id TEXT NOT NULL,
            quote_id TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN
                ('PROVIDER_API','SPORTSBOOK_DISPLAY','SPORTSBOOK_SCREENSHOT')),
            raw_evidence BLOB NOT NULL,
            raw_evidence_hash TEXT NOT NULL,
            attached_at TEXT NOT NULL,
            PRIMARY KEY (decision_id,quote_id),
            FOREIGN KEY (decision_id,quote_id) REFERENCES parlay_quote(decision_id,quote_id)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_wager_source (
            decision_id TEXT NOT NULL,
            grading_version INTEGER NOT NULL,
            raw_response BLOB NOT NULL,
            raw_response_hash TEXT NOT NULL,
            attached_at TEXT NOT NULL,
            PRIMARY KEY (decision_id,grading_version),
            FOREIGN KEY (decision_id,grading_version)
                REFERENCES parlay_result(decision_id,grading_version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_result_source (
            decision_id TEXT NOT NULL,
            grading_version INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            available_at TEXT NOT NULL,
            raw_response BLOB NOT NULL,
            raw_response_hash TEXT NOT NULL,
            attached_at TEXT NOT NULL,
            PRIMARY KEY (decision_id,grading_version),
            FOREIGN KEY (decision_id,grading_version)
                REFERENCES parlay_result(decision_id,grading_version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_closing_source (
            decision_id TEXT NOT NULL,
            grading_version INTEGER NOT NULL,
            raw_response BLOB NOT NULL,
            raw_response_hash TEXT NOT NULL,
            attached_at TEXT NOT NULL,
            PRIMARY KEY (decision_id,grading_version),
            FOREIGN KEY (decision_id,grading_version)
                REFERENCES parlay_result(decision_id,grading_version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_outcome (
            decision_id TEXT NOT NULL REFERENCES parlay_validation_candidate(decision_id),
            grading_version INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN
                ('WIN','LOSS','PUSH','VOID','PENDING','NEEDS_REVIEW')),
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY (decision_id, grading_version),
            FOREIGN KEY (decision_id, grading_version)
                REFERENCES parlay_result(decision_id, grading_version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_artifact (
            artifact_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES parlay_validation_plan(plan_id),
            created_at TEXT NOT NULL,
            report_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('UNVALIDATED','VALIDATION_PASSED')),
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parlay_deployment_evidence (
            deployment_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES parlay_validation_plan(plan_id),
            artifact_id TEXT NOT NULL REFERENCES parlay_validation_artifact(artifact_id),
            product_type TEXT NOT NULL CHECK (product_type IN
                ('STANDARD_PARLAY','SAME_GAME_PARLAY','CROSS_GAME_PARLAY')),
            validation_id TEXT NOT NULL,
            validation_state TEXT NOT NULL CHECK (validation_state IN
                ('PROVISIONAL_VALIDATED','STANDARD_VALIDATED')),
            reviewer_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS parlay_validation_candidate_cohort
            ON parlay_validation_candidate(plan_id,cohort,first_event_start);
        CREATE INDEX IF NOT EXISTS parlay_deployment_product
            ON parlay_deployment_evidence(product_type,reviewed_at);
    """)
    candidate_columns = {item[1] for item in db.execute("PRAGMA table_info(parlay_validation_candidate)")}
    if "last_event_start" not in candidate_columns:
        db.execute("ALTER TABLE parlay_validation_candidate ADD COLUMN last_event_start TEXT")
    for table in ("parlay_validation_plan", "parlay_validation_candidate",
                  "parlay_validation_quote_source", "parlay_validation_wager_source",
                  "parlay_validation_result_source", "parlay_validation_closing_source",
                  "parlay_validation_outcome", "parlay_validation_artifact",
                  "parlay_deployment_evidence"):
        for action in ("UPDATE", "DELETE"):
            db.execute(
                f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                f"BEFORE {action} ON {table} BEGIN "
                "SELECT RAISE(ABORT, 'prospective evidence is append-only'); END"
            )
    return db


def attach_quote_source(path: str | Path | None, decision_id: str, *,
                        raw_response: Mapping | None = None,
                        artifact_bytes: bytes | None = None) -> bool:
    """Retain replayable source material for an exact provider or owner quote.

    Provider material must be the canonical response returned by the adapter;
    owner material is the authenticated owner's retained sportsbook artifact.
    Neither path creates a validation ID. Provider authenticity still requires
    the external adapter and audit of its upstream connection.
    """
    snapshot = store.load_decision(path, decision_id)
    if snapshot is None or snapshot["quote"] is None:
        raise ValueError("saved decision with exact quote required")
    quote = snapshot["quote"]
    if (raw_response is None) == (artifact_bytes is None):
        raise ValueError("provide exactly one raw provider response or owner artifact")
    if raw_response is not None:
        if quote.get("source") != "SPORTSBOOK" or quote.get("source_type") != "PROVIDER_API":
            raise ValueError("provider response source mismatch")
        raw = _mapping(raw_response, "raw_response")
        if raw.get("price_origin") != "EXECUTABLE_TICKET" or any(
                raw.get(name) != quote.get(name) for name in
                ("provider", "product_type", "sportsbook", "quote_id", "provider_ticket_id",
                 "provider_response_id", "selection_bindings", "component_hashes",
                 "sgp_components", "american_odds", "decimal_odds", "quoted_at",
                 "expires_at", "settlement_rules_id")):
            raise ValueError("raw provider response does not match exact saved ticket")
        raw_bytes = _serialized(raw).encode("utf-8")
        source_type = "PROVIDER_API"
    else:
        if quote.get("source") != "OWNER_CONFIRMED" or quote.get("source_type") not in {
                "SPORTSBOOK_DISPLAY", "SPORTSBOOK_SCREENSHOT"}:
            raise ValueError("owner artifact source mismatch")
        if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
            raise ValueError("owner artifact bytes required")
        raw_bytes = artifact_bytes
        source_type = quote["source_type"]
    if len(raw_bytes) > 2_000_000:
        raise ValueError("quote source exceeds retained-evidence size limit")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != quote.get("raw_evidence_hash"):
        raise ValueError("quote raw evidence hash mismatch")
    if not _quote_verified(snapshot["decision"], snapshot["legs"], quote, _clock()):
        raise ValueError("saved quote fails exact current binding")
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        found = db.execute("SELECT raw_evidence_hash,raw_evidence FROM parlay_validation_quote_source "
                           "WHERE decision_id=? AND quote_id=?",
                           (decision_id,quote["quote_id"])).fetchone()
        if found:
            if found != (digest,raw_bytes):
                raise store.EvidenceConflict("quote source already bound to different bytes")
            return False
        db.execute("INSERT INTO parlay_validation_quote_source VALUES (?,?,?,?,?,?)", (
            decision_id, quote["quote_id"], source_type, raw_bytes, digest, _clock().isoformat()))
    return True


def _has_quote_source(path: str | Path | None, decision_id: str, quote_id: str,
                      freeze_at: datetime) -> bool:
    with closing(connect(path)) as db:
        row = db.execute("SELECT source_type,raw_evidence,raw_evidence_hash,attached_at "
                         "FROM parlay_validation_quote_source WHERE decision_id=? AND quote_id=?",
                         (decision_id,quote_id)).fetchone()
    return bool(row and hashlib.sha256(row[1]).hexdigest() == row[2] and
                _time(row[3], "quote source attached_at") <= freeze_at)


def attach_accepted_wager_source(path: str | Path | None, decision_id: str,
                                 grading_version: int, raw_response: Mapping) -> bool:
    """Bind a replayable sportsbook acceptance response to a graded wager.

    A bare accepted_wager_id in a result is never enough to count ROI. The
    response must independently reproduce exact ticket, price, stake and time.
    """
    _integer(grading_version, "grading_version", minimum=1)
    snapshot = store.load_decision(path, decision_id)
    revisions = store.result_revisions(path, decision_id)
    result = next((item for item in revisions if item.get("grading_version") == grading_version), None)
    if snapshot is None or result is None or result.get("result_kind") != "ACCEPTED_WAGER":
        raise ValueError("saved accepted-wager grading revision required")
    raw = _mapping(raw_response, "raw_response")
    decision = snapshot["decision"]
    quote = snapshot["quote"]
    if (quote is None or raw.get("acceptance_state") != "ACCEPTED" or
            not raw.get("provider_response_id") or
            raw.get("provider") != quote.get("provider") or
            raw.get("sportsbook") != decision.get("sportsbook") or
            raw.get("ticket_hash") != decision.get("ticket_hash") or
            raw.get("quote_id") != quote.get("quote_id") or
            raw.get("accepted_wager_id") != result.get("accepted_wager_id") or
            raw.get("decimal_odds") != result.get("accepted_decimal_odds") or
            raw.get("stake") != result.get("accepted_stake") or
            raw.get("placed_at") != result.get("placed_at")):
        raise ValueError("acceptance response does not match saved wager")
    if not (_time(quote["quoted_at"], "quoted_at") <=
            _time(raw["placed_at"], "placed_at") <
            min(_time(leg.get("game_start_at") or leg.get("start"), "leg.start")
                for leg in snapshot["legs"])):
        raise ValueError("accepted wager chronology invalid")
    raw_bytes = _serialized(raw).encode("utf-8")
    if len(raw_bytes) > 2_000_000 or hashlib.sha256(raw_bytes).hexdigest() != result.get(
            "accepted_wager_evidence_hash"):
        raise ValueError("accepted wager source hash mismatch")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT raw_response_hash,raw_response FROM parlay_validation_wager_source "
                           "WHERE decision_id=? AND grading_version=?",
                           (decision_id,grading_version)).fetchone()
        if prior:
            if prior != (digest,raw_bytes):
                raise store.EvidenceConflict("accepted-wager source already differs")
            return False
        db.execute("INSERT INTO parlay_validation_wager_source VALUES (?,?,?,?,?)", (
            decision_id,grading_version,raw_bytes,digest,_clock().isoformat()))
    return True


def _has_wager_source(db: sqlite3.Connection, decision_id: str, grading_version: int) -> bool:
    source = db.execute("SELECT raw_response,raw_response_hash FROM parlay_validation_wager_source "
                        "WHERE decision_id=? AND grading_version=?",
                        (decision_id,grading_version)).fetchone()
    return bool(source and hashlib.sha256(source[0]).hexdigest() == source[1])


def attach_result_source(path: str | Path | None, decision_id: str,
                         grading_version: int, raw_response: Mapping) -> bool:
    """Retain the source result and its availability before scoring a ticket."""
    _integer(grading_version,"grading_version",minimum=1)
    raw = _mapping(raw_response,"raw_response")
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        candidate_row = db.execute("SELECT payload,payload_hash FROM parlay_validation_candidate "
                                   "WHERE decision_id=?",(decision_id,)).fetchone()
        result_row = db.execute("SELECT payload,payload_hash FROM parlay_result "
                                "WHERE decision_id=? AND grading_version=?",
                                (decision_id,grading_version)).fetchone()
        if not candidate_row or not result_row:
            raise ValueError("frozen candidate and grading revision required")
        candidate,result = _verified(*candidate_row),_verified(*result_row)
        if raw.get("source_type") not in {"OFFICIAL_RESULT","SPORTSBOOK_SETTLEMENT"} or (
                result.get("result_kind") == "ACCEPTED_WAGER" and
                raw.get("source_type") != "SPORTSBOOK_SETTLEMENT"):
            raise ValueError("result source type invalid")
        leg_outcomes = result.get("leg_outcomes")
        if not isinstance(leg_outcomes,list) or len(leg_outcomes) != len(candidate["leg_hashes"]) or any(
                item not in OUTCOMES for item in leg_outcomes):
            raise ValueError("leg outcomes are required for reproducible settlement")
        if any(raw.get(name) != value for name,value in (
                ("source_id",result.get("source_id")),
                ("ticket_hash",candidate["ticket_hash"]),
                ("event_keys",candidate["event_keys"]),
                ("outcome",result.get("outcome")),
                ("leg_outcomes",leg_outcomes),
                ("settlement_rule",result.get("settlement_rule")))) or not raw.get("provider_response_id"):
            raise ValueError("result source does not match frozen ticket and grading")
        if result.get("result_kind") == "ACCEPTED_WAGER" and any(raw.get(name) != result.get(target)
                for name,target in (("accepted_wager_id","accepted_wager_id"),
                                    ("net_return","net_return"),
                                    ("accepted_stake","accepted_stake"),
                                    ("accepted_decimal_odds","accepted_decimal_odds"))):
            raise ValueError("sportsbook settlement does not match accepted wager")
        available = _time(raw.get("available_at"),"raw available_at")
        if not _time(candidate["last_event_start"],"last_event_start") <= available <= _clock():
            raise ValueError("result source availability chronology invalid")
        raw_bytes = _serialized(raw).encode("utf-8")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if len(raw_bytes) > 2_000_000 or digest != result.get("result_evidence_hash"):
            raise ValueError("result source hash mismatch")
        prior = db.execute("SELECT raw_response_hash,raw_response FROM parlay_validation_result_source "
                           "WHERE decision_id=? AND grading_version=?",
                           (decision_id,grading_version)).fetchone()
        if prior:
            if prior != (digest,raw_bytes):
                raise store.EvidenceConflict("result source already differs")
            return False
        db.execute("INSERT INTO parlay_validation_result_source VALUES (?,?,?,?,?,?,?)",(
            decision_id,grading_version,raw["source_id"],available.isoformat(),
            raw_bytes,digest,_clock().isoformat()))
    return True


def attach_closing_source(path: str | Path | None, decision_id: str,
                          grading_version: int, raw_response: Mapping) -> bool:
    """Retain a comparable exact closing ticket quote for optional CLV."""
    _integer(grading_version,"grading_version",minimum=1)
    raw = _mapping(raw_response,"raw_response")
    snapshot = store.load_decision(path,decision_id)
    result = next((item for item in store.result_revisions(path,decision_id)
                   if item.get("grading_version") == grading_version),None)
    if snapshot is None or result is None or snapshot["quote"] is None:
        raise ValueError("saved decision, opening quote and result are required")
    quote = snapshot["quote"]
    if raw.get("price_origin") != "CLOSING_TICKET" or not raw.get("provider_response_id") or any(
            raw.get(name) != value for name,value in (
                ("source_id",result.get("closing_source_id")),
                ("ticket_hash",snapshot["decision"]["ticket_hash"]),
                ("sportsbook",quote.get("sportsbook")),
                ("settlement_rules_id",quote.get("settlement_rules_id")),
                ("selection_bindings",quote.get("selection_bindings")),
                ("component_hashes",quote.get("component_hashes")),
                ("sgp_components",quote.get("sgp_components")),
                ("decimal_odds",result.get("closing_decimal_odds")),
                ("quoted_at",result.get("closing_quoted_at")))):
        raise ValueError("closing quote is not comparable to opening ticket")
    closing_at = _time(raw["quoted_at"],"closing quoted_at")
    if not _time(quote["quoted_at"],"opening quoted_at") <= closing_at < min(
            _time(leg.get("game_start_at") or leg.get("start"),"leg.start")
            for leg in snapshot["legs"]):
        raise ValueError("closing quote chronology invalid")
    _finite(raw.get("decimal_odds"),"closing decimal_odds",low=1.000001)
    raw_bytes = _serialized(raw).encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if len(raw_bytes) > 2_000_000 or digest != result.get("closing_evidence_hash"):
        raise ValueError("closing source hash mismatch")
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT raw_response_hash,raw_response FROM parlay_validation_closing_source "
                           "WHERE decision_id=? AND grading_version=?",
                           (decision_id,grading_version)).fetchone()
        if prior:
            if prior != (digest,raw_bytes):
                raise store.EvidenceConflict("closing source already differs")
            return False
        db.execute("INSERT INTO parlay_validation_closing_source VALUES (?,?,?,?,?)",(
            decision_id,grading_version,raw_bytes,digest,_clock().isoformat()))
    return True


def _source_matches(db: sqlite3.Connection, table: str, decision_id: str,
                    grading_version: int) -> bool:
    row = db.execute(f"SELECT raw_response,raw_response_hash FROM {table} "
                     "WHERE decision_id=? AND grading_version=?",
                     (decision_id,grading_version)).fetchone()
    return bool(row and hashlib.sha256(row[0]).hexdigest() == row[1])


def _validate_plan_fields(plan: dict, now: datetime) -> tuple[dict, dict]:
    product = plan.get("product_type")
    if product not in PRODUCTS:
        raise ValueError("invalid parlay product")
    _text(plan.get("plan_id"), "plan_id")
    _integer(plan.get("version"), "version", minimum=1)
    _text(plan.get("product_policy_version"), "product_policy_version")
    scope = _mapping(plan.get("model_scope"), "model_scope")
    for name in ("joint_model_id", "joint_model_version", "calibration_version", "method"):
        _text(scope.get(name), f"model_scope.{name}")
    if product == "STANDARD_PARLAY" and "INDEPEND" not in scope["method"].upper():
        _text(scope.get("dependence_methodology_id"), "model_scope.dependence_methodology_id")
    if product == "SAME_GAME_PARLAY":
        if "INDEPEND" in scope["method"].upper():
            raise ValueError("SGP independent multiplication is prohibited")
        _text(scope.get("correlation_method"), "model_scope.correlation_method")
    if product == "CROSS_GAME_PARLAY":
        if scope.get("preserve_sgp_blocks") is not True:
            raise ValueError("cross-game plan must preserve SGP component blocks")
        for name in ("component_dependence_method", "shared_factor_method",
                     "final_calibration_id", "final_calibration_version"):
            _text(scope.get(name), f"model_scope.{name}")
    scopes = plan.get("sport_market_scope")
    if not isinstance(scopes, list) or not scopes or any(
            not isinstance(item, Mapping) or not item.get("sport") or not item.get("market") for item in scopes):
        raise ValueError("sport_market_scope requires sport and market")
    training = _time(plan.get("training_cutoff"), "training_cutoff")
    manifest = _mapping(plan.get("training_manifest"), "training_manifest")
    if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_manifest_hash"))) is None:
        raise ValueError("training source manifest hash is required")
    if any(_time(manifest.get(key), f"training_manifest.{key}") > training for key in
           ("latest_outcome_available_at", "latest_feature_available_at")):
        raise ValueError("training inputs became available after training cutoff")
    windows = _mapping(plan.get("windows"), "windows")
    validation = _mapping(windows.get("validation"), "validation window")
    holdout = _mapping(windows.get("holdout"), "holdout window")
    vs, ve = (_time(validation.get(key), f"validation.{key}") for key in ("start", "end"))
    hs, he = (_time(holdout.get(key), f"holdout.{key}") for key in ("start", "end"))
    if not training < now <= vs < ve <= hs < he:
        raise ValueError("plan chronology must freeze before validation and holdout")
    minimums = _mapping(plan.get("minimum_independent_units"), "minimum_independent_units")
    for cohort in COHORTS:
        _integer(minimums.get(cohort), f"minimum_independent_units.{cohort}", minimum=1)
    probability = _mapping(plan.get("probability_thresholds"), "probability_thresholds")
    for name in ("max_brier", "max_log_loss", "max_calibration_error"):
        _finite(probability.get(name), f"probability_thresholds.{name}", low=0)
    _finite(probability.get("min_coverage"), "probability_thresholds.min_coverage", low=0, high=1)
    if probability.get("coverage_denominator") != "ALL_CANDIDATES":
        raise ValueError("coverage denominator must include pending and void candidates")
    betting = _mapping(plan.get("betting_thresholds"), "betting_thresholds")
    _integer(betting.get("min_accepted_wagers"), "betting_thresholds.min_accepted_wagers", minimum=1)
    if betting.get("min_roi") is not None:
        _finite(betting["min_roi"], "betting_thresholds.min_roi")
    quote_policy = _mapping(plan.get("price_evidence_requirements"), "price_evidence_requirements")
    if quote_policy.get("require_verified_exact_quote") is not True:
        raise ValueError("verified exact ticket prices are mandatory")
    clv = _mapping(plan.get("clv_policy"), "clv_policy")
    if type(clv.get("required")) is not bool or clv.get("comparable_closing_only") is not True:
        raise ValueError("CLV policy must require comparable closing evidence")
    roi = _mapping(plan.get("roi_reporting_policy"), "roi_reporting_policy")
    if roi.get("accepted_wagers_only") is not True or roi.get("include_void_in_denominator") is not True:
        raise ValueError("ROI must use accepted wagers and report voids")
    deployment = _mapping(plan.get("deployment_state_criteria"), "deployment_state_criteria")
    if deployment.get("target_state") not in {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED"}:
        raise ValueError("deployment target state is required")
    if plan.get("independence_method") != INDEPENDENCE_METHOD:
        raise ValueError("unsupported independent-unit method")
    return scope, {"training": training, "validation_start": vs,
                   "validation_end": ve, "holdout_start": hs, "holdout_end": he}


def freeze_plan(path: str | Path | None, plan: Mapping) -> dict:
    """Freeze a new, prospectively timed plan; changes require a new version."""
    row = _mapping(plan, "plan")
    if any(key in row for key in ("created_at", "frozen_at", "artifact_hash")):
        raise ValueError("plan timestamps and hash are stamped by the store")
    now = _clock()
    _, times = _validate_plan_fields(row, now)
    row = {**row, "created_at": now.isoformat(), "frozen_at": now.isoformat()}
    row["artifact_hash"] = _hash(row)
    payload = _serialized(row)
    digest = _hash(row)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload,payload_hash FROM parlay_validation_plan WHERE plan_id=?",
                              (row["plan_id"],)).fetchone()
        if existing:
            raise store.EvidenceConflict("plan_id is already frozen; create a new version")
        parent = row.get("supersedes_plan_id")
        if parent is None:
            if row["version"] != 1 or db.execute(
                    "SELECT 1 FROM parlay_validation_plan WHERE product_type=?",
                    (row["product_type"],)).fetchone():
                raise ValueError("new product plan must start at version 1")
        else:
            previous = db.execute("SELECT product_type,version FROM parlay_validation_plan WHERE plan_id=?",
                                  (parent,)).fetchone()
            if previous is None or previous != (row["product_type"], row["version"] - 1):
                raise ValueError("new plan must version the prior plan for the same product")
        db.execute("INSERT INTO parlay_validation_plan VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            row["plan_id"], row["product_type"], row["version"], parent, row["frozen_at"],
            times["training"].isoformat(), times["validation_start"].isoformat(),
            times["validation_end"].isoformat(), times["holdout_start"].isoformat(),
            times["holdout_end"].isoformat(), row["artifact_hash"], payload, digest))
    return row


def load_plan(path: str | Path | None, plan_id: str) -> dict | None:
    with closing(connect(path)) as db:
        found = db.execute("SELECT payload,payload_hash FROM parlay_validation_plan WHERE plan_id=?",
                           (plan_id,)).fetchone()
    return _verified(*found) if found else None


def _event_key(leg: dict) -> str:
    sport = _text(leg.get("sport"), "leg.sport")
    namespace = _text(leg.get("provider_namespace"), "leg.provider_namespace")
    event = _text(leg.get("provider_event_id"), "leg.provider_event_id")
    return f"{sport}|{namespace}|{event}"


def _cohort_for(plan: dict, starts: list[datetime]) -> str:
    for name in COHORTS:
        window = plan["windows"][name]
        start = _time(window["start"], f"{name}.start")
        end = _time(window["end"], f"{name}.end")
        if all(start <= item < end for item in starts):
            return name
    raise ValueError("ticket events span or fall outside frozen cohorts")


def _quote_verified(decision: dict, legs: list[dict], quote: dict | None,
                    now: datetime) -> bool:
    """Recheck adapter-bound exact evidence; a saved arbitrary price is insufficient."""
    if not isinstance(quote, dict):
        return False
    try:
        from app_core.parlay_ticket_quotes import bind_ticket_request
        from core.wager_decisions import decimal_price

        if decision["product_type"] == "CROSS_GAME_PARLAY" and decision.get("sgp_component_count"):
            expected = _mapping(decision.get("ticket_binding"), "ticket_binding")
        else:
            components = decision.get("components") or legs
            expected = bind_ticket_request({
                "product_type": decision["product_type"], "sportsbook": decision["sportsbook"],
                "components": components})
        if expected["ticket_hash"] != decision.get("ticket_hash"):
            return False
        if expected["product_type"] != decision["product_type"] or expected["sportsbook"] != decision["sportsbook"]:
            return False
        if sorted(leg.get("leg_hash") for leg in legs) != expected["leg_hashes"]:
            return False
        if any(quote.get(key) != expected[key] for key in
               ("ticket_hash", "leg_hashes", "selection_bindings", "component_hashes", "sgp_components")):
            return False
        if quote.get("verification_state") != "VERIFIED" or quote.get("sportsbook") != expected["sportsbook"]:
            return False
        for key in ("quote_id", "provider", "provider_ticket_id", "settlement_rules_id"):
            _text(quote.get(key), key)
        if re.fullmatch(r"[0-9a-f]{64}", str(quote.get("raw_evidence_hash"))) is None:
            return False
        if quote.get("source") == "SPORTSBOOK":
            if quote.get("source_type") != "PROVIDER_API" or not quote.get("provider_response_id"):
                return False
        elif quote.get("source") == "OWNER_CONFIRMED":
            if quote.get("source_type") not in {"SPORTSBOOK_DISPLAY", "SPORTSBOOK_SCREENSHOT"} or not all(
                    quote.get(key) for key in
                    ("owner_id", "owner_authorization_id", "owner_confirmed_at", "artifact_reference")):
                return False
            if not _time(quote["quoted_at"], "quoted_at") <= _time(
                    quote["owner_confirmed_at"], "owner_confirmed_at") <= now:
                return False
        else:
            return False
        decimal = _finite(quote.get("decimal_odds"), "decimal_odds", low=1.000001)
        american = _finite(quote.get("american_odds"), "american_odds")
        converted = decimal_price(american)
        if converted is None or abs(converted-decimal) > 0.002:
            return False
        quoted_at = _time(quote["quoted_at"], "quoted_at")
        expires_at = _time(quote["expires_at"], "expires_at")
        decided_at = _time(decision["decision_at"], "decision_at")
        return (quoted_at <= decided_at <= now < expires_at and
                (now-quoted_at).total_seconds() <= 1800 and
                (expires_at-quoted_at).total_seconds() <= 1800)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def freeze_candidate(path: str | Path | None, plan_id: str, decision_id: str) -> dict:
    """Capture one saved exact decision before every underlying event starts."""
    with closing(connect(path)) as db:
        previous = db.execute("SELECT payload,payload_hash FROM parlay_validation_candidate WHERE decision_id=?",
                              (decision_id,)).fetchone()
    if previous:
        prior = _verified(*previous)
        if prior["plan_id"] != plan_id:
            raise store.EvidenceConflict("candidate already belongs to a frozen plan")
        return prior
    plan = load_plan(path, plan_id)
    snapshot = store.load_decision(path, decision_id)
    if plan is None or snapshot is None:
        raise ValueError("known plan and saved parlay decision are required")
    decision, legs, quote = snapshot["decision"], snapshot["legs"], snapshot["quote"]
    if decision.get("product_type") != plan["product_type"] or not legs:
        raise ValueError("product-scoped plan/decision mismatch")
    now = _clock()
    frozen_at = _time(plan["frozen_at"], "plan.frozen_at")
    if now < frozen_at:
        raise ValueError("plan freeze is in the future")
    starts = [_time(leg.get("game_start_at") or leg.get("start"), "leg.game_start_at") for leg in legs]
    if any(start <= now for start in starts):
        raise ValueError("prospective evidence must predate every event")
    cohort = _cohort_for(plan, starts)
    event_keys = sorted({_event_key(leg) for leg in legs})
    blockers = set(decision.get("blockers") or [])
    fatal = set()
    decided_at = _time(decision.get("decision_at"), "decision_at")
    joint_generated = _optional_time(decision.get("joint_generated_at"), "joint_generated_at")
    if not (joint_generated is not None and frozen_at <= joint_generated <= decided_at <= now < min(starts)):
        fatal.add("DECISION_CHRONOLOGY_INVALID")
    joint_evidence_frozen = _optional_time(decision.get("joint_evidence_frozen_at"),
                                           "joint_evidence_frozen_at")
    if joint_evidence_frozen is None or joint_generated is None or joint_evidence_frozen > joint_generated:
        fatal.add("JOINT_EVIDENCE_CHRONOLOGY_INVALID")
    scope = plan["model_scope"]
    if (decision.get("joint_model_id") or decision.get("model_id")) != scope["joint_model_id"] or (
            decision.get("joint_model_version") or decision.get("model_version")) != scope["joint_model_version"] or (
            decision.get("calibration_version") != scope["calibration_version"] or
             decision.get("probability_method") != scope["method"]):
        fatal.add("MODEL_SCOPE_MISMATCH")
    if decision.get("policy_version") != plan["product_policy_version"]:
        fatal.add("PRODUCT_POLICY_VERSION_MISMATCH")
    allowed_scopes = {(item["sport"], item["market"]) for item in plan["sport_market_scope"]}
    if any((leg.get("sport"), leg.get("market_type") or leg.get("market")) not in allowed_scopes
           for leg in legs):
        fatal.add("SPORT_MARKET_SCOPE_MISMATCH")
    trained = _optional_time(decision.get("joint_model_trained_through"), "joint_model_trained_through")
    joint_times = [
        _optional_time(decision.get(key), key) for key in
        ("joint_model_available_at", "joint_calibration_available_at", "joint_generated_at")
    ]
    if trained is None or trained > _time(plan["training_cutoff"], "training_cutoff") or (
            any(value is None or value > now for value in joint_times) or
            (joint_times[2] is not None and any(value is not None and value > joint_times[2]
                                                   for value in joint_times[:2]))):
        fatal.add("JOINT_MODEL_CHRONOLOGY_INVALID")
    for leg in legs:
        available = _optional_time(leg.get("model_available_at"), "leg.model_available_at")
        calibrated = _optional_time(leg.get("calibration_available_at"), "leg.calibration_available_at")
        trained_leg = _optional_time(leg.get("model_trained_through"), "leg.model_trained_through")
        frozen_leg = _optional_time(leg.get("evidence_frozen_at"), "leg.evidence_frozen_at")
        if any(value is None or value > now for value in (available, calibrated, trained_leg)) or (
                joint_generated is not None and any(value is not None and value > joint_generated
                                                    for value in (available,calibrated,frozen_leg))) or (
                frozen_leg is None) or (
                trained_leg is not None and trained_leg > _time(plan["training_cutoff"], "training_cutoff")):
            fatal.add("LEG_MODEL_CHRONOLOGY_INVALID")
    try:
        mean = _finite(decision.get("probability_mean"), "probability_mean", low=0, high=1)
        conservative = _finite(decision.get("probability_conservative"),
                               "probability_conservative", low=0, high=1)
    except ValueError:
        mean = conservative = None
        fatal.add("JOINT_PROBABILITY_MISSING")
    try:
        push = _finite(decision.get("probability_push"), "probability_push", low=0, high=1)
        partial = _finite(decision.get("probability_partial", 0), "probability_partial", low=0, high=1)
    except ValueError:
        push = partial = None
        fatal.add("JOINT_PROBABILITY_MISSING")
    if (mean is not None and conservative is not None and conservative > mean) or (
            mean is not None and push is not None and partial is not None and mean+push+partial > 1+1e-9):
        fatal.add("JOINT_PROBABILITY_INCOHERENT")
    method = str(decision.get("probability_method") or "").upper()
    dependence = decision.get("dependence_status")
    if plan["product_type"] == "STANDARD_PARLAY":
        if "INDEPEND" in method and dependence != "INDEPENDENT_VERIFIED":
            fatal.add("STANDARD_INDEPENDENCE_UNVERIFIED")
        elif "INDEPEND" not in method and (
                dependence != "LOW_DEPENDENCE" or
                decision.get("joint_dependence_methodology_id") != scope.get("dependence_methodology_id")):
            fatal.add("STANDARD_DEPENDENCE_METHOD_UNVERIFIED")
    if plan["product_type"] == "SAME_GAME_PARLAY":
        if "INDEPEND" in method:
            fatal.add("SGP_INDEPENDENT_MULTIPLICATION_PROHIBITED")
        if decision.get("joint_correlation_method") != scope["correlation_method"]:
            fatal.add("SGP_CORRELATION_METHOD_UNVERIFIED")
    if plan["product_type"] == "CROSS_GAME_PARLAY":
        if any(decision.get(source) != scope[target] for source, target in (
                ("joint_component_dependence_method", "component_dependence_method"),
                ("joint_shared_factor_method", "shared_factor_method"),
                ("joint_final_calibration_id", "final_calibration_id"),
                ("joint_final_calibration_version", "final_calibration_version"))):
            fatal.add("CROSS_DEPENDENCE_OR_CALIBRATION_UNVERIFIED")
        sgp_count = decision.get("sgp_component_count")
        sgp_ids = decision.get("sgp_component_validation_ids")
        if sgp_count and (
                not decision.get("joint_component_hashes") or
                not isinstance(sgp_ids, list) or len(sgp_ids) != sgp_count or
                any(not isinstance(item, str) or not item.strip() for item in sgp_ids) or
                "SGP_COMPONENT_UNVALIDATED" in blockers):
            fatal.add("SGP_COMPONENT_INTEGRITY_UNVERIFIED")
        if sgp_count and "SGP_COMPONENT_INTEGRITY_UNVERIFIED" not in fatal:
            sgp_readiness = product_readiness(path, "SAME_GAME_PARLAY")
            if sgp_readiness["validation_id"] is None or any(
                    item != sgp_readiness["validation_id"] for item in sgp_ids):
                fatal.add("SGP_VALIDATION_ARTIFACT_MISSING")
    fatal.update(blockers & {
        "JOINT_INPUT_MISMATCH", "JOINT_CHRONOLOGY_INVALID", "JOINT_PROBABILITY_INVALID",
        "JOINT_INDEPENDENCE_MISMATCH", "JOINT_MASS_INCONSISTENT_WITH_LEGS",
        "MUTUALLY_EXCLUSIVE_LEGS", "DUPLICATE_LEG", "LEG_COUNT_INVALID",
        "DUPLICATE_GAME", "NOT_SAME_GAME", "COMPONENT_GAME_CONFLICT",
        "SGP_COMPONENT_HASH_MISMATCH", "SGP_COMPONENT_UNVALIDATED",
        "SGP_COMPONENT_SHAPE_INVALID", "SGP_COMPONENT_NOT_ALLOWED",
        "SGP_INDEPENDENCE_FORBIDDEN", "SGP_CORRELATION_METHOD_UNAVAILABLE",
        "LEG_CHRONOLOGY_INVALID", "LEG_PROBABILITY_INVALID", "LEG_CRITICAL_INPUT",
        "LEG_EVIDENCE_MISSING", "LEG_IDENTITY_MISSING", "LEG_IDENTITY_UNVERIFIED",
        "LEG_PROVIDER_EVENT_MISSING", "LEG_SELECTION_LINE_MISMATCH",
        "LEG_STARTED_OR_TIME_UNKNOWN", "LEG_TEAM_IDENTITY_MISSING",
        "LEG_MATERIAL_NEWS", "LEG_ANALYSIS_STALE", "BOOK_MISMATCH",
        "DEPENDENCE_UNKNOWN", "DEPENDENCE_JOINT_MODEL_REQUIRED",
        "CROSS_GAME_DEPENDENCE_UNAVAILABLE", "INDEPENDENCE_NOT_VERIFIED",
        "COMPONENTS_INVALID", "LEGS_MISSING", "HALF_POINT_PUSH_IMPOSSIBLE",
    })
    quote_id = quote.get("quote_id") if quote else None
    verified_price = (_quote_verified(decision, legs, quote, now) and
                      _has_quote_source(path, decision_id, quote_id, now)) if quote_id else False
    if not verified_price:
        blockers.add("PRICE_UNAVAILABLE")
    elif plan["product_type"] == "CROSS_GAME_PARLAY" and (
            quote.get("component_hashes") != decision.get("joint_component_hashes") or
            (decision.get("sgp_component_count", 0) and not quote.get("sgp_components"))):
        blockers.add("PRICE_UNAVAILABLE")
        blockers.add("CROSS_COMPONENT_PRICE_MISMATCH")
        verified_price = False
    conservative_ev = break_even = conservative_edge = None
    if verified_price and conservative is not None and push is not None:
        from core.true_parlay_engine import _partial_return
        joint_settlement = {"probability_push": push,
                            "partial_outcomes": decision.get("joint_partial_outcomes") or [],
                            "settlement_rules_id": decision.get("settlement_rules_id")}
        other_return, settlement_blockers = _partial_return(joint_settlement, quote, legs)
        if settlement_blockers or other_return is None:
            fatal.add("SETTLEMENT_MODEL_UNAVAILABLE")
        else:
            decimal = _finite(quote.get("decimal_odds"), "decimal_odds", low=1.000001)
            conservative_ev = conservative*decimal+other_return-1
            break_even = (1-other_return)/decimal
            conservative_edge = conservative-break_even
            for key, actual in (("conservative_ev", conservative_ev),
                                ("break_even_probability", break_even),
                                ("conservative_edge", conservative_edge)):
                try:
                    supplied = _finite(decision.get(key), key)
                except ValueError:
                    supplied = None
                if supplied is None or abs(supplied-actual) > 1e-6:
                    fatal.add("VALUE_FACT_MISMATCH")
    blockers.update(fatal)
    candidate = {
        "plan_id": plan_id, "decision_id": decision_id, "parlay_id": decision["parlay_id"],
        "product_type": plan["product_type"], "cohort": cohort,
        "frozen_at": now.isoformat(), "first_event_start": min(starts).isoformat(),
        "last_event_start": max(starts).isoformat(),
        "event_keys": event_keys, "ticket_hash": decision.get("ticket_hash"),
        "leg_hashes": [leg.get("leg_hash") for leg in legs],
        "joint_model_id": decision.get("joint_model_id"),
        "joint_model_version": decision.get("joint_model_version"),
        "calibration_version": decision.get("calibration_version"),
        "joint_generated_at": decision.get("joint_generated_at"),
        "joint_model_available_at": decision.get("joint_model_available_at"),
        "joint_calibration_available_at": decision.get("joint_calibration_available_at"),
        "joint_correlation_method": decision.get("joint_correlation_method"),
        "joint_component_dependence_method": decision.get("joint_component_dependence_method"),
        "joint_shared_factor_method": decision.get("joint_shared_factor_method"),
        "joint_final_calibration_id": decision.get("joint_final_calibration_id"),
        "joint_final_calibration_version": decision.get("joint_final_calibration_version"),
        "sgp_component_validation_ids": decision.get("sgp_component_validation_ids"),
        "policy_version": decision.get("policy_version"),
        "probability_mean": mean, "probability_conservative": conservative,
        "probability_push": push,
        "probability_partial": partial,
        "quote_id": quote_id, "quote_source": quote.get("source") if quote else None,
        "quoted_at": quote.get("quoted_at") if quote else None,
        "break_even_probability": break_even,
        "conservative_ev": conservative_ev,
        "conservative_edge": conservative_edge,
        "dependence_status": dependence,
        "deployment_state_at_prediction": decision.get("validation_state") or "UNVALIDATED",
        "probability_admissible": not fatal,
        "blockers": sorted(blockers),
    }
    if not isinstance(candidate["ticket_hash"], str) or not candidate["ticket_hash"]:
        raise ValueError("saved ticket hash is required")
    payload, digest = _serialized(candidate), _hash(candidate)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload,payload_hash FROM parlay_validation_candidate WHERE decision_id=?",
                              (decision_id,)).fetchone()
        if existing:
            prior = _verified(*existing)
            if prior["plan_id"] != plan_id:
                raise store.EvidenceConflict("candidate already belongs to a frozen plan")
            return prior
        db.execute("INSERT INTO parlay_validation_candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            decision_id, plan_id, candidate["parlay_id"], candidate["product_type"], cohort,
            candidate["frozen_at"], candidate["first_event_start"], candidate["last_event_start"],
            candidate["ticket_hash"],
            quote_id, mean, conservative, int(candidate["probability_admissible"]),
            _serialized(event_keys), payload, digest))
    return candidate


def append_outcome(path: str | Path | None, decision_id: str, grading_version: int,
                   *, observed_at: str, available_at: str, source_id: str) -> bool:
    """Attach source availability and our observation to a grading revision."""
    _integer(grading_version, "grading_version", minimum=1)
    _text(source_id, "source_id")
    now = _clock()
    observed, available = _time(observed_at, "observed_at"), _time(available_at, "available_at")
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        candidate_row = db.execute(
            "SELECT payload,payload_hash FROM parlay_validation_candidate WHERE decision_id=?",
            (decision_id,)).fetchone()
        result_row = db.execute(
            "SELECT payload,payload_hash FROM parlay_result WHERE decision_id=? AND grading_version=?",
            (decision_id, grading_version)).fetchone()
        if candidate_row is None or result_row is None:
            raise ValueError("candidate and saved grading revision are required")
        candidate, result = _verified(*candidate_row), _verified(*result_row)
        if not _time(candidate["last_event_start"], "last_event_start") <= available <= observed <= now:
            raise ValueError("outcome chronology invalid or not yet available")
        graded = _time(result.get("graded_at"), "graded_at")
        if not observed <= graded <= now:
            raise ValueError("grading predates observation or lies in the future")
        if result.get("source_id") != source_id:
            raise ValueError("result source mismatch")
        source_row = db.execute("SELECT source_id,available_at FROM parlay_validation_result_source "
                                "WHERE decision_id=? AND grading_version=?",
                                (decision_id,grading_version)).fetchone()
        if (source_row != (source_id,available.isoformat()) or not
                _source_matches(db,"parlay_validation_result_source",decision_id,grading_version)):
            raise ValueError("replayable result source and availability are required")
        outcome = result.get("outcome")
        if outcome not in OUTCOMES:
            raise ValueError("unsupported ticket outcome")
        if result.get("settlement_rule") is None:
            raise ValueError("settlement rule is required")
        quote_row = db.execute("SELECT payload,payload_hash FROM parlay_quote WHERE decision_id=?",
                               (decision_id,)).fetchone()
        quote = _verified(*quote_row) if quote_row else None
        if result.get("selection_return") is not None:
            if quote is None:
                raise ValueError("selection return requires captured ticket price")
            selection_return = _finite(result["selection_return"],"selection_return")
            decimal = _finite(quote.get("decimal_odds"),"decimal_odds",low=1.000001)
            expected = {"WIN":decimal-1,"LOSS":-1.0,"PUSH":0.0,"VOID":0.0}.get(outcome)
            if expected is None or abs(selection_return-expected) > 1e-6:
                raise ValueError("selection return conflicts with ticket outcome")
        if result.get("result_kind") == "ACCEPTED_WAGER":
            if not all(result.get(name) is not None for name in
                       ("accepted_wager_id", "accepted_decimal_odds", "accepted_stake", "placed_at", "net_return")):
                raise ValueError("accepted return requires actual accepted-wager receipt")
            if _time(result["placed_at"], "placed_at") >= _time(candidate["first_event_start"], "first_event_start"):
                raise ValueError("accepted wager must predate the event")
            if not _has_wager_source(db, decision_id, grading_version):
                raise ValueError("accepted wager source response is required")
            stake = _finite(result["accepted_stake"],"accepted_stake",low=0.000001)
            decimal = _finite(result["accepted_decimal_odds"],"accepted_decimal_odds",low=1.000001)
            actual_return = _finite(result["net_return"],"net_return")
            expected = {"WIN":stake*(decimal-1),"LOSS":-stake,"PUSH":0.0,"VOID":0.0}.get(outcome)
            if expected is None or abs(actual_return-expected) > 1e-6:
                raise ValueError("accepted return conflicts with ticket outcome or price")
        elif result.get("net_return") is not None:
            raise ValueError("selection result cannot claim accepted-wager return")
        if grading_version > 1:
            earlier = db.execute("SELECT payload,payload_hash FROM parlay_result "
                                 "WHERE decision_id=? AND grading_version<? ORDER BY grading_version",
                                 (decision_id,grading_version)).fetchall()
            accepted_receipts = {_verified(*entry).get("accepted_wager_id") for entry in earlier
                                 if _verified(*entry).get("result_kind") == "ACCEPTED_WAGER"}
            if accepted_receipts and (result.get("result_kind") != "ACCEPTED_WAGER" or
                                      result.get("accepted_wager_id") not in accepted_receipts):
                raise ValueError("accepted wager cannot be replaced by selection grading")
        closing_at = _optional_time(result.get("closing_quoted_at"), "closing_quoted_at")
        comparable = (
            result.get("closing_quote_comparable") is True and quote is not None and
            result.get("closing_ticket_hash") == candidate["ticket_hash"] and
            result.get("closing_sportsbook") == quote.get("sportsbook") and
            result.get("closing_settlement_rules_id") == quote.get("settlement_rules_id") and
            result.get("closing_source_id") and closing_at is not None and
            _time(quote["quoted_at"], "quote.quoted_at") <= closing_at <
            _time(candidate["first_event_start"], "first_event_start") and
            result.get("closing_decimal_odds") is not None and
            _source_matches(db,"parlay_validation_closing_source",decision_id,grading_version))
        evidence = {
            "decision_id": decision_id, "grading_version": grading_version,
            "source_id": source_id, "observed_at": observed.isoformat(),
            "available_at": available.isoformat(), "outcome": outcome,
            "settlement_rule": result.get("settlement_rule"),
            "leg_outcomes": result.get("leg_outcomes"),
            "selection_return": result.get("selection_return"),
            "accepted_wager_return": result.get("net_return") if result.get("result_kind") == "ACCEPTED_WAGER" else None,
            "accepted_wager_id": result.get("accepted_wager_id"),
            "accepted_stake": result.get("accepted_stake"),
            "closing_decimal_odds": result.get("closing_decimal_odds"),
            "closing_quote_comparable": bool(comparable),
            "closing_source_id": result.get("closing_source_id"),
        }
        payload, digest = _serialized(evidence), _hash(evidence)
        existing = db.execute("SELECT payload_hash FROM parlay_validation_outcome "
                              "WHERE decision_id=? AND grading_version=?",
                              (decision_id, grading_version)).fetchone()
        if existing:
            if existing[0] != digest:
                raise store.EvidenceConflict("grading availability already has different facts")
            return False
        db.execute("INSERT INTO parlay_validation_outcome VALUES (?,?,?,?,?,?,?,?)", (
            decision_id, grading_version, source_id, observed.isoformat(), available.isoformat(),
            outcome, payload, digest))
    return True


def _components(rows: list[dict]) -> list[list[int]]:
    """Conservatively cluster every ticket sharing an underlying event."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in [f"ticket:{row['ticket_hash']}", *(f"event:{item}" for item in row["event_keys"])]:
            if key in first:
                parents[find(index)] = find(first[key])
            else:
                first[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _binary_metrics(rows: list[dict], *, denominator_count: int | None = None) -> dict:
    resolved = [row for row in rows if row["outcome"] in {"WIN", "LOSS"}]
    usable = [row for row in resolved if row["probability_admissible"] and
              row["probability_mean"] is not None]
    denominator = len(rows) if denominator_count is None else denominator_count
    if not usable:
        return {"brier": None, "log_loss": None, "calibration_error": None,
                "calibration_curve": [], "coverage": 0.0 if denominator else None,
                "mean_predicted_probability": None, "observed_frequency": None,
                "resolved_raw": len(resolved), "scored_raw": 0, "scored_independent_units": 0}
    groups = _components(usable)
    weighted: list[tuple[float, float, float]] = []
    for indices in groups:
        weight = 1/len(indices)
        weighted.extend((usable[index]["probability_mean"],
                         float(usable[index]["outcome"] == "WIN"),weight) for index in indices)
    brier = sum(weight*(p-y)**2 for p,y,weight in weighted)/len(groups)
    log_loss = -sum(weight*(y*math.log(max(p,1e-15))+
                            (1-y)*math.log(max(1-p,1e-15)))
                    for p,y,weight in weighted)/len(groups)
    buckets: dict[int,list[tuple[float,float,float]]] = {}
    for p,y,weight in weighted:
        buckets.setdefault(min(9,int(p*10)),[]).append((p,y,weight))
    curve = []
    for key,points in sorted(buckets.items()):
        weight = sum(item[2] for item in points)
        curve.append({"bin":key,"count":len(points),"effective_weight":weight,
                      "mean_probability":sum(p*w for p,_,w in points)/weight,
                      "observed_frequency":sum(y*w for _,y,w in points)/weight})
    ece = sum(item["effective_weight"]*abs(item["mean_probability"]-item["observed_frequency"])
              for item in curve)/len(groups)
    return {"brier": brier, "log_loss": log_loss, "calibration_error": ece,
            "calibration_curve": curve, "coverage": len(usable)/denominator,
            "mean_predicted_probability": sum(p*w for p,_,w in weighted)/len(groups),
            "observed_frequency": sum(y*w for _,y,w in weighted)/len(groups),
            "resolved_raw": len(resolved), "scored_raw": len(usable),
            "scored_independent_units": len(groups)}


def _cohort_report(rows: list[dict]) -> dict:
    status = Counter(row["outcome"] for row in rows)
    for outcome in OUTCOMES:
        status.setdefault(outcome, 0)
    priced = [row for row in rows if row["quote_verified"]]
    accepted = [row for row in rows if row["accepted_wager_return"] is not None]
    provider_rows = [row for row in rows if row["quote_verified"] and row["quote_source"] == "SPORTSBOOK"]
    provider_accepted = [row for row in provider_rows if row["accepted_wager_return"] is not None]
    accepted_stake = sum(row["accepted_stake"] for row in accepted)
    provider_stake = sum(row["accepted_stake"] for row in provider_accepted)
    clv = [math.log(row["quote_decimal_odds"]/row["closing_decimal_odds"])
           for row in rows if row["closing_quote_comparable"] and row["closing_source_id"] and
           row["quote_decimal_odds"] and row["closing_decimal_odds"] and row["closing_decimal_odds"] > 1]
    predictions = [row["conservative_ev"] for row in rows if row["conservative_ev"] is not None]
    realized = [row["selection_return"] for row in rows if row["selection_return"] is not None]
    return {
        "raw_candidates": len(rows), "unique_tickets": len({row["ticket_hash"] for row in rows}),
        "independent_units": len(_components(rows)), "outcomes": dict(sorted(status.items())),
        "probability": _binary_metrics(rows),
        "provider_probability": _binary_metrics(provider_rows,denominator_count=len(rows)),
        "matched_provider_price_probability_units": len(_components([
            row for row in rows if row["outcome"] in {"WIN","LOSS"} and
            row["probability_admissible"] and row["quote_verified"] and
            row["quote_source"] == "SPORTSBOOK"])),
        "value": {
            "average_predicted_ev": sum(predictions)/len(predictions) if predictions else None,
            "average_realized_selection_return": sum(realized)/len(realized) if realized else None,
            "predicted_ev_vs_realized_return": {
                "predicted_ev": sum(predictions)/len(predictions) if predictions else None,
                "realized_return": sum(realized)/len(realized) if realized else None},
            "accepted_wagers": len(accepted), "accepted_stake": accepted_stake,
            "accepted_net_return": sum(row["accepted_wager_return"] for row in accepted),
            "roi": sum(row["accepted_wager_return"] for row in accepted)/accepted_stake if accepted_stake else None,
            "provider_accepted_wagers":len(provider_accepted),
            "provider_roi":(sum(row["accepted_wager_return"] for row in provider_accepted)/provider_stake
                            if provider_stake else None),
            "price_available": len(priced), "price_rejected": len(rows)-len(priced),
            "price_availability_rate": len(priced)/len(rows) if rows else None,
            "clv_comparable_count": len(clv), "mean_log_clv": sum(clv)/len(clv) if clv else None,
        },
    }


def evaluate_plan(path: str | Path | None, plan_id: str) -> dict:
    """Report every frozen cohort member, including pending and void tickets."""
    plan = load_plan(path, plan_id)
    if plan is None:
        raise ValueError("unknown validation plan")
    with closing(connect(path)) as db:
        frozen_rows = db.execute(
            "SELECT payload,payload_hash FROM parlay_validation_candidate WHERE plan_id=? "
            "ORDER BY frozen_at,decision_id", (plan_id,)).fetchall()
        candidates = [_verified(*entry) for entry in frozen_rows]
        latest = {}
        for candidate in candidates:
            found = db.execute(
                "SELECT payload,payload_hash FROM parlay_validation_outcome WHERE decision_id=? "
                "ORDER BY grading_version DESC LIMIT 1", (candidate["decision_id"],)).fetchone()
            latest[candidate["decision_id"]] = _verified(*found) if found else None
        quote_odds = {}
        for candidate in candidates:
            quote = db.execute("SELECT payload,payload_hash FROM parlay_quote WHERE decision_id=? "
                               "AND quote_id=?", (candidate["decision_id"],candidate["quote_id"])).fetchone()
            if quote:
                quote_data = _verified(*quote)
                quote_odds[candidate["decision_id"]] = quote_data.get("odds_decimal", quote_data.get("decimal_odds"))
            else:
                quote_odds[candidate["decision_id"]] = None
    rows = []
    for candidate in candidates:
        outcome = latest[candidate["decision_id"]] or {}
        rows.append({**candidate,
                     "outcome": outcome.get("outcome", "PENDING"),
                     "selection_return": outcome.get("selection_return"),
                     "accepted_wager_return": outcome.get("accepted_wager_return"),
                     "accepted_stake": outcome.get("accepted_stake"),
                     "closing_decimal_odds": outcome.get("closing_decimal_odds"),
                     "closing_quote_comparable": outcome.get("closing_quote_comparable") is True,
                     "closing_source_id": outcome.get("closing_source_id"),
                     "quote_decimal_odds": quote_odds[candidate["decision_id"]],
                     "quote_verified": "PRICE_UNAVAILABLE" not in candidate["blockers"] and
                                       candidate.get("quote_id") is not None})
    cohorts = {cohort: _cohort_report([row for row in rows if row["cohort"] == cohort])
               for cohort in COHORTS}
    blockers = []
    if _clock() < _time(plan["windows"]["holdout"]["end"], "holdout.end"):
        blockers.append("HOLDOUT_WINDOW_OPEN")
    for cohort in COHORTS:
        result = cohorts[cohort]
        threshold = plan["minimum_independent_units"][cohort]
        if result["provider_probability"]["scored_independent_units"] < threshold:
            blockers.append(f"{cohort.upper()}_INDEPENDENT_UNITS_INSUFFICIENT")
        p = result["provider_probability"]
        limits = plan["probability_thresholds"]
        for metric, bound in (("brier", "max_brier"), ("log_loss", "max_log_loss"),
                              ("calibration_error", "max_calibration_error")):
            if p[metric] is None or p[metric] > limits[bound]:
                blockers.append(f"{cohort.upper()}_{metric.upper()}_FAIL")
        if p["coverage"] is None or p["coverage"] < limits["min_coverage"]:
            blockers.append(f"{cohort.upper()}_COVERAGE_FAIL")
        if result["value"]["price_available"] < threshold:
            blockers.append(f"{cohort.upper()}_TICKET_PRICE_EVIDENCE_INSUFFICIENT")
        if result["matched_provider_price_probability_units"] < threshold:
            blockers.append(f"{cohort.upper()}_MATCHED_PROVIDER_EVIDENCE_INSUFFICIENT")
    holdout_value = cohorts["holdout"]["value"]
    betting = plan["betting_thresholds"]
    if holdout_value["provider_accepted_wagers"] < betting["min_accepted_wagers"]:
        blockers.append("ACCEPTED_WAGER_EVIDENCE_INSUFFICIENT")
    if betting.get("min_roi") is not None and (holdout_value["provider_roi"] is None or
                                               holdout_value["provider_roi"] < betting["min_roi"]):
        blockers.append("HOLDOUT_ROI_FAIL")
    if plan["clv_policy"]["required"] and holdout_value["clv_comparable_count"] == 0:
        blockers.append("COMPARABLE_CLV_EVIDENCE_MISSING")
    if any("SGP_COMPONENT_INTEGRITY_UNVERIFIED" in row["blockers"] for row in rows):
        blockers.append("SGP_COMPONENT_INTEGRITY_UNVERIFIED")
    if plan["product_type"] == "CROSS_GAME_PARLAY":
        sgp_ready = product_readiness(path, "SAME_GAME_PARLAY")
        if any(row.get("sgp_component_validation_ids") and (
                sgp_ready["validation_id"] is None or any(
                    item != sgp_ready["validation_id"]
                    for item in row["sgp_component_validation_ids"])) for row in rows):
            blockers.append("SGP_VALIDATION_ARTIFACT_MISSING")
    report = {"plan_id": plan_id, "plan_artifact_hash": plan["artifact_hash"],
              "product_type": plan["product_type"], "independence_method": INDEPENDENCE_METHOD,
              "cohorts": cohorts, "raw_candidates": len(rows),
              "unique_tickets": len({row["ticket_hash"] for row in rows}),
              "effective_independent_units": len(_components(rows)),
              "candidate_decision_ids": [row["decision_id"] for row in rows],
              "outcome_revision_fingerprint": _hash([(row["decision_id"],
                  latest[row["decision_id"]]) for row in rows]),
              "blockers": sorted(set(blockers)),
              "status": "UNVALIDATED" if blockers else "VALIDATION_PASSED",
              "activation_status": "NOT_READY" if blockers else "ACTIVATION_PENDING",
              "stake_authority": False, "recommended_stake": 0.0}
    report["report_hash"] = _hash(report)
    return report


def freeze_report(path: str | Path | None, plan_id: str) -> dict:
    """Persist a point-in-time report; never change product or stake authority."""
    report = evaluate_plan(path, plan_id)
    artifact = {"artifact_id": report["report_hash"], "created_at": _clock().isoformat(),
                "report": report}
    payload, digest = _serialized(artifact), _hash(artifact)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload,payload_hash FROM parlay_validation_artifact WHERE artifact_id=?",
                              (artifact["artifact_id"],)).fetchone()
        if existing:
            return _verified(*existing)
        db.execute("INSERT INTO parlay_validation_artifact VALUES (?,?,?,?,?,?,?)", (
            artifact["artifact_id"], plan_id, artifact["created_at"], report["report_hash"],
            report["status"], payload, digest))
    return artifact


def record_deployment_review(path: str | Path | None, *, artifact_id: str,
                             validation_id: str, reviewer_id: str,
                             validation_state: str) -> dict:
    """Explicitly review a passed artifact; no owner activation is created."""
    if validation_state not in {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED"}:
        raise ValueError("invalid validated deployment state")
    _text(validation_id, "validation_id")
    _text(reviewer_id, "reviewer_id")
    with closing(connect(path)) as db:
        stored = db.execute("SELECT plan_id,status,payload,payload_hash FROM parlay_validation_artifact "
                            "WHERE artifact_id=?", (artifact_id,)).fetchone()
        if not stored or stored[1] != "VALIDATION_PASSED":
            raise ValueError("passed immutable validation artifact required")
        artifact = _verified(stored[2], stored[3])
        plan = db.execute("SELECT payload,payload_hash FROM parlay_validation_plan WHERE plan_id=?",
                          (stored[0],)).fetchone()
        plan = _verified(*plan)
        if plan["deployment_state_criteria"]["target_state"] != validation_state:
            raise ValueError("deployment state differs from frozen plan")
    if evaluate_plan(path, stored[0])["report_hash"] != artifact["report"]["report_hash"]:
        raise ValueError("validation artifact is stale after evidence revisions")
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        reviewed = _clock().isoformat()
        record = {"deployment_id": _hash([artifact_id, validation_id, reviewer_id, reviewed]),
                  "plan_id": stored[0], "artifact_id": artifact_id,
                  "product_type": plan["product_type"], "validation_id": validation_id,
                  "validation_state": validation_state, "reviewer_id": reviewer_id,
                  "reviewed_at": reviewed, "activation_status": "ACTIVATION_PENDING",
                  "stake_authority": False}
        payload, digest = _serialized(record), _hash(record)
        db.execute("INSERT INTO parlay_deployment_evidence VALUES (?,?,?,?,?,?,?,?,?,?)", (
            record["deployment_id"], record["plan_id"], record["artifact_id"],
            record["product_type"], validation_id, validation_state, reviewer_id,
            reviewed, payload, digest))
    return record


def product_readiness(path: str | Path | None, product_type: str) -> dict:
    """Validation status only; owner authorization, ledger, quote, and gates remain separate."""
    if product_type not in PRODUCTS:
        raise ValueError("invalid parlay product")
    with closing(connect(path)) as db:
        found = db.execute("SELECT payload,payload_hash FROM parlay_deployment_evidence "
                           "WHERE product_type=? ORDER BY reviewed_at DESC,deployment_id DESC LIMIT 1",
                           (product_type,)).fetchone()
        latest_plan = db.execute("SELECT plan_id FROM parlay_validation_plan WHERE product_type=? "
                                 "ORDER BY version DESC LIMIT 1", (product_type,)).fetchone()
    state = _verified(*found) if found else None
    if state is not None:
        try:
            artifact_current = (latest_plan is not None and latest_plan[0] == state["plan_id"] and
                                evaluate_plan(path, state["plan_id"])["report_hash"] == state["artifact_id"])
        except (ValueError, store.EvidenceConflict):
            artifact_current = False
        if not artifact_current:
            state = None
    plan = load_plan(path, state["plan_id"]) if state else None
    return {"product_type": product_type,
            "validation_state": state["validation_state"] if state else "UNVALIDATED",
            "validation_id": state["validation_id"] if state else None,
            "validation_artifact_id": state["artifact_id"] if state else None,
            "validation_plan_id": state["plan_id"] if state else None,
            "model_scope": plan["model_scope"] if plan else None,
            "product_policy_version": plan["product_policy_version"] if plan else None,
            "readiness": "VALIDATION_PASSED" if state else "UNVALIDATED",
            "activation_status": "ACTIVATION_PENDING" if state else "NOT_READY",
            "stake_authority": False, "production_eligible": False,
            "recommended_stake": 0.0,
            "activation_blockers": ["OWNER_AUTHORIZATION_REQUIRED", "BANKROLL_EXPOSURE_LEDGER_REQUIRED",
                                    "FRESH_EXACT_TICKET_QUOTE_REQUIRED", "DETERMINISTIC_GATES_REQUIRED",
                                    "PRODUCT_ALLOCATION_CAPS_REQUIRED"] if state else ["PRODUCT_UNVALIDATED"]}


def validation_policy_projection(path: str | Path | None, product_type: str,
                                 current_model_scope: Mapping | None) -> dict:
    """Expose reviewed validation to a policy builder without granting a stake.

    The caller must separately obtain owner authorization, a bankroll/exposure
    ledger, a fresh exact quote, deterministic gate results, and allocation caps.
    ``core.true_parlay_engine.evaluate_ticket`` remains the stake decision.
    """
    readiness = product_readiness(path, product_type)
    blockers = list(readiness["activation_blockers"])
    expected = readiness["model_scope"]
    current = dict(current_model_scope) if isinstance(current_model_scope, Mapping) else {}
    if expected is None:
        blockers.append("VALIDATION_ARTIFACT_MISSING")
    elif any(current.get(name) != expected.get(name) for name in expected):
        blockers.append("CURRENT_MODEL_OR_CALIBRATION_MISMATCH")
    current_validation = expected is not None and "CURRENT_MODEL_OR_CALIBRATION_MISMATCH" not in blockers
    return {
        "product_type": product_type,
        "validation_id": readiness["validation_id"] if current_validation else None,
        "validation_state": readiness["validation_state"] if current_validation else "UNVALIDATED",
        "validation_artifact_id": readiness["validation_artifact_id"],
        "product_policy_version": readiness["product_policy_version"],
        "readiness": readiness["readiness"] if current_validation else "UNVALIDATED",
        "activation_status": "ACTIVATION_PENDING" if current_validation else "NOT_READY",
        "activation_blockers": sorted(set(blockers)),
        "stake_authority": False, "production_eligible": False,
        "recommended_stake": 0.0,
    }
