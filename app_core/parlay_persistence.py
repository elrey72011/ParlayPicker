"""Append-only SQLite evidence for true-parlay decisions and grading.

This store records facts supplied by the decision engine. It does not grant
validation, authorization, or wager placement authority. Missing research facts
stay NULL in normalized columns and unchanged in the frozen JSON payload.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = {"STANDARD_PARLAY", "SAME_GAME_PARLAY", "CROSS_GAME_PARLAY"}
VALIDATION_STATES = {"UNVALIDATED", "PROVISIONAL_VALIDATED", "STANDARD_VALIDATED"}
RESULT_KINDS = {"SELECTION", "ACCEPTED_WAGER"}


class EvidenceConflict(ValueError):
    """An immutable identity was reused with different facts."""


def database_path() -> Path:
    directory = Path(os.environ.get("PARLAYPICKER_EVIDENCE_DIR", ROOT / "data/prediction_evidence"))
    return directory / "parlay-evidence.sqlite3"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _time(value: object, name: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if isinstance(value, datetime):
        parsed = value
        value = value.isoformat()
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{name} requires a timezone-aware timestamp") from None
    else:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    return str(value)


def _first(row: Mapping, *names: str):
    for name in names:
        if name in row:
            return row[name]
    return None


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Run the idempotent, isolated parlay schema migration on connection."""
    location = Path(path or database_path())
    location.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(location, timeout=30)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS parlay_identity (
            parlay_id TEXT PRIMARY KEY,
            product_type TEXT NOT NULL,
            sportsbook TEXT,
            leg_hash TEXT,
            ticket_hash TEXT,
            identity_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parlay_ticket (
            decision_id TEXT PRIMARY KEY,
            parlay_id TEXT NOT NULL REFERENCES parlay_identity(parlay_id),
            product_type TEXT NOT NULL CHECK (product_type IN
                ('STANDARD_PARLAY','SAME_GAME_PARLAY','CROSS_GAME_PARLAY')),
            status TEXT NOT NULL,
            sportsbook TEXT,
            quote_id TEXT,
            leg_hash TEXT,
            ticket_hash TEXT,
            policy_id TEXT,
            policy_version TEXT,
            validation_id TEXT,
            validation_state TEXT CHECK (validation_state IS NULL OR validation_state IN
                ('UNVALIDATED','PROVISIONAL_VALIDATED','STANDARD_VALIDATED')),
            deployment_state TEXT,
            model_id TEXT,
            model_version TEXT,
            calibration_id TEXT,
            calibration_version TEXT,
            probability_mean REAL,
            probability_conservative REAL,
            probability_break_even REAL,
            probability_method TEXT,
            conservative_ev REAL,
            conservative_edge REAL,
            recommended_stake_dollars REAL,
            recommended_stake_fraction REAL,
            production_eligible INTEGER CHECK (production_eligible IN (0,1)),
            decision_at TEXT NOT NULL,
            analysis_at TEXT,
            evidence_id TEXT,
            blockers TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            CHECK (recommended_stake_dollars IS NULL OR recommended_stake_dollars <= 0
                   OR production_eligible = 1),
            CHECK (recommended_stake_dollars IS NULL OR recommended_stake_dollars >= 0),
            CHECK (recommended_stake_fraction IS NULL OR recommended_stake_fraction BETWEEN 0 AND 1),
            CHECK (status != 'ACTIONABLE' OR
                   (production_eligible = 1 AND recommended_stake_dollars > 0 AND
                    quote_id IS NOT NULL AND validation_id IS NOT NULL)),
            FOREIGN KEY (decision_id, quote_id) REFERENCES parlay_quote(decision_id, quote_id)
                DEFERRABLE INITIALLY DEFERRED
        );
        CREATE TABLE IF NOT EXISTS parlay_leg (
            decision_id TEXT NOT NULL REFERENCES parlay_ticket(decision_id),
            leg_index INTEGER NOT NULL CHECK (leg_index >= 0),
            candidate_id TEXT,
            game_id TEXT,
            event_id TEXT,
            sport TEXT,
            market TEXT,
            selection TEXT,
            line REAL,
            sportsbook TEXT,
            leg_hash TEXT,
            model_id TEXT,
            model_version TEXT,
            calibration_id TEXT,
            calibration_version TEXT,
            evidence_id TEXT,
            policy_version TEXT,
            model_trained_through TEXT,
            model_available_at TEXT,
            calibration_available_at TEXT,
            evidence_frozen_at TEXT,
            analysis_at TEXT,
            game_start_at TEXT,
            quote_at TEXT,
            probability_semantics TEXT,
            probability_mean REAL,
            probability_conservative REAL,
            probability_push REAL,
            probability_loss REAL,
            critical_input_state TEXT,
            material_news_status TEXT,
            production_eligible INTEGER CHECK (production_eligible IN (0,1)),
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY (decision_id, leg_index)
        );
        CREATE TABLE IF NOT EXISTS parlay_quote (
            decision_id TEXT NOT NULL REFERENCES parlay_ticket(decision_id),
            quote_id TEXT NOT NULL,
            parlay_id TEXT NOT NULL REFERENCES parlay_identity(parlay_id),
            sportsbook TEXT,
            odds_american REAL,
            odds_decimal REAL,
            quoted_at TEXT,
            expires_at TEXT,
            quote_source TEXT,
            provider_ticket_id TEXT,
            ticket_hash TEXT,
            leg_hash TEXT,
            verification_state TEXT,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY (decision_id, quote_id)
        );
        CREATE TABLE IF NOT EXISTS parlay_gate (
            decision_id TEXT NOT NULL REFERENCES parlay_ticket(decision_id),
            gate_index INTEGER NOT NULL CHECK (gate_index >= 0),
            gate_name TEXT NOT NULL,
            passed INTEGER CHECK (passed IN (0,1)),
            evaluated_at TEXT,
            blockers TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY (decision_id, gate_index)
        );
        CREATE TABLE IF NOT EXISTS parlay_result (
            decision_id TEXT NOT NULL REFERENCES parlay_ticket(decision_id),
            parlay_id TEXT NOT NULL REFERENCES parlay_identity(parlay_id),
            grading_version INTEGER NOT NULL CHECK (grading_version > 0),
            result_kind TEXT NOT NULL CHECK (result_kind IN ('SELECTION','ACCEPTED_WAGER')),
            outcome TEXT,
            settlement TEXT,
            selection_return REAL,
            net_return REAL,
            accepted_wager_id TEXT,
            accepted_decimal_odds REAL,
            accepted_stake REAL,
            placed_at TEXT,
            settlement_rule TEXT,
            closing_decimal_odds REAL,
            source_id TEXT,
            graded_at TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY (decision_id, grading_version)
        );
        CREATE TABLE IF NOT EXISTS parlay_validation_evidence (
            evidence_id TEXT PRIMARY KEY,
            decision_id TEXT REFERENCES parlay_ticket(decision_id),
            parlay_id TEXT REFERENCES parlay_identity(parlay_id),
            product_type TEXT NOT NULL CHECK (product_type IN
                ('STANDARD_PARLAY','SAME_GAME_PARLAY','CROSS_GAME_PARLAY')),
            validation_state TEXT NOT NULL CHECK (validation_state IN
                ('UNVALIDATED','PROVISIONAL_VALIDATED','STANDARD_VALIDATED')),
            validation_id TEXT,
            policy_id TEXT,
            policy_version TEXT,
            model_id TEXT,
            model_version TEXT,
            calibration_id TEXT,
            calibration_version TEXT,
            evidence_at TEXT,
            available_at TEXT,
            recorded_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS parlay_ticket_product_at
            ON parlay_ticket(product_type, decision_at);
        CREATE INDEX IF NOT EXISTS parlay_ticket_identity
            ON parlay_ticket(parlay_id, decision_at);
        CREATE INDEX IF NOT EXISTS parlay_result_at
            ON parlay_result(graded_at);
        CREATE INDEX IF NOT EXISTS parlay_validation_product
            ON parlay_validation_evidence(product_type, validation_state);
        CREATE TRIGGER IF NOT EXISTS parlay_ticket_identity_conflict BEFORE INSERT ON parlay_ticket
        WHEN EXISTS (
            SELECT 1 FROM parlay_ticket WHERE parlay_id=NEW.parlay_id AND
                (product_type IS NOT NEW.product_type OR sportsbook IS NOT NEW.sportsbook OR
                 leg_hash IS NOT NEW.leg_hash OR ticket_hash IS NOT NEW.ticket_hash)
        ) BEGIN SELECT RAISE(ABORT, 'parlay identity conflict'); END;
        CREATE TRIGGER IF NOT EXISTS parlay_quote_identity_conflict BEFORE INSERT ON parlay_quote
        WHEN EXISTS (
            SELECT 1 FROM parlay_quote WHERE quote_id=NEW.quote_id AND
                (parlay_id IS NOT NEW.parlay_id OR payload_hash IS NOT NEW.payload_hash)
        ) BEGIN SELECT RAISE(ABORT, 'quote identity conflict'); END;
        CREATE TRIGGER IF NOT EXISTS parlay_quote_decision_conflict BEFORE INSERT ON parlay_quote
        WHEN (SELECT parlay_id FROM parlay_ticket WHERE decision_id=NEW.decision_id) IS NOT NEW.parlay_id
        BEGIN SELECT RAISE(ABORT, 'quote decision mismatch'); END;
        CREATE TRIGGER IF NOT EXISTS parlay_result_decision_conflict BEFORE INSERT ON parlay_result
        WHEN (SELECT parlay_id FROM parlay_ticket WHERE decision_id=NEW.decision_id) IS NOT NEW.parlay_id
        BEGIN SELECT RAISE(ABORT, 'result decision mismatch'); END;
        CREATE TRIGGER IF NOT EXISTS parlay_receipt_decision_conflict BEFORE INSERT ON parlay_result
        WHEN NEW.accepted_wager_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM parlay_result WHERE accepted_wager_id=NEW.accepted_wager_id
              AND decision_id IS NOT NEW.decision_id
        ) BEGIN SELECT RAISE(ABORT, 'accepted receipt belongs to another decision'); END;
    """)
    for table in ("parlay_identity", "parlay_ticket", "parlay_leg", "parlay_quote", "parlay_gate",
                  "parlay_result", "parlay_validation_evidence"):
        for action in ("UPDATE", "DELETE"):
            db.execute(
                f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'parlay evidence is append-only'); END"
            )
    return db


def _boolean(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return int(value)


def _blockers(value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("blockers must be an array of strings")
    return _json(value)


def _record(row: Mapping) -> tuple[str, str]:
    payload = _json(dict(row))
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_decision(
    path: str | Path | None,
    decision: Mapping,
    legs: Sequence[Mapping],
    *,
    quote: Mapping | None = None,
    gates: Sequence[Mapping] = (),
) -> str:
    """Freeze one exact decision with its legs, quote, and gate trace atomically.

    Returns the deterministic decision_id on insertion and identical retries.
    Repricing adds a new decision under the same stable parlay_id. Reusing a
    quote_id for changed quote facts, or a decision_id for changed facts, fails.
    """
    ticket = _mapping(decision, "decision")
    parlay_id = _nonempty(ticket.get("parlay_id"), "parlay_id")
    product = ticket.get("product_type")
    if product not in PRODUCTS:
        raise ValueError("invalid product_type")
    _nonempty(ticket.get("status"), "status")
    decision_at = _time(ticket.get("decision_at"), "decision_at", required=True)
    if ticket.get("validation_state") is not None and ticket["validation_state"] not in VALIDATION_STATES:
        raise ValueError("invalid validation_state")
    if not isinstance(legs, (list, tuple)) or not legs:
        raise ValueError("at least one leg is required")
    leg_rows = [_mapping(leg, "leg") for leg in legs]
    if "legs" in ticket and ticket["legs"] != leg_rows:
        raise ValueError("decision legs mismatch")
    hashes = [leg.get("leg_hash") for leg in leg_rows]
    bound_leg_hash = _digest(sorted(hashes)) if all(isinstance(h, str) and h for h in hashes) else None
    if ticket.get("leg_hash") is not None and bound_leg_hash is not None and ticket["leg_hash"] != bound_leg_hash:
        raise ValueError("decision leg_hash mismatch")
    bound_leg_hash = ticket.get("leg_hash") or bound_leg_hash
    gate_rows = [_mapping(gate, "gate") for gate in gates]
    quote_row = _mapping(quote, "quote") if quote is not None else None
    quote_id = ticket.get("quote_id")
    quote_leg_hash = None
    if quote_row is not None:
        quote_id = _nonempty(quote_row.get("quote_id"), "quote_id")
        if ticket.get("quote_id") not in (None, quote_id):
            raise ValueError("quote_id mismatch")
        if quote_row.get("parlay_id") not in (None, parlay_id):
            raise ValueError("quote parlay_id mismatch")
        for field in ("sportsbook", "ticket_hash"):
            if ticket.get(field) is not None and quote_row.get(field) is not None and ticket[field] != quote_row[field]:
                raise ValueError(f"quote {field} mismatch")
        quote_facts = (
            ("provider_ticket_id", "provider_ticket_id"),
            ("quote_source", "source"),
            ("quote_verification_state", "verification_state"),
            ("quoted_decimal_odds", "decimal_odds"),
            ("quoted_american_odds", "american_odds"),
            ("quoted_at", "quoted_at"),
            ("expires_at", "expires_at"),
        )
        for ticket_field, quote_field in quote_facts:
            if ticket.get(ticket_field) is not None and quote_row.get(quote_field) is not None \
                    and ticket[ticket_field] != quote_row[quote_field]:
                raise ValueError(f"quote {ticket_field} mismatch")
        quoted_hashes = quote_row.get("leg_hashes")
        if quoted_hashes is not None:
            if not (isinstance(quoted_hashes, list) and
                    all(isinstance(value, str) and value for value in quoted_hashes)):
                raise ValueError("quote leg_hashes invalid")
            if not all(isinstance(value, str) and value for value in hashes) or sorted(quoted_hashes) != sorted(hashes):
                raise ValueError("quote leg_hashes mismatch")
            quote_leg_hash = _digest(sorted(quoted_hashes))
        if quote_row.get("leg_hash") is not None and quote_leg_hash is not None and quote_row["leg_hash"] != quote_leg_hash:
            raise ValueError("quote leg_hash mismatch")
        quote_leg_hash = quote_row.get("leg_hash") or quote_leg_hash
        if bound_leg_hash is not None and quote_leg_hash is not None and bound_leg_hash != quote_leg_hash:
            raise ValueError("quote leg_hash mismatch")
    elif quote_id is not None:
        raise ValueError("quote_id requires an exact quote")
    stake = _number(_first(ticket, "recommended_stake_dollars", "recommended_dollars", "recommended_stake"),
                    "recommended_stake_dollars")
    fraction = _number(_first(ticket, "recommended_stake_fraction", "recommended_fraction"),
                       "recommended_stake_fraction")
    if stake is not None and stake < 0:
        raise ValueError("recommended stake cannot be negative")
    if fraction is not None and not 0 <= fraction <= 1:
        raise ValueError("recommended fraction must be between zero and one")
    eligible = _boolean(ticket.get("production_eligible"), "production_eligible")
    if stake is not None and stake > 0 and eligible != 1:
        raise ValueError("positive stake requires production eligibility")
    if stake is not None and stake > 0 and ticket["status"] != "ACTIONABLE":
        raise ValueError("positive stake requires ACTIONABLE status")
    if ticket["status"] == "ACTIONABLE":
        price_decimal = _number(_first(quote_row or {}, "odds_decimal", "decimal_odds"), "odds_decimal")
        price_american = _number(_first(quote_row or {}, "odds_american", "american_odds"), "odds_american")
        converted = (1 + (price_american / 100 if price_american > 0 else 100 / -price_american)
                     if price_american is not None and abs(price_american) >= 100 else None)
        mean = _number(_first(ticket, "probability_mean", "mean_probability"), "probability_mean")
        conservative = _number(_first(ticket, "probability_conservative", "conservative_probability"),
                               "probability_conservative")
        push = _number(_first(ticket, "probability_push", "push_probability"), "probability_push")
        if not (eligible == 1 and stake is not None and stake > 0 and quote_row is not None
                and quote_row.get("verification_state") == "VERIFIED"
                and ticket.get("validation_id") and ticket.get("validation_state") in
                {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED"}
                and ticket.get("ticket_hash") and bound_leg_hash
                and ticket.get("policy_version")
                and _first(ticket, "model_id", "joint_model_id")
                and _first(ticket, "model_version", "joint_model_version")
                and ticket.get("calibration_version")
                and _first(ticket, "evidence_id", "joint_evidence_snapshot_id")
                and quote_row.get("ticket_hash") == ticket.get("ticket_hash")
                and quote_leg_hash == bound_leg_hash
                and quote_row.get("sportsbook") == ticket.get("sportsbook")
                and _first(quote_row, "quote_source", "source")
                and quote_row.get("provider_ticket_id")
                and price_decimal is not None and price_decimal > 1
                and converted is not None and abs(converted - price_decimal) <= 0.002
                and mean is not None and conservative is not None and push is not None
                and 0 < conservative <= mean < 1 and 0 <= push < 1 and mean + push <= 1
                and (_number(ticket.get("conservative_ev"), "conservative_ev") or 0) > 0):
            raise ValueError("ACTIONABLE requires stake, validation, and verified exact quote")
        quoted = _time(quote_row.get("quoted_at"), "quoted_at", required=True)
        expires = _time(quote_row.get("expires_at"), "expires_at", required=True)
        if not (datetime.fromisoformat(quoted.replace("Z", "+00:00"))
                <= datetime.fromisoformat(decision_at.replace("Z", "+00:00"))
                < datetime.fromisoformat(expires.replace("Z", "+00:00"))):
            raise ValueError("ACTIONABLE quote is not current at decision time")
    blockers = _blockers(ticket.get("blockers", []))
    snapshot = {"decision": ticket, "legs": leg_rows, "quote": quote_row, "gates": gate_rows}
    snapshot_hash = _digest(snapshot)
    decision_id = ticket.get("decision_id") or snapshot_hash
    _nonempty(decision_id, "decision_id")
    payload, payload_hash = _record(ticket)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        identity = {key: ticket.get(key) for key in
                    ("parlay_id", "product_type", "sportsbook", "leg_hash", "ticket_hash")}
        identity["leg_hash"] = bound_leg_hash
        identity_hash = _digest(identity)
        saved_identity = db.execute("SELECT identity_hash FROM parlay_identity WHERE parlay_id=?",
                                    (parlay_id,)).fetchone()
        if saved_identity is not None and saved_identity[0] != identity_hash:
            raise EvidenceConflict("parlay_id already has different leg/product/book identity")
        existing = db.execute("SELECT snapshot_hash FROM parlay_ticket WHERE decision_id=?", (decision_id,)).fetchone()
        if existing is not None:
            if existing[0] != snapshot_hash:
                raise EvidenceConflict("decision_id already has different immutable facts")
            return decision_id
        if saved_identity is None:
            db.execute("INSERT INTO parlay_identity VALUES (?,?,?,?,?,?)", (
                parlay_id, product, ticket.get("sportsbook"), bound_leg_hash,
                ticket.get("ticket_hash"), identity_hash,
            ))
        ticket_values = (
            decision_id, parlay_id, product, ticket["status"], ticket.get("sportsbook"), quote_id,
            bound_leg_hash, ticket.get("ticket_hash"), ticket.get("policy_id"),
            ticket.get("policy_version"), ticket.get("validation_id"), ticket.get("validation_state"),
            ticket.get("deployment_state"), _first(ticket, "model_id", "joint_model_id"),
            _first(ticket, "model_version", "joint_model_version"),
            ticket.get("calibration_id"), ticket.get("calibration_version"),
            _number(_first(ticket, "probability_mean", "mean_probability"), "probability_mean"),
            _number(_first(ticket, "probability_conservative", "conservative_probability"), "probability_conservative"),
            _number(_first(ticket, "probability_break_even", "break_even_probability"), "probability_break_even"),
            ticket.get("probability_method"), _number(ticket.get("conservative_ev"), "conservative_ev"),
            _number(ticket.get("conservative_edge"), "conservative_edge"), stake, fraction, eligible,
            decision_at, _time(ticket.get("analysis_at"), "analysis_at"),
            _first(ticket, "evidence_id", "joint_evidence_snapshot_id"),
            blockers, payload, payload_hash, snapshot_hash,
        )
        db.execute(f"INSERT INTO parlay_ticket VALUES ({','.join('?' for _ in ticket_values)})", ticket_values)
        for index, leg in enumerate(leg_rows):
            leg_payload, leg_hash = _record(leg)
            leg_values = (
                decision_id, index, leg.get("candidate_id"), leg.get("game_id"), leg.get("event_id"),
                leg.get("sport"), _first(leg, "market", "market_type"), leg.get("selection"),
                _number(leg.get("line"), "leg line"), leg.get("sportsbook"), leg.get("leg_hash"),
                leg.get("model_id"), leg.get("model_version"), leg.get("calibration_id"),
                leg.get("calibration_version"), _first(leg, "evidence_id", "evidence_snapshot_id"),
                leg.get("policy_version"),
                _time(leg.get("model_trained_through"), "model_trained_through"),
                _time(leg.get("model_available_at"), "model_available_at"),
                _time(leg.get("calibration_available_at"), "calibration_available_at"),
                _time(leg.get("evidence_frozen_at"), "evidence_frozen_at"),
                _time(_first(leg, "analysis_at", "analysis_timestamp"), "analysis_at"),
                _time(_first(leg, "game_start_at", "start"), "game_start_at"),
                _time(_first(leg, "quote_at", "quote_timestamp", "quote_time"), "quote_at"),
                leg.get("probability_semantics"),
                _number(_first(leg, "probability_mean", "mean_probability"), "leg probability_mean"),
                _number(_first(leg, "probability_conservative", "conservative_probability"), "leg probability_conservative"),
                _number(_first(leg, "probability_push", "push_probability"), "leg probability_push"),
                _number(leg.get("probability_loss"), "leg probability_loss"),
                leg.get("critical_input_state"), leg.get("material_news_status"),
                _boolean(leg.get("production_eligible"), "leg production_eligible"), leg_payload, leg_hash,
            )
            db.execute(f"INSERT INTO parlay_leg VALUES ({','.join('?' for _ in leg_values)})", leg_values)
        if quote_row is not None:
            quote_payload, quote_hash = _record(quote_row)
            saved_quote = db.execute("SELECT parlay_id,payload_hash FROM parlay_quote WHERE quote_id=? LIMIT 1",
                                     (quote_id,)).fetchone()
            if saved_quote is not None and saved_quote != (parlay_id, quote_hash):
                raise EvidenceConflict("quote identity conflict")
            quote_values = (
                decision_id, quote_id, parlay_id, quote_row.get("sportsbook"),
                _number(_first(quote_row, "odds_american", "american_odds"), "odds_american"),
                _number(_first(quote_row, "odds_decimal", "decimal_odds"), "odds_decimal"),
                _time(quote_row.get("quoted_at"), "quoted_at"), _time(quote_row.get("expires_at"), "expires_at"),
                _first(quote_row, "quote_source", "source"), quote_row.get("provider_ticket_id"),
                quote_row.get("ticket_hash"), quote_leg_hash,
                quote_row.get("verification_state"), quote_payload, quote_hash,
            )
            db.execute(f"INSERT INTO parlay_quote VALUES ({','.join('?' for _ in quote_values)})", quote_values)
        for index, gate in enumerate(gate_rows):
            gate_payload, gate_hash = _record(gate)
            db.execute("INSERT INTO parlay_gate VALUES (?,?,?,?,?,?,?,?)", (
                decision_id, index, _nonempty(gate.get("gate_name"), "gate_name"),
                _boolean(gate.get("passed"), "gate passed"),
                _time(gate.get("evaluated_at"), "evaluated_at"),
                _blockers(gate.get("blockers", [])), gate_payload, gate_hash,
            ))
    return decision_id


def append_result(path: str | Path | None, decision_id: str, result: Mapping) -> bool:
    """Append a grading revision; same facts/version are idempotent.

    `graded_at` records the first write's grading time. A retry with identical
    facts and version may supply a new graded_at, but it cannot change that
    frozen timestamp or create a second revision.
    """
    _nonempty(decision_id, "decision_id")
    row = _mapping(result, "result")
    version = row.get("grading_version")
    if type(version) is not int or version <= 0:
        raise ValueError("grading_version must be a positive integer")
    kind = row.get("result_kind")
    if kind not in RESULT_KINDS:
        raise ValueError("result_kind must distinguish selection from accepted wager")
    if kind == "SELECTION":
        if any(row.get(field) is not None for field in
               ("net_return", "accepted_wager_id", "accepted_decimal_odds", "accepted_stake", "placed_at")):
            raise ValueError("selection grading cannot claim accepted wager return")
    else:
        if row.get("selection_return") is not None:
            raise ValueError("accepted wager return must be separate from selection return")
        if (not row.get("accepted_wager_id") or
                (_number(row.get("accepted_decimal_odds"), "accepted_decimal_odds") or 0) <= 1 or
                (_number(row.get("accepted_stake"), "accepted_stake") or 0) <= 0):
            raise ValueError("accepted wager requires receipt, actual price, and stake")
        _time(row.get("placed_at"), "placed_at", required=True)
    graded_at = _time(row.get("graded_at"), "graded_at", required=True)
    material = {key: value for key, value in row.items()
                if key not in {"grading_version", "graded_at", "recorded_at"}}
    input_hash = _digest(material)
    payload, payload_hash = _record(row)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        parent = db.execute("SELECT parlay_id FROM parlay_ticket WHERE decision_id=?", (decision_id,)).fetchone()
        if parent is None:
            raise ValueError("unknown decision_id")
        parlay_id = parent[0]
        existing = db.execute(
            "SELECT input_hash FROM parlay_result WHERE decision_id=? AND grading_version=?",
            (decision_id, version),
        ).fetchone()
        if existing is not None:
            if existing[0] != input_hash:
                raise EvidenceConflict("grading version already has different inputs")
            return False
        latest = db.execute("SELECT grading_version,input_hash FROM parlay_result WHERE decision_id=? "
                            "ORDER BY grading_version DESC LIMIT 1", (decision_id,)).fetchone()
        expected = (latest[0] if latest else 0) + 1
        if version != expected:
            raise ValueError("grading revisions must be consecutive")
        if latest and latest[1] == input_hash:
            raise ValueError("unchanged grading inputs do not need a new revision")
        result_values = (
            decision_id, parlay_id, version, kind, row.get("outcome"), row.get("settlement"),
            _number(row.get("selection_return"), "selection_return"),
            _number(row.get("net_return"), "net_return"),
            row.get("accepted_wager_id"),
            _number(row.get("accepted_decimal_odds"), "accepted_decimal_odds"),
            _number(row.get("accepted_stake"), "accepted_stake"),
            _time(row.get("placed_at"), "placed_at"), row.get("settlement_rule"),
            _number(row.get("closing_decimal_odds"), "closing_decimal_odds"),
            row.get("source_id"), graded_at, datetime.now(timezone.utc).isoformat(),
            input_hash, payload, payload_hash,
        )
        db.execute(f"INSERT INTO parlay_result VALUES ({','.join('?' for _ in result_values)})", result_values)
    return True


def append_validation_evidence(path: str | Path | None, evidence: Mapping) -> bool:
    """Record a validation claim without activating a product or stake policy."""
    row = _mapping(evidence, "validation evidence")
    evidence_id = _nonempty(row.get("evidence_id"), "evidence_id")
    if row.get("product_type") not in PRODUCTS or row.get("validation_state") not in VALIDATION_STATES:
        raise ValueError("invalid validation product or state")
    payload, payload_hash = _record(row)
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload_hash FROM parlay_validation_evidence WHERE evidence_id=?",
                              (evidence_id,)).fetchone()
        if existing is not None:
            if existing[0] != payload_hash:
                raise EvidenceConflict("evidence_id already has different facts")
            return False
        decision_id = row.get("decision_id")
        if decision_id is not None:
            parent = db.execute("SELECT parlay_id,product_type FROM parlay_ticket WHERE decision_id=?",
                                (decision_id,)).fetchone()
            if parent is None or (row.get("parlay_id") not in (None, parent[0])
                                  or row["product_type"] != parent[1]):
                raise ValueError("validation evidence ticket identity mismatch")
        elif row.get("parlay_id") is not None and db.execute(
                "SELECT 1 FROM parlay_identity WHERE parlay_id=?", (row["parlay_id"],)).fetchone() is None:
            raise ValueError("unknown parlay_id")
        evidence_values = (
            evidence_id, decision_id, row.get("parlay_id"), row["product_type"], row["validation_state"],
            row.get("validation_id"), row.get("policy_id"), row.get("policy_version"),
            row.get("model_id"), row.get("model_version"), row.get("calibration_id"),
            row.get("calibration_version"), _time(row.get("evidence_at"), "evidence_at"),
            _time(row.get("available_at"), "available_at"),
            datetime.now(timezone.utc).isoformat(), payload, payload_hash,
        )
        db.execute(f"INSERT INTO parlay_validation_evidence VALUES ({','.join('?' for _ in evidence_values)})",
                   evidence_values)
    return True


def _verified_payload(payload: str, digest: str) -> dict:
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
        raise EvidenceConflict("stored parlay evidence hash mismatch")
    return json.loads(payload)


def load_decision(path: str | Path | None, decision_id: str) -> dict | None:
    """Read and verify the frozen decision and every normalized child record."""
    with closing(connect(path)) as db:
        ticket = db.execute("SELECT payload,payload_hash,snapshot_hash,quote_id FROM parlay_ticket WHERE decision_id=?",
                            (decision_id,)).fetchone()
        if ticket is None:
            return None
        decision = _verified_payload(ticket[0], ticket[1])
        legs = [_verified_payload(payload, digest) for payload, digest in db.execute(
            "SELECT payload,payload_hash FROM parlay_leg WHERE decision_id=? ORDER BY leg_index", (decision_id,))]
        quote_row = db.execute("SELECT payload,payload_hash FROM parlay_quote WHERE quote_id=? AND decision_id=?",
                               (ticket[3], decision_id)).fetchone() if ticket[3] else None
        quote = _verified_payload(*quote_row) if quote_row else None
        gates = [_verified_payload(payload, digest) for payload, digest in db.execute(
            "SELECT payload,payload_hash FROM parlay_gate WHERE decision_id=? ORDER BY gate_index", (decision_id,))]
    snapshot = {"decision": decision, "legs": legs, "quote": quote, "gates": gates}
    if _digest(snapshot) != ticket[2]:
        raise EvidenceConflict("stored parlay snapshot hash mismatch")
    return dict(snapshot, decision_id=decision_id)


def decision_history(path: str | Path | None, parlay_id: str) -> list[str]:
    """Return frozen decision IDs for one stable leg/product/book identity."""
    with closing(connect(path)) as db:
        rows = db.execute("SELECT decision_id FROM parlay_ticket WHERE parlay_id=? ORDER BY decision_at,decision_id",
                          (parlay_id,)).fetchall()
    return [row[0] for row in rows]


def result_revisions(path: str | Path | None, decision_id: str) -> list[dict]:
    with closing(connect(path)) as db:
        rows = db.execute("SELECT payload,payload_hash FROM parlay_result WHERE decision_id=? ORDER BY grading_version",
                          (decision_id,)).fetchall()
    return [_verified_payload(payload, digest) for payload, digest in rows]


def validation_evidence(path: str | Path | None, product_type: str) -> list[dict]:
    """Return verified, product-scoped evidence without treating it as authority."""
    if product_type not in PRODUCTS:
        raise ValueError("invalid product_type")
    with closing(connect(path)) as db:
        rows = db.execute("SELECT payload,payload_hash FROM parlay_validation_evidence "
                          "WHERE product_type=? ORDER BY recorded_at,evidence_id", (product_type,)).fetchall()
    return [_verified_payload(payload, digest) for payload, digest in rows]
