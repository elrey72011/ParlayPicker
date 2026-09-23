"""Persisted trial consent and recommendation reservations; no bet placement.

Consent is an append-only owner action. A recommendation reserves risk until an
owner explicitly releases it; a browser refresh cannot restore the budget.
Committed straight and parlay risk comes from the existing exposure ledger.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3

from core.exposure_ledger import digest, events, snapshot, verify_snapshot
from core.wager_decisions import aware, finite


CONSENT_PATH_ENV = "PARLAYPICKER_CONTROLLED_TRIAL_CONSENT_LEDGER"
RESERVATION_PATH_ENV = "PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER"
EXPOSURE_PATH_ENV = "PARLAYPICKER_EXPOSURE_LEDGER"
DEFAULT_CONSENT_PATH = "data/exposure/controlled-trial-consent.sqlite3"
DEFAULT_RESERVATION_PATH = "data/exposure/controlled-trial-reservations.sqlite3"
DEFAULT_EXPOSURE_PATH = "data/exposure/exposure.sqlite3"


def _connect(path, *, create):
    target = Path(path)
    if not create and not target.is_file():
        raise ValueError("LEDGER_MISSING")
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(target), timeout=30)
    db.row_factory = sqlite3.Row
    return db


def _init_consent(db):
    db.execute("""CREATE TABLE IF NOT EXISTS consent_events (
        event_id TEXT PRIMARY KEY, prior_event_id TEXT, status TEXT NOT NULL,
        owner TEXT NOT NULL, recorded_at TEXT NOT NULL, expires_at TEXT)""")
    for action in ("UPDATE", "DELETE"):
        db.execute(f"""CREATE TRIGGER IF NOT EXISTS consent_no_{action.lower()}
            BEFORE {action} ON consent_events
            BEGIN SELECT RAISE(ABORT, 'consent is append-only'); END""")


def record_consent(path, status, owner, *, expires_at=None, confirmed=False, now=None):
    """Append an owner-confirmed grant or revocation. Never called by analysis."""
    if not confirmed or status not in {"GRANTED", "REVOKED"} or not str(owner).strip():
        raise ValueError("EXPLICIT_OWNER_CONFIRMATION_REQUIRED")
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError("CONSENT_TIME_UNVERIFIED")
    at = at.astimezone(timezone.utc)
    expiry = aware(expires_at)
    if status == "GRANTED" and (expiry is None or expiry <= at):
        raise ValueError("CONSENT_EXPIRY_REQUIRED")
    if status == "REVOKED" and expires_at is not None:
        raise ValueError("REVOCATION_HAS_NO_EXPIRY")
    with closing(_connect(path, create=True)) as db:
        db.execute("BEGIN IMMEDIATE")
        _init_consent(db)
        previous = db.execute("SELECT event_id FROM consent_events ORDER BY rowid DESC LIMIT 1").fetchone()
        payload = dict(prior_event_id=previous[0] if previous else None, status=status,
                       owner=str(owner).strip(), recorded_at=at.isoformat(),
                       expires_at=expiry.isoformat() if expiry else None)
        event_id = digest(payload)
        db.execute("INSERT INTO consent_events VALUES (?, ?, ?, ?, ?, ?)",
                   (event_id, *payload.values()))
        db.commit()
    return event_id


def consent_status(path=None, *, now=None):
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None:
        return None, "CONSENT_TIME_UNVERIFIED"
    path = path or os.getenv(CONSENT_PATH_ENV, DEFAULT_CONSENT_PATH)
    try:
        with closing(_connect(path, create=False)) as db:
            rows = db.execute("SELECT * FROM consent_events ORDER BY rowid").fetchall()
        previous = None
        latest = None
        for row in rows:
            payload = {key: row[key] for key in ("prior_event_id", "status", "owner", "recorded_at", "expires_at")}
            recorded = aware(payload["recorded_at"])
            if (digest(payload) != row["event_id"] or payload["prior_event_id"] != previous
                    or payload["status"] not in {"GRANTED", "REVOKED"}
                    or not payload["owner"] or recorded is None or recorded > at
                    or (latest and recorded < aware(latest["recorded_at"]))):
                return None, "CONSENT_LEDGER_INVALID"
            previous = row["event_id"]
            latest = dict(payload, event_id=row["event_id"])
        if not latest:
            return None, "CONSENT_MISSING"
        if latest["status"] != "GRANTED":
            return None, "CONSENT_REVOKED"
        expiry = aware(latest["expires_at"])
        if expiry is None or expiry <= at:
            return None, "CONSENT_EXPIRED"
        return latest, "AUTHORIZED"
    except (sqlite3.Error, OSError, ValueError):
        return None, "CONSENT_LEDGER_UNAVAILABLE"


def exposure_status(bankroll, *, now=None, path=None):
    at = now or datetime.now(timezone.utc)
    path = path or os.getenv(EXPOSURE_PATH_ENV, DEFAULT_EXPOSURE_PATH)
    if not Path(path).is_file():
        return None, None, "EXPOSURE_LEDGER_UNAVAILABLE"
    try:
        current = verify_snapshot(snapshot(path, now=at), now=at)
        if finite(bankroll) != finite(current.get("bankroll")):
            return None, None, "BANKROLL_LEDGER_MISMATCH"
        # Core snapshot counts placed commitments. Unresolved owner-recorded
        # recommendations also reserve capacity until ledger reconciliation.
        history = events(path)
        latest = {event.get("bet_id"): event["status"] for event in history if event.get("bet_id")}
        extra = {}
        for event in history:
            if event["status"] != "RECOMMENDED" or latest.get(event.get("bet_id")) != "RECOMMENDED":
                continue
            fraction = finite(event.get("stake_dollars")) / current["bankroll"]
            keys = {"total", "daily", "weekly"}
            for leg in event["legs"]:
                sport, game = leg["sport"], leg["game_id"]
                keys.update({f"sport:{sport}", f"game:{sport}:{game}"})
                keys.update(f"team:{sport}:{team}" for team in leg["team_ids"])
            for key in keys:
                extra[key] = extra.get(key, 0.0) + fraction
        return current, extra, "LEDGER_CURRENT"
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, ZeroDivisionError):
        return None, None, "EXPOSURE_LEDGER_UNAVAILABLE"


def _init_reservations(db):
    db.execute("""CREATE TABLE IF NOT EXISTS reservations (
        reservation_id TEXT PRIMARY KEY, reserved_at TEXT NOT NULL,
        consent_id TEXT NOT NULL, slate_key TEXT NOT NULL, sport TEXT NOT NULL,
        game_id TEXT NOT NULL, team_ids TEXT NOT NULL, stake_dollars REAL NOT NULL,
        bankroll REAL NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS releases (
        release_id TEXT PRIMARY KEY, reservation_id TEXT NOT NULL UNIQUE,
        released_at TEXT NOT NULL, owner TEXT NOT NULL,
        FOREIGN KEY (reservation_id) REFERENCES reservations(reservation_id))""")
    for table in ("reservations", "releases"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()}
                BEFORE {action} ON {table}
                BEGIN SELECT RAISE(ABORT, 'trial risk history is append-only'); END""")


def release_reservation(path, reservation_id, owner, *, confirmed=False, now=None):
    """Owner-only release after verifying a recommendation was not placed."""
    if not confirmed or not str(owner).strip():
        raise ValueError("EXPLICIT_OWNER_CONFIRMATION_REQUIRED")
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError("RELEASE_TIME_UNVERIFIED")
    with closing(_connect(path, create=False)) as db:
        db.execute("BEGIN IMMEDIATE")
        _init_reservations(db)
        if not db.execute("SELECT 1 FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone():
            raise ValueError("RESERVATION_NOT_FOUND")
        payload = dict(reservation_id=reservation_id, released_at=at.astimezone(timezone.utc).isoformat(), owner=str(owner).strip())
        release_id = digest(payload)
        db.execute("INSERT INTO releases VALUES (?, ?, ?, ?)", (release_id, *payload.values()))
        db.commit()
    return release_id


def reserve_recommendation(identity, amount, *, consent, exposure, external_recommended,
                           slate_key, sport, game_id, team_ids, now=None, max_picks=2,
                           path=None):
    """Atomically reserve downward-capped risk for one exact candidate quote."""
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None or not consent or not isinstance(exposure, dict):
        return 0.0, None, "AUTHORITY_UNAVAILABLE"
    try:
        verify_snapshot(exposure, now=at)
    except (ValueError, TypeError, KeyError):
        return 0.0, None, "EXPOSURE_SNAPSHOT_STALE"
    bank = finite(exposure.get("bankroll"))
    requested = finite(amount)
    if (bank is None or bank <= 0 or requested is None or requested <= 0
            or not all(isinstance(x, str) and x.strip() for x in (slate_key, sport, game_id))
            or not isinstance(team_ids, (list, tuple)) or len(team_ids) != 2
            or len(set(team_ids)) != 2 or not all(isinstance(x, str) and x.strip() for x in team_ids)):
        return 0.0, None, "TRIAL_EXPOSURE_IDENTITY_MISSING"
    reservation_id = digest(dict(identity=identity, consent_id=consent["event_id"]))
    path = path or os.getenv(RESERVATION_PATH_ENV)
    if not path:
        return 0.0, None, "RESERVATION_LEDGER_NOT_CONFIGURED"
    consent_path = os.getenv(CONSENT_PATH_ENV, DEFAULT_CONSENT_PATH)
    exposure_path = os.getenv(EXPOSURE_PATH_ENV, DEFAULT_EXPOSURE_PATH)
    if not Path(consent_path).is_file() or not Path(exposure_path).is_file():
        return 0.0, None, "AUTHORITY_LEDGER_UNAVAILABLE"
    try:
        with closing(_connect(path, create=True)) as db:
            # Hold writer locks on the authority ledgers while evaluating and
            # reserving, so a concurrent revocation/commit cannot slip between
            # the source snapshot and this durable reservation.
            db.execute("ATTACH DATABASE ? AS consent_authority", (str(consent_path),))
            db.execute("ATTACH DATABASE ? AS exposure_authority", (str(exposure_path),))
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute("SELECT event_id FROM consent_authority.consent_events ORDER BY rowid DESC LIMIT 1").fetchone()
            if latest is None or latest[0] != consent["event_id"]:
                db.rollback()
                return 0.0, None, "CONSENT_CHANGED"
            current_exposure, current_external, current_reason = exposure_status(
                bank, now=at, path=exposure_path
            )
            if (current_exposure is None or current_reason != "LEDGER_CURRENT"
                    or current_exposure["ledger_hash"] != exposure["ledger_hash"]
                    or current_external != external_recommended):
                db.rollback()
                return 0.0, None, "EXPOSURE_LEDGER_CHANGED"
            _init_reservations(db)
            rows = db.execute("""SELECT r.* FROM reservations r LEFT JOIN releases x
                ON x.reservation_id=r.reservation_id WHERE x.reservation_id IS NULL""").fetchall()
            existing = next((row for row in rows if row["reservation_id"] == reservation_id), None)
            if existing is None and db.execute("SELECT 1 FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone():
                db.rollback()
                return 0.0, reservation_id, "RESERVATION_ALREADY_RELEASED"
            slate_count = sum(row["slate_key"] == slate_key for row in rows)
            if existing is None and slate_count >= max_picks:
                db.rollback()
                return 0.0, reservation_id, "SLATE_CAP_EXHAUSTED"
            used = dict(exposure["committed"])
            for key, value in external_recommended.items():
                used[key] = used.get(key, 0.0) + value
            for row in rows:
                fraction = float(row["stake_dollars"]) / bank
                keys = {"total", "daily", "weekly", f"sport:{row['sport']}",
                        f"game:{row['sport']}:{row['game_id']}"}
                keys.update(f"team:{row['sport']}:{team}" for team in json.loads(row["team_ids"]))
                for key in keys:
                    used[key] = used.get(key, 0.0) + fraction
            keys = {"total": exposure["total_cap"], "daily": exposure["daily_cap"],
                    "weekly": exposure["weekly_cap"],
                    f"game:{sport}:{game_id}": exposure["game_cap"]}
            keys.update({f"team:{sport}:{team}": exposure["team_cap"] for team in team_ids})
            self_fraction = float(existing["stake_dollars"]) / bank if existing else 0.0
            remaining = min(limit - used.get(key, 0.0) + self_fraction
                            for key, limit in keys.items())
            cap_amount = math.floor(max(0.0, min(requested, remaining * bank)) * 100 + 1e-9) / 100
            if existing:
                old = float(existing["stake_dollars"])
                if cap_amount + 0.001 < old:
                    db.rollback()
                    return 0.0, reservation_id, "EXISTING_RESERVATION_EXCEEDS_CURRENT_CAP"
                db.commit()
                return old, reservation_id, "RESERVATION_REUSED"
            if cap_amount <= 0:
                db.rollback()
                return 0.0, reservation_id, "EXPOSURE_CAP_EXHAUSTED"
            db.execute("""INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                       (reservation_id, at.astimezone(timezone.utc).isoformat(),
                        consent["event_id"], slate_key, sport, game_id,
                        json.dumps(list(team_ids)), cap_amount, bank))
            db.commit()
            return cap_amount, reservation_id, "RESERVED"
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
        return 0.0, None, "RESERVATION_LEDGER_UNAVAILABLE"
