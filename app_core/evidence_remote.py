"""Immutable Google Drive replication of evidence records; SQLite remains the local cache.

Remote restore merges records, never replaces a database. Credentials use a Google service account and are never included in evidence or diagnostics.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import threading

from app_core.evidence_config import safe_error

TABLES = {
    "bundles": ("version", "frozen_at", "manifest"),
    "snapshots": ("snapshot_id", "version", "generated_at", "candidates", "decisions", "inputs", "payload_hash"),
    "snapshot_runtime": ("snapshot_id", "process_instance"),
    "score_revisions": ("snapshot_id", "evidence_hash", "recorded_at", "scores"),
}
KEYS = {"bundles": (0,), "snapshots": (0,), "snapshot_runtime": (0,), "score_revisions": (0, 1)}
_lock = threading.RLock()
_restored = set()
_status = {}


def settings():
    return os.environ.get("PARLAYPICKER_DRIVE_FOLDER_ID", "").strip(), os.environ.get("PARLAYPICKER_DRIVE_PREFIX", "parlaypicker/evidence-v1").strip("/")


def remote_status():
    bucket, _ = settings()
    return {"provider": "google_workspace_shared_drive", "configured": bool(bucket), "status": "not_configured" if not bucket else _status.get("status", "not_checked"),
            "restored_snapshots": _status.get("restored_snapshots", 0) if bucket else 0,
            "last_success_at": _status.get("last_success_at") if bucket else None,
            "error": _status.get("error") if bucket else None}


def _client():
    from app_core.evidence_drive import DriveStore
    return DriveStore(settings()[0])


def _encode(table, row):
    return json.dumps({"schema": 1, "table": table, "row": list(row)}, sort_keys=True, separators=(",", ":")).encode()


def _key(table, row):
    _, prefix = settings()
    identity = hashlib.sha256(json.dumps([row[i] for i in KEYS[table]], separators=(",", ":")).encode()).hexdigest()
    return f"{prefix}/{table}/{identity}.json"


def _decode(raw, table):
    item = json.loads(raw)
    if item.get("schema") != 1 or item.get("table") != table:
        raise ValueError("Unexpected remote evidence schema")
    row = item.get("row")
    if not isinstance(row, list) or len(row) != len(TABLES[table]) or not all(isinstance(v, str) for v in row):
        raise ValueError("Invalid remote evidence record")
    if table == "bundles" and hashlib.sha256(row[2].encode()).hexdigest() != row[0]:
        raise ValueError("Remote model manifest hash mismatch")
    if table == "snapshots" and hashlib.sha256("\0".join(row[3:6]).encode()).hexdigest() != row[6]:
        raise ValueError("Remote prediction hash mismatch")
    if table == "score_revisions" and hashlib.sha256(row[3].encode()).hexdigest() != row[1]:
        raise ValueError("Remote score hash mismatch")
    return tuple(row)


def _get(client, key):
    bucket, _ = settings()
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return body.read()
    finally:
        body.close()


def _put(client, table, row, *, choose_existing=False):
    bucket, _ = settings()
    key, raw = _key(table, row), _encode(table, row)
    _decode(raw, table)
    try:
        client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        existing = _decode(_get(client, key), table)
        if choose_existing and table == "bundles" and existing[2] == row[2]:
            return existing
        # Idempotent score corrections may be observed at different times.
        same_score = table == "score_revisions" and existing[:2] == tuple(row[:2]) and existing[3] == row[3]
        if existing != tuple(row) and not same_score:
            raise ValueError("Remote immutable record conflicts with local evidence")
    # Read back the object: successful PUT alone is not the verification result.
    verified = _decode(_get(client, key), table)
    if verified != tuple(row) and not (table == "score_revisions" and verified[:2] == tuple(row[:2]) and verified[3] == row[3]):
        raise ValueError("Remote read-back differs from local evidence")
    return verified


def register_bundle(version, frozen, manifest):
    """Concurrent fresh instances adopt the same first remote freeze timestamp."""
    if not settings()[0]:
        return frozen
    return _put(_client(), "bundles", (version, frozen, manifest), choose_existing=True)[1]


def restore(path=None, *, client=None):
    from app_core.prediction_evidence import connect, database_path
    if not settings()[0]:
        return 0
    client = client or _client()
    bucket, prefix = settings()
    rows = {table: [] for table in TABLES}
    # Table-specific listing preserves referential order; all pages are consumed.
    for table in TABLES:
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/{table}/")
        for page in pages:
            for item in page.get("Contents", []):
                row = _decode(_get(client, item["Key"]), table)
                if item["Key"] != _key(table, row):
                    raise ValueError("Remote record key does not match its identity")
                rows[table].append(row)
    imported = 0
    with closing(connect(path or database_path())) as db, db:
        for table, entries in rows.items():
            if table == "score_revisions":
                entries.sort(key=lambda row: (row[2], row[1]))
            for row in entries:
                where = " AND ".join(f"{TABLES[table][i]}=?" for i in KEYS[table])
                identity = tuple(row[i] for i in KEYS[table])
                existing = db.execute(f"SELECT {','.join(TABLES[table])} FROM {table} WHERE {where}", identity).fetchone()
                same_score = existing and table == "score_revisions" and existing[:2] == row[:2] and existing[3] == row[3]
                if existing and existing != row and not same_score:
                    raise ValueError("Restore conflicts with immutable local evidence")
                if not existing:
                    db.execute(f"INSERT INTO {table} ({','.join(TABLES[table])}) VALUES ({','.join('?' for _ in row)})", row)
                    imported += int(table == "snapshots")
    _status.update(status="restored", restored_snapshots=_status.get("restored_snapshots", 0) + imported, error=None)
    return imported


def restore_once(path=None):
    from app_core.prediction_evidence import database_path
    bucket, prefix = settings()
    if not bucket:
        return
    identity = (bucket, prefix, str(Path(path or database_path()).resolve()))
    with _lock:
        if identity not in _restored:
            try:
                restore(path)
                _restored.add(identity)
            except Exception as exc:
                _status.update(status="error", error=safe_error(exc, "Restore"))
                raise RuntimeError(_status["error"]) from None


def sync(path=None, *, client=None):
    from app_core.prediction_evidence import connect, database_path, now_utc
    if not settings()[0]:
        return False
    with _lock:
        try:
            client = client or _client()
            with closing(connect(path or database_path())) as db:
                db.execute("BEGIN")
                records = {table: db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall() for table, columns in TABLES.items()}
            for table, rows in records.items():
                for row in rows:
                    _put(client, table, row)
            _status.update(status="synced", last_success_at=now_utc(), error=None)
            return True
        except Exception as exc:
            # Local evidence survives a network outage; the UI explicitly shows it
            # is pending replication and retries on the next capture or button.
            _status.update(status="error", error=safe_error(exc, "Backup") + " Local evidence is saved; retry synchronization after correcting the error.")
            return False
