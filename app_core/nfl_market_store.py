"""Append-only research records, isolated from approved wager evidence."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from app_core.prediction_evidence import database_path

PREFIX = "parlaypicker/nfl-market-v1/"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def connect(path=None):
    path = path or database_path().with_name("nfl-market.sqlite3")
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    for action in ("UPDATE", "DELETE"):
        db.execute(f"CREATE TRIGGER IF NOT EXISTS records_{action} BEFORE {action} ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END")
    return db


def insert(record, path=None):
    if record.get("schema") != 1 or record.get("kind") not in ("capture", "scores"):
        raise ValueError("Invalid prospective record")
    if record.get("data", {}).get("sport") != "NFL":
        raise ValueError("Invalid NFL record")
    raw = encode(record)
    key = hashlib.sha256(raw).hexdigest()
    with closing(connect(path)) as db, db:
        db.execute("INSERT OR IGNORE INTO records VALUES (?, ?)", (key, raw.decode()))
    return key


def save(kind, data, path=None):
    created = datetime.now(timezone.utc).isoformat()
    if kind in ("capture", "closing"):
        from app_core.ncaaf_history import timestamp
        data = dict(data)
        data["events"] = [e for e in data["events"] if timestamp(e["start"]) > timestamp(created)]
    return insert({"schema": 1, "kind": kind, "created_at": created, "data": data}, path)


def records(path=None):
    with closing(connect(path)) as db:
        result = []
        for key, raw in db.execute("SELECT id,payload FROM records ORDER BY rowid"):
            if hashlib.sha256(raw.encode()).hexdigest() != key:
                raise ValueError("Prospective record integrity failure")
            result.append({"id": key, **json.loads(raw)})
        return sorted(result, key=lambda r: (r["created_at"], r["id"]))


def sync(path=None, *, client=None, folder=None, session=None):
    from app_core.evidence_remote import settings
    from app_core.evidence_drive import DriveStore
    from app_core.prospective_sync import sync_records
    import sys
    folder = folder or settings()[0]
    client = client or DriveStore(folder)
    return sync_records(sys.modules[__name__], path, client, folder, session)
