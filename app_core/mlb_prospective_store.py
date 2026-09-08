"""Append-only research records, isolated from approved wager evidence."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from app_core.prediction_evidence import database_path

PREFIX = "parlaypicker/mlb-prospective-v1/"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def connect(path=None):
    path = path or database_path().with_name("mlb-prospective.sqlite3")
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    for action in ("UPDATE", "DELETE"):
        db.execute(f"CREATE TRIGGER IF NOT EXISTS records_{action} BEFORE {action} ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END")
    return db


def insert(record, path=None):
    if record.get("schema") != 1 or record.get("kind") not in ("model", "capture", "scores", "closing"):
        raise ValueError("Invalid prospective record")
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


def sync(path=None, *, client=None, folder=None):
    from app_core.evidence_remote import settings
    from app_core.evidence_drive import DriveStore, AlreadyExists
    folder = folder or settings()[0]
    client = client or DriveStore(folder)
    restored = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder, Prefix=PREFIX):
        for item in page.get("Contents", []):
            with client.get_object(Bucket=folder, Key=item["Key"])["Body"] as body:
                raw = body.read(40_000_001)
            if len(raw) > 40_000_000 or PREFIX + hashlib.sha256(raw).hexdigest() + ".json" != item["Key"]:
                raise ValueError("Prospective backup integrity failure")
            insert(json.loads(raw), path)
            restored += 1
    saved = 0
    for r in records(path):
        key = PREFIX + r["id"] + ".json"
        raw = encode({k: v for k, v in r.items() if k != "id"})
        try:
            client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
        except AlreadyExists:
            pass
        with client.get_object(Bucket=folder, Key=key)["Body"] as body:
            if body.read() != raw:
                raise ValueError("Prospective backup read-back failed")
        saved += 1
    return {"remote_records_read": restored, "records_verified": saved}
