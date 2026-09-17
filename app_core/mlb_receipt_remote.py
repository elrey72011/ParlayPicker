"""Immutable full-store receipt backups with verified, transactional recovery."""
import json
from contextlib import closing
from app_core import mlb_pregame_receipts as r
from app_core.mlb_receipt_audit import backup_bundle
from app_core.mlb_spread_total_model import digest, canonical, receipt_features, prepare_rows

PREFIX = "parlaypicker/mlb-receipt-backup-v1/"


def restore(bundle, path=None):
    p = bundle["payload"]
    if p.get("schema") != "mlb-receipt-backup-v1" or digest(p) != bundle.get("sha256"):
        raise ValueError("Receipt backup hash/schema mismatch")
    tables = p["tables"]
    if set(tables) != {"observations", "receipts", "outcomes"}:
        raise ValueError("Receipt backup tables mismatch")
    for key, observation in tables["observations"].items():
        if digest(observation) != key:
            raise ValueError("Observation hash mismatch")
    events = set()
    for key, snapshot in tables["receipts"].items():
        payload, _ = receipt_features(snapshot)
        event = r.event_key(payload)
        events.add(event)
        if key != r.receipt_key(payload):
            raise ValueError("Receipt identity mismatch")
        if any(ref not in tables["observations"] for ref in payload["source_observations"].values()):
            raise ValueError("Receipt observation missing")
        outcome = tables["outcomes"].get(event)
        if outcome is not None:
            prepare_rows([{"snapshot": snapshot, "outcome": outcome}])
    for key, outcome in tables["outcomes"].items():
        if key not in events or key != r.event_key(outcome) or outcome.get("observation_hash") not in tables["observations"]:
            raise ValueError("Outcome provenance missing")
    # Recovery copies previously recorded bytes; it never records a new pregame
    # observation or rewrites a first receipt. Any local conflict rolls back all.
    count = 0
    with closing(r.connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        for table, records in tables.items():
            for key, payload in records.items():
                raw, sha = canonical(payload).decode(), digest(payload)
                old = db.execute(f"SELECT sha256,payload FROM {table} WHERE id=?", (key,)).fetchone()
                if old is not None:
                    if old != (sha, raw):
                        raise ValueError("Receipt recovery conflict")
                    continue
                db.execute(f"INSERT INTO {table} VALUES (?,?,?)", (key, sha, raw))
                count += 1
    return count


def connection():
    from app_core.evidence_remote import settings
    from app_core.evidence_drive import DriveStore
    folder = settings()[0]
    return DriveStore(folder), folder


def recover(client, path=None):
    count = 0
    for key, raw in client.read_objects(Prefix=PREFIX):
        if len(raw) > 40_000_000:
            raise ValueError("Receipt backup too large")
        bundle = json.loads(raw)
        if key != PREFIX + digest(bundle) + ".json":
            raise ValueError("Receipt remote key mismatch")
        count += restore(bundle, path)
    return count


def backup(client, folder, path=None):
    from app_core.evidence_drive import AlreadyExists
    bundle = backup_bundle(path)
    raw = canonical(bundle)
    if len(raw) > 40_000_000:
        raise ValueError("Receipt backup too large")
    key = PREFIX + digest(bundle) + ".json"
    try:
        client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    except AlreadyExists:
        pass
    with client.get_object(Bucket=folder, Key=key)["Body"] as stream:
        if stream.read(40_000_001) != raw:
            raise ValueError("Receipt backup read-back failed")
    return {"remote_backup_verified": True, "backup_id": digest(bundle)}


def collect_durable(games):
    client, folder = connection()
    restored = recover(client)
    games, health = r.capture_live_games(games)
    # Preserve new pregame receipts before any outcome network work.
    health.update(backup(client, folder))
    health["records_restored"] = restored
    try:
        health["reconciliation"] = r.reconcile(max_games=10)
    finally:
        health.update(backup(client, folder))
    return games, health
