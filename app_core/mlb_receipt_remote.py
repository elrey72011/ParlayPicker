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


RECORD_PREFIX = "parlaypicker/mlb-receipt-records-v2/"
MANIFEST_PREFIX = "parlaypicker/mlb-receipt-manifests-v2/"
MAX_OBJECT_BYTES = 40_000_000


def recover(client, path=None):
    count = 0
    for key, raw in client.read_objects(Prefix=PREFIX):
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Legacy receipt backup too large")
        bundle = json.loads(raw)
        if key != PREFIX + digest(bundle) + ".json":
            raise ValueError("Receipt remote key mismatch")
        count += restore(bundle, path)
    records = {}
    for key, raw in client.read_objects(Prefix=RECORD_PREFIX):
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Receipt record too large")
        value = json.loads(raw)
        sha = digest(value)
        if key != RECORD_PREFIX + sha + ".json":
            raise ValueError("Receipt record hash mismatch")
        records[sha] = value
    for key, raw in client.read_objects(Prefix=MANIFEST_PREFIX):
        manifest = json.loads(raw)
        if len(raw) > MAX_OBJECT_BYTES or key != MANIFEST_PREFIX + digest(manifest) + ".json" or manifest.get("schema") != "mlb-receipt-manifest-v2":
            raise ValueError("Receipt manifest mismatch")
        tables = {}
        for table, refs in manifest["tables"].items():
            tables[table] = {}
            for record_id, sha in refs.items():
                value = records.get(sha)
                if value is None or value.get("table") != table or value.get("id") != record_id:
                    raise ValueError("Receipt manifest record missing or mismatched")
                tables[table][record_id] = value["payload"]
        payload = {"schema": "mlb-receipt-backup-v1", "tables": tables}
        count += restore({"payload": payload, "sha256": digest(payload)}, path)
    return count


def backup(client, folder, path=None):
    from app_core.evidence_drive import AlreadyExists

    def verified_put(prefix, value):
        raw = canonical(value)
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Receipt individual object too large")
        sha = digest(value)
        key = prefix + sha + ".json"
        try:
            client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
        except AlreadyExists:
            pass
        with client.get_object(Bucket=folder, Key=key)["Body"] as stream:
            if stream.read(MAX_OBJECT_BYTES + 1) != raw:
                raise ValueError("Receipt backup read-back failed")
        return sha

    bundle = backup_bundle(path)
    manifest = {"schema": "mlb-receipt-manifest-v2", "tables": {}}
    for table, records in bundle["payload"]["tables"].items():
        manifest["tables"][table] = {}
        for key, payload in records.items():
            manifest["tables"][table][key] = verified_put(RECORD_PREFIX,
                {"table": table, "id": key, "payload": payload})
    # Publish the manifest only after every referenced object passed read-back.
    sha = verified_put(MANIFEST_PREFIX, manifest)
    return {"remote_backup_verified": True, "backup_id": sha, "backup_format": "records-v2"}


def collect_durable(games):
    stage = "connect"
    health = {"receipts_created": 0, "receipts_skipped": len(games)*4}
    try:
        client, folder = connection()
        stage = "restore"
        restored = recover(client)
        stage = "capture"
        games, health = r.capture_live_games(games)
        health["records_restored"] = restored
        stage = "backup_before_reconciliation"
        health.update(backup(client, folder))
        stage = "reconcile"
        health["reconciliation"] = r.reconcile(max_games=10)
        stage = "backup_after_reconciliation"
        health.update(backup(client, folder))
    except Exception as exc:
        # Report only stage and class; provider exception text can contain secrets.
        health.update(remote_backup_verified=False, failed_stage=stage, error_type=type(exc).__name__)
        health.setdefault("reasons", {})["receipt_" + stage + "_failed"] = 1
    return games, health
