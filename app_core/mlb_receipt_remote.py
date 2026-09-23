"""Immutable full-store receipt backups with verified, transactional recovery."""
import json
from contextlib import closing
from app_core import mlb_pregame_receipts as r
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
    # A fresh remote read establishes verification only for this client/run.
    client._receipt_verified = set()
    verified = set()
    count = 0
    for key, raw in client.read_objects(Prefix=PREFIX):
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Legacy receipt backup too large")
        bundle = json.loads(raw)
        if key != PREFIX + digest(bundle) + ".json":
            raise ValueError("Receipt remote key mismatch")
        count += restore(bundle, path)
    records = {}
    # Cache is expendable and never replaces a fresh remote inventory/checksum.
    # Legacy full backups retain the existing explicit read path.
    def read_v2(prefix):
        if not hasattr(client, 'read_cached_objects'):
            return client.read_objects(Prefix=prefix)
        from pathlib import Path
        db_path = Path(path or r.database_path().with_name('mlb-pregame-receipts.sqlite3'))
        return client.read_cached_objects(Prefix=prefix, cache_dir=db_path.with_suffix('.remote-cache'))
    for key, raw in read_v2(RECORD_PREFIX):
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Receipt record too large")
        value = json.loads(raw)
        sha = digest(value)
        if key != RECORD_PREFIX + sha + ".json":
            raise ValueError("Receipt record hash mismatch")
        if canonical(value) != raw:
            raise ValueError("Receipt record canonical mismatch")
        verified.add(key)
        records[sha] = value
    merged_tables = {t: {} for t in ("observations", "receipts", "outcomes")}
    manifest_count = 0
    for key, raw in read_v2(MANIFEST_PREFIX):
        manifest = json.loads(raw)
        if len(raw) > MAX_OBJECT_BYTES or key != MANIFEST_PREFIX + digest(manifest) + ".json" or manifest.get("schema") != "mlb-receipt-manifest-v2":
            raise ValueError("Receipt manifest mismatch")
        if canonical(manifest) != raw:
            raise ValueError("Receipt manifest canonical mismatch")
        verified.add(key)
        tables = {}
        for table, refs in manifest["tables"].items():
            tables[table] = {}
            for record_id, sha in refs.items():
                value = records.get(sha)
                if value is None or value.get("table") != table or value.get("id") != record_id:
                    raise ValueError("Receipt manifest record missing or mismatched")
                tables[table][record_id] = value["payload"]
        if set(tables) != set(merged_tables):
            raise ValueError("Receipt backup tables mismatch")
        # Verify each manifest's dependency closure before combining snapshots.
        # A later snapshot must not hide a missing dependency in an earlier one.
        events = set()
        for record_id, snapshot in tables["receipts"].items():
            payload, _ = receipt_features(snapshot)
            event = r.event_key(payload)
            events.add(event)
            if record_id != r.receipt_key(payload):
                raise ValueError("Receipt identity mismatch")
            if any(ref not in tables["observations"] for ref in payload["source_observations"].values()):
                raise ValueError("Receipt observation missing")
            if event in tables["outcomes"]:
                prepare_rows([{"snapshot": snapshot, "outcome": tables["outcomes"][event]}])
        for record_id, outcome in tables["outcomes"].items():
            if record_id not in events or record_id != r.event_key(outcome) or outcome.get("observation_hash") not in tables["observations"]:
                raise ValueError("Outcome provenance missing")
        for table, values in tables.items():
            for record_id, payload in values.items():
                old = merged_tables[table].get(record_id)
                if old is not None and old != payload:
                    raise ValueError("Receipt recovery conflict")
                merged_tables[table][record_id] = payload
        manifest_count += 1
    # Restore the union once rather than repeatedly hash and scan every large
    # observation in every cumulative historical manifest.
    if manifest_count:
        payload = {"schema": "mlb-receipt-backup-v1", "tables": merged_tables}
        count += restore({"payload": payload, "sha256": digest(payload)}, path)
    import logging
    logging.getLogger(__name__).warning("RECEIPT RECOVERY manifests=%s unique_records=%s",
        manifest_count, sum(len(v) for v in merged_tables.values()))
    client._receipt_verified = verified
    return count


def backup(client, folder, path=None):
    from app_core.evidence_drive import AlreadyExists
    verified = getattr(client, "_receipt_verified", set())
    counters = {"objects_reused": 0, "objects_uploaded_verified": 0}

    def verified_put(prefix, value):
        raw = canonical(value)
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Receipt individual object too large")
        sha = digest(value)
        key = prefix + sha + ".json"
        if key in verified:
            counters["objects_reused"] += 1
            return sha
        try:
            client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
        except AlreadyExists:
            pass
        with client.get_object(Bucket=folder, Key=key)["Body"] as stream:
            if stream.read(MAX_OBJECT_BYTES + 1) != raw:
                raise ValueError("Receipt backup read-back failed")
        verified.add(key)
        counters["objects_uploaded_verified"] += 1
        return sha

    manifest = {"schema": "mlb-receipt-manifest-v2", "tables": {}}
    # One consistent SQLite snapshot; keep only one decoded raw feed in memory.
    # Do not materialize/hash the entire archive just to upload individual objects.
    with closing(r.connect(path)) as db:
        db.execute("BEGIN")
        for table in ("observations", "receipts", "outcomes"):
            manifest["tables"][table] = {}
            for key, expected, raw in db.execute(f"SELECT id,sha256,payload FROM {table} ORDER BY rowid"):
                payload = json.loads(raw)
                if digest(payload) != expected:
                    raise r.Rejected("stored_hash_mismatch")
                manifest["tables"][table][key] = verified_put(RECORD_PREFIX,
                    {"table": table, "id": key, "payload": payload})
                del payload, raw
    # Publish the manifest only after every referenced object passed read-back.
    sha = verified_put(MANIFEST_PREFIX, manifest)
    client._receipt_verified = verified
    return {"remote_backup_verified": True, "backup_id": sha, "backup_format": "records-v2", **counters}


def collect_durable(games, *, max_feeds=20, reconcile_history=True):
    import logging
    import time
    timings = {}
    def timed(name, operation):
        started = time.monotonic()
        try:
            return operation()
        finally:
            timings[name] = round(time.monotonic() - started, 3)
            logging.getLogger(__name__).warning("PERFORMANCE mlb_receipts stage=%s seconds=%.3f", name, timings[name])
    stage = "connect"
    health = {"receipts_created": 0, "receipts_skipped": len(games)*4}
    try:
        client, folder = timed(stage, connection)
        stage = "restore"
        restored = timed(stage, lambda: recover(client))
        stage = "capture"
        games, health = timed(stage, lambda: r.capture_live_games(games, max_feeds=max_feeds))
        health["records_restored"] = restored
        stage = "backup_before_reconciliation"
        health.update(timed(stage, lambda: backup(client, folder)))
        if reconcile_history:
            stage = "reconcile"
            health["reconciliation"] = timed(stage, lambda: r.reconcile(max_games=10))
            stage = "backup_after_reconciliation"
            health.update(timed(stage, lambda: backup(client, folder)))
        else:
            health["reconciliation_deferred"] = True
    except Exception as exc:
        # Report only stage and class; provider exception text can contain secrets.
        health.update(remote_backup_verified=False, failed_stage=stage, error_type=type(exc).__name__)
        health.setdefault("reasons", {})["receipt_" + stage + "_failed"] = 1
    health["stage_timings_seconds"] = timings
    return games, health


def reconcile_durable(*, path=None, max_games=100):
    """Restore saved receipts, append verified finals, and publish the new state."""
    client, folder = connection()
    restored = recover(client, path)
    reconciliation = r.reconcile(path, max_games=max_games)
    verified = backup(client, folder, path)
    return {"records_restored": restored, "reconciliation": reconciliation, **verified}


def restore_diagnostic(stage, exc):
    """Never expose arbitrary exception text or provider URLs/credentials."""
    known = {
        "Receipt recovery conflict": "LOCAL_RECORD_CONFLICT",
        "Receipt backup hash/schema mismatch": "BACKUP_HASH_OR_SCHEMA_INVALID",
        "Receipt manifest record missing or mismatched": "REMOTE_RECORD_MISSING",
        "Receipt backup read-back failed": "REMOTE_READBACK_MISMATCH",
        "Receipt individual object too large": "INDIVIDUAL_OBJECT_TOO_LARGE",
        "Receipt remote key mismatch": "REMOTE_KEY_MISMATCH",
        "Receipt record hash mismatch": "REMOTE_RECORD_HASH_MISMATCH",
    }
    return {"failed_stage": stage, "error_type": type(exc).__name__,
            "reason": known.get(str(exc), "RESTORE_OPERATION_FAILED"),
            "remote_backup_verified": False}


def catch_up_history():
    """Bounded receipt-only run with fresh quotes, no full analysis or Gemini."""
    from core.streamlit_pipeline import _get_odds_api_key
    from app_core.odds_api import TheOddsAPIClient
    key = _get_odds_api_key()
    if not key:
        raise ValueError("Missing odds API configuration")
    games = TheOddsAPIClient(key, markets="spreads,totals").get_odds("baseball_mlb")
    _, health = collect_durable(games, max_feeds=100)
    return health
