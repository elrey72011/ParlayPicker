"""Immutable Google Drive replication of evidence records; SQLite remains the local cache.

Remote restore merges records, never replaces a database. Credentials use a Google service account and are never included in evidence or diagnostics.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import threading

from app_core.evidence_config import safe_error, EvidenceStorageError
from app_core.performance_spans import PerformanceSpan, opaque_hash, operation_ids

TABLES = {
    "validation_plans": ("plan_id", "sport", "payload"),
    "closing_observations": ("observation_id", "snapshot_id", "candidate_id", "payload"),
    "bundles": ("version", "frozen_at", "manifest"),
    "snapshots": ("snapshot_id", "version", "generated_at", "candidates", "decisions", "inputs", "payload_hash"),
    "snapshot_runtime": ("snapshot_id", "process_instance"),
    "score_revisions": ("snapshot_id", "evidence_hash", "recorded_at", "scores"),
}
KEYS = {"validation_plans": (0,),"closing_observations": (0,),"bundles": (0,), "snapshots": (0,), "snapshot_runtime": (0,), "score_revisions": (0, 1)}
_lock = threading.RLock()
_restored = set()
_status = {}
# Process-local receipts only: restored or read-back-verified immutable bytes.
# Explicit sync remains a full remote verification and can detect external edits.
_verified = {}


def _shared_inventory_enabled():
    return os.environ.get("PARLAYPICKER_SHARED_INVENTORY", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _database_generation(path):
    """Process-local file identity that changes when a DB is replaced in place."""
    try:
        stat = Path(path).stat()
    except OSError:
        return "missing"
    inode = getattr(stat, "st_ino", 0)
    device = getattr(stat, "st_dev", 0)
    # st_ino/st_dev are stable across ordinary SQLite writes. On a filesystem
    # without a usable file ID, ctime is a conservative invalidation fallback.
    return opaque_hash(device, inode) if inode else opaque_hash(device, stat.st_ctime_ns)


def _scope(path):
    from app_core.prediction_evidence import database_path
    resolved = Path(path or database_path()).resolve()
    return (*settings(), str(resolved), _database_generation(resolved))


def _receipt(table, row):
    return (_key(table, row), hashlib.sha256(_encode(table, row)).hexdigest())


def settings():
    return os.environ.get("PARLAYPICKER_DRIVE_FOLDER_ID", "").strip(), os.environ.get("PARLAYPICKER_DRIVE_PREFIX", "parlaypicker/evidence-v1").strip("/")


def remote_status():
    bucket, _ = settings()
    return {"provider": "google_workspace_shared_drive", "configured": bool(bucket), "status": "not_configured" if not bucket else _status.get("status", "not_checked"),
            "restored_snapshots": _status.get("restored_snapshots", 0) if bucket else 0,
            "last_success_at": _status.get("last_success_at") if bucket else None,
            "error": _status.get("error") if bucket else None,
            "operation": _status.get("operation") if bucket else None}


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
    try:
        item = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EvidenceStorageError("Remote evidence file is not valid JSON; preserve the file for inspection.") from None
    if not isinstance(item, dict):
        raise EvidenceStorageError("Remote evidence file must contain a JSON object")
    if item.get("schema") != 1 or item.get("table") != table:
        raise EvidenceStorageError("Unexpected remote evidence schema")
    row = item.get("row")
    if not isinstance(row, list) or len(row) != len(TABLES[table]) or not all(isinstance(v, str) for v in row):
        raise EvidenceStorageError("Invalid remote evidence record")
    if table == "validation_plans":
        from core.exposure_ledger import digest
        value=json.loads(row[2])
        if digest({k:v for k,v in value.items() if k!="plan_hash"}) != row[0]:
            raise EvidenceStorageError("Development plan hash mismatch")
    if table == "closing_observations":
        from core.exposure_ledger import digest
        if digest(json.loads(row[3])) != row[0]:
            raise EvidenceStorageError("Closing observation hash mismatch")
    if table == "bundles" and hashlib.sha256(row[2].encode()).hexdigest() != row[0]:
        raise EvidenceStorageError("Remote model manifest hash mismatch")
    if table == "snapshots" and hashlib.sha256("\0".join(row[3:6]).encode()).hexdigest() != row[6]:
        raise EvidenceStorageError("Remote prediction hash mismatch")
    if table == "score_revisions" and hashlib.sha256(row[3].encode()).hexdigest() != row[1]:
        raise EvidenceStorageError("Remote score hash mismatch")
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
            raise EvidenceStorageError("Remote immutable record conflicts with local evidence")
    # Read back the object: successful PUT alone is not the verification result.
    verified = _decode(_get(client, key), table)
    if verified != tuple(row) and not (table == "score_revisions" and verified[:2] == tuple(row[:2]) and verified[3] == row[3]):
        raise EvidenceStorageError("Remote read-back differs from local evidence")
    return verified


def register_bundle(version, frozen, manifest):
    """Concurrent fresh instances adopt the same first remote freeze timestamp."""
    if not settings()[0]:
        return frozen
    return _put(_client(), "bundles", (version, frozen, manifest), choose_existing=True)[1]


def restore(path=None, *, client=None, full_verification=False, ids=None):
    from app_core.prediction_evidence import connect, database_path
    if not settings()[0]:
        return 0
    client = client or _client()
    bucket, prefix = settings()
    database = Path(path or database_path()).resolve()
    ids = ids or operation_ids(refresh_run_id=os.urandom(8).hex())
    rows = {table: [] for table in TABLES}
    prefixes = {table: f"{prefix}/{table}/" for table in TABLES}
    scope_hash = opaque_hash(bucket, prefix)
    with PerformanceSpan("evidence_restore", ids=ids, database_generation=_database_generation(database),
                         storage_scope_hash=scope_hash) as restore_span:
        optimized = (_shared_inventory_enabled()
                     and callable(getattr(client, "discover_complete_inventory", None))
                     and callable(getattr(client, "read_verified_prefixes", None)))
        if optimized:
            _status["operation"] = "restore:discover"
            inventory = client.discover_complete_inventory(
                operation_id=ids["action_id"], namespace=prefix, ids=ids)
            objects_by_prefix = client.read_verified_prefixes(
                Prefixes=list(prefixes.values()), inventory=inventory,
                cache_dir=database.with_name(database.name + ".remote-cache") / inventory.scope_hash,
                full_verify=full_verification, ids=ids)
            read_report = getattr(client, "last_read_report", None)
            if read_report is not None:
                _status["read_report"] = dict(read_report.__dict__)
        else:
            objects_by_prefix = {}
            for table, table_prefix in prefixes.items():
                if callable(getattr(client, "read_objects", None)):
                    objects_by_prefix[table_prefix] = client.read_objects(Prefix=table_prefix)
                else:
                    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=table_prefix)
                    objects_by_prefix[table_prefix] = [
                        (item["Key"], _get(client, item["Key"]))
                        for page in pages for item in page.get("Contents", [])]
        for table, table_prefix in prefixes.items():
            _status["operation"] = f"restore:{table}"
            with PerformanceSpan("evidence_decode_validate", ids=ids,
                                 database_generation=_database_generation(database),
                                 storage_scope_hash=scope_hash, table_or_kind=table) as table_span:
                for key, raw in objects_by_prefix[table_prefix]:
                    row = _decode(raw, table)
                    if key != _key(table, row):
                        raise EvidenceStorageError("Remote record key does not match its identity")
                    rows[table].append(row)
                table_span.set(records_returned=len(rows[table]), verification_status="payload_verified")
        restore_span.set(
            listing_traversals=(getattr(getattr(client, "last_read_report", None), "listing_traversals", None)
                                if optimized else len(TABLES)),
            listing_pages=getattr(getattr(client, "last_read_report", None), "listing_pages", None),
            metadata_items_seen=getattr(getattr(client, "last_read_report", None), "metadata_items_seen", None),
            objects_matched=sum(len(value) for value in objects_by_prefix.values()),
            objects_downloaded=getattr(getattr(client, "last_read_report", None), "objects_downloaded", None),
            objects_reused=getattr(getattr(client, "last_read_report", None), "objects_reused", None),
            bytes_downloaded=getattr(getattr(client, "last_read_report", None), "bytes_downloaded", None),
            cache_hits=getattr(getattr(client, "last_read_report", None), "cache_hits", None),
            cache_misses=getattr(getattr(client, "last_read_report", None), "cache_misses", None),
            records_returned=sum(len(entries) for entries in rows.values()),
            verification_status="full_bytes_verified" if full_verification else "verified")
    imported = 0
    unchanged = 0
    with PerformanceSpan("evidence_local_merge", ids=ids, database_generation=_database_generation(database),
                         storage_scope_hash=scope_hash) as merge_span:
        with closing(connect(database)) as db, db:
            for table, entries in rows.items():
                if table == "score_revisions":
                    entries.sort(key=lambda row: (row[2], row[1]))
                for row in entries:
                    where = " AND ".join(f"{TABLES[table][i]}=?" for i in KEYS[table])
                    identity = tuple(row[i] for i in KEYS[table])
                    existing = db.execute(f"SELECT {','.join(TABLES[table])} FROM {table} WHERE {where}", identity).fetchone()
                    same_score = existing and table == "score_revisions" and existing[:2] == row[:2] and existing[3] == row[3]
                    if existing and existing != row and not same_score:
                        raise EvidenceStorageError("Restore conflicts with immutable local evidence")
                    if not existing:
                        db.execute(f"INSERT INTO {table} ({','.join(TABLES[table])}) VALUES ({','.join('?' for _ in row)})", row)
                        imported += int(table == "snapshots")
                    else:
                        unchanged += 1
        merge_span.set(records_imported=imported, records_unchanged=unchanged,
                       records_returned=sum(len(entries) for entries in rows.values()),
                       verification_status="transaction_committed")
    with _lock:
        _verified.setdefault(_scope(path), set()).update(
            _receipt(table, row) for table, entries in rows.items() for row in entries)
    _status.update(status="restored", restored_snapshots=_status.get("restored_snapshots", 0) + imported, error=None)
    return imported


def restore_once(path=None, *, full_verification=False):
    from app_core.prediction_evidence import database_path
    bucket, prefix = settings()
    if not bucket:
        return
    identity = _scope(path)
    with _lock:
        if full_verification or identity not in _restored:
            try:
                restore(path, full_verification=full_verification)
                _restored.add(_scope(path))
            except Exception as exc:
                _status.update(status="error", error=safe_error(exc, "Restore"))
                raise RuntimeError(_status["error"]) from None


def sync(path=None, *, client=None, incremental=False):
    from app_core.prediction_evidence import connect, database_path, now_utc
    if not settings()[0]:
        return False
    with _lock:
        scope = _scope(path)
        verified = _verified.setdefault(scope, set())
        try:
            client = client or _client()
            with closing(connect(path or database_path())) as db:
                db.execute("BEGIN")
                records = {table: db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall() for table, columns in TABLES.items()}
            for table, rows in records.items():
                for row in rows:
                    receipt = _receipt(table, row)
                    if incremental and receipt in verified:
                        continue
                    _status["operation"] = f"upload_verify:{table}"
                    _put(client, table, row)
                    verified.add(receipt)
            _status.update(status="synced", last_success_at=now_utc(), error=None, operation="complete")
            return True
        except Exception as exc:
            if not incremental:
                verified.clear()  # A failed full audit invalidates this scope's cached receipts.
            # Local evidence survives a network outage; the UI explicitly shows it
            # is pending replication and retries on the next capture or button.
            _status.update(status="error", error=safe_error(exc, "Backup") + " Local evidence is saved; retry synchronization after correcting the error.")
            return False
