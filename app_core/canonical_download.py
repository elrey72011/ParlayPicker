"""Private, bounded SQLite backup of an existing canonical store; never restore.

Only the authenticated owner UI calls this module. The backup preserves original
record bytes and committed WAL transactions. Its manifest inventories evidence,
not scientific acceptance or wagering authority.
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
import zipfile

from app_core.canonical_schema import CANONICAL_PRIMARY_KEYS, CANONICAL_SCHEMA_VERSION
from app_core.prediction_evidence import database_path

MAX_BYTES = 128 * 1024 * 1024
MAX_SECONDS = 30
MAX_REFERENCES = 10000
NAME = "prospective-evidence.sqlite3"
SECRET_NAME = re.compile(r"(?:api[_-]?key|password|credential|private[_-]?key|secret|authorization|(?:^|_)token(?:$|_))", re.I)
SECRET_FIELD = re.compile(r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|credentials?|password|private[_-]?key|client[_-]?secret|authorization)", re.I)
SECRET_BYTES = re.compile(
    rb'''(?:["'](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|credential|password|private[_-]?key|client[_-]?secret|authorization)["']\s*[:=]\s*["'][^"']+|[?&](?:api[_-]?key|access[_-]?token|token|password)=[^&\s"']+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^/\s:@]+:[^/\s@]+@)''', re.I)
REFERENCE_FIELDS = frozenset({
    "model_id", "model_version", "artifact_hash", "feature_version", "feature_snapshot_id",
    "feature_frozen_at", "model_available_at", "model_trained_through", "training_code_commit",
    "training_cutoff", "training_start", "available_at", "observed_at", "created_at",
    "source_id", "source_key", "source_hash", "payload_hash", "evidence_snapshot_id",
    "evidence_hash", "runtime_hash", "source_commit", "result_id", "quote_id", "event_id",
    "calibration_id", "calibration_version", "artifact_id", "sport", "market_family",
    "prediction_timestamp", "feature_dependencies", "dependency_receipts", "dependencies",
})


class DownloadUnavailable(ValueError):
    """Fixed, credential-free reason code; never wrap a raw storage exception."""


def effective_path():
    return database_path().with_name(NAME)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reference(value):
    # Only named references enter the manifest; raw dependencies stay in the
    # private database. Absence of a receipt is never inferred as availability.
    if isinstance(value, dict):
        return {k: _reference(v) for k, v in value.items() if k in REFERENCE_FIELDS}
    if isinstance(value, list):
        return [_reference(v) for v in value]
    return value


def _credential_object(value):
    if isinstance(value, dict):
        return any((SECRET_FIELD.fullmatch(str(k)) and v not in (None, "", [], {}))
                   or _credential_object(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_credential_object(v) for v in value)
    if isinstance(value, str):
        if SECRET_BYTES.search(value.encode()):
            return True
        # JSON embedded inside a retained string can escape credential keys.
        if value.lstrip().startswith(("{", "[")):
            try:
                return _credential_object(json.loads(value))
            except (ValueError, TypeError):
                pass
    return False


def build_download(path=None, *, forbidden_values=(), max_bytes=MAX_BYTES,
                   max_seconds=MAX_SECONDS, max_references=MAX_REFERENCES):
    """Return (ZIP bytes, manifest) without initializing or writing the source.

    mode=ro deliberately includes WAL; immutable=1 is used only for the closed
    destination. A bounded backup callback aborts busy/concurrently growing
    stores. Credentials cause refusal, never redaction of historical records.
    """
    target = Path(path) if path is not None else effective_path()
    if target.name != NAME:
        raise DownloadUnavailable("WRONG_STORE")
    try:
        exists = target.is_file()
    except OSError:
        raise DownloadUnavailable("LOCAL_CANONICAL_STORE_INACCESSIBLE_REMOTE_UNKNOWN") from None
    if not exists:
        raise DownloadUnavailable("LOCAL_CANONICAL_STORE_MISSING_REMOTE_UNKNOWN")
    started = time.monotonic()
    prepared_at = datetime.now(timezone.utc).isoformat()
    secrets = [str(v).encode() for v in forbidden_values if v]
    secrets.extend(v.encode() for k, v in os.environ.items() if v and SECRET_NAME.search(k))

    def bounded():
        if time.monotonic() - started > max_seconds:
            raise DownloadUnavailable("SNAPSHOT_TIME_LIMIT")

    def progress(status, remaining, total):
        bounded()
        if total * page_size > max_bytes:
            raise DownloadUnavailable("SNAPSHOT_SIZE_LIMIT")

    try:
        with tempfile.TemporaryDirectory(prefix="private-canonical-backup-") as directory:
            snapshot = Path(directory) / NAME
            with closing(sqlite3.connect(target.resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as source:
                source.execute("PRAGMA query_only=ON")
                page_size = source.execute("PRAGMA page_size").fetchone()[0]
                if source.execute("PRAGMA page_count").fetchone()[0] * page_size > max_bytes:
                    raise DownloadUnavailable("SNAPSHOT_SIZE_LIMIT")
                with closing(sqlite3.connect(snapshot)) as destination:
                    source.backup(destination, pages=128, progress=progress, sleep=0.05)
            bounded()
            if snapshot.stat().st_size > max_bytes:
                raise DownloadUnavailable("SNAPSHOT_SIZE_LIMIT")
            raw = snapshot.read_bytes()
            if SECRET_BYTES.search(raw) or any(secret in raw for secret in secrets):
                raise DownloadUnavailable("CREDENTIAL_MATERIAL_DETECTED")
            with closing(sqlite3.connect(snapshot.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
                db.row_factory = sqlite3.Row
                db.set_progress_handler(lambda: int(time.monotonic()-started > max_seconds), 1000)
                if db.execute("PRAGMA quick_check").fetchall()[0][0] != "ok":
                    raise DownloadUnavailable("SNAPSHOT_INTEGRITY_FAILED")
                objects = [dict(row) for row in db.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
                tables = [row["name"] for row in objects if row["type"] == "table"]
                if not tables or "prospective_event" not in tables or any(t not in CANONICAL_PRIMARY_KEYS for t in tables):
                    raise DownloadUnavailable("CANONICAL_SCHEMA_UNSUPPORTED")
                counts, schema, references = {}, {}, []
                reference_count = 0
                for table in sorted(tables):
                    bounded()
                    details = [dict(row) for row in db.execute('PRAGMA table_info("'+table+'")')]
                    primary = tuple(row["name"] for row in sorted(details, key=lambda row: row["pk"]) if row["pk"])
                    if primary != CANONICAL_PRIMARY_KEYS[table]:
                        raise DownloadUnavailable("CANONICAL_SCHEMA_UNSUPPORTED")
                    schema[table] = details
                    counts[table] = db.execute('SELECT count(*) FROM "'+table+'"').fetchone()[0]
                    for row in db.execute('SELECT * FROM "'+table+'" ORDER BY '+','.join('"'+p+'"' for p in primary)):
                        bounded()
                        for column in row.keys():
                            value = row[column]
                            if SECRET_FIELD.fullmatch(column) and value not in (None, "", b""):
                                raise DownloadUnavailable("CREDENTIAL_MATERIAL_DETECTED")
                            if isinstance(value, (str, bytes)):
                                try:
                                    retained = json.loads(value)
                                except (ValueError, TypeError, UnicodeDecodeError):
                                    continue
                                if _credential_object(retained):
                                    raise DownloadUnavailable("CREDENTIAL_MATERIAL_DETECTED")
                        item = {k: row[k] for k in row.keys() if k in REFERENCE_FIELDS}
                        if "payload" in row.keys():
                            try:
                                payload = json.loads(row["payload"])
                                if isinstance(payload, dict):
                                    item["payload_references"] = _reference(payload)
                            except (ValueError, TypeError):
                                item["payload_reference_status"] = "UNREADABLE_ORIGINAL_PRESERVED"
                        if item:
                            reference_count += 1
                            if len(references) < max_references:
                                references.append(dict(table=table, identity={p: row[p] for p in primary}, references=item))
                manifest = dict(schema_version=1, kind="PRIVATE_CANONICAL_SQLITE_BACKUP",
                    prepared_at=prepared_at, completed_at=datetime.now(timezone.utc).isoformat(),
                    source_filename=NAME, source_configuration="PARLAYPICKER_EVIDENCE_DIR" if "PARLAYPICKER_EVIDENCE_DIR" in os.environ else "APPLICATION_ROOT_DEFAULT",
                    snapshot=dict(filename=NAME, sha256=_sha(raw), bytes=len(raw)),
                    canonical_contract_version=CANONICAL_SCHEMA_VERSION,
                    sqlite_user_version=db.execute("PRAGMA user_version").fetchone()[0],
                    schema_objects=objects, schema_sha256=_sha(_json(objects)), tables=schema,
                    record_counts=counts, absent_contract_tables=sorted(set(CANONICAL_PRIMARY_KEYS)-set(tables)),
                    available_references=references, reference_rows=reference_count,
                    references_truncated=reference_count > len(references),
                    external_reference_contents="NOT_INCLUDED_AVAILABILITY_UNKNOWN",
                    backup_method="sqlite3.Connection.backup; committed WAL included; uncommitted transactions excluded",
                    original_record_bytes_preserved=True, source_mutated=False,
                    scientific_acceptance=False, wagering_authority=False)
            manifest_raw = _json(manifest)
            if SECRET_BYTES.search(manifest_raw) or any(secret in manifest_raw for secret in secrets):
                raise DownloadUnavailable("CREDENTIAL_MATERIAL_DETECTED")
            bounded()
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(NAME, raw)
                archive.writestr("manifest.json", manifest_raw)
            bounded()
            return output.getvalue(), manifest
    except DownloadUnavailable:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, RecursionError):
        bounded()
        raise DownloadUnavailable("SNAPSHOT_STORAGE_OR_INTEGRITY_FAILURE") from None
