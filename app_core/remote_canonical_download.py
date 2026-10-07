"""Bounded read-only canonical JSON export. Never connect to or hydrate SQLite.

Reuse Drive's pagination/checksum/duplicate protections and the canonical wire
decoder. Original media bytes and every remote file ID survive in the ZIP.
Partial exports are evidence inventories, never complete dependency chains.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
import zipfile

from app_core import canonical_download as private
from app_core.canonical_remote_contract import SCHEMA, DEPENDENCIES
from app_core.canonical_schema import CANONICAL_SCHEMA_VERSION
from app_core.evidence_config import EvidenceStorageError
from app_core.evidence_drive import API, DriveInventory, DriveStore, _checksum, _read
from app_core.prospective_remote import PREFIX, MAX_OBJECT_BYTES, _decode


@dataclass(frozen=True)
class Limits:
    max_pages: int = 100
    max_metadata: int = 100000
    max_objects: int = 50000
    max_bytes: int = 128 * 1024 * 1024
    max_object_bytes: int = MAX_OBJECT_BYTES
    max_seconds: float = 300
    max_requests: int = 100000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("INVALID_EXPORT_LIMIT")
            if name != "max_seconds" and type(value) is not int:
                raise ValueError("INVALID_EXPORT_LIMIT")
            if value > self.__dataclass_fields__[name].default:
                raise ValueError("INVALID_EXPORT_LIMIT")


class ExportUnavailable(ValueError):
    """Only fixed reason codes may leave the private reader."""


class _BoundedStop(Exception):
    pass


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.started = time.monotonic()
        self.requests = self.received = 0
        self.stats = {"listing_pages": 0, "metadata_items_seen": 0}

    def check(self):
        if time.monotonic() - self.started >= self.limits.max_seconds:
            raise _BoundedStop("TIME_LIMIT")


class _Session:
    """Read-only request boundary around the existing Drive session.

    DriveStore._files owns pagination, incompleteSearch and token validation.
    No POST/PATCH/DELETE methods are exposed. Per-request timeout is bounded by
    remaining time; a stalled stream has at most one such timeout outstanding.
    """
    def __init__(self, session, budget):
        self.session, self.budget = session, budget
        self.credentials = getattr(session, "credentials", None)

    def get(self, url, **kwargs):
        b = self.budget
        b.check()
        if b.requests >= b.limits.max_requests:
            raise _BoundedStop("REQUEST_LIMIT")
        if url == API:
            if b.stats["listing_pages"] >= b.limits.max_pages:
                raise _BoundedStop("PAGE_LIMIT")
            remaining = b.limits.max_metadata - b.stats["metadata_items_seen"]
            if remaining <= 0:
                raise _BoundedStop("METADATA_LIMIT")
            kwargs["params"] = dict(kwargs["params"], pageSize=min(1000, remaining))
        kwargs["timeout"] = max(0.001, min(20, b.limits.max_seconds - (time.monotonic()-b.started)))
        b.requests += 1
        return self.session.get(url, **kwargs)


def _read_media(store, items, budget, secrets):
    """Verify every duplicate by streaming original media, with no cache/write.

    Limit detection may receive one final <=64 KiB chunk; that explicit maximum
    overrun is recorded. A stopped object is never put into the archive.
    """
    first, identities = None, []
    for item in items:
        budget.check()
        response = _read(store.session, f"{API}/{item['id']}",
            params={"alt": "media", "supportsAllDrives": "true"}, stream=True, timeout=20)
        try:
            parts, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                budget.received += len(chunk)
                size += len(chunk)
                budget.check()
                if budget.received > budget.limits.max_bytes:
                    raise _BoundedStop("BYTE_LIMIT")
                if size > budget.limits.max_object_bytes:
                    raise _BoundedStop("OBJECT_BYTE_LIMIT")
                parts.append(chunk)
            raw = b"".join(parts)
        finally:
            response.close()
        checksum = _checksum(item)
        digest = hashlib.sha256(raw).hexdigest()
        if checksum and checksum != digest:
            raise ExportUnavailable("REMOTE_CHECKSUM_CONFLICT")
        if first is not None and raw != first:
            raise ExportUnavailable("REMOTE_IDENTITY_CONFLICT")
        _credentials(raw, secrets)
        first = raw
        identities.append(dict(file_id=item["id"], name=item["name"],
                               provider_sha256=checksum or None, sha256=digest))
    return first, identities


def _credentials(raw, secrets, decoded=None):
    if private.SECRET_BYTES.search(raw) or any(value in raw for value in secrets):
        raise ExportUnavailable("CREDENTIAL_MATERIAL_DETECTED")
    if decoded is not None and private._credential_object(decoded):
        raise ExportUnavailable("CREDENTIAL_MATERIAL_DETECTED")


def _dependencies(rows, export_complete):
    identities = {table: set() for table in SCHEMA}
    for table, row in rows:
        columns, primary = SCHEMA[table]
        identities[table].add(tuple(row[columns.index(field)] for field in primary))
    absent, unrecorded = [], []
    count = 0
    for table, row in rows:
        columns, primary = SCHEMA[table]
        values = dict(zip(columns, row))
        origin = {field: values[field] for field in primary}
        for field, parent, parent_field in DEPENDENCIES.get(table, ()):
            value = values[field]
            if value is None:
                continue
            count += 1
            if (value,) not in identities[parent]:
                absent.append(dict(table=table, identity=origin, field=field,
                    target_table=parent, target_field=parent_field, target_identity=value,
                    status="MISSING_FROM_COMPLETE_EXPORT" if export_complete else "NOT_INCLUDED_REMOTE_UNKNOWN"))
        if table == "prospective_prediction":
            needed = ("quote_id", "model_id", "feature_snapshot_id", "feature_frozen_at",
                      "model_available_at", "runtime_hash", "source_commit")
            missing = [field for field in needed if values[field] in (None, "")]
            if missing:
                unrecorded.append(dict(table=table, identity=origin, fields=missing,
                                       status="UNRECORDED_IN_ORIGINAL_ROW"))
    return dict(declared_reference_count=count, unresolved_references=absent,
                unrecorded_prediction_bindings=unrecorded,
                external_feature_artifact_source_contents="NOT_FETCHED_AVAILABILITY_UNKNOWN",
                scientific_chain_completeness="NOT_ESTABLISHED")


def build_download(folder, *, forbidden_values=(), limits=None, store_factory=None):
    """Return original-object ZIP and truthful manifest; retrieval only.

    Listing interruption and resource caps can return a labelled partial export.
    Corrupt objects and identity/credential conflicts reject the entire export.
    No database API, canonical sync, upload or probability execution is called.
    """
    limits = limits or Limits()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder or ""):
        raise ExportUnavailable("REMOTE_FOLDER_NOT_CONFIGURED")
    budget = _Budget(limits)
    prepared = datetime.now(timezone.utc).isoformat()
    secrets = [str(value).encode() for value in forbidden_values if value]
    secrets.extend(value.encode() for name, value in os.environ.items()
                   if value and private.SECRET_NAME.search(name))
    store, original_session = None, None
    try:
        if store_factory is None:
            from app_core.evidence_drive import _authorized_session
            original_session = _authorized_session()
            try:
                store = DriveStore(folder, session=_Session(original_session, budget))
            except Exception:
                original_session.close()
                raise
            owned_session = True
        else:
            store = store_factory(folder)
            original_session = store.session
            owned_session = False
        # Preserve authenticated scope before wrapping; do not serialize credentials.
        scope = store.storage_scope_hash()
        store.session = _Session(original_session, budget)
        files, ids, listing_reason = [], {}, None
        try:
            for item in store._files(stats=budget.stats):
                budget.check()
                if budget.stats["metadata_items_seen"] > limits.max_metadata:
                    raise _BoundedStop("METADATA_LIMIT")
                if not isinstance(item["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
                    raise ExportUnavailable("REMOTE_IDENTITY_INVALID")
                if item["id"] in ids and ids[item["id"]] != item:
                    raise ExportUnavailable("REMOTE_IDENTITY_CONFLICT")
                ids[item["id"]] = item
                if item["name"].startswith(PREFIX):
                    try:
                        _checksum(item)
                    except EvidenceStorageError:
                        raise ExportUnavailable("REMOTE_CHECKSUM_INVALID") from None
                    files.append(item)
        except _BoundedStop as exc:
            listing_reason = str(exc)
        except Exception as exc:
            if isinstance(exc, ExportUnavailable):
                raise
            listing_reason = "LISTING_INTERRUPTED"
        inventory = DriveInventory(scope, uuid.uuid4().hex, PREFIX, tuple(files),
            budget.stats["listing_pages"], budget.stats["metadata_items_seen"], listing_reason is None)
        groups = {}
        for item in inventory.files:
            groups.setdefault(item["name"], {})[item["id"]] = item
        output = io.BytesIO()
        entries, rows, counts, stop = [], [], {}, None
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key in sorted(groups):
                if len(entries) >= limits.max_objects:
                    stop = "OBJECT_LIMIT"
                    break
                try:
                    raw, remote_ids = _read_media(store, list(groups[key].values()), budget, secrets)
                except _BoundedStop as exc:
                    stop = str(exc)
                    break
                except ExportUnavailable:
                    raise
                except EvidenceStorageError:
                    raise ExportUnavailable("REMOTE_CHECKSUM_OR_IDENTITY_CONFLICT") from None
                except Exception:
                    stop = "MEDIA_READ_INTERRUPTED"
                    break
                try:
                    table, row = _decode(key, raw, SCHEMA)
                except (ValueError, TypeError, KeyError, RecursionError):
                    raise ExportUnavailable("CANONICAL_OBJECT_CORRUPTED") from None
                # The wire decoder recovers base64 blobs; inspect their original
                # bytes too so credentials cannot hide in an encoded raw source.
                for value in row:
                    if isinstance(value, bytes):
                        _credentials(value, secrets)
                        try:
                            _credentials(value, secrets, json.loads(value))
                        except (ValueError, UnicodeDecodeError) as exc:
                            if isinstance(exc, ExportUnavailable):
                                raise
                    elif isinstance(value, str):
                        _credentials(value.encode(), secrets, value)
                try:
                    budget.check()
                except _BoundedStop as exc:
                    stop = str(exc)
                    break
                archive.writestr(key, raw)
                entries.append(dict(path=key, sha256=hashlib.sha256(raw).hexdigest(),
                                    bytes=len(raw), remote_objects=remote_ids))
                rows.append((table, row))
                counts[table] = counts.get(table, 0) + 1
            if time.monotonic()-budget.started >= limits.max_seconds:
                stop = stop or "TIME_LIMIT"
            complete = inventory.complete and stop is None
            events, games = set(), set()
            for table, row in rows:
                values = dict(zip(SCHEMA[table][0], row))
                if values.get("event_id"):
                    events.add(values["event_id"])
                if values.get("game_id"):
                    games.add((values.get("sport"), values["game_id"]))
            manifest = dict(schema_version=1, kind="PRIVATE_REMOTE_CANONICAL_JSON_EXPORT",
                prepared_at=prepared, completed_at=datetime.now(timezone.utc).isoformat(),
                canonical_contract_version=CANONICAL_SCHEMA_VERSION,
                inventory=dict(folder_id=folder, scope_hash=inventory.scope_hash,
                    operation_id=inventory.operation_id, prefix=PREFIX,
                    listing_pages=inventory.listing_pages, metadata_items_seen=inventory.metadata_items_seen,
                    canonical_remote_files_listed=len({item["id"] for item in files}),
                    canonical_paths_listed=len(groups), complete=inventory.complete,
                    incomplete_reason=listing_reason, traversal="ENTIRE_CONFIGURED_FOLDER; canonical prefix filtered locally",
                    point_in_time_consistency="PAGINATED_LISTING_NOT_AN_ATOMIC_REMOTE_SNAPSHOT"),
                export_complete=complete, stop_reason=stop, limits=asdict(limits),
                resources=dict(requests=budget.requests, media_bytes_received=budget.received,
                    maximum_detection_chunk_overrun_bytes=65536),
                objects=entries, counts=dict(exported_paths=len(entries),
                    exported_remote_files=sum(len(entry["remote_objects"]) for entry in entries),
                    duplicate_remote_names=sum(len(entry["remote_objects"])-1 for entry in entries),
                    rows_by_table=counts, unique_event_id_references=len(events),
                    unique_sport_game_id_references=len(games)),
                dependencies=_dependencies(rows, complete),
                original_object_bytes_preserved=True, sqlite_created_or_hydrated=False,
                remote_mutated=False, probabilities_reconstructed=False,
                scientific_acceptance=False, wagering_authority=False)
            manifest_raw = private._json(manifest)
            _credentials(manifest_raw, secrets, manifest)
            archive.writestr("manifest.json", manifest_raw)
        return output.getvalue(), manifest
    except ExportUnavailable:
        raise
    except Exception:
        raise ExportUnavailable("REMOTE_EXPORT_ACCESS_OR_INTEGRITY_FAILURE") from None
    finally:
        if store is not None and original_session is not None:
            store.session = original_session
            if owned_session:
                original_session.close()
