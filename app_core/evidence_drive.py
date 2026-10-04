"""Create/read-only evidence objects in a Google Workspace Shared Drive folder."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import local
from app_core.scoped_reads import singleflight
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import uuid
import time
import requests

from app_core.performance_spans import PerformanceSpan, opaque_hash, operation_ids


def _read(session, url, **kwargs):
    """Retry transient read failures only; never retry an uncertain upload."""
    metrics = kwargs.pop("_metrics", None)
    for attempt in range(3):
        try:
            response = session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        wait_seconds = attempt + 1
        if metrics is not None:
            metrics["retries"] = metrics.get("retries", 0) + 1
            metrics["retry_wait_ms"] = metrics.get("retry_wait_ms", 0) + wait_seconds * 1000
        time.sleep(wait_seconds)

from app_core.evidence_config import EvidenceStorageError

API = "https://www.googleapis.com/drive/v3/files"


def _checksum(item):
    checksum = item.get("sha256Checksum") or ""
    if checksum and (not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)):
        raise EvidenceStorageError("Drive evidence has an invalid SHA-256 checksum")
    return checksum


class AlreadyExists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


@dataclass(frozen=True)
class DriveInventory:
    """A complete, operation-scoped Drive folder inventory.

    The object is deliberately immutable and is never kept as global authority.
    Callers must discover a new inventory at each required action boundary.
    """
    scope_hash: str
    operation_id: str
    namespace: str | None
    files: tuple
    listing_pages: int
    metadata_items_seen: int
    complete: bool = True


@dataclass(frozen=True)
class VerifiedReadReport:
    inventory_scope_hash: str
    operation_id: str
    requested_prefixes: tuple
    listing_traversals: int
    listing_pages: int
    metadata_items_seen: int
    objects_matched: int
    objects_downloaded: int
    objects_reused: int
    bytes_downloaded: int
    retries: int
    retry_wait_ms: int
    cache_hits: int
    cache_misses: int
    verification_status: str
    full_verification: bool


def _authorized_session():
    from google.oauth2.service_account import Credentials
    from google.auth.transport.requests import AuthorizedSession
    from app_core.evidence_config import service_account_info, EvidenceConfigurationError
    info = service_account_info()
    try:
        credentials = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive"])
    except (ValueError, TypeError) as exc:
        raise EvidenceConfigurationError(
            "Service-account JSON parsed, but its signing key could not be loaded. "
            "Replace the secret with the complete original downloaded JSON in triple single quotes; "
            "do not edit the private-key contents.") from None
    return AuthorizedSession(credentials)


class DriveStore:
    """Small object-store interface used by the evidence replication layer.

Drive permits duplicate names. Identical duplicates are harmless retries;
conflicting duplicates fail closed. No update/delete operation is implemented.
"""
    def __init__(self, folder, session=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder):
            raise EvidenceStorageError("Use the Shared Drive folder ID, not a URL")
        self.folder = folder
        self.created_ids = {}
        self._session_factory = _authorized_session if session is None else None
        if session is None:
            session = self._session_factory()
        self.session = session
        response = _read(self.session, f"{API}/{folder}", params={"supportsAllDrives": "true", "fields": "id,driveId,mimeType,trashed"}, timeout=20)
        response.raise_for_status()
        metadata = response.json()
        if not metadata.get("driveId") or metadata.get("trashed") or metadata.get("mimeType") != "application/vnd.google-apps.folder":
            raise EvidenceStorageError("Evidence folder must be an active Google Workspace Shared Drive folder")
        self.drive = metadata["driveId"]
        self.last_read_report = None

    def storage_scope_hash(self):
        session = getattr(self, "session", None)
        credentials = getattr(session, "credentials", None)
        identity = (getattr(credentials, "service_account_email", None)
                    or getattr(credentials, "client_email", None)
                    or ("injected-session", id(session) if session is not None else id(self)))
        subject = getattr(credentials, "_subject", None)
        principal = getattr(credentials, "quota_project_id", None)
        return getattr(self, "_authenticated_scope", None) or opaque_hash("google_workspace_shared_drive", identity, subject, principal,
                           getattr(self, "drive", "injected"), getattr(self, "folder", "injected"))

    def _files(self, name=None, stats=None):
        query = f"'{self.folder}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            query += f" and name = '{escaped}'"
        token = None
        seen_tokens = set()
        while True:
            params = {"q": query, "fields": "nextPageToken,incompleteSearch,files(id,name,sha256Checksum)", "pageSize": 1000,
                      "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "corpora": "drive", "driveId": self.drive}
            if token:
                params["pageToken"] = token
            response = _read(self.session, API, params=params, timeout=20, _metrics=stats)
            response.raise_for_status()
            data = response.json()
            if stats is not None:
                stats["listing_pages"] = stats.get("listing_pages", 0) + 1
            if not isinstance(data, dict) or data.get("incompleteSearch"):
                raise EvidenceStorageError("Drive listing was incomplete; restore cannot be verified")
            files = data.get("files")
            if not isinstance(files, list) or any(
                    not isinstance(item, dict) or not item.get("id") or not item.get("name")
                    for item in files):
                raise EvidenceStorageError("Drive listing was incomplete; restore cannot be verified")
            if stats is not None:
                stats["metadata_items_seen"] = stats.get("metadata_items_seen", 0) + len(files)
            yield from files
            token = data.get("nextPageToken")
            if not token:
                return
            if not isinstance(token, str) or token in seen_tokens:
                raise EvidenceStorageError("Drive listing was incomplete; repeated page token")
            seen_tokens.add(token)

    def get_object(self, *, Key, **kwargs):
        files = list(self._files(Key))
        # A successful upload returns an authoritative file ID. Search indexing
        # need not be used to rediscover the file before read-back verification.
        created_id = self.created_ids.get(Key)
        if created_id and all(item["id"] != created_id for item in files):
            files.append({"id": created_id, "name": Key})
        if not files:
            raise EvidenceStorageError("Remote evidence object is missing")
        return {"Body": BytesIO(self._read_files(files))}

    def _read_files(self, files, metrics=None):
        contents = []
        for item in files:
            response = _read(self.session, f"{API}/{item['id']}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=20,
                             _metrics=metrics)
            response.raise_for_status()
            checksum = _checksum(item)
            if re.fullmatch(r"[a-f0-9]{64}", checksum) and hashlib.sha256(response.content).hexdigest() != checksum:
                raise EvidenceStorageError("Remote evidence checksum changed during read")
            contents.append(response.content)
            if metrics is not None:
                metrics["objects_downloaded"] = metrics.get("objects_downloaded", 0) + 1
                metrics["bytes_downloaded"] = metrics.get("bytes_downloaded", 0) + len(response.content)
        if any(raw != contents[0] for raw in contents):
            raise EvidenceStorageError("Drive contains conflicting duplicate evidence names")
        return contents[0]

    def run_parallel(self, operation, items, progress=None):
        """Bounded I/O with a separate authenticated session per worker.

        Callbacks run on the caller thread. Injected sessions stay sequential
        unless their owner explicitly supplies a worker-session factory.
        """
        items = list(items)
        if not items:
            return []
        if self._session_factory is None or len(items) == 1:
            result = []
            for item in items:
                result.append(operation(self, item))
                if progress:
                    progress(len(result), len(items))
            return result
        state, sessions = local(), []
        def run(item):
            if not hasattr(state, 'store'):
                worker = object.__new__(DriveStore)
                worker.folder, worker.drive = self.folder, self.drive
                worker.created_ids = self.created_ids
                worker._authenticated_scope = self.storage_scope_hash()
                worker._session_factory = None
                worker.session = self._session_factory()
                sessions.append(worker.session)
                state.store = worker
            return operation(state.store, item)
        pool = ThreadPoolExecutor(max_workers=min(4, len(items)))
        futures = {}
        try:
            futures = {pool.submit(run, item): i for i, item in enumerate(items)}
            result = [None] * len(items)
            for done, future in enumerate(as_completed(futures), 1):
                result[futures[future]] = future.result()
                if progress:
                    progress(done, len(items))
            return result
        finally:
            # Failed/uncertain writes are never retried. Finish running calls
            # before reporting an error; cancel calls that have not started.
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            for session in sessions:
                session.close()

    def discover_complete_inventory(self, *, operation_id=None, namespace=None, ids=None, coalesce=True):
        if not coalesce:
            return self._discover_complete_inventory(operation_id=operation_id, namespace=namespace, ids=ids)
        inventory, shared = singleflight(
            ("inventory", self.storage_scope_hash(), namespace),
            lambda: self._discover_complete_inventory(operation_id=operation_id, namespace=namespace, ids=ids))
        if shared:
            with PerformanceSpan("drive_inventory_join", ids=ids, storage_scope_hash=self.storage_scope_hash()) as span:
                span.set(listing_traversals=0, cache_hits=1, verification_status="inflight_complete_inventory")
        return inventory

    def _discover_complete_inventory(self, *, operation_id=None, namespace=None, ids=None):
        """List the entire folder exactly once for one bounded read phase."""
        ids = ids or operation_ids(action_id=operation_id)
        operation_id = operation_id or ids["action_id"]
        stats = {"listing_pages": 0, "metadata_items_seen": 0, "retries": 0, "retry_wait_ms": 0}
        with PerformanceSpan("drive_inventory", ids=ids, storage_scope_hash=self.storage_scope_hash()) as span:
            try:
                files = list(self._files(stats=stats))
            except TypeError:
                # Compatibility for deterministic injected stores that override
                # _files() without the optional instrumentation argument.
                files = list(self._files())
                stats["metadata_items_seen"] = len(files)
                stats["listing_pages"] = None
            span.set(listing_traversals=1, listing_pages=stats["listing_pages"],
                     metadata_items_seen=stats["metadata_items_seen"], retries=stats["retries"],
                     retry_wait_ms=stats["retry_wait_ms"], verification_status="complete")
        return DriveInventory(self.storage_scope_hash(), operation_id, namespace, tuple(files),
                              stats["listing_pages"], stats["metadata_items_seen"])

    def _inventory_groups(self, inventory, prefixes):
        if not inventory.complete or inventory.scope_hash != self.storage_scope_hash():
            raise EvidenceStorageError("Drive inventory is incomplete or belongs to a different storage scope")
        grouped = {prefix: {} for prefix in prefixes}
        for item in inventory.files:
            for prefix in prefixes:
                if item["name"].startswith(prefix):
                    grouped[prefix].setdefault(item["name"], {})[item["id"]] = item
        for name, file_id in self.created_ids.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    grouped[prefix].setdefault(name, {}).setdefault(file_id, {"id": file_id, "name": name})
        return grouped

    def verified_cache_root(self, cache_dir, namespace):
        return Path(cache_dir) / opaque_hash(self.storage_scope_hash(), namespace)

    def read_verified_prefixes(self, *, Prefixes, inventory=None, cache_dir=None,
                               full_verify=False, progress=None, ids=None):
        """Read several prefixes from one complete inventory.

        Cached bytes are reusable only when their local SHA-256 agrees with the
        fresh provider checksum. Missing checksums and full verification always
        download media. Duplicate names retain fail-closed byte comparison.
        """
        prefixes = tuple(dict.fromkeys(Prefixes))
        ids = ids or operation_ids(action_id=getattr(inventory, "operation_id", None))
        inventory = inventory or self.discover_complete_inventory(ids=ids, namespace=opaque_hash(*sorted(prefixes)))
        grouped_by_prefix = self._inventory_groups(inventory, prefixes)
        by_name = {}
        for groups in grouped_by_prefix.values():
            for name, items in groups.items():
                by_name.setdefault(name, {}).update(items)
        root = (Path(cache_dir) / opaque_hash(inventory.scope_hash, inventory.namespace or tuple(sorted(prefixes)))
                if cache_dir is not None else None)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

        def read(worker, name):
            local_metrics = {"objects_downloaded": 0, "objects_reused": 0,
                             "bytes_downloaded": 0, "cache_hits": 0,
                             "cache_misses": 0, "retries": 0, "retry_wait_ms": 0}
            contents = []
            for item in by_name[name].values():
                checksum = _checksum(item)
                target = root / checksum if root is not None and re.fullmatch(r"[a-f0-9]{64}", checksum) else None
                raw = None
                if not full_verify and target is not None:
                    try:
                        candidate = target.read_bytes()
                        if hashlib.sha256(candidate).hexdigest() == checksum:
                            raw = candidate
                            local_metrics["objects_reused"] += 1
                            local_metrics["cache_hits"] += 1
                    except OSError:
                        pass
                if raw is None:
                    local_metrics["cache_misses"] += 1
                    def download():
                        counts = {"objects_downloaded": 0, "bytes_downloaded": 0, "retries": 0, "retry_wait_ms": 0}
                        try:
                            value = worker._read_files([item], metrics=counts)
                        except TypeError:
                            value = worker._read_files([item])
                        if not counts["objects_downloaded"]:
                            counts["objects_downloaded"] = 1
                            counts["bytes_downloaded"] = len(value)
                        # Checksums are validated even with caching disabled.
                        if re.fullmatch(r"[a-f0-9]{64}", checksum) and hashlib.sha256(value).hexdigest() != checksum:
                            raise EvidenceStorageError("Remote evidence checksum changed during read")
                        return value, counts
                    (raw, counts), shared = singleflight(
                        ("media", inventory.scope_hash, inventory.namespace or tuple(sorted(prefixes)),
                         item["id"], checksum, full_verify), download)
                    if shared:
                        local_metrics["objects_reused"] += 1
                        local_metrics["cache_hits"] += 1
                    else:
                        for counter, count in counts.items():
                            local_metrics[counter] += count
                    if target is not None:
                        if hashlib.sha256(raw).hexdigest() != checksum:
                            raise EvidenceStorageError("Remote evidence checksum changed during read")
                        temporary = root / (checksum + "." + uuid.uuid4().hex + ".tmp")
                        try:
                            temporary.write_bytes(raw)
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
                contents.append(raw)
            if any(raw != contents[0] for raw in contents):
                raise EvidenceStorageError("Drive contains conflicting duplicate evidence names")
            return name, contents[0], local_metrics

        with PerformanceSpan("drive_verified_prefixes", ids=ids,
                             storage_scope_hash=inventory.scope_hash) as span:
            results = self.run_parallel(read, sorted(by_name), progress=progress)
            totals = {name: sum(result[2][name] for result in results) for name in (
                "objects_downloaded", "objects_reused", "bytes_downloaded", "cache_hits",
                "cache_misses", "retries", "retry_wait_ms")}
            values = {prefix: [] for prefix in prefixes}
            raw_by_name = {name: raw for name, raw, _ in results}
            for prefix, groups in grouped_by_prefix.items():
                values[prefix] = [(name, raw_by_name[name]) for name in sorted(groups)]
            report = VerifiedReadReport(
                inventory.scope_hash, inventory.operation_id, prefixes, 1,
                inventory.listing_pages, inventory.metadata_items_seen, len(by_name),
                totals["objects_downloaded"], totals["objects_reused"], totals["bytes_downloaded"],
                totals["retries"], totals["retry_wait_ms"], totals["cache_hits"],
                totals["cache_misses"], "full_bytes_verified" if full_verify else "checksum_or_bytes_verified",
                full_verify)
            self.last_read_report = report
            span.set(listing_traversals=0, listing_pages=0, metadata_items_seen=0,
                     objects_matched=len(by_name), objects_downloaded=totals["objects_downloaded"],
                     objects_reused=totals["objects_reused"], bytes_downloaded=totals["bytes_downloaded"],
                     retries=totals["retries"], retry_wait_ms=totals["retry_wait_ms"],
                     cache_hits=totals["cache_hits"], cache_misses=totals["cache_misses"],
                     records_returned=sum(len(items) for items in values.values()),
                     verification_status=report.verification_status)
        return values

    def read_objects(self, *, Prefix):
        """Compatibility wrapper using a fresh single-phase inventory."""
        inventory = self.discover_complete_inventory(namespace=Prefix)
        return self.read_verified_prefixes(Prefixes=[Prefix], inventory=inventory)[Prefix]

    def read_cached_objects(self, *, Prefix, cache_dir):
        """Fresh complete listing; reuse only bytes matching Drive's SHA-256.

        Missing checksums always force media reads. Every duplicate is checked;
        cached bytes alone never establish remote presence or durability.
        """
        inventory = self.discover_complete_inventory(namespace=Prefix)
        return self.read_verified_prefixes(Prefixes=[Prefix], inventory=inventory,
                                           cache_dir=cache_dir)[Prefix]

    def put_object(self, *, Key, Body, IfNoneMatch, **kwargs):
        if IfNoneMatch != "*":
            raise EvidenceStorageError("Only create-only evidence uploads are supported")
        if list(self._files(Key)):
            raise AlreadyExists()
        boundary = "evidence_" + uuid.uuid4().hex
        metadata = json.dumps({"name": Key, "parents": [self.folder], "mimeType": "application/json"}).encode()
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode() + metadata
                + f"\r\n--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode() + Body
                + f"\r\n--{boundary}--\r\n".encode())
        try:
            response = self.session.post("https://www.googleapis.com/upload/drive/v3/files",
                                         params={"uploadType": "multipart", "supportsAllDrives": "true", "fields": "id"},
                                         headers={"Content-Type": f"multipart/related; boundary={boundary}"}, data=body, timeout=30)
        except (requests.Timeout, requests.ConnectionError):
            # The server may have committed the immutable object before the
            # response was lost. Never repeat an uncertain upload. A bounded
            # listing and full-byte readback may establish that it completed.
            for attempt in range(3):
                if attempt:
                    time.sleep(attempt * 2)
                files = list(self._files(Key))
                if not files:
                    continue
                if self._read_files(files) != Body:
                    raise EvidenceStorageError("Timed-out upload conflicts with remote evidence")
                self.created_ids[Key] = files[0]["id"]
                return
            raise
        response.raise_for_status()
        file_id = response.json().get("id")
        if not isinstance(file_id, str) or not file_id:
            raise EvidenceStorageError("Drive accepted upload but returned no file ID; retry synchronization.")
        self.created_ids[Key] = file_id
        # The caller reads every matching object back, detecting conflicts even
        # when another writer races this creation or a timed-out upload is retried.

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise EvidenceStorageError("Unsupported listing operation")
        return self

    def paginate(self, *, Prefix, **kwargs):
        names = sorted({item["name"] for item in self._files() if item["name"].startswith(Prefix)})
        yield {"Contents": [{"Key": name} for name in names]}
