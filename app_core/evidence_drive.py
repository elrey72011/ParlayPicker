"""Create/read-only evidence objects in a Google Workspace Shared Drive folder."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from io import BytesIO
import json
import re
import uuid

from app_core.evidence_config import EvidenceStorageError

API = "https://www.googleapis.com/drive/v3/files"


class AlreadyExists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


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
        response = self.session.get(f"{API}/{folder}", params={"supportsAllDrives": "true", "fields": "id,driveId,mimeType,trashed"}, timeout=20)
        response.raise_for_status()
        metadata = response.json()
        if not metadata.get("driveId") or metadata.get("trashed") or metadata.get("mimeType") != "application/vnd.google-apps.folder":
            raise EvidenceStorageError("Evidence folder must be an active Google Workspace Shared Drive folder")
        self.drive = metadata["driveId"]

    def _files(self, name=None):
        query = f"'{self.folder}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            query += f" and name = '{escaped}'"
        token = None
        while True:
            params = {"q": query, "fields": "nextPageToken,incompleteSearch,files(id,name)", "pageSize": 1000,
                      "supportsAllDrives": "true", "includeItemsFromAllDrives": "true", "corpora": "drive", "driveId": self.drive}
            if token:
                params["pageToken"] = token
            response = self.session.get(API, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data.get("incompleteSearch"):
                raise EvidenceStorageError("Drive listing was incomplete; restore cannot be verified")
            yield from data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                return

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

    def _read_files(self, files):
        contents = []
        for item in files:
            response = self.session.get(f"{API}/{item['id']}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=20)
            response.raise_for_status()
            contents.append(response.content)
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

    def read_objects(self, *, Prefix):
        """One fresh listing, then verified reads by ID; no persistent cache."""
        grouped = {}
        for item in self._files():
            if item['name'].startswith(Prefix):
                grouped.setdefault(item['name'], {})[item['id']] = item
        for name, file_id in self.created_ids.items():
            if name.startswith(Prefix):
                grouped.setdefault(name, {})[file_id] = {'id': file_id, 'name': name}
        return self.run_parallel(
            lambda worker, name: (name, worker._read_files(list(grouped[name].values()))),
            sorted(grouped))

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
        response = self.session.post("https://www.googleapis.com/upload/drive/v3/files",
                                     params={"uploadType": "multipart", "supportsAllDrives": "true", "fields": "id"},
                                     headers={"Content-Type": f"multipart/related; boundary={boundary}"}, data=body, timeout=30)
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
