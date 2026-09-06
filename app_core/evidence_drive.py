"""Create/read-only evidence objects in a Google Workspace Shared Drive folder."""
from io import BytesIO
import json
import os
import re
import uuid

API = "https://www.googleapis.com/drive/v3/files"


class AlreadyExists(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class DriveStore:
    """Small object-store interface used by the evidence replication layer.

Drive permits duplicate names. Identical duplicates are harmless retries;
conflicting duplicates fail closed. No update/delete operation is implemented.
"""
    def __init__(self, folder, session=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder):
            raise ValueError("Use the Shared Drive folder ID, not a URL")
        self.folder = folder
        if session is None:
            from google.oauth2.service_account import Credentials
            from google.auth.transport.requests import AuthorizedSession
            info = json.loads(os.environ.get("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "{}"))
            credentials = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive"])
            session = AuthorizedSession(credentials)
        self.session = session
        response = self.session.get(f"{API}/{folder}", params={"supportsAllDrives": "true", "fields": "id,driveId,mimeType,trashed"}, timeout=20)
        response.raise_for_status()
        metadata = response.json()
        if not metadata.get("driveId") or metadata.get("trashed") or metadata.get("mimeType") != "application/vnd.google-apps.folder":
            raise ValueError("Evidence folder must be an active Google Workspace Shared Drive folder")
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
                raise ValueError("Drive listing was incomplete; restore cannot be verified")
            yield from data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                return

    def get_object(self, *, Key, **kwargs):
        files = list(self._files(Key))
        if not files:
            raise ValueError("Remote evidence object is missing")
        contents = []
        for item in files:
            response = self.session.get(f"{API}/{item['id']}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=20)
            response.raise_for_status()
            contents.append(response.content)
        if any(raw != contents[0] for raw in contents):
            raise ValueError("Drive contains conflicting duplicate evidence names")
        return {"Body": BytesIO(contents[0])}

    def put_object(self, *, Key, Body, IfNoneMatch, **kwargs):
        if IfNoneMatch != "*":
            raise ValueError("Only create-only evidence uploads are supported")
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
        # The caller reads every matching object back, detecting conflicts even
        # when another writer races this creation or a timed-out upload is retried.

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise ValueError("Unsupported listing operation")
        return self

    def paginate(self, *, Prefix, **kwargs):
        names = sorted({item["name"] for item in self._files() if item["name"].startswith(Prefix)})
        yield {"Contents": [{"Key": name} for name in names]}
