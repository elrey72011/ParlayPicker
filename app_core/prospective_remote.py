"""Immutable, read-back-verified remote replication of canonical research rows.

The remote key is derived from the table and primary key, rather than the row
contents. A changed row under the same identity therefore fails closed on
restore or backup. This module has no wager-activation path.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import re

from app_core import prospective_evidence as evidence


PREFIX = "parlaypicker/canonical-prospective-v1/"
MAX_OBJECT_BYTES = 60_000_000


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _schema(db):
    tables = {}
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'prospective_%' ORDER BY name"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", table):
            raise ValueError("canonical_remote_schema_invalid")
        details = db.execute(f"PRAGMA table_info({table})").fetchall()
        columns = tuple(row[1] for row in details)
        primary = tuple(row[1] for row in sorted((row for row in details if row[5]), key=lambda row: row[5]))
        if not columns or not primary:
            raise ValueError("canonical_remote_schema_invalid")
        tables[table] = (columns, primary)
    return tables


def _table_order(db, schema):
    """Insert parent rows first; defer only self-referencing revision chains."""
    ordered, active, visited = [], set(), set()

    def visit(table):
        if table in visited:
            return
        if table in active:
            raise ValueError("canonical_remote_foreign_key_cycle")
        active.add(table)
        for edge in db.execute(f"PRAGMA foreign_key_list({table})"):
            parent = edge[2]
            if parent in schema and parent != table:
                visit(parent)
        active.remove(table)
        visited.add(table)
        ordered.append(table)

    for table in schema:
        visit(table)
    return ordered


def _key(table, primary_values):
    ident = hashlib.sha256(_json([table, list(primary_values)])).hexdigest()
    return f"{PREFIX}{table}/{ident}.json"


def _encode(table, columns, primary, row):
    values = [({"blob_base64": base64.b64encode(value).decode("ascii")}
               if isinstance(value, bytes) else value) for value in row]
    raw = _json({"schema": 1, "table": table, "columns": columns, "row": values})
    if len(raw) > MAX_OBJECT_BYTES:
        raise ValueError("canonical_remote_object_too_large")
    identity = tuple(row[columns.index(column)] for column in primary)
    return _key(table, identity), raw


def _decode(key, raw, schema):
    if len(raw) > MAX_OBJECT_BYTES:
        raise ValueError("canonical_remote_object_too_large")
    try:
        item = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise ValueError("canonical_remote_json_invalid") from None
    if not isinstance(item, dict) or item.get("schema") != 1:
        raise ValueError("canonical_remote_schema_invalid")
    table = item.get("table")
    if table not in schema:
        raise ValueError("canonical_remote_table_invalid")
    columns, primary = schema[table]
    if item.get("columns") != list(columns) or not isinstance(item.get("row"), list) or len(item["row"]) != len(columns):
        raise ValueError("canonical_remote_columns_invalid")
    row = []
    for value in item["row"]:
        if isinstance(value, dict):
            if set(value) != {"blob_base64"}:
                raise ValueError("canonical_remote_blob_invalid")
            try:
                value = base64.b64decode(value["blob_base64"], validate=True)
            except (TypeError, ValueError, binascii.Error):
                raise ValueError("canonical_remote_blob_invalid") from None
        row.append(value)
    row = tuple(row)
    expected, canonical = _encode(table, columns, primary, row)
    if key != expected or raw != canonical:
        raise ValueError("canonical_remote_integrity_conflict")
    for field, digest_field in (("payload", "payload_hash"), ("raw_source", "source_hash")):
        if field in columns and digest_field in columns:
            value, digest = row[columns.index(field)], row[columns.index(digest_field)]
            if not isinstance(value, (str, bytes)) or hashlib.sha256(
                    value.encode() if isinstance(value, str) else value).hexdigest() != digest:
                raise ValueError("canonical_remote_evidence_hash_conflict")
    return table, row


def _remote_objects(client, folder):
    if callable(getattr(client, "read_objects", None)):
        yield from client.read_objects(Prefix=PREFIX)
    else:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder, Prefix=PREFIX):
            for item in page.get("Contents", []):
                key = item["Key"]
                with client.get_object(Bucket=folder, Key=key)["Body"] as body:
                    yield key, body.read(MAX_OBJECT_BYTES + 1)


def _readback(client, folder, key):
    with client.get_object(Bucket=folder, Key=key)["Body"] as body:
        return body.read(MAX_OBJECT_BYTES + 1)


def sync(path: str | Path, client, folder: str, session: dict | None = None) -> dict:
    """Restore before upload; merge only absent rows and verify every new upload.

    The session cache is valid only for this process and this local/remote scope.
    A fresh process re-reads the remote and detects immutable conflicts.
    """
    from app_core.evidence_drive import AlreadyExists
    from app_core.prospective_reconciliation import ensure_reconciliation_schema

    ensure_reconciliation_schema(path)
    session = {} if session is None else session
    scope = (str(Path(path).resolve()), folder, id(client))
    if session.get("scope") != scope:
        session.clear()
        session["scope"] = scope
    with closing(evidence.connect(path)) as db:
        schema = _schema(db)
        table_order = _table_order(db, schema)
    restored = 0
    read_count = 0
    verified = session.setdefault("verified", {})
    if not session.get("restored"):
        remote = {}
        for key, raw in _remote_objects(client, folder):
            read_count += 1
            table, row = _decode(key, raw, schema)
            if key in remote and remote[key][1] != raw:
                raise ValueError("canonical_remote_identity_conflict")
            remote[key] = (table, raw, row)
        with closing(evidence.connect(path)) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("PRAGMA defer_foreign_keys=ON")
                # Foreign keys are deferred until every immutable remote row is present.
                by_table = {table: [] for table in table_order}
                for key, item in remote.items():
                    by_table[item[0]].append((key, item))
                for table in table_order:
                    columns, primary = schema[table]
                    for key, (_, raw, row) in sorted(by_table[table]):
                        where = " AND ".join(f"{column}=?" for column in primary)
                        ident = tuple(row[columns.index(column)] for column in primary)
                        old = db.execute(f"SELECT {','.join(columns)} FROM {table} WHERE {where}", ident).fetchone()
                        if old is not None:
                            if tuple(old) != row:
                                raise ValueError("canonical_remote_local_conflict")
                        else:
                            try:
                                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", row)
                            except sqlite3.IntegrityError:
                                raise ValueError("canonical_remote_source_conflict") from None
                            restored += 1
                if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("canonical_remote_foreign_key_conflict")
        verified.update({key: hashlib.sha256(raw).hexdigest() for key, (_, raw, _) in remote.items()})
        session["restored"] = True
    pending = []
    with closing(evidence.connect(path)) as db:
        for table, (columns, primary) in schema.items():
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                key, raw = _encode(table, columns, primary, tuple(row))
                if key in verified:
                    if verified[key] != hashlib.sha256(raw).hexdigest():
                        raise ValueError("canonical_remote_local_conflict")
                    continue
                _decode(key, raw, schema)
                pending.append((key, raw))
    if pending:
        print(json.dumps({"stage": "CANONICAL_UPLOAD", "pending_records": len(pending)}), flush=True)
    def upload(worker, item):
        key, raw = item
        try:
            worker.put_object(Bucket=folder, Key=key, Body=raw,
                              ContentType="application/json", IfNoneMatch="*")
        except AlreadyExists:
            pass
        if _readback(worker, folder, key) != raw:
            raise ValueError("canonical_remote_readback_conflict")
        return key, hashlib.sha256(raw).hexdigest()
    def upload_progress(done, total):
        if done == 1 or done % 100 == 0 or done == total:
            print(json.dumps({"stage": "CANONICAL_UPLOAD", "verified_records": done,
                              "pending_records": total}), flush=True)
    if callable(getattr(client, "run_parallel", None)):
        uploaded = client.run_parallel(upload, pending, progress=upload_progress)
    else:
        uploaded = []
        for item in pending:
            uploaded.append(upload(client, item))
            upload_progress(len(uploaded), len(pending))
    verified.update(uploaded)
    return {"remote_records_read": read_count, "records_restored": restored,
            "records_verified": len(verified),
            "new_records_verified": len(uploaded)}
