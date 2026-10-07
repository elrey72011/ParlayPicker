"""Synthetic remote objects only; actual Drive/UI path, all network blocked."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import zipfile

import pytest
from app_core import remote_canonical_download as export
from app_core.canonical_remote_contract import COLUMNS, DEPENDENCIES, SCHEMA
from app_core.canonical_schema import CANONICAL_PRIMARY_KEYS
from app_core.evidence_drive import API, DriveStore
from app_core.prospective_remote import _encode
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Remote export must not create/hydrate/query SQLite")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with blocked_network():
        yield


def record(table="prospective_event", identity="synthetic-event", **values):
    columns, primary = SCHEMA[table]
    row = dict.fromkeys(columns)
    for field in primary:
        row[field] = identity
    row.update(sport="NCAAF", game_id="synthetic-game")
    if "raw_source" in columns:
        row.update(raw_source=b'{"synthetic":true}',
                   source_hash=hashlib.sha256(b'{"synthetic":true}').hexdigest())
    if "payload" in columns:
        row.update(payload='{"synthetic":true}',
                   payload_hash=hashlib.sha256(b'{"synthetic":true}').hexdigest())
    row.update(values)
    key, raw = _encode(table, columns, primary, tuple(row[c] for c in columns))
    return dict(id="id-"+hashlib.sha256(key.encode()).hexdigest(), name=key,
                sha256Checksum=hashlib.sha256(raw).hexdigest(), raw=raw)


class Response:
    def __init__(self, data=None, raw=b""):
        self.data, self.raw = data, raw
    def raise_for_status(self): pass
    def json(self): return self.data
    def iter_content(self, chunk_size):
        for i in range(0, len(self.raw), chunk_size):
            yield self.raw[i:i+chunk_size]
    def close(self): pass


class Session:
    def __init__(self, items, page_size=1, mode=None):
        self.items, self.page_size, self.mode = items, page_size, mode
        self.list_calls = self.media_calls = 0
    def get(self, url, params, timeout, **kwargs):
        assert timeout > 0 and params["supportsAllDrives"] == "true"
        if url == API+"/synthetic-folder":
            return Response(dict(id="synthetic-folder", driveId="synthetic-drive",
                mimeType="application/vnd.google-apps.folder"))
        if params.get("alt") == "media":
            self.media_calls += 1
            assert kwargs["stream"] is True
            if self.mode == "media_failure":
                raise RuntimeError("synthetic-secret-network-message")
            return Response(raw=next(item["raw"] for item in self.items if url.endswith("/"+item["id"])))
        self.list_calls += 1
        if self.mode == "list_failure" and self.list_calls == 2:
            raise RuntimeError("synthetic-secret-network-message")
        assert params["corpora"] == "drive" and params["driveId"] == "synthetic-drive"
        assert "'synthetic-folder' in parents" in params["q"]
        offset = int(params.get("pageToken", "0"))
        size = min(self.page_size, params["pageSize"])
        data = dict(files=[{k:v for k,v in item.items() if k != "raw"}
                           for item in self.items[offset:offset+size]])
        if offset + size < len(self.items): data["nextPageToken"] = str(offset+size)
        if self.mode == "repeated_token": data["nextPageToken"] = "1"
        if self.mode == "incomplete": data["incompleteSearch"] = True
        return Response(data)
    def post(self, *args, **kwargs): pytest.fail("No remote writes")
    def patch(self, *args, **kwargs): pytest.fail("No remote writes")
    def delete(self, *args, **kwargs): pytest.fail("No remote writes")
    def close(self): self.closed = True


def run(items, limits=None, mode=None, page_size=1, forbidden_values=()):
    session = Session(items, page_size, mode)
    data, manifest = export.build_download("synthetic-folder", limits=limits,
        forbidden_values=forbidden_values,
        store_factory=lambda folder: DriveStore(folder, session=session))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("manifest.json")) == manifest
        for entry in manifest["objects"]:
            assert entry["path"].startswith(export.PREFIX)
            raw = archive.read(entry["path"])
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return data, manifest, session


def test_all_pages_original_bytes_prefix_and_remote_identities():
    one, two = record(), record(identity="synthetic-two")
    unrelated = dict(id="generic", name="parlaypicker/evidence-v1/private.json", raw=b"never-read")
    data, manifest, session = run([one, unrelated, two])
    assert manifest["export_complete"] and manifest["inventory"]["complete"]
    assert manifest["inventory"]["listing_pages"] == 3
    assert manifest["counts"]["exported_paths"] == 2 and session.media_calls == 2
    assert manifest["counts"]["unique_sport_game_id_references"] == 1
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.read(one["name"]) == one["raw"]
        assert unrelated["name"] not in archive.namelist()
    assert {r["file_id"] for e in manifest["objects"] for r in e["remote_objects"]} == {one["id"],two["id"]}
    assert not manifest["sqlite_created_or_hydrated"] and not manifest["probabilities_reconstructed"]


@pytest.mark.parametrize("limits,reason,listing", [
    (export.Limits(max_pages=1), "PAGE_LIMIT", False),
    (export.Limits(max_metadata=1), "METADATA_LIMIT", False),
    (export.Limits(max_objects=1), "OBJECT_LIMIT", True),
    (export.Limits(max_bytes=1), "BYTE_LIMIT", True),
    (export.Limits(max_object_bytes=1), "OBJECT_BYTE_LIMIT", True),
    (export.Limits(max_requests=1), "REQUEST_LIMIT", False),
])
def test_bounded_partial_exports_never_claim_complete(limits, reason, listing):
    _, manifest, _ = run([record(), record(identity="synthetic-two")], limits)
    assert manifest["export_complete"] is False
    assert manifest["inventory"]["complete"] is listing
    assert reason in (manifest["stop_reason"], manifest["inventory"]["incomplete_reason"])
    assert manifest["dependencies"]["scientific_chain_completeness"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("mode", ["list_failure", "repeated_token", "incomplete", "media_failure"])
def test_interruption_and_invalid_pagination_are_explicitly_partial(mode):
    data, manifest, _ = run([record(), record(identity="synthetic-two")], mode=mode)
    assert not manifest["export_complete"]
    assert "synthetic-secret-network-message".encode() not in data


def test_time_bound_before_listing(monkeypatch):
    ticks = iter([0, 301])
    monkeypatch.setattr(export.time, "monotonic", lambda: next(ticks, 301))
    _, manifest, session = run([record()])
    assert not manifest["inventory"]["complete"] and not manifest["export_complete"]
    assert manifest["inventory"]["incomplete_reason"] == "TIME_LIMIT"
    assert session.list_calls == 0


def test_duplicate_remote_identities_are_retained_and_conflicts_reject():
    one = record(); duplicate = dict(one, id="another-id")
    _, manifest, _ = run([one, duplicate])
    assert manifest["counts"]["exported_paths"] == 1
    assert manifest["counts"]["duplicate_remote_names"] == 1
    assert manifest["counts"]["exported_remote_files"] == 2
    bad = dict(duplicate, raw=one["raw"]+b" ")
    bad["sha256Checksum"] = hashlib.sha256(bad["raw"]).hexdigest()
    with pytest.raises(export.ExportUnavailable, match="REMOTE_IDENTITY_CONFLICT"):
        run([one, bad])
    with pytest.raises(export.ExportUnavailable, match="REMOTE_IDENTITY_CONFLICT"):
        run([one, dict(record(identity="synthetic-other"), id=one["id"])])


@pytest.mark.parametrize("kind", ["checksum", "invalid_checksum", "payload", "source", "key", "columns", "json"])
def test_corruption_rejects_entire_export(kind):
    item = record()
    if kind == "checksum": item["sha256Checksum"] = "0"*64
    elif kind == "invalid_checksum": item["sha256Checksum"] = "invalid"
    elif kind == "key": item["name"] = export.PREFIX+"prospective_event/"+"0"*64+".json"
    elif kind == "json": item["raw"] = b"corrupted"
    else:
        wire = json.loads(item["raw"])
        if kind == "columns": wire["columns"][0] = "altered"
        elif kind == "payload": wire["row"][wire["columns"].index("payload")] = '{"synthetic":false}'
        else: wire["row"][wire["columns"].index("raw_source")] = {"blob_base64": base64.b64encode(b"changed").decode()}
        item["raw"] = export.private._json(wire)
    if kind not in ("checksum", "invalid_checksum"):
        item["sha256Checksum"] = hashlib.sha256(item["raw"]).hexdigest()
    with pytest.raises(export.ExportUnavailable): run([item])


@pytest.mark.parametrize("kind", ["payload", "blob", "embedded_json", "configured", "environment"])
def test_credentials_reject_without_redacting_originals(kind, monkeypatch):
    values, forbidden = {}, ()
    secret = "synthetic-opaque-credential-value"
    if kind in ("payload", "embedded_json"):
        payload = json.dumps({"token": secret}) if kind == "payload" else json.dumps({"nested": json.dumps({"secret": secret})})
        values.update(payload=payload, payload_hash=hashlib.sha256(payload.encode()).hexdigest())
    elif kind == "blob":
        raw = json.dumps({"private_key": secret}).encode()
        values.update(raw_source=raw, source_hash=hashlib.sha256(raw).hexdigest())
    else:
        values["home_team"] = secret
        if kind == "configured": forbidden = (secret,)
        else: monkeypatch.setenv("SYNTHETIC_API_KEY", secret)
    item = record(**values)
    with pytest.raises(export.ExportUnavailable, match="CREDENTIAL_MATERIAL_DETECTED"):
        run([item], forbidden_values=forbidden)


def test_missing_dependencies_complete_partial_and_unrecorded_bindings():
    prediction = record("prospective_prediction", "synthetic-prediction",
        event_id="synthetic-missing", model_id="synthetic-model")
    _, manifest, _ = run([prediction])
    unresolved = manifest["dependencies"]["unresolved_references"]
    assert {d["field"] for d in unresolved} == {"event_id", "model_id"}
    assert all(d["status"] == "MISSING_FROM_COMPLETE_EXPORT" for d in unresolved)
    assert "quote_id" in manifest["dependencies"]["unrecorded_prediction_bindings"][0]["fields"]
    _, manifest, _ = run([prediction, record()], export.Limits(max_pages=1))
    assert all(d["status"] == "NOT_INCLUDED_REMOTE_UNKNOWN" for d in manifest["dependencies"]["unresolved_references"])


def test_present_dependencies_and_missing_provider_checksum():
    event = record(); event.pop("sha256Checksum")
    quote = record("prospective_quote", "synthetic-quote", event_id="synthetic-event")
    _, manifest, _ = run([event, quote])
    assert manifest["dependencies"]["declared_reference_count"] == 1
    assert not manifest["dependencies"]["unresolved_references"]


def test_configured_default_drive_path_counts_requests_and_closes_session(monkeypatch):
    from app_core import evidence_drive
    session = Session([record()])
    monkeypatch.setattr(evidence_drive, "_authorized_session", lambda: session)
    raw, manifest = export.build_download("synthetic-folder")
    assert raw and manifest["export_complete"]
    assert manifest["resources"]["requests"] == 3  # folder, listing, media
    assert session.closed


def test_default_access_failure_never_echoes_credentials(monkeypatch):
    from app_core import evidence_drive
    def denied(): raise RuntimeError("synthetic-private-key-access-message")
    monkeypatch.setattr(evidence_drive, "_authorized_session", denied)
    with pytest.raises(export.ExportUnavailable, match="^REMOTE_EXPORT_ACCESS_OR_INTEGRITY_FAILURE$"):
        export.build_download("synthetic-folder")


def test_manifest_credential_refusal_and_remote_scope_identities(monkeypatch):
    with pytest.raises(export.ExportUnavailable, match="CREDENTIAL_MATERIAL_DETECTED"):
        run([record()], forbidden_values=("synthetic-folder",))


def test_wire_contract_matches_existing_ddl_statically():
    root = Path(__file__).resolve().parents[1]
    columns, deps = {}, {}
    for filename in ("app_core/prospective_evidence.py", "app_core/prospective_reconciliation.py"):
        for node in ast.walk(ast.parse((root/filename).read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "executescript" and node.args and isinstance(node.args[0], ast.Constant)):
                continue
            for match in re.finditer(r"CREATE TABLE IF NOT EXISTS (prospective_\w+) \((.*?)\n\s*\);", node.args[0].value, re.S):
                table, body = match.groups(); fields, refs = [], []
                for line in body.splitlines():
                    field = re.match(r"\s*(\w+)\s+(TEXT|BLOB|REAL|INTEGER)\b", line)
                    if not field: continue
                    fields.append(field[1])
                    ref = re.search(r"REFERENCES (\w+)\((\w+)\)", line)
                    if ref: refs.append((field[1],ref[1],ref[2]))
                columns[table] = tuple(fields)
                if refs: deps[table] = tuple(refs)
    assert columns == COLUMNS and deps == DEPENDENCIES
    assert set(columns) == set(CANONICAL_PRIMARY_KEYS)


def owner_app():
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(None, None, lazy_history=True)


def test_actual_owner_gate_prepare_private_download_and_scope_invalidation(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import source_evidence_panel, activation_panel, public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "synthetic-owner-token")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "synthetic-folder")
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    calls = []
    original = export.build_download
    def actual(folder, **kwargs):
        calls.append(folder)
        return original(folder, **kwargs, store_factory=lambda name: DriveStore(name, session=Session([record()])))
    monkeypatch.setattr(export, "build_download", actual)
    at = AppTest.from_function(owner_app).run()
    assert not at.exception and not at.button and not calls
    at.text_input(key="publication_token").set_value("wrong").run()
    assert not at.button and not at.get("download_button") and not calls
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    assert not at.exception and any(b.key == "canonical_download_prepare" for b in at.button)
    at.button(key="remote_canonical_prepare").click().run()
    assert not at.exception and calls == ["synthetic-folder"]
    prepared = at.session_state["private_remote_canonical_download"]
    assert prepared["manifest"]["export_complete"] and len(at.get("download_button")) == 1
    assert "publication_preview" not in at.session_state
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "changed-folder")
    at.run()
    assert not at.get("download_button")
    at.text_input(key="publication_token").set_value("wrong").run()
    assert not at.button and not at.get("download_button")


def test_owner_partial_status_and_sanitized_access_failure(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import source_evidence_panel, activation_panel, public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "synthetic-owner-token")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "synthetic-folder")
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    original = export.build_download
    monkeypatch.setattr(export, "build_download", lambda folder, **kwargs:
        original(folder, **kwargs, limits=export.Limits(max_pages=1),
                 store_factory=lambda name: DriveStore(name, session=Session([record(), record(identity="two")]))))
    at = AppTest.from_function(owner_app).run()
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    at.button(key="remote_canonical_prepare").click().run()
    assert not at.exception and any("PARTIAL remote export" in w.value for w in at.warning)
    assert len(at.get("download_button")) == 1
    def denied(folder): raise RuntimeError("synthetic-secret-access-message")
    monkeypatch.setattr(export, "build_download", lambda folder, **kwargs:
        original(folder, **kwargs, store_factory=denied))
    at.button(key="remote_canonical_prepare").click().run()
    assert not at.exception and not at.get("download_button")
    assert any("REMOTE_EXPORT_ACCESS_OR_INTEGRITY_FAILURE" in e.value for e in at.error)
    assert all("synthetic-secret" not in e.value for e in at.error)


@pytest.mark.parametrize("kwargs", [{"max_pages":0}, {"max_seconds":float("nan")},
    {"max_metadata":True}, {"max_objects":1.5}, {"max_bytes":129*1024*1024}])
def test_limits_cannot_be_disabled_or_increased(kwargs):
    with pytest.raises(ValueError, match="INVALID_EXPORT_LIMIT"): export.Limits(**kwargs)
