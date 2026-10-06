"""Synthetic stores only; actual exporter and authenticated UI, network blocked."""
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
import zipfile

import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from app_core import canonical_download as download
from app_core import prospective_evidence as evidence


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def store(tmp_path):
    path = tmp_path / download.NAME
    with closing(evidence.connect(path)) as db:
        db.commit()
    return path


def insert(db, identity="synthetic-event", raw=b'{"fixture":"synthetic"}'):
    payload = json.dumps({"event_id": identity, "fixture": "synthetic"})
    db.execute("""INSERT INTO prospective_event
        (event_id,sport,game_id,provider_namespace,provider_event_id,home_team,away_team,
         scheduled_start,observed_at,ingested_at,source_id,source_hash,raw_source,payload,payload_hash)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (identity,"MLB",identity,"synthetic",identity,
        "Synthetic Home","Synthetic Away","2026-10-06T22:00:00Z","2026-10-06T17:00:00Z",
        "2026-10-06T17:01:00Z","synthetic-source",hashlib.sha256(raw).hexdigest(),raw,payload,
        hashlib.sha256(payload.encode()).hexdigest()))


def unpack(raw, tmp_path):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert set(archive.namelist()) == {download.NAME, "manifest.json"}
        snapshot = archive.read(download.NAME)
        manifest = json.loads(archive.read("manifest.json"))
    assert hashlib.sha256(snapshot).hexdigest() == manifest["snapshot"]["sha256"]
    assert len(snapshot) == manifest["snapshot"]["bytes"]
    assert hashlib.sha256(download._json(manifest["schema_objects"])).hexdigest() == manifest["schema_sha256"]
    output = tmp_path / "downloaded.sqlite3"
    output.write_bytes(snapshot)
    return output, manifest


def test_committed_wal_uncommitted_transaction_and_original_bytes(tmp_path):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        original = b'{ "fixture": "synthetic", "original_spacing": true }'
        insert(writer, raw=original)
        writer.commit()
        insert(writer, "synthetic-uncommitted")
        before = {p.name: p.read_bytes() for p in (path, Path(str(path)+"-wal"))}
        raw, returned = download.build_download(path)
        copied, manifest = unpack(raw, tmp_path)
        assert returned == manifest
        with closing(sqlite3.connect(copied)) as db:
            rows = db.execute("SELECT event_id,raw_source FROM prospective_event").fetchall()
        assert rows == [("synthetic-event", original)]
        assert manifest["record_counts"]["prospective_event"] == 1
        assert manifest["available_references"][0]["identity"] == {"event_id": "synthetic-event"}
        assert not manifest["scientific_acceptance"] and not manifest["wagering_authority"]
        assert all(p.read_bytes() == before[p.name] for p in (path, Path(str(path)+"-wal")))
        writer.rollback()


def test_coherent_backup_with_concurrent_committed_rows(tmp_path):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as setup:
        setup.execute("PRAGMA journal_mode=WAL")
    ready, finish = threading.Event(), threading.Event()
    failures = []
    def write():
        try:
            with closing(sqlite3.connect(path)) as db:
                insert(db, "synthetic-first"); db.commit(); ready.set()
                finish.wait(5)
                insert(db, "synthetic-later"); db.commit()
        except Exception as exc:
            failures.append(exc); ready.set()
    thread = threading.Thread(target=write); thread.start()
    try:
        assert ready.wait(5) and not failures
        raw, _ = download.build_download(path)
        copied, manifest = unpack(raw, tmp_path)
        finish.set(); thread.join(5)
        assert not thread.is_alive() and not failures
        with closing(sqlite3.connect(copied)) as db:
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert db.execute("SELECT count(*) FROM prospective_event").fetchone()[0] == manifest["record_counts"]["prospective_event"] == 1
    finally:
        finish.set(); thread.join(5)


def test_model_and_dependency_references_without_raw_manifest(tmp_path):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as db:
        insert(db)
        payload = json.dumps({"artifact_hash": "synthetic-artifact", "dependencies": [{"source_hash": "synthetic-dependency", "available_at": "2026-10-05T00:00:00Z", "raw_secret_free_dependency": "private-original"}], "fixture": "synthetic"})
        db.execute("""INSERT INTO prospective_model VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("synthetic-model", "MLB", "RUN_LINE", "synthetic-v1", "2023-01-01T00:00:00Z",
             "2025-01-01T00:00:00Z", 0, 0, "synthetic-features", "synthetic-code",
             "2026-10-05T00:00:00Z", "2026-10-05T00:00:00Z", "synthetic-artifact", payload,
             hashlib.sha256(payload.encode()).hexdigest()))
        db.commit()
    raw, manifest = download.build_download(path)
    copied, _ = unpack(raw, tmp_path)
    refs = next(row for row in manifest["available_references"] if row["table"] == "prospective_model")
    assert refs["references"]["artifact_hash"] == "synthetic-artifact"
    assert refs["references"]["payload_references"]["dependencies"] == [{"available_at": "2026-10-05T00:00:00Z", "source_hash": "synthetic-dependency"}]
    assert b"private-original" not in download._json(manifest)
    assert manifest["external_reference_contents"] == "NOT_INCLUDED_AVAILABILITY_UNKNOWN"
    with closing(sqlite3.connect(copied)) as db:
        assert db.execute("SELECT payload FROM prospective_model").fetchone()[0] == payload


@pytest.mark.parametrize("content", [b'{"api_key":"synthetic-key"}', b'{"Authorization":"Bearer synthetic"}', b'https://example.invalid/?apiKey=synthetic', b'-----BEGIN PRIVATE KEY-----synthetic', b'https://user:password@example.invalid/', json.dumps({"nested": json.dumps({"token": "synthetic-old-token"})}).encode()])
def test_credential_material_refuses_without_redacting(tmp_path, content):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as db:
        insert(db, raw=content); db.commit()
    before = path.read_bytes()
    with pytest.raises(download.DownloadUnavailable, match="CREDENTIAL_MATERIAL_DETECTED"):
        download.build_download(path)
    assert path.read_bytes() == before


def test_opaque_active_credential_refused(tmp_path, monkeypatch):
    path = store(tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "synthetic-opaque-credential")
    with closing(sqlite3.connect(path)) as db:
        insert(db, raw=b"synthetic-opaque-credential"); db.commit()
    with pytest.raises(download.DownloadUnavailable, match="CREDENTIAL_MATERIAL_DETECTED"):
        download.build_download(path)


def test_credential_column_refused_without_mutating_original(tmp_path):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("ALTER TABLE prospective_event ADD COLUMN api_key TEXT DEFAULT 'synthetic-expired-opaque-credential'")
        insert(db)
        db.commit()
    before = path.read_bytes()
    with pytest.raises(download.DownloadUnavailable, match="CREDENTIAL_MATERIAL_DETECTED"):
        download.build_download(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode,reason", [("missing","LOCAL_CANONICAL_STORE_MISSING_REMOTE_UNKNOWN"),("corrupt","SNAPSHOT_STORAGE_OR_INTEGRITY_FAILURE"),("unknown_table","CANONICAL_SCHEMA_UNSUPPORTED"),("wrong_schema","CANONICAL_SCHEMA_UNSUPPORTED"),("wrong_store","WRONG_STORE")])
def test_missing_corrupt_or_unsupported_never_initializes(tmp_path, mode, reason):
    path = tmp_path / download.NAME
    if mode == "corrupt": path.write_bytes(b"synthetic corrupt bytes")
    elif mode == "unknown_table":
        path = store(tmp_path)
        with closing(sqlite3.connect(path)) as db: db.execute("CREATE TABLE credentials (value TEXT)"); db.commit()
    elif mode == "wrong_schema":
        with closing(sqlite3.connect(path)) as db: db.execute("CREATE TABLE prospective_event (wrong TEXT PRIMARY KEY)"); db.commit()
    elif mode == "wrong_store": path = tmp_path / "evidence.sqlite3"
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(download.DownloadUnavailable, match=reason): download.build_download(path)
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_limits_and_reference_truncation_are_explicit(tmp_path):
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as db: insert(db); db.commit()
    for kwargs, code in [({"max_bytes":1}, "SNAPSHOT_SIZE_LIMIT"), ({"max_seconds":0}, "SNAPSHOT_TIME_LIMIT")]:
        with pytest.raises(download.DownloadUnavailable, match=code): download.build_download(path, **kwargs)
    _, manifest = download.build_download(path, max_references=0)
    assert manifest["references_truncated"] and manifest["reference_rows"] == 1
    assert manifest["available_references"] == []


def test_busy_store_has_bounded_refusal_and_no_source_mutation(tmp_path):
    path = store(tmp_path)
    before = path.read_bytes()
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        with pytest.raises(download.DownloadUnavailable, match="SNAPSHOT_TIME_LIMIT"):
            download.build_download(path, max_seconds=0.01)
        writer.rollback()
    assert path.read_bytes() == before


def test_empty_directory_setting_preserves_actual_producer_resolution(tmp_path, monkeypatch):
    path = store(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", "")
    assert download.effective_path().resolve() == path.resolve()
    _, manifest = download.build_download()
    assert manifest["source_configuration"] == "PARLAYPICKER_EVIDENCE_DIR"
    monkeypatch.delenv("PARLAYPICKER_EVIDENCE_DIR")
    assert download.effective_path().is_absolute()
    assert download.effective_path().parent.name == "prediction_evidence"


def test_inaccessible_probe_is_unknown_and_sanitized(tmp_path, monkeypatch):
    path = tmp_path / download.NAME
    def denied(self):
        raise PermissionError("synthetic-private-path-and-credential")
    monkeypatch.setattr(Path, "is_file", denied)
    with pytest.raises(download.DownloadUnavailable, match="^LOCAL_CANONICAL_STORE_INACCESSIBLE_REMOTE_UNKNOWN$"):
        download.build_download(path)
    assert not list(tmp_path.iterdir())


def owner_app():
    import streamlit as st
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(None, None, lazy_history=True)


def test_actual_owner_gate_empty_analysis_and_scope_invalidation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import source_evidence_panel, activation_panel, public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "synthetic-owner-token")
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    path = store(tmp_path)
    with closing(sqlite3.connect(path)) as db: insert(db); db.commit()
    at = AppTest.from_function(owner_app).run()
    assert not at.exception and not at.button and not at.get("download_button")
    at.text_input(key="publication_token").set_value("wrong").run()
    assert not at.button and not at.get("download_button")
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    assert not at.exception and any(b.key == "canonical_download_prepare" for b in at.button)
    assert not at.get("download_button")
    at.button(key="canonical_download_prepare").click().run()
    assert not at.exception and len(at.get("download_button")) == 1
    prepared = at.session_state["private_canonical_download"]
    assert prepared["manifest"]["record_counts"]["prospective_event"] == 1
    assert "publication_preview" not in at.session_state
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path / "other"))
    at.run()
    assert not at.get("download_button")
    at.button(key="canonical_download_prepare").click().run()
    assert not at.exception and not (tmp_path/"other").exists()
    assert any("LOCAL_CANONICAL_STORE_MISSING_REMOTE_UNKNOWN" in error.value for error in at.error)


def test_owner_diagnostic_does_not_echo_storage_exception(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import source_evidence_panel, activation_panel, public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "synthetic-owner-token")
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    (tmp_path/download.NAME).write_bytes(b"synthetic-secret-corruption")
    at=AppTest.from_function(owner_app).run()
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    at.button(key="canonical_download_prepare").click().run()
    assert not at.exception
    assert "synthetic-secret-corruption" not in at.error[0].value
    assert not at.get("download_button")


def test_actual_owner_preparation_resolves_after_lazy_configuration_load(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import publish_panel, source_evidence_panel, activation_panel, public_results
    old, effective = tmp_path/"before-secrets", tmp_path/"effective-producer"
    old.mkdir(); effective.mkdir()
    old_path, effective_path = store(old), store(effective)
    with closing(sqlite3.connect(old_path)) as db: insert(db, "synthetic-old-location"); db.commit()
    with closing(sqlite3.connect(effective_path)) as db:
        insert(db, "synthetic-effective-one"); insert(db, "synthetic-effective-two"); db.commit()
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(old))
    def lazy_setting(name, default=""):
        if name == "PARLAYPICKER_PUBLISH_TOKEN": return "synthetic-owner-token"
        if name == "ODDS_API_KEY":
            monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(effective))
            return "synthetic-opaque-provider-credential"
        return default
    monkeypatch.setattr(publish_panel, "setting", lazy_setting)
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    at=AppTest.from_function(owner_app).run()
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    at.button(key="canonical_download_prepare").click().run()
    assert not at.exception
    prepared=at.session_state["private_canonical_download"]
    assert prepared["binding"] == str(effective_path.resolve())
    assert prepared["manifest"]["record_counts"]["prospective_event"] == 2
    assert len(at.get("download_button")) == 1
    at.run()
    assert len(at.get("download_button")) == 1
