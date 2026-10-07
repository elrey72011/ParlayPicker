"""Synthetic actual exporter/UI path; no historical inference or real network."""
import io
import json
import pytest
import zipfile

from app_core import remote_canonical_download as export
from app_core.evidence_drive import DriveStore
from test_remote_canonical_download import Session, record, owner_app, isolated


def run(items, **kwargs):
    session = Session(items)
    data, manifest = export.build_download("synthetic-folder",
        store_factory=lambda name: DriveStore(name, session=session), **kwargs)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        assert json.loads(z.read("manifest.json")) == manifest
        for entry in manifest["objects"]:
            assert z.read(entry["path"]) == next(i["raw"] for i in items if i["name"] == entry["path"])
    return data, manifest, session


def test_reproduced_timeout_starvation_can_target_research_sources(monkeypatch):
    coverage = [record("prospective_football_coverage", "synthetic-coverage-"+str(i)) for i in range(4)]
    source = record("prospective_reconciled_source", "synthetic-original-research-source")
    items = coverage+[source]
    now = [0]
    class SlowSession(Session):
        def get(self, url, **kwargs):
            result = super().get(url, **kwargs)
            if self.media_calls >= 3:
                now[0] = 301
            return result
    monkeypatch.setattr(export.time, "monotonic", lambda: now[0])
    session = SlowSession(items)
    _, before = export.build_download("synthetic-folder",
        store_factory=lambda name: DriveStore(name, session=session))
    assert before["stop_reason"] == "TIME_LIMIT"
    assert before["counts"]["rows_by_table"] == {"prospective_football_coverage":2}
    assert not before["export_complete"]
    now[0] = 0
    _, after, session = run(items, table="prospective_reconciled_source")
    assert after["selection_complete"] and not after["export_complete"]
    assert after["counts"]["rows_by_table"] == {"prospective_reconciled_source":1}
    assert session.media_calls == 1
    assert after["request_scope"]["listed_paths_outside_requested_range"] == 4
    assert after["inventory"]["canonical_paths_by_table"] == {
        "prospective_football_coverage":4, "prospective_reconciled_source":1}


def test_continuation_reads_each_original_once_without_claiming_whole_store_complete():
    items = [record(identity="synthetic-event-"+str(i)) for i in range(5)]
    received, cursor, pages = [], None, []
    while True:
        _, m, session = run(items, limits=export.Limits(max_objects=2), start_after=cursor)
        received += [entry["path"] for entry in m["objects"]]
        pages.append(m)
        assert session.media_calls == len(m["objects"])
        assert not m["export_complete"]
        cursor = m["continuation"]["next_start_after"]
        if cursor is None:
            break
    assert received == sorted(i["name"] for i in items)
    assert len(set(received)) == 5
    assert [m["counts"]["exported_paths"] for m in pages] == [2,2,1]
    assert [m["selection_complete"] for m in pages] == [False,False,True]
    assert pages[-1]["continuation"]["cross_download_consistency"] == "NOT_AN_ATOMIC_REMOTE_SNAPSHOT"


def test_full_default_compatibility_empty_scoped_table_and_dependencies_remain_unknown():
    event = record()
    prediction = record("prospective_prediction", "synthetic-prediction",
                        event_id="synthetic-event", model_id="synthetic-unseen-model")
    _, default, _ = run([event, prediction])
    assert default["export_complete"] and default["selection_complete"]
    _, filtered, _ = run([event, prediction], table="prospective_prediction")
    assert not filtered["export_complete"] and filtered["selection_complete"]
    assert {d["field"] for d in filtered["dependencies"]["unresolved_references"]} == {"event_id","model_id"}
    assert all(d["status"] == "NOT_INCLUDED_REMOTE_UNKNOWN" for d in filtered["dependencies"]["unresolved_references"])
    _, empty, session = run([event], table="prospective_model")
    assert empty["selection_complete"] and not empty["export_complete"]
    assert empty["counts"]["exported_paths"] == session.media_calls == 0


def test_incomplete_listing_never_supplies_a_continuation_cursor():
    _, manifest, _ = run([record(),record(identity="synthetic-two")],
                        limits=export.Limits(max_pages=1))
    assert not manifest["inventory"]["complete"]
    assert manifest["continuation"]["next_start_after"] is None
    assert not manifest["selection_complete"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"table":"not-a-table"}, "INVALID_CANONICAL_EXPORT_TABLE"),
    ({"table":True}, "INVALID_CANONICAL_EXPORT_TABLE"),
    ({"start_after":"../../credentials.json"}, "INVALID_CANONICAL_EXPORT_CURSOR"),
    ({"start_after":True}, "INVALID_CANONICAL_EXPORT_CURSOR"),
    ({"table":"prospective_model", "start_after":record()["name"]}, "INVALID_CANONICAL_EXPORT_CURSOR"),
])
def test_invalid_scope_refuses_before_access_without_echoing_input(kwargs, reason):
    def forbidden(*args): pytest.fail("Invalid scope must not access remote storage")
    with pytest.raises(export.ExportUnavailable, match="^"+reason+"$"):
        export.build_download("synthetic-folder", store_factory=forbidden, **kwargs)


def test_new_remote_objects_before_cursor_are_explicitly_outside_range():
    items = sorted([record(identity="synthetic-"+str(i)) for i in range(3)], key=lambda i:i["name"])
    _, manifest, _ = run(items, start_after=items[1]["name"])
    assert manifest["selection_complete"] and not manifest["export_complete"]
    assert manifest["request_scope"]["listed_paths_outside_requested_range"] == 2
    assert [e["path"] for e in manifest["objects"]] == [items[2]["name"]]


@pytest.mark.parametrize("attack", ["corruption", "duplicate_conflict", "credential"])
def test_selected_originals_still_fail_closed(attack):
    import hashlib
    source = record("prospective_reconciled_source", "synthetic-source")
    items = [record("prospective_football_coverage"), source]
    if attack == "corruption":
        source["raw"] += b" "
        source["sha256Checksum"] = hashlib.sha256(source["raw"]).hexdigest()
    elif attack == "duplicate_conflict":
        raw = source["raw"]+b" "
        items.append(dict(source,id="duplicate-source",raw=raw,sha256Checksum=hashlib.sha256(raw).hexdigest()))
    else:
        raw = b'{"private_key":"synthetic-secret"}'
        items[1] = record("prospective_reconciled_source", "synthetic-source",
                          raw_source=raw,source_hash=hashlib.sha256(raw).hexdigest())
    with pytest.raises(export.ExportUnavailable):
        run(items,table="prospective_reconciled_source")


def test_owner_scope_changes_invalidate_zip_and_wrong_token_has_no_controls(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import source_evidence_panel, activation_panel, public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "synthetic-owner-token")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "synthetic-folder")
    monkeypatch.setattr(source_evidence_panel, "render", lambda: None)
    monkeypatch.setattr(activation_panel, "render", lambda games: None)
    monkeypatch.setattr(public_results, "render_history", lambda *args, **kwargs: [])
    original = export.build_download
    calls = []
    def actual(folder, **kwargs):
        calls.append(kwargs)
        return original(folder, **kwargs,
            store_factory=lambda name: DriveStore(name, session=Session([record()])))
    monkeypatch.setattr(export, "build_download", actual)
    at = AppTest.from_function(owner_app).run()
    assert not calls and not any(s.key == "remote_canonical_table" for s in at.selectbox)
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    at.button(key="remote_canonical_prepare").click().run()
    assert not at.exception and len(at.get("download_button")) == 1
    at.selectbox(key="remote_canonical_table").set_value("prospective_model").run()
    assert not at.get("download_button")
    at.button(key="remote_canonical_prepare").click().run()
    assert not at.exception and calls[-1]["table"] == "prospective_model"
    assert any("scoped download" in i.value for i in at.info)
    assert len(at.get("download_button")) == 1
    at.text_input(key="remote_canonical_cursor").set_value(record("prospective_model")["name"]).run()
    assert not at.get("download_button")
    at.text_input(key="publication_token").set_value("wrong").run()
    assert not at.get("download_button") and not any(s.key == "remote_canonical_table" for s in at.selectbox)
