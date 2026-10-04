"""Actual UI callers, 39k synthetic inventories and adversarial reads; network blocked."""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import time

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import public_results
from app_core import evidence_remote as remote
from app_core.evidence_drive import API, DriveInventory, DriveStore
from app_core.public_history import History, digest
from scripts.benchmark_drive_history_loading import SyntheticDrive, UI, setting, blocked_network, benchmark, Response


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("PARLAYPICKER_NETLIFY_SITE_ID", "site-1234")
    monkeypatch.setenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "synthetic-principal-a")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_PREFIX", "parlaypicker/evidence-v1")
    monkeypatch.setattr(remote, "_restored", set())
    monkeypatch.setattr(remote, "_verified", {})
    with blocked_network():
        yield


def ui_store(monkeypatch, session=None):
    session = session or SyntheticDrive()
    drive = DriveStore("folder", session=session)
    history = History("site-1234", "folder", drive)
    ui = UI()
    monkeypatch.setattr(public_results, "history", lambda unused:history)
    monkeypatch.setattr(public_results, "st", ui)
    return session, history, ui


def test_actual_restore_one_inventory_and_warm_bytes(monkeypatch):
    session, store, ui = ui_store(monkeypatch)
    assert public_results.restore_history(setting)
    assert not ui.errors and session.listings == 1 and session.downloads == 8
    assert "lock_removals" in ui.session_state["public_results_site-1234"]
    assert public_results.restore_history(setting)
    assert session.listings == 2 and session.downloads == 8
    assert store.client.last_read_report.objects_reused == 8
    assert public_results.restore_history(setting, full_verification=True)
    assert session.listings == 3 and session.downloads == 16


def test_benchmark_before_after(tmp_path):
    result = benchmark(tmp_path)
    assert result["baseline_ui_restore"]["inventories"] == 8
    assert result["baseline_lock_display"]["inventories"] == 1
    assert result["batched_ui_restore"]["inventories"] == 1
    assert result["batched_ui_restore"]["downloads"] == 8
    assert result["warm_explicit_restore"]["downloads"] == 0


def test_actual_completed_analysis_publish_caller_defers_history(monkeypatch):
    # Execute the completed-analysis publication block from the real app.
    tree = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                 and any(isinstance(call, ast.Call) and any(k.arg=="lazy_history" for k in call.keywords)
                         for call in ast.walk(node)))
    from test_publish_panel import app
    source = inspect.getsource(app).replace("def app():", "def main():")
    source = source.replace("    render_publish_panel(frame, None)", "    publication_games=frame\n"
        "    publication_props=None\n    publication_dfs={}\n    diagnostics={}\n"
        "    from streamlit_app import _publication_candidates\n"
        "    publish_tab=st.container()\n" + "\n".join("    "+line for line in ast.unparse(block).splitlines()))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("restore")
        raise AssertionError("Display-only rerun must not restore")
    monkeypatch.setattr(public_results, "restore_history", forbidden)
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", "test-only-publish-token")
    at = AppTest.from_string(source+"\nmain()").run()
    at.text_input(key="publication_token").set_value("test-only-publish-token").run()
    at.run()
    at.button(key="publication_build").click().run()
    assert not at.exception and not calls
    assert "publication_preview" in at.session_state
    assert at.button(key="public_history_load")


def test_lazy_display_reruns_zero_inventories_and_scope_invalidation(monkeypatch):
    session = SyntheticDrive()
    store = History("site-1234", "folder", DriveStore("folder", session=session))
    monkeypatch.setattr(public_results, "history", lambda unused:store)
    source = "from app.ui.public_results import render_history\nfrom scripts.benchmark_drive_history_loading import setting\nrender_history(setting,lazy=True)"
    at = AppTest.from_string(source).run()
    at.run()
    assert session.listings == session.downloads == 0
    at.button(key="public_history_load").click().run()
    assert not at.exception and session.listings == 1
    for _ in range(3):
        at.run()
    assert session.listings == 1 and session.downloads == 8
    monkeypatch.setattr(public_results, "display_scope", lambda unused:"changed-scope")
    at.run()
    assert not at.exception and session.listings == 1
    assert "public_results_site-1234" not in at.session_state


def test_recovery_changes_membership_requires_fresh_second_phase(monkeypatch):
    session, store, ui = ui_store(monkeypatch)
    session.files = [f for f in session.files if "/confirmed/" not in f["name"]]
    pending = next(json.loads(f["content"]) for f in session.files if "/deployments/" in f["name"])
    monkeypatch.setattr("app_core.netlify_publishing.deployment_status", lambda *args:{"state":"ready"})
    def confirm(deploy, package):
        session.add(store.prefix+"confirmed/"+deploy+".json",
                    dict(package_hash=package, confirmed_at="2026-10-03T15:01:00+00:00"))
    monkeypatch.setattr(store, "confirm", confirm)
    assert public_results.restore_history(lambda name:"synthetic-token" if name=="PARLAYPICKER_NETLIFY_TOKEN" else setting(name))
    assert session.listings == 2 and len(ui.session_state["public_results_site-1234"]["publications"]) == 1


@pytest.mark.parametrize("fault", ["corrupt", "duplicate", "incomplete", "malformed", "missing_package"])
def test_failed_restore_preserves_display_and_cannot_confirm(monkeypatch, fault):
    session, store, ui = ui_store(monkeypatch)
    old = {"preserved":True}
    ui.session_state["public_results_site-1234"] = old
    item = next(f for f in session.files if "/confirmed/" in f["name"])
    if fault == "corrupt":
        item["content"] = b"corrupt"
    elif fault == "duplicate":
        session.add(item["name"], {"different":True})
    elif fault == "incomplete":
        session.incomplete = True
    elif fault == "malformed":
        session.get = lambda *args, **kwargs:Response({})
    else:
        session.files = [f for f in session.files if "/packages/" not in f["name"]]
    monkeypatch.setattr(store, "confirm", lambda *args:pytest.fail("Corruption must not authorize recovery"))
    assert not public_results.restore_history(setting)
    assert ui.session_state["public_results_site-1234"] is old


def test_missing_checksum_downloads_every_phase_and_cache_corruption_reloads(monkeypatch):
    session, store, ui = ui_store(monkeypatch)
    item = next(f for f in session.files if "/scores/" in f["name"])
    item.pop("sha256Checksum")
    assert public_results.restore_history(setting)
    assert public_results.restore_history(setting)
    assert session.downloads == 9
    item = next(f for f in session.files if "/confirmed/" in f["name"])
    cache = store.client.verified_cache_root(store.cache_dir, store.prefix)/item["sha256Checksum"]
    cache.write_bytes(b"corrupt")
    assert public_results.restore_history(setting)
    assert session.downloads == 11


def test_equivalent_inflight_history_and_downloads_are_coalesced(monkeypatch):
    session = SyntheticDrive()
    drive = DriveStore("folder", session=session)
    store = History("site-1234", "folder", drive)
    entered, release = Event(), Event()
    original = drive._files
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        yield from original(*args, **kwargs)
    monkeypatch.setattr(drive, "_files", slow)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(store.history_phase)
        assert entered.wait(5)
        second = pool.submit(store.history_phase)
        time.sleep(.05)
        release.set()
        assert first.result() == second.result()
    assert session.listings == 1 and session.downloads == 8
    assert store.history_phase()
    assert session.listings == 2  # Completed membership is never cached.


def test_equivalent_media_read_joins_and_authority_inventory_stays_fresh(monkeypatch, tmp_path):
    session = SyntheticDrive()
    drive = DriveStore("folder", session=session)
    prefix = "parlaypicker/public-history-v1/site-1234/confirmed/"
    inventory = drive.discover_complete_inventory(namespace="parlaypicker/public-history-v1/site-1234/")
    entered, release = Event(), Event()
    original = drive._read_files
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(drive, "_read_files", slow)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(drive.read_verified_prefixes, Prefixes=[prefix], inventory=inventory)
        assert entered.wait(5)
        second = pool.submit(drive.read_verified_prefixes, Prefixes=[prefix], inventory=inventory)
        time.sleep(.05)
        fresh = drive.discover_complete_inventory(namespace=inventory.namespace, coalesce=False)
        release.set()
        assert first.result() == second.result()
    assert session.listings == 2 and session.downloads == 1


@pytest.mark.parametrize("change", ["principal", "folder", "namespace", "site"])
def test_scope_isolation_requires_new_bytes(monkeypatch, tmp_path, change):
    session = SyntheticDrive()
    first = DriveStore("folder", session=session)
    prefix = "parlaypicker/public-history-v1/site-1234/confirmed/"
    inv = first.discover_complete_inventory(namespace=prefix)
    first.read_verified_prefixes(Prefixes=[prefix], inventory=inv, cache_dir=tmp_path)
    if change == "principal":
        session.credentials.service_account_email = "other@example.invalid"
    if change == "folder":
        first.folder = "other-folder"
    namespace = prefix if change in {"principal", "folder"} else "parlaypicker/public-history-v1/"
    if change == "site":
        prefix = prefix.replace("site-1234", "site-5678")
        session.add(prefix+"deploy-2.json", {"new":True})
        namespace = prefix
    fresh = first.discover_complete_inventory(namespace=namespace)
    first.read_verified_prefixes(Prefixes=[prefix], inventory=fresh, cache_dir=tmp_path)
    assert session.downloads == 2
    if change in {"principal", "folder"}:
        with pytest.raises(ValueError, match="scope"):
            first.read_verified_prefixes(Prefixes=[prefix], inventory=inv, cache_dir=tmp_path)


def test_concurrent_removal_and_lock_membership_is_fresh(monkeypatch):
    session, store, ui = ui_store(monkeypatch)
    first = dict(id="one", date="2026-10-03", legs=[])
    second = dict(id="two", date="2026-10-03", legs=[])
    session.add(store.prefix+"locks/one.json", first)
    assert store.active_lock_snapshot().active == (first,)
    session.add(store.prefix+"lock_removals/one.json", {"lock_hash":digest(first)})
    session.add(store.prefix+"locks/two.json", second)
    assert store.active_lock_snapshot().active == (second,)
    assert session.listings == 2


def test_initialization_concurrent_first_generation_explicit_restore_and_failure(monkeypatch, tmp_path):
    session = SyntheticDrive()
    monkeypatch.setattr(remote, "_client", lambda:DriveStore("folder", session=session))
    path = tmp_path/"evidence.sqlite3"
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda unused:remote.restore_once(path), range(3)))
    assert session.listings == 1
    remote.restore(path)
    remote.restore_once(path)
    assert session.listings == 2
    session.incomplete = True
    with pytest.raises(ValueError):
        remote.restore(path, full_verification=True)
    session.incomplete = False
    remote.restore_once(path)
    assert session.listings == 4


@pytest.mark.parametrize("change", ["replacement", "principal", "folder", "namespace", "site"])
def test_initialization_invalidates_generation_and_scope(monkeypatch, tmp_path, change):
    session = SyntheticDrive()
    monkeypatch.setattr(remote, "_client", lambda:DriveStore("folder", session=session))
    path = tmp_path/"evidence.sqlite3"
    remote.restore_once(path)
    remote.restore_once(path)
    assert session.listings == 1
    if change == "replacement":
        from app_core.prediction_evidence import connect
        replacement = tmp_path/"replacement.sqlite3"
        connect(replacement).close()
        replacement.replace(path)
    else:
        name = {"principal":"PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT","folder":"PARLAYPICKER_DRIVE_FOLDER_ID",
                "namespace":"PARLAYPICKER_DRIVE_PREFIX","site":"PARLAYPICKER_NETLIFY_SITE_ID"}[change]
        monkeypatch.setenv(name, "changed-synthetic-scope")
    remote.restore_once(path)
    assert session.listings == 2


def test_market_instrumentation_returns_exact_values_and_errors(caplog):
    from app_core.market_stage_metrics import measured_call
    value = object()
    assert measured_call("fixture", lambda:value) is value
    with pytest.raises(ValueError, match="synthetic failure"):
        measured_call("fixture", lambda:(_ for _ in ()).throw(ValueError("synthetic failure")))
    assert any("market_enrichment_component" in record.message for record in caplog.records)


def test_repeated_page_token_and_no_cache_checksum_mismatch_fail(monkeypatch):
    session = SyntheticDrive()
    drive = DriveStore("folder", session=session)
    monkeypatch.setattr(drive, "_files", lambda *args, **kwargs:iter([
        {"id":"bad","name":"test/key","sha256Checksum":"a"*64}]))
    monkeypatch.setattr(drive, "_read_files", lambda *args, **kwargs:b"corrupt")
    with pytest.raises(ValueError, match="checksum changed"):
        drive.read_verified_prefixes(Prefixes=["test/"])
    monkeypatch.undo()
    session = SyntheticDrive()
    drive = DriveStore("folder", session=session)
    monkeypatch.setattr(session, "get", lambda *args, **kwargs:Response(
        {"files":[],"nextPageToken":"repeated"}))
    with pytest.raises(ValueError, match="repeated page token"):
        drive.discover_complete_inventory()


def test_equivalent_explicit_evidence_restores_share_inflight_work(monkeypatch, tmp_path):
    session = SyntheticDrive()
    drive = DriveStore("folder", session=session)
    entered, release = Event(), Event()
    original = drive._files
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        yield from original(*args, **kwargs)
    monkeypatch.setattr(drive, "_files", slow)
    path = tmp_path/"shared.sqlite3"
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(remote.restore, path, client=drive)
        assert entered.wait(5)
        second = pool.submit(remote.restore, path, client=drive)
        time.sleep(.05)
        release.set()
        assert first.result() == second.result() == 0
    assert session.listings == 1


def test_failed_shared_read_is_not_reused_on_retry(monkeypatch):
    session = SyntheticDrive()
    session.incomplete = True
    drive = DriveStore("folder", session=session)
    with pytest.raises(ValueError, match="incomplete"):
        drive.discover_complete_inventory(namespace="test/")
    session.incomplete = False
    assert drive.discover_complete_inventory(namespace="test/").complete
    assert session.listings == 2


def test_scope_changes_during_history_restore_cannot_relabel_display(monkeypatch):
    session, store, ui = ui_store(monkeypatch)
    values = {"PARLAYPICKER_NETLIFY_SITE_ID":"site-1234","PARLAYPICKER_DRIVE_FOLDER_ID":"folder"}
    old = {"preserved":True}
    ui.session_state["public_results_site-1234"] = old
    phase = store.history_phase
    def changed(**kwargs):
        result = phase(**kwargs)
        values["PARLAYPICKER_DRIVE_FOLDER_ID"] = "changed-folder"
        return result
    monkeypatch.setattr(store, "history_phase", changed)
    assert not public_results.restore_history(lambda name:values.get(name,""))
    assert ui.session_state["public_results_site-1234"] is old


def test_scope_change_during_initialization_invalidates_success(monkeypatch, tmp_path):
    session = SyntheticDrive()
    client = DriveStore("folder", session=session)
    monkeypatch.setattr(remote, "_client", lambda:client)
    original = remote._restore
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setenv("PARLAYPICKER_DRIVE_PREFIX", "changed-prefix")
        return result
    monkeypatch.setattr(remote, "_restore", changed)
    with pytest.raises(RuntimeError, match="scope or database generation changed"):
        remote.restore_once(tmp_path/"evidence.sqlite3")
    assert not remote._restored


@pytest.mark.parametrize("checksum", [None, "", "invalid-sha256"])
def test_missing_and_malformed_checksums_without_cache(checksum):
    session = SyntheticDrive()
    item = next(f for f in session.files if "/confirmed/" in f["name"])
    item["sha256Checksum"] = checksum
    drive = DriveStore("folder", session=session)
    prefix = "parlaypicker/public-history-v1/site-1234/confirmed/"
    if checksum == "invalid-sha256":
        with pytest.raises(ValueError, match="invalid SHA-256"):
            drive.read_verified_prefixes(Prefixes=[prefix])
    else:
        drive.read_verified_prefixes(Prefixes=[prefix])
        drive.read_verified_prefixes(Prefixes=[prefix])
        assert session.downloads == 2
