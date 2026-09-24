from copy import deepcopy
import pytest
import requests
from app_core import research_model_recovery as recovery
from app_core import research_scheduler as scheduler
from app_core.evidence_drive import _read


def test_read_retry_is_bounded_and_auth_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr("app_core.evidence_drive.time.sleep", lambda _: None)
    class Session:
        calls = 0
        def get(self, *a, **kw):
            self.calls += 1
            raise requests.ReadTimeout("private url")
    session = Session()
    with pytest.raises(requests.ReadTimeout):
        _read(session, "unused", timeout=20)
    assert session.calls == 3
    response = requests.Response()
    response.status_code = 403
    session.calls = 0
    def forbidden(*a, **kw):
        session.calls += 1
        return response
    session.get = forbidden
    with pytest.raises(requests.HTTPError):
        _read(session, "unused")
    assert session.calls == 1


def test_transient_read_recovers(monkeypatch):
    monkeypatch.setattr("app_core.evidence_drive.time.sleep", lambda _: None)
    class Session:
        calls = 0
        def get(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                raise requests.ReadTimeout()
            r = requests.Response()
            r.status_code = 200
            return r
    session = Session()
    assert _read(session, "unused").status_code == 200
    assert session.calls == 2


def model():
    artifact = {"protocol": {"production_eligible": False}}
    return {"kind": "model", "id": "old", "data": {"artifact": artifact,
        "artifact_hash": recovery.ARTIFACT, "runtime_hash": recovery.SOURCE_RUNTIME}}


def test_reviewed_recovery_appends_new_cohort_and_is_idempotent(monkeypatch, tmp_path):
    original = model()
    before = deepcopy(original)
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: recovery.TARGET_RUNTIME)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    path = tmp_path / "cohorts.sqlite3"
    new = recovery.recover_ncaaf(original, path)
    assert original == before
    assert new["id"] != original["id"] and new["created_at"]
    assert new["data"]["artifact"] == original["data"]["artifact"]
    assert new["data"]["recovery"]["production_eligible"] is False
    assert recovery.recover_ncaaf(original, path)["id"] == new["id"]
    assert recovery.recover_ncaaf(new, path) == new
    assert len(recovery.store.records(path)) == 1


@pytest.mark.parametrize("fault", ["source", "target", "artifact", "production"])
def test_unknown_or_invalid_model_never_recovers(monkeypatch, tmp_path, fault):
    record = model()
    target = recovery.TARGET_RUNTIME
    digest = recovery.ARTIFACT
    if fault == "source": record["data"]["runtime_hash"] = "unknown"
    if fault == "target": target = "unknown"
    if fault == "artifact": digest = "tampered"
    if fault == "production": record["data"]["artifact"]["protocol"]["production_eligible"] = True
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: target)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: digest)
    with pytest.raises(ValueError, match="stale_frozen_model"):
        recovery.recover_ncaaf(record, tmp_path / "absent.sqlite3")
    assert not (tmp_path / "absent.sqlite3").exists()


def test_scheduler_reports_restore_and_backup_stage_without_secrets(monkeypatch, tmp_path):
    from test_research_scheduler import Cloud
    monkeypatch.setattr(scheduler, "is_open", lambda: True)
    def fail(*a, **kw): raise requests.ReadTimeout("secret-token")
    monkeypatch.setattr(scheduler.ms, "sync", fail)
    result = scheduler.run(["MLB"], tmp_path, Cloud(), "folder")
    assert result["failure_stages"]["MLB"] == {"stage": "restore", "code": "ReadTimeout"}
    assert "secret-token" not in str(result)
    calls = []
    def backup(*a, **kw):
        calls.append(1)
        if len(calls) > 1: fail()
    monkeypatch.setattr(scheduler.ms, "sync", backup)
    monkeypatch.setattr(scheduler, "run_mlb", lambda *a: {"errors": []})
    result = scheduler.run(["MLB"], tmp_path, Cloud(), "folder")
    assert result["failure_stages"]["MLB"]["stage"] == "backup"


def test_schedule_failure_identifies_provider_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler.ms, "records", lambda *a: [{"kind": "model", "data": {"runtime_hash": "h"}}])
    monkeypatch.setattr(scheduler.mlb, "runtime_hash", lambda: "h")
    monkeypatch.setattr(scheduler, "retry_schedule", lambda f: f())
    def fail(): raise requests.ReadTimeout("private url")
    monkeypatch.setattr(scheduler, "upcoming_mlb", fail)
    with pytest.raises(scheduler.ResearchStageError) as error:
        scheduler.run_mlb(tmp_path / "x", {}, lambda: None)
    assert error.value.stage == "mlb_schedule"
    assert error.value.code == "ReadTimeout"
    assert "private" not in str(error.value)


def test_uncertain_upload_is_never_retried(monkeypatch):
    from test_evidence_drive import DriveSession
    from app_core.evidence_drive import DriveStore
    session = DriveSession()
    store = DriveStore("folder", session=session)
    calls = []
    def upload(*a, **kw):
        calls.append(1)
        raise requests.ReadTimeout("private upload")
    monkeypatch.setattr(session, "post", upload)
    with pytest.raises(requests.ReadTimeout):
        store.put_object(Key="record", Body=b"{}", IfNoneMatch="*")
    assert len(calls) == 1


def test_canonical_gap_repairs_only_from_independent_source_backup(monkeypatch, tmp_path):
    from test_research_scheduler import Cloud
    from app_core import prospective_remote, prospective_reconciliation, prospective_validation_plans
    from app_core.prospective_sport_adapters import get_adapter

    calls = []
    def sync(*args):
        calls.append("canonical_sync")
        if calls.count("canonical_sync") == 1:
            raise prospective_remote.CanonicalMissingDependencies({"NFL"})
        return {"records_verified": 0}
    monkeypatch.setattr(prospective_remote, "sync", sync)
    monkeypatch.setattr(prospective_validation_plans, "freeze_current_validation_plans",
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(prospective_reconciliation, "reconcile_sport",
                        lambda *args: calls.append("reconcile") or {})
    adapter = get_adapter("NFL")
    monkeypatch.setattr(adapter, "restore", lambda *args: calls.append("native_restore") or {})
    monkeypatch.setattr(adapter, "backup", lambda *args: calls.append("native_backup") or {})
    monkeypatch.setattr(adapter, "run_cycle", lambda *args: {"errors": []})

    result = scheduler.run(["NFL"], tmp_path, Cloud(), "folder")
    assert result["canonical_repair_sports"] == ["NFL"]
    assert result["requested_slate_success"] is True
    assert calls[:4] == ["canonical_sync", "native_restore", "reconcile", "canonical_sync"]


def test_canonical_failure_code_has_no_provider_text():
    assert scheduler.canonical_failure_code(ValueError("canonical_remote_foreign_key_conflict")) == \
        "CANONICAL_REMOTE_FOREIGN_KEY_CONFLICT"
    assert scheduler.canonical_failure_code(RuntimeError("https://private.example/api-key")) == \
        "RUNTIMEERROR"
    response = requests.Response()
    response.status_code = 429
    error = requests.HTTPError("https://private.example/api-key", response=response)
    assert scheduler.canonical_failure_code(error) == "DRIVE_STATUS_429"
