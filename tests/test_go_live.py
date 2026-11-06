"""Go-live checklist (C4 prep): read-only status per requirement. SYNTHETIC inputs."""
from datetime import datetime, timezone

import pytest

from app_core import go_live
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe

NOW = datetime(2026, 12, 20, tzinfo=timezone.utc)
VALIDATED = dict(sport="NCAAB", market_family="SPREAD", deployment_state="PROVISIONAL_VALIDATED",
                 validation_state="PROVISIONAL_VALIDATED", validation_id="validation-1", artifact_id="artifact-1",
                 model_id="model-1", model_version="v1", calibration_id="cal-1", calibration_version="v1")


def _by_key(items):
    return {item["key"]: item for item in items}


def test_empty_store_lists_every_step_and_writes_nothing(tmp_path):
    store, ledger = tmp_path / "store.sqlite3", tmp_path / "ledger.sqlite3"
    items = go_live.checklist("NCAAB", "SPREAD", path=store, ledger_path=ledger, now=NOW,
                              setting=lambda *a: None, activations_dir=tmp_path / "active")
    assert [i["key"] for i in items] == ["bound_plan", "validated", "exposure_ledger", "activation",
                                         "decision_capture", "bridge", "drive", "gemini"]
    s = _by_key(items)
    assert s["bound_plan"]["status"] == "ACTION" and s["validated"]["status"] == "WAIT"
    assert s["activation"]["status"] == "WAIT" and s["decision_capture"]["status"] == "WAIT"
    assert all(i["next_step"] for i in items if i["status"] != "PASS")
    assert not store.exists() and not ledger.exists() and not (tmp_path / "active").exists()
    head = go_live.summary(items)
    assert head["ready"] is False and head["next"]["key"] == "bound_plan"
    assert "Go-live checklist: NCAAB SPREAD" in go_live.markdown("NCAAB", "SPREAD", items)


def test_validated_market_points_at_the_owner_activation_then_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "bound_plan", lambda *a: {"validation_plan_id": "plan", "version": 2})
    monkeypatch.setattr(pe, "deployment_state", lambda *a: dict(VALIDATED))
    monkeypatch.setattr(pe, "market_readiness", lambda *a: {"validation_count": 210, "holdout_count": 205})
    monkeypatch.setattr(pe, "read_records", lambda *a, **k: [
        {"observation_id": "decision-prediction:x", "prediction_timestamp": "2026-12-19T23:45:00+00:00"},
        {"observation_id": "bound-prediction:y", "prediction_timestamp": "2026-12-01T00:00:00+00:00"}])
    snapshot = {"bankroll": 1000.0, "currency": "USD", "unit_value": 10.0}
    monkeypatch.setattr("core.exposure_ledger.snapshot", lambda path, now=None: snapshot)
    settings = {"PARLAYPICKER_QUALIFIED_BRIDGE": "true", "PARLAYPICKER_DRIVE_FOLDER_ID": "folder",
                "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT": "{...}", "GEMINI_API_KEY": "k"}
    active = tmp_path / "active"

    def run():
        return _by_key(go_live.checklist("NCAAB", "SPREAD", now=NOW, setting=lambda n, d=None: settings.get(n, d),
                                         activations_dir=active))
    s = run()
    assert s["validated"]["status"] == "PASS" and s["exposure_ledger"]["status"] == "PASS"
    assert s["activation"]["status"] == "ACTION" and "no record" in s["activation"]["detail"]
    assert s["decision_capture"]["status"] == "PASS"
    active.mkdir()
    (active / "NCAAB-SPREAD.json").write_text("{}")
    seen = {}

    def verify(record, state, exposure, now=None):
        seen.update(state=state, exposure=exposure)
        return "policy"
    monkeypatch.setattr("core.sport_market_activation.verify_market_activation", verify)
    s = run()
    assert s["activation"]["status"] == "PASS" and seen["exposure"] is snapshot
    assert go_live.summary(list(s.values()))["ready"] is True
    monkeypatch.setattr("core.sport_market_activation.verify_market_activation",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("Owner market activation expired or invalid")))
    assert "expired" in run()["activation"]["detail"]


def test_stale_decision_capture_and_secret_values_never_shown(monkeypatch):
    monkeypatch.setattr(pc, "bound_plan", lambda *a: None)
    monkeypatch.setattr(pe, "deployment_state", lambda *a: dict(VALIDATED))
    monkeypatch.setattr(pe, "market_readiness", lambda *a: {})
    monkeypatch.setattr(pe, "read_records", lambda *a, **k: [
        {"observation_id": "decision-prediction:x", "prediction_timestamp": "2026-12-01T00:00:00+00:00"}])
    items = go_live.checklist("NCAAB", "SPREAD", now=NOW, setting=lambda n, d=None: {"GEMINI_API_KEY": "secret-value"}.get(n, d))
    s = _by_key(items)
    assert s["decision_capture"]["status"] == "ACTION" and "PARLAYPICKER_DECISION_MARKETS" in s["decision_capture"]["next_step"]
    assert s["gemini"]["status"] == "PASS" and "secret-value" not in str(items)
    with pytest.raises(ValueError):
        go_live.checklist("NCAAB", "PUCK_LINE")
