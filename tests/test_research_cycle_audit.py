import json
import uuid

from app_core.research_cycle_audit import sanitize_cycle


SPORTS = ["NFL", "NCAAF", "NBA", "NCAAB", "MLB", "NHL"]


def test_six_sport_audit_is_complete_and_redacts_untrusted_provider_text():
    report = {"run_id": str(uuid.uuid4()), "source_commit": "a" * 40,
              "started_at": "2026-09-23T12:00:00+00:00",
              "finished_at": "2026-09-23T12:01:00+00:00",
              "requested_sports": SPORTS, "requested_slate_success": False,
              "errors": ["NBA:https://private.example/secret-api-key"],
              "health": {sport: {"restore": "success", "capture": "success",
                                 "grade": "success", "backup": "success",
                                 "verified_backup": True, "discovered_events": 0,
                                 "provider_blockers": ["SECRET_API_KEY", "MISSING_CALIBRATION"]}
                         for sport in SPORTS},
              "sports": {"NBA": {"model_cycle": {"predictions": 0,
                         "blockers": ["INSUFFICIENT_EVIDENCE"],
                         "raw_provider_url": "https://private.example/secret-api-key"}}}}
    audit = sanitize_cycle(report)
    raw = json.dumps(audit)
    assert set(audit["sports"]) == set(SPORTS)
    assert audit["requested_slate_success"] is False
    assert "private.example" not in raw
    assert "SECRET_API_KEY" not in raw
    assert audit["sports"]["NBA"]["model_cycle"]["predictions"] == 0
    assert audit["sports"]["NBA"]["model_blockers"] == ["INSUFFICIENT_EVIDENCE"]


def test_zero_event_slate_audit_keeps_all_sports():
    report = {"requested_sports": SPORTS, "requested_slate_success": True,
              "health": {sport: {"restore": "success", "capture": "success",
                                 "grade": "success", "backup": "success",
                                 "verified_backup": True} for sport in SPORTS}}
    audit = sanitize_cycle(report)
    assert audit["requested_slate_success"] is True
    assert all(row["captured_events"] == 0 and row["verified_backup"]
               for row in audit["sports"].values())


def test_missing_credentials_fail_explicitly_before_provider_or_remote_calls(tmp_path, monkeypatch):
    from scripts import run_research_scheduler as script
    for name in ("PARLAYPICKER_DRIVE_FOLDER_ID", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
                 "ODDS_API_KEY", "CFBD_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RESEARCH_SPORTS", ",".join(SPORTS))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("RESEARCH_AUDIT_PATH", str(tmp_path / "audit.json"))
    monkeypatch.setattr(script, "is_open", lambda: True)
    monkeypatch.setattr(script, "DriveStore", lambda _: (_ for _ in ()).throw(
        AssertionError("remote connection attempted")))
    assert script.main() == 1
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["requested_slate_success"] is False
    assert set(audit["requested_sports"]) == set(SPORTS)
    assert set(audit["missing_environment_variables"]) == {
        "PARLAYPICKER_DRIVE_FOLDER_ID", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
        "ODDS_API_KEY", "CFBD_API_KEY"}


def test_audit_checkpoint_survives_an_interrupted_research_cycle(tmp_path, monkeypatch):
    from scripts import run_research_scheduler as script
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("RESEARCH_AUDIT_PATH", str(audit_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("RESEARCH_SPORTS", "MLB")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "test-only-placeholder")
    monkeypatch.delenv("PARLAYPICKER_NETLIFY_SITE_ID", raising=False)
    monkeypatch.setattr(script, "is_open", lambda: True)
    monkeypatch.setattr(script, "settings", lambda: ("folder", None))
    monkeypatch.setattr(script, "DriveStore", lambda _: object())

    def run_until_cancelled(*args):
        checkpoint = args[-1]
        checkpoint({"requested_sports": ["MLB"], "health": {"MLB": {"restore": "success"}},
                    "active_sport": "MLB", "active_stage": "CANONICAL_RECONCILIATION",
                    "execution_state": "IN_PROGRESS", "errors": ["https://secret.example/key"]})
        written = json.loads(audit_path.read_text())
        assert written["execution_state"] == "IN_PROGRESS"
        assert written["active_stage"] == "CANONICAL_RECONCILIATION"
        assert written["sports"]["MLB"]["restore"] == "success"
        assert "secret.example" not in audit_path.read_text()
        assert not audit_path.with_name("audit.json.tmp").exists()
        raise TimeoutError("simulated cancellation")

    monkeypatch.setattr(script, "run", run_until_cancelled)
    assert script.main() == 1
    assert json.loads(audit_path.read_text())["execution_state"] == "FAILED"


def test_workflow_has_room_for_first_restore_and_keeps_audit_on_failure():
    from pathlib import Path
    import yaml
    workflow = yaml.safe_load(Path(".github/workflows/research-scheduler.yml").read_text())
    research = workflow["jobs"]["research"]
    assert research["timeout-minutes"] >= 60
    artifact = next(step for step in research["steps"] if step.get("name") == "Retain sanitized cycle audit")
    assert "always()" in artifact["if"]
