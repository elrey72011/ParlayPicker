"""A score backlog alone does not fail the research run (owner 2026-11-06); real grading problems still do."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_core import research_scheduler as scheduler  # noqa: E402


def _run(monkeypatch, tmp_path, cycle):
    from test_research_scheduler import Cloud
    from app_core import prospective_remote, prospective_reconciliation, prospective_validation_plans
    from app_core.prospective_sport_adapters import get_adapter
    monkeypatch.setattr(prospective_remote, "sync", lambda *args: {"records_verified": 0})
    monkeypatch.setattr(prospective_validation_plans, "freeze_current_validation_plans", lambda *a, **k: [])
    monkeypatch.setattr(prospective_reconciliation, "reconcile_sport", lambda *args: {})
    adapter = get_adapter("NFL")
    monkeypatch.setattr(adapter, "restore", lambda *args: {})
    monkeypatch.setattr(adapter, "backup", lambda *args: {})
    monkeypatch.setattr(adapter, "run_cycle", lambda *args: cycle)
    return scheduler.run(["NFL"], tmp_path, Cloud(), "folder")


@pytest.mark.parametrize("blockers,success", [
    (["DUE_SCORE_BATCH_LIMIT"], True),                                               # backlog only
    (["DUE_SCORE_BATCH_LIMIT", "SCORE_SOURCE_IDENTITY_OR_TIMING_REJECTED"], False),  # rejected scores
    (["DUE_SCORE_BATCH_LIMIT", "SCORE_UNAVAILABLE_AFTER_PROVIDER_WINDOW"], False),   # scores never came
    ([], False),                                                                     # partial, reason unknown
])
def test_backlog_only_grading_counts_as_success(monkeypatch, tmp_path, blockers, success):
    result = _run(monkeypatch, tmp_path, {"errors": [], "grade_status": "partial", "blockers": blockers})
    assert result["requested_slate_success"] is success
    assert result["health"]["NFL"]["grade"] == "partial"              # still visible as partial
    assert result["grade_backlog_only"] == (["NFL"] if success else [])


def test_errors_still_fail_a_backlog_run(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, {"errors": ["capture_failed"], "grade_status": "partial",
                                          "blockers": ["DUE_SCORE_BATCH_LIMIT"]})
    assert result["requested_slate_success"] is False


@pytest.mark.parametrize("cycle,success", [
    # 2026-11-07 NCAAB: daily credit cap reached for capture, closes and grading.
    ({"errors": [], "budget_paused": True, "capture_status": "budget_paused",
      "close_capture_status": "budget_paused", "grade_status": "budget_paused"}, True),
    ({"errors": [], "budget_paused": True, "capture_status": "success",
      "close_capture_status": "budget_paused"}, True),
    # A pause never hides a real failure.
    ({"errors": ["capture_failed"], "budget_paused": True,
      "capture_status": "failed", "close_capture_status": "budget_paused"}, False),
    ({"errors": [], "budget_paused": True, "capture_status": "failed"}, False),
    ({"errors": [], "budget_paused": True, "capture_status": "budget_paused",
      "close_capture_status": "failed"}, False),
])
def test_budget_pause_is_not_a_failure(monkeypatch, tmp_path, cycle, success):
    result = _run(monkeypatch, tmp_path, cycle)
    assert result["requested_slate_success"] is success
    assert result["execution_state"] == ("COMPLETE" if success else "FAILED")
    assert result["budget_paused_sports"] == ["NFL"]                  # still visible


@pytest.mark.parametrize("blockers,success", [
    (["SCORE_WINDOW_EXPIRED_REPORTED"], True),                                        # already reported once
    (["SCORE_WINDOW_EXPIRED_REPORTED", "DUE_SCORE_BATCH_LIMIT"], True),
    (["SCORE_WINDOW_EXPIRED_REPORTED", "SCORE_SOURCE_IDENTITY_OR_TIMING_REJECTED"], False),  # a new rejection
    (["SCORE_WINDOW_EXPIRED_REPORTED", "SCORE_UNAVAILABLE_AFTER_PROVIDER_WINDOW"], False),   # a newly expired game
])
def test_a_game_already_reported_past_the_score_window_does_not_fail_every_cycle(monkeypatch, tmp_path,
                                                                                blockers, success):
    result = _run(monkeypatch, tmp_path, {"errors": [], "grade_status": "partial", "blockers": blockers})
    assert result["requested_slate_success"] is success
    assert result["health"]["NFL"]["grade"] == "partial"
