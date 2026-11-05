"""SYNTHETIC review-and-freeze path for bound plans (roadmap C2, D11). Not live evidence."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest

from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core.prospective_validation_plans import freeze_current_validation_plans
from tests.test_prospective_research_models import _at, _event

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    """A store with the v1 plans, a pinned NBA SPREAD candidate and its calibration."""
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, state, "NBA", index)
    state[0] = _at(71)
    models.train_scope(path, "NBA", "SPREAD")
    for index in range(72, 72 + 48):
        _event(path, state, "NBA", index)
        state[0] = _at(index, 1)
        pc.predict_candidate(path, "NBA", "SPREAD", now=state[0])
    state[0] = _at(72 + 48 + 1)
    pc.settle_scope(path, "NBA", "SPREAD")
    assert pc.fit_calibration(path, "NBA", "SPREAD", pc.candidate_model(path, "NBA", "SPREAD")["model_id"],
                              now=state[0])["status"] == "FITTED"
    now = state[0]
    state[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
    freeze_current_validation_plans(path, source_commit="b" * 40)
    state[0] = now
    return path, state


def test_status_reports_the_next_blocker_per_research_market(calibrated):
    path, state = calibrated
    status = pc.binding_status(path, "NBA", "SPREAD", now=state[0])
    assert status["candidate_model_id"] and status["calibration_id"]
    assert status["candidate_decided"] >= pc.MIN_CALIBRATION_DECIDED
    assert status["games_per_day_14d"] > 0 and status["proposed_window_days"] >= 7
    assert status["next_blocker"] == "READY_FOR_OWNER_REVIEW"
    assert pc.binding_status(path, "NBA", "TOTAL", now=state[0])["next_blocker"].startswith("CANDIDATE_NEEDS_")
    assert pc.binding_status(path, "NHL", "PUCK_LINE", now=state[0])["next_blocker"] == "SCORE_SETTLEMENT_NOT_ENABLED"
    rows = _script("market_readiness_report").research_bindings(path, now=state[0])
    assert {(r["sport"], r["market_family"]) for r in rows} >= {("NBA", "SPREAD"), ("NCAAB", "SPREAD")}


def test_draft_changes_nothing_and_freeze_needs_the_reviewed_hash(calibrated):
    path, state = calibrated
    script = _script("bound_plan")
    plans_before = len(pe.read_records(path, "prospective_validation_plan"))
    start = (state[0] + timedelta(days=2)).isoformat()
    result = script.draft(path, sport="NBA", market="SPREAD", validation_start=start, now=state[0])
    item = result["draft"]
    assert item["status"] == "DRAFTED_NOT_FROZEN"
    assert len(pe.read_records(path, "prospective_validation_plan")) == plans_before  # nothing frozen
    assert "plan hash" in script.markdown(result)

    inputs = item["freeze_inputs"]
    with pytest.raises(ValueError, match="PLAN_HASH_MISMATCH"):
        pc.freeze_reviewed_plan(path, expected_hash="0" * 64, now=state[0], **dict(inputs))
    with pytest.raises(ValueError, match="VALIDATION_ALREADY_STARTED"):
        pc.freeze_reviewed_plan(path, expected_hash=item["plan_hash"],
                                now=state[0] + timedelta(days=3), **dict(inputs))
    assert len(pe.read_records(path, "prospective_validation_plan")) == plans_before

    frozen = pc.freeze_reviewed_plan(path, expected_hash=item["plan_hash"], now=state[0], **dict(inputs))
    assert frozen["validation_plan_id"] == item["plan"]["validation_plan_id"]
    assert pc.bound_plan(path, "NBA", "SPREAD")["validation_plan_id"] == frozen["validation_plan_id"]
    assert pc.binding_status(path, "NBA", "SPREAD", now=state[0])["next_blocker"] == "VALIDATION_IN_PROGRESS"


def test_an_undraftable_market_is_reported_not_raised(calibrated):
    path, state = calibrated
    result = _script("bound_plan").draft(path, sport="NBA", market="TOTAL", now=state[0])
    assert result["draft"]["status"] == "NOT_DRAFTABLE" and result["draft"]["reason"] == "no candidate model"
