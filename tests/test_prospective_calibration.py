"""SYNTHETIC end-to-end research chain (roadmap C2): settle -> calibrate -> bind -> bound predictions.

Provider-shaped synthetic events reuse the research-model test builder. Nothing here
is live evidence or a model-quality claim.
"""
from datetime import timedelta
import math

import pytest

from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core.prospective_validation_plans import freeze_current_validation_plans
from tests.test_prospective_research_models import _at, _event


@pytest.fixture
def clock(monkeypatch):
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    return state


def test_score_settlement_rules():
    rule = pe.score_settlement_outcome
    assert rule("NBA", "SPREAD", "A", "A", "B", 101, 96, -2.5) == "WIN"     # 5 - 2.5
    assert rule("NBA", "SPREAD", "B", "A", "B", 101, 96, 2.5) == "LOSS"     # -5 + 2.5
    assert rule("NBA", "SPREAD", "A", "A", "B", 100, 97, -3) == "PUSH"
    assert rule("NBA", "TOTAL", "Over", "A", "B", 101, 96, 202.5) == "LOSS"
    assert rule("NBA", "TOTAL", "Under", "A", "B", 101, 96, 196.5) == "LOSS"
    assert rule("NBA", "TOTAL", "Under", "A", "B", 101, 96, 202.5) == "WIN"
    with pytest.raises(ValueError):
        rule("NBA", "SPREAD", "Nobody", "A", "B", 1, 0, -1.5)


def test_platt_fit_recovers_a_known_miscalibration():
    import random
    rng = random.Random(3)
    pairs = []
    for _ in range(4000):
        p = rng.uniform(0.2, 0.8)
        true = 1 / (1 + math.exp(-(0.3 + 0.5 * math.log(p / (1 - p)))))
        pairs.append((p, int(rng.random() < true)))
    a, b = pc.platt_fit(pairs)
    assert abs(a - 0.3) < 0.15 and abs(b - 0.5) < 0.15
    out = pc.calibrated({"win": 0.6, "push": 0.1, "loss": 0.3}, {"a": 0.0, "b": 1.0})
    assert math.isclose(out["win"] + out["push"] + out["loss"], 1) and out["push"] == 0.1


def test_end_to_end_settle_calibrate_bind_and_bound_predictions(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    fit = models.train_scope(path, "NBA", "SPREAD")
    assert fit["status"] == "FITTED_RESEARCH_ONLY"
    candidate = pc.candidate_model(path, "NBA", "SPREAD")
    assert candidate["model_id"] == fit["model_id"]

    # Calibration window: the pinned candidate predicts each game before it starts.
    for index in range(72, 72 + 48):
        event_id = _event(path, clock, "NBA", index)
        clock[0] = _at(index, 1)  # pregame
        assert pc.predict_candidate(path, "NBA", "SPREAD", now=clock[0])["predictions"] == 1
    clock[0] = _at(72 + 48 + 1)
    settled = pc.settle_scope(path, "NBA", "SPREAD")
    assert settled["settled"] >= 40 and settled["errors"] == []
    assert pc.settle_scope(path, "NBA", "SPREAD")["settled"] == 0  # idempotent

    too_few = pc.fit_calibration(path, "NBA", "SPREAD", candidate["model_id"], now=clock[0], min_decided=10_000)
    assert too_few["status"] == "INSUFFICIENT_EVIDENCE"
    calibration = pc.fit_calibration(path, "NBA", "SPREAD", candidate["model_id"], now=clock[0])
    assert calibration["status"] == "FITTED"
    stored = pe.load_record(path, "prospective_calibration", calibration["calibration_id"])
    assert stored["model_id"] == candidate["model_id"]

    # Freeze v1 (unbound) at its real-world date, then the bound v2 drafted for review.
    from datetime import datetime, timezone
    now = clock[0]
    clock[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
    freeze_current_validation_plans(path, source_commit="b" * 40)
    clock[0] = now
    spec = pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=(clock[0] + timedelta(days=1)).isoformat(),
                               rate=7)  # sized by game count (D14)
    assert spec["version"] == 2 and spec["model_id"] == candidate["model_id"]
    assert spec["calibration_id"] == calibration["calibration_id"]
    assert spec["minimum_independent_sample"] == 200  # v1 thresholds are kept
    assert (pc._time(spec["validation_end"]) - pc._time(spec["validation_start"])).days == pc.window_days(200, 7)
    pe.freeze_validation_plan(path, spec)
    assert pc.bound_plan(path, "NBA", "SPREAD")["validation_plan_id"] == spec["validation_plan_id"]

    # A game inside the validation window gets a bound, fully provenanced prediction.
    future = 72 + 48 + 3
    future_id = _event(path, clock, "NBA", future)
    clock[0] = _at(future, 1)
    made = pc.predict_bound(path, "NBA", "SPREAD", now=clock[0])
    assert made["predictions"] == 1 and made["blocker"] is None
    bound = [row for row in pe.read_records(path, "prospective_prediction", sport="NBA", market_family="SPREAD")
             if row["event_id"] == future_id and row["calibration_id"] == calibration["calibration_id"]]
    assert len(bound) == 1
    row = bound[0]
    assert all(row[field] is not None for field in (
        "model_version", "model_trained_through", "model_available_at", "feature_version", "feature_snapshot_id",
        "feature_frozen_at", "calibration_version", "calibration_available_at", "policy_version",
        "mean_probability", "conservative_probability", "push_probability", "loss_probability",
        "probability_semantics", "evidence_snapshot_id", "evidence_hash", "runtime_hash", "source_commit"))
    clock[0] = _at(future + 1)
    pc.settle_scope(path, "NBA", "SPREAD")
    report = pe.evaluate_validation_plan(path, spec["validation_plan_id"], as_of=clock[0])
    assert report["validation"]["unique_events"] == 1 and report["validation"]["missing_provenance_count"] == 0
    assert report["validation"]["scored_observations"] == 1  # settled, so it is scored rather than PENDING
    assert report["status"] != "VALIDATION_PASSED"  # 1 game is nowhere near the frozen minimum


def test_a_changed_research_runtime_blocks_visibly(tmp_path, clock, monkeypatch):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    models.train_scope(path, "NBA", "SPREAD")
    monkeypatch.setattr(models, "runtime_hash", lambda: "c" * 64)
    _event(path, clock, "NBA", 72)
    clock[0] = _at(72, 1)
    assert pc.predict_candidate(path, "NBA", "SPREAD", now=clock[0])["blocker"] == "CANDIDATE_MODEL_RUNTIME_CHANGED"


def test_nhl_is_not_settled_until_its_shootout_rule_is_confirmed(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    assert pc.settle_scope(path, "NHL", "PUCK_LINE")["errors"] == ["SCORE_SETTLEMENT_NOT_ENABLED"]
    assert pc.run_cycle(path, "NHL")["skipped"] == "SCORE_SETTLEMENT_NOT_ENABLED"


def test_windows_are_sized_by_game_count():
    assert pc.window_days(200, 30) == 11          # NCAAB-like slate: 200 / (30 * 0.8) * 1.25
    assert pc.window_days(200, 7) == 45           # NBA-like slate
    assert pc.window_days(200, 400) == 7          # never shorter than a week
    with pytest.raises(ValueError, match='cannot be sized'):
        pc.window_days(200, 0)
    from app_core.research_api_budget import ALL_CERTIFIED_CLOSE_LIMITS, Budget
    assert ALL_CERTIFIED_CLOSE_LIMITS['NCAAB'] == {'daily': 400, 'rolling_31_days': 4500}
    budget = Budget({}, lambda: None, clock=lambda: _at(0))
    budget.reserve_certified_close('NCAAB', 400)
    assert budget.usage('NCAAB', budget.close_ledger)['daily'] == 400
