"""D19 option B: plans scored against the de-vigged entry price. SYNTHETIC store."""
import math

import pytest

from app_core import market_skill_validation as ms
from app_core import prospective_evidence as pe
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    _trained_market, at, clock, close, event, prediction, quote, result)


def _plan(path, *, skill=True):
    spec = dict(validation_plan_id="plan", sport="NBA", market_family="SPREAD", version=1, supersedes_plan_id=None,
                model_id="model", calibration_id="calibration", training_cutoff=at(8),
                validation_start=at(9), validation_end=at(15), holdout_start=at(16), holdout_end=at(25),
                minimum_independent_sample=1, minimum_effective_sample=1,
                probability_thresholds=dict(max_brier=0.24, max_log_loss=0.68, max_calibration_error=1, min_coverage=1),
                calibration_requirements=dict(required=True),
                price_evidence_requirements=dict(min_verified_entry_coverage=1),
                clv_policy=dict(required=False, min_comparable_close_coverage=0),
                value_roi_policy=dict(min_paper_roi=0),
                deployment_criteria=dict(target_state="PROVISIONAL_VALIDATED", maturity_rules={"synthetic_only": True}),
                independence_method="ONE_EARLIEST_PREDICTION_PER_EVENT_V1")
    return pe.freeze_validation_plan(path, ms.template(spec) if skill else spec)


def _game(path, clock, name, start, *, both_sides=True, mean=.65, loss=.35, outcome="WIN", close_stamp=None):  # noqa: F811
    clock(hours=start - 2)
    event(path, event_id=name, start=start, clock_hour=start - 2)
    quote(path, event_id=name, quote_id=f"{name}-home", stamp=start - 2)
    if both_sides:
        quote(path, event_id=name, quote_id=f"{name}-away", stamp=start - 2, selection="Away", line=2.5,
              raw_line=-2.5)
    prediction(path, event_id=name, quote_id=f"{name}-home", observation_id=f"{name}-pred", stamp=start - 2,
               mean=mean, loss=loss)
    if close_stamp is not None:
        clock(hours=close_stamp)
        close(path, event_id=name, close_id=f"{name}-close", stamp=close_stamp)
    clock(hours=start + 1)
    result(path, event_id=name, result_id=f"{name}-result", observed=start + 1, available=start + 1, outcome=outcome)


def test_template_keeps_v1_rules_and_sets_coin_flip_bounds():
    base = {"probability_thresholds": {"max_brier": 0.24, "max_log_loss": 0.68, "max_calibration_error": 0.08,
                                       "min_coverage": 0.9}, "other": 1}
    spec = ms.template(base)
    assert spec["probability_thresholds"] == {"max_brier": 0.25, "max_log_loss": round(math.log(2), 6),
                                              "max_calibration_error": 0.08, "min_coverage": 0.9}
    assert spec["other"] == 1 and spec[ms.KEY]["version"] == ms.VERSION
    assert base["probability_thresholds"]["max_brier"] == 0.24          # the v1 template is not mutated


def test_market_skill_plan_passes_when_the_model_beats_the_no_vig_price(tmp_path, clock):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan(path)
    stored = pe.load_record(path, "prospective_validation_plan", "plan")
    assert ms.KEY in stored["payload"]                                  # frozen into the plan artifact
    _game(path, clock, "validation", 11)
    _game(path, clock, "holdout", 20, close_stamp=19.5)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    skill = report["market_skill"]["holdout"]
    assert skill["with_market_reference"] == 1 and skill["market_brier"] == pytest.approx(0.25)
    assert skill["model_brier"] == pytest.approx(0.1225)
    # The model beats the market on both cohorts; only certified closes (none here) are missing.
    assert report["blockers"] == ["VALIDATION_MEAN_CLV_BELOW_MINIMUM", "HOLDOUT_MEAN_CLV_BELOW_MINIMUM"]


def test_gate_passes_with_skill_reference_coverage_and_non_negative_clv():
    method = {ms.KEY: ms.methodology()}
    good = {"market_reference_coverage": 0.95, "model_brier": 0.247, "market_brier": 0.249,
            "model_log_loss": 0.687, "market_log_loss": 0.691}
    assert ms.blockers("HOLDOUT", {"average_valid_clv": 0.004}, good, method) == []
    assert ms.blockers("HOLDOUT", {"average_valid_clv": -0.01}, good, method) == ["HOLDOUT_MEAN_CLV_BELOW_MINIMUM"]
    tied = dict(good, model_brier=0.249)                                 # equal to the market still passes
    assert ms.blockers("HOLDOUT", {"average_valid_clv": 0.0}, tied, method) == []


def test_worse_than_market_missing_reference_and_v1_plans_unchanged(tmp_path, clock):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan(path)
    _game(path, clock, "validation", 11, mean=.65, loss=.35, outcome="LOSS", close_stamp=None)
    _game(path, clock, "holdout", 20, both_sides=False, close_stamp=19.5)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    assert "VALIDATION_BRIER_WORSE_THAN_MARKET" in report["blockers"]
    assert "VALIDATION_LOG_LOSS_WORSE_THAN_MARKET" in report["blockers"]
    assert "HOLDOUT_MARKET_REFERENCE_COVERAGE_TOO_LOW" in report["blockers"]

    v1 = tmp_path / "v1.sqlite3"
    clock(hours=0)
    _trained_market(v1, clock)
    clock(hours=8)
    _plan(v1, skill=False)
    _game(v1, clock, "validation", 11)
    _game(v1, clock, "holdout", 20, close_stamp=19.5)
    clock(hours=26)
    old = pe.evaluate_validation_plan(v1, "plan")
    assert "market_skill" not in old                                    # v1 report shape unchanged
    assert not any("MARKET" in b or "CLV" in b for b in old["blockers"])


def test_freeze_inputs_carry_the_methodology_so_the_reviewed_hash_reproduces():
    from app_core import prospective_calibration as pc
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2026-11-08T00:00:00+00:00", "validation_end": "2026-11-28T00:00:00+00:00",
            "holdout_start": "2026-11-28T00:00:00+00:00", "holdout_end": "2026-12-18T00:00:00+00:00"}
    assert "methodology" not in pc.freeze_inputs(spec)                  # v1 drafts: inputs unchanged
    assert pc.freeze_inputs(dict(spec, **{ms.KEY: ms.methodology()}))["methodology"] == "market_skill"


def test_draft_freeze_and_evaluate_a_market_skill_bound_plan(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app_core import prospective_calibration as pc
    from app_core import prospective_research_models as models
    from app_core.prospective_validation_plans import freeze_current_validation_plans
    from tests.test_prospective_research_models import _at, _event
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, state, "NBA", index)
    state[0] = _at(71)
    model_id = models.train_scope(path, "NBA", "SPREAD")["model_id"]
    for index in range(72, 120):
        _event(path, state, "NBA", index)
        state[0] = _at(index, 1)
        pc.predict_candidate(path, "NBA", "SPREAD", now=state[0])
    state[0] = _at(121)
    pc.settle_scope(path, "NBA", "SPREAD")
    assert pc.fit_calibration(path, "NBA", "SPREAD", model_id, now=state[0])["status"] == "FITTED"
    now = state[0]
    state[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
    freeze_current_validation_plans(path, source_commit="b" * 40)
    state[0] = now
    start = (now + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="unknown plan methodology"):
        pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=start, rate=7, methodology="other")
    spec = pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=start, rate=7, methodology="market_skill")
    assert spec[ms.KEY]["version"] == ms.VERSION and spec["probability_thresholds"]["max_brier"] == 0.25
    inputs = pc.freeze_inputs(spec)
    frozen = pc.freeze_reviewed_plan(path, expected_hash=pc.plan_hash(spec), now=now, **inputs)
    report = pe.evaluate_validation_plan(path, frozen["validation_plan_id"], as_of=now)
    assert "market_skill" in report and report["status"] == "UNVALIDATED"
