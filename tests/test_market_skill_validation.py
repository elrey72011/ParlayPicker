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
    assert skill["bets"] == 1 and skill["bet_roi"] == pytest.approx(0.909090909)   # home at -110, won
    # The model beats the market on both cohorts; what is left is sample size and certified closes.
    assert not any("WORSE_THAN_MARKET" in b or "REFERENCE" in b or "BET_ROI" in b for b in report["blockers"])
    for label in ("VALIDATION", "HOLDOUT"):
        assert {f"{label}_NEED_49_MORE_BETS", f"{label}_BET_CLOSE_COVERAGE_TOO_LOW",
                f"{label}_BET_MEAN_CLV_BELOW_MINIMUM"} <= set(report["blockers"])


def test_gate_passes_with_skill_reference_coverage_and_non_negative_clv():
    method = {ms.KEY: ms.methodology()}
    good = {"market_reference_coverage": 0.95, "model_brier": 0.247, "market_brier": 0.249,
            "model_log_loss": 0.687, "market_log_loss": 0.691, "bets": 80, "bet_roi": 0.03,
            "bet_close_coverage": 0.9, "bet_mean_clv": 0.004}
    assert ms.blockers("HOLDOUT", {}, good, method) == []
    assert ms.blockers("HOLDOUT", {}, dict(good, bet_mean_clv=-0.01), method) == ["HOLDOUT_BET_MEAN_CLV_BELOW_MINIMUM"]
    assert ms.blockers("HOLDOUT", {}, dict(good, bet_roi=-0.01), method) == ["HOLDOUT_BET_ROI_BELOW_MINIMUM"]
    assert ms.blockers("HOLDOUT", {}, dict(good, bets=30), method) == ["HOLDOUT_NEED_20_MORE_BETS"]
    tied = dict(good, model_brier=0.249)                                 # equal to the market still passes
    assert ms.blockers("HOLDOUT", {}, tied, method) == []


def test_bets_take_the_side_with_the_edge_and_skip_thin_edges(monkeypatch):
    home = {"quote_id": "h", "selection": "Home", "decimal_odds": 1.909090909}
    away = {"quote_id": "a", "selection": "Away", "decimal_odds": 1.909090909}
    monkeypatch.setattr(ms, "opposite", lambda db, quote: away)
    monkeypatch.setattr("app_core.prospective_evidence._comparable_close",
                        lambda db, quote, as_of: {"decimal_odds": 1.8} if quote is away else None)
    # Model says home 40%: the away side has EV 0.6 x 1.909 - 1 = +14.5%; away won (home lost).
    out = ms.skill(None, [(0.40, 0, home), (0.52, 1, home)], as_of="now")
    assert out["bets"] == 1 and out["bet_roi"] == pytest.approx(0.909090909)   # 52% home: EV -0.7%, no bet
    assert out["bet_mean_clv"] == pytest.approx(1.909090909 / 1.8 - 1) and out["bet_close_coverage"] == 1.0
    assert out["with_market_reference"] == 2 and out["market_brier"] == pytest.approx(0.25)


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
    # Every-game paper ROI (home side, lost) is not a gate for market-skill plans; the bets are.
    assert report["validation"]["paper_roi"] < 0 and "VALIDATION_PAPER_ROI_POLICY_NOT_MET" not in report["blockers"]
    assert "VALIDATION_BET_ROI_BELOW_MINIMUM" in report["blockers"]

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


def test_research_cycle_predictions_do_not_block_a_market_skill_plan(tmp_path, monkeypatch):
    """The research cycle writes an uncalibrated prediction for every game; v1 plans block on it."""
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

    def plan_with(methodology, path):
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
        pc.fit_calibration(path, "NBA", "SPREAD", model_id, now=state[0])
        now = state[0]
        state[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
        freeze_current_validation_plans(path, source_commit="b" * 40)
        state[0] = now
        spec = pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=(now + timedelta(days=1)).isoformat(),
                                   rate=7, methodology=methodology)
        pe.freeze_validation_plan(path, spec)
        _event(path, state, "NBA", 123)
        state[0] = _at(123, 1)
        assert models.predict_scope(path, "NBA", "SPREAD", as_of=state[0])["predictions"] == 1   # research cycle
        assert pc.predict_bound(path, "NBA", "SPREAD", now=state[0])["predictions"] == 1
        state[0] = _at(124)
        pc.settle_scope(path, "NBA", "SPREAD")
        return pe.evaluate_validation_plan(path, spec["validation_plan_id"], as_of=state[0])

    v1 = plan_with("v1", tmp_path / "v1.sqlite3")
    assert "VALIDATION_MODEL_CALIBRATION_SCOPE_MISMATCH" in v1["blockers"]           # unchanged for v1
    state[0] = _at(0)
    skill = plan_with("market_skill", tmp_path / "skill.sqlite3")
    assert skill["validation"]["wrong_model_or_calibration_count"] == 1                 # still reported
    assert skill["validation"]["unique_events"] == 1
    assert not any("SCOPE_MISMATCH" in b for b in skill["blockers"])
