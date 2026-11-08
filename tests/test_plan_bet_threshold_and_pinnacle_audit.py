"""Bet threshold chosen at the freeze; Pinnacle allowance in the cycle audit (owner 2026-11-07)."""
import importlib

import pytest

from app_core import market_skill_validation as ms
from app_core import prospective_calibration as pc
from app_core.research_cycle_audit import sanitize_cycle

BASE = {"probability_thresholds": {"max_brier": 0.24, "max_log_loss": 0.68, "max_calibration_error": 0.08,
                                   "min_coverage": 0.9}}


def test_bet_threshold_is_one_or_two_percent_and_defaults_to_two():
    assert ms.methodology()["min_edge"] == 0.02 == ms.template(BASE)[ms.KEY]["min_edge"]
    assert ms.template(BASE, 0.01)[ms.KEY]["min_edge"] == 0.01
    with pytest.raises(ValueError, match="min_edge"):
        ms.methodology(0.03)


def _spec(min_edge):
    return {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": ms.methodology(min_edge)}


def test_freeze_inputs_carry_a_non_default_threshold_so_the_freeze_redrafts_the_same_plan():
    assert "min_edge" not in pc.freeze_inputs(_spec(0.02))          # default plans: inputs unchanged
    inputs = pc.freeze_inputs(_spec(0.01))
    assert inputs["min_edge"] == 0.01 and inputs["methodology"] == "market_skill"
    assert pc.plan_hash(_spec(0.01)) != pc.plan_hash(_spec(0.02))    # the threshold is part of the reviewed hash


def test_draft_summary_states_the_bet_rule():
    bound_plan = importlib.import_module("scripts.bound_plan")
    plan = dict(_spec(0.01), validation_plan_id="p2", supersedes_plan_id="p1", minimum_independent_sample=200)
    text = bound_plan.markdown({"mode": "DRAFT_READ_ONLY", "markets": [], "draft": {
        "sport": "NCAAB", "market_family": "SPREAD", "status": "DRAFTED_NOT_FROZEN", "plan": plan,
        "plan_hash": "h", "freeze_inputs": pc.freeze_inputs(plan)}})
    assert "- bet rule: EV at least 1% at the entry price, at least 50 bets per window" in text
    assert '"min_edge": 0.01' in text


def test_cycle_audit_shows_pinnacle_allowance_and_whether_capture_included_pinnacle():
    report = {"requested_sports": ["NCAAB", "NBA"],
              "health": {s: {"restore": "success"} for s in ("NCAAB", "NBA")},
              "sports": {"NCAAB": {"pinnacle": "allowance_spent"}, "NBA": {"pinnacle": "on"}},
              "api_budget": {"pinnacle_usage": {"daily": 60, "rolling_31_days": 410},
                             "pinnacle_limits": {"daily": 60, "rolling_31_days": 1500},
                             "paused": ["PINNACLE:daily", "ODDS:daily", "https://leak.example"]}}
    audit = sanitize_cycle(report)
    assert audit["pinnacle_allowance"] == {"used_today": 60, "daily_limit": 60, "used_31_days": 410,
                                           "limit_31_days": 1500, "spent": ["DAILY"]}
    assert audit["sports"]["NCAAB"]["pinnacle_capture"] == "ALLOWANCE_SPENT"
    assert audit["sports"]["NBA"]["pinnacle_capture"] == "ON"
    assert "leak.example" not in str(audit)
    assert "pinnacle_allowance" not in sanitize_cycle({"requested_sports": []})
