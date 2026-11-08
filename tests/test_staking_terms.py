"""Staking terms frozen with a plan (app_core.staking_terms; owner delegation 2026-11-08)."""
import pytest

from app_core import market_skill_validation as ms
from app_core import prospective_calibration as pc
from app_core.staking_terms import PINNACLE_PROVISIONAL_V1, terms
from core.sport_market_activation import _policy_matches_frozen_plan
from core.sport_policy import SportPolicy


def test_terms_carry_an_activatable_policy_and_provisional_tier_rules():
    t = terms(PINNACLE_PROVISIONAL_V1, "NCAAB")
    policy = t["validated_policy"]
    assert (policy["sport"], policy["deployment_state"], policy["kelly_fraction"],
            policy["provisional_stake_cap"], policy["sport_exposure_cap"]) == (
        "NCAAB", "PROVISIONAL_VALIDATED", 0.10, 0.005, 0.03)
    assert t["maturity_rules"]["PROVISIONAL"] == {"max_uncertainty": 0.03, "min_prior_clv": 0.0}
    assert t["regime_brake"] == {"window_bets": 50, "min_mean_clv": 0.0}
    # The owner's activation policy is this exact policy plus the reviewed validation id:
    # the activation check (digest comparison) accepts it.
    state = {"validated_policy": policy, "validation_id": "val-1"}
    assert _policy_matches_frozen_plan(state, SportPolicy(**policy, validation_id="val-1"))
    assert not _policy_matches_frozen_plan(state, SportPolicy(**dict(policy, kelly_fraction=0.25),
                                                              validation_id="val-1"))
    with pytest.raises(ValueError, match="unknown staking terms"):
        terms("other", "NCAAB")


def test_freeze_inputs_carry_the_terms_so_the_freeze_redrafts_the_same_plan():
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": ms.methodology(),
            "deployment_criteria": dict(target_state="PROVISIONAL_VALIDATED", **terms(PINNACLE_PROVISIONAL_V1, "NCAAB"))}
    assert pc.freeze_inputs(spec)["staking_terms"] == PINNACLE_PROVISIONAL_V1
    assert "staking_terms" not in pc.freeze_inputs(dict(spec, deployment_criteria={"target_state": "X"}))


def test_holdout_bet_clv_lower_bound(monkeypatch):
    assert ms._lower_95([0.01]) is None
    assert ms._lower_95([0.02, 0.0, 0.01, 0.03]) == pytest.approx(0.015 - 1.96 * (0.0129099 / 2), abs=1e-6)
    home = {"quote_id": "h", "selection": "Home", "decimal_odds": 2.2}
    away = {"quote_id": "a", "selection": "Away", "decimal_odds": 1.7}
    monkeypatch.setattr(ms, "opposite", lambda db, quote: away)
    monkeypatch.setattr("app_core.prospective_evidence._comparable_close", lambda db, quote, as_of: {"decimal_odds": 2.1})
    out = ms.skill(None, [(0.55, 1, home), (0.56, 0, home), (0.57, 1, home)], as_of="now")
    assert out["bets"] == 3 and out["bet_clv_lower_95"] == pytest.approx(2.2 / 2.1 - 1)   # identical CLVs: no spread
