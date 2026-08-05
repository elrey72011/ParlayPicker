from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.production_gate import evaluate_absolute_production_gate
from core.streamlit_pipeline import optimize_portfolio_allocation


def test_gate_requires_two_point_calibrated_edge_and_positive_model_ev():
    result = evaluate_absolute_production_gate(
        pd.Series([0.55, 0.54, 0.60]),
        pd.Series([0.52, 0.52, 0.55]),
        pd.Series([0.03, 0.03, -0.01]),
    )

    assert result["production_gate_pass"].tolist() == [True, True, False]
    assert result.loc[2, "production_gate_reason"] == "model EV is not positive"


def test_gate_fails_closed_when_probability_or_price_is_missing():
    result = evaluate_absolute_production_gate(
        pd.Series([pd.NA, 0.60]),
        pd.Series([0.52, pd.NA]),
        pd.Series([0.05, 0.05]),
    )

    assert not result["production_gate_pass"].any()
    assert result.loc[0, "production_gate_reason"] == "missing calibrated probability"
    assert result.loc[1, "production_gate_reason"] == "missing sportsbook break-even price"


def _portfolio_row(*, empirical_probability: float, expected_value: float) -> dict:
    return {
        "league": "MLB",
        "home_team": "Home",
        "away_team": "Away",
        "best_pick": "Home +1.5",
        "Pick_Status": "Actionable",
        "market_type": "spread_home",
        "market_line_source": "live",
        "market_line_used": 1.5,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "line_provenance_warning": "",
        "model_status": "ML Model",
        "stats_source": "live",
        "fallback_summary_by_league": "",
        "run_health_warning": "",
        "degraded_feature_subset_flag": False,
        "odds_american": -110,
        "decimal_odds": 1.0 + (100.0 / 110.0),
        "empirical_win_probability": empirical_probability,
        "effective_win_probability": empirical_probability,
        "effective_expected_value": expected_value,
        "expected_value": expected_value,
    }


def test_portfolio_allocator_refuses_thin_actionable_price_edge():
    # -110 breaks even at 52.38%; 53.5% is positive but below the 2-point safety margin.
    weak = pd.DataFrame([_portfolio_row(empirical_probability=0.535, expected_value=0.03)])
    result = optimize_portfolio_allocation(weak, bankroll=1000.0)

    assert not bool(result.iloc[0]["production_eligible"])
    assert not bool(result.iloc[0]["absolute_production_gate_pass"])
    assert float(result.iloc[0]["recommended_bet"]) == 0.0


def test_portfolio_allocator_funds_actionable_pick_that_clears_gate():
    strong = pd.DataFrame([_portfolio_row(empirical_probability=0.58, expected_value=0.05)])
    result = optimize_portfolio_allocation(strong, bankroll=1000.0)

    assert bool(result.iloc[0]["production_eligible"])
    assert bool(result.iloc[0]["absolute_production_gate_pass"])
    assert float(result.iloc[0]["recommended_bet"]) > 0.0


def test_controlled_value_allocator_uses_empirical_exact_price_probability():
    """A recovered plus-money value row must survive its second sizing pass."""
    row = _portfolio_row(empirical_probability=0.4485899256, expected_value=0.0430743718)
    row.update({
        "home_team": "New York Yankees",
        "away_team": "Saint Louis",
        "best_pick": "New York Yankees -1.5",
        "market_line_used": -1.5,
        "odds_american": 141,
        "decimal_odds": 2.41,
        "production_win_probability": 0.4328109426,
        "production_expected_value": 0.0430743718,
        "consensus_agreement": "Disagrees",
        "controlled_card_recovery": True,
    })

    result = optimize_portfolio_allocation(pd.DataFrame([row]), bankroll=1000.0)
    recovered = result.iloc[0]

    assert recovered["kelly_probability_source"] == (
        "controlled_value_empirical_price_probability"
    )
    assert float(recovered["kelly_probability_used"]) == 0.4485899256
    assert bool(recovered["production_eligible"])
    assert bool(recovered["absolute_production_gate_pass"])
    assert float(recovered["recommended_bet"]) > 0.0


def test_controlled_value_allocator_rechecks_stricter_disagrees_margin():
    row = _portfolio_row(empirical_probability=0.44, expected_value=0.04)
    row.update({
        "odds_american": 140,
        "decimal_odds": 2.4,
        "production_win_probability": 0.43,
        "production_expected_value": 0.04,
        "consensus_agreement": "Disagrees",
        "controlled_card_recovery": True,
    })

    result = optimize_portfolio_allocation(pd.DataFrame([row]), bankroll=1000.0)
    rejected = result.iloc[0]

    # +140 breaks even at 41.67%; 44% clears the normal 2-point gate but not
    # the controlled contrarian 3-point requirement.
    assert not bool(rejected["production_eligible"])
    assert not bool(rejected["absolute_production_gate_pass"])
    assert rejected["absolute_production_gate_reason"] == (
        "controlled value exact-price gate failed"
    )
    assert float(rejected["recommended_bet"]) == 0.0

