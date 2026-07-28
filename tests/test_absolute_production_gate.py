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

