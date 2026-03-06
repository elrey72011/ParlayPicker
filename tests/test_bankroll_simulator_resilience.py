import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.bankroll_simulator import simulate_bankroll


def test_simulate_bankroll_missing_decimal_odds_does_not_crash():
    portfolio_df = pd.DataFrame(
        {
            "calibrated_probability": [0.55, 0.52],
            "recommended_bet": [10.0, 5.0],
        }
    )

    result = simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=1000.0, days=5, simulations=10)

    assert isinstance(result, dict)
    assert "expected_bankroll" in result
    assert "mean_final_bankroll" in result
    assert "bankroll_curves" in result


def test_simulate_bankroll_missing_calibrated_probability_does_not_crash():
    portfolio_df = pd.DataFrame(
        {
            "decimal_odds": [1.9, 2.1],
            "recommended_bet": [10.0, 5.0],
        }
    )

    result = simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=1000.0, days=5, simulations=10)

    assert isinstance(result, dict)
    assert result["inputs_used"] == 2


def test_simulate_bankroll_missing_recommended_bet_returns_default_payload():
    portfolio_df = pd.DataFrame(
        {
            "decimal_odds": [1.9, 2.1],
            "calibrated_probability": [0.55, 0.52],
        }
    )

    result = simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=1000.0, days=5, simulations=10)

    assert result["expected_bankroll"] == 1000.0
    assert result["ruin_probability"] == 0.0
    assert result["inputs_used"] == 0


def test_simulate_bankroll_empty_portfolio_returns_default_payload():
    result = simulate_bankroll(portfolio_df=pd.DataFrame(), starting_bankroll=250.0, days=5, simulations=10)

    assert result["expected_bankroll"] == 250.0
    assert result["mean_final_bankroll"] == 250.0
    assert result["median_final_bankroll"] == 250.0
    assert result["bankroll_curves"] == []
