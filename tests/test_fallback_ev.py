import pandas as pd
import numpy as np
import pytest
from streamlit_app import _recompute_consensus_from_kalshi

def test_fallback_ev_computation():
    # Setup dataframe with stale decimal_odds but updated odds_american
    data = {
        "best_pick": ["TEAM A"],
        "odds_american": [-110.0],
        "odds_source": ["fallback_novig"],
        "decimal_odds": [1.5], # Stale odds (e.g. -200)
        "kalshi_probability": [0.65],
        "market_probability": [0.66],
        "ml_probability": [0.65],
        "theover_probability": [0.65],
        "sentiment_diff": [0.5],
        "league": ["NHL"],
        "market_type": ["spread_home"]
    }
    df = pd.DataFrame(data)

    # Process
    out = _recompute_consensus_from_kalshi(df)

    # Verify decimal_odds was recomputed from -110 (should be 1.909...)
    # 1.91 rounded or approx 1.90909
    assert np.isclose(out.loc[0, "decimal_odds"], 1.90909, atol=0.01)

    # Expected EV with 65% prob at -110 is positive: 0.65 * 0.909 - 0.35 = 0.59085 - 0.35 = 0.24085
    # The stale 1.5 odds would give negative EV: 0.65 * 0.5 - 0.35 = 0.325 - 0.35 = -0.025
    assert out.loc[0, "expected_value"] > 0
    assert np.isclose(out.loc[0, "expected_value"], 0.2409, atol=0.05)

if __name__ == "__main__":
    test_fallback_ev_computation()
    print("Test passed.")
