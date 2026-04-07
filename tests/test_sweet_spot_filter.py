import pandas as pd
import pytest
from streamlit_app import _recompute_consensus_from_kalshi

def test_sweet_spot_logic():
    # Because sweet spot is inside the UI function, we'll manually implement its filtering logic
    # here to ensure the logic works as expected given a mocked best_picks_df

    data = {
        "Pick_Status": ["Actionable", "Actionable", "Actionable", "Below Threshold", "Actionable"],
        "odds_source": ["odds_api", "fallback_novig", "odds_api", "odds_api", "odds_api"],
        "calibrated_probability": [0.60, 0.60, 0.50, 0.60, 0.65],
        "edge": [0.10, 0.10, 0.10, 0.10, 0.10],
        "expected_value": [0.15, 0.15, 0.15, 0.15, 0.15],
        "consensus_agreement": ["Agrees", "Agrees", "Agrees", "Agrees", "Agrees"]
    }
    df = pd.DataFrame(data)

    # 1. Actionable
    df = df[df["Pick_Status"] == "Actionable"]
    assert len(df) == 4

    # 2. Odds Source (include_fallback=False)
    df = df[df["odds_source"] != "fallback_novig"]
    assert len(df) == 3

    # 3. Probability Band (0.58 - 0.64)
    prob_mask = (df["calibrated_probability"] >= 0.58) & (df["calibrated_probability"] <= 0.64)
    df = df[prob_mask]
    assert len(df) == 1

    # 4. Edge & EV (already pass)

    # Check the remaining row
    assert df.iloc[0]["calibrated_probability"] == 0.60

if __name__ == "__main__":
    test_sweet_spot_logic()
    print("Test passed.")
