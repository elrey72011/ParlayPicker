
import pandas as pd
import pytest
from streamlit_app import ml_allowed, calculate_best_pick_metrics

def test_ml_allowed():
    # Allow cases
    assert ml_allowed(100, -110) == True
    assert ml_allowed(-300, 300) == True
    assert ml_allowed(None, 150) == True
    assert ml_allowed("200", "-250") == True

    # Deny cases
    assert ml_allowed(301, -200) == False
    assert ml_allowed(-301, 200) == False
    assert ml_allowed(100, 301) == False

    # Edge cases
    assert ml_allowed(None, None) == False
    assert ml_allowed("invalid", 100) == False

def test_calculate_best_pick_metrics_suppression():
    # Setup row with extreme ML odds
    data = {
        "Home_ML": [350],
        "Away_ML": [-450],
        "Pick": ["Home"],
        "final_probability": [0.60],
        "Implied_Prob": [0.22],
        "Spread & Pick": ["Home -5"],
        "spread_prob": [0.55],
        "spread_edge": [0.05],
        "Total & Pick": ["Over 45"],
        "total_prob": [0.52],
        "total_edge": [0.02]
    }
    df = pd.DataFrame(data)

    # Run calculation
    result = calculate_best_pick_metrics(df)
    row = result.iloc[0]

    # Verify suppression
    assert row["ml_eligible"] == False
    assert row["moneyline_disabled"] == True
    assert "abs(ML)>300" in row["ml_suppressed_reason"]

    # Verify it picked Spread (since ML suppressed and S > T in this fake setup?)
    # Score logic: (0.55-0.5) + (0.05*2) = 0.05 + 0.1 = 0.15
    # Total: (0.52-0.5) + (0.02*2) = 0.02 + 0.04 = 0.06
    # Spread > Total.
    assert row["best_pick_type"] == "SPREAD"

def test_calculate_best_pick_metrics_allowed():
    # Setup row with normal ML odds
    data = {
        "Home_ML": [150],
        "Away_ML": [-170],
        "Pick": ["Home"],
        "final_probability": [0.65], # Strong ML
        "Implied_Prob": [0.40],
        "Spread & Pick": ["Home +3"],
        "spread_prob": [0.51], # Weak spread
        "spread_edge": [0.01],
        "Total & Pick": [None], # No total
        "total_prob": [None],
        "total_edge": [None]
    }
    df = pd.DataFrame(data)

    # Run calculation
    result = calculate_best_pick_metrics(df)
    row = result.iloc[0]

    # Verify allowed
    assert row["ml_eligible"] == True
    assert row["moneyline_disabled"] == False

    # Since Spread is valid (though weak), it might still prioritize SPREAD/TOTAL over ML
    # based on the priority logic "Prefer Spread/Total if 'Available'".
    # Let's check the logic in calculate_best_pick_metrics:
    # "Priority: Prefer Spread/Total if 'Available' (Valid)"
    # s_valid = (s_prob is not None and s_pick is not None) -> True here.
    # So it should pick SPREAD even if ML is stronger?
    # "Moneyline can still be computed optionally for context, but it must never be selected as best_pick_type when odds exceed threshold."
    # The requirement didn't explicitly say "Always pick ML if allowed and stronger".
    # It said "Moneyline should STOP driving best picks when odds are extreme."
    # The current implementation prioritizes S/T if valid.

    assert row["best_pick_type"] == "SPREAD"

if __name__ == "__main__":
    test_ml_allowed()
    test_calculate_best_pick_metrics_suppression()
    test_calculate_best_pick_metrics_allowed()
    print("All tests passed!")
