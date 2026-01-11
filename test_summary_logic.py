import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from app_core
sys.path.append(os.getcwd())

from app_core.new_summary_logic import build_game_summary_v2, reorder_for_spread_total_focus_v2

def test_summary_logic():
    print("Testing build_game_summary_v2...")

    # Mock Data
    data = [
        # Game 1: Complete (ML, Spread, Total)
        {
            "league": "NBA", "Home": "Bulls", "Away": "Heat", "Commence (UTC)": "2023-10-01T00:00:00Z",
            "Commence (Local)": "2023-09-30T19:00:00", "Local Date": "2023-09-30",
            "Market": "Moneyline", "Pick": "Bulls", "final_probability": 0.60,
            "consensus_prob_adj": 0.60, "Line": None
        },
        {
            "league": "NBA", "Home": "Bulls", "Away": "Heat", "Commence (UTC)": "2023-10-01T00:00:00Z",
            "Commence (Local)": "2023-09-30T19:00:00", "Local Date": "2023-09-30",
            "Market": "Spread", "Pick": "Bulls", "Line": -5.0, "final_probability": 0.55,
            "spread_prob_adj": 0.55, "Spread & Pick": "Bulls -5.0"
        },
        {
            "league": "NBA", "Home": "Bulls", "Away": "Heat", "Commence (UTC)": "2023-10-01T00:00:00Z",
            "Commence (Local)": "2023-09-30T19:00:00", "Local Date": "2023-09-30",
            "Market": "Total", "Pick": "Over", "Line": 210.0, "final_probability": 0.52,
            "total_prob_adj": 0.52, "Total & Pick": "Over 210.0"
        },
        # Game 2: No Moneyline, Spread Better
        {
            "league": "NFL", "Home": "Packers", "Away": "Bears", "Commence (UTC)": "2023-10-02T00:00:00Z",
            "Commence (Local)": "2023-10-01T12:00:00", "Local Date": "2023-10-01",
            "Market": "Spread", "Pick": "Packers", "Line": -3.0, "final_probability": 0.58,
            "spread_prob_adj": 0.58, "Spread & Pick": "Packers -3.0"
        },
        {
            "league": "NFL", "Home": "Packers", "Away": "Bears", "Commence (UTC)": "2023-10-02T00:00:00Z",
            "Commence (Local)": "2023-10-01T12:00:00", "Local Date": "2023-10-01",
            "Market": "Total", "Pick": "Under", "Line": 45.0, "final_probability": 0.51,
            "total_prob_adj": 0.51, "Total & Pick": "Under 45.0"
        },
         # Game 3: No Moneyline, Total Better
        {
            "league": "NHL", "Home": "Leafs", "Away": "Habs", "Commence (UTC)": "2023-10-03T00:00:00Z",
            "Commence (Local)": "2023-10-02T19:00:00", "Local Date": "2023-10-02",
            "Market": "Spread", "Pick": "Leafs", "Line": -1.5, "final_probability": 0.53,
            "spread_prob_adj": 0.53, "Spread & Pick": "Leafs -1.5"
        },
        {
            "league": "NHL", "Home": "Leafs", "Away": "Habs", "Commence (UTC)": "2023-10-03T00:00:00Z",
            "Commence (Local)": "2023-10-02T19:00:00", "Local Date": "2023-10-02",
            "Market": "Total", "Pick": "Over", "Line": 6.5, "final_probability": 0.59,
            "total_prob_adj": 0.59, "Total & Pick": "Over 6.5"
        }
    ]

    master_df = pd.DataFrame(data)

    summary_df = build_game_summary_v2(master_df)

    # Assertions
    assert len(summary_df) == 3, f"Expected 3 games, got {len(summary_df)}"

    # Game 1: ML exists, so Overall = ML
    g1 = summary_df[summary_df["Home"] == "Bulls"].iloc[0]
    assert g1["ML"] == "Bulls"
    assert g1["ML Prob"] == 0.60
    assert g1["Overall Pick"] == "Bulls"
    assert g1["Spread"] == "Bulls -5.0"
    assert g1["Total"] == "Over 210.0"

    # Game 2: No ML, Spread (0.58) > Total (0.51) -> Overall = Spread
    g2 = summary_df[summary_df["Home"] == "Packers"].iloc[0]
    assert pd.isna(g2["ML"])
    assert g2["Spread"] == "Packers -3.0"
    assert g2["Spread Prob"] == 0.58
    assert g2["Overall Pick"] == "Packers -3.0"
    assert g2["Overall Prob"] == 0.58

    # Game 3: No ML, Total (0.59) > Spread (0.53) -> Overall = Total
    g3 = summary_df[summary_df["Home"] == "Leafs"].iloc[0]
    assert pd.isna(g3["ML"])
    assert g3["Total"] == "Over 6.5"
    assert g3["Total Prob"] == 0.59
    assert g3["Overall Pick"] == "Over 6.5"
    assert g3["Overall Prob"] == 0.59

    print("build_game_summary_v2 passed!")

    print("Testing reorder_for_spread_total_focus_v2...")

    # Add dummy columns
    summary_df["random_col"] = "data"

    reordered = reorder_for_spread_total_focus_v2(summary_df)

    cols = list(reordered.columns)
    expected_start = [
        "league", "Home", "Away", "Commence (UTC)", "Commence (Local)", "Local Date",
        "Overall Pick", "Overall Prob", "Spread", "Spread Prob", "Total", "Total Prob", "ML", "ML Prob"
    ]

    # Check fixed front
    assert cols[:len(expected_start)] == expected_start, f"Column order mismatch. Got {cols[:len(expected_start)]}"
    assert "random_col" in cols[len(expected_start):]

    print("reorder_for_spread_total_focus_v2 passed!")

if __name__ == "__main__":
    test_summary_logic()
