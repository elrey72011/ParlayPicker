import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# The streamlit_app uses st.selectbox and st.checkbox to filter.
# It's hard to test the Streamlit app directly, but we can verify the export logic handles market families.
# We will create a test that verifies Sweet Spot filtering correctly filters a mocked dataframe.

def test_sweet_spot_market_segmentation():
    # We'll just write a functional test that simulates the filtering block in streamlit_app.py

    best_picks_df = pd.DataFrame({
        "Pick_Status": ["Actionable", "Actionable", "Actionable", "Actionable", "No Play"],
        "odds_source": ["odds_api", "odds_api", "odds_api", "odds_api", "odds_api"],
        "calibrated_probability": [0.60, 0.60, 0.60, 0.60, 0.60],
        "edge": [0.10, 0.10, 0.10, 0.10, 0.10],
        "expected_value": [0.15, 0.15, 0.15, 0.15, 0.15],
        "consensus_agreement": ["Agrees", "Agrees", "Agrees", "Agrees", "Agrees"],
        "market_type": ["total_over", "total_over", "total_under", "spread_home", "total_over"],
        "home_team": ["A", "B", "C", "D", "E"],
        "best_pick": ["Over", "Over", "Under", "Spread", "Over"],
    })

    # 1. Actionable filter
    df = best_picks_df[best_picks_df["Pick_Status"] == "Actionable"].copy()
    assert len(df) == 4

    # Simulate "Overs only"
    m_type_str = df["market_type"].astype(str).str.lower()
    overs_df = df[m_type_str.str.contains("total_over")]
    assert len(overs_df) == 2
    assert all("over" in m for m in overs_df["market_type"])

    # Simulate "Unders only"
    unders_df = df[m_type_str.str.contains("total_under")]
    assert len(unders_df) == 1
    assert all("under" in m for m in unders_df["market_type"])

    # Simulate "Sides only"
    sides_df = df[~m_type_str.str.contains("total")]
    assert len(sides_df) == 1
    assert all("spread" in m for m in sides_df["market_type"])

def test_sweet_spot_export_columns():
    best_picks_df = pd.DataFrame({
        "Pick_Status": ["Actionable"],
        "Status_Reason": ["Passed"],
        "Triple_Filter_Rank": [1],
        "Pick_Quality": ["A"],
        "parlay_rank": [1],
        "league": ["NFL"],
        "home_team": ["A"],
        "away_team": ["B"],
        "game_date": ["2023-10-10"],
        "game_time_est": ["1:00 PM ET"],
        "market_type": ["total_over"],
        "candidate_source": ["src"],
        "orientation_source": ["src"],
        "upload_match_reason": ["match"],
        "best_pick": ["Over 50"],
        "calibrated_probability": [0.60],
        "expected_value": [0.15],
        "edge": [0.10],
        "Conviction_Score": [0.8],
        "consensus_agreement": ["Agrees"],
        "odds_american": [-110],
        "odds_source": ["odds_api"],
        "market_probability": [0.5],
        "kalshi_probability": [0.6],
        "ml_probability": [0.6],
        "gemini_explanation": ["Good"],
        "gemini_risk_notes": ["None"]
    })

    # Simulate export rename
    csv_rename_map = {
        "home_team": "Home",
        "away_team": "Away",
        "game_date": "Local Date",
        "game_time_est": "Commence (Local)",
        "calibrated_probability": "WinProbability"
    }
    export_prep_ss = best_picks_df.rename(columns=csv_rename_map)

    target_export_cols = [
        "Pick_Status", "Status_Reason", "Triple_Filter_Rank", "Pick_Quality", "parlay_rank", "league", "Home", "Away", "Local Date",
        "Commence (Local)", "market_type", "candidate_source", "orientation_source", "upload_match_reason", "best_pick", "WinProbability", "expected_value",
        "edge", "Conviction_Score", "consensus_agreement", "odds_american", "odds_source", "market_probability",
        "kalshi_probability", "ml_probability", "gemini_explanation", "gemini_risk_notes"
    ]

    final_export_cols = [c for c in target_export_cols if c in export_prep_ss.columns]

    # Assert all target columns are present
    assert set(final_export_cols) == set(target_export_cols)
    assert len(final_export_cols) == len(target_export_cols)
