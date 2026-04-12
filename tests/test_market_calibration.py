import pytest
import pandas as pd
import numpy as np

from app_core.weights_config import (
    NBA_STAR_ACTIVE_TOTAL_OVER_BOOST,
    TOTAL_OVER_MIN_EV,
    TOTAL_OVER_MIN_EDGE
)

from core.streamlit_pipeline import build_best_picks_df, run_analysis_pipeline

def test_nba_over_boost_applied():
    """Test that the NBA_STAR_ACTIVE_TOTAL_OVER_BOOST is correctly applied to total_over rows in run_analysis_pipeline."""
    merged = pd.DataFrame({
        "league": ["NBA", "NBA", "NHL"],
        "market_type": ["total_over", "total_under", "total_over"],
        "feature_star_active": [1.0, 1.0, 1.0],
    })

    model_probability = pd.Series([0.5, 0.5, 0.5], dtype=float)

    is_nba = merged["league"].str.upper() == "NBA"
    has_star_active = pd.Series([False] * len(merged), index=merged.index)
    if "feature_star_active" in merged.columns:
        has_star_active = pd.to_numeric(merged["feature_star_active"], errors="coerce").fillna(0) == 1.0

    star_impact = is_nba & has_star_active
    affected_over = star_impact & (merged["market_type"] == "total_over")
    affected_under = star_impact & (merged["market_type"] == "total_under")

    model_probability = model_probability.where(~affected_over, model_probability + NBA_STAR_ACTIVE_TOTAL_OVER_BOOST)
    model_probability = model_probability.where(~affected_under, model_probability - NBA_STAR_ACTIVE_TOTAL_OVER_BOOST)

    # Over is boosted
    assert model_probability[0] == 0.5 + NBA_STAR_ACTIVE_TOTAL_OVER_BOOST
    # Under is penalized
    assert model_probability[1] == 0.5 - NBA_STAR_ACTIVE_TOTAL_OVER_BOOST
    # Non-NBA over is unchanged
    assert model_probability[2] == 0.5


def test_total_over_strict_guardrails():
    """Test that total_over requires higher edge and EV thresholds to become Actionable."""
    analysis_df = pd.DataFrame({
        "league": ["NFL", "NFL", "NFL", "NFL"],
        "home_team": ["Team A", "Team B", "Team C", "Team D"],
        "away_team": ["Team X", "Team Y", "Team Z", "Team W"],
        "game_date": ["2023-10-10", "2023-10-10", "2023-10-10", "2023-10-10"],
        "market_type": ["total_over", "total_over", "spread_home", "total_under"],
        "expected_value": [0.02, TOTAL_OVER_MIN_EV + 0.01, 0.02, 0.02],
        "edge": [0.03, TOTAL_OVER_MIN_EDGE + 0.01, 0.03, 0.03],
        "calibrated_probability": [0.55, 0.55, 0.55, 0.55],
        "ml_probability": [0.55, 0.55, 0.55, 0.55],
        "theover_probability": [0.55, 0.55, 0.55, 0.55],
        "market_probability": [0.5, 0.5, 0.5, 0.5],
        "odds_american": [-110, -110, -110, -110],
        "total_line": [50.5, 50.5, np.nan, 50.5],
        "spread_line": [np.nan, np.nan, -3.5, np.nan],
        "matchup_id": ["A", "B", "C", "D"],
        "best_pick": ["Over 50.5", "Over 50.5", "Team C -3.5", "Under 50.5"]
    })

    best_picks_df = build_best_picks_df(analysis_df)

    # We find the rows based on best_pick and home_team because matchup_id isn't guaranteed to be exported in BEST_PICK_COLUMNS
    row_a = best_picks_df[(best_picks_df["home_team"] == "Team A") & (best_picks_df["best_pick"] == "Over 50.5")].iloc[0]
    row_b = best_picks_df[(best_picks_df["home_team"] == "Team B") & (best_picks_df["best_pick"] == "Over 50.5")].iloc[0]
    row_c = best_picks_df[(best_picks_df["home_team"] == "Team C") & (best_picks_df["best_pick"] == "Team C -3.5")].iloc[0]
    row_d = best_picks_df[(best_picks_df["home_team"] == "Team D") & (best_picks_df["best_pick"] == "Under 50.5")].iloc[0]

    # 1st row (A): total_over with 0.02 EV (fails strict 0.03 EV requirement) -> Below Threshold
    assert row_a["Pick_Status"] == "Below Threshold"

    # 2nd row (B): total_over with passing EV/Edge -> Below Threshold (fails 0.56 TOTAL_MIN_WIN_PROB)
    assert row_b["Pick_Status"] == "Below Threshold"
    assert "Win Probability for Totals" in row_b["Status_Reason"]

    # 3rd row (C): side with 0.02 EV/0.03 Edge (passes baseline 0.01/0.02) -> Actionable (0.55 passes 0.52 side floor)
    assert row_c["Pick_Status"] == "Actionable"

    # 4th row (D): total_under with 0.02 EV/0.03 Edge (passes baseline 0.01/0.02) -> Below Threshold (fails 0.56 TOTAL_MIN_WIN_PROB)
    assert row_d["Pick_Status"] == "Below Threshold"
    assert "Win Probability for Totals" in row_d["Status_Reason"]
