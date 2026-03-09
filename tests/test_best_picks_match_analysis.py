import pandas as pd
import pytest

from core.streamlit_pipeline import build_best_picks_df, BEST_PICK_COLUMNS

def test_best_picks_df_matches_analysis_df_values():
    """
    Verify that `best_picks_df` purely reflects the fully enriched values of `analysis_df`
    (including Kalshi probabilities, calibrated probability, and expected value),
    without recalculating or diverging.
    """
    # Simulate a fully enriched analysis_df row
    data = {
        "league": ["NBA", "NBA"],
        "home_team": ["Denver Nuggets", "Denver Nuggets"],
        "away_team": ["Oklahoma City Thunder", "Oklahoma City Thunder"],
        "game_date": [pd.Timestamp("2026-03-08", tz="UTC")] * 2,
        "market_type": ["total_over", "total_under"],
        "total_line": [225.5, 225.5],
        "expected_value": [0.05, -0.01],
        "edge": [0.06, -0.02],
        "calibrated_probability": [0.60, 0.40],
        "kalshi_probability": [0.62, 0.38],
        "model_probability": [0.55, 0.45],
        "theover_probability": [0.58, 0.42],
        "ml_probability": [0.55, 0.45],
        "odds_american": [-110, -110],
        "market_probability": [0.5238, 0.5238],
    }

    analysis_df = pd.DataFrame(data)

    best_picks_df = build_best_picks_df(analysis_df)

    # Verify we got exactly one best pick (the positive EV/edge one)
    assert len(best_picks_df) == 1

    best_row = best_picks_df.iloc[0]

    # Verify values strictly match the Over row from analysis_df
    assert best_row["league"] == "NBA"
    assert best_row["best_pick"] == "Over 225.5"
    assert best_row["expected_value"] == 0.05
    assert best_row["edge"] == 0.06
    assert best_row["calibrated_probability"] == 0.60
    assert best_row["kalshi_probability"] == 0.62

    # Verify correct filtering of columns
    for col in BEST_PICK_COLUMNS:
        assert col in best_picks_df.columns
