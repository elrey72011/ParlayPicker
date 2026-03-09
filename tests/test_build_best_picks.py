import pandas as pd
import pytest
from core.streamlit_pipeline import build_best_picks_df

def test_build_best_picks_df_competes_spread_vs_total():
    # Construct a sample analysis_df where we have a Spread and a Total for the same game
    # but the Home/Away order is flipped.

    data = [
        {
            "league": "NCAAB",
            "home_team": "DUKE",
            "away_team": "NORTH CAROLINA",
            "game_date": pd.Timestamp("2024-03-01 00:00:00+00:00"),
            "market_type": "spread_home",
            "has_signal_probability": True,
            "expected_value": 0.05,
            "edge": 0.05,
            "calibrated_probability": 0.55,
            "market_probability": 0.50
        },
        {
            # Flipped orientation for Total
            "league": "NCAAB",
            "home_team": "NORTH CAROLINA",
            "away_team": "DUKE",
            "game_date": pd.Timestamp("2024-03-01 02:00:00+00:00"), # Slightly different time
            "market_type": "total_over",
            "has_signal_probability": True,
            "expected_value": 0.10, # Higher EV, should win
            "edge": 0.10,
            "calibrated_probability": 0.60,
            "market_probability": 0.50
        }
    ]

    df = pd.DataFrame(data)

    best = build_best_picks_df(df)

    # We expect only 1 row because the orientation-insensitive key should group them
    assert len(best) == 1

    # The winning row should be the Total (EV=0.10)
    assert best.iloc[0]["expected_value"] == 0.10
    assert best.iloc[0]["market_type"] == "total_over"
