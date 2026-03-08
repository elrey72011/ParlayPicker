import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import build_best_picks_df


def test_best_picks_prefers_rows_with_identity_when_available():
    analysis_df = pd.DataFrame(
        {
            "league": ["", "NBA"],
            "home_team": ["", "Boston Celtics"],
            "away_team": ["", "Miami Heat"],
            "game_date": [pd.NaT, pd.Timestamp("2026-03-08", tz="UTC")],
            "market_type": ["total_under", "spread_home"],
            "spread_line": [pd.NA, -3.5],
            "total_line": [210.5, pd.NA],
            "expected_value": [0.20, 0.05],
            "edge": [0.10, 0.03],
            "model_probability": [0.60, 0.55],
            "theover_probability": [0.60, pd.NA],
            "ml_probability": [pd.NA, 0.55],
            "calibrated_probability": [0.60, 0.55],
            "odds_american": [-110, -110],
            "odds_source": ["uploaded", "uploaded"],
            "market_probability": [0.52, 0.52],
        }
    )

    best = build_best_picks_df(analysis_df)

    assert len(best) == 1
    assert best.loc[0, "league"] == "NBA"
    assert best.loc[0, "home_team"] == "Boston Celtics"
    assert best.loc[0, "away_team"] == "Miami Heat"
