import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import build_best_picks_df


def test_best_picks_returns_one_pick_per_game_key_even_without_positive_ev():
    analysis_df = pd.DataFrame(
        {
            "game_key": ["G1", "G1", "G2", "G2"],
            "league": ["NBA", "NBA", "NBA", "NBA"],
            "home_team": ["Boston Celtics", "Boston Celtics", "Lakers", "Lakers"],
            "away_team": ["Miami Heat", "Miami Heat", "Warriors", "Warriors"],
            "game_date": [pd.Timestamp("2026-03-10", tz="UTC")] * 4,
            "market_type": ["spread_home", "total_under", "spread_home", "total_under"],
            "spread_line": [-3.5, pd.NA, -1.5, pd.NA],
            "total_line": [pd.NA, 222.5, pd.NA, 229.5],
            "expected_value": [0.01, -0.02, -0.03, -0.01],
            "edge": [0.01, -0.02, -0.03, -0.01],
            "model_probability": [0.53, 0.48, 0.49, 0.50],
            "theover_probability": [pd.NA, 0.48, pd.NA, 0.50],
            "ml_probability": [0.53, pd.NA, 0.49, pd.NA],
            "calibrated_probability": [0.53, 0.48, 0.49, 0.50],
            "odds_american": [-110, -110, -110, -110],
            "odds_source": ["uploaded", "uploaded", "uploaded", "uploaded"],
            "market_probability": [0.5238, 0.5238, 0.5238, 0.5238],
        }
    )

    best = build_best_picks_df(analysis_df)

    assert len(best) == 2
    matchups = set(zip(best["home_team"].tolist(), best["away_team"].tolist()))
    assert matchups == {("Boston Celtics", "Miami Heat"), ("Lakers", "Warriors")}
    # G2 should still be present even though both candidate EVs are negative.
    g2_row = best[(best["home_team"] == "Lakers") & (best["away_team"] == "Warriors")].iloc[0]
    assert g2_row["expected_value"] == -0.01
