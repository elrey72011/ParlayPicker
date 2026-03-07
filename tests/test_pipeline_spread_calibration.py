import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_spread_calibrated_probability_uses_ml(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "odds_american": [-120],
            "ml_probability": [0.62],
        }
    )
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA", "NBA"],
            "home_team": ["Boston Celtics", "Boston Celtics"],
            "away_team": ["Los Angeles Lakers", "Los Angeles Lakers"],
            "market_type": ["spread_home", "total_over"],
            "spread_line": [-4.5, pd.NA],
            "total_line": [pd.NA, 220.5],
            "theover_probability": [0.40, 0.57],
            "odds_american": [-110, -110],
            "ml_probability": [0.61, 0.51],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    analysis_df, _best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=20, use_ml=False, spreads_df=None, totals_df=None
    )

    spread_row = analysis_df[analysis_df["market_type"] == "spread_home"].iloc[0]
    total_row = analysis_df[analysis_df["market_type"] == "total_over"].iloc[0]

    assert round(float(spread_row["calibrated_probability"]), 2) == 0.59
    assert round(float(total_row["calibrated_probability"]), 2) == 0.56
    assert pd.notna(spread_row["spread"])
    assert pd.notna(total_row["total"])
