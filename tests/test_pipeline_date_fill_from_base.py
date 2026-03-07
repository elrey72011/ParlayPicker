import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_run_analysis_pipeline_fills_game_date_from_base_date_column(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "date": ["2026-03-10T00:00:00Z"],
            "odds_american": [-120],
            "ml_probability": [0.57],
        }
    )
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "market_type": ["spread_home"],
            "spread_line": [-2.5],
            "theover_probability": [0.55],
            "odds_american": [-110],
            "game_date": [pd.NaT],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    analysis_df, _best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=20, use_ml=False, spreads_df=None, totals_df=None
    )

    assert pd.notna(analysis_df.loc[0, "game_date"])
    assert diagnostics["rows_with_game_date"] >= 1
