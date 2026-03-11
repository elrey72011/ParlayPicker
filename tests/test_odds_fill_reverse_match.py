import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_run_analysis_pipeline_fills_odds_from_reversed_base_matchup(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Los Angeles Lakers"],
            "away_team": ["Boston Celtics"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "odds_american": [-125],
            "ml_probability": [0.58],
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
            "ml_probability": [pd.NA],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analysis_df, _best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    row = analysis_df.iloc[0]
    assert float(row["odds_american"]) == -110
