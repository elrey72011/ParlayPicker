import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_fallback_odds_gets_stronger_market_shrink(monkeypatch):
    base_df = pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "odds_american", "ml_probability"])
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA", "NBA"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "market_type": ["total_over", "total_over"],
            "total_line": [220.5, 221.5],
            "theover_probability": [0.70, 0.70],
            "odds_american": [-110, -130],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analysis_df, _best, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=20, use_ml=False, spreads_df=None, totals_df=None
    )
