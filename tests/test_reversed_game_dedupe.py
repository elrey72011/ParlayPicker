import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_best_picks_dedupes_reversed_home_away_matchups(monkeypatch):
    base_df = pd.DataFrame()
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NCAAB", "NCAAB"],
            "home_team": ["Auburn", "Alabama"],
            "away_team": ["Alabama", "Auburn"],
            "market_type": ["total_over", "total_under"],
            "total_line": [176.5, 176.5],
            "theover_probability": [0.52, 0.61],
            "odds_american": [-110, -110],
            "game_date": ["2026-03-07T00:00:00Z", "2026-03-07T00:00:00Z"],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    _analysis_df, best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NCAAB"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    assert len(best_picks_df) == 1
    assert best_picks_df.iloc[0]["best_pick"].startswith("Under")
