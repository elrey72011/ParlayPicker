import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_game_time_est_is_converted_from_utc_game_date(monkeypatch):
    base_df = pd.DataFrame()
    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "market_type": ["total_over"],
            "total_line": [220.5],
            "theover_probability": [0.57],
            "odds_american": [-110],
            "game_date": ["2026-03-07T00:00:00Z"],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    analysis_df, best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=5, use_ml=False, spreads_df=None, totals_df=None
    )

    # With the updated logic, exact midnight UTC timestamps are treated as date-only
    # placeholders, not valid times. They should yield the date formatted as %Y-%m-%d
    assert analysis_df.loc[0, "game_time_est"] == "2026-03-07"
    assert best_picks_df.loc[0, "game_time_est"] == "2026-03-07"
