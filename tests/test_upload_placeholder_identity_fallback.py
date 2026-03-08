import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_literal_none_home_away_falls_back_to_team1_team2(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    totals_df = pd.DataFrame(
        {
            "Sport": ["NBA"],
            "Home Team": ["None"],
            "Away Team": ["None"],
            "Team 1": ["Boston Celtics"],
            "Team 2": ["Miami Heat"],
            "Game Date": ["None"],
            "Pick": ["Under 210.5"],
            "Line": [210.5],
            "Win Probability": [0.58],
            "Odds": [-110],
        }
    )

    _, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not best_picks_df.empty
    assert best_picks_df.loc[0, "home_team"] == "Boston Celtics"
    assert best_picks_df.loc[0, "away_team"] == "Miami Heat"
    assert pd.notna(pd.to_datetime(best_picks_df.loc[0, "game_date"], errors="coerce", utc=True))
