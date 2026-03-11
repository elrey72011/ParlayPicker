import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_literal_none_home_away_falls_back_to_team1_team2(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

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

    analysis_df, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert "Boston" in analysis_df.loc[0, "home_team"]
    assert "Miami" in analysis_df.loc[0, "away_team"]
