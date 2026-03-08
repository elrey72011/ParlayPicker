import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_auto_detect_matchup_from_unknown_column_name(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    totals_df = pd.DataFrame(
        {
            "League": ["NBA"],
            "Home Team": ["None"],
            "Away Team": ["None"],
            "Game Name": ["Boston Celtics vs Miami Heat"],
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
    assert best_picks_df.loc[0, "home_team"] == "Miami Heat"
    assert best_picks_df.loc[0, "away_team"] == "Boston Celtics"
    assert best_picks_df.loc[0, "league"] == "NBA"
