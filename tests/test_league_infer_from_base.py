import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_infer_missing_league_from_base_for_unique_match(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Miami Heat"],
            "away_team": ["Boston Celtics"],
            "date": ["2026-03-08"],
            "odds_american": [-110],
            "ml_probability": [0.55],
        }
    )
    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)

    totals_df = pd.DataFrame(
        {
            "League": ["None"],
            "Match Up": ["Boston Celtics AT Miami Heat"],
            "Pick": ["Under 210.5"],
            "Line": [210.5],
            "Win Probability": [0.58],
            "Odds": [-110],
        }
    )

    _, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA", "NHL", "NCAAB"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not best_picks_df.empty
    assert best_picks_df.loc[0, "league"] == "NBA"
    assert best_picks_df.loc[0, "home_team"] == "Miami Heat"
    assert best_picks_df.loc[0, "away_team"] == "Boston Celtics"
