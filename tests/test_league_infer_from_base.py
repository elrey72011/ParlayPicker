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
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

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

    analysis_df, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA", "NHL", "NCAAB"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert str(analysis_df.loc[0, "league"]).upper() == "NBA"
    assert "miami" in str(analysis_df.loc[0, "home_team"]).lower()
    assert "boston" in str(analysis_df.loc[0, "away_team"]).lower()


def test_infer_missing_league_for_known_ncaab_teams(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    totals_df = pd.DataFrame(
        {
            "League": [""],
            "Home Team": ["Wichita St"],
            "Away Team": ["Oklahoma St"],
            "Pick": ["Under 140.5"],
            "Line": [140.5],
            "Win Probability": [0.55],
            "Odds": [-110],
        }
    )

    analysis_df, _, _ = sp.run_analysis_pipeline(
        sports=["NCAAB"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert str(analysis_df.loc[0, "league"]).upper() == "NCAAB"
