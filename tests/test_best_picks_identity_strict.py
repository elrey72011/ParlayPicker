import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import build_best_picks_df
from core import streamlit_pipeline as sp


def test_best_picks_drops_rows_without_identity():
    analysis_df = pd.DataFrame(
        {
            "league": ["None"],
            "home_team": ["None"],
            "away_team": ["None"],
            "game_date": [pd.NaT],
            "market_type": ["total_under"],
            "total_line": [210.5],
            "expected_value": [0.2],
            "edge": [0.1],
            "calibrated_probability": [0.6],
            "odds_american": [-110],
            "market_probability": [0.52],
        }
    )

    best = build_best_picks_df(analysis_df)
    assert best.empty


def test_matchup_parser_handles_uppercase_at(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    totals_df = pd.DataFrame(
        {
            "League": ["NBA"],
            "Home Team": ["None"],
            "Away Team": ["None"],
            "Matchup": ["Boston Celtics AT Miami Heat"],
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
