import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_punctuation_headers_are_normalized_to_identity_fields(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    totals_df = pd.DataFrame(
        {
            "LEAGUE!!": ["NBA"],
            "Home-Team-Name": ["Miami Heat"],
            "Visitor_Team": ["Boston Celtics"],
            "Win-Probability": [0.58],
            "Line": [210.5],
            "Pick": ["Under 210.5"],
            "Odds-American": [-110],
        }
    )

    _, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not best_picks_df.empty
    assert best_picks_df.loc[0, "league"] == "NBA"
    assert best_picks_df.loc[0, "home_team"] == "Miami Heat"
    assert best_picks_df.loc[0, "away_team"] == "Boston Celtics"
