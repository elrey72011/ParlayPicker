import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_punctuation_headers_are_normalized_to_identity_fields(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

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

    analysis_df, best_picks_df, _ = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert analysis_df.loc[0, "league"] == "NBA"
    assert "Miami" in analysis_df.loc[0, "home_team"]
    assert "Boston" in analysis_df.loc[0, "away_team"]
