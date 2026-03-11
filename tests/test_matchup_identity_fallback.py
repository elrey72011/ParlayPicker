import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_matchup_text_and_selected_sport_fill_identity(monkeypatch):
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    # Provide a mock live odds dataframe so rows aren't dropped
    # Let it be empty, we patched streamlit_pipeline so empty live odds doesn't drop rows
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda sports: pd.DataFrame(columns=["home_team", "away_team"]))

    totals_df = pd.DataFrame(
        {
            "Home Team": ["None"],
            "Away Team": ["None"],
            "Matchup": ["Boston Celtics @ Miami Heat"],
            "Pick": ["Under 210.5"],
            "Line": [210.5],
            "Win Probability": [0.58],
            "Odds": [-110],
        }
    )

    # Note: run_analysis_pipeline no longer returns a fully populated best_picks_df.
    # We should assert the parsing happened correctly on analysis_df instead.
    analysis_df, _, _ = sp.run_analysis_pipeline(
        sports=["NBA"],
        max_rows=10,
        use_ml=False,
        spreads_df=None,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert analysis_df.iloc[0]["league"] == "NBA"
    assert analysis_df.iloc[0]["home_team"] == "Miami"
    assert analysis_df.iloc[0]["away_team"] == "Boston"
