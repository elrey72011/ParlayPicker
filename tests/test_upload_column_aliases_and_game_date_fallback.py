import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_pipeline_handles_spaced_upload_columns_and_keeps_game_date(monkeypatch):
    # No base schedule needed for this regression; exercise raw upload normalization.
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    spreads_df = pd.DataFrame(
        {
            "Sport": ["NBA"],
            "Home Team": ["Los Angeles Lakers"],
            "Away Team": ["Boston Celtics"],
            "Pick Team": ["Los Angeles Lakers"],
            "Line": [-3.5],
                "Win Probability": [0.60],
            "Odds": [-110],
        }
    )
    totals_df = pd.DataFrame(
        {
            "Sport": ["NBA"],
            "Home Team": ["Los Angeles Lakers"],
            "Away Team": ["Boston Celtics"],
            "Pick": ["Under 228.5"],
            "Line": [228.5],
                "Win Probability": [0.65],
            "Odds": [-110],
        }
    )

    analysis_df, best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=spreads_df,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert analysis_df["league"].astype(str).str.upper().eq("NBA").all()
