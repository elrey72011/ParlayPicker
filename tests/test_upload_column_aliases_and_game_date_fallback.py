import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_pipeline_handles_spaced_upload_columns_and_keeps_game_date(monkeypatch):
    # No base schedule needed for this regression; exercise raw upload normalization.
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())

    spreads_df = pd.DataFrame(
        {
            "Sport": ["NBA"],
            "Home Team": ["Los Angeles Lakers"],
            "Away Team": ["Boston Celtics"],
            "Pick Team": ["Los Angeles Lakers"],
            "Line": [-3.5],
            "Win Probability": [0.56],
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
            "Win Probability": [0.58],
            "Odds": [-110],
        }
    )

    analysis_df, best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"],
        spreads_df=spreads_df,
        totals_df=totals_df,
    )

    assert not analysis_df.empty
    assert not best_picks_df.empty
    assert best_picks_df["league"].astype(str).str.upper().eq("NBA").all()
    assert best_picks_df["home_team"].astype(str).str.len().gt(0).all()
    assert best_picks_df["away_team"].astype(str).str.len().gt(0).all()
    assert pd.to_datetime(best_picks_df["game_date"], errors="coerce", utc=True).notna().all()
    assert diagnostics["best_picks"] > 0
