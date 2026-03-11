import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_totals_under_pick_maps_probability_to_total_under(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "odds_american": [-110],
        }
    )

    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "line": [220.5],
            "pick": ["Under 220.5"],
            "winprobability": [0.62],
            "odds_american": [-110],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analysis_df, best_picks_df, _diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"],
        max_rows=20,
        use_ml=False,
        spreads_df=None,
        totals_df=totals_df,
    )

    over_row = analysis_df[analysis_df["market_type"] == "total_over"].iloc[0]
    under_row = analysis_df[analysis_df["market_type"] == "total_under"].iloc[0]

    assert round(float(over_row["theover_probability"]), 2) == 0.38
    assert round(float(under_row["theover_probability"]), 2) == 0.62

    # Re-calculate to match best_pick
    best = sp.build_best_picks_df(analysis_df)
    if not best.empty:
        assert best.iloc[0]["best_pick"].startswith("Under")
