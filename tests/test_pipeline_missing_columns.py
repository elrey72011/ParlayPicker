import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp


def test_run_analysis_pipeline_handles_base_without_game_time_est(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "odds_american": [-120],
            "ml_probability": [0.57],
            # intentionally omits game_time_est
        }
    )

    bet_rows_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "market_type": ["spread_home"],
            "spread_line": [-2.5],
            "theover_probability": [0.55],
            "odds_american": [-110],
            "game_time_est": ["7:30 PM"],
        }
    )

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)

    analysis_df, best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"],
        max_rows=20,
        use_ml=False,
        spreads_df=None,
        totals_df=None,
    )

    assert not analysis_df.empty
    assert not best_picks_df.empty
    assert "game_time_est" in analysis_df.columns
    assert diagnostics["odds_schedule_loaded"] is True
    assert analysis_df["game_key"].str.len().gt(0).all()
