import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.odds_api import iso8601_to_est
from app_core import kalshi_integrator as ki
from app_core.prediction_engine import _to_et_game_date_string as _to_et_game_date


def test_iso8601_to_est_handles_utc_boundary_day_shift():
    est = iso8601_to_est("2026-03-18T03:10:00Z")
    assert est.strftime("%Y-%m-%d") == "2026-03-17"


def test_prediction_et_game_date_preserves_date_only_keys():
    series = pd.Series(["2026-03-17", "2026-03-18T03:10:00Z"])
    out = _to_et_game_date(series)
    assert out.iloc[0] == "2026-03-17"
    assert out.iloc[1] == "2026-03-17"


def test_kalshi_enrich_na_safe_output_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["spread_home"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": [pd.NA],
            "spread_line": [-4.5],
        }
    )

    out = ki.enrich_with_kalshi_markets(df)

    assert "is_matched" in out.columns
    assert bool(out.loc[0, "is_matched"]) is False
    assert float(out.loc[0, "kalshi_probability"]) == 0.0

def test_run_analysis_pipeline_handles_missing_live_game_date(monkeypatch):
    import core.streamlit_pipeline as sp

    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(
        sp,
        "build_theover_bet_rows",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "league": ["NBA"],
                "home_team": ["Boston Celtics"],
                "away_team": ["Los Angeles Lakers"],
                "market_type": ["spread_home"],
                "odds_american": [-110],
                "theover_probability": [0.55],
            }
        ),
    )
    monkeypatch.setattr(sp, "is_stale_schedule", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        sp,
        "fetch_live_odds_dataframe",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "league": ["NBA"],
                "home_team": ["Boston Celtics"],
                "away_team": ["Los Angeles Lakers"],
            }
        ),
    )

    analysis_df, bet_rows_df, diagnostics = sp.run_analysis_pipeline(use_ml=False)

    assert isinstance(analysis_df, pd.DataFrame)
    assert isinstance(bet_rows_df, pd.DataFrame)
    assert isinstance(diagnostics, dict)
