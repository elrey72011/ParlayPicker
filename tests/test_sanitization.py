import pytest
import pandas as pd
from core.streamlit_pipeline import _apply_analysis_calculations, build_best_picks_df

from core import streamlit_pipeline as sp

def test_sanitization_drops_extreme_odds(monkeypatch):
    base_df = pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "odds_american", "ml_probability"])
    bet_rows_df = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "A",
            "away_team": "B",
            "game_date": "2026-03-01T00:00:00Z",
            "market_type": "spread_home",
            "spread_line": -1.0,
            "odds_american": -99900.0,
            "theover_probability": 0.50,
        },
        {
            "league": "NBA",
            "home_team": "C",
            "away_team": "D",
            "game_date": "2026-03-01T00:00:00Z",
            "market_type": "spread_home",
            "spread_line": -1.0,
            "odds_american": -110.0,
            "theover_probability": 0.50,
        }
    ])

    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "build_theover_bet_rows", lambda *_args, **_kwargs: bet_rows_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda x: pd.DataFrame())

    analysis_df, best_picks_df, diagnostics = sp.run_analysis_pipeline(
        sports=["NBA"], max_rows=10, use_ml=False, spreads_df=None, totals_df=None
    )

    # Expect 1 row since the extreme one should be dropped
    assert len(analysis_df) == 1
    assert analysis_df.iloc[0]["odds_american"] == -110.0

def test_best_picks_drops_negative_ev():
    df = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "A",
            "away_team": "B",
            "game_date": pd.Timestamp("2024-03-01 00:00:00+00:00"),
            "market_type": "spread_home",
            "expected_value": -0.1232,
            "has_signal_probability": True,
            "edge": -0.10,
            "calibrated_probability": 0.30,
            "market_probability": 0.40
        },
        {
            "league": "NBA",
            "home_team": "C",
            "away_team": "D",
            "game_date": pd.Timestamp("2024-03-01 00:00:00+00:00"),
            "market_type": "spread_home",
            "expected_value": 0.05,
            "has_signal_probability": True,
            "edge": 0.05,
            "calibrated_probability": 0.55,
            "market_probability": 0.50
        }
    ])

    best = build_best_picks_df(df)

    # We expect 1 row because the -0.1232 row should be dropped by dynamic EV thresholding
    assert len(best) == 1
    assert best["expected_value"].iloc[0] == 0.05
