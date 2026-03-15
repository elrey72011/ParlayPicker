import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import build_best_picks_df


def _base_rows() -> dict:
    return {
        "league": "NBA",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "game_date": pd.Timestamp("2026-03-08", tz="UTC"),
        "market_type": "spread_home",
        "spread_line": -3.5,
        "total_line": pd.NA,
        "edge": 0.03,
        "model_probability": 0.55,
        "theover_probability": 0.55,
        "ml_probability": pd.NA,
        "calibrated_probability": 0.55,
        "odds_american": -110,
        "market_probability": 0.52,
    }


def test_best_picks_prioritizes_novig_live_over_fallback_when_present():
    base = _base_rows()
    analysis_df = pd.DataFrame(
        [
            {
                **base,
                "market_type": "spread_home",
                "expected_value": 0.05,
                "odds_source": "novig_live",
            },
            {
                **base,
                "market_type": "spread_away",
                "expected_value": 0.10,
                "odds_source": "fallback_novig",
            },
        ]
    )

    best = build_best_picks_df(analysis_df)

    assert len(best) == 1
    assert best.loc[0, "odds_source"] == "novig_live"
    assert float(best.loc[0, "expected_value"]) == 0.05


def test_best_picks_uses_highest_ev_when_no_novig_live_rows_exist():
    base = _base_rows()
    analysis_df = pd.DataFrame(
        [
            {
                **base,
                "market_type": "spread_home",
                "expected_value": 0.05,
                "odds_source": "fallback_novig",
            },
            {
                **base,
                "market_type": "spread_away",
                "expected_value": 0.10,
                "odds_source": "uploaded",
            },
        ]
    )

    best = build_best_picks_df(analysis_df)

    assert len(best) == 1
    assert best.loc[0, "odds_source"] == "uploaded"
    assert float(best.loc[0, "expected_value"]) == 0.10


def test_best_picks_skips_game_when_only_fallback_rows_exist():
    base = _base_rows()
    analysis_df = pd.DataFrame(
        [
            {
                **base,
                "market_type": "spread_home",
                "expected_value": 0.05,
                "odds_source": "fallback_novig",
            },
            {
                **base,
                "market_type": "spread_away",
                "expected_value": 0.10,
                "odds_source": "fallback_novig",
            },
        ]
    )

    best = build_best_picks_df(analysis_df)

    assert best.empty
