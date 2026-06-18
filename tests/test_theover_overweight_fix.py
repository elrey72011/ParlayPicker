"""TheOver over-weight fixes (18 Jun):
  A) symmetric fade — the OVER hit-rate source is now faded like the UNDER one;
  B) ML-contradiction guard raised to 0.45 — never stake an over our ML leans against.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _fade_theover, build_best_picks_df
from app_core.weights_config import MLB_THEOVER_FADE_SOURCES, MLB_THEOVER_FADE_SHRINK


def test_A_over_hit_rate_is_now_faded():
    # model_hit_rate (the OVER source) is in the fade set now, so 0.875 shrinks toward 0.5.
    out = _fade_theover([0.875], ["model_hit_rate"], MLB_THEOVER_FADE_SOURCES, MLB_THEOVER_FADE_SHRINK)
    expected = 0.5 + (0.875 - 0.5) * (1.0 - MLB_THEOVER_FADE_SHRINK)
    assert abs(float(out[0]) - expected) < 1e-9
    assert float(out[0]) < 0.875  # genuinely shrunk


def _over_row(ml_prob):
    return {
        "league": "MLB", "home_team": "Seattle", "away_team": "Baltimore",
        "game_date": "2026-06-18", "matchup_id": "2026-06-18|Seattle|Baltimore",
        "market_type": "total_over", "best_pick": "Over 7.5",
        "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.57,
        "ml_probability": ml_prob, "model_probability": ml_prob,
        "odds_american": -108, "total_line": 7.5, "spread_line": pd.NA,
        "market_probability": 0.49, "line_source": "live", "live_total_line": 7.5,
        "is_live_data": True, "used_stale_features": False, "odds_source": "odds_api",
        "kalshi_probability": 0.525,
    }


def test_B_over_below_ml_floor_is_no_play():
    # ML 0.40 < 0.45 -> the over our model leans against is blocked.
    best = build_best_picks_df(pd.DataFrame([_over_row(0.40)]))
    row = best.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "ml" in str(row["status_blocker_stage"]).lower()


def test_B_over_above_ml_floor_not_blocked_by_contradiction():
    # ML 0.50 >= 0.45 -> not blocked by the contradiction guard (may pass/other status).
    best = build_best_picks_df(pd.DataFrame([_over_row(0.50)]))
    assert str(best.iloc[0]["status_blocker_stage"]) != "ml_contradiction_guardrail"
