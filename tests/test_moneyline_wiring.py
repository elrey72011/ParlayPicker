"""Moneyline parlay-only wiring (gated). Enforcement + build_best_picks_df integration."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_core.weights_config as wc
from core.streamlit_pipeline import _enforce_moneyline_parlay_only, build_best_picks_df


def _ml_rows():
    return pd.DataFrame([
        # Value dog: model 0.48 > +150 implied 0.40 -> +8% edge, in range.
        {"market_type": "moneyline_home", "effective_win_probability": 0.48, "WinProbability": 0.48,
         "odds_american": 150, "Pick_Status": "Actionable", "Status_Reason": "x",
         "Kelly_Bet_Size": 99.0, "production_eligible": True, "status_blocker_stage": ""},
        # Heavy favorite: -300 outside the [-250, 250] cap.
        {"market_type": "moneyline_away", "effective_win_probability": 0.62, "WinProbability": 0.62,
         "odds_american": -300, "Pick_Status": "High Variance/Speculative", "Status_Reason": "x",
         "Kelly_Bet_Size": 50.0, "production_eligible": True, "status_blocker_stage": ""},
        # No-edge dog: model 0.40 == +150 implied 0.40.
        {"market_type": "moneyline_home", "effective_win_probability": 0.40, "WinProbability": 0.40,
         "odds_american": 150, "Pick_Status": "High Variance/Speculative", "Status_Reason": "x",
         "Kelly_Bet_Size": 0.0, "production_eligible": False, "status_blocker_stage": ""},
        # A non-moneyline row must be untouched.
        {"market_type": "total_over", "effective_win_probability": 0.60, "WinProbability": 0.60,
         "odds_american": -110, "Pick_Status": "Actionable", "Status_Reason": "x",
         "Kelly_Bet_Size": 75.0, "production_eligible": True, "status_blocker_stage": ""},
    ])


def test_eligible_moneyline_is_parlay_only_never_single():
    out = _enforce_moneyline_parlay_only(_ml_rows())
    dog = out.iloc[0]
    assert bool(dog["parlay_only"])
    assert float(dog["Kelly_Bet_Size"]) == 0.0
    assert not bool(dog["production_eligible"])
    # Actionable moneyline is capped to High Variance so it stays a leg, not a single.
    assert dog["Pick_Status"] == "High Variance/Speculative"


def test_heavy_favorite_moneyline_no_play():
    out = _enforce_moneyline_parlay_only(_ml_rows())
    fav = out.iloc[1]
    assert fav["Pick_Status"] == "No Play"
    assert float(fav["Kelly_Bet_Size"]) == 0.0
    assert "moneyline parlay gate" in fav["Status_Reason"].lower()


def test_no_edge_moneyline_no_play():
    out = _enforce_moneyline_parlay_only(_ml_rows())
    assert out.iloc[2]["Pick_Status"] == "No Play"


def test_non_moneyline_rows_untouched():
    out = _enforce_moneyline_parlay_only(_ml_rows())
    tot = out.iloc[3]
    assert tot["Pick_Status"] == "Actionable"
    assert float(tot["Kelly_Bet_Size"]) == 75.0
    assert not bool(tot["parlay_only"])


def test_build_best_picks_forces_moneyline_parlay_only(monkeypatch):
    # With the flag on, a moneyline pick through the full builder is never single-staked.
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_PARLAY_LEGS", True)
    df = pd.DataFrame([{
        "league": "MLB", "home_team": "HomeA", "away_team": "AwayA",
        "game_date": "2026-04-24", "matchup_id": "2026-04-24|HomeA|AwayA",
        "market_type": "moneyline_home", "best_pick": "HomeA ML",
        "expected_value": 0.10, "edge": 0.08,
        "calibrated_probability": 0.48, "ml_probability": 0.48, "model_probability": 0.48,
        "odds_american": 150, "spread_line": pd.NA, "total_line": pd.NA,
        "line_source": "live", "live_spread_line": pd.NA, "live_total_line": pd.NA,
        "is_live_data": True, "used_stale_features": False, "odds_source": "odds_api",
        "kalshi_probability": None,
    }])
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert float(row["Kelly_Bet_Size"]) == 0.0
    assert bool(row.get("parlay_only", False))


def test_flag_off_leaves_moneyline_unenforced(monkeypatch):
    # Flag off: the enforcement is skipped (no parlay_only column added by it).
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False)
    out = build_best_picks_df(pd.DataFrame([{
        "league": "MLB", "home_team": "HomeB", "away_team": "AwayB",
        "game_date": "2026-04-24", "matchup_id": "2026-04-24|HomeB|AwayB",
        "market_type": "total_over", "best_pick": "Over 8.5",
        "expected_value": 0.05, "edge": 0.05,
        "calibrated_probability": 0.60, "ml_probability": 0.60, "model_probability": 0.60,
        "odds_american": -110, "spread_line": pd.NA, "total_line": 8.5,
        "line_source": "live", "live_spread_line": pd.NA, "live_total_line": 8.5,
        "is_live_data": True, "used_stale_features": False, "odds_source": "odds_api",
        "kalshi_probability": 0.55,
    }]))
    assert not out.empty
