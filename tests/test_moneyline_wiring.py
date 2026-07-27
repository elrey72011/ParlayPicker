"""Moneyline parlay-only wiring (gated). Enforcement + build_best_picks_df integration."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_core.weights_config as wc
from core.streamlit_pipeline import _enforce_moneyline_parlay_only, _format_best_pick, build_best_picks_df


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



@pytest.mark.parametrize(
    ("market_type", "expected_pick"),
    [
        ("moneyline_home", "Home Team"),
        ("moneyline_away", "Away Team"),
        ("h2h_home", "Home Team"),
        ("h2h_away", "Away Team"),
    ],
)
def test_moneyline_formatter_uses_selected_team_without_a_line(market_type, expected_pick):
    row = pd.Series({
        "market_type": market_type,
        "home_team": "Home Team",
        "away_team": "Away Team",
        "spread_line": pd.NA,
        "total_line": pd.NA,
    })

    assert _format_best_pick(row) == expected_pick


@pytest.mark.parametrize(
    ("market_type", "expected_pick", "odds"),
    [
        ("moneyline_home", "HomeA", -147),
        ("moneyline_away", "AwayA", -167),
    ],
)
def test_best_picks_preserves_live_line_less_moneyline_identity(
    monkeypatch, market_type, expected_pick, odds
):
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_BEST_AVAILABLE", True)
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False)
    df = pd.DataFrame([{
        "league": "MLB", "home_team": "HomeA", "away_team": "AwayA",
        "game_date": "2026-07-27", "matchup_id": "2026-07-27|HomeA|AwayA",
        "market_type": market_type,
        "expected_value": 0.05, "edge": 0.03,
        "calibrated_probability": 0.60, "ml_probability": 0.60,
        "model_probability": 0.60, "odds_american": odds,
        "spread_line": pd.NA, "total_line": pd.NA,
        "line_source": "live_odds", "live_spread_line": pd.NA,
        "live_total_line": pd.NA, "is_live_data": True,
        "used_stale_features": False, "odds_source": "odds_api",
        "candidate_source": "live_market_only",
        "orientation_source": "exact_match",
        "kalshi_probability": None,
    }])

    out = build_best_picks_df(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["best_pick"] == expected_pick
    assert "unresolved" not in str(row["best_pick"]).lower()
    assert pd.isna(row["market_line_used"])
    assert row["market_line_source"] == "live"
    assert row["market_line_source_detail"] == "live_odds"
    assert bool(row["line_consistency_flag"])
    assert bool(row["line_event_identity_match_flag"])
    assert row["line_event_identity_reason"] == "exact_live_event_identity"
    assert int(row["line_candidate_count"]) == 1
    assert not bool(row["production_eligible"])
    assert float(row["Kelly_Bet_Size"]) == 0.0

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

def test_best_picks_excludes_moneyline_when_both_gates_are_disabled(monkeypatch):
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_BEST_AVAILABLE", False)
    monkeypatch.setattr(wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False)
    common = {
        "league": "MLB", "home_team": "HomeC", "away_team": "AwayC",
        "game_date": "2026-07-27", "matchup_id": "2026-07-27|HomeC|AwayC",
        "calibrated_probability": 0.60, "ml_probability": 0.60,
        "model_probability": 0.60, "spread_line": pd.NA,
        "line_source": "live_odds", "live_spread_line": pd.NA,
        "is_live_data": True, "used_stale_features": False,
        "odds_source": "odds_api", "candidate_source": "live_market_only",
        "orientation_source": "exact_match", "kalshi_probability": 0.55,
    }
    df = pd.DataFrame([
        {
            **common,
            "market_type": "moneyline_home", "expected_value": 0.50,
            "edge": 0.30, "odds_american": 120, "total_line": pd.NA,
            "live_total_line": pd.NA,
        },
        {
            **common,
            "market_type": "total_over", "expected_value": 0.01,
            "edge": 0.01, "odds_american": -110, "total_line": 8.5,
            "live_total_line": 8.5,
        },
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)

    assert len(out) == 1
    assert out.iloc[0]["market_type"] == "total_over"
    audit = diagnostics["candidate_audit_df"]
    assert not audit["market_type"].astype(str).str.startswith("moneyline").any()

