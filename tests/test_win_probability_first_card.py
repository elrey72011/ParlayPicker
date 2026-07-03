"""Win-probability-first across the games card (owner preference, 3 Jul).

"I don't want the highest ROI, I want the best pick that has the highest chance
of winning." Three enforcement points on the main card (props got the same
treatment in app_core/prop_runner.py):
  * MIN_STAKE_WIN_PROBABILITY: no stake below the floor regardless of EV — the
    terminal override of the high-EV longshot exception paths.
  * Card ranking: within a tier, higher win probability outranks higher EV.
  * Empty-card recovery: selects the most-likely-to-win candidate, not the
    best-paying one, and applies the same floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.streamlit_pipeline as sp
from app_core.weights_config import MIN_STAKE_WIN_PROBABILITY


def _row(**overrides) -> dict:
    row = {
        "league": "MLB",
        "home_team": "Philadelphia",
        "away_team": "Pittsburgh",
        "game_date": pd.Timestamp("2026-07-10", tz="UTC"),
        "market_type": "total_under",
        "total_line": 8.5,
        "spread_line": pd.NA,
        "expected_value": 0.05,
        "edge": 0.04,
        "calibrated_probability": 0.58,
        "effective_win_probability": 0.58,
        "market_probability": 0.50,
        "ml_probability": 0.58,
        "odds_american": -110,
        "odds_source": "odds_api",
        "consensus_agreement": "Agrees",
        "is_live_data": True,
        "candidate_source": "upload",
        "orientation_source": "model",
        "upload_match_reason": "matched",
    }
    row.update(overrides)
    return row


def test_stake_floor_unstakes_high_ev_longshot():
    # The MLB_SPREAD_HIGH_EV underdog shape: huge EV, sub-50% win probability.
    # It must end the pipeline unstaked (not Actionable), whatever promoted it.
    df = pd.DataFrame([
        _row(
            home_team="Texas", away_team="Seattle", market_type="spread_home",
            total_line=pd.NA, spread_line=-1.5,
            expected_value=0.27, edge=0.11,
            calibrated_probability=0.47, effective_win_probability=0.47,
            ml_probability=0.47, odds_american=175,
        )
    ])
    best = sp.build_best_picks_df(df)
    row = best.iloc[0]
    assert str(row["Pick_Status"]) != "Actionable"
    assert float(pd.to_numeric(row["Kelly_Bet_Size"], errors="coerce") or 0) == 0.0


def test_stake_floor_leaves_qualified_favorites_alone():
    # A pick above the floor must not be touched by the floor stage (whether it
    # ends Actionable depends on the other gates, but never via the floor).
    df = pd.DataFrame([_row(effective_win_probability=0.62, calibrated_probability=0.62)])
    best = sp.build_best_picks_df(df)
    assert str(best.iloc[0].get("status_blocker_stage")) != "min_win_probability_floor"


def test_ranking_orders_by_win_probability_within_tier():
    # Two same-status picks: A pays better (higher EV), B is likelier to win.
    # B must outrank A in the final card order.
    df = pd.DataFrame([
        _row(home_team="Atlanta", away_team="Miami",
             expected_value=0.20, edge=0.10,
             calibrated_probability=0.56, effective_win_probability=0.56),
        _row(home_team="Boston", away_team="Toronto",
             expected_value=0.06, edge=0.04,
             calibrated_probability=0.64, effective_win_probability=0.64),
    ])
    best = sp.build_best_picks_df(df)
    # Compare positions among rows sharing the same Pick_Status bucket.
    statuses = best["Pick_Status"].astype(str)
    same_status = best[statuses.eq(statuses.iloc[0])] if len(set(statuses)) == 1 else best
    order = same_status["home_team"].tolist()
    assert order.index("Boston") < order.index("Atlanta"), (
        f"likelier winner must rank first, got {order}"
    )


def test_recovery_floor_blocks_sub_floor_candidates():
    # Recovery-eligible shape but below the win-probability floor: with the whole
    # card empty, recovery must NOT stake it.
    df = pd.DataFrame([
        _row(
            expected_value=0.12, edge=0.06,
            calibrated_probability=MIN_STAKE_WIN_PROBABILITY - 0.03,
            effective_win_probability=MIN_STAKE_WIN_PROBABILITY - 0.03,
        )
    ])
    best = sp.build_best_picks_df(df)
    assert not bool(best["production_eligible"].fillna(False).astype(bool).any())
