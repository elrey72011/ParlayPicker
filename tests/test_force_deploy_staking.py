"""Force-deploy daily stake budget (user-directed, 17 Jun).

The day's eligible card is staked to SUM to DAILY_STAKE_BUDGET, split Actionable
(ACTIONABLE_STAKE_SHARE) vs viable non-Actionable, overriding the Kelly caps. A
tier with no picks deploys nothing; a health-suspended slate deploys nothing.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import optimize_portfolio_allocation
from app_core.weights_config import DAILY_STAKE_BUDGET, ACTIONABLE_STAKE_SHARE


def _row(pick, status, prob, odds=-110, health=""):
    return {
        "best_pick": pick,
        "Pick_Status": status,
        "calibrated_probability": prob,
        "odds_american": odds,
        "market_line_source": "live",
        "line_provenance_warning": "",
        "market_line_used": 7.5,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "canonical_pick_key": pick,
        "run_health_warning": health,
    }


def _df(rows):
    return pd.DataFrame(rows)


def test_force_deploy_splits_60_40_and_sums_to_budget():
    df = _df([
        _row("A Over 7.5", "Actionable", 0.60),
        _row("B Over 8.5", "Actionable", 0.58),
        _row("C Over 9.5", "High Variance/Speculative", 0.56),
        _row("D Over 6.5", "Below Threshold", 0.54),
    ])
    out = optimize_portfolio_allocation(df, bankroll=1000.0)
    status = out["Pick_Status"].str.lower()
    act = out.loc[status.eq("actionable"), "production_bet_amount"].sum()
    nonact = out.loc[status.isin(["high variance/speculative", "below threshold"]), "production_bet_amount"].sum()
    total = out["production_bet_amount"].sum()
    assert abs(act - DAILY_STAKE_BUDGET * ACTIONABLE_STAKE_SHARE) < 1.0   # ~3000
    assert abs(nonact - DAILY_STAKE_BUDGET * (1 - ACTIONABLE_STAKE_SHARE)) < 1.0  # ~2000
    assert abs(total - DAILY_STAKE_BUDGET) < 2.0   # ~5000


def test_force_deploy_actionable_only_card_deploys_just_60pct():
    # No non-Actionable picks -> only the Actionable 60% deploys (its budget is not
    # pushed onto a non-existent tier).
    df = _df([
        _row("A Over 7.5", "Actionable", 0.60),
        _row("B Over 8.5", "Actionable", 0.58),
    ])
    out = optimize_portfolio_allocation(df, bankroll=1000.0)
    total = out["production_bet_amount"].sum()
    assert abs(total - DAILY_STAKE_BUDGET * ACTIONABLE_STAKE_SHARE) < 1.0  # ~3000


def test_force_deploy_suspended_slate_stakes_nothing():
    warn = "slate_direction_imbalance: 100% of 13 totals are over — big-Kelly staking suspended"
    df = _df([
        _row("A Over 7.5", "Actionable", 0.60, health=warn),
        _row("C Over 9.5", "High Variance/Speculative", 0.56, health=warn),
    ])
    out = optimize_portfolio_allocation(df, bankroll=1000.0)
    assert float(out["production_bet_amount"].sum()) == 0.0


def test_force_deploy_skips_unsafe_lines():
    # A pick without a clean live line must not receive a forced stake.
    rows = [
        _row("A Over 7.5", "Actionable", 0.60),
        _row("B Over 8.5", "Actionable", 0.58),
    ]
    rows[1]["market_line_source"] = "upload"  # not live -> excluded
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    staked = out.loc[out["best_pick"].eq("B Over 8.5"), "production_bet_amount"].iloc[0]
    assert float(staked) == 0.0
    # The whole Actionable budget lands on the one safe pick.
    assert abs(out["production_bet_amount"].sum() - DAILY_STAKE_BUDGET * ACTIONABLE_STAKE_SHARE) < 1.0
