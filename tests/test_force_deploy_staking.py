"""Force-deploy daily stake budget (user-directed, 17 Jun) + concentration controls.

The day's eligible card is staked toward DAILY_STAKE_BUDGET, split Actionable
(ACTIONABLE_STAKE_SHARE) vs High Variance, with:
  - Below Threshold excluded from the non-Actionable tier,
  - a per-pick cap (FORCE_DEPLOY_MAX_PICK_PCT) so a thin tier under-deploys instead
    of dumping its whole budget onto one marginal pick,
  - nothing staked on a health-suspended slate or an unsafe line.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import optimize_portfolio_allocation
from app_core.weights_config import (
    DAILY_STAKE_BUDGET,
    ACTIONABLE_STAKE_SHARE,
    FORCE_DEPLOY_MAX_PICK_PCT,
)

MAX_PICK = DAILY_STAKE_BUDGET * FORCE_DEPLOY_MAX_PICK_PCT  # $750 on a $5000 budget


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


def test_force_deploy_splits_60_40_under_per_pick_cap():
    # Enough picks per tier that none hits the cap -> tiers fill to 60% / 40%.
    rows = [_row(f"A{i} Over 7.5", "Actionable", 0.60) for i in range(5)]
    rows += [_row(f"H{i} Over 8.5", "High Variance/Speculative", 0.56) for i in range(4)]
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    status = out["Pick_Status"].str.lower()
    act = out.loc[status.eq("actionable"), "production_bet_amount"]
    hv = out.loc[status.eq("high variance/speculative"), "production_bet_amount"]
    assert abs(act.sum() - DAILY_STAKE_BUDGET * ACTIONABLE_STAKE_SHARE) < 2.0   # ~3000
    assert abs(hv.sum() - DAILY_STAKE_BUDGET * (1 - ACTIONABLE_STAKE_SHARE)) < 2.0  # ~2000
    assert out["production_bet_amount"].max() <= MAX_PICK + 1e-6


def test_force_deploy_caps_a_lone_pick_instead_of_concentrating():
    # One Actionable pick must NOT absorb the whole $3000 — it's capped, and the tier
    # under-deploys by design.
    out = optimize_portfolio_allocation(_df([_row("A Over 7.5", "Actionable", 0.60)]), bankroll=1000.0)
    staked = float(out["production_bet_amount"].iloc[0])
    assert abs(staked - MAX_PICK) < 1e-6
    assert out["production_bet_amount"].sum() < DAILY_STAKE_BUDGET * ACTIONABLE_STAKE_SHARE


def test_force_deploy_excludes_below_threshold():
    # The 17 Jun case: a lone Below Threshold pick must get NO forced stake.
    rows = [
        _row("A Over 7.5", "Actionable", 0.60),
        _row("B Under 9.5", "Below Threshold", 0.49),
    ]
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    bt = float(out.loc[out["best_pick"].eq("B Under 9.5"), "production_bet_amount"].iloc[0])
    assert bt == 0.0


def test_force_deploy_suspended_slate_stakes_nothing():
    warn = "slate_direction_imbalance: 100% of 13 totals are over — big-Kelly staking suspended"
    rows = [
        _row("A Over 7.5", "Actionable", 0.60, health=warn),
        _row("H Over 9.5", "High Variance/Speculative", 0.56, health=warn),
    ]
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    assert float(out["production_bet_amount"].sum()) == 0.0


def test_force_deploy_skips_unsafe_lines():
    rows = [
        _row("A Over 7.5", "Actionable", 0.60),
        _row("B Over 8.5", "Actionable", 0.58),
    ]
    rows[1]["market_line_source"] = "upload"  # not live -> excluded
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    unsafe = float(out.loc[out["best_pick"].eq("B Over 8.5"), "production_bet_amount"].iloc[0])
    safe = float(out.loc[out["best_pick"].eq("A Over 7.5"), "production_bet_amount"].iloc[0])
    assert unsafe == 0.0
    assert abs(safe - MAX_PICK) < 1e-6  # lone safe pick capped, not given full $3000
