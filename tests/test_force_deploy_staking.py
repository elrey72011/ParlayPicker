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
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import optimize_portfolio_allocation
from app_core import weights_config
from app_core.weights_config import (
    PRODUCTION_ABSOLUTE_MAX_PICK_DOLLARS,
    PRODUCTION_ABSOLUTE_MAX_SLATE_DOLLARS,
    PRODUCTION_MAX_PICK_PCT,
    PRODUCTION_MAX_SLATE_PCT,
)


@pytest.fixture(autouse=True)
def _enable_force_deploy(monkeypatch):
    # Production default is now DAILY_STAKE_FORCE_DEPLOY=False (turned off 20 Jun after it
    # force-staked corrupt-data rows on no-edge slates). These tests validate the
    # force-deploy LOGIC for when it is explicitly re-enabled, so switch it on here.
    monkeypatch.setattr(weights_config, "DAILY_STAKE_FORCE_DEPLOY", True)

TEST_BANKROLL = 1000.0
MAX_PICK = min(TEST_BANKROLL * PRODUCTION_MAX_PICK_PCT, PRODUCTION_ABSOLUTE_MAX_PICK_DOLLARS)
MAX_SLATE = min(TEST_BANKROLL * PRODUCTION_MAX_SLATE_PCT, PRODUCTION_ABSOLUTE_MAX_SLATE_DOLLARS)


def _row(pick, status, prob, odds=-110, health="", consensus="Agrees", league="MLB"):
    decimal_odds = 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    return {
        "best_pick": pick,
        "league": league,
        "Pick_Status": status,
        "calibrated_probability": prob,
        "odds_american": odds,
        "expected_value": (prob * decimal_odds) - 1.0,
        "consensus_agreement": consensus,
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


def test_force_deploy_remains_bounded_by_final_production_caps():
    rows = [_row(f"A{i} Over 7.5", "Actionable", 0.60) for i in range(5)]
    rows += [_row(f"H{i} Over 8.5", "High Variance/Speculative", 0.56) for i in range(4)]
    out = optimize_portfolio_allocation(_df(rows), bankroll=TEST_BANKROLL)
    assert out["production_bet_amount"].max() <= MAX_PICK + 1e-6
    assert out["production_bet_amount"].sum() <= MAX_SLATE + 1e-6


def test_force_deploy_caps_a_lone_pick_instead_of_concentrating():
    # One Actionable pick must NOT absorb the whole $3000 â€” it's capped, and the tier
    # under-deploys by design.
    out = optimize_portfolio_allocation(_df([_row("A Over 7.5", "Actionable", 0.60)]), bankroll=1000.0)
    staked = float(out["production_bet_amount"].iloc[0])
    assert abs(staked - MAX_PICK) < 1e-6
    assert out["production_bet_amount"].sum() <= MAX_SLATE


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
    warn = "slate_direction_imbalance: 100% of 13 totals are over â€” big-Kelly staking suspended"
    rows = [
        _row("A Over 7.5", "Actionable", 0.60, health=warn),
        _row("H Over 9.5", "High Variance/Speculative", 0.56, health=warn),
    ]
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    assert float(out["production_bet_amount"].sum()) == 0.0


def test_force_deploy_non_actionable_excludes_disagrees():
    # A High Variance pick where Kalshi DISAGREES (backs the other side) must not be
    # staked â€” we never bet against the market on the speculative tier.
    rows = [
        _row("A Over 7.5", "High Variance/Speculative", 0.56, consensus="Disagrees"),
        _row("B Over 8.5", "High Variance/Speculative", 0.56, consensus="Neutral"),
    ]
    out = optimize_portfolio_allocation(_df(rows), bankroll=1000.0)
    dis = float(out.loc[out["best_pick"].eq("A Over 7.5"), "production_bet_amount"].iloc[0])
    neu = float(out.loc[out["best_pick"].eq("B Over 8.5"), "production_bet_amount"].iloc[0])
    assert dis == 0.0          # Disagrees gets nothing
    assert neu > 0.0           # Neutral still staked


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
    assert abs(safe - MAX_PICK) < 1e-6


def test_force_deploy_does_not_bypass_degraded_model_guard():
    row = _row("A Over 7.5", "Actionable", 0.60)
    row["model_status"] = "statistical fallback"
    out = optimize_portfolio_allocation(_df([row]), bankroll=TEST_BANKROLL)
    assert float(out["production_bet_amount"].sum()) == 0.0


def test_force_deploy_does_not_equal_weight_zero_edge_rows():
    out = optimize_portfolio_allocation(
        _df([_row("A Over 7.5", "Actionable", 0.50)]),
        bankroll=TEST_BANKROLL,
    )
    assert float(out["production_bet_amount"].sum()) == 0.0


def test_empty_fallback_summary_does_not_block_clean_row():
    row = _row("Boston Over 7.5", "Actionable", 0.60)
    row["fallback_summary_by_league"] = "{}"
    out = optimize_portfolio_allocation(_df([row]), bankroll=TEST_BANKROLL)
    assert float(out["production_bet_amount"].iloc[0]) > 0.0


def test_unrelated_league_fallback_does_not_block_clean_row():
    row = _row("Boston Over 7.5", "Actionable", 0.60, league="MLB")
    row["fallback_summary_by_league"] = "{'WNBA': 12}"
    row["run_health_warning"] = "Run health warning: WNBA fallback usage is elevated."
    out = optimize_portfolio_allocation(_df([row]), bankroll=TEST_BANKROLL)
    assert float(out["production_bet_amount"].iloc[0]) > 0.0


def test_matching_league_fallback_blocks_row():
    row = _row("Dallas +4.5", "Actionable", 0.60, league="WNBA")
    row["fallback_summary_by_league"] = "{'WNBA': 12}"
    row["run_health_warning"] = "Run health warning: WNBA fallback usage is elevated."
    out = optimize_portfolio_allocation(_df([row]), bankroll=TEST_BANKROLL)
    assert float(out["production_bet_amount"].iloc[0]) == 0.0

