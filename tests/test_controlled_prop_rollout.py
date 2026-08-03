import pandas as pd

from app_core.export_scope import label_wager_export
from app_core.prop_calibration import load_prop_results_log
from app_core.prop_runner import (
    apply_controlled_prop_rollout,
    apply_prop_stake_status,
)


def _qualified_card(markets, stakes=None):
    stakes = stakes or [8.0] * len(markets)
    return pd.DataFrame({
        "player": [f"Player {index}" for index in range(len(markets))],
        "market_type": markets,
        "best_pick": [f"Pick {index}" for index in range(len(markets))],
        "Pick_Status": ["Actionable"] * len(markets),
        "Prop_Tier": ["Core"] * len(markets),
        "production_eligible": [True] * len(markets),
        "production_gate_reason": ["Core production qualified"] * len(markets),
        "Kelly_Bet_Size": stakes,
    })


def _ledger(market, wins, losses, *, odds=-110, date="2026-08-03"):
    count = wins + losses
    return pd.DataFrame({
        "game_date": [date] * count,
        "market_type": [market] * count,
        "pick": ["Historical pick"] * count,
        "result": ["WIN"] * wins + ["LOSS"] * losses,
        "RawWinProbability": [0.65] * count,
        "odds_american": [odds] * count,
    })


def test_rollout_funds_only_batter_hits_and_caps_pilot_at_one_dollar():
    card = _qualified_card([
        "batter_hits_over",
        "pitcher_strikeouts_over",
        "batter_total_bases_under",
    ])

    out = apply_controlled_prop_rollout(
        card,
        results_history=None,
        as_of_date="2026-08-03",
    )
    out = apply_prop_stake_status(out)
    exported = label_wager_export(out)

    assert out["production_eligible"].tolist() == [True, False, False]
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 0.0, 0.0]
    assert out["Stake_Status"].tolist() == [
        "Funded", "Research / No Stake", "Research / No Stake"
    ]
    assert exported["Bettable"].tolist() == [True, False, False]
    assert exported["Wager_Instruction"].tolist() == [
        "BET - APP APPROVED",
        "DO NOT BET - $0 PASS / RESEARCH",
        "DO NOT BET - $0 PASS / RESEARCH",
    ]
    assert bool(out.loc[0, "controlled_pilot_mode"])
    assert bool(out.loc[0, "controlled_pilot_stake_cap_applied"])
    assert "batter-hit pilot" in out.loc[0, "production_gate_reason"]
    assert "batter-hit props only" in out.loc[1, "production_gate_reason"]


def test_rollout_graduates_direction_after_fifty_profitable_post_fix_results():
    card = _qualified_card(["batter_hits_over"], stakes=[8.0])
    history = _ledger("batter_hits_over", wins=35, losses=15)

    out = apply_controlled_prop_rollout(
        card,
        history,
        as_of_date="2026-08-04",
    )

    assert bool(out.loc[0, "controlled_pilot_graduated"])
    assert not bool(out.loc[0, "controlled_pilot_mode"])
    assert out.loc[0, "controlled_pilot_post_fix_graded"] == 50
    assert out.loc[0, "controlled_pilot_post_fix_hit_rate"] == 0.70
    assert out.loc[0, "controlled_pilot_post_fix_roi"] > 0
    assert out.loc[0, "Kelly_Bet_Size"] == 8.0


def test_rollout_does_not_graduate_on_sample_size_without_positive_roi():
    card = _qualified_card(["batter_hits_under"], stakes=[8.0])
    history = _ledger(
        "batter_hits_under", wins=35, losses=15, odds=-300
    )

    out = apply_controlled_prop_rollout(
        card,
        history,
        as_of_date="2026-08-04",
    )

    assert out.loc[0, "controlled_pilot_post_fix_graded"] == 50
    assert out.loc[0, "controlled_pilot_post_fix_hit_rate"] == 0.70
    assert out.loc[0, "controlled_pilot_post_fix_roi"] < 0
    assert not bool(out.loc[0, "controlled_pilot_graduated"])
    assert out.loc[0, "Kelly_Bet_Size"] == 1.0


def test_rollout_excludes_pre_fix_and_same_day_results_from_graduation():
    card = _qualified_card(["batter_hits_over"], stakes=[8.0])
    history = pd.concat([
        _ledger(
            "batter_hits_over", wins=100, losses=0, date="2026-08-01"
        ),
        _ledger(
            "batter_hits_over", wins=50, losses=0, date="2026-08-03"
        ),
    ], ignore_index=True)

    out = apply_controlled_prop_rollout(
        card,
        history,
        as_of_date="2026-08-03",
    )

    assert out.loc[0, "controlled_pilot_post_fix_graded"] == 0
    assert not bool(out.loc[0, "controlled_pilot_graduated"])
    assert out.loc[0, "Kelly_Bet_Size"] == 1.0


def test_bundled_ledger_restores_current_hit_pilot_evidence_after_restart():
    """A fresh deployment must not silently reset the active pilot to zero."""
    card = _qualified_card(
        ["batter_hits_over", "batter_hits_under"], stakes=[8.0, 8.0]
    )

    out = apply_controlled_prop_rollout(
        card,
        load_prop_results_log(),
        as_of_date="2026-08-03",
    )

    assert out["controlled_pilot_post_fix_graded"].tolist() == [20, 15]
    assert out["controlled_pilot_post_fix_hit_rate"].tolist() == [0.65, 8 / 15]
    assert out.loc[0, "controlled_pilot_post_fix_roi"] > 0
    assert out.loc[1, "controlled_pilot_post_fix_roi"] < 0
    assert out["controlled_pilot_graduated"].tolist() == [False, False]
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 1.0]
