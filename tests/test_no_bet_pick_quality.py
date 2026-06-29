"""Display hygiene: benched ($0-stake) picks must not wear a 'Value' tier label.

The S/A/B/C/D-Tier labels are scored from raw edge/EV and stamped onto every candidate,
including the picks the engine declines to stake. On an under-heavy card that made a stack
of $0 picks read as ranked 'C-Tier (Value)' bets when only the proven-bucket play carried
money. apply_no_bet_pick_quality relabels benched rows to 'No Bet - <status>' without
touching staking. See streamlit_pipeline.apply_no_bet_pick_quality.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.streamlit_pipeline import apply_no_bet_pick_quality


def _card():
    return pd.DataFrame(
        {
            "Pick_Quality": ["C-Tier (Value)", "C-Tier (Value)", "D-Tier (Weak/Negative)"],
            "Kelly_Bet_Size": [3.43, 0.0, 0.0],
            "Pick_Status": ["Actionable", "Below Threshold", "No Play"],
        }
    )


def test_staked_pick_keeps_its_tier_label():
    out = apply_no_bet_pick_quality(_card())
    assert out.loc[0, "Pick_Quality"] == "C-Tier (Value)"  # Kelly > 0 -> real bet, untouched


def test_benched_picks_are_relabeled_no_bet_with_status():
    out = apply_no_bet_pick_quality(_card())
    assert out.loc[1, "Pick_Quality"] == "No Bet - Below Threshold"
    assert out.loc[2, "Pick_Quality"] == "No Bet - No Play"


def test_staking_columns_are_untouched():
    card = _card()
    out = apply_no_bet_pick_quality(card)
    # Display-only: Kelly / status must be byte-for-byte unchanged.
    assert list(out["Kelly_Bet_Size"]) == [3.43, 0.0, 0.0]
    assert list(out["Pick_Status"]) == ["Actionable", "Below Threshold", "No Play"]


def test_empty_or_missing_columns_are_noops():
    assert apply_no_bet_pick_quality(pd.DataFrame()).empty
    no_quality = pd.DataFrame({"Kelly_Bet_Size": [0.0]})
    # No Pick_Quality column -> returned unchanged, no crash.
    assert "Pick_Quality" not in apply_no_bet_pick_quality(no_quality).columns


def test_all_staked_card_is_unchanged():
    card = pd.DataFrame(
        {
            "Pick_Quality": ["S-Tier (The Hammer)", "A-Tier (High Value)"],
            "Kelly_Bet_Size": [5.0, 2.0],
            "Pick_Status": ["Actionable", "Actionable"],
        }
    )
    out = apply_no_bet_pick_quality(card)
    assert list(out["Pick_Quality"]) == ["S-Tier (The Hammer)", "A-Tier (High Value)"]
