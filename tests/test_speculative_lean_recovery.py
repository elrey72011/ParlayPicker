"""Speculative-lean empty-card recovery (22 Jun, user-directed thin-edge action).

When the card is otherwise empty, the best CLEAN positive-EV near-miss is surfaced at small
size. Guards: positive EV/edge, win >= 0.50, consensus not Disagrees, overs excluded, small
Kelly. These tests pin those guards so the bleed stays bounded.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_best_picks_calibration_pass import _row
from core.streamlit_pipeline import build_best_picks_df
from app_core.weights_config import EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT


def _under(kalshi, win=0.512, ev=0.034, edge=0.038, line=7.5, idx=970):
    df = pd.DataFrame([_row(idx=idx, league="MLB", market_type="total_under",
                            win_prob=win, ev=ev, edge=edge, kalshi_probability=kalshi)])
    df["live_total_line"] = [line]
    df["total_line"] = [line]
    df["best_pick"] = [f"Under {line}"]
    return build_best_picks_df(df).iloc[0]


def test_thin_positive_ev_under_is_recovered_small():
    # Below the 55% Actionable floor, but positive EV/edge + Agrees -> recovered at small size.
    row = _under(kalshi=0.52)
    assert row["Pick_Status"] == "Actionable"
    assert 0.0 < float(row["Kelly_Bet_Size"]) <= EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT + 1e-9
    assert row["consensus_agreement"] != "Disagrees"


def test_disagrees_under_not_recovered():
    # Never bet against Kalshi (the graded-loser bucket).
    row = _under(kalshi=0.45)
    assert row["consensus_agreement"] == "Disagrees"
    assert row["Pick_Status"] != "Actionable"
    assert float(row["Kelly_Bet_Size"]) == 0.0


def test_negative_ev_under_not_recovered():
    row = _under(kalshi=0.52, ev=-0.02, edge=0.01)
    assert row["Pick_Status"] != "Actionable"
    assert float(row["Kelly_Bet_Size"]) == 0.0


def test_over_not_recovered_even_if_thin_positive():
    # Overs bleed worst -> excluded from speculative recovery.
    df = pd.DataFrame([_row(idx=971, league="MLB", market_type="total_over",
                            win_prob=0.55, ev=0.05, edge=0.05, kalshi_probability=0.55)])
    df["live_total_line"] = [9.5]
    df["total_line"] = [9.5]
    df["best_pick"] = ["Over 9.5"]
    row = build_best_picks_df(df).iloc[0]
    assert row["Pick_Status"] != "Actionable"
    assert float(row["Kelly_Bet_Size"]) == 0.0
