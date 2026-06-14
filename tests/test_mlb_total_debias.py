"""Tests for apply_mlb_total_market_debias — the shared helper now wired into BOTH
the Analysis-tab and production best-picks blend paths (the production path was
missed by #1919, leaving the card all-Over)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.streamlit_pipeline import apply_mlb_total_market_debias


def _df(rows):
    return pd.DataFrame(rows, columns=["league", "market_type", "market_probability"])


def test_debias_lowers_overs_and_raises_unders():
    df = _df([("MLB", "total_over", 0.46)] * 6 + [("MLB", "total_under", 0.54)] * 6)
    cal = pd.Series([0.58] * 6 + [0.42] * 6)
    corrected, bias = apply_mlb_total_market_debias(cal, df)
    assert abs(bias - 0.12) < 1e-6              # mean(over 0.58) - mean(mkt 0.46)
    assert abs(corrected.iloc[0] - (0.58 - 0.12)) < 1e-6   # over lowered
    assert abs(corrected.iloc[6] - (0.42 + 0.12)) < 1e-6   # under raised


def test_debias_leaves_non_mlb_and_non_totals_untouched():
    df = _df([("NBA", "total_over", 0.46)] * 6 + [("MLB", "spread_home", 0.46)] * 6)
    cal = pd.Series([0.58] * 12)
    corrected, bias = apply_mlb_total_market_debias(cal, df)
    assert bias == 0.0
    assert corrected.equals(cal)


def test_debias_rebalances_real_jun14_card():
    # The 15 MLB total_over rows from run 133905Z: blended WinProb vs de-vig market.
    mdl = [0.5415, 0.5516, 0.5465, 0.5992, 0.5734, 0.522, 0.6068, 0.5616,
           0.5341, 0.6057, 0.5078, 0.5596, 0.5538, 0.5273, 0.5358]
    mkt = [0.4398, 0.4553, 0.4513, 0.5128, 0.4761, 0.4455, 0.5105, 0.5059,
           0.4593, 0.5366, 0.4533, 0.4988, 0.5012, 0.4789, 0.4890]
    df = _df([("MLB", "total_over", m) for m in mkt])
    corrected, bias = apply_mlb_total_market_debias(pd.Series(mdl), df)
    assert bias > 0.07
    overs = int((corrected > 0.5).sum())
    # Was 15/15 Over; only the genuinely strong overs survive the de-bias.
    assert overs <= 5
