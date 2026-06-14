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


# ---- regression: the post-Kalshi re-blend in streamlit_app must NOT undo the de-bias ----

def test_recompute_consensus_reapplies_mlb_total_debias():
    """_recompute_consensus_from_kalshi re-blends calibrated_probability/EV/edge from
    scratch (the THIRD blend path). Before the fix it dropped the MLB total market
    de-bias, so the production card on 14 Jun rebuilt on the raw over-lean and shipped
    15/15 Over. The de-bias must be re-applied here so EV/edge/selection rebalance."""
    from streamlit_app import _recompute_consensus_from_kalshi

    # 8 MLB total_over games whose post-Kalshi blend sits systematically above the
    # de-vig market (model over-lean) — exactly the 14 Jun pattern.
    n = 8
    df = pd.DataFrame({
        "best_pick": [f"Over {i}" for i in range(n)],
        "odds_american": [-110.0] * n,
        "market_probability": [0.46] * n,
        "kalshi_probability": [0.575] * n,   # tier 1
        "ml_probability": [0.72] * n,
        "model_probability": [0.72] * n,
        "theover_probability": [float("nan")] * n,
        "sentiment_diff": [0.5] * n,
        "league": ["MLB"] * n,
        "market_type": ["total_over"] * n,
    })

    out = _recompute_consensus_from_kalshi(df)

    # The de-bias fired in THIS function (only place that could add the column here).
    assert "mlb_total_market_debias" in out.columns
    assert float(out["mlb_total_market_debias"].iloc[0]) > 0.0
    # Over calibrated_probability was pulled down toward the market; the raw blend sat
    # well above 0.5, so after de-bias the systematic over-lean is removed.
    assert out["calibrated_probability"].mean() < 0.52
    # edge recomputed from the de-biased probability (was the un-debiased gap before).
    assert (out["edge"] < 0.10).all()
