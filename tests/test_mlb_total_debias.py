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


# ---- overshoot guard: the correction must not push a game past the market ----

def test_debias_does_not_overshoot_market_over_games():
    # Slate-mean over-bias is large (5 strongly over-leaning games), but one game's
    # de-vig market actually favors Over (0.51) with only a small model lean. The old
    # flat -bias shift pushed it UNDER (0.54-0.15=0.39); the market-floored correction
    # keeps it on Over (floored at its 0.51 market) so the card follows the market
    # instead of overshooting to all-Under.
    mkt = [0.51, 0.46, 0.45, 0.44, 0.43, 0.42]
    cal = [0.54, 0.60, 0.61, 0.62, 0.63, 0.64]
    df = _df([("MLB", "total_over", m) for m in mkt])
    corrected, bias = apply_mlb_total_market_debias(pd.Series(cal), df)
    assert bias > 0.1
    assert corrected.iloc[0] >= 0.51 - 1e-9   # market-over game stays Over
    assert corrected.iloc[0] > 0.5
    # No over row is pushed below its own market (no overshoot past the anchor).
    assert bool((corrected.to_numpy() >= pd.Series(mkt).to_numpy() - 1e-9).all())


def test_debias_preserves_genuine_under_read_on_over_row():
    # An over row whose model already sits at/below the market (a genuine under read)
    # is left untouched — the correction only removes OVER excess.
    mkt = [0.50, 0.46, 0.45, 0.44, 0.43, 0.42]
    cal = [0.40, 0.60, 0.61, 0.62, 0.63, 0.64]
    df = _df([("MLB", "total_over", m) for m in mkt])
    corrected, _ = apply_mlb_total_market_debias(pd.Series(cal), df)
    assert abs(corrected.iloc[0] - 0.40) < 1e-9


def test_debias_under_rows_raised_toward_market_but_capped():
    over_rows = [("MLB", "total_over", 0.45)] * 6           # market P(over) 0.45
    under_rows = [("MLB", "total_under", 0.55), ("MLB", "total_under", 0.55)]  # market P(under) 0.55
    df = _df(over_rows + under_rows)
    cal = pd.Series([0.58] * 6 + [0.48, 0.70])             # bias ~0.13 from the over rows
    corrected, bias = apply_mlb_total_market_debias(cal, df)
    assert bias > 0.1
    assert 0.48 < corrected.iloc[6] <= 0.55 + 1e-9         # below-market under raised, capped at market
    assert abs(corrected.iloc[7] - 0.70) < 1e-9           # genuine strong-under read untouched
