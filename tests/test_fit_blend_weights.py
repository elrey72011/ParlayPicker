"""Tests for the blend-weight backtest harness (scripts/fit_blend_weights.py)."""
import numpy as np
import pandas as pd
import pytest

from scripts.fit_blend_weights import (
    SIGNAL_COLS,
    _project_simplex,
    _to_outcome,
    _pct_to_float,
    blended_prob,
    brier,
    fit_weights,
    log_loss,
    load_exports,
    run_fit,
)


def test_blended_prob_matches_weighted_average_when_all_present():
    signals = np.array([[0.6, 0.5, 0.7, 0.65]])
    w = np.array([0.43, 0.17, 0.35, 0.05])  # Kalshi, Market, TheOver, ML order
    # NOTE: SIGNAL order in the harness is kalshi, market, ml, theover.
    # Here we just check the math: dot product of normalised weights.
    expected = float((signals[0] * w).sum() / w.sum())
    assert blended_prob(signals, w)[0] == pytest.approx(expected, abs=1e-9)


def test_blended_prob_drops_nan_and_renormalises():
    # Missing signal must be dropped and its weight redistributed — NOT treated as 0.5.
    signals = np.array([[0.6, 0.5, np.nan, 0.7]])
    w = np.array([0.4, 0.2, 0.1, 0.3])
    present = np.array([0.6, 0.5, 0.7])
    pw = np.array([0.4, 0.2, 0.3])
    expected = float((present * pw).sum() / pw.sum())
    got = blended_prob(signals, w)[0]
    assert got == pytest.approx(expected, abs=1e-9)
    # And it must differ from the buggy "fill 0.5" behaviour.
    buggy = float((np.array([0.6, 0.5, 0.5, 0.7]) * w).sum())
    assert abs(got - buggy) > 1e-3


def test_blended_prob_all_nan_falls_back_to_half():
    signals = np.array([[np.nan, np.nan, np.nan, np.nan]])
    assert blended_prob(signals, np.array([0.25, 0.25, 0.25, 0.25]))[0] == pytest.approx(0.5)


def test_fit_recovers_known_weights():
    # Generate data whose true win prob is a known blend of independent signals,
    # then confirm the fitter recovers weights close to the truth.
    rng = np.random.default_rng(42)
    n = 4000
    true_w = np.array([0.5, 0.1, 0.3, 0.1])
    sig = rng.uniform(0.2, 0.8, size=(n, 4))
    p_true = (sig * true_w).sum(axis=1)
    y = (rng.uniform(size=n) < p_true).astype(float)

    w = fit_weights(sig, y)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-9)
    # The dominant signal must be identified as dominant.
    assert np.argmax(w) == np.argmax(true_w)
    # Fitted weights should be in the right ballpark.
    assert np.allclose(w, true_w, atol=0.12)


def test_fitted_beats_uniform_on_logloss():
    rng = np.random.default_rng(1)
    n = 3000
    true_w = np.array([0.6, 0.05, 0.3, 0.05])
    sig = rng.uniform(0.3, 0.7, size=(n, 4))
    y = (rng.uniform(size=n) < (sig * true_w).sum(axis=1)).astype(float)
    w = fit_weights(sig, y)
    uniform = np.full(4, 0.25)
    assert log_loss(y, blended_prob(sig, w)) <= log_loss(y, blended_prob(sig, uniform)) + 1e-9


def test_project_simplex_sums_to_one_and_nonneg():
    v = np.array([0.9, -0.3, 0.5, 0.2])
    p = _project_simplex(v)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p >= -1e-12)


def test_outcome_mapping():
    s = pd.Series([" W ", "L", "WIN", "loss", "push", "x"])
    out = _to_outcome(s)
    assert list(out[:4]) == [1.0, 0.0, 1.0, 0.0]
    assert np.isnan(out[4]) and np.isnan(out[5])


def test_pct_parsing():
    s = pd.Series(["67.9%", "0.55", "72%", "0.5"])
    out = _pct_to_float(s)
    assert out.tolist() == pytest.approx([0.679, 0.55, 0.72, 0.5])


def test_brier_and_logloss_sanity():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    perfect = np.array([0.99, 0.01, 0.99, 0.01])
    coin = np.array([0.5, 0.5, 0.5, 0.5])
    assert brier(y, perfect) < brier(y, coin)
    assert log_loss(y, perfect) < log_loss(y, coin)
    assert brier(y, coin) == pytest.approx(0.25)
    assert log_loss(y, coin) == pytest.approx(0.693, abs=1e-3)


def test_run_fit_end_to_end(tmp_path):
    # A small graded export with the oriented blend-input columns.
    rng = np.random.default_rng(7)
    n = 400
    sig = rng.uniform(0.3, 0.7, size=(n, 4))
    true_w = np.array([0.5, 0.1, 0.3, 0.1])
    y = (rng.uniform(size=n) < (sig * true_w).sum(axis=1)).astype(int)
    df = pd.DataFrame({
        "league": "MLB",
        "market_type": "total_under",
        "blend_in_kalshi": sig[:, 0],
        "blend_in_market": sig[:, 1],
        "blend_in_ml": sig[:, 2],
        "blend_in_theover": sig[:, 3],
        "Pick_Outcome": np.where(y == 1, "WIN", "LOSS"),
    })
    p = tmp_path / "best_picks_export_test.csv"
    df.to_csv(p, index=False)

    loaded = load_exports(str(p), None)
    assert "__y" in loaded.columns
    assert set(SIGNAL_COLS).issubset(loaded.columns)
    # Should run without raising and return 0.
    assert run_fit(loaded, "MLB", "total", min_samples=100, folds=3) == 0
