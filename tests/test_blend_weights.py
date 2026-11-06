"""Market-anchored blend weights fitted from settled picks. SYNTHETIC rows."""
import random

import numpy as np

from core import blend_weights as bw


def _rows(n, *, kalshi_signal, seed=1):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        market = rng.uniform(0.4, 0.6)
        edge = rng.gauss(0, 0.6)                       # information the market lacks
        truth = 1 / (1 + np.exp(-(np.log(market / (1 - market)) + kalshi_signal * edge)))
        k_logit = np.log(market / (1 - market)) + edge  # Kalshi sees the edge
        kalshi = float(1 / (1 + np.exp(-k_logit)))
        ml = min(0.95, max(0.05, 0.5 + rng.gauss(0, 0.1)))   # pure noise
        rows.append({"game_start_utc": f"2026-10-{1 + i // 40:02d}T{i % 24:02d}:00:00+00:00",
                     "fair_market_probability": market, "kalshi_probability": kalshi, "ml_probability": ml,
                     "calibrated_probability": 0.5 * market + 0.5 * kalshi,
                     "candidate_outcome": "WIN" if rng.random() < truth else "LOSS"})
    return rows


def test_informative_kalshi_gets_weight_and_noise_does_not():
    report = bw.study(_rows(1500, kalshi_signal=1.0))
    assert report["status"] == "FITTED"
    assert report["weights"]["kalshi"]["fitted"] > 0.5 and report["weights"]["kalshi"]["low_90"] > 0
    ml = report["weights"]["ml"]
    assert ml["low_90"] < 0 < ml["high_90"]                         # noise: interval spans 0
    s = report["test_scores"]
    assert s["fitted_blend"]["log_loss"] < s["market_only"]["log_loss"]


def test_uninformative_kalshi_shrinks_toward_the_market():
    report = bw.study(_rows(1500, kalshi_signal=0.0, seed=2))
    k = report["weights"]["kalshi"]
    assert k["low_90"] < 0 < k["high_90"] and abs(k["fitted"]) < 0.3


def test_fit_recovers_a_known_weight():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 1))
    offset = rng.normal(scale=0.3, size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(offset + 0.8 * x[:, 0])))).astype(float)
    assert abs(bw.fit(offset, x, y)[0] - 0.8) < 0.1


def test_small_samples_and_markdown():
    small = bw.study(_rows(50, kalshi_signal=1.0))
    assert small["status"].startswith("TOO_FEW") and "TOO_FEW" in bw.markdown(small)
    text = bw.markdown(bw.study(_rows(600, kalshi_signal=1.0)))
    assert "| kalshi |" in text and "market only" in text and "TheOver and sentiment" in text


def test_theover_is_read_as_a_pick_side_fraction_and_its_no_prediction_value_is_missing():
    from core.blend_weights import _theover, rows_from, INPUTS
    assert INPUTS["theover"] == "theover_probability"
    assert _theover(0.61) == 0.61 and _theover(61) == 0.61       # percentage read as a fraction
    assert _theover(0.5) is None and _theover(50) is None         # TheOver's "no prediction" value
    assert _theover(None) is None and _theover("x") is None and _theover(0) is None
    rows = rows_from([{"fair_market_probability": 0.5, "theover_probability": 58, "candidate_outcome": "WIN",
                       "game_start_utc": "2026-11-07T00:00:00Z"}])
    assert rows[0][2]["theover"] == 0.58


def test_theover_is_saved_with_each_pick():
    from app_core.candidate_evidence_schema import FIELDS
    assert "theover_probability" in FIELDS
