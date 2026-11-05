"""Market-anchored NCAAF model screen. SYNTHETIC seasons; no network."""
import random

from app_core import ncaaf_line_model as lm
from app_core.ncaaf_research import FEATURES


def _season(seed, n, signal):
    rng = random.Random(seed)
    rows, lines = [], {}
    for i in range(n):
        gid = seed * 10_000 + i
        feats = {k: rng.uniform(10, 40) for k in FEATURES}
        feats["neutral_site"] = 0.0
        hidden = rng.gauss(0, 7)                      # what the market misses, visible via home_ppg
        feats["home_ppg"] = 25 + hidden * signal
        spread = -round(rng.gauss(0, 10) * 2) / 2 - 0.5 * (rng.random() < 0.5)
        margin = round(-spread + hidden + rng.gauss(0, 10))
        total = 55.5
        rows.append({"game_id": gid, **feats, "y_margin": margin, "y_total": round(total + rng.gauss(0, 12))})
        lines[gid] = {"spread": spread, "total": total, "provider": "consensus"}
    return rows, lines


def test_no_signal_means_near_zero_centres_and_few_confident_picks():
    train, l1 = _season(1, 400, signal=0.0)
    cal, l2 = _season(2, 400, signal=0.0)
    test, l3 = _season(3, 400, signal=0.0)
    lines = {**l1, **l2, **l3}
    model = lm.fit(train, cal, lines, "SPREAD")
    result = lm.evaluate(model, test, lines)
    assert result["games"] == 400 and result["mean_abs_centre"] < 2.0
    assert result["by_threshold"]["0.58"]["picks"] < 60


def test_a_real_signal_is_found_and_beats_break_even():
    train, l1 = _season(1, 600, signal=1.0)
    cal, l2 = _season(2, 600, signal=1.0)
    test, l3 = _season(3, 600, signal=1.0)
    lines = {**l1, **l2, **l3}
    result = lm.evaluate(lm.fit(train, cal, lines, "SPREAD"), test, lines)
    assert result["by_threshold"]["0.55"]["hit_rate"] > 0.55
    assert lm.fit(train[:10], cal, lines, "SPREAD") is None   # too few rows
