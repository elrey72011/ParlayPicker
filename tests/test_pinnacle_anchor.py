"""Pinnacle shadow anchor and the anchor comparison. SYNTHETIC rows."""
from datetime import datetime, timedelta, timezone
import random

from core import blend_weights as bw
from core import pinnacle_anchor as pa

NOW = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)


def row(**over):
    base = {"market_type": "spread_home", "spread_line": -3.5, "pinnacle_point": -3.5,
            "pinnacle_odds_american": -125, "pinnacle_opposing_odds_american": 105,
            "pinnacle_updated_at": (NOW - timedelta(minutes=5)).isoformat()}
    base.update(over)
    return base


def test_fair_needs_the_exact_line_both_sides_and_a_fresh_quote():
    assert 0.53 < pa.fair(row(), NOW) < 0.535
    assert pa.fair(row(pinnacle_point=-4.5), NOW) is None                      # different line
    assert pa.fair(row(pinnacle_opposing_odds_american=None), NOW) is None     # one side only
    assert pa.fair(row(pinnacle_updated_at=(NOW - timedelta(hours=1)).isoformat()), NOW) is None   # stale
    assert pa.fair(row(market_type="total_over", total_line=52.5, pinnacle_point=52.5), NOW) is not None
    assert pa.fair(row(market_line_used=-2.5), NOW) is None                     # the used line wins


def test_anchor_comparison_prefers_the_sharper_price():
    rng = random.Random(4)
    rows = []
    for _ in range(600):
        truth = rng.uniform(0.35, 0.65)
        rows.append({"fair_market_probability": min(0.95, max(0.05, truth + rng.gauss(0, 0.08))),
                     "pinnacle_fair_probability": min(0.95, max(0.05, truth + rng.gauss(0, 0.02))),
                     "candidate_outcome": "WIN" if rng.random() < truth else "LOSS"})
    out = bw.anchor_comparison(rows)
    assert out["rows"] == 600 and out["pinnacle"]["log_loss"] < out["book"]["log_loss"]
    assert bw.anchor_comparison(rows[:10])["status"].startswith("TOO_FEW")
