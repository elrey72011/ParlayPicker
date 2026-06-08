#!/usr/bin/env python3
"""Edge-by-bucket report for graded MLB totals.

Buckets graded MLB Over/Under picks by direction, line, consensus, Kalshi
direction, and effective win probability, and reports hit-rate / ROI per bucket
so production no-stake gates can be set from data rather than a hunch.

Breakeven at -110 is 52.38%; buckets below that with a real sample are bleeding.
ROI is STAKE-WEIGHTED (a few big hammers dominate it), so judge gating by HIT
RATE, not ROI. This is the source for weights_config.MLB_TOTAL_NEUTRAL_NO_STAKE
and MLB_OVER_MID_LINE_NO_STAKE.

Usage
-----
    python3 scripts/edge_by_bucket.py data/backtest_exports
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.backtest_low_line_over import Pick, load_dir

BREAKEVEN = 0.5238  # -110


def _line_bucket(p: Pick) -> str:
    if p.line is None:
        return "?"
    if p.line < 8:
        return "<8.0 (low)"
    if p.line < 9.5:
        return "8.0-9.0 (mid)"
    if p.line < 11:
        return "9.5-10.5 (high)"
    return ">=11 (xhigh)"


def _block(title: str, picks: list[Pick]) -> None:
    g = [p for p in picks if p.graded]
    n = len(g)
    w = sum(1 for p in g if p.result == "WIN")
    stake = sum(p.kelly for p in g)
    profit = sum(p.profit for p in g)
    hit = (w / n) if n else 0.0
    roi = (profit / stake) if stake else 0.0
    flag = "" if n < 8 else ("  <-- EDGE" if hit >= BREAKEVEN else "  (bleed)")
    print(f"  {title:<34} n={n:>3}  {w}-{n - w}  hit={hit:5.1%}  ROI={roi:7.1%}{flag}")


def run(picks: list[Pick]) -> None:
    tot = [p for p in picks if p.league == "MLB"
           and p.market_type in ("total_over", "total_under") and p.graded]
    print(f"\nMLB graded totals: n={len(tot)}  (breakeven {BREAKEVEN:.1%} at -110)\n")

    print("=== by direction ===")
    for mt, lbl in (("total_over", "Over"), ("total_under", "Under")):
        _block(lbl, [p for p in tot if p.market_type == mt])

    print("\n=== direction x line bucket (the decisive view) ===")
    for mt, lbl in (("total_over", "Over"), ("total_under", "Under")):
        for b in ["<8.0 (low)", "8.0-9.0 (mid)", "9.5-10.5 (high)", ">=11 (xhigh)"]:
            _block(f"{lbl} / {b}", [p for p in tot if p.market_type == mt and _line_bucket(p) == b])

    print("\n=== by consensus ===")
    for c in ["Agrees", "Neutral", "Disagrees", "No Kalshi"]:
        _block(f"consensus={c}", [p for p in tot if p.consensus == c])

    print("\n=== direction x consensus ===")
    for mt, lbl in (("total_over", "Over"), ("total_under", "Under")):
        for c in ["Agrees", "Neutral", "Disagrees"]:
            _block(f"{lbl} / {c}", [p for p in tot if p.market_type == mt and p.consensus == c])

    print("\n=== by effective win probability bucket ===")
    for lo, hi in [(0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]:
        _block(f"eff_win [{lo:.2f},{hi:.2f})",
               [p for p in tot if p.eff_win_prob is not None and lo <= p.eff_win_prob < hi])


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_exports")
    run(load_dir(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
