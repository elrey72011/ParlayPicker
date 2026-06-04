#!/usr/bin/env python3
"""
Backtest: does overriding the market (Kalshi) on MLB totals help or hurt?

Why this proxy
--------------
The graded historical exports (data/backtest_exports) do NOT carry TheOver's
WinProbSource tag — it was never persisted before 2 Jun — so a literal
source-by-source split (model_hit_rate vs model_hit_rate_flipped vs
public_betting_pct) is not reconstructable from saved data.

But `consensus_agreement` IS in every graded row, and it encodes the exact thing
the source-gating is a proxy for:
  * Agrees    -> Kalshi favors the pick's direction (market-aligned)
  * Disagrees -> Kalshi favors the OTHER direction (the model/TheOver overrode the
                 market — the same situation model_hit_rate_flipped creates)
  * Neutral   -> Kalshi present but no strong directional lean

So "Disagrees" win-rate/ROI vs "Agrees" answers: when our model overrides the
market on an MLB total, do we win? That is the data behind gating low-quality
TheOver sources and trusting the market-led setup.

Usage
-----
    python3 scripts/backtest_theover_direction.py data/backtest_exports
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.backtest_low_line_over import Pick, load_dir, _summary, _print_block


def _block(title: str, picks: list[Pick]) -> None:
    _print_block(title, picks)


def run(picks: list[Pick]) -> None:
    totals = [p for p in picks if p.league == "MLB" and p.market_type in ("total_over", "total_under")]

    print("\n" + "=" * 84)
    print("MLB totals: market-aligned (Agrees) vs market-overridden (Disagrees)")
    print("=" * 84)
    print("CAVEAT: ROI below is STAKE-WEIGHTED. A few big Actionable Over hammers (genuine")
    print("  model_hit_rate/public picks) dominate it and can swamp hit-rate. Do NOT use the")
    print("  ROI buckets alone to set MLB_THEOVER_FADE_SHRINK — the fade only governs flipped")
    print("  (TheOver-Under) games; judge it by the flipped-game Over-vs-Under counterfactual.")
    _block("All MLB totals", totals)
    print("\n-- by consensus_agreement --")
    for c in ("Agrees", "Neutral", "Disagrees"):
        _block(f"  {c}", [p for p in totals if p.consensus == c])

    print("\n-- by direction --")
    for mt, label in (("total_over", "Overs"), ("total_under", "Unders")):
        _block(f"  {label}", [p for p in totals if p.market_type == mt])

    print("\n-- direction x consensus (the decisive cells) --")
    for mt, label in (("total_over", "Over"), ("total_under", "Under")):
        for c in ("Agrees", "Neutral", "Disagrees"):
            sub = [p for p in totals if p.market_type == mt and p.consensus == c]
            _block(f"  {label} / {c}", sub)

    # Sharper metric than the consensus label: kalshi_probability is oriented to the
    # pick side, so < 0.50 means Kalshi prices the OTHER direction — a *true* market
    # override (what the source-gating targets). The `consensus=Disagrees` tag is noisier:
    # it also fires when Kalshi agrees on direction but is merely more confident than the
    # model, which is not an override at all.
    kal = [p for p in totals if p.kalshi_prob is not None]
    print("\n-- by Kalshi direction (kalshi_probability oriented to pick side) --")
    _block("  Kalshi OPPOSES pick (<0.50) = true market override", [p for p in kal if p.kalshi_prob < 0.50])
    _block("  Kalshi favors pick (>=0.50) = market-aligned", [p for p in kal if p.kalshi_prob >= 0.50])
    _block("  (no kalshi value on row)", [p for p in totals if p.kalshi_prob is None])

    print("\n-- spotlight: Unders where the market disagreed (the flipped-source analogue) --")
    flipped_like = [p for p in totals if p.market_type == "total_under" and p.consensus == "Disagrees"]
    _block("  Under / Disagrees", flipped_like)
    over_when_market_agrees = [p for p in totals if p.market_type == "total_over" and p.consensus in ("Agrees", "Neutral")]
    _block("  Over / Agrees+Neutral (market-aligned overs)", over_when_market_agrees)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_exports")
    picks = load_dir(path)
    run(picks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
