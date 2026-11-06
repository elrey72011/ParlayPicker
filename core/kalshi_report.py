"""Does Kalshi add information beyond the sportsbook price? (owner request 2026-11-06)

The app's probability blend gives Kalshi the largest weight (``KALSHI_WEIGHT`` 0.40), but
only when Kalshi says >= 0.55 for the pick (tier 1). This measures that choice on the
same verified, settled saved best picks as the paper track record
(``core.research_performance.accepted_rows``), using the probabilities recorded at pick
time, both oriented to the pick side:

* ``brier_kalshi`` vs ``brier_market``: which predicted the results better (lower wins).
  ``market`` is the de-vigged book price (``fair_market_probability``, else
  ``market_probability``).
* ``mean_abs_gap``: how far Kalshi sat from the book price.
* ``tier1``: picks where Kalshi said >= 0.55. Their hit rate vs what the book implied.
* ``kalshi_above_book``: picks where Kalshi was >= 5 points above the book. If Kalshi had
  real information these would beat the book-implied rate.

Descriptive only; it changes no weight. A verdict needs ``MIN_DECIDED`` decided picks.
"""
from __future__ import annotations

from collections import defaultdict

from core.track_record import SIDE, wilson

MIN_DECIDED = 100
TIER1 = 0.55
GAP = 0.05


def _prob(value):
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


def _rows(accepted):
    out = []
    for row in accepted:
        k = _prob(row.get("kalshi_probability"))
        m = _prob(row.get("fair_market_probability")) or _prob(row.get("market_probability"))
        market = SIDE.get(row.get("market_type"))
        if k is None or m is None or market is None or row.get("candidate_outcome") not in ("WIN", "LOSS"):
            continue
        out.append({"group": f"{row.get('sport')} {market}", "k": k, "m": m,
                    "y": 1 if row["candidate_outcome"] == "WIN" else 0})
    return out


def _cohort(rows, key):
    picked = [r for r in rows if key(r)]
    wins = sum(r["y"] for r in picked)
    low, high = wilson(wins, len(picked))
    return {"picks": len(picked), "hit_rate": wins / len(picked) if picked else None,
            "hit_rate_95_low": low, "hit_rate_95_high": high,
            "book_implied": sum(r["m"] for r in picked) / len(picked) if picked else None,
            "kalshi_implied": sum(r["k"] for r in picked) / len(picked) if picked else None}


def _stats(rows):
    n = len(rows)
    brier_k = sum((r["k"] - r["y"]) ** 2 for r in rows) / n
    brier_m = sum((r["m"] - r["y"]) ** 2 for r in rows) / n
    if n < MIN_DECIDED:
        verdict = f"TOO_FEW_DECIDED (need {MIN_DECIDED})"
    else:
        verdict = "KALSHI_MORE_ACCURATE" if brier_k < brier_m else "KALSHI_NOT_MORE_ACCURATE"
    return {"decided": n, "brier_kalshi": brier_k, "brier_market": brier_m,
            "mean_abs_gap": sum(abs(r["k"] - r["m"]) for r in rows) / n,
            "tier1": _cohort(rows, lambda r: r["k"] >= TIER1),
            "kalshi_above_book": _cohort(rows, lambda r: r["k"] - r["m"] >= GAP),
            "verdict": verdict}


def build_from_rows(accepted):
    rows = _rows(accepted)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
        groups["ALL"].append(r)
    return {"schema": "kalshi-information-v1", "picks_with_kalshi": len(rows),
            "by_market": {k: _stats(v) for k, v in sorted(groups.items())},
            "min_decided_for_verdict": MIN_DECIDED, "weight_authority": False}


def build(database):
    from core.research_performance import accepted_rows, saved_rows
    accepted, _ = accepted_rows(saved_rows(database))
    return build_from_rows(accepted)


def markdown(report):
    def pct(v):
        return "" if v is None else f"{v:.1%}"
    lines = ["## Kalshi vs the book price (saved best picks)", "",
             f"{report['picks_with_kalshi']} settled picks with a Kalshi price. Lower Brier predicts results better. "
             "Descriptive only; it changes no weight.", "",
             "| market | decided | Brier Kalshi | Brier book | mean gap | Kalshi >= 55%: hit vs book-implied | "
             "Kalshi 5+ pts above book: hit vs book-implied | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for key, s in report["by_market"].items():
        t, a = s["tier1"], s["kalshi_above_book"]
        lines.append(f"| {key} | {s['decided']} | {s['brier_kalshi']:.3f} | {s['brier_market']:.3f} | "
                     f"{s['mean_abs_gap']:.1%} | {pct(t['hit_rate'])} vs {pct(t['book_implied'])} ({t['picks']}) | "
                     f"{pct(a['hit_rate'])} vs {pct(a['book_implied'])} ({a['picks']}) | {s['verdict']} |")
    if not report["by_market"]:
        lines.append("| | 0 | | | | | | no settled picks with a Kalshi price yet |")
    return "\n".join(lines) + "\n"
