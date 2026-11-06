"""Forward shadow: the app's blend vs the market price alone (owner request 2026-11-06).

The first paper track record (748 settled picks, 2026-11-06) showed neither the current
blend nor a fitted one beating the de-vigged market price on later picks (log loss
0.6658 / 0.6664 vs 0.6639). Before changing any weight, this scores both on picks saved
**from ``SHADOW_START`` on**, so the comparison is prospective and the decision is not
fitted to the picks that suggested it.

For each settled saved pick (``core.research_performance.accepted_rows``), with both
probabilities oriented to the pick side as saved:

* blend: ``calibrated_probability`` (what the app showed and selected on);
* market: ``fair_market_probability``, else ``market_probability``;
* Pinnacle: ``pinnacle_fair_probability`` (no-vig, same line), on the picks that have it.

Reported per sport and market and overall: picks, log loss and Brier for each, and the
difference. ``MARKET_AS_GOOD_OR_BETTER`` / ``BLEND_BETTER`` only after ``MIN_DECIDED``
picks; before that ``TOO_FEW_DECIDED``. Descriptive only; it changes no weight.
"""
from __future__ import annotations

import math
from collections import defaultdict

from core.track_record import SIDE

SHADOW_START = "2026-11-06"
MIN_DECIDED = 100
CLIP = 1e-4


def _prob(value):
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


def _scores(pairs):
    log_loss = brier = 0.0
    for p, y in pairs:
        q = min(max(p, CLIP), 1 - CLIP)
        log_loss -= y * math.log(q) + (1 - y) * math.log(1 - q)
        brier += (p - y) ** 2
    return log_loss / len(pairs), brier / len(pairs)


def _summary(rows):
    blend_ll, blend_brier = _scores([(b, y) for b, _, y, _ in rows])
    market_ll, market_brier = _scores([(m, y) for _, m, y, _ in rows])
    pinned = [(b, m, y, p) for b, m, y, p in rows if p is not None]
    pinnacle = None
    if pinned:
        pin_ll, pin_brier = _scores([(p, y) for _, _, y, p in pinned])
        same_market_ll, _ = _scores([(m, y) for _, m, y, _ in pinned])
        same_blend_ll, _ = _scores([(b, y) for b, _, y, _ in pinned])
        pinnacle = {"decided": len(pinned), "pinnacle_log_loss": pin_ll, "pinnacle_brier": pin_brier,
                    "market_log_loss_same_picks": same_market_ll, "blend_log_loss_same_picks": same_blend_ll}
    verdict = (f"TOO_FEW_DECIDED (need {MIN_DECIDED})" if len(rows) < MIN_DECIDED
               else "MARKET_AS_GOOD_OR_BETTER" if market_ll <= blend_ll else "BLEND_BETTER")
    return {"decided": len(rows), "blend_log_loss": blend_ll, "market_log_loss": market_ll,
            "blend_brier": blend_brier, "market_brier": market_brier,
            "log_loss_gain_vs_market": market_ll - blend_ll, "verdict": verdict, "pinnacle": pinnacle}


def build_from_rows(accepted, *, start=SHADOW_START):
    groups = defaultdict(list)
    for row in accepted:
        if str(row.get("game_start_utc") or "")[:10] < start or row.get("candidate_outcome") not in ("WIN", "LOSS"):
            continue
        blend = _prob(row.get("calibrated_probability"))
        market = _prob(row.get("fair_market_probability")) or _prob(row.get("market_probability"))
        side = SIDE.get(row.get("market_type"))
        if blend is None or market is None or side is None:
            continue
        item = (blend, market, 1 if row["candidate_outcome"] == "WIN" else 0,
                _prob(row.get("pinnacle_fair_probability")))
        groups[f"{row.get('sport')} {side}"].append(item)
        groups["ALL"].append(item)
    return {"schema": "blend-shadow-v1", "start": start, "min_decided": MIN_DECIDED, "weight_authority": False,
            "by_market": {key: _summary(rows) for key, rows in sorted(groups.items())}}


def build(database):
    from core.research_performance import accepted_rows, saved_rows
    accepted, _ = accepted_rows(saved_rows(database))
    return build_from_rows(accepted)


def markdown(report):
    lines = ["## Blend vs market price: forward shadow", "",
             f"Settled saved picks from {report['start']} on, each scored with the app's blend "
             "(calibrated probability) and with the market's no-vig price alone. Lower is better. "
             "A positive gain means the blend beat the market. Descriptive only; it changes no weight.", "",
             "| market | decided | log loss blend | log loss market | gain vs market | Brier blend | Brier market | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for key, s in report["by_market"].items():
        lines.append(f"| {key} | {s['decided']} | {s['blend_log_loss']:.4f} | {s['market_log_loss']:.4f} | "
                     f"{s['log_loss_gain_vs_market']:+.4f} | {s['blend_brier']:.4f} | {s['market_brier']:.4f} | "
                     f"{s['verdict']} |")
    if not report["by_market"]:
        lines.append(f"| | 0 | | | | | | no settled picks since {report['start']} yet |")
    pinned = {k: s["pinnacle"] for k, s in report["by_market"].items() if s.get("pinnacle")}
    if pinned:
        lines += ["", "On the picks that also have Pinnacle's no-vig price at the same line:", "",
                  "| market | decided | log loss Pinnacle | log loss market | log loss blend |", "|---|---|---|---|---|"]
        for key, s in pinned.items():
            lines.append(f"| {key} | {s['decided']} | {s['pinnacle_log_loss']:.4f} | "
                         f"{s['market_log_loss_same_picks']:.4f} | {s['blend_log_loss_same_picks']:.4f} |")
    return "\n".join(lines) + "\n"
