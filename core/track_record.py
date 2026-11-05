"""Paper track record of the app's saved best picks: hit rate, ROI and CLV by sport and market.

Built on exactly the filters of ``core.research_performance`` (verified pregame saved best
pick, latest per event, exact quote binding, provider-settled outcome). Adds what that
descriptive report lacks: units won at the recorded price, ROI per pick, a 95% interval
on the hit rate, the break-even rate at the average price, and closing-line value from
``closing_observations`` where a verified close exists. Descriptive only: it grants no
ranking or wager authority.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math

SIDE = {"spread_home": "SPREAD", "spread_away": "SPREAD", "total_over": "TOTAL", "total_under": "TOTAL"}


def _decimal(row):
    for key in ("decimal_odds",):
        try:
            value = float(row.get(key))
            if value > 1:
                return value
        except (TypeError, ValueError):
            pass
    try:
        american = float(row.get("odds_american"))
    except (TypeError, ValueError):
        return None
    if -100 < american < 100:
        return None
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def wilson(wins, decided, z=1.96):
    """95% Wilson interval for a hit rate (low, high), or (None, None)."""
    if not decided:
        return None, None
    p = wins / decided
    denominator = 1 + z * z / decided
    centre = (p + z * z / (2 * decided)) / denominator
    half = z * math.sqrt(p * (1 - p) / decided + z * z / (4 * decided * decided)) / denominator
    return centre - half, centre + half


def summarize(accepted, closes=None):
    """{"SPORT MARKET": stats} plus an "ALL" row. ``closes``: {candidate_id: beat_close bool}."""
    closes = closes or {}
    groups = defaultdict(list)
    for row in accepted:
        market = SIDE.get(row.get("market_type"))
        if market:
            groups[f"{row.get('sport')} {market}"].append(row)
            groups["ALL"].append(row)
    out = {}
    for key, rows in sorted(groups.items()):
        counts = Counter(r["candidate_outcome"] for r in rows)
        decided = counts["WIN"] + counts["LOSS"]
        units, priced, decimals = 0.0, 0, []
        for r in rows:
            d = _decimal(r)
            if d is None:
                continue
            priced += 1
            decimals.append(d)
            units += (d - 1) if r["candidate_outcome"] == "WIN" else (-1 if r["candidate_outcome"] == "LOSS" else 0)
        beat = [closes[r.get("candidate_id")] for r in rows if r.get("candidate_id") in closes
                and closes[r.get("candidate_id")] is not None]
        low, high = wilson(counts["WIN"], decided)
        mean_decimal = sum(decimals) / len(decimals) if decimals else None
        out[key] = {
            "picks": len(rows), "wins": counts["WIN"], "losses": counts["LOSS"], "pushes": counts["PUSH"],
            "hit_rate": counts["WIN"] / decided if decided else None,
            "hit_rate_95_low": low, "hit_rate_95_high": high,
            "break_even_hit_rate": 1 / mean_decimal if mean_decimal else None,
            "units": units if priced else None, "roi_per_pick": units / priced if priced else None,
            "with_close": len(beat), "beat_close_share": sum(beat) / len(beat) if beat else None}
    return out


def closes_by_candidate(database):
    """{candidate_id: beat_close} from verified closing observations."""
    from contextlib import closing
    from app_core.prediction_evidence import connect
    out = {}
    with closing(connect(database)) as db:
        for _key, _snapshot, candidate, payload in db.execute("SELECT * FROM closing_observations"):
            try:
                value = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if value.get("quote_verified") is True:
                out[candidate] = value.get("beat_close")
    return out


def build(database):
    from core.research_performance import accepted_rows, saved_rows
    accepted, rejected = accepted_rows(saved_rows(database))
    return {"schema": "paper-track-record-v1", "settled_picks": len(accepted), "exclusions": dict(rejected),
            "by_market": summarize(accepted, closes_by_candidate(database)),
            "ranking_authority": False, "wager_authority": False}


def markdown(report):
    def pct(v):
        return "" if v is None else f"{v:.1%}"
    lines = ["## Paper track record (saved best picks)", "",
             f"{report['settled_picks']} settled picks. Descriptive only; no wager authority.", "",
             "| market | picks | W-L-P | hit rate (95%) | break-even | units | ROI/pick | beat close |",
             "|---|---|---|---|---|---|---|---|"]
    for key, s in report["by_market"].items():
        interval = (f"{pct(s['hit_rate'])} ({pct(s['hit_rate_95_low'])}–{pct(s['hit_rate_95_high'])})"
                    if s["hit_rate"] is not None else "")
        units = "" if s["units"] is None else f"{s['units']:+.2f}"
        roi = "" if s["roi_per_pick"] is None else f"{s['roi_per_pick']:+.1%}"
        close = f"{pct(s['beat_close_share'])} of {s['with_close']}" if s["with_close"] else "no closes"
        lines.append(f"| {key} | {s['picks']} | {s['wins']}-{s['losses']}-{s['pushes']} | {interval} | "
                     f"{pct(s['break_even_hit_rate'])} | {units} | {roi} | {close} |")
    if report["exclusions"]:
        lines += ["", "Excluded rows: " + ", ".join(f"{k} {v}" for k, v in sorted(report["exclusions"].items()))]
    return "\n".join(lines) + "\n"
