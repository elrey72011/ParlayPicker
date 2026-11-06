"""NCAAF Pinnacle value picks from the NCAAF capture records (D4 option B+, owner: "NCAAF picks this season").

NCAAF quotes do not flow through the canonical store the way NBA/NCAAB quotes do; the
NCAAF capture (``ncaaf_prospective``) saves, per game and capture, the US-book quotes
it scored (``models.*.candidates``) and Pinnacle's quotes as ``reference_quotes``. This
reads those records directly (``ncaaf_prospective_store``), no model:

* Spread: Pinnacle's no-vig price for the home side at home line ``p`` (home ``p`` /
  away ``-p``); total: Over / Under at the same point.
* Pick: the best US-book price beating that fair price by ``MIN_EV`` (2%), one per game.
* ``board``: upcoming games, from each game's latest capture.
* ``replay``: finished games, decided at the first capture with a qualifying price,
  graded on the CFBD final score in the store's ``scores`` records, with CLV against
  Pinnacle's no-vig price at the same line in the last capture before kickoff.

Research only: no stake, plan or wager authority. Captures happen a few times a day, so
the "close" is the last captured Pinnacle line, not the true close.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.clv import no_vig_prob
from app_core.ncaaf_history import timestamp
from app_core.pinnacle_value import MIN_EV
from app_core.research_api_budget import REFERENCE_BOOKS

METHOD = "pinnacle_value"
PAIRS = (("spread_home", "spread_away", "SPREAD"), ("total_over", "total_under", "TOTAL"))


def _decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def _price(q):
    p = q.get("price")
    return p if isinstance(p, (int, float)) and not isinstance(p, bool) and abs(p) >= 100 else None


def _key(market_type, point):
    """(family, home-perspective line) so both sides of one line share a key."""
    if point is None:
        return None
    point = float(point)
    if market_type == "spread_away":
        point = -point
    return ("SPREAD" if market_type.startswith("spread") else "TOTAL", round(point, 2))


def _us_quotes(event):
    seen, out = set(), []
    for model in (event.get("models") or {}).values():
        for q in model.get("candidates") or []:
            ident = (q.get("book"), q.get("market_type"), q.get("point"))
            if (ident in seen or str(q.get("book", "")).lower() in REFERENCE_BOOKS or _price(q) is None
                    or not str(q.get("market_type", "")).startswith(("spread", "total"))):
                continue
            seen.add(ident)
            out.append(q)
    return out


def _fair(reference, family, line):
    """{market_type: fair probability} from Pinnacle's two sides at one home-perspective line."""
    sides = {}
    for q in reference:
        if _key(q.get("market_type", ""), q.get("point")) == (family, line) and _price(q) is not None:
            sides[q["market_type"]] = q
    first, second = (("spread_home", "spread_away") if family == "SPREAD" else ("total_over", "total_under"))
    if first not in sides or second not in sides:
        return None
    p = no_vig_prob(sides[first]["price"], sides[second]["price"])
    return None if p is None else {first: p, second: 1 - p}


def pick(event, min_ev=MIN_EV):
    """Highest-EV US price beating Pinnacle's no-vig line by ``min_ev`` for one captured event, or None."""
    reference = event.get("reference_quotes") or []
    best = None
    for q in _us_quotes(event):
        key = _key(q["market_type"], q.get("point"))
        fair = key and _fair(reference, *key)
        if not fair:
            continue
        p = fair[q["market_type"]]
        decimal = _decimal(q["price"])
        ev = p * decimal - 1
        if ev < min_ev or (best and (ev, decimal) <= (best["ev"], best["decimal_odds"])):
            continue
        best = {"market_type": q["market_type"], "market_family": key[0], "home_line": key[1],
                "line": float(q["point"]), "sportsbook": q["book"], "american_odds": q["price"],
                "decimal_odds": decimal, "quote_timestamp": q.get("recorded_at"), "win_probability": p,
                "consensus_probability": p, "ev": ev, "method": METHOD}
    return best


def _captures(records):
    """[(captured_at, event)] for every saved capture event, oldest first."""
    out = []
    for r in records:
        if r.get("kind") == "capture":
            for event in r["data"].get("events", []):
                out.append((r["data"].get("captured_at") or r["created_at"], event))
    return sorted(out, key=lambda item: (item[0], str(item[1].get("event_id"))))


def board(records, *, now=None, min_ev=MIN_EV):
    now = now or datetime.now(timezone.utc)
    report = {"sport": "NCAAF", "market_family": "SPREAD+TOTAL", "method": METHOD, "model_id": None,
              "training_games": None, "blocker": None, "picks": [], "skipped": {}}
    latest = {}
    for captured, event in _captures(records):
        if timestamp(captured) <= now < timestamp(event["start"]):
            latest[event["event_id"]] = event
    if not any(e.get("reference_quotes") for e in latest.values()):
        report["blocker"] = "NO_PINNACLE_QUOTES_YET"
        return report
    for event in latest.values():
        chosen = pick(event, min_ev)
        if chosen:
            report["picks"].append(dict(chosen, sport="NCAAF", home_team=event["home"], away_team=event["away"],
                                        scheduled_start=event["start"],
                                        selection=event["home"] if chosen["market_type"] == "spread_home" else
                                        event["away"] if chosen["market_type"] == "spread_away" else
                                        "Over" if chosen["market_type"] == "total_over" else "Under"))
    report["picks"].sort(key=lambda p: -p["ev"])
    return report


def settle(market_type, line, home_score, away_score):
    if market_type == "spread_home":
        delta = home_score - away_score + line
    elif market_type == "spread_away":
        delta = away_score - home_score + line
    elif market_type == "total_over":
        delta = home_score + away_score - line
    else:
        delta = line - home_score - away_score
    return "WIN" if delta > 0 else "LOSS" if delta < 0 else "PUSH"


def replay(records, *, now=None, min_ev=MIN_EV):
    now = now or datetime.now(timezone.utc)
    scores = {}
    for r in records:
        if r.get("kind") == "scores":
            for s in r["data"].get("scores", []):
                scores.setdefault(str(s.get("cfbd_id")), s)
    by_event = {}
    for captured, event in _captures(records):
        by_event.setdefault(event["event_id"], []).append((captured, event))
    graded = []
    for snaps in by_event.values():
        first = snaps[0][1]
        score = scores.get(str(first.get("cfbd_id")))
        if (score is None or timestamp(first["start"]) > now
                or (score.get("home_id"), score.get("away_id")) != (first.get("home_id"), first.get("away_id"))):
            continue
        decided = None
        for captured, event in snaps:
            if timestamp(captured) >= timestamp(event["start"]):
                break
            decided = pick(event, min_ev)
            if decided:
                break
        if not decided:
            continue
        close = None
        for captured, event in snaps:
            if timestamp(captured) < timestamp(event["start"]):
                fair = _fair(event.get("reference_quotes") or [], decided["market_family"], decided["home_line"])
                close = fair[decided["market_type"]] if fair else close
        decided["clv"] = None if close is None else close - 1 / decided["decimal_odds"]
        decided["outcome"] = settle(decided["market_type"], decided["line"], score["home_score"], score["away_score"])
        graded.append(decided)
    return graded
