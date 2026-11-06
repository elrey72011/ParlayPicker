"""Pinnacle value picks: US-book prices that beat Pinnacle's no-vig line (D4 option A).

No model. For each game, line and side, the fair probability is Pinnacle's two-way
no-vig price at the same line (``core.clv.no_vig_prob``), using each book's newest
verified quote at the moment of the decision. The pick is the best US-book price
(Pinnacle is reference-only, ``REFERENCE_BOOKS``) whose EV against that fair price is at
least ``MIN_EV``; one pick per game, the highest EV.

* ``board``: upcoming games, as of now.
* ``replay``: finished games. Each capture moment is checked in time order and the game
  is decided at the first one where a qualifying price existed (as a bettor watching
  the market would), using only quotes seen by then. Graded on the replay-verified final score and on CLV against
  Pinnacle's last pregame no-vig price at the same line (the standard sharp-close
  benchmark): CLV = close fair probability - 1 / entry decimal.

Research only: no stake, no plan, no wager authority. Pinnacle odds come from its public
site via the Odds API and may be delayed, so an apparent edge can be a stale reference.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.clv import no_vig_prob
from app_core import prospective_evidence as evidence
from app_core import prospective_research_models as research
from app_core import research_picks as rp
from app_core.research_api_budget import PINNACLE_SPORTS, REFERENCE_BOOKS

MIN_EV = 0.02
METHOD = "pinnacle_value"
_EPSILON = timedelta(microseconds=1)


def _book_quotes(event, quotes, cutoff):
    """{home_line: {book: {selection: quote}}}: each book's newest verified quote before ``cutoff``."""
    ordered = sorted((q for q in quotes if q["event_id"] == event["event_id"] and q["quote_verified"]
                      and q["line"] is not None and rp._time(q["quote_timestamp"]) < cutoff),
                     key=lambda q: (rp._time(q["quote_timestamp"]), q["quote_id"]))
    out = {}
    for quote in ordered:
        out.setdefault(rp._home_line(event, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    return out


def _fair(books, first, second, side):
    pair = next((books[b] for b in REFERENCE_BOOKS if b in books), None)
    if not pair or first not in pair or second not in pair:
        return None
    p = no_vig_prob(pair[first]["american_odds"], pair[second]["american_odds"])
    return None if p is None else (p if side == first else 1 - p)


def pick(event, market, by_line, min_ev=MIN_EV):
    """The highest-EV US price beating Pinnacle's no-vig line by ``min_ev``, or None."""
    first, second = rp._sides(event, market)
    best = None
    for line, books in by_line.items():
        for side in (first, second):
            fair = _fair(books, first, second, side)
            prices = [p[side] for b, p in books.items() if b not in REFERENCE_BOOKS and side in p]
            if fair is None or not prices:
                continue
            quote = max(prices, key=lambda q: (q["decimal_odds"], q["sportsbook"]))
            ev = fair * quote["decimal_odds"] - 1
            if ev < min_ev or (best and (ev, quote["decimal_odds"]) <= (best["ev"], best["decimal_odds"])):
                continue
            best = {"event_id": event["event_id"], "selection": side, "line": quote["line"],
                    "home_line": line, "sportsbook": quote["sportsbook"], "american_odds": quote["american_odds"],
                    "decimal_odds": quote["decimal_odds"], "quote_id": quote["quote_id"],
                    "quote_timestamp": quote["quote_timestamp"], "win_probability": fair, "ev": ev,
                    "consensus_probability": fair, "method": METHOD}
    return best


def board(path, sport, market, *, now=None, min_ev=MIN_EV):
    now = now or datetime.now(timezone.utc)
    report = {"sport": sport, "market_family": market, "method": METHOD, "model_id": None,
              "training_games": None, "blocker": None, "picks": [], "skipped": {}}
    if sport not in PINNACLE_SPORTS:
        report["blocker"] = "PINNACLE_NOT_IN_SCOPE"
        return report
    events, quotes, _ = research._read_scope(path, sport, market)
    if not any(q["sportsbook"] in REFERENCE_BOOKS for q in quotes):
        report["blocker"] = "NO_PINNACLE_QUOTES_YET"
        return report
    for event in sorted(events.values(), key=lambda e: (e["scheduled_start"], e["event_id"])):
        if rp._time(event["scheduled_start"]) <= now:
            continue
        chosen = pick(event, market, _book_quotes(event, quotes, now), min_ev)
        if chosen:
            report["picks"].append(dict(chosen, sport=sport, market_family=market, home_team=event["home_team"],
                                        away_team=event["away_team"], scheduled_start=event["scheduled_start"]))
    report["picks"].sort(key=lambda p: -p["ev"])
    return report


def replay(path, sport, market, *, now=None, min_ev=MIN_EV):
    """Graded picks on finished games, each decided at the first capture moment with a qualifying price."""
    now = now or datetime.now(timezone.utc)
    if sport not in PINNACLE_SPORTS or sport not in evidence.SCORE_SETTLEMENT_SPORTS:
        return []
    events, quotes, scores = research._read_scope(path, sport, market)
    latest = {}
    for score in scores:
        if evidence.verify_provider_score(events[score["event_id"]], score):
            prev = latest.get(score["event_id"])
            if prev is None or score["grading_version"] > prev["grading_version"]:
                latest[score["event_id"]] = score
    graded = []
    for event_id, score in latest.items():
        event = events[event_id]
        start = rp._time(event["scheduled_start"])
        if start > now:
            continue
        own = [q for q in quotes if q["event_id"] == event_id and q["quote_verified"]
               and rp._time(q["quote_timestamp"]) < start]
        decided = None
        for stamp in sorted({rp._time(q["quote_timestamp"]) for q in own}):
            decided = pick(event, market, _book_quotes(event, own, stamp + _EPSILON), min_ev)
            if decided:
                break
        if not decided:
            continue
        first, second = rp._sides(event, market)
        close_books = _book_quotes(event, own, start).get(decided["home_line"], {})
        close = _fair(close_books, first, second, decided["selection"])
        decided["clv"] = None if close is None else close - 1 / decided["decimal_odds"]
        decided["outcome"] = rp.settle(sport, market, event, score, decided["selection"], decided["line"])
        graded.append(decided)
    return graded


def record(graded):
    """``research_picks.record`` plus CLV against Pinnacle's close."""
    clv = [g["clv"] for g in graded if g.get("clv") is not None]
    return dict(rp.record(graded), clv_measured=len(clv), mean_clv=sum(clv) / len(clv) if clv else None,
                beat_close_share=sum(1 for c in clv if c > 0) / len(clv) if clv else None)

