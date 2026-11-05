"""Research picks for NBA, NCAAB and NHL from the in-season score models (roadmap C5).

The app's pick pipeline has no spread/total model for these sports (its ML model targets
the home win and is blocked for spreads/totals). The research cycle trains in-season score
models (``prospective_research_models``) on this season's captured games. This turns them
into picks:

* For each upcoming game, the newest model available now, scored at every captured line
  and side (``score_probabilities``; "canonical" is home/Over, "opposite" is away/Under).
* The pick is the side and line with the best expected value at the **best available
  price** across books: EV = P(win) x (decimal - 1) - P(loss); a push returns the stake.
* Alongside it, the cross-book no-vig consensus (``price_edge_screen``) when 3+ books post
  both sides, so a pick can be read against the market.

``replay`` grades the same rule on finished games without any stored board: it recomputes
each game's pick **as of its first captured price**, using only the model available then,
and settles it on the replay-verified provider final score. Research only: no stake, no
plan, no wager authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
from statistics import median

from core.clv import no_vig_prob
from app_core import prospective_evidence as evidence
from app_core import prospective_research_models as research

SPORTS = ("NBA", "NCAAB", "NHL")


def _time(value):
    return research._time(value)


def _home_line(event, quote):
    line = float(quote["line"])
    if quote["market_family"] != "TOTAL" and quote["selection"] != event["home_team"]:
        line = -line
    return round(line, 2)


def _sides(event, market):
    return (event["home_team"], event["away_team"]) if market != "TOTAL" else ("Over", "Under")


def _offers(event, quotes, market, cutoff):
    """{home_line: {sportsbook: {selection: quote}}} for verified quotes observed by ``cutoff``."""
    out = {}
    for quote in quotes:
        if (quote["event_id"] != event["event_id"] or not quote["quote_verified"] or quote["line"] is None
                or _time(quote["quote_timestamp"]) > cutoff):
            continue
        out.setdefault(_home_line(event, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    return out


def _pick(event, market, offers, fit, features):
    """Best-EV (line, side) at the best available price, or None."""
    first, second = _sides(event, market)
    mean = research._expected(fit, features)
    best = None
    for line, books in offers.items():
        fair = [no_vig_prob(p[first]["american_odds"], p[second]["american_odds"])
                for p in books.values() if first in p and second in p]
        fair = [f for f in fair if f is not None]
        consensus = median(fair) if len(fair) >= 3 else None
        for side in (first, second):
            prices = [p[side] for p in books.values() if side in p]
            if not prices:
                continue
            quote = max(prices, key=lambda q: (q["decimal_odds"], q["sportsbook"]))
            try:
                prob = research.score_probabilities(mean, fit["residual_sd"], market, line,
                                                    selection="canonical" if side == first else "opposite")
            except ValueError:  # quarter lines need a split-stake rule
                continue
            ev = prob["win"] * (quote["decimal_odds"] - 1) - prob["loss"]
            row = {"event_id": event["event_id"], "selection": side, "line": quote["line"],
                   "sportsbook": quote["sportsbook"], "american_odds": quote["american_odds"],
                   "decimal_odds": quote["decimal_odds"], "quote_id": quote["quote_id"],
                   "win_probability": prob["win"], "push_probability": prob["push"],
                   "loss_probability": prob["loss"], "ev": ev, "books": len(books),
                   "consensus_probability": None if consensus is None else
                   (consensus if side == first else 1 - consensus),
                   "model_center": mean}
            if best is None or (ev, quote["decimal_odds"]) > (best["ev"], best["decimal_odds"]):
                best = row
    return best


def _model(path, sport, market, when):
    try:
        model, artifact = research._latest_model(path, sport, market, when)
    except ValueError as exc:
        return None, None, str(exc)
    return model, artifact, None if model else "NO_MODEL_YET"


def board(path, sport, market, *, now=None):
    """Picks for upcoming games: one per game, the best-EV side and line at the best price."""
    now = now or datetime.now(timezone.utc)
    model, artifact, blocker = _model(path, sport, market, now)
    report = {"sport": sport, "market_family": market, "model_id": model and model["model_id"],
              "training_games": model and model["independent_event_count"], "blocker": blocker,
              "picks": [], "skipped": {}}
    if model is None:
        return report
    fit = artifact["model"]
    events, quotes, scores = research._read_scope(path, sport, market)
    for event in sorted(events.values(), key=lambda e: (e["scheduled_start"], e["event_id"])):
        if _time(event["scheduled_start"]) <= now:
            continue
        offers = _offers(event, quotes, market, now)
        if not offers:
            continue
        built = research._features(event, now, events, scores, market)
        if built is None:
            report["skipped"]["insufficient_team_history"] = report["skipped"].get("insufficient_team_history", 0) + 1
            continue
        pick = _pick(event, market, offers, fit, built[0])
        if pick:
            report["picks"].append(dict(pick, sport=sport, market_family=market, home_team=event["home_team"],
                                        away_team=event["away_team"], scheduled_start=event["scheduled_start"]))
    report["picks"].sort(key=lambda p: -p["ev"])
    return report


def replay(path, sport, market, *, now=None, min_ev=0.0):
    """Grade the board rule on finished games, as of each game's first captured price."""
    now = now or datetime.now(timezone.utc)
    if sport not in evidence.SCORE_SETTLEMENT_SPORTS:
        return []  # NHL grades need the per-game official-score check (D17), not done offline
    events, quotes, scores = research._read_scope(path, sport, market)
    latest = {}
    for score in scores:
        if evidence.verify_provider_score(events[score["event_id"]], score):
            previous = latest.get(score["event_id"])
            if previous is None or score["grading_version"] > previous["grading_version"]:
                latest[score["event_id"]] = score
    graded = []
    for event_id, score in latest.items():
        event = events[event_id]
        start = _time(event["scheduled_start"])
        stamps = [_time(q["quote_timestamp"]) for q in quotes
                  if q["event_id"] == event_id and q["quote_verified"] and _time(q["quote_timestamp"]) < start]
        if not stamps or start > now:
            continue
        as_of = min(stamps)
        model, artifact, _ = _model(path, sport, market, as_of)
        if model is None:
            continue
        built = research._features(event, as_of, events, scores, market)
        if built is None:
            continue
        pick = _pick(event, market, _offers(event, quotes, market, as_of), artifact["model"], built[0])
        if pick is None or pick["ev"] < min_ev:
            continue
        pick["outcome"] = evidence.score_settlement_outcome(
            sport, market, pick["selection"], event["home_team"], event["away_team"],
            score["home_score"], score["away_score"], pick["line"])
        graded.append(pick)
    return graded


def record(graded):
    """W-L-P, hit rate with a 95% interval, units and ROI per pick for graded picks."""
    from core.track_record import wilson
    wins = sum(1 for g in graded if g["outcome"] == "WIN")
    losses = sum(1 for g in graded if g["outcome"] == "LOSS")
    pushes = sum(1 for g in graded if g["outcome"] == "PUSH")
    units = sum(g["decimal_odds"] - 1 if g["outcome"] == "WIN" else (-1 if g["outcome"] == "LOSS" else 0)
                for g in graded)
    low, high = wilson(wins, wins + losses)
    return {"picks": len(graded), "wins": wins, "losses": losses, "pushes": pushes,
            "hit_rate": wins / (wins + losses) if wins + losses else None,
            "hit_rate_95_low": low, "hit_rate_95_high": high,
            "units": units, "roi_per_pick": units / len(graded) if graded else None}
