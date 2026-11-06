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
from app_core.research_api_budget import REFERENCE_BOOKS

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
                or _time(quote["quote_timestamp"]) > cutoff or quote["sportsbook"] in REFERENCE_BOOKS):
            continue
        out.setdefault(_home_line(event, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    return out


def _pick(event, market, offers, mean, sd):
    """Best-EV (line, side) at the best available price for a score distribution N(mean, sd), or None."""
    first, second = _sides(event, market)
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
                prob = research.score_probabilities(mean, sd, market, line,
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


ANCHOR_MIN_GAMES = 60
ANCHOR_RIDGE = 10.0


def anchored_fit(events, quotes, scores, market, as_of, *, min_games=None):
    """Line-anchored ridge on this season's games finished by ``as_of``.

    Target: the home cover margin (margin + home line) or the total's error (points -
    line), with the line as an extra input. None until ``min_games`` games exist.
    """
    min_games = ANCHOR_MIN_GAMES if min_games is None else min_games
    rows = []
    for example in research._examples(events, quotes, scores, market, as_of):
        line = float(example["line"])
        target = example["target"] + line if market != "TOTAL" else example["target"] - line
        rows.append({"features": list(example["features"]) + [line], "target": target})
    if len(rows) < min_games:
        return None
    fit = research._fit(rows, ANCHOR_RIDGE)
    fit["training_games"] = len(rows)
    return fit


def anchored_mean(fit, features, line, market):
    """Expected margin (spread) or total implied by the anchored prediction at ``line``."""
    predicted = research._expected(fit, list(features) + [float(line)])
    return predicted - float(line) if market != "TOTAL" else predicted + float(line)


def _reference_line(events, quotes, event_id, when):
    quote = research._reference_quotes(events, quotes, when).get(event_id)
    return None if quote is None else quote["line"]


def _model(path, sport, market, when):
    try:
        model, artifact = research._latest_model(path, sport, market, when)
    except ValueError as exc:
        return None, None, str(exc)
    return model, artifact, None if model else "NO_MODEL_YET"


def _mean(fit, features, method, market, events, quotes, event_id, when):
    if method != "anchored":
        return research._expected(fit, features)
    line = _reference_line(events, quotes, event_id, when)
    return None if line is None else anchored_mean(fit, features, line, market)


def board(path, sport, market, *, now=None, method="score"):
    """Picks for upcoming games: one per game, the best-EV side and line at the best price.

    ``method``: "score" uses the stored research score model; "anchored" fits the
    line-anchored model on this season's games finished by ``now``.
    """
    now = now or datetime.now(timezone.utc)
    events, quotes, scores = research._read_scope(path, sport, market)
    if method == "anchored":
        fit = anchored_fit(events, quotes, scores, market, now)
        report = {"sport": sport, "market_family": market, "method": method, "model_id": None,
                  "training_games": fit and fit["training_games"],
                  "blocker": None if fit else f"ANCHORED_NEEDS_{ANCHOR_MIN_GAMES}_GAMES", "picks": [], "skipped": {}}
    else:
        model, artifact, blocker = _model(path, sport, market, now)
        fit = artifact and artifact["model"]
        report = {"sport": sport, "market_family": market, "method": method, "model_id": model and model["model_id"],
                  "training_games": model and model["independent_event_count"], "blocker": blocker,
                  "picks": [], "skipped": {}}
    if fit is None:
        return report
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
        mean = _mean(fit, built[0], method, market, events, quotes, event["event_id"], now)
        pick = None if mean is None else _pick(event, market, offers, mean, fit["residual_sd"])
        if pick:
            report["picks"].append(dict(pick, sport=sport, market_family=market, method=method,
                                        home_team=event["home_team"],
                                        away_team=event["away_team"], scheduled_start=event["scheduled_start"]))
    report["picks"].sort(key=lambda p: -p["ev"])
    return report


def replay_games(path, sport, market, *, now=None, method="score"):
    """Finished games as of each one's first captured price: (event, score, as_of, mean, sd, quotes).

    Only the model available at ``as_of`` is used. Shared by ``replay`` and the selector
    shadow comparison so both grade exactly the same games.
    """
    now = now or datetime.now(timezone.utc)
    if sport not in evidence.SCORE_SETTLEMENT_SPORTS:
        return  # NHL grades need the per-game official-score check (D17), not done offline
    events, quotes, scores = research._read_scope(path, sport, market)
    latest = {}
    for score in scores:
        if evidence.verify_provider_score(events[score["event_id"]], score):
            previous = latest.get(score["event_id"])
            if previous is None or score["grading_version"] > previous["grading_version"]:
                latest[score["event_id"]] = score
    for event_id, score in latest.items():
        event = events[event_id]
        start = _time(event["scheduled_start"])
        stamps = [_time(q["quote_timestamp"]) for q in quotes
                  if q["event_id"] == event_id and q["quote_verified"] and _time(q["quote_timestamp"]) < start]
        if not stamps or start > now:
            continue
        as_of = min(stamps)
        if method == "anchored":
            fit = anchored_fit(events, quotes, scores, market, as_of)
        else:
            model, artifact, _ = _model(path, sport, market, as_of)
            fit = artifact and artifact["model"]
        if fit is None:
            continue
        built = research._features(event, as_of, events, scores, market)
        if built is None:
            continue
        mean = _mean(fit, built[0], method, market, events, quotes, event_id, as_of)
        if mean is None:
            continue
        yield event, score, as_of, mean, fit["residual_sd"], [q for q in quotes if q["event_id"] == event_id]


def settle(sport, market, event, score, selection, line):
    return evidence.score_settlement_outcome(sport, market, selection, event["home_team"], event["away_team"],
                                             score["home_score"], score["away_score"], line)


def replay(path, sport, market, *, now=None, min_ev=0.0, method="score"):
    """Grade the board rule on finished games, as of each game's first captured price."""
    graded = []
    for event, score, as_of, mean, sd, quotes in replay_games(path, sport, market, now=now, method=method):
        pick = _pick(event, market, _offers(event, quotes, market, as_of), mean, sd)
        if pick is None or pick["ev"] < min_ev:
            continue
        pick["outcome"] = settle(sport, market, event, score, pick["selection"], pick["line"])
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
