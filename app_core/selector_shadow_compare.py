"""Shadow comparison of two pick rules on the same finished games (roadmap D7).

D7 (switch the selector to EV with No Bet) waits for a side-by-side comparison. This
replays every finished research game as of its first captured price (the same games as
``research_picks.replay``) and applies both rules to the same model and the same prices:

* **current**: today's research rule. The best-EV side and line at the best available
  price from the raw model probability; bet whenever EV >= 0.
* **ev_selector**: ``app_core.selection.ev_selector``. The market is the two-way
  no-vig price of the book offering the bet; the model only nudges it on the log-odds
  scale (weight 0.25, at most +/-5 points). Bet only when edge >= 2 points and EV > 0.
  Qualification is not applied here: this is a shadow test of the rule itself.

Each bet is graded on the replay-verified final score and against the **last captured
pregame consensus** for the same line and side (median no-vig across books). CLV is
close fair probability minus the entry break-even (1 / decimal). Captured last quotes
are not certified closes, so CLV here is a screen, not plan evidence. Research only.
"""
from __future__ import annotations

from statistics import median

from core.clv import no_vig_prob
from app_core import prospective_research_models as research
from app_core import research_picks as rp
from app_core.selection import ev_selector

RULES = ("current", "ev_selector")


def _latest_quotes(event, quotes, market, cutoff):
    """{home_line: {book: {selection: quote}}} keeping each book's newest quote before ``cutoff``."""
    ordered = sorted((q for q in quotes if q["quote_verified"] and q["line"] is not None
                      and rp._time(q["quote_timestamp"]) < cutoff),
                     key=lambda q: (rp._time(q["quote_timestamp"]), q["quote_id"]))
    out = {}
    for quote in ordered:
        out.setdefault(rp._home_line(event, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    return out


def close_fair(event, quotes, market, selection, line):
    """Median no-vig probability of ``selection`` at ``line`` from the last pregame quotes, or None."""
    first, second = rp._sides(event, market)
    start = rp._time(event["scheduled_start"])
    home_line = line if selection in (first, "Over") or market == "TOTAL" else -line
    books = _latest_quotes(event, quotes, market, start).get(round(home_line, 2), {})
    fair = [no_vig_prob(p[first]["american_odds"], p[second]["american_odds"])
            for p in books.values() if first in p and second in p]
    fair = [f for f in fair if f is not None]
    if not fair:
        return None
    p = median(fair)
    return p if selection == first else 1 - p


def ev_selector_pick(event, market, offers, mean, sd, policy=ev_selector.Policy()):
    """Highest-EV candidate under the EV selector, with its decision. None without a paired price."""
    first, second = rp._sides(event, market)
    best = None
    for line, books in offers.items():
        for side, other in ((first, second), (second, first)):
            paired = [p for p in books.values() if side in p and other in p]
            if not paired:
                continue
            book = max(paired, key=lambda p: (p[side]["decimal_odds"], p[side]["sportsbook"]))
            quote = book[side]
            market_p = ev_selector.fair_probability(quote["american_odds"], book[other]["american_odds"])
            if market_p is None:
                continue
            try:
                prob = research.score_probabilities(mean, sd, market, line,
                                                    selection="canonical" if side == first else "opposite")
            except ValueError:
                continue
            decided = prob["win"] + prob["loss"]
            if decided <= 0:
                continue
            model_p = prob["win"] / decided
            anchored = ev_selector.anchored_probability(market_p, model_p, policy)
            ev = (anchored * (quote["decimal_odds"] - 1) - (1 - anchored)) * decided
            row = {"event_id": event["event_id"], "selection": side, "line": quote["line"],
                   "sportsbook": quote["sportsbook"], "american_odds": quote["american_odds"],
                   "decimal_odds": quote["decimal_odds"], "market_probability": market_p,
                   "model_probability": model_p, "anchored_probability": anchored,
                   "edge": anchored - market_p, "ev": ev}
            if best is None or (ev, quote["decimal_odds"]) > (best["ev"], best["decimal_odds"]):
                best = row
    if best is not None:
        best["bet"] = best["edge"] >= policy.min_edge and best["ev"] > policy.min_ev
    return best


def compare(path, sport, market, *, now=None, method="score", policy=ev_selector.Policy()):
    """Per-game decisions of both rules, graded on results and against the last pregame consensus."""
    games = []
    for event, score, as_of, mean, sd, quotes in rp.replay_games(path, sport, market, now=now, method=method):
        offers = rp._offers(event, quotes, market, as_of)
        current = rp._pick(event, market, offers, mean, sd)
        if current is not None:
            current["bet"] = current["ev"] >= 0
        shadow = ev_selector_pick(event, market, offers, mean, sd, policy)
        game = {"event_id": event["event_id"], "scheduled_start": event["scheduled_start"]}
        for rule, pick in (("current", current), ("ev_selector", shadow)):
            if pick is None or not pick["bet"]:
                game[rule] = None
                continue
            fair = close_fair(event, quotes, market, pick["selection"], pick["line"])
            game[rule] = {"selection": pick["selection"], "line": pick["line"], "decimal_odds": pick["decimal_odds"],
                          "ev": pick["ev"], "outcome": rp.settle(sport, market, event, score, pick["selection"],
                                                                  pick["line"]),
                          "clv": None if fair is None else fair - 1 / pick["decimal_odds"]}
        games.append(game)
    return games


def summarize(games):
    """Per rule: the replayed record plus CLV; and how often the rules agree."""
    out = {"games": len(games)}
    for rule in RULES:
        bets = [g[rule] for g in games if g.get(rule)]
        clv = [b["clv"] for b in bets if b["clv"] is not None]
        out[rule] = dict(rp.record(bets), no_bet=len(games) - len(bets),
                         clv_measured=len(clv), mean_clv=sum(clv) / len(clv) if clv else None,
                         beat_close_share=sum(1 for c in clv if c > 0) / len(clv) if clv else None)
    both = [g for g in games if g.get("current") and g.get("ev_selector")]
    out["both_bet"] = len(both)
    out["same_side"] = sum(1 for g in both if (g["current"]["selection"], g["current"]["line"])
                           == (g["ev_selector"]["selection"], g["ev_selector"]["line"]))
    return out
