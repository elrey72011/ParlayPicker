"""Cross-book price screen: do the best available prices beat the market, and the close? (roadmap C5)

The research score models do not beat the line (``docs/evaluation/ncaaf-line-backtest-
2026-11-05.md``). The approach with the strongest record is price shopping: estimate the
fair probability **from the market itself**, then take only prices that beat it. This
screen measures that on captured evidence, without any model:

1. **Consensus.** For each event, market and line (spread lines in home perspective),
   every sportsbook with both sides posted is de-vigged (``core.clv.no_vig_prob``). The
   fair probability is the median across books; at least ``min_books`` are required.
2. **Flag.** For each side, the best price across books; EV = fair × decimal − 1.
   Flagged at EV ≥ threshold.
3. **Grade.** Against that book's certified close for the same selection and line (price
   CLV = entry decimal / close decimal − 1, positive means it beat the close), and against
   the final score (``prospective_evidence.score_settlement_outcome``).

Research only. Pushes on whole-number lines are ignored in EV, which overstates EV
slightly on those lines. Beating the close is the leading indicator. ROI needs hundreds
of settled picks before it means anything.
"""
from __future__ import annotations

from statistics import median

from core.clv import no_vig_prob
from app_core import prospective_evidence as evidence

THRESHOLDS = (0.0, 0.02, 0.04)
# How long before the start the best price was quoted (roadmap M5: outcomes by book and timing).
TIMING = ((0, 1, "under 1 h"), (1, 3, "1-3 h"), (3, 6, "3-6 h"), (6, 24, "6-24 h"), (24, None, "24 h or more"))
REFERENCE_BOOK = "pinnacle"  # D4: sharp reference, never a bettable price or a consensus member


def _key(event, quote):
    """(event, market, home-perspective line) so both sides of one line share a key."""
    line = quote["line"]
    if quote["market_family"] != "TOTAL" and quote["selection"] != event["home_team"]:
        line = -line
    return quote["event_id"], quote["market_family"], round(float(line), 2)


def _sides(event, market):
    return (event["home_team"], event["away_team"]) if market != "TOTAL" else ("Over", "Under")


def _order(quote):
    return str(quote.get("quote_timestamp") or ""), str(quote.get("quote_id") or "")


def screen(events, quotes, closes, scores, *, min_books=3):
    """Flagged candidates (one per event, market, line and side) with their grades."""
    events = {e["event_id"]: e for e in events}
    books = {}
    for quote in quotes:
        event = events.get(quote["event_id"])
        if event is None or not quote.get("quote_verified") or quote.get("line") is None:
            continue
        sides = books.setdefault(_key(event, quote), {}).setdefault(quote["sportsbook"], {})
        held = sides.get(quote["selection"])
        if held is None or _order(quote) < _order(held):
            sides[quote["selection"]] = quote  # each book's first quote at this line (later ones are C6 decision quotes)
    close_by = {(c["event_id"], c["market_family"], c["selection"], c["sportsbook"]): c
                for c in closes if c.get("close_verified")}
    score_by = {s["event_id"]: s for s in scores}
    candidates = []
    for (event_id, market, line), by_book in sorted(books.items()):
        event = events[event_id]
        first, second = _sides(event, market)
        reference = by_book.pop(REFERENCE_BOOK, None)
        fair = [no_vig_prob(pair[first]["american_odds"], pair[second]["american_odds"])
                for pair in by_book.values() if first in pair and second in pair]
        fair = [p for p in fair if p is not None]
        if len(fair) < min_books:
            continue
        consensus = {first: median(fair)}
        consensus[second] = 1 - consensus[first]
        sharp = None
        if reference and first in reference and second in reference:
            sharp = no_vig_prob(reference[first]["american_odds"], reference[second]["american_odds"])
        for side in (first, second):
            offers = [pair[side] for pair in by_book.values() if side in pair]
            if not offers:
                continue
            best = max(offers, key=lambda q: (q["decimal_odds"], q["sportsbook"]))
            row = {"event_id": event_id, "sport": event["sport"], "market_family": market, "selection": side,
                   "line": best["line"], "sportsbook": best["sportsbook"], "decimal_odds": best["decimal_odds"],
                   "fair_probability": consensus[side], "books": len(fair),
                   "ev": consensus[side] * best["decimal_odds"] - 1}
            lead = _hours_before_start(event, best)
            if lead is not None:
                row["hours_before_start"] = lead
            if sharp is not None:
                row["pinnacle_fair"] = sharp if side == first else 1 - sharp
                row["ev_vs_pinnacle"] = row["pinnacle_fair"] * best["decimal_odds"] - 1
            close = close_by.get((event_id, market, side, best["sportsbook"]))
            if close is not None and close["line"] == best["line"]:
                row["price_clv"] = best["decimal_odds"] / close["decimal_odds"] - 1
            score = score_by.get(event_id)
            if score is not None:
                row["outcome"] = evidence.score_settlement_outcome(
                    event["sport"], market, side, event["home_team"], event["away_team"],
                    score["home_score"], score["away_score"], best["line"])
            candidates.append(row)
    return candidates


def _hours_before_start(event, quote):
    try:
        start = evidence._time(event["scheduled_start"], "scheduled_start")
        return (start - evidence._time(quote["quote_timestamp"], "quote_timestamp")).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return None


def timing_label(hours):
    if hours is None or hours < 0:
        return None
    return next(label for low, high, label in TIMING if hours >= low and (high is None or hours < high))


def _mean_se(values):
    """(mean, standard error); the error is None with fewer than 2 values."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, (variance / len(values)) ** 0.5


def group(picks):
    """One breakdown row: EV, closing-line value and results for ``picks``, with standard errors."""
    clv = [c["price_clv"] for c in picks if "price_clv" in c]
    settled = [c for c in picks if c.get("outcome") in ("WIN", "LOSS", "PUSH")]
    returns = [c["decimal_odds"] - 1 if c["outcome"] == "WIN" else (-1.0 if c["outcome"] == "LOSS" else 0.0)
               for c in settled]
    mean_clv, clv_se = _mean_se(clv)
    roi, roi_se = _mean_se(returns)
    return {"picks": len(picks), "mean_ev": sum(c["ev"] for c in picks) / len(picks) if picks else None,
            "with_comparable_close": len(clv), "mean_price_clv": mean_clv, "price_clv_se": clv_se,
            "beat_close_share": sum(1 for v in clv if v > 0) / len(clv) if clv else None,
            "settled": len(settled), "wins": sum(1 for c in settled if c["outcome"] == "WIN"),
            "roi_per_pick": roi, "roi_se": roi_se}


def breakdown(candidates):
    """The best-price candidates split by the book that had the best price, and by how long before
    the start it was quoted. Every candidate counts (no EV threshold): the question is where the
    best available price beats the close, not which picks a threshold would have kept."""
    by_book, by_timing = {}, {}
    for c in candidates:
        by_book.setdefault(c["sportsbook"], []).append(c)
        label = timing_label(c.get("hours_before_start"))
        if label is not None:
            by_timing.setdefault(label, []).append(c)
    return {"by_book": {book: group(picks) for book, picks in sorted(by_book.items())},
            "by_timing": {label: group(by_timing[label]) for _, _, label in TIMING if label in by_timing}}


def summarize(candidates, thresholds=THRESHOLDS, ev_key="ev"):
    """By EV threshold, plus ``all``: the best price on every candidate, as a baseline.

    Beating a no-vig consensus needs a price at or above fair odds (+100 at 50%), which
    is rare, so ``all`` shows whether the best available price beats the close anyway.
    """
    out = {}
    for threshold in (None,) + tuple(thresholds):
        picks = [c for c in candidates if ev_key in c and (threshold is None or c[ev_key] >= threshold)]
        clv = [c["price_clv"] for c in picks if "price_clv" in c]
        settled = [c for c in picks if c.get("outcome") in ("WIN", "LOSS", "PUSH")]
        units = sum(c["decimal_odds"] - 1 if c["outcome"] == "WIN" else (-1 if c["outcome"] == "LOSS" else 0)
                    for c in settled)
        out["all" if threshold is None else f"{threshold:.2f}"] = {
            "picks": len(picks),
            "mean_ev": sum(c[ev_key] for c in picks) / len(picks) if picks else None,
            "with_comparable_close": len(clv),
            "mean_price_clv": sum(clv) / len(clv) if clv else None,
            "beat_close_share": sum(1 for v in clv if v > 0) / len(clv) if clv else None,
            "settled": len(settled),
            "wins": sum(1 for c in settled if c["outcome"] == "WIN"),
            "roi_per_pick": units / len(settled) if settled else None}
    return out


def from_store(path, sport, market, *, min_books=3):
    """Read one sport and market from the canonical store and summarize it."""
    events = evidence.read_records(path, "prospective_event", sport=sport)
    quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=market)
    closes = evidence.read_records(path, "prospective_close", sport=sport, market_family=market)
    by_id = {e["event_id"]: e for e in events}
    scores = [r for r in evidence.read_records(path, "prospective_result", sport=sport)
              if r["market_family"] is None and r["home_score"] is not None and r["away_score"] is not None
              and r["event_id"] in by_id and evidence.verify_provider_score(by_id[r["event_id"]], r)]
    candidates = screen(events, quotes, closes, scores, min_books=min_books)
    return {"sport": sport, "market_family": market, "candidates": len(candidates),
            "by_threshold": summarize(candidates),
            "with_pinnacle": sum(1 for c in candidates if "ev_vs_pinnacle" in c),
            "by_threshold_vs_pinnacle": summarize(candidates, ev_key="ev_vs_pinnacle"),
            **breakdown(candidates)}
