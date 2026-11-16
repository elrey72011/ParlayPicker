"""Two notes beside each candidate: the best price at its line, and whether it tied (owner 2026-11-15).

Display and audit only. Nothing reads them back: they change no pick, probability, gate or stake.

**Best price at the line.** A pick takes its price from the first book with a consistent line,
Novig first. In five app exports (10 to 15 November 2026, 342 picks) another fetched book had a
better price for the same side at the same line on 87 picks (25%), worth a median 1.4 points of
expected value. ``prices`` names the best such price and the book(s) offering it.

**Ties.** Candidates are ranked by win probability and an exact tie is broken by market name, so
a spread beats a total and the away spread beats the home spread. 36 of those 342 picks were such
ties. ``ties`` marks them, so a tie-break is not read as a lean.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import pandas as pd

from app_core.public_quote_policy import canonical_book_label

STAKE_BOOKS = ("novig", "fanduel", "draftkings", "betmgm")   # Pinnacle is a reference, never an entry
MAX_AMERICAN = 10000
MAX_QUOTE_LAG_MINUTES = 20      # a quote this much older than the newest one at the line is left out
TIE_TOLERANCE = 1e-9
PRICE_FIELDS = ("best_line_price", "best_line_price_book", "best_line_price_books")
TIE_FIELDS = ("best_available_tie_count", "best_available_tied_with", "best_available_no_lean")
NO_LEAN = "No lean (tie)"


def decimal(american):
    """Decimal odds for a real American price (100 to 10000 either way), else None."""
    american = _number(american)
    if american is None or not 100 <= abs(american) <= MAX_AMERICAN:
        return None
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo is not None else None


def _parse(value):
    if isinstance(value, list):
        return value
    try:
        quotes = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return quotes if isinstance(quotes, list) else []


def best_price(row, quotes=None):
    """``PRICE_FIELDS`` for one candidate: the best price for its own side at its exact line.

    ``quotes`` is the row's parsed ``provider_quotes`` (parsed here when not given). A quote counts
    when it is for the row's market type, at the row's point, from a stake book, with a real
    American price and a provider update time. Quotes more than ``MAX_QUOTE_LAG_MINUTES`` older
    than the newest of those are left out as stale. ``best_line_price_book`` lists every book at
    the best price and ``best_line_price_books`` counts the stake books that quoted the side at
    that line. A book quoting the line twice counts once, at its newest (then lower) price.
    """
    empty = {"best_line_price": float("nan"), "best_line_price_book": "", "best_line_price_books": 0}
    kind = str(row.get("market_type") or "")
    moneyline = kind.startswith("moneyline")
    line = None if moneyline else _number(row.get("total_line" if kind.startswith("total") else "spread_line"))
    if not kind or (line is None and not moneyline):
        return empty
    offers = {}
    for quote in _parse(row.get("provider_quotes")) if quotes is None else quotes:
        if not isinstance(quote, dict) or quote.get("market_type") != kind:
            continue
        book = str(quote.get("book") or "").strip().casefold()
        book = "novig" if book.startswith("novig") else book
        price, stamp = decimal(quote.get("price")), _stamp(quote.get("recorded_at"))
        if book not in STAKE_BOOKS or price is None or stamp is None:
            continue
        if not moneyline:
            point = _number(quote.get("point"))
            if point is None or abs(point - line) >= 1e-8:
                continue
        held = offers.get(book)
        if held is None or (stamp, -price) > (held[0], -held[1]):
            offers[book] = (stamp, price, _number(quote.get("price")))
    if not offers:
        return empty
    newest = max(offer[0] for offer in offers.values())
    fresh = {book: offer for book, offer in offers.items()
             if newest - offer[0] <= timedelta(minutes=MAX_QUOTE_LAG_MINUTES)}
    best = max(offer[1] for offer in fresh.values())
    at_best = [book for book in STAKE_BOOKS if book in fresh and fresh[book][1] == best]
    return {"best_line_price": fresh[at_best[0]][2],
            "best_line_price_book": ", ".join(canonical_book_label(book) for book in at_best),
            "best_line_price_books": len(fresh)}


def prices(frame):
    """``PRICE_FIELDS`` for every row of ``frame``, on its index; each game's quotes are parsed once."""
    needed = frame.reindex(columns=["market_type", "spread_line", "total_line", "provider_quotes"])
    parsed, rows = {}, []
    for record in needed.to_dict("records"):
        text = record["provider_quotes"]
        key = text if isinstance(text, str) else ""
        if key not in parsed:
            parsed[key] = _parse(text)
        rows.append(best_price(record, parsed[key]))
    out = pd.DataFrame(rows, columns=list(PRICE_FIELDS))
    out.index = frame.index
    out["best_line_price"] = pd.to_numeric(out["best_line_price"], errors="coerce").astype("float64")
    out["best_line_price_book"] = out["best_line_price_book"].fillna("").astype(object)
    out["best_line_price_books"] = pd.to_numeric(out["best_line_price_books"], errors="coerce").fillna(0).astype("int64")
    return out


def _family(market_type):
    return market_type.split("_")[0]


def _level(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TIE_TOLERANCE


def ties(frame):
    """``TIE_FIELDS`` for every row of ``frame``, on its index.

    Among a game's ranked candidates (``best_available_rank`` set), two are tied when their
    ``best_available_probability`` is within ``TIE_TOLERANCE``, or when neither has one.
    ``best_available_tie_count`` is the number sharing the row's probability, itself included;
    ``best_available_tied_with`` names the others in rank order; ``best_available_no_lean`` is
    True when one of them is the other side of the row's own market, so the row is a coin flip
    there and its place in the ranking is the tie-break's. Unranked rows get no count.
    """
    size = len(frame)
    count, tied_with, no_lean = [pd.NA] * size, [""] * size, [False] * size
    needed = ("matchup_id", "best_available_rank", "best_available_probability", "market_type", "best_pick")
    if size and all(column in frame for column in needed):
        work = pd.DataFrame({column: frame[column].to_numpy() for column in needed})
        work["best_available_rank"] = pd.to_numeric(work["best_available_rank"], errors="coerce")
        work = work[work["best_available_rank"].notna()]
        for _, group in work.groupby("matchup_id", sort=False):
            group = group.sort_values("best_available_rank", kind="mergesort")
            values = [_number(value) for value in group["best_available_probability"]]
            markets = [str(value) for value in group["market_type"]]
            names = [str(value) for value in group["best_pick"]]
            for i, position in enumerate(group.index):
                same = [j for j in range(len(values)) if j != i and _level(values[i], values[j])]
                count[position] = len(same) + 1
                tied_with[position] = " | ".join(names[j] for j in same)
                no_lean[position] = any(markets[j] != markets[i] and _family(markets[j]) == _family(markets[i])
                                        for j in same)
    return pd.DataFrame({"best_available_tie_count": pd.array(count, dtype="Int64"),
                         "best_available_tied_with": pd.Series(tied_with, dtype=object),
                         "best_available_no_lean": pd.Series(no_lean, dtype=bool)}).set_axis(frame.index)


def tie_note(count, tied_with, no_lean, *, other_markets=True):
    """The grid's wording for one candidate's tie: ``NO_LEAN``, 'Level with <picks>', or ''.

    ``other_markets`` False leaves out a tie with another market's side (a Sides or Totals board
    shows one market, so only the coin flip inside it matters there).
    """
    if isinstance(no_lean, bool) and no_lean or str(no_lean).strip().lower() == "true":
        return NO_LEAN
    count = _number(count)
    others = "" if tied_with is None or (not isinstance(tied_with, str) and pd.isna(tied_with)) else str(tied_with).strip()
    if other_markets and count is not None and count > 1 and others:
        return "Level with " + others.replace(" | ", ", ")
    return ""
