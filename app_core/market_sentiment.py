"""Market sentiment read from real-money prices (owner 2026-11-15; research only, read-only).

"Sentiment" here is where money sits, not opinion or news: one venue (the *lean book*) rates a
side higher than the retail books do at the same line. Six nights of app exports (10 to 15
November 2026, 850 graded markets) had the side the Novig exchange rated higher than DraftKings,
FanDuel and BetMGM at 136-100 on spreads and 105-107 on totals. That is suggestive, not proof,
and it came from one look at a small sample. This keeps the running record over every stored
capture, so more nights answer it instead of those six.

The lean is read ``CUTOFF_MINUTES`` before the start, while a bet can still be placed. It READS
results. Nothing reads it back: no pick, plan or stake depends on it, and it is outside the
hashed prediction functions.

``direction_results`` is the base rate beside it (2026-11-16): how each side finished at the retail
books' main line at the same cutoff, Under against Over, away against home, underdog against
favourite. The same six nights of exports had Unders at 246-177 (college basketball 135-88), one
result among some 45 looks at those nights, so it proves nothing by itself; this counts every
stored game so the nights that follow can.
"""
from __future__ import annotations

from datetime import timedelta

from app_core import pinnacle_model as pm
from app_core import prospective_evidence as evidence
from app_core.public_quote_policy import canonical_book_label

EXCHANGE = "novig"
SHARP = "pinnacle"
RETAIL = ("draftkings", "fanduel", "betmgm")
CUTOFF_MINUTES = 60                # the lean is read this long before the start
MIN_GAP = 0.005                    # below half a point of probability the venues agree: no lean
STRONG_GAP = 0.01
FLAT_DECIMAL = 1 + 100 / 110       # a direction test pays every graded side a flat -110


def _book(quote):
    return canonical_book_label(quote["sportsbook"]).casefold()


def book_fairs(event, quotes, family, limit):
    """``{capture_time: {book: {home_line: (p_canonical, {selection: quote})}}}`` for captures up to ``limit``.

    One entry per book and line where the book quoted both sides in the same response. ``p_canonical``
    is the home side's or the Over's probability with the book's margin removed. A pair whose two
    sides sum above ``pinnacle_model.MAX_OVERROUND`` is an alternate line nobody would take and is
    left out. When one response holds a line more than once, the newest quote wins.
    """
    grouped = {}
    for q in quotes:
        if (q["event_id"] != event["event_id"] or q["market_family"] != family or not q["quote_verified"]
                or q["line"] is None or not q.get("decimal_odds") or q["decimal_odds"] <= 1):
            continue
        captured = pm.capture_time(event, q)
        if captured > limit:
            continue
        key = (captured, _book(q), pm._home_line(event, q))
        held = grouped.setdefault(key, {})
        stamp = str(q["quote_timestamp"])
        sides = held.setdefault(stamp, {})
        sides[q["selection"]] = q
    canonical = pm._canonical(event, family)
    out = {}
    for (captured, book, line), stamps in grouped.items():
        for stamp in sorted(stamps, reverse=True):
            sides = stamps[stamp]
            if len(sides) != 2 or canonical not in sides:
                continue
            mine = 1 / sides[canonical]["decimal_odds"]
            other = 1 / next(q for s, q in sides.items() if s != canonical)["decimal_odds"]
            if mine + other > pm.MAX_OVERROUND:
                continue
            out.setdefault(captured, {}).setdefault(book, {})[line] = (mine / (mine + other), sides)
            break
    return out


def lean(fairs, event, family, *, lean_book=EXCHANGE, against=RETAIL, min_gap=MIN_GAP):
    """The lean in the newest usable capture of ``fairs`` (``book_fairs``' result), or None.

    Usable: ``lean_book`` and at least one ``against`` book quote both sides at the same line. The
    line is the lean book's main one among those (price nearest even). ``gap`` is the lean book's
    probability for the home side or the Over minus the ``against`` books' average. ``selection``
    is the side the lean book rates higher, or None when the gap is under ``min_gap``; ``line`` is
    that side's own line and ``best_decimal`` the best price for it among the ``against`` books.
    """
    canonical = pm._canonical(event, family)
    for captured in sorted(fairs, reverse=True):
        books = fairs[captured]
        mine = books.get(lean_book)
        if not mine:
            continue
        shared = [line for line in mine if any(line in books.get(b, {}) for b in against)]
        if not shared:
            continue
        home_line = min(shared, key=lambda line: (abs(mine[line][0] - 0.5), line))
        others = [books[b][home_line] for b in against if home_line in books.get(b, {})]
        retail = sum(p for p, _ in others) / len(others)
        gap = mine[home_line][0] - retail
        found = {"captured": captured, "home_line": home_line, "lean_probability": mine[home_line][0],
                 "against_probability": retail, "against_books": len(others), "gap": gap,
                 "selection": None, "line": None, "best_decimal": None}
        if abs(gap) < min_gap:
            return found
        if gap > 0:
            selection = canonical
        else:
            selection = next(s for s in mine[home_line][1] if s != canonical)
        own_line = home_line if (family == "TOTAL" or selection == canonical) else -home_line
        prices = [sides[selection]["decimal_odds"] for _, sides in others if selection in sides]
        found.update(selection=selection, line=own_line, best_decimal=max(prices) if prices else None)
        return found
    return None


def read(event, quotes, family, *, lean_book=EXCHANGE, against=RETAIL, cutoff_minutes=CUTOFF_MINUTES,
         min_gap=MIN_GAP):
    """``lean`` over the captures at least ``cutoff_minutes`` before the start."""
    limit = pm._t(event["scheduled_start"]) - timedelta(minutes=cutoff_minutes)
    return lean(book_fairs(event, quotes, family, limit), event, family, lean_book=lean_book, against=against,
                min_gap=min_gap)


def _scope(path, sport, family, now, nights):
    """``(started games by id, newest final by game id, quotes by game id)`` for the last ``nights`` days."""
    from app_core import prospective_research_models as research
    events, quotes, scores = research._read_scope(path, sport, family)
    start = now - timedelta(days=nights)
    started = {e["event_id"]: e for e in events.values() if start <= pm._t(e["scheduled_start"]) < now}
    final = {}
    for score in scores:
        held = final.get(score["event_id"])
        if held is None or (score["grading_version"], score["available_at"], score["result_id"]) > \
                (held["grading_version"], held["available_at"], held["result_id"]):
            final[score["event_id"]] = score
    return started, final, pm.by_event(pm.slim_sources([q for q in quotes if q["event_id"] in started]))


def results_by_venue(path, sport, family, *, now, nights=14, lean_books=(EXCHANGE, SHARP), against=RETAIL):
    """One record per lean book: per slate night, how the side it rated higher than ``against`` finished.

    For every started game with a verified final, the side from ``lean``, graded at its own line.
    Each graded side is counted twice: paid at a flat -110 (a direction test) and paid at the best
    price an ``against`` book offered for it in that capture (what following the lean at a retail
    book would have returned). The other side is the mirror at -110. The store is read once and
    each game's quotes are paired once, whatever the number of lean books.
    """
    from app_core.price_edge_screen import _mean_se
    started, final, grouped = _scope(path, sport, family, now, nights)
    by_night = {book: {} for book in lean_books}
    for event_id, event in started.items():
        limit = pm._t(event["scheduled_start"]) - timedelta(minutes=CUTOFF_MINUTES)
        fairs = book_fairs(event, grouped.get(event_id, []), family, limit)
        for book in lean_books:
            row = by_night[book].setdefault(pm.slate_night(event), {
                "games": 0, "no_quote": 0, "no_lean": 0, "ungraded": 0, "WIN": 0, "LOSS": 0, "PUSH": 0,
                "flat": [], "priced": [], "strong": [0, 0]})
            row["games"] += 1
            found = lean(fairs, event, family, lean_book=book, against=against)
            if found is None:
                row["no_quote"] += 1
                continue
            if found["selection"] is None:
                row["no_lean"] += 1
                continue
            score = final.get(event_id)
            if score is None:
                row["ungraded"] += 1
                continue
            outcome = evidence.score_settlement_outcome(
                sport, family, found["selection"], event["home_team"], event["away_team"],
                int(score["home_score"]), int(score["away_score"]), float(found["line"]))
            row[outcome] += 1
            row["flat"].append(FLAT_DECIMAL - 1 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0))
            if found["best_decimal"]:
                row["priced"].append(found["best_decimal"] - 1 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0))
            if outcome != "PUSH" and abs(found["gap"]) >= STRONG_GAP:
                row["strong"][outcome == "LOSS"] += 1

    def summary(rows):
        flat, flat_se = _mean_se([v for r in rows for v in r["flat"]])
        priced, priced_se = _mean_se([v for r in rows for v in r["priced"]])
        return {"games_started": sum(r["games"] for r in rows), "no_quote": sum(r["no_quote"] for r in rows),
                "no_lean": sum(r["no_lean"] for r in rows), "ungraded": sum(r["ungraded"] for r in rows),
                "wins": sum(r["WIN"] for r in rows), "losses": sum(r["LOSS"] for r in rows),
                "pushes": sum(r["PUSH"] for r in rows),
                "strong": [sum(r["strong"][i] for r in rows) for i in (0, 1)],
                "roi_flat": flat, "roi_flat_se": flat_se, "roi_priced": priced, "roi_priced_se": priced_se,
                "priced_bets": sum(len(r["priced"]) for r in rows)}
    return [{"sport": sport, "market_family": family, "lean_book": book, "against": list(against),
             "cutoff_minutes": CUTOFF_MINUTES, "min_gap": MIN_GAP, "strong_gap": STRONG_GAP,
             "nights": [dict(summary([row]), night=night.isoformat()) for night, row in sorted(by_night[book].items())],
             "total": summary(list(by_night[book].values()))} for book in lean_books]


def results(path, sport, family, *, now, nights=14, lean_book=EXCHANGE, against=RETAIL):
    """``results_by_venue`` for one lean book."""
    return results_by_venue(path, sport, family, now=now, nights=nights, lean_books=(lean_book,), against=against)[0]


def main_line(fairs, books=RETAIL):
    """The line most of ``books`` quote both sides at, in the newest capture any of them is in; else None.

    ``fairs`` is ``book_fairs``' result. Ties go to the line priced nearest even, then the lower
    one. Returns ``(captured, home_line, best)`` with ``best`` the best decimal price for each
    selection among those books at that line.
    """
    for captured in sorted(fairs, reverse=True):
        held = {}
        for book in books:
            for line, pair in fairs[captured].get(book, {}).items():
                held.setdefault(line, []).append(pair)
        if not held:
            continue
        line = min(held, key=lambda at: (-len(held[at]), abs(sum(p for p, _ in held[at]) / len(held[at]) - 0.5), at))
        best = {}
        for _, sides in held[line]:
            for selection, quote in sides.items():
                best[selection] = max(best.get(selection, 0.0), quote["decimal_odds"])
        return captured, line, best
    return None


def direction_results(path, sport, family, *, now, nights=14, books=RETAIL):
    """Per slate night, how each side finished at the ``books``' main line ``CUTOFF_MINUTES`` before the start.

    A base rate, not a signal. One record per split: Under against Over for a total; away against
    home and underdog against favourite otherwise (a pick'em has no favourite and is left out of
    that split). Wins and losses are the first-named side's. Each side is also paid at the best
    price one of ``books`` offered for it at that line.
    """
    from app_core.price_edge_screen import _mean_se
    started, final, grouped = _scope(path, sport, family, now, nights)
    splits = (("Under", "Over"),) if family == "TOTAL" else (("Away", "Home"), ("Underdog", "Favourite"))
    by_night = {split: {} for split in splits}

    def blank():
        return {"games": 0, "no_quote": 0, "ungraded": 0, "no_side": 0, "WIN": 0, "LOSS": 0, "PUSH": 0,
                "first": [], "second": []}

    def paid(outcome, decimal):
        return decimal - 1 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0)
    for event_id, event in started.items():
        night = pm.slate_night(event)
        rows = [by_night[split].setdefault(night, blank()) for split in splits]
        for row in rows:
            row["games"] += 1
        limit = pm._t(event["scheduled_start"]) - timedelta(minutes=CUTOFF_MINUTES)
        found = main_line(book_fairs(event, grouped.get(event_id, []), family, limit), books)
        score = final.get(event_id)
        if found is None or score is None:
            for row in rows:
                row["no_quote" if found is None else "ungraded"] += 1
            continue
        _, home_line, best = found
        canonical = pm._canonical(event, family)
        other = next(selection for selection in best if selection != canonical)

        def grade(selection):
            line = home_line if (family == "TOTAL" or selection == canonical) else -home_line
            return evidence.score_settlement_outcome(sport, family, selection, event["home_team"], event["away_team"],
                                                     int(score["home_score"]), int(score["away_score"]), float(line))
        for split, row in zip(splits, rows):
            if split[0] == "Underdog":
                if home_line == 0:
                    row["no_side"] += 1
                    continue
                first, second = (canonical, other) if home_line > 0 else (other, canonical)
            else:
                first, second = other, canonical            # Under, or the away side
            outcome = grade(first)
            row[outcome] += 1
            row["first"].append(paid(outcome, best[first]))
            row["second"].append(paid(grade(second), best[second]))

    def summary(rows):
        first, first_se = _mean_se([v for r in rows for v in r["first"]])
        second, second_se = _mean_se([v for r in rows for v in r["second"]])
        return {"games_started": sum(r["games"] for r in rows), "no_quote": sum(r["no_quote"] for r in rows),
                "ungraded": sum(r["ungraded"] for r in rows), "no_side": sum(r["no_side"] for r in rows),
                "wins": sum(r["WIN"] for r in rows), "losses": sum(r["LOSS"] for r in rows),
                "pushes": sum(r["PUSH"] for r in rows), "roi_first": first, "roi_first_se": first_se,
                "roi_second": second, "roi_second_se": second_se}
    return [{"sport": sport, "market_family": family, "first": split[0], "second": split[1], "books": list(books),
             "cutoff_minutes": CUTOFF_MINUTES,
             "nights": [dict(summary([row]), night=night.isoformat()) for night, row in sorted(by_night[split].items())],
             "total": summary(list(by_night[split].values()))} for split in splits]
