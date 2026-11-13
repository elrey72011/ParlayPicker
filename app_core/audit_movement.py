"""Pinnacle line movement columns for the candidate audit export (owner 2026-11-12; research only).

The price watch publishes, per upcoming game, where Pinnacle's main line and no-vig price
started and where they are now (``price_gap_candidates.movement``). This joins those rows onto
the audit's rows when it is exported. The audit itself, the evidence saved from it and every
pick are untouched: the columns exist only in the downloaded file.

The audit names a team by a short name ("Vegas", "New York") and the price store by the
provider's full name ("Vegas Golden Knights"). A game matches when the league is the same, the
start times are within ``START_TOLERANCE``, and each audit name is the leading words of the
store's name for the same side. A game that does not match exactly one movement row, and every
sport the price watch does not capture, gets blank columns: a blank means "not measured", never
"did not move".

The line is the home line for a side market and the total for a total; the probability is the
home side's or the Over's, whichever side the audit row is on.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import re

START_TOLERANCE = timedelta(minutes=20)
COLUMNS = ("pinnacle_first_line", "pinnacle_current_line", "pinnacle_line_move", "pinnacle_first_probability",
           "pinnacle_current_probability", "pinnacle_probability_shift", "pinnacle_moved_toward",
           "pinnacle_move_with_pick", "pinnacle_stake_books_at_line", "pinnacle_stake_books_off_line",
           "pinnacle_captures", "pinnacle_movement_at")
_FAMILIES = {"spread": ("SPREAD", "PUCK_LINE"), "total": ("TOTAL",)}
_SOURCE = {"pinnacle_first_line": "pinnacle_first_line", "pinnacle_current_line": "pinnacle_line",
           "pinnacle_line_move": "line_move", "pinnacle_first_probability": "pinnacle_first_probability",
           "pinnacle_current_probability": "pinnacle_probability", "pinnacle_probability_shift": "probability_shift",
           "pinnacle_moved_toward": "moved_toward", "pinnacle_captures": "captures",
           "pinnacle_movement_at": "last_capture_at"}


# The provider shortens names the audit spells out, and the audit drops hyphens (found 2026-11-13 on
# two real exports: 24 of 195 names did not match, 18 of them "State" against "St").
_TOKENS = {"state": "st", "saint": "st"}
# A spelled-out "North Carolina Wilmington" against the provider's "UNC Wilmington" is deliberately
# left unmatched (an existing test pins it): a blank is safer than an alias table that can go stale.
_ALIASES = {"california baptist": "cal baptist"}


def _words(name):
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", "", text.replace("-", " ")).split()
    words = [_TOKENS.get(word, word) for word in words]
    alias = _ALIASES.get(" ".join(words))
    return alias.split() if alias else words


def _leads(short, full):
    """Whether the audit's short name is the start of the provider's full name.

    Compared word by word, and also with the spaces removed, because the audit writes
    "Arkansaspine Bluff" for the provider's "Arkansas-Pine Bluff Golden Lions".
    """
    short, full = _words(short), _words(full)
    if not short:
        return False
    if full[:len(short)] == short:
        return True
    joined = "".join(short)
    return len(joined) >= 8 and "".join(full).startswith(joined)


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _match(row, moves):
    """The one movement row for this audit row's game and market, or None."""
    kind = str(row.get("market_type") or "").split("_")[0]
    start = _time(row.get("game_start_utc"))
    if kind not in _FAMILIES or start is None:
        return None
    found = []
    for move in moves:
        other = _time(move.get("scheduled_start"))
        if (move.get("market_family") in _FAMILIES[kind] and other is not None
                and str(move.get("sport", "")).upper() == str(row.get("league", "")).upper()
                and abs(other - start) <= START_TOLERANCE
                and _leads(row.get("home_team"), move.get("home_team"))
                and _leads(row.get("away_team"), move.get("away_team"))):
            found.append(move)
    return found[0] if len(found) == 1 else None


def _implied(price):
    price = float(price)
    if abs(price) < 100:
        raise ValueError("not an American price")
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def _refreshed(row, move):
    """``move`` brought up to the audit row's own Pinnacle quote when that quote is the newer one.

    The published movement can be hours old for a game the price watch has not reached yet: on
    2026-11-13 it showed Pinnacle's total at 154 from 12:26 PM beside a fair price the 5:06 PM
    Refresh had read at 152.5. The row's saved provider quotes are from that Refresh, so when they
    hold both Pinnacle sides at one line, recorded after the movement's last capture, the current
    line, price, direction and the books at that line are taken from them. The first line and
    price stay the watch's. Anything unreadable leaves ``move`` as it was.
    """
    import json
    try:
        total = str(row.get("market_type") or "").startswith("total")
        first, second = ("total_over", "total_under") if total else ("spread_home", "spread_away")
        quotes = json.loads(row.get("provider_quotes") or "[]")
        mine = {q.get("market_type"): q for q in quotes
                if isinstance(q, dict) and "pinnacle" in str(q.get("book", "")).lower()}
        one, other = mine[first], mine[second]
        line = float(one["point"])
        if abs(float(other["point"]) - (line if total else -line)) > 1e-8:
            return move
        seen = min(_time(one.get("recorded_at")), _time(other.get("recorded_at")))
        if seen <= _time(move.get("last_capture_at")):
            return move
        a, b = _implied(one["price"]), _implied(other["price"])
        probability = round(a / (a + b), 4)
        line_move = line - float(move["pinnacle_first_line"])
        shift = round(probability - float(move["pinnacle_first_probability"]), 4)
        from app_core.price_gap_candidates import _toward
        from app_core.public_quote_policy import canonical_book_label
        at, off = set(), set()
        for q in quotes:
            book = str(q.get("book", "")) if isinstance(q, dict) else ""
            if q.get("market_type") == first and book and "pinnacle" not in book.lower() and q.get("point") is not None:
                (at if abs(float(q["point"]) - line) < 1e-8 else off).add(canonical_book_label(book))
        return {**move, "pinnacle_line": line, "line_move": line_move, "pinnacle_probability": probability,
                "probability_shift": shift,
                "moved_toward": _toward(move, "TOTAL" if total else "SPREAD", line_move, shift),
                "captures": int(move.get("captures") or 0) + 1, "last_capture_at": seen.isoformat(timespec="seconds"),
                "stake_books_at_line": sorted(at), "stake_books_off_line": sorted(off - at)}
    except (KeyError, TypeError, ValueError, AttributeError):
        return move


def _with_pick(row, move):
    """Whether Pinnacle moved toward this row's side: True, False, or None when it did not move."""
    toward = move.get("moved_toward")
    if not toward:
        return None
    side = str(row.get("market_type") or "")
    mine = {"spread_home": move.get("home_team"), "spread_away": move.get("away_team"),
            "total_over": "Over", "total_under": "Under"}.get(side)
    return None if mine is None else toward == mine


def with_movement(audit, moves):
    """A copy of ``audit`` with ``COLUMNS`` appended; blank where a game has no single movement row.

    ``audit`` is returned unchanged (the same object) when it is empty or there is nothing to join,
    so an export without the price watch is byte-for-byte what it was.
    """
    if audit is None or getattr(audit, "empty", True) or not moves:
        return audit
    needed = {"league", "home_team", "away_team", "market_type", "game_start_utc"}
    if not needed <= set(audit.columns):
        return audit
    out = audit.copy()
    values = {column: [] for column in COLUMNS}
    for row in out.to_dict("records"):
        move = _match(row, moves)
        if move is not None:
            move = _refreshed(row, move)
        for column in COLUMNS:
            if move is None:
                values[column].append(None)
            elif column == "pinnacle_move_with_pick":
                values[column].append(_with_pick(row, move))
            elif column in ("pinnacle_stake_books_at_line", "pinnacle_stake_books_off_line"):
                values[column].append(", ".join(move.get(column.replace("pinnacle_", ""), []) or []))
            else:
                values[column].append(move.get(_SOURCE[column]))
    for column in COLUMNS:
        out[column] = values[column]
    return out
