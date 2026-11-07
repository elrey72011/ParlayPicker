"""How often an entry quote has a comparable certified close, by how early it was taken (2026-11-07).

A plan's bets need a certified close at the **same sportsbook, selection and line** as
the entry quote (``prospective_evidence._comparable_close``), for at least
``min_bet_close_coverage`` (80%) of them. Certified closes are recorded
(``prospective_closes``: the provider's last pregame snapshot, one per book, market and
selection), but at the snapshot's own line. A quote taken hours before tip-off often sits
at a line the market later moved off, so it has no comparable close.

This measures, for started games whose certified close was fetched, the share of verified
US-book entry quotes (Pinnacle excluded; it is never an entry price) that have a
comparable close, by minutes before the scheduled start. It shows how late a plan must
price its bets to reach the 80% bar. Read-only; it changes nothing.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app_core import prospective_evidence as evidence
from app_core.research_api_budget import REFERENCE_BOOKS

BUCKETS = ((0, 30, "under 30 min"), (30, 60, "30-60 min"), (60, 180, "1-3 h"), (180, 360, "3-6 h"),
           (360, 10 ** 9, "over 6 h"))


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def coverage(events, quotes, closes, *, now=None):
    """{bucket: {"quotes", "with_close_fetched", "comparable", "share"}} over started games."""
    now = now or datetime.now(timezone.utc)
    events = {e["event_id"]: e for e in events}
    closed = {}
    for c in closes:
        if c.get("close_verified"):
            closed.setdefault((c["event_id"], c["market_family"], c["selection"], c["sportsbook"]), set()).add(c["line"])
    out = {label: {"quotes": 0, "with_close_fetched": 0, "comparable": 0} for _, _, label in BUCKETS}
    for q in quotes:
        event = events.get(q["event_id"])
        if (event is None or not q.get("quote_verified") or q.get("line") is None
                or q["sportsbook"] in REFERENCE_BOOKS):
            continue
        start = _time(event["scheduled_start"])
        taken = _time(q["quote_timestamp"])
        if start > now or taken > start:
            continue
        minutes = (start - taken).total_seconds() / 60
        label = next(lbl for low, high, lbl in BUCKETS if low <= minutes < high)
        row = out[label]
        row["quotes"] += 1
        lines = closed.get((q["event_id"], q["market_family"], q["selection"], q["sportsbook"]))
        if lines is None:
            continue
        row["with_close_fetched"] += 1
        row["comparable"] += int(q["line"] in lines)
    for row in out.values():
        row["share"] = row["comparable"] / row["with_close_fetched"] if row["with_close_fetched"] else None
    return out


def from_store(path, sport, family, *, now=None):
    events = evidence.read_records(path, "prospective_event", sport=sport)
    quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=family)
    closes = evidence.read_records(path, "prospective_close", sport=sport, market_family=family)
    return {"sport": sport, "market_family": family, "by_minutes_before_start": coverage(events, quotes, closes, now=now)}
