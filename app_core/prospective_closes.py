"""Certified closing prices for captured prospective quotes (roadmap C1). Off by default.

A *certified close* is The Odds API's last historical snapshot before the scheduled
start, retrieved after the game has started (``prospective_evidence.verify_provider_close``).
Once a game has started no later pregame price can exist, and the snapshot can be
fetched again and replayed, so this satisfies the frozen plans' rules
``certified_replayable_close_only`` and ``near_start_candidate_is_verified_close: False``.

For each recently started event with verified entry quotes and no certified close
yet, this module makes **one** historical event-odds request. It is limited to
the quotes' sportsbooks, the spread-side and total markets, and the second before
the start. It then stores one certified close per quoted sportsbook, market and
selection.

**Cost.** Historical event odds cost 10 credits per market per 10 bookmakers. The
default is the spread-side market only (``PARLAYPICKER_CERTIFIED_CLOSE_MARKETS``;
``SPREAD,TOTAL`` widens it), so about 10 credits per game. Every request is reserved
first against its own allowance (``Budget.reserve_certified_close``;
``ALL_CERTIFIED_CLOSE_LIMITS``: NBA 100/day and 2,500 per 31 days (D13), NCAAB 400/day and 4,500 (D14); owner
decision D13), so it never crowds out capture. A response whose ``x-requests-last``
exceeds the reservation is charged in full and stops the batch. Nothing runs
unless ``PARLAYPICKER_CERTIFIED_CLOSE_SPORTS`` names the sports. The default is none.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os

from app_core import prospective_evidence as evidence
from app_core.research_api_budget import REFERENCE_BOOKS

HISTORICAL_EVENT_ODDS = "https://api.the-odds-api.com/v4/historical/sports/{sport_key}/events/{event_id}/odds"
CREDITS_PER_MARKET_PER_10_BOOKS = 10
# Started events older than this are left for a manual backfill. Research runs start
# only a few times a day, so the window spans a full day (D16); the snapshot before
# the start can be fetched at any time after it, at the same cost.
LOOKBACK = timedelta(hours=24)


def enabled_sports(setting=os.environ.get):
    raw = str(setting("PARLAYPICKER_CERTIFIED_CLOSE_SPORTS", "") or "")
    return tuple(s for s in (p.strip().upper() for p in raw.split(",")) if s in evidence.ODDS_API_SPORT_KEYS)


def enabled_families(sport, setting=os.environ.get):
    """Market families to close. The default is the spread-side family only (D13, option B)."""
    side = evidence.ODDS_API_SIDE_MARKETS[sport]
    raw = str(setting("PARLAYPICKER_CERTIFIED_CLOSE_MARKETS", "") or "").upper()
    wanted = {p.strip() for p in raw.split(",") if p.strip()}
    families = {side if w in ("SPREAD", side) else w for w in wanted} & {side, "TOTAL"}
    return families or {side}


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def _market_key(sport, family):
    return "spreads" if family == evidence.ODDS_API_SIDE_MARKETS[sport] else "totals"


def due(path, now, *, sports, lookback=LOOKBACK, setting=os.environ.get):
    """[(event, [verified quotes])] for started events still missing a certified close per quote scope."""
    out = []
    for sport in sports:
        families = enabled_families(sport, setting)
        events = {e["event_id"]: e for e in evidence.read_records(path, "prospective_event", sport=sport)}
        closes = {(c["event_id"], c["market_family"], c["selection"], c["sportsbook"])
                  for c in evidence.read_records(path, "prospective_close", sport=sport) if c["close_verified"]}
        by_event = {}
        for quote in evidence.read_records(path, "prospective_quote", sport=sport):
            event = events.get(quote["event_id"])
            if (not quote["quote_verified"] or event is None or event["provider_namespace"] != "THE_ODDS_API" or
                    quote["market_family"] not in families or quote["sportsbook"] in REFERENCE_BOOKS):
                continue
            start = _time(event["scheduled_start"])
            if not now - lookback <= start <= now:
                continue
            if (quote["event_id"], quote["market_family"], quote["selection"], quote["sportsbook"]) in closes:
                continue
            by_event.setdefault(quote["event_id"], []).append(quote)
        out.extend((events[event_id], quotes) for event_id, quotes in sorted(by_event.items()))
    return out


def request_plan(event, quotes):
    """(url, params, credits) for the single historical request covering ``quotes``."""
    sport = event["sport"]
    books = sorted({q["sportsbook"] for q in quotes})
    markets = sorted({_market_key(sport, q["market_family"]) for q in quotes})
    before = _time(event["scheduled_start"]) - timedelta(seconds=1)
    params = dict(bookmakers=",".join(books), markets=",".join(markets), oddsFormat="american",
                  date=before.strftime("%Y-%m-%dT%H:%M:%SZ"))
    credits = CREDITS_PER_MARKET_PER_10_BOOKS * len(markets) * math.ceil(len(books) / 10)
    url = HISTORICAL_EVENT_ODDS.format(sport_key=evidence.ODDS_API_SPORT_KEYS[sport],
                                       event_id=event["provider_event_id"])
    return url, params, credits


def closes_from_snapshot(event, quotes, response, retrieved_at):
    """Certified close rows, one per quoted sportsbook, market and selection present in the snapshot.

    The close line is the snapshot's own line. When it differs from the entry line,
    the close is still recorded, but it is not *comparable* for that quote.
    """
    sport = event["sport"]
    raw = json.dumps({"retrieved_at": retrieved_at.isoformat(), "response": response},
                     sort_keys=True, separators=(",", ":"))
    data = response.get("data") or {}
    snapshot = response.get("timestamp")
    rows, seen = [], set()
    for quote in quotes:
        scope = (quote["sportsbook"], quote["market_family"], quote["selection"])
        if scope in seen:
            continue
        seen.add(scope)
        market_key = _market_key(sport, quote["market_family"])
        for book in data.get("bookmakers") or []:
            if book.get("key") != quote["sportsbook"]:
                continue
            for market in book.get("markets") or []:
                if market.get("key") != market_key:
                    continue
                for outcome in market.get("outcomes") or []:
                    if outcome.get("name") != quote["selection"]:
                        continue
                    price = outcome.get("price")
                    if type(price) is not int:
                        continue
                    source_id = evidence.certified_close_source_id(
                        event["provider_event_id"], quote["sportsbook"], market_key, quote["selection"], snapshot)
                    rows.append(dict(
                        close_id="certified-close:" + hashlib.sha256(source_id.encode()).hexdigest()[:32],
                        event_id=event["event_id"], sport=sport, market_family=quote["market_family"],
                        selection=quote["selection"], line=outcome.get("point"), american_odds=price,
                        decimal_odds=1 + (price / 100 if price > 0 else 100 / abs(price)),
                        sportsbook=quote["sportsbook"], close_timestamp=snapshot,
                        close_source=evidence.HISTORICAL_CLOSE_SOURCE, close_verified=True,
                        source_id=source_id, raw_source=raw))
    return rows


def _charged(response):
    try:
        return int((getattr(response, "headers", None) or {}).get("x-requests-last"))
    except (TypeError, ValueError):
        return None


def record(path, *, now, key, request_get, budget=None, sports=None, setting=os.environ.get, max_events=20):
    """Fetch and store certified closes for due events. Returns a sanitized report; never raises on one event."""
    sports = enabled_sports(setting) if sports is None else tuple(sports)
    report = dict(sports=list(sports), events=0, requests=0, credits=0, stored=0, skipped=[], errors=[])
    if not sports:
        report["skipped"].append("CERTIFIED_CLOSES_DISABLED")
        return report
    if not key:
        report["skipped"].append("ODDS_API_KEY_MISSING")
        return report
    for event, quotes in due(path, now, sports=sports, setting=setting)[:max_events]:
        report["events"] += 1
        url, params, credits = request_plan(event, quotes)
        try:
            if budget is not None:
                budget.reserve_certified_close(event["sport"], credits)
            response = request_get(url, params=dict(params, apiKey=key), timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # budget limits and provider errors end this event only
            report["errors"].append(f"{event['event_id']}:{type(exc).__name__}:{str(exc)[:80]}")
            if type(exc).__name__ == "BudgetLimit":
                break
            continue
        report["requests"] += 1
        report["credits"] += credits
        for row in closes_from_snapshot(event, quotes, payload, now):
            try:
                evidence.insert_certified_close(path, row)
                report["stored"] += 1
            except (ValueError, evidence.EvidenceConflict) as exc:
                report["errors"].append(f"{row['close_id']}:{str(exc)[:80]}")
        actual = _charged(response)
        if actual is not None and actual > credits:
            # The provider charged more than planned: record the difference, then stop.
            report["credits"] += actual - credits
            report["errors"].append(f"{event['event_id']}:COST_MISMATCH:planned={credits}:charged={actual}")
            if budget is not None:
                budget.reserve_certified_close(event["sport"], actual - credits, already_spent=True)
            report["stopped"] = "COST_MISMATCH"
            break
    return report
