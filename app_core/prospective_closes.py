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


def _bet_scopes(path, sport):
    """{(event_id, market_family, sportsbook)} for recorded predictions' quotes: the books a plan bets at."""
    try:
        predictions = evidence.read_records(path, "prospective_prediction", sport=sport)
    except (KeyError, ValueError, LookupError, OSError):
        return set()  # no readable predictions: plain order, never a failed close run
    if not predictions:
        return set()
    quotes = {q["quote_id"]: q for q in evidence.read_records(path, "prospective_quote", sport=sport)}
    scopes = set()
    for prediction in predictions:
        quote = quotes.get(prediction.get("quote_id"))
        if quote is not None:
            scopes.add((quote["event_id"], quote["market_family"], quote["sportsbook"]))
    return scopes


REPLAY_EDGE = 0.01     # the replay's lower edge: its bets at 2% are among these


def _replay_scopes(path, sport, now, lookback, setting=os.environ.get):
    """{(event_id, market_family, sportsbook)} of the replay's first-edge bets on started games (2026-11-10).

    The closing-line value of those bets is what a go/no-go rests on, and on 2026-11-09 13 of
    the 22 had no close: a whole-game request costs 40 credits (two markets, 12 books), so the
    daily allowance covered two NBA games. Closing only the bet's own book and market costs 10.
    Both entry rules count (any capture, and the last hour only). Read-only; an unreadable scope
    is an empty set, never a failed close run.
    """
    try:
        from app_core import pinnacle_model as pm
        from app_core.prospective_research_models import _read_scope
        scopes = set()
        for family in sorted(enabled_families(sport, setting)):
            events, quotes, _ = _read_scope(path, sport, family)
            started = {e["event_id"]: e for e in events.values() if now - lookback <= _time(e["scheduled_start"]) <= now}
            if not started:
                continue
            kept, _ = pm.without_junk(pm.slim_sources([q for q in quotes if q["event_id"] in started]))
            for sample in (kept, pm.late_captures(started, kept, pm.LATE_ENTRY_MINUTES)):
                for event_id, bet, _ in pm.first_edge_bets(started, sample, now, REPLAY_EDGE, pm.PLAN_BOOKS):
                    scopes.add((event_id, family, bet["sportsbook"]))
        return scopes
    except Exception:
        return set()


def due(path, now, *, sports, lookback=LOOKBACK, setting=os.environ.get):
    """[(event, [verified quotes])] for started events still missing a certified close per quote scope.

    Plan bets first (owner request 2026-11-07): a plan needs a comparable close for its own
    bets, and each request is charged per 10 bookmakers. So for every game with a recorded
    prediction, the books of the predicted quotes (both sides, that market) come first as
    their own entries, at 10 credits per market; every other due game and book follows in
    the usual order, while the daily allowance lasts. Without predictions nothing changes.

    The replay's first-edge bets count as bets too (2026-11-10, ``_replay_scopes``): their book
    and market are closed first, so the allowance goes to the closes that can be measured.
    """
    priority, rest = due_split(path, now, sports=sports, lookback=lookback, setting=setting)
    return priority + rest


REST_SHARE_ENV = "PARLAYPICKER_CERTIFIED_CLOSE_REST_SHARE"


def rest_share(setting=os.environ.get):
    """The share of a sport's daily close allowance that whole-game closes (no bet) may use: 0 to 1.

    1 (the default) is the behaviour before 2026-11-10. A whole game costs 40 credits and a bet's
    own book and market 10. Left alone, the first games to start each day spent the allowance
    on whole-game closes, and bets on later games had none (2026-11-10: NCAAB at 400 of 400).
    With a share of 0.25, a whole-game request is made only while the day's use stays within a
    quarter of the allowance; the rest is kept for bets. Anything unreadable is 1.
    """
    try:
        share = float(str(setting(REST_SHARE_ENV, "") or "").strip())
    except ValueError:
        return 1.0
    return share if 0 <= share <= 1 else 1.0


PINNACLE_ENV = "PARLAYPICKER_CERTIFIED_CLOSE_PINNACLE"


def pinnacle_closes(setting=os.environ.get):
    """Whether a bet's close request also asks for Pinnacle's close (off unless the variable says so).

    Found 2026-11-12: Novig's closing snapshots sat 4 to 25 points from the market minutes before
    tip-off, so the close at the stake book cannot judge a bet there. Pinnacle's close is the usual
    benchmark, and until now it was never fetched (reference books are left out of the closes).
    A request is charged per 10 bookmakers, so adding Pinnacle to a bet's own book costs nothing.
    """
    return str(setting(PINNACLE_ENV, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _pinnacle_quotes(path, sport, families):
    """{(event_id, market_family): [one verified Pinnacle quote per selection still without a close]}."""
    closed = {(c["event_id"], c["market_family"], c["selection"])
              for c in evidence.read_records(path, "prospective_close", sport=sport)
              if c["close_verified"] and c["sportsbook"] in REFERENCE_BOOKS}
    found = {}
    for quote in evidence.read_records(path, "prospective_quote", sport=sport):
        scope = (quote["event_id"], quote["market_family"], quote["selection"])
        if (quote["sportsbook"] in REFERENCE_BOOKS and quote["quote_verified"] and quote["market_family"] in families
                and scope not in closed):
            found.setdefault(scope[:2], {}).setdefault(quote["selection"], quote)
    return {key: list(by_selection.values()) for key, by_selection in found.items()}


def _with_pinnacle(first, reference):
    """``first`` plus Pinnacle's quotes for the same games and markets, while the request costs the same."""
    books = {q["sportsbook"] for q in first}
    extra = [q for key in sorted({(q["event_id"], q["market_family"]) for q in first}) for q in reference.get(key, [])]
    added = {q["sportsbook"] for q in extra}
    if not extra or math.ceil(len(books | added) / 10) > math.ceil(len(books) / 10):
        return first
    return first + extra


def due_split(path, now, *, sports, lookback=LOOKBACK, setting=os.environ.get):
    """``due`` as two lists: (bet scopes at the bet's book, every other due game and book).

    With ``PARLAYPICKER_CERTIFIED_CLOSE_PINNACLE`` on, a bet's request also asks for Pinnacle's close
    for the same game and market (``pinnacle_closes``), at no extra cost.
    """
    priority, rest = [], []
    for sport in sports:
        bets = _bet_scopes(path, sport) | _replay_scopes(path, sport, now, lookback, setting)
        reference = (_pinnacle_quotes(path, sport, enabled_families(sport, setting))
                     if pinnacle_closes(setting) else {})
        for event, quotes in _due_sport(path, now, sport, lookback, setting):
            first = [q for q in quotes if (q["event_id"], q["market_family"], q["sportsbook"]) in bets]
            later = [q for q in quotes if (q["event_id"], q["market_family"], q["sportsbook"]) not in bets]
            if first and reference:
                first = _with_pinnacle(first, reference)
            if first:
                priority.append((event, first))
            if later:
                rest.append((event, later))
    return priority, rest


def _due_sport(path, now, sport, lookback, setting):
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
    return [(events[event_id], quotes) for event_id, quotes in sorted(by_event.items())]


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


REQUESTED_STATE = "certified_close_requested_v1"
COST_FLAG = "certified_close_cost_review_required"


def _scope(quote):
    return f"{quote['market_family']}|{quote['sportsbook']}|{quote['selection']}"


def _requested_scopes(budget, now):
    """{event_id: {start, scopes}} of snapshots already fetched, kept in the budget's own state
    (saved with it) and pruned once the event is past the lookback; None without a budget."""
    state = getattr(budget, "state", None)
    if not isinstance(state, dict):
        return None
    requested = state.setdefault(REQUESTED_STATE, {})
    if not isinstance(requested, dict):
        requested = state[REQUESTED_STATE] = {}
    for event_id in [e for e, v in requested.items()
                     if not isinstance(v, dict) or _time(v.get("start")) < now - LOOKBACK - timedelta(days=1)]:
        del requested[event_id]
    return requested


def _over_rest_share(budget, sport, credits, share):
    """True when a whole-game request would take the day's close use past ``share`` of the allowance."""
    try:
        from app_core.research_api_budget import certified_close_limits
        limit = (certified_close_limits().get(sport) or {}).get("daily")
        if budget is None or limit is None:
            return False
        return budget.usage(sport, budget.close_ledger)["daily"] + credits > share * limit
    except Exception:  # an unreadable ledger never blocks a close the allowance itself would allow
        return False


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
    state = getattr(budget, "state", None)
    if isinstance(state, dict) and (state.get(COST_FLAG) or state.get("api_budget_cost_mismatch")):
        # A provider charge above plan (here or in research capture) stops certified closes too,
        # until the owner reviews it (audit 2026-11-09).
        report["skipped"].append("PROVIDER_COST_REVIEW_REQUIRED")
        return report
    requested = _requested_scopes(budget, now)
    pending = []
    bets_first, others = due_split(path, now, sports=sports, setting=setting)
    whole_game = {id(quotes) for _, quotes in others}
    share = rest_share(setting)
    for event, quotes in bets_first + others:
        # A historical snapshot at the last pregame second never changes, so a book missing from
        # it stays missing. Ask once per scope, never again every cycle (audit 2026-11-09).
        done = set(((requested or {}).get(event["event_id"]) or {}).get("scopes") or ())
        left = [q for q in quotes if _scope(q) not in done]
        if left:
            pending.append((event, left, id(quotes) in whole_game))
        elif quotes:
            report.setdefault("already_requested", 0)
            report["already_requested"] += 1
    for event, quotes, is_whole_game in pending[:max_events]:
        url, params, credits = request_plan(event, quotes)
        if is_whole_game and share < 1 and _over_rest_share(budget, event["sport"], credits, share):
            report.setdefault("held_for_bets", 0)     # allowance kept for bets on games still to start
            report["held_for_bets"] += 1
            continue
        report["events"] += 1
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
        if requested is not None:
            entry = requested.setdefault(event["event_id"], {"start": event["scheduled_start"], "scopes": []})
            entry["scopes"] = sorted(set(entry["scopes"]) | {_scope(q) for q in quotes})
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
                if isinstance(state, dict):
                    state[COST_FLAG] = f"planned={credits}:charged={actual}"  # sticky until reviewed
                    budget.persist()
            report["stopped"] = "COST_MISMATCH"
            break
    return report


def summary(report, sport=None, budget=None):
    """One log line's worth of a ``record`` report: counts and codes only (owner 2026-11-10).

    The report was kept in the cycle result but printed nowhere, so a night with no closes
    certified left no trace of why. Errors are reduced to their kind: a provider error's text
    can carry a request URL, so it is never copied. With ``budget``, the sport's
    certified-close credits used against its allowance are added.
    """
    kinds = {}
    for error in report.get("errors") or []:
        parts = str(error).split(":")
        named = [p for p in parts if p.endswith(("Error", "Exception", "Limit", "Conflict", "Timeout")) and p.isalnum()]
        if "BudgetLimit" in parts:                                 # an event id can hold a colon (run 2026-11-10)
            kind = "BudgetLimit:" + parts[-1][:20]                 # the window that ran out: daily or rolling_31_days
        elif "COST_MISMATCH" in parts:
            kind = "COST_MISMATCH"
        elif named:
            kind = named[0]
        else:
            kind = parts[1] if len(parts) > 1 and parts[1].replace("_", "").isalnum() else "ERROR"
        kinds[kind] = kinds.get(kind, 0) + 1
    out = {key: int(report.get(key) or 0) for key in ("events", "requests", "credits", "stored", "already_requested")}
    if report.get("held_for_bets"):
        out["held_for_bets"] = int(report["held_for_bets"])
    out["skipped"] = [str(code)[:60] for code in report.get("skipped") or []]
    out["errors"] = kinds
    if budget is not None and sport is not None:
        try:
            from app_core.research_api_budget import certified_close_limits
            out["allowance"] = {"used": budget.usage(sport, budget.close_ledger),
                                "limits": certified_close_limits().get(sport)}
        except Exception:  # a summary never fails a cycle
            pass
    return out

