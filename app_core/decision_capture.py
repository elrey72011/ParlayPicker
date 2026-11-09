"""C6 step 1: fresh decision-time quotes and bound predictions for owner-switched qualified markets.

``docs/roadmap/C6-PLAN-2026-11-06.md``. The market gate accepts a row only when its quote is
at most 30 minutes old, but the research cycle predicts once per game from the game's
*first* captured quote, hours before the start. This job, run every ``INTERVAL_MINUTES``
through a slate (``scripts/decision_capture_loop.py``), does for each market named in
``PARLAYPICKER_DECISION_MARKETS`` (for example ``NCAAB:SPREAD``) that is also qualified
(``app_core.selection.qualification``):

1. ``capture``: one Odds API request (US books, that market only: 1 credit) for the stored
   games starting within ``HORIZON``. Each quote is stored with its own provider response
   (``prospective_evidence.quote_source``), so it replays exactly like a first-capture quote.
2. ``predict``: a bound prediction from the plan's exact model and calibration against the
   newest fresh quote on **both** sides of each game (``decision_quotes``). The bridge
   (``app_core.qualified_bridge``) then offers the better-EV side to the gate and Gemini.

Off unless the owner names a market. Credits come from their own allowance
(``DECISION_LIMITS``, kept in a separate ledger so this loop never rewrites the research
scheduler's state). It grants no authority: rows still need the gate, activation and review.
Validation is unaffected: a plan scores each game's first prediction, and a decision
prediction is made only for a game that already has one.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from app_core import prospective_evidence as evidence
from app_core import prospective_research_models as research
from app_core.research_api_budget import REFERENCE_BOOKS, BudgetLimit

DECISION_ENV = "PARLAYPICKER_DECISION_MARKETS"
DECISION_LIMITS = {"daily": 60, "rolling_31_days": 1500}
HORIZON = timedelta(minutes=60)
MAX_QUOTE_AGE = timedelta(minutes=30)
INTERVAL_MINUTES = 15
MAX_EVENTS_PER_REQUEST = 24
SPORT_KEYS = {"NBA": "basketball_nba", "NCAAB": "basketball_ncaab", "NHL": "icehockey_nhl"}


def markets(setting=None):
    """Owner-switched {(sport, family)} from ``PARLAYPICKER_DECISION_MARKETS`` ("NCAAB:SPREAD,NBA:TOTAL")."""
    raw = (setting or os.environ.get)(DECISION_ENV, "") or ""
    out = set()
    for item in raw.split(","):
        sport, _, family = item.strip().upper().partition(":")
        if sport in SPORT_KEYS and family in research.SCOPES.get(sport, ()):
            out.add((sport, family))
    return frozenset(out)


def _time(value):
    return evidence._time(value, "timestamp")


def due(path, sport, now, *, horizon=HORIZON):
    """Stored Odds API events starting within ``horizon`` of ``now``, soonest first."""
    events = [e for e in evidence.read_records(path, "prospective_event", sport=sport)
              if e["provider_namespace"] == "THE_ODDS_API" and now < _time(e["scheduled_start"]) <= now + horizon]
    return sorted(events, key=lambda e: (e["scheduled_start"], e["event_id"]))[:MAX_EVENTS_PER_REQUEST]


def _market_key(sport, family):
    return "spreads" if family == evidence.ODDS_API_SIDE_MARKETS[sport] else "totals"


def _utcnow():
    return datetime.now(timezone.utc)


COST_FLAG = "decision_cost_review_required"


def capture(path, sport, family, *, key, request_get, budget, clock=_utcnow, pinnacle=False):
    """Store fresh verified quotes for the games due in one market. Returns counts."""
    from app_core import odds_research_adapter as adapter
    result = {"due_events": 0, "requested": False, "stored_quotes": 0, "replay_blocked": 0, "blocker": None}
    events = {e["provider_event_id"]: e for e in due(path, sport, clock())}
    result["due_events"] = len(events)
    if not events:
        return result
    if budget.state.get(COST_FLAG):
        result["blocker"] = COST_FLAG  # sticky until the owner reviews the provider's charge
        return result
    # A Pinnacle-model plan needs Pinnacle's price in the same response: the 11 US books plus
    # Pinnacle are 12 bookmakers, 2 region equivalents, so 2 credits for one market.
    cost = 2 if pinnacle else 1
    try:
        budget.reserve("ODDS", cost)
    except BudgetLimit as exc:
        result["blocker"] = str(exc)
        return result
    result["requested"] = True
    charged_get = charging(request_get, budget, cost, COST_FLAG, result)

    from app_core.research_api_budget import capture_bookmakers
    books = {"bookmakers": capture_bookmakers(sport)} if pinnacle else {"regions": "us"}
    try:
        payload = adapter.fetch(sport, "odds", key, charged_get, params={
            **books, "markets": _market_key(sport, family), "oddsFormat": "american",
            "eventIds": ",".join(sorted(events))})
    except BudgetLimit as exc:  # a retry the allowance could not cover
        result["blocker"] = str(exc)
        return result
    stored, blocked = store_quotes(path, sport, family, events, payload, clock=clock, reference=pinnacle)
    result["stored_quotes"] += stored
    result["replay_blocked"] += blocked
    return result


def charging(request_get, budget, cost, flag, result):
    """``request_get`` that charges every provider attempt (audit 2026-11-09).

    The caller reserves the first attempt. ``odds_research_adapter.fetch`` retries a timeout,
    429 or 5xx up to twice, and the provider may charge each attempt, so every retry reserves
    ``cost`` again first (a ``BudgetLimit`` ends the request). A charge above ``cost`` sets the
    sticky ``flag`` until the owner reviews it.
    """
    attempts = [0]

    def charged_get(url, **kwargs):
        attempts[0] += 1
        if attempts[0] > 1:
            budget.reserve("ODDS", cost)
        response = request_get(url, **kwargs)
        spent = getattr(response, "headers", {}).get("x-requests-last")
        if spent is not None and str(spent).strip() not in {str(n) for n in range(cost + 1)}:
            budget.state[flag] = str(spent)
            try:  # the provider already charged it: count the credits above the reservation
                budget.record_overage("ODDS", int(str(spent).strip()) - cost)
            except ValueError:
                budget.persist()
            result["blocker"] = flag
        return response
    return charged_get


def store_quotes(path, sport, family, events, payload, *, clock=_utcnow, reference=False):
    """Store each verified ``family`` quote in ``payload`` for ``events`` ({provider id: event}).

    Each quote keeps its own provider response, so it replays like a first-capture quote.
    Reference books (Pinnacle) are stored only with ``reference``. Returns (stored, replay-blocked).

    The quote id includes the response time (audit 2026-11-09): every capture stores the full
    snapshot, including a book whose price did not change since an earlier capture. A first-edge
    plan compares each stake-book price with Pinnacle's price **from the same response**
    (``pinnacle_model.plan_quotes``), so a book that has not moved after Pinnacle moved (the lag the
    news study found) must be stored again with the new response. It also keeps two loops
    (decision and news capture) from writing the same quote id with different response bytes.
    """
    from app_core import odds_research_adapter as adapter
    stored = blocked = 0
    have = {q["quote_id"] for q in evidence.read_records(path, "prospective_quote", sport=sport, market_family=family)}
    for row in payload:
        event = events.get(row.get("id")) if isinstance(row, dict) else None
        if event is None:
            continue
        observed = clock()
        valid, _ = adapter.quotes(sport, row, observed)
        for quote in valid:
            # Pinnacle is stored only as the Pinnacle model's reference; it is never an entry price
            # (decision_quotes skips it).
            if quote["market_family"] != family or (quote["sportsbook"] in REFERENCE_BOOKS and not reference):
                continue
            quote_id = (f"odds-quote:{sport}:"
                        f"{research._digest([event['provider_event_id'], quote['source_id'], observed.isoformat()])[:24]}")
            if quote_id in have:
                continue
            try:
                evidence.insert_quote(path, {
                    "quote_id": quote_id, "event_id": event["event_id"], "sport": sport,
                    "market_family": family, "selection": quote["selection"], "line": quote["line"],
                    "american_odds": quote["american_odds"], "decimal_odds": quote["decimal_odds"],
                    "sportsbook": quote["sportsbook"], "quote_timestamp": quote["quote_timestamp"],
                    "quote_source": quote["quote_source"], "quote_verified": True, "source_id": quote["source_id"],
                    "raw_source": evidence.quote_source(row, quote["raw_source"], observed.isoformat())})
            except evidence.EvidenceConflict:
                raise
            except ValueError:
                blocked += 1
                continue
            have.add(quote_id)
            stored += 1
    return stored, blocked


def decision_quotes(events, quotes, now, *, predicted=None, horizon=HORIZON, max_age=MAX_QUOTE_AGE, books=None):
    """{(event_id, selection): newest fresh verified quote} for games starting within ``horizon``.

    Both sides of each game, any bettable book, quote at most ``max_age`` old and not after
    ``now``; half-point or whole lines only (the research settlement contract). With
    ``predicted``, only games in it: games that already have their research prediction. With
    ``books`` (canonical labels: the plan's frozen ``entry_books``, go-live 2026-11-08), only
    quotes at those books, the books a pick can be staked at.
    """
    from app_core.public_quote_policy import canonical_book_label
    chosen = {}
    for quote in quotes:
        event = events.get(quote["event_id"])
        if (event is None or (predicted is not None and quote["event_id"] not in predicted) or quote["quote_verified"] != 1 or quote["line"] is None
                or quote["sportsbook"] in REFERENCE_BOOKS
                or (books is not None and canonical_book_label(quote["sportsbook"]) not in books)):
            continue
        start, stamp = _time(event["scheduled_start"]), _time(quote["quote_timestamp"])
        if not (now < start <= now + horizon and now - max_age <= stamp <= now):
            continue
        if abs(2 * quote["line"] - round(2 * quote["line"])) > 1e-8:
            continue
        if not evidence.verify_provider_offer(event, quote):
            continue
        key = (quote["event_id"], quote["selection"])
        held = chosen.get(key)
        if held is None or (quote["quote_timestamp"], quote["decimal_odds"], quote["quote_id"]) > (
                held["quote_timestamp"], held["decimal_odds"], held["quote_id"]):
            chosen[key] = quote
    return chosen


def predict(path, sport, family, *, now):
    """Bound predictions against the fresh decision quotes, with the plan's exact model and calibration."""
    from app_core import prospective_calibration as calibration
    plan = calibration.bound_plan(path, sport, family)
    if plan is None:
        return {"predictions": 0, "blocker": "NO_BOUND_PLAN"}
    model = evidence.load_record(path, "prospective_model", plan["model_id"])
    cal = evidence.load_record(path, "prospective_calibration", plan["calibration_id"])
    # Validation scores each game's *first* prediction, so a decision prediction is made only
    # for games that already have one from this exact model and calibration.
    predicted = {p["event_id"] for p in evidence.read_records(path, "prospective_prediction", sport=sport,
                                                               market_family=family)
                 if p["model_id"] == plan["model_id"] and p["calibration_id"] == plan["calibration_id"]
                 and _time(p["prediction_timestamp"]) <= now}
    # Only the books the plan froze as its entry books (the books a pick can be staked at).
    books = frozenset(calibration._plan_rules(plan).get("entry_books") or ()) or None
    result = calibration._predict(path, sport, family, model, cal, now=now,
                                  policy_version=calibration.POLICY_VERSION, tag="DECISION",
                                  select=lambda events, quotes, at: decision_quotes(events, quotes, at,
                                                                                    predicted=predicted,
                                                                                    books=books))
    return dict(result, validation_plan_id=plan["validation_plan_id"])


def _plan_uses_pinnacle(path, sport, family):
    from app_core import pinnacle_model
    from app_core import prospective_calibration as calibration
    plan = calibration.bound_plan(path, sport, family)
    model = plan and evidence.load_record(path, "prospective_model", plan["model_id"])
    return bool(model) and pinnacle_model.is_pinnacle(model)


def run_once(path, *, key, request_get, budget, setting=None, assess=None, clock=_utcnow):
    """One cycle over every owner-switched market that is also qualified."""
    switched = markets(setting)
    report = {"switched_markets": sorted(" ".join(m) for m in switched), "markets": {}}
    if not switched:
        report["status"] = "OFF"
        return report
    if assess is None:
        from app_core.selection import qualification
        qualified = qualification.qualified_set(qualification.assess(path))
    else:
        qualified = frozenset(assess(path))
    for sport, family in sorted(switched):
        name = f"{sport} {family}"
        if (sport, family) not in qualified:
            report["markets"][name] = {"blocker": "NOT_QUALIFIED"}
            continue
        entry = report["markets"][name] = {"capture": capture(path, sport, family, key=key, request_get=request_get,
                                                              budget=budget, clock=clock,
                                                              pinnacle=_plan_uses_pinnacle(path, sport, family))}
        entry["predictions"] = predict(path, sport, family, now=clock())  # after the quotes it uses
    report["status"] = "RAN"
    return report


STATE_PREFIX = "parlaypicker/decision-capture-v1/state/"


def load_state(client, folder, *, prefix=STATE_PREFIX):
    """The loop's own credit ledger, newest object under ``prefix`` (``{}`` when none)."""
    import hashlib
    import json
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder, Prefix=prefix)
            for o in page.get("Contents", [])]
    if not keys:
        return {}
    key = max(keys)
    with client.get_object(Bucket=folder, Key=key)["Body"] as body:
        raw = body.read(1_000_001)
    if len(raw) > 1_000_000 or not key.endswith(hashlib.sha256(raw).hexdigest() + ".json"):
        raise ValueError("decision_state_integrity")
    return json.loads(raw)


def save_state(client, folder, state, *, clock=_utcnow, prefix=STATE_PREFIX):
    """Write a new immutable state object and read it back before any provider call."""
    import hashlib
    import json
    raw = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    key = prefix + clock().strftime("%Y%m%dT%H%M%S%f") + "-" + hashlib.sha256(raw).hexdigest() + ".json"
    client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    with client.get_object(Bucket=folder, Key=key)["Body"] as body:
        if body.read() != raw:
            raise ValueError("decision_state_readback")


def budget_for(state, persist, *, clock=_utcnow):
    """A ``Budget`` whose only Odds allowance is ``DECISION_LIMITS``, over the loop's own state."""
    from app_core.research_api_budget import Budget
    return Budget(state, persist, clock=clock, limits={"ODDS": DECISION_LIMITS})
