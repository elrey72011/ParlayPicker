"""News-triggered NBA price capture (owner 2026-11-09, step 3 of the news-edge plan).

``docs/roadmap/NEWS-EDGE-PLAN-2026-11-08.md``. The lag study found that after injury news
Pinnacle moves first, and a stake book (almost always Novig) is still 1% or more better than
Pinnacle's new price in 11 of 29 cases (7 of 51 in the control). The gaps last about 10 to 30
minutes. Research capture prices each game once, hours before the start, so it never sees them.

Through a slate, every ``POLL_MINUTES`` (``scripts/news_capture_loop.py``):

1. **News** (no credits): read the newest official injury report (``nba_injury_reports``) and
   diff it with the previous one. A material change is a player moving into or out of
   Out/Doubtful, or a first listing as Out/Doubtful within 3 hours of tip-off, at least
   10 minutes before the start (the lag study's ``candidates``).
2. **Capture**: one Odds API request (``capture_bookmakers``: US books, Pinnacle and Novig where
   switched on; spreads and totals: ``CREDITS_PER_REQUEST`` credits) for the stored games with
   news, and again ``RECHECK_MINUTES`` later. Each quote keeps its own provider response
   (``decision_capture.store_quotes``), so it replays like any captured quote.
3. **Predict**: where a frozen plan binds a market (``prospective_calibration.predict_bound``),
   it predicts at the fresh quote at once, so a ``FIRST_EDGE_CAPTURE`` plan can enter while the
   gap is open. With no bound plan the quotes still feed the replay (``pinnacle_model.replay_estimate``).

Off unless ``PARLAYPICKER_NEWS_CAPTURE`` names NBA. Credits come from their own allowance
(``NEWS_LIMITS``, owner 2026-11-09: 100 a day) in a separate ledger on the evidence remote. It
grants no authority: picks still need a frozen plan, review, activation and the gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from app_core import decision_capture as decision
from app_core import nba_injury_reports as reports
from app_core import news_lag_study as lag
from app_core import prospective_evidence as evidence

NEWS_ENV = "PARLAYPICKER_NEWS_CAPTURE"
SPORT = "NBA"
FAMILIES = ("SPREAD", "TOTAL")
NEWS_LIMITS = {"daily": 100, "rolling_31_days": 2500}
CREDITS_PER_REQUEST = 4        # 13 books = 2 region equivalents, times 2 markets
POLL_MINUTES = 5
RECHECK_MINUTES = 10
LOOKBACK_SLOTS = 4             # the newest report is looked for in the last hour of slots
HORIZON = timedelta(hours=12)  # stored games starting within this of now
STATE_PREFIX = "parlaypicker/news-capture-v1/state/"
COST_FLAG = "news_cost_review_required"


def enabled(setting=None) -> bool:
    raw = (setting or os.environ.get)(NEWS_ENV, "") or ""
    return SPORT in {s.strip().upper() for s in raw.split(",")}


def _utcnow():
    return datetime.now(timezone.utc)


def newest_report(fetch, now, *, after=None):
    """(slot, rows) of the newest published report at or before ``now`` and after ``after``, or None."""
    local = now.astimezone(reports.ET)
    slot = local.replace(minute=local.minute - local.minute % 15, second=0, microsecond=0)
    for back in range(LOOKBACK_SLOTS):
        candidate = slot - timedelta(minutes=15 * back)
        if after is not None and candidate <= after:
            return None
        try:
            status, body = fetch(reports.slot_url(candidate))
        except Exception:
            continue
        if status == 200 and body:
            try:
                return candidate, reports.parse(body)
            except Exception:
                continue
    return None


def news(slot, previous_rows, rows):
    """Material news groups ({slot, matchup, changes, ...}) between two consecutive reports."""
    entry = {"slot": slot.isoformat(), "rows": rows, "changes": reports.changes(previous_rows, rows)}
    return lag.candidates([{"reports": [entry]}])


def due_events(path, now, *, sport=SPORT, horizon=HORIZON):
    """Stored Odds API events of ``sport`` starting within ``horizon`` of ``now``."""
    return [e for e in evidence.read_records(path, "prospective_event", sport=sport)
            if e["provider_namespace"] == "THE_ODDS_API"
            and now < evidence._time(e["scheduled_start"], "scheduled_start") <= now + horizon]


def capture(path, events, *, key, request_get, budget, clock=_utcnow, sport=SPORT, flag=COST_FLAG):
    """One request for ``events`` ({provider id: event}); store spreads and totals, Pinnacle included."""
    from app_core import odds_research_adapter as adapter
    from app_core.research_api_budget import BudgetLimit, capture_bookmakers
    result = {"events": len(events), "requested": False, "stored_quotes": 0, "replay_blocked": 0, "blocker": None}
    if not events:
        return result
    if budget.state.get(flag):
        result["blocker"] = flag  # sticky until the owner reviews the provider's charge
        return result
    try:
        budget.reserve("ODDS", CREDITS_PER_REQUEST)
    except BudgetLimit as exc:
        result["blocker"] = str(exc)
        return result
    result["requested"] = True
    charged_get = decision.charging(request_get, budget, CREDITS_PER_REQUEST, flag, result)
    try:
        payload = adapter.fetch(sport, "odds", key, charged_get, params={
            "bookmakers": capture_bookmakers(sport), "markets": "spreads,totals", "oddsFormat": "american",
            "eventIds": ",".join(sorted(events))})
    except BudgetLimit as exc:  # a retry the allowance could not cover
        result["blocker"] = str(exc)
        return result
    for family in FAMILIES:
        stored, blocked = decision.store_quotes(path, sport, family, events, payload, clock=clock, reference=True)
        result["stored_quotes"] += stored
        result["replay_blocked"] += blocked
    return result


def predict(path, now, *, sport=SPORT):
    """Bound predictions at the fresh quotes, where a frozen plan binds the market."""
    from app_core import prospective_calibration as calibration
    out = {}
    for family in FAMILIES:
        try:
            out[family] = calibration.predict_bound(path, sport, family, now=now)
        except Exception as exc:  # a prediction failure never loses the stored quotes
            out[family] = {"error": type(exc).__name__}
    return out


class Watcher:
    """The loop's memory between polls: the last report read and the rechecks still due."""

    def __init__(self):
        self.slot = None
        self.rows = None
        self.rechecks = {}            # provider event id -> due time (UTC)

    def poll(self, path, *, fetch, key, request_get, budget, clock=_utcnow):
        now = clock()
        report = {"at": now.isoformat(), "news": [], "capture": None}
        found = newest_report(fetch, now, after=self.slot)
        if found is not None:
            slot, rows = found
            if self.rows is not None:  # the first report read is the baseline, not news
                report["news"] = news(slot, self.rows, rows)
            self.slot, self.rows = slot, rows
            report["slot"] = slot.isoformat()
        events = due_events(path, now)
        wanted = {}
        for group in report["news"]:
            event = lag.match_event(events, group["matchup"])
            group["event_id"] = event and event["provider_event_id"]
            if event is not None:
                wanted[event["provider_event_id"]] = event
                self.rechecks[event["provider_event_id"]] = now + timedelta(minutes=RECHECK_MINUTES)
        by_id = {e["provider_event_id"]: e for e in events}
        for event_id, at in list(self.rechecks.items()):
            if event_id not in by_id:
                del self.rechecks[event_id]          # started, or no longer stored
            elif at <= now and event_id not in wanted:
                wanted[event_id] = by_id[event_id]
                del self.rechecks[event_id]
        if wanted:
            report["capture"] = capture(path, wanted, key=key, request_get=request_get, budget=budget, clock=clock)
            if report["capture"]["stored_quotes"]:
                report["predictions"] = predict(path, clock())
        return report


def load_state(client, folder):
    """The loop's own credit ledger (newest object under ``STATE_PREFIX``)."""
    return decision.load_state(client, folder, prefix=STATE_PREFIX)


def save_state(client, folder, state, *, clock=_utcnow):
    return decision.save_state(client, folder, state, clock=clock, prefix=STATE_PREFIX)


def budget_for(state, persist, *, clock=_utcnow):
    from app_core.research_api_budget import Budget
    return Budget(state, persist, clock=clock, limits={"ODDS": NEWS_LIMITS})
