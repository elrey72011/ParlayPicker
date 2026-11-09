"""Fixed-interval price watch for games in their last hours (owner D28, 2026-11-09; roadmap M2).

``docs/roadmap/VALIDATION-ROADMAP-2026-11-09.md``. Research capture prices each game once, hours
before the start. Two things need prices near the start instead:

* the Pinnacle/Novig lag (a stake book still at its old price after Pinnacle moved) lasts about
  10 to 30 minutes, with or without injury news, so one early price never sees it;
* an entry 30 to 60 minutes before the start gets a certified close 90% of the time; one taken
  more than 6 hours ahead gets it 42% of the time, and a plan needs 80%.

Through a slate (``scripts/price_watch_loop.py``), every ``INTERVAL_MINUTES[sport]``: one Odds API
request for the stored games of that sport starting within ``WINDOW`` (US books, Pinnacle and
Novig where switched on; spreads and totals; ``news_capture.capture``). Where a frozen plan binds
the market it predicts at the fresh quotes; otherwise the quotes feed the replay
(``pinnacle_model.replay_estimate``).

Off unless ``PARLAYPICKER_PRICE_WATCH`` names the sport, and off for good from ``UNTIL`` (the owner
approved 7 days, then a review). Credits come from their own allowance (``WATCH_LIMITS``: 200 a
day, 1,400 in all) in a separate ledger on the evidence remote. It grants no authority.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from app_core import decision_capture as decision
from app_core import news_capture as news

WATCH_ENV = "PARLAYPICKER_PRICE_WATCH"
SPORTS = ("NBA", "NCAAB")
WINDOW = timedelta(hours=3)
INTERVAL_MINUTES = {"NBA": 15, "NCAAB": 30}     # NCAAB slates run twice as long; both stay near 100 credits a day
MAX_EVENTS = 100                                # one request whatever the number of games; soonest first
WATCH_LIMITS = {"daily": 200, "rolling_31_days": 1400}
UNTIL = datetime(2026, 11, 17, 7, 0, tzinfo=timezone.utc)   # 2 AM ET: the night of 2026-11-16 is the last
STATE_PREFIX = "parlaypicker/price-watch-v1/state/"
COST_FLAG = "price_watch_cost_review_required"


def _utcnow():
    return datetime.now(timezone.utc)


def sports(setting=None, *, now=None):
    """The sports switched on (``PARLAYPICKER_PRICE_WATCH``), none from ``UNTIL``."""
    if (now or _utcnow()) >= UNTIL:
        return ()
    raw = (setting or os.environ.get)(WATCH_ENV, "") or ""
    named = {s.strip().upper() for s in raw.split(",")}
    return tuple(s for s in SPORTS if s in named)


def enabled(setting=None, *, now=None) -> bool:
    return bool(sports(setting, now=now))


class Watcher:
    """The loop's memory between polls: when each sport is next due."""

    def __init__(self, setting=None):
        self.setting = setting
        self.due = {}

    def poll(self, path, *, key, request_get, budget, clock=_utcnow, fetch=None):
        now = clock()
        report = {"at": now.isoformat(), "news": [], "capture": None, "sports": {}}
        for sport in sports(self.setting, now=now):
            if self.due.get(sport, now) > now:
                continue
            events = sorted(news.due_events(path, now, sport=sport, horizon=WINDOW),
                            key=lambda e: (e["scheduled_start"], e["provider_event_id"]))[:MAX_EVENTS]
            if not events:
                continue                 # no game in its window: ask again at the next poll, at no cost
            self.due[sport] = now + timedelta(minutes=INTERVAL_MINUTES[sport])
            result = news.capture(path, {e["provider_event_id"]: e for e in events}, key=key, request_get=request_get,
                                  budget=budget, clock=clock, sport=sport, flag=COST_FLAG)
            if result["stored_quotes"]:
                result["predictions"] = news.predict(path, clock(), sport=sport)
            report["sports"][sport] = result
        if report["sports"]:
            results = report["sports"].values()
            report["capture"] = {"requested": any(r["requested"] for r in results),
                                 "stored_quotes": sum(r["stored_quotes"] for r in results)}
        return report


def load_state(client, folder):
    """The watch's own credit ledger (newest object under ``STATE_PREFIX``)."""
    return decision.load_state(client, folder, prefix=STATE_PREFIX)


def save_state(client, folder, state, *, clock=_utcnow):
    return decision.save_state(client, folder, state, clock=clock, prefix=STATE_PREFIX)


def budget_for(state, persist, *, clock=_utcnow):
    from app_core.research_api_budget import Budget
    return Budget(state, persist, clock=clock, limits={"ODDS": WATCH_LIMITS})
