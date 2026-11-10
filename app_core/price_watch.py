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

**NHL** (owner 2026-11-09: widen the lag pool) is an extra sport with its own switch,
``PARLAYPICKER_PRICE_WATCH_EXTRA_SPORTS``: puck lines and totals every 15 minutes. Each extra sport
adds ``EXTRA_LIMITS`` to the allowance (100 a day, 700 in all; three times that in fast mode), so
the two-sport allowance the owner approved first (D28) is unchanged when NHL is off.

**Fast mode** (prepared 2026-11-09, off; turning it on is a new spend decision for the owner).
The gaps last about 10 to 30 minutes, so a 15-minute poll can miss one entirely. With
``PARLAYPICKER_PRICE_WATCH_FAST=true`` both sports are polled every ``FAST_INTERVAL_MINUTES`` and the
allowance is ``FAST_LIMITS`` (600 a day, 4,200 in all): about three times the credits. ``UNTIL``
is unchanged, and the same ledger counts both modes.

**Automatic pace** (prepared 2026-11-10, off). With ``PARLAYPICKER_PRICE_WATCH_AUTO_PACE=true`` each
sport's interval is the slowest one that kept 85% of the bets the replay found over the last
``AUTO_LOOKBACK_DAYS`` days (``pinnacle_model.recommended_pace``), where that is slower than the
interval above. It can only slow a sport down, so it never spends more than the mode it runs in,
and it changes nothing when the stored captures are too sparse or too few bets were found.
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
EXTRA_ENV = "PARLAYPICKER_PRICE_WATCH_EXTRA_SPORTS"
EXTRA_SPORTS = ("NHL",)
EXTRA_INTERVAL_MINUTES = 15
EXTRA_LIMITS = {"daily": 100, "rolling_31_days": 700}        # per extra sport switched on
FAST_ENV = "PARLAYPICKER_PRICE_WATCH_FAST"
FAST_INTERVAL_MINUTES = 5                       # the loop's own poll interval: every poll captures
FAST_LIMITS = {"daily": 600, "rolling_31_days": 4200}
AUTO_ENV = "PARLAYPICKER_PRICE_WATCH_AUTO_PACE"
AUTO_LOOKBACK_DAYS = 7
WATCH_START = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)   # games before the watch have one capture each
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
    return tuple(s for s in SPORTS if s in named) + extra_sports(setting)


def extra_sports(setting=None):
    """The extra sports switched on (``PARLAYPICKER_PRICE_WATCH_EXTRA_SPORTS``, e.g. "NHL")."""
    raw = (setting or os.environ.get)(EXTRA_ENV, "") or ""
    named = {s.strip().upper() for s in raw.split(",")}
    return tuple(s for s in EXTRA_SPORTS if s in named)


def enabled(setting=None, *, now=None) -> bool:
    return bool(sports(setting, now=now))


def fast(setting=None) -> bool:
    """Fast mode is on only for the exact value ``true`` (any case); anything else is the normal watch."""
    return str((setting or os.environ.get)(FAST_ENV, "") or "").strip().lower() == "true"


def interval_minutes(sport, setting=None) -> int:
    return FAST_INTERVAL_MINUTES if fast(setting) else INTERVAL_MINUTES.get(sport, EXTRA_INTERVAL_MINUTES)


def auto_pace(setting=None) -> bool:
    """Automatic pace is on only for the exact value ``true`` (any case)."""
    return str((setting or os.environ.get)(AUTO_ENV, "") or "").strip().lower() == "true"


def recommended(path, sport, now):
    """``pinnacle_model.recommended_pace`` for ``sport`` over the last ``AUTO_LOOKBACK_DAYS`` days of started
    games, none from before the watch began."""
    from app_core import pinnacle_model
    days = min(float(AUTO_LOOKBACK_DAYS), max((now - WATCH_START) / timedelta(days=1), 0.0))
    return pinnacle_model.recommended_pace([
        pinnacle_model.pace_comparison(path, sport, family, now=now, lookback_days=days,
                                       books=pinnacle_model.PLAN_BOOKS) for family in news.families(sport)])


def limits(setting=None) -> dict:
    """The allowance: the two-sport base (normal or fast), plus ``EXTRA_LIMITS`` per extra sport on."""
    base = dict(FAST_LIMITS if fast(setting) else WATCH_LIMITS)
    scale = len(extra_sports(setting)) * (FAST_LIMITS["daily"] // WATCH_LIMITS["daily"] if fast(setting) else 1)
    return {window: base[window] + scale * EXTRA_LIMITS[window] for window in base}


class Watcher:
    """The loop's memory between polls: when each sport is next due."""

    def __init__(self, setting=None):
        self.setting = setting
        self.due = {}
        self.paces = {}

    def interval(self, path, sport, now) -> int:
        """Minutes to the sport's next capture: the mode's interval, or a slower recommended one."""
        base = interval_minutes(sport, self.setting)
        if not auto_pace(self.setting):
            return base
        if sport not in self.paces:                     # once a run: the replay reads the whole scope
            try:
                self.paces[sport] = recommended(path, sport, now)
            except Exception as exc:                    # a recommendation must never stop the watch
                self.paces[sport] = {"status": f"ERROR:{type(exc).__name__}", "minutes": None}
        return max(base, self.paces[sport].get("minutes") or base)

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
            minutes = self.interval(path, sport, now)
            self.due[sport] = now + timedelta(minutes=minutes)
            result = news.capture(path, {e["provider_event_id"]: e for e in events}, key=key, request_get=request_get,
                                  budget=budget, clock=clock, sport=sport, flag=COST_FLAG)
            if result["stored_quotes"]:
                result["predictions"] = news.predict(path, clock(), sport=sport)
            if sport in self.paces:
                result["pace"] = {"interval_minutes": minutes, "status": self.paces[sport].get("status")}
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


def budget_for(state, persist, *, clock=_utcnow, setting=None):
    from app_core.research_api_budget import Budget
    return Budget(state, persist, clock=clock, limits={"ODDS": limits(setting)})


def after_poll(path, client, folder, report, *, setting=None, clock=_utcnow):
    """After a poll that stored prices: publish the live price-gap candidates for the app (research only).

    Called by the shared loop before the Drive backup, which can take minutes. It never raises:
    a failure is returned as a status and the watch goes on.
    """
    if not (report.get("capture") or {}).get("stored_quotes"):
        return None
    try:
        from app_core import price_gap_candidates as gaps
        now = clock()
        watched = sports(setting, now=now)
        rows = gaps.current(path, now, sports=watched)
        return gaps.publish(client, folder, rows, now=now, sports=watched)
    except Exception as exc:
        return {"status": "CANDIDATES_FAILED", "error": type(exc).__name__}

