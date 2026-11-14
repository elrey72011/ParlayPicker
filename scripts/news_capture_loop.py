"""News-triggered NBA price capture through a slate (``app_core.news_capture``).

    python scripts/news_capture_loop.py --minutes 340 --interval 5

Off unless ``PARLAYPICKER_NEWS_CAPTURE`` names NBA. Restores the canonical evidence store once,
then every ``--interval`` minutes reads the newest official injury report and, for games with
material news, captures prices at once and again 10 minutes later. Backs the store up after any
poll that stored a quote. Credits come from the loop's own allowance (``NEWS_LIMITS``, 100 a day).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REFRESH_MINUTES = 60


def slate_over(now) -> bool:
    """2 AM to 8 AM Eastern: the night's games are over and the next slate has not started.

    The window ended at 11 AM until 2026-11-14, when a weekend start at 8:50 AM ET (to give noon
    games their three hours of captures) stopped after one poll and nothing ran until 12:07 PM.
    """
    from zoneinfo import ZoneInfo
    return 2 <= now.astimezone(ZoneInfo("America/New_York")).hour < 8


def fetch_report(url):
    import requests
    response = requests.get(url, timeout=30, headers={"User-Agent": "ParlayPicker research recorder"})
    return response.status_code, (response.content if response.status_code == 200 else b"")


WAKE_PAD_SECONDS = 3   # wake just after the interval, never a moment before a sport is due again


def poll_wait(interval, now, began):
    """Seconds to sleep so polls start ``interval`` minutes apart, not ``interval`` after the backup ends.

    Run 2026-11-10: with a 5-minute interval a capture was stored every 9 minutes, because the
    loop slept the whole interval after a backup that took 3 to 5. The wait is now the interval
    less the time the poll and its backup took, plus ``WAKE_PAD_SECONDS``. The first poll
    (``began`` unknown) waits the whole interval, as before; work longer than the interval waits
    only the pad.
    """
    if began is None:
        return interval * 60
    return max(0.0, interval * 60 - (now - began)) + WAKE_PAD_SECONDS


def run(minutes, interval, *, root, client, folder, key, request_get, fetch=fetch_report, setting=None, sync=None,
        clock=time.monotonic, sleep=time.sleep, watcher=None, wall=None, source=None):
    """``source``: the module that decides what to capture (``news_capture``, or ``price_watch``)."""
    if source is None:
        from app_core import news_capture as source
    news = source
    if not news.enabled(setting):
        return {"status": "OFF", "polls": 0}
    if sync is None:
        from app_core.prospective_remote import sync
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prospective-evidence.sqlite3"
    session = {}
    started = clock()  # the slate's minutes count from the start, restore included (audit 2026-11-09)
    deadline = started + minutes * 60
    last_restore = started
    sync(path, client, folder, session)  # restore before anything is read or written
    state = news.load_state(client, folder)
    budget = news.budget_for(state, lambda: news.save_state(client, folder, state))
    watcher = watcher or news.Watcher()
    totals = {"status": "RAN", "polls": 0, "news": 0, "captures": 0, "stored_quotes": 0, "backups": 0, "errors": 0}
    pending = False
    began = None   # when this poll began, on ``clock``: the last reading plus the sleep that followed it
    while True:
        totals["polls"] += 1
        try:
            report = watcher.poll(path, fetch=fetch, key=key, request_get=request_get, budget=budget)
        except Exception as exc:  # one failed poll must not end the slate
            totals["errors"] += 1
            report = {"error": type(exc).__name__}
        totals["news"] += len(report.get("news") or [])
        capture = report.get("capture") or {}
        totals["captures"] += int(bool(capture.get("requested")))
        totals["stored_quotes"] += capture.get("stored_quotes", 0)
        print(json.dumps({"poll": totals["polls"], **report}, default=str), flush=True)
        after_poll = getattr(news, "after_poll", None)
        if after_poll is not None:   # the price watch publishes its live candidates before the slow backup
            published = after_poll(path, client, folder, report, setting=setting)
            if published is not None:
                print(json.dumps({"poll": totals["polls"], "candidates": published}), flush=True)
        pending = pending or bool(capture.get("stored_quotes"))
        if pending:
            try:  # a failed backup is retried at the next poll; it never ends the slate
                sync(path, client, folder, session)
            except Exception as exc:
                totals["backup_errors"] = totals.get("backup_errors", 0) + 1
                print(json.dumps({"poll": totals["polls"], "backup_error": type(exc).__name__}), flush=True)
            else:
                pending = False
                totals["backups"] += 1
        now = clock()
        if now + interval * 60 > deadline:
            return totals
        if wall is not None and slate_over(wall()):
            totals["ended"] = "slate_over"   # a backup start that began late stops with the slate
            return totals
        if now - last_restore >= REFRESH_MINUTES * 60:
            # Games the research loop captured after this slate began are only in the remote
            # store: restore again with a fresh session, every REFRESH_MINUTES (audit 2026-11-09).
            try:
                session = {}
                sync(path, client, folder, session)
                last_restore = now
            except Exception as exc:
                print(json.dumps({"poll": totals["polls"], "restore_error": type(exc).__name__}), flush=True)
        wait = poll_wait(interval, now, began)
        began = now + wait
        sleep(wait)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--root", default="output/news-capture")
    args = parser.parse_args(argv)
    from app_core import news_capture as news
    if not news.enabled():
        print(json.dumps({"status": "OFF", "detail": f"{news.NEWS_ENV} does not name NBA"}))
        return 0
    import requests
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    folder, _ = settings()
    result = run(args.minutes, args.interval, root=Path(args.root), client=DriveStore(folder), folder=folder,
                 key=os.getenv("ODDS_API_KEY"), request_get=requests.get,
                 wall=lambda: datetime.now(timezone.utc))
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
