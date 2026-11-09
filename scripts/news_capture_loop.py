"""News-triggered NBA price capture through a slate (``app_core.news_capture``).

    python scripts/news_capture_loop.py --minutes 340 --interval 5

Off unless ``PARLAYPICKER_NEWS_CAPTURE`` names NBA. Restores the canonical evidence store once,
then every ``--interval`` minutes reads the newest official injury report and, for games with
material news, captures prices at once and again 10 minutes later. Backs the store up after any
poll that stored a quote. Credits come from the loop's own allowance (``NEWS_LIMITS``, 100 a day).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fetch_report(url):
    import requests
    response = requests.get(url, timeout=30, headers={"User-Agent": "ParlayPicker research recorder"})
    return response.status_code, (response.content if response.status_code == 200 else b"")


def run(minutes, interval, *, root, client, folder, key, request_get, fetch=fetch_report, setting=None, sync=None,
        clock=time.monotonic, sleep=time.sleep, watcher=None):
    from app_core import news_capture as news
    if not news.enabled(setting):
        return {"status": "OFF", "polls": 0}
    if sync is None:
        from app_core.prospective_remote import sync
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prospective-evidence.sqlite3"
    session = {}
    sync(path, client, folder, session)  # restore before anything is read or written
    state = news.load_state(client, folder)
    budget = news.budget_for(state, lambda: news.save_state(client, folder, state))
    watcher = watcher or news.Watcher()
    deadline = clock() + minutes * 60
    totals = {"status": "RAN", "polls": 0, "news": 0, "captures": 0, "stored_quotes": 0, "backups": 0, "errors": 0}
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
        if capture.get("stored_quotes"):
            sync(path, client, folder, session)
            totals["backups"] += 1
        if clock() + interval * 60 > deadline:
            return totals
        sleep(interval * 60)


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
                 key=os.getenv("ODDS_API_KEY"), request_get=requests.get)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
