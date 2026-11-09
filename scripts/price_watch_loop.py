"""Fixed-interval price watch through a slate (``app_core.price_watch``; owner D28, 2026-11-09).

    python scripts/price_watch_loop.py --minutes 340 --interval 5

Off unless ``PARLAYPICKER_PRICE_WATCH`` names NBA or NCAAB, and off from ``price_watch.UNTIL``.
The same loop as ``scripts/news_capture_loop.py`` (restore, poll, back up after stored quotes),
with ``price_watch`` deciding what to capture: every 15 minutes (NBA) or 30 (NCAAB), the stored
games starting within 3 hours. Credits come from the watch's own allowance (200 a day, 1,400 in all).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--root", default="output/price-watch")
    args = parser.parse_args(argv)
    from app_core import price_watch
    if not price_watch.enabled():
        print(json.dumps({"status": "OFF", "detail": f"{price_watch.WATCH_ENV} names no sport, or the watch has ended"}))
        return 0
    import requests
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    from scripts import news_capture_loop as loop
    folder, _ = settings()
    result = loop.run(args.minutes, args.interval, root=Path(args.root), client=DriveStore(folder), folder=folder,
                      key=os.getenv("ODDS_API_KEY"), request_get=requests.get,
                      wall=lambda: datetime.now(timezone.utc), source=price_watch)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
