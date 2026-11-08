"""Record the NBA's official injury reports for completed ET days, write-once to Drive.

Research only (owner 2026-11-08, option 3): no Odds API credits, no picks. Each run records
every completed day from ``--start`` (default: three days ago) through yesterday that is not
recorded yet, so a skipped scheduled run is caught up by the next one. The league keeps this
season's 15-minute reports at fixed URLs (``app_core.nba_injury_reports.slot_url``).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_core import nba_injury_reports as reports  # noqa: E402


def _fetch(url):
    import requests
    response = requests.get(url, timeout=30, headers={"User-Agent": "ParlayPicker research recorder"})
    return response.status_code, (response.content if response.status_code == 200 else b"")


def _drive():
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    folder = settings()[0]
    if not folder or not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip():
        raise SystemExit("MISSING_CREDENTIALS: PARLAYPICKER_DRIVE_FOLDER_ID / PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT")
    return DriveStore(folder)


def run(client, *, start, end, now, fetch=_fetch):
    results, day = [], start
    while day <= end:
        try:
            results.append(reports.record_day(client, day, fetch, now=now))
        except Exception as exc:  # one bad day never blocks the others
            results.append({"day": day.isoformat(), "status": "ERROR", "error": type(exc).__name__})
        day += timedelta(days=1)
    return results


def markdown(results):
    lines = ["## NBA injury reports (research)", "", "| day | status | reports | changes | material |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['day']} | {r['status']} | {r.get('reports', '')} | {r.get('changes', '')} | "
                     f"{r.get('material_changes', '')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", help="first ET day (YYYY-MM-DD); default three days ago")
    parser.add_argument("--end", help="last ET day; default yesterday")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    today = now.astimezone(reports.ET).date()
    start = date.fromisoformat(args.start) if args.start else today - timedelta(days=3)
    end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    results = run(_drive(), start=start, end=end, now=now)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    text = markdown(results)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 1 if any(r["status"] == "ERROR" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
