"""One-time study: do the stake books lag Pinnacle after NBA injury news? (owner 2026-11-08, 5,000 credits)

Reads the archived injury reports (``NBA injury reports`` workflow) for ``--start``..``--end``,
claims the label's write-once start marker on Drive (a label runs once), spends at most
``--max-credits`` on Odds API historical snapshots, and writes every raw response and the
analysis to Drive and the run artifact. See ``app_core.news_lag_study``.
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
from app_core import news_lag_study as study  # noqa: E402

OWNER_BUDGET = 5000


def _get(url, params):
    import requests
    response = requests.get(url, params=params, timeout=30)
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, response.headers, body


def manifests(client, start, end):
    found, missing, day = [], [], start
    while day <= end:
        try:
            found.append(json.loads(client.get_object(Key=reports.day_key(day))["Body"].read()))
        except Exception:
            missing.append(day.isoformat())
        day += timedelta(days=1)
    return found, missing


def markdown(result):
    s = result.get("summary") or {}
    lines = ["## News lag study (research)", "",
             f"- label `{result.get('label')}`, credits spent {result.get('spent')} of {result.get('max_credits')}"
             f" (stopped: {result.get('stopped') or 'no'})",
             f"- report days missing from the archive: {', '.join(result.get('missing_days') or []) or 'none'}",
             f"- news studied: {s.get('news_studied')}, with a Pinnacle move (>= 0.5 pt or 2 points of probability): "
             f"{s.get('pinnacle_moved')}"]
    for key, label in (("after_news_1pct", "after news, Pinnacle moved: a stake book >= 1% better"),
                       ("after_news_2pct", "after news, Pinnacle moved: a stake book >= 2% better"),
                       ("all_after_1pct", "after any news: >= 1%"), ("control_before_1pct", "30 min before (control): >= 1%")):
        row = s.get(key) or {}
        lines.append(f"- {label}: {row.get('at_least')} of {row.get('with_price')}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--label", default="v1")
    parser.add_argument("--max-credits", type=int, default=OWNER_BUDGET)
    parser.add_argument("--min-remaining", type=int, default=4000,
                        help="stop when the account's remaining monthly credits fall below this")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_credits > OWNER_BUDGET:
        raise SystemExit(f"max credits above the owner's budget ({OWNER_BUDGET})")
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("MISSING_CREDENTIALS: ODDS_API_KEY")
    from record_nba_injury_reports import _drive
    client = _drive()
    found, missing = manifests(client, date.fromisoformat(args.start), date.fromisoformat(args.end))
    groups = study.candidates(found)
    now = datetime.now(timezone.utc)
    result = {"label": args.label, "max_credits": args.max_credits, "started_at": now.isoformat(),
              "missing_days": missing, "candidates": len(groups)}
    if not groups:
        result.update(status="NO_NEWS", spent=0)
    elif not study.claim(client, args.label, max_credits=args.max_credits, now=now):
        raise SystemExit(f"label {args.label} already ran: its budget is spent (use a new label only with owner approval)")
    else:
        result.update(study.run(groups, _get, api_key=key, max_credits=args.max_credits,
                                min_remaining=args.min_remaining, log=print), status="DONE")
        body = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode()
        client.put_object(Key=study.result_key(args.label), Body=body, IfNoneMatch="*")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    text = markdown(result)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
