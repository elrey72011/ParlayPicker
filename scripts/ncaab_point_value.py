"""Measure the value of a point of line in college basketball from archived lines (research only).

    python scripts/ncaab_point_value.py --output OUT.json [--seasons 2025,2026]

Needs CBBD_API_KEY (falls back to CFBD_API_KEY: the same account serves both APIs). About 15 CBBD
calls per season and endpoint (games and lines, in date windows). No Odds API calls; writes nothing
to Drive or the evidence store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_line_backtest as bt  # noqa: E402
from app_core import ncaab_point_value as pv  # noqa: E402


def run(token, seasons, get=None):
    if get is None:
        import requests
        get = requests.get
    games, lines = [], {}
    for season in seasons:
        games += bt.parse_games(bt.fetch("games", season, token, get))
        lines.update(bt.select_lines(bt.fetch("lines", season, token, get)))
    return {"seasons": list(seasons), "games": len(games), "lined": sum(1 for g in games if g["id"] in lines),
            "providers": sorted({v["provider"] for v in lines.values() if v["provider"]}),
            "report": pv.measure(games, lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasons", default="2025,2026",
                        help="CBBD season numbers, comma-separated (2026 = 2025-26)")
    args = parser.parse_args(argv)
    token = (os.getenv("CBBD_API_KEY", "") or os.getenv("CFBD_API_KEY", "")).strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CBBD_API_KEY or CFBD_API_KEY")
    seasons = [int(part) for part in args.seasons.split(",") if part.strip()]
    if not seasons or len(seasons) > 4:
        raise SystemExit("give one to four seasons")
    result = run(token, seasons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = pv.markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
