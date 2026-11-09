"""NCAAF ratings and CFBD Elo vs archived opening and closing lines (research only).

    python scripts/ncaaf_ratings_backtest.py --output OUT.json [--first-season 2021] [--last-season 2026]

Ratings start at ``--first-season`` (that season only builds them), the next three seasons calibrate,
and every later season up to ``--last-season`` is scored (the current season to date). About 18
CFBD calls (games, and regular and postseason lines, per season). Needs CFBD_API_KEY. No Odds API
calls; writes nothing to Drive or the store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaaf_ratings_backtest as bt  # noqa: E402

CALIBRATION_SEASONS = 3


def run(token, first_season, last_season, get=None):
    if get is None:
        import requests
        get = requests.get
    years = list(range(first_season, last_season + 1))
    calibration_years = years[1:1 + CALIBRATION_SEASONS]
    scored_years = years[1 + CALIBRATION_SEASONS:]
    if not scored_years:
        raise SystemExit("need at least five seasons: one to build ratings, three to calibrate, one to score")
    seasons = {y: bt.parse_games(bt.fetch("games", y, token, get)) for y in years}
    lines = {y: bt.select_lines(bt.fetch("lines", y, token, get)) for y in years[1:]}
    return {"seasons": years, "calibration_years": calibration_years, "scored_years": scored_years,
            "games": {str(y): len(g) for y, g in seasons.items()},
            "lined": {str(y): f"{sum(1 for v in l.values() if v['spread'] is not None)} / "
                              f"{sum(1 for v in l.values() if v['spread_open'] is not None)}"
                      for y, l in lines.items()},
            "providers": sorted({v["provider"] for l in lines.values() for v in l.values() if v["provider"]}),
            "report": bt.run(seasons, lines, calibration_years, scored_years)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--last-season", type=int, default=2026)
    args = parser.parse_args(argv)
    token = os.getenv("CFBD_API_KEY", "").strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CFBD_API_KEY")
    result = run(token, args.first_season, args.last_season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = bt.markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
