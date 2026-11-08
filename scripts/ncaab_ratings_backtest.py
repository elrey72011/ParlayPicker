"""NCAAB opponent-adjusted ratings vs archived opening and closing lines (CBBD, research only).

    python scripts/ncaab_ratings_backtest.py --output OUT.json [--scored-season 2025]

Seasons: scored-2 sets the priors, scored-1 calibrates, scored is scored. About 45 CBBD calls
(games for three seasons, lines for two, in date windows). Needs CBBD_API_KEY (falls back to
CFBD_API_KEY: the same account serves both). No Odds API calls; writes nothing to Drive or the store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_line_backtest as data  # noqa: E402
from app_core import ncaab_ratings_backtest as bt  # noqa: E402


def run(token, scored_season, get=None):
    if get is None:
        import requests
        get = requests.get
    prior_season, calibration_season = scored_season - 2, scored_season - 1
    games = {s: data.parse_games(data.fetch("games", s, token, get))
             for s in (prior_season, calibration_season, scored_season)}
    lines = {s: bt.select_lines(data.fetch("lines", s, token, get)) for s in (calibration_season, scored_season)}
    scored_lines = lines[scored_season]
    return {"prior_season": prior_season, "calibration_season": calibration_season, "scored_season": scored_season,
            "games": {str(s): len(g) for s, g in games.items()},
            "closing_lined": sum(1 for v in scored_lines.values() if v["spread"] is not None),
            "opening_lined": sum(1 for v in scored_lines.values() if v["spread_open"] is not None),
            "providers": sorted({v["provider"] for v in scored_lines.values() if v["provider"]}),
            "report": bt.run(games[prior_season], games[calibration_season], games[scored_season],
                             lines[calibration_season], scored_lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-season", type=int, default=2025)
    args = parser.parse_args(argv)
    token = (os.getenv("CBBD_API_KEY", "") or os.getenv("CFBD_API_KEY", "")).strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CBBD_API_KEY or CFBD_API_KEY")
    result = run(token, args.scored_season)
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
