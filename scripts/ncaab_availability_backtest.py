"""NCAAB: does the closing line already price a missing key player? (CBBD, research only.)

    python scripts/ncaab_availability_backtest.py --output OUT.json [--scored-season 2026]

Seasons: scored-1 and scored are scored; scored-2 only ranks the conferences for the first.
About 110 CBBD calls: games for three seasons, lines and player box scores for two, in date
windows. Needs CBBD_API_KEY (falls back to CFBD_API_KEY: the same account serves both). No Odds
API calls; writes nothing to Drive or the store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_availability as model  # noqa: E402
from app_core import ncaab_line_backtest as data  # noqa: E402
from app_core import ncaab_possession_model as possession  # noqa: E402
from app_core import ncaab_ratings_backtest as ratings  # noqa: E402


def run(token, scored_season, get=None):
    if get is None:
        import requests
        get = requests.get
    scored = (scored_season - 1, scored_season)
    games = {season: possession.parse_games(data.fetch("games", season, token, get))
             for season in (scored_season - 2, *scored)}
    seasons = []
    for season in scored:
        lines = ratings.select_lines(data.fetch("lines", season, token, get))
        seasons.append((season, games[season], games[season - 1], model.fetch_players(season, token, get), lines))
    return {"seasons": list(scored), "tier_season": scored_season - 2, "report": model.run(seasons)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-season", type=int, default=2026)
    args = parser.parse_args(argv)
    token = (os.getenv("CBBD_API_KEY", "") or os.getenv("CFBD_API_KEY", "")).strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CBBD_API_KEY or CFBD_API_KEY")
    result = run(token, args.scored_season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = model.markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
