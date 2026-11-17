"""NCAAB possession model beside the points ratings, against archived lines (CBBD, research only).

    python scripts/ncaab_possession_backtest.py --output OUT.json [--scored-season 2026]

Seasons: scored-2 sets the priors, scored-1 picks the settings and calibrates, scored is scored.
About 90 CBBD calls: games and team box scores for three seasons, lines for two, in date windows.
Needs CBBD_API_KEY (falls back to CFBD_API_KEY: the same account serves both). No Odds API calls;
writes nothing to Drive or the store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_line_backtest as data  # noqa: E402
from app_core import ncaab_possession_model as model  # noqa: E402
from app_core import ncaab_ratings_backtest as ratings  # noqa: E402


def run(token, scored_season, get=None):
    if get is None:
        import requests
        get = requests.get
    seasons = (scored_season - 2, scored_season - 1, scored_season)
    games = {season: model.parse_games(data.fetch("games", season, token, get)) for season in seasons}
    # One season of box scores at a time: the rows are large and only two numbers a game are kept.
    box = {season: model.parse_box(data.fetch("games/teams", season, token, get)) for season in seasons}
    raw_lines = {season: data.fetch("lines", season, token, get) for season in seasons[1:]}
    lines = {season: ratings.select_lines(rows) for season, rows in raw_lines.items()}
    scored_lines = lines[scored_season]
    report = model.run(*((games[season], box[season]) for season in seasons), lines[seasons[1]], scored_lines,
                       ratings.select_pairs(raw_lines[scored_season]))
    return {"prior_season": seasons[0], "calibration_season": seasons[1], "scored_season": scored_season,
            "games": {str(season): len(rows) for season, rows in games.items()},
            "box_scores": {str(season): len(rows) for season, rows in box.items()},
            "closing_lined": sum(1 for line in scored_lines.values() if line["spread"] is not None),
            "opening_lined": sum(1 for line in scored_lines.values() if line["spread_open"] is not None),
            "providers": sorted({line["provider"] for line in scored_lines.values() if line["provider"]}),
            "report": report}


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
    text = (model.markdown(result)
            + f"\n- team box scores read: {result['box_scores']}; finished games: {result['games']}\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
