"""Screen a market-anchored NCAAB model on archived CollegeBasketballData lines (research only).

    python scripts/ncaab_line_backtest.py --output OUT.json [--train-season 2024 --test-season 2025]

Needs CBBD_API_KEY (falls back to CFBD_API_KEY: the same account serves both APIs).
About 30 CBBD calls (games and lines for two seasons, in date windows). No Odds API calls; writes nothing to
Drive or the evidence store.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_line_backtest as bt  # noqa: E402


def run(token, train_season, test_season, get=None):
    if get is None:
        import requests
        get = requests.get
    train_games = bt.parse_games(bt.fetch("games", train_season, token, get))
    test_games = bt.parse_games(bt.fetch("games", test_season, token, get))
    train_lines = bt.select_lines(bt.fetch("lines", train_season, token, get))
    test_lines = bt.select_lines(bt.fetch("lines", test_season, token, get))
    return {"train_season": train_season, "test_season": test_season,
            "train_games": len(train_games), "test_games": len(test_games),
            "train_lined": sum(1 for g in train_games if g["id"] in train_lines),
            "test_lined": sum(1 for g in test_games if g["id"] in test_lines),
            "providers": sorted({v["provider"] for v in test_lines.values() if v["provider"]}),
            "report": bt.run(train_games, test_games, train_lines, test_lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    args = parser.parse_args(argv)
    token = (os.getenv("CBBD_API_KEY", "") or os.getenv("CFBD_API_KEY", "")).strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CBBD_API_KEY or CFBD_API_KEY")
    result = run(token, args.train_season, args.test_season)
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
