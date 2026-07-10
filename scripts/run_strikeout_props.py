#!/usr/bin/env python3
"""Run the pitcher-strikeout prop card end-to-end for a date.

Lists the day's MLB events (Odds API), pulls strikeout props, resolves each starter's recent
form + opponent K rate (StatsAPI), scores every prop, and prints the card with the actionable
picks first. The model is the standalone prop slice â€” separate from the main best-picks card â€”
so it can be run and calibrated before any integration.

Usage:
    ODDS_API_KEY=... python3 scripts/run_strikeout_props.py [--date YYYY-MM-DD] [--min-edge 0.04]
                                                            [--season YYYY] [--csv out.csv]

The Odds API key is read from --api-key, then $ODDS_API_KEY / $THE_ODDS_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.odds_api import TheOddsAPIClient  # noqa: E402
from app_core.prop_pipeline import PROP_MIN_EDGE  # noqa: E402
from app_core.prop_runner import build_strikeout_card  # noqa: E402

_CARD_COLS = [
    "pitcher", "home_team", "away_team", "market_type", "line", "expected_stat",
    "expected_count", "expected_ks", "expected_walks", "expected_outs",
    "model_p_over", "market_p_over", "best_side", "best_edge", "best_ev",
    "recommendation", "book",
]


def _today_et() -> str:
    return datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="slate date YYYY-MM-DD (default: today ET)")
    ap.add_argument("--season", type=int, default=None, help="StatsAPI season (default: date's year)")
    ap.add_argument("--min-edge", type=float, default=PROP_MIN_EDGE)
    ap.add_argument("--api-key", default=None, help="Odds API key (else $ODDS_API_KEY)")
    ap.add_argument("--csv", default=None, help="optional path to write the full scored card")
    args = ap.parse_args()

    date = args.date or _today_et()
    season = args.season or int(date[:4])
    api_key = args.api_key or os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("no Odds API key (pass --api-key or set $ODDS_API_KEY)", file=sys.stderr)
        return 2

    client = TheOddsAPIClient(api_key=api_key, markets="h2h")  # markets unused for props
    card = build_strikeout_card(client, date, season, min_edge=args.min_edge)

    if not card:
        print(f"{date}: no strikeout props scored (no MLB props on feed).")
        return 0

    df = pd.DataFrame(card)
    for c in _CARD_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[_CARD_COLS]

    actionable = df[df["recommendation"].isin(["over", "under"])]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\n{date}: scored {len(df)} strikeout props, {len(actionable)} actionable "
          f"(edge >= {args.min_edge:.0%}, EV > 0)\n")
    print((actionable if not actionable.empty else df.head(15)).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote full card -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

