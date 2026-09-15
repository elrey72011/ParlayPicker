"""Bounded live MLB receipt capture, separate reconciliation and training export."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core import mlb_pregame_receipts as receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "reconcile", "export", "status"])
    parser.add_argument("--database", type=Path)
    parser.add_argument("--max-feeds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "capture":
        from core.streamlit_pipeline import _get_odds_api_key
        from app_core.odds_api import TheOddsAPIClient
        key = _get_odds_api_key()
        if not key:
            parser.exit(2, "Capture blocked: missing ODDS_API_KEY\n")
        games = TheOddsAPIClient(key, markets="spreads,totals").get_odds("baseball_mlb")
        _, report = receipts.capture_live_games(games, path=args.database, max_feeds=args.max_feeds)
    elif args.command == "reconcile":
        report = receipts.reconcile(args.database, max_games=args.max_feeds)
    elif args.command == "export":
        if args.output is None:
            parser.error("export requires --output")
        records = receipts.export_records(args.database, settled_only=not args.include_pending)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(records, file, indent=2, allow_nan=False)
        report = {"records_exported": len(records), "includes_pending": args.include_pending}
    else:
        report = receipts.health(args.database)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
