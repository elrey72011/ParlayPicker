"""Go-live checklist for one sport and market (roadmap C4 prep). Read-only.

    python scripts/go_live_check.py --sport NCAAB --market-family SPREAD \
        [--database <prospective-evidence.sqlite3>] [--ledger <exposure.sqlite3>] [--output checklist.md]

Prints each requirement for the market to reach a staked pick, its status (PASS, ACTION,
WAIT) and the next step. It checks the settings of the environment it runs in; run it
where the app runs (or use "Go-live checklist" in the app's readiness report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sport", required=True)
    parser.add_argument("--market-family", required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from app_core.go_live import checklist, markdown, summary
    sport, family = args.sport.upper(), args.market_family.upper()
    items = checklist(sport, family, path=args.database, ledger_path=args.ledger)
    text = json.dumps({"items": items, **summary(items)}, indent=2) if args.json else markdown(sport, family, items)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
