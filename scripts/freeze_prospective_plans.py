"""Freeze the 12 reviewed prospective market plans in an authenticated store.

Run only after restoring the canonical remote evidence database. The command
does not inspect outcomes or create any deployment or wager authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app_core.prospective_validation_plans import freeze_current_validation_plans, plan_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True,
                        help="restored canonical prospective SQLite path")
    parser.add_argument("--source-commit", required=True,
                        help="40-character commit SHA of the running reviewed source")
    parser.add_argument("--dry-run", action="store_true",
                        help="print prospective policy IDs and hashes without writing")
    args = parser.parse_args(argv)
    if args.dry_run:
        rows = [{"sport": p["sport"], "market_family": p["market_family"],
                 "validation_plan_id": p["validation_plan_id"],
                 "policy_source_hash": p["policy_source_hash"]}
                for p in plan_specs(source_commit=args.source_commit)]
    else:
        rows = freeze_current_validation_plans(args.database,
                                               source_commit=args.source_commit)
    print(json.dumps({"count": len(rows), "dry_run": args.dry_run,
                      "database": str(args.database), "plans": rows},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
