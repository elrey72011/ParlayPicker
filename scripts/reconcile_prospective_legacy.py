"""Read-only legacy-source reconciliation into the canonical research ledger."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.prospective_reconciliation import reconcile_all, reconcile_sport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, help="canonical prospective SQLite path")
    parser.add_argument("--source-dir", type=Path, help="directory of restored legacy stores")
    parser.add_argument("--sport", choices=("NFL", "NCAAF", "MLB"),
                        help="reconcile only one legacy sport")
    args = parser.parse_args(argv)
    directory = args.source_dir
    if args.sport:
        from app_core.prospective_source_view import SOURCE_FILENAMES
        result = {args.sport: reconcile_sport(args.sport, args.canonical,
                  directory / SOURCE_FILENAMES[args.sport] if directory else None)}
    else:
        result = reconcile_all(args.canonical, directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
