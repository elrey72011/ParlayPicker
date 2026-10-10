"""Explicit private local setup. Default is read-only assessment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core import ncaaf_pilot_setup as local


def main(argv=None):
    parser = argparse.ArgumentParser(description='Assess or explicitly prepare an isolated prediction-store copy. No provider or inference activity.')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--backup-root', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorization-ref')
    args = parser.parse_args(argv)
    if args.execute:
        result = local.setup(args.source, args.root, args.backup_root, authorization_ref=args.authorization_ref)
    else:
        result = local.assess(args.source, args.root, args.backup_root)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] != 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
