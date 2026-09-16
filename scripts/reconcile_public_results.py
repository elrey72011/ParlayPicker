"""Read saved history only. Never fetch new scores or mutate remote history."""
import argparse
import json
from pathlib import Path
from app_core.public_reconciliation import load_source, reconcile, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local immutable-history JSON")
    parser.add_argument("--site")
    parser.add_argument("--folder")
    parser.add_argument("--date", default="2026-09-15")
    parser.add_argument("--output", type=Path, default=Path("output/september-15-reconciliation"))
    args = parser.parse_args()
    if args.source:
        source = json.loads(args.source.read_text(encoding="utf-8"))
    elif args.site and args.folder:
        from app_core.public_history import History
        source = load_source(History(args.site, args.folder))
    else:
        parser.error("Supply --source or --site and --folder with configured Drive credentials")
    result = reconcile(source, args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    return 1 if result["source_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
