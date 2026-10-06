"""Explicit local files only. No acquisition, database initialization or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from app_core.nfl_calibration_evidence import build_dataset, export_private, read_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--specification", required=True, type=Path)
    parser.add_argument("--specification-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    specification = read_review(args.specification, args.specification_sha256)
    dataset = build_dataset(path=args.database, **specification)
    raw, digest = export_private(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A new private output is required; prior evidence is never overwritten.
    with args.output.open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(dataset_sha256=digest, report=dataset["report"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
