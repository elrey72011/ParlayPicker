"""Read-only prospective MLB totals quality comparison."""
import argparse
import json
from pathlib import Path
from app_core.total_quality_report import read_candidates, summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/total-quality-cohorts.json"))
    args = parser.parse_args()
    report = summarize(*read_candidates(args.database))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
