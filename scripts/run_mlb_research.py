"""Evaluate the fixed MLB protocol from a completed local checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.mlb_research import run_research, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_research(json.loads(args.checkpoint.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (args.output / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.md").write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))


if __name__ == "__main__":
    main()
