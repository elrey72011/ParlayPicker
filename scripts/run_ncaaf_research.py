"""Run the fixed research protocol against a completed collection ZIP."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.ncaaf_history import load_checkpoint
from app_core.ncaaf_research import run_research, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.collection) as archive:
        if archive.getinfo("checkpoint.json").file_size > 40_000_000:
            raise ValueError("Checkpoint is too large")
        state = load_checkpoint(archive.read("checkpoint.json"))
    result = run_research(state)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (args.output / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.md").write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))


if __name__ == "__main__":
    main()
