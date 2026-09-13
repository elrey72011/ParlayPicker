"""Compare saved rankings and sources without changing live picks or weights."""
import argparse
import glob
import json
from pathlib import Path
from app_core.pick_accuracy import build_accuracy_report, render_accuracy_markdown
from core.selector_validation import read_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", required=True, help="Glob of full graded candidate ledgers")
    parser.add_argument("--evaluation-start", required=True, help="First evaluation slate, YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    frame, sources = read_inputs(glob.glob(args.audits))
    report = build_accuracy_report(frame, evaluation_start=args.evaluation_start)
    report["input_files"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_accuracy_markdown(report), encoding="utf-8")
    print(report["status"], report["validation"]["inventory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
