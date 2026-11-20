"""Write the paper track record of saved best picks (core.track_record). Read-only.

    python scripts/track_record.py --database data/prediction_evidence/evidence.sqlite3 --output output/track-record.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import blend_shadow, blend_weights, kalshi_report, theover_plan, track_record  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = track_record.build(args.database)
    report["kalshi"] = kalshi_report.build(args.database)
    report["blend_weights"] = blend_weights.build(args.database)
    report["blend_shadow"] = blend_shadow.build(args.database)
    # TheOver's college totals plan (owner 2026-11-19): a tally against its fixed rules. A fault in it is
    # said in the report and never costs the track record above.
    try:
        report["theover_plan"] = theover_plan.build(args.database)
        plan_text = theover_plan.markdown(report["theover_plan"])
    except Exception as exc:
        report["theover_plan"] = {"error": type(exc).__name__}
        plan_text = f"## TheOver's college totals plan\n\nThe tally could not be built ({type(exc).__name__}).\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = (track_record.markdown(report) + "\n" + kalshi_report.markdown(report["kalshi"]) + "\n"
            + blend_weights.markdown(report["blend_weights"]) + "\n" + blend_shadow.markdown(report["blend_shadow"])
            + "\n" + plan_text)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
