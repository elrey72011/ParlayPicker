"""Create a run-readiness report from exact audit/final exports or saved snapshots."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.run_readiness import build_readiness, game_table, render_readiness
from core.selector_validation import read_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--output", type=Path, default=Path("output/run-readiness"))
    args = parser.parse_args(argv)
    inputs = []
    if args.audit:
        if args.database or args.snapshot_id:
            parser.error("Use either CSV inputs or a saved database snapshot")
        audit, inputs = read_inputs([args.audit])
        final = None
        if args.final:
            final, more = read_inputs([args.final])
            inputs += more
    else:
        if args.final:
            parser.error("--final requires --audit")
        from app_core.prediction_evidence import load_snapshots
        snapshots = load_snapshots(args.database)
        if args.snapshot_id:
            snapshots = [item for item in snapshots if item[0] == args.snapshot_id]
        if not snapshots:
            parser.error("No matching saved snapshot exists")
        _, audit, final = snapshots[-1]
    report = build_readiness(audit, final)
    report["input_files"] = inputs
    outputs = [args.output.with_suffix(suffix) for suffix in (".json", ".md", ".csv")]
    sources = {p.resolve() for p in (args.audit, args.final, args.database) if p}
    if any(p.resolve() in sources for p in outputs):
        parser.error("Report output must not overwrite an input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs[1].write_text(render_readiness(report), encoding="utf-8")
    game_table(report).to_csv(outputs[2], index=False)
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
