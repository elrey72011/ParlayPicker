"""Generate reproducible selector validation reports from graded candidate CSVs."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from core.selector_validation import REPORT_VERSION, build_report, read_inputs, render_markdown


def evaluator_identity():
    root = Path(__file__).resolve().parents[1]
    # Include identity normalization and runtime aliases, which can change the
    # event population even when the report implementation is unchanged.
    files = ["core/selector_validation.py", "core/team_mapper.py", "scripts/validate_selector.py", "data/dynamic_aliases.json"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              if (root / name).exists() else None for name in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Record a specification before collecting future slates")
    run = sub.add_parser("report", help="Evaluate exported candidates without changing production")
    for command in (freeze, run):
        command.add_argument("--train-through", required=True, help="Last development slate, YYYY-MM-DD Eastern")
        command.add_argument("--probability-column", default="calibrated_probability")
        command.add_argument("--market-column", default="market_probability")
        command.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--model-version", required=True)
    run.add_argument("--audits", nargs="+", required=True, help="Graded audit CSV paths or glob patterns")
    run.add_argument("--selections", nargs="+", help="Final-pick export paths or globs for approval joins")
    run.add_argument("--specification", type=Path)
    args = parser.parse_args(argv)
    cutoff = pd.Timestamp(args.train_through)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        parser.error("--train-through must be a calendar date")
    config = {"report_version": REPORT_VERSION, "train_through": cutoff.strftime("%Y-%m-%d"),
              "probability_column": args.probability_column, "market_column": args.market_column,
              "snapshot_policy": "latest_export_before_start", "slate_timezone": "America/New_York"}
    evaluator_hash, code_hashes = evaluator_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        now = pd.Timestamp.now(tz="UTC")
        if now >= (cutoff + pd.Timedelta(days=1)).tz_localize("America/New_York"):
            parser.error("A new freeze must precede evaluation; choose a current/future development cutoff")
        payload = {"configuration": config, "frozen_at": now.isoformat(),
                   "model_version": args.model_version, "evaluator_sha256": evaluator_hash}
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, allow_nan=False)
        print(f"Frozen specification: {args.output}")
        return 0

    def expand(patterns):
        paths = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            if not matches:
                raise ValueError(f"No CSV inputs matched: {pattern}")
            paths.extend(matches)
        return sorted(set(paths))

    try:
        audit_paths = expand(args.audits)
        audits, inputs = read_inputs(audit_paths)
        selections, final_inputs = None, []
        if args.selections:
            selections, final_inputs = read_inputs(expand(args.selections))
        specification = json.loads(args.specification.read_text(encoding="utf-8")) if args.specification else None
        if specification and specification.get("evaluator_sha256") != evaluator_hash:
            raise ValueError("Frozen evaluator hash differs; use the frozen code or freeze a new future evaluation")
        report = build_report(audits, train_through=args.train_through,
                              probability_column=args.probability_column, market_column=args.market_column,
                              selections=selections, specification=specification)
        report["reproducibility"] = {"audits": inputs, "final_selections": final_inputs,
                                     "evaluator_sha256": evaluator_hash, "code_hashes": code_hashes,
                                     "pandas_version": pd.__version__, "specification": specification}
        json_path, md_path = args.output.with_suffix(".json"), args.output.with_suffix(".md")
        all_inputs = {Path(p).resolve() for p in audit_paths}
        if args.selections:
            all_inputs.update(Path(p).resolve() for p in expand(args.selections))
        if args.specification:
            all_inputs.add(args.specification.resolve())
        if {json_path.resolve(), md_path.resolve()} & all_inputs:
            raise ValueError("Report output must not overwrite an input or frozen specification")
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {report['inventory']['eligible_events']} verified games")
    print(f"Report: {md_path}\nData: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
