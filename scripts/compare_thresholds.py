"""Compare development-only selection thresholds from saved evidence or audit exports."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.prediction_evidence import materialize, database_path
from core.selector_validation import read_inputs
from core.threshold_validation import compare_thresholds, render_threshold_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audits", nargs="+")
    source.add_argument("--database", type=Path)
    parser.add_argument("--selections", nargs="+")
    parser.add_argument("--train-through", required=True)
    parser.add_argument("--development-through", required=True)
    parser.add_argument("--scope", choices=["qualified_wagers", "all_selected"], default="qualified_wagers")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.5, .55, .6, .65, .7, .75, .8, .85, .9])
    parser.add_argument("--output", type=Path, default=Path("output/threshold-comparison"))
    args = parser.parse_args(argv)

    def expand(patterns):
        files = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            if not matches:
                raise ValueError(f"No CSV files matched {pattern}")
            files.extend(matches)
        return sorted(set(files))

    try:
        inputs = []
        if args.audits:
            files = expand(args.audits)
            audit, inventory = read_inputs(files)
            inputs.extend(files)
            final = None
            if args.selections:
                files = expand(args.selections)
                final, more = read_inputs(files)
                inventory += more
                inputs.extend(files)
        else:
            if args.selections:
                raise ValueError("--selections is only used with --audits")
            audit, final = materialize(args.database)
            inventory = [{"source": str(args.database or database_path()),
                          "materialized_audit_sha256": hashlib.sha256(audit.to_csv(index=False).encode()).hexdigest(),
                          "materialized_decisions_sha256": hashlib.sha256(final.to_csv(index=False).encode()).hexdigest()}]
            if audit.empty:
                raise ValueError("No saved snapshots exist; run Master Analysis first or supply graded audit CSVs")
            inputs.append(str(args.database or database_path()))
        result = compare_thresholds(audit, train_through=args.train_through,
                                    development_through=args.development_through, selections=final,
                                    thresholds=args.thresholds, scope=args.scope)
        root = Path(__file__).resolve().parents[1]
        code = ["core/threshold_validation.py", "core/selector_validation.py", "core/team_mapper.py", "scripts/compare_thresholds.py"]
        result["reproducibility"] = {"inputs": inventory,
                                     "code_hashes": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in code}}
        targets = [args.output.with_suffix(".json"), args.output.with_suffix(".md")]
        if set(p.resolve() for p in targets) & set(Path(p).resolve() for p in inputs):
            raise ValueError("Report output must not overwrite an input")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        targets[1].write_text(render_threshold_report(result), encoding="utf-8")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{result['status']}: {targets[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
