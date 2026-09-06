"""Refresh saved game results or export immutable prediction evidence for validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from app_core.prediction_evidence import load_snapshots, materialize, record_scores, write_validation_reports
from core.selector_validation import build_report, render_markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh", "import-scores", "report"])
    parser.add_argument("--database", type=Path)
    parser.add_argument("--scores", type=Path, help="Final scores with snapshot_id and matchup_id")
    parser.add_argument("--train-through", help="Final development slate in Eastern time")
    parser.add_argument("--output", type=Path, default=Path("output/live-selector-validation"))
    args = parser.parse_args(argv)
    if args.command == "import-scores":
        if not args.scores:
            parser.error("--scores is required")
        print(f"Saved {record_scores(pd.read_csv(args.scores), path=args.database)} score revisions")
        write_validation_reports(args.database)
        return 0
    if args.command == "refresh":
        from app_core.performance_pipeline import grade_picks_with_live_results
        yesterday = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frames = []
        for _, _, final in load_snapshots(args.database):
            starts = pd.to_datetime(final.game_start_utc, errors="coerce", utc=True)
            frames.append(final[starts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").eq(yesterday)])
        pending = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if pending.empty:
            print("No saved picks from yesterday to refresh")
            return 0
        graded = grade_picks_with_live_results(pending)
        print(f"Saved {record_scores(graded, path=args.database)} score revisions")
        write_validation_reports(args.database)
        return 0
    if not args.train_through:
        parser.error("--train-through is required for report")
    audits, finals = materialize(args.database)
    if audits.empty:
        parser.error("No saved prediction snapshots exist yet; run Master Analysis first")
    report = build_report(audits, train_through=args.train_through, selections=finals)
    report["evidence_store"] = {"snapshot_ids": sorted(audits.snapshot_id.unique().tolist()),
                                "database": str(args.database or "default local evidence store")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audits.to_csv(args.output.with_name(args.output.name + "-candidates.csv"), index=False)
    finals.to_csv(args.output.with_name(args.output.name + "-decisions.csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"{report['status']}: {report['inventory']['eligible_events']} verified games; {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
