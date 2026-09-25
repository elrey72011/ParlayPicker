"""Run Stage 2 only after an authenticated Stage 1 refresh and read-back."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.football_stage2 import build_reports, write_reports


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--stage1-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    audit = json.loads(Path(args.stage1_audit).read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    reports = build_reports(args.database, Path(__file__).resolve().parents[1],
                            stage1_report=audit, source_commit=commit)
    write_reports(reports, args.output_dir)
    print(json.dumps({"stage1_run_id": audit["run_id"],
                      "scopes": {key: {"legal_independent_n": item["legal_independent_n"],
                                        "model_status": reports["model_inventory"]["scopes"][key]["model_status"]}
                                 for key, item in reports["training_audit"]["scopes"].items()},
                      "output_dir": args.output_dir}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
