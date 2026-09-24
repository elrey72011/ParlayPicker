"""Restore, capture and read-back-verify football Stage 1 evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.football_stage1_cycle import run_cycle, write_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--theover-sides")
    parser.add_argument("--theover-totals")
    args = parser.parse_args(argv)
    folder = os.getenv("PARLAYPICKER_DRIVE_FOLDER_ID", "").strip()
    odds_key = os.getenv("ODDS_API_KEY", "").strip()
    cfbd_key = os.getenv("CFBD_API_KEY", "").strip()
    report = {"schema": "football-stage1-cycle-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "requested_slate_success": False, "execution_state": "PREFLIGHT_FAILED", "sports": {},
              "remote": {}, "no_model_calibration_prediction_activation_stake_or_wager": True}
    if not folder or not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip() or not odds_key or not cfbd_key:
        report["preflight_error"] = "MISSING_AUTHENTICATED_STAGE1_SETTINGS"
    else:
        try:
            from app_core.evidence_drive import DriveStore
            client = DriveStore(folder)
            report = run_cycle(args.database, folder, client, odds_key, cfbd_key,
                               theover_files=[p for p in (args.theover_sides, args.theover_totals) if p])
        except Exception as exc:
            report["execution_state"] = "AUTHENTICATED_STAGE1_FAILURE"
            report["error_type"] = type(exc).__name__
    write_artifacts(report, args.output_dir)
    print(json.dumps({"execution_state": report["execution_state"],
                      "requested_slate_success": report["requested_slate_success"],
                      "output_dir": args.output_dir}, sort_keys=True))
    return 0 if report["requested_slate_success"] else 1


if __name__ == "__main__":
    sys.exit(main())
