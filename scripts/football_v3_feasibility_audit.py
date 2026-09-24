"""Run a read-only authenticated V3 feasibility audit in GitHub Actions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from app_core.football_v3_feasibility import restore_and_audit


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder = os.environ.get("PARLAYPICKER_DRIVE_FOLDER_ID", "").strip()
        if not folder or not os.environ.get("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip():
            raise ValueError("MISSING_AUTHENTICATED_DRIVE_SETTINGS")
        from app_core.evidence_drive import DriveStore
        report = restore_and_audit(Path(args.root), DriveStore(folder), folder)
        status = 0
    except Exception as exc:
        report = {"schema": "football-v3-feasibility-audit-v1",
                  "generated_at": datetime.now(timezone.utc).isoformat(),
                  "authenticated_remote_verified": False,
                  "result": "AUTHENTICATED_RESTORE_FAILED",
                  "error_type": type(exc).__name__,
                  "v3_plans_created": 0, "no_wager_or_activation": True}
        status = 1
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": report["result"], "authenticated_remote_verified":
                      report["authenticated_remote_verified"], "audit_path": str(output)}))
    return status


if __name__ == "__main__":
    sys.exit(main())
