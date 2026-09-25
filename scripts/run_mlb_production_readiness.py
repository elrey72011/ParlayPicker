"""Authenticated, read-only MLB exact-market readiness audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_core import mlb_production_readiness as readiness


def run(directory: Path, output: Path, source_commit: str):
    from app_core import evidence_remote, mlb_receipt_remote
    from app_core.evidence_drive import DriveStore

    folder, _ = evidence_remote.settings()
    if not folder:
        raise ValueError("MISSING_DRIVE_FOLDER_ID")
    client = DriveStore(folder)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "mlb-pregame-receipts.sqlite3"
    canonical_path = directory / "prospective-evidence.sqlite3"
    snapshot_path = directory / "evidence.sqlite3"
    restored = mlb_receipt_remote.recover(client, receipt_path)
    verified = getattr(client, "_receipt_verified", set())
    manifests = sorted(key for key in verified if key.startswith(mlb_receipt_remote.MANIFEST_PREFIX))
    if not manifests:
        raise ValueError("NO_VERIFIED_REMOTE_RECEIPT_MANIFEST")
    backup_ids = []
    for key in manifests:
        backup_id = key.removeprefix(mlb_receipt_remote.MANIFEST_PREFIX).removesuffix(".json")
        mlb_receipt_remote.verify_backup(client, folder,
            {"remote_backup_verified": True, "backup_id": backup_id})
        backup_ids.append(backup_id)
    canonical, canonical_count = readiness.read_canonical_remote(client, folder, canonical_path)
    native = readiness.read_native_remote(client)
    snapshots = readiness.read_snapshot_remote(client, snapshot_path)
    verification = {
        "receipt_restore_verified": True, "receipt_readback_verified": True,
        "canonical_restore_verified": True, "native_restore_verified": True,
        "snapshot_restore_verified": True, "receipt_records_restored": restored,
        "receipt_objects_verified": len(verified), "receipt_backup_ids": backup_ids,
        "canonical_objects_verified": canonical_count,
        "native_objects_verified": len(native),
        "snapshot_records_restored": snapshots["snapshots_restored"],
    }
    reports = readiness.build_reports(receipt_path, ROOT, source_commit=source_commit,
        remote_verification=verification, canonical=canonical, native=native, snapshots=snapshots)
    readiness.write_reports(reports, output)
    return {"status": "COMPLETE", "source_commit": source_commit,
            "remote_verification": verification,
            "scopes": {key: {"legal_independent_n": value["legal_independent_n"],
                             "legal_by_season": value["legal_by_season"],
                             "blockers": reports["validation"]["scopes"][key]["blockers"],
                             "validation_readiness": reports["validation"]["scopes"][key]["status"]}
                       for key, value in reports["training"]["scopes"].items()},
            "reports_written": len(readiness.REPORTS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args(argv)
    try:
        report = run(a.database_dir, a.output_dir, a.source_commit)
    except Exception as exc:
        # Never put provider URLs, credentials or raw remote payloads in CI logs.
        a.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"status": "FAILED", "reason": "MLB_READINESS_AUDIT_FAILED",
                  "error_type": type(exc).__name__, "remote_verification": False}
        (a.output_dir / "mlb-production-readiness-failure.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
