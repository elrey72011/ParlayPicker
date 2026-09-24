"""Twelve-market readiness with explicit local/remote evidence provenance."""

from __future__ import annotations

import os
from pathlib import Path

from app_core.prediction_evidence import database_path
from app_core.prospective_evidence import all_market_readiness, read_records
from app_core.prospective_source_view import all_source_readiness, SOURCE_FILENAMES


def _latest(path, table, sport, market, time_field):
    rows = read_records(path, table, sport=sport, market_family=market)
    return max(rows, key=lambda row: (row[time_field], row[next(iter(row))])) if rows else None


def _reconciliation(path):
    from app_core.prospective_reconciliation import reconciliation_readiness
    return reconciliation_readiness(path)


def _load_remote(root, client, folder):
    from app_core.prospective_reconciliation import ensure_reconciliation_schema, reconcile_all
    from app_core.prospective_remote import sync
    from app_core.prospective_sport_adapters import ADAPTERS
    from app_core.mlb_receipt_remote import recover

    canonical = root / "prospective-evidence.sqlite3"
    ensure_reconciliation_schema(canonical)
    canonical_restore = sync(canonical, client, folder, {})
    restored = {}
    for sport, adapter in ADAPTERS.items():
        session = {}
        restored[sport] = adapter.restore(root / adapter.path_name, client, folder, session)
    receipt_path = root / SOURCE_FILENAMES["MLB"]
    restored["MLB_RECEIPTS"] = {"records_restored": recover(client, receipt_path)}
    reconciliation = reconcile_all(canonical, root)
    canonical_backup = sync(canonical, client, folder, {})
    return {"canonical_restore": canonical_restore, "source_restore": restored,
            "reconciliation": reconciliation, "canonical_backup": canonical_backup}


def load_readiness(directory=None, *, authenticate=False, client=None, folder=None):
    """Never call a local zero a verified remote zero.

    An authenticated request restores every source before reconciliation and
    verifies the canonical backup afterwards. Errors remain sanitized.
    """
    root = Path(directory) if directory is not None else database_path().parent
    canonical = root / "prospective-evidence.sqlite3"
    remote = {"status": "REMOTE_NOT_RESTORED", "verified": False, "blocker": "REMOTE_EVIDENCE_UNKNOWN"}
    if authenticate:
        from app_core.evidence_remote import settings
        folder = folder or settings()[0]
        if not folder or (client is None and not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip()):
            remote = {"status": "MISSING_CREDENTIALS", "verified": False,
                      "blocker": "OWNER_ACTION_CONFIGURE_DRIVE_CREDENTIALS"}
        else:
            try:
                if client is None:
                    from app_core.evidence_drive import DriveStore
                    client = DriveStore(folder)
                result = _load_remote(root, client, folder)
                remote = {"status": "RESTORED_AND_READBACK_VERIFIED", "verified": True,
                          "blocker": None, "counts": result}
            except Exception as exc:
                remote = {"status": "RESTORE_OR_BACKUP_FAILED", "verified": False,
                          "blocker": type(exc).__name__}
    canonical_rows = all_market_readiness(canonical)
    source_rows = all_source_readiness(root)
    sources = {(row["sport"], row["market_family"]): row for row in source_rows}
    reconciled = _reconciliation(canonical)
    reconciliation_rows = {(row["sport"], row["market_family"]): row for row in reconciled}
    rows = []
    for row in canonical_rows:
        sport, market = row["sport"], row["market_family"]
        source = sources[(sport, market)]
        model = _latest(canonical, "prospective_model", sport, market, "available_at")
        calibration = _latest(canonical, "prospective_calibration", sport, market, "available_at")
        plan = _latest(canonical, "prospective_validation_plan", sport, market, "version")
        merged = {**row,
                  "source_observations": source["research_quote_rows"],
                  "source_status": source["local_source_status"],
                  "canonical_predictions": row["pregame_observations"],
                  "independent_validation_count": row["validation_count"],
                  "independent_holdout_count": row["holdout_count"],
                  "model_id": model["model_id"] if model else None,
                  "calibration_id": calibration["calibration_id"] if calibration else None,
                  "validation_plan_id": plan["validation_plan_id"] if plan else None,
                  "reconciliation": reconciliation_rows.get((sport, market)),
                  "remote_evidence_status": remote["status"],
                  "remote_evidence_verified": remote["verified"]}
        if not remote["verified"]:
            merged["remote_count_interpretation"] = "LOCAL_COUNTS_ONLY_REMOTE_UNKNOWN"
        else:
            merged["remote_count_interpretation"] = "RESTORED_REMOTE_AND_LOCAL_CANONICAL"
        rows.append(merged)
    return {"markets": rows, "sources": source_rows, "remote": remote}
