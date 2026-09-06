"""Read-only evidence-store checks. No database is created by a health probe."""
from contextlib import closing
import hashlib
from io import StringIO
import os
from pathlib import Path
import sqlite3

import pandas as pd


def evidence_health(path=None, process_instance=None):
    from app_core.prediction_evidence import database_path, PROCESS_INSTANCE

    from app_core.evidence_remote import remote_status
    location = Path(path or database_path()).resolve()
    result = {"remote_storage": remote_status(), "status": "missing", "snapshots": 0, "score_revisions": 0,
              "storage_directory_configured": bool(os.environ.get("PARLAYPICKER_EVIDENCE_DIR")),
              "prior_process_snapshots_accessible": None,
              "durability_across_redeployment_verified": False,
              "latest_snapshot_id": None, "latest_generated_at": None,
              "latest_candidates": 0, "latest_exact_quote_times": 0, "latest_model_versions": [],
              "persistence_note": "A directory setting or healthy SQLite file does not prove persistent storage across redeployments."}
    if not location.exists():
        return result
    try:
        with closing(sqlite3.connect(location.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            result["snapshots"] = db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            result["score_revisions"] = db.execute("SELECT COUNT(*) FROM score_revisions").fetchone()[0]
            latest = db.execute("SELECT snapshot_id,generated_at,candidates,decisions,inputs,payload_hash FROM snapshots ORDER BY generated_at DESC LIMIT 1").fetchone()
            has_runtime = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot_runtime'").fetchone()
            if has_runtime:
                result["prior_process_snapshots_accessible"] = bool(db.execute(
                    "SELECT COUNT(*) FROM snapshot_runtime WHERE process_instance != ?", (process_instance or PROCESS_INSTANCE,)).fetchone()[0])
            if latest:
                sid, generated, candidates, decisions, inputs, digest = latest
                if hashlib.sha256("\0".join([candidates, decisions, inputs]).encode()).hexdigest() != digest:
                    raise ValueError("Latest prediction payload failed its hash check")
                frame = pd.read_csv(StringIO(candidates))
                result.update(latest_snapshot_id=sid, latest_generated_at=generated, latest_candidates=len(frame))
                result["latest_exact_quote_times"] = int(frame.get("quote_binding_verified", pd.Series(False, index=frame.index)).astype(str).str.lower().isin(["true", "1"]).sum())
                result["latest_model_versions"] = sorted(frame.get("model_version", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        result["status"] = "healthy" if latest else "empty"
    except (sqlite3.Error, ValueError, OSError) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result
