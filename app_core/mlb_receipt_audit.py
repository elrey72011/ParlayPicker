"""Read-only receipt inventory and portable backup. Never grants model authority."""
from collections import Counter
from app_core import mlb_pregame_receipts as receipts
from app_core.mlb_spread_total_model import CONFIG, prepare_rows, receipt_features, digest


def audit_store(path=None):
    records = receipts.export_records(path)
    settled = [r for r in records if r["outcome"] is not None]
    blockers = []
    for record in records:
        receipt_features(record["snapshot"])
    try:
        rows = prepare_rows(settled)
    except (ValueError, KeyError, TypeError):
        rows = []
        blockers.append("settled_dataset_integrity_failed")
    counts = Counter(r["family"] for r in rows if r["outcome"] in {"WIN", "LOSS"})
    days = sorted({r["slate"] for r in rows})
    if not records:
        blockers.append("no_pregame_receipts")
    if not settled:
        blockers.append("no_settled_receipts")
    if len(days) < 3:
        blockers.append("fewer_than_three_settled_slate_dates")
    # Necessary lower bound only: the trainer also deduplicates paired sides
    # and checks chronological partitions, classes and outcome availability.
    for family in ("spread", "total"):
        if counts[family] < 3 * CONFIG["minimum_rows_per_family_split"]:
            blockers.append(f"insufficient_{family}_decided_rows")
    return {"schema": "mlb-receipt-audit-v1", "receipts": len(records),
            "unique_events": len({receipts.event_key(r["snapshot"]["payload"]) for r in records}),
            "settled_receipts": len(settled), "pending_receipts": len(records)-len(settled),
            "settled_slate_dates": days, "decided_rows_by_family": dict(counts),
            "minimum_rows_per_family_per_split": CONFIG["minimum_rows_per_family_split"],
            "blockers": blockers, "training_authorized": False,
            "split_validation": "Not performed; requires predeclared chronological cutoffs and trainer checks",
            "remote_backup_verified": False,
            "persistence": "Local store; this audit does not prove recovery after redeployment"}


def backup_bundle(path=None):
    # Include the raw observations needed to verify provenance and avoid
    # recollecting cached history. Hash protects bytes, not provenance by itself.
    tables = {name: receipts.read(name, path) for name in ("observations", "receipts", "outcomes")}
    payload = {"schema": "mlb-receipt-backup-v1", "tables": tables}
    return {"payload": payload, "sha256": digest(payload)}
