import json
from app_core import mlb_receipt_audit as audit
from app_core import mlb_pregame_receipts as receipts
from test_mlb_pregame_receipts import fixture, collect


def test_empty_store_never_claims_training_readiness(tmp_path):
    report = audit.audit_store(tmp_path / "empty.sqlite3")
    assert report["receipts"] == 0
    assert "no_pregame_receipts" in report["blockers"]
    assert report["training_authorized"] is False
    assert report["remote_backup_verified"] is False


def test_backup_preserves_all_records_and_audit_counts_pending(fixture):
    collect(fixture)
    path = fixture[0]
    before = {t: receipts.read(t, path) for t in ("observations", "receipts", "outcomes")}
    report = audit.audit_store(path)
    bundle = audit.backup_bundle(path)
    assert report["receipts"] == report["pending_receipts"] == 4
    assert report["unique_events"] == 1
    assert report["settled_receipts"] == 0
    assert "no_settled_receipts" in report["blockers"]
    assert bundle["sha256"] == audit.digest(bundle["payload"])
    assert bundle["payload"]["tables"] == before
    assert {t: receipts.read(t, path) for t in before} == before
    json.dumps(bundle, allow_nan=False)
