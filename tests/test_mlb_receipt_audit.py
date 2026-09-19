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

def test_training_inventory_does_not_count_complementary_sides_twice():
    rows=[dict(family='spread',event=('mlb','1'),x=[1,2,1.5],outcome='WIN',reference_outcome='WIN',slate='2026-09-17'),
          dict(family='spread',event=('mlb','1'),x=[1,2,1.5],outcome='LOSS',reference_outcome='WIN',slate='2026-09-17')]
    result=audit.training_inventory(rows)['spread']
    assert result['decided_market_rows']==2
    assert result['independent_event_line_units']==1
    assert result['additional_units_to_training_floor_only']==19
    assert result['both_training_classes_observed'] is False


def test_training_inventory_rejects_complementary_conflicts():
    import pytest
    row=dict(family='total',event=('mlb','1'),x=[1,8.5],outcome='WIN',reference_outcome='WIN',slate='2026-09-17')
    with pytest.raises(ValueError,match='conflicting'):
        audit.training_inventory([row,dict(row,reference_outcome='LOSS')])


def test_audit_downloads_never_load_raw_backup(fixture, monkeypatch):
    collect(fixture)
    original = receipts.read
    def guarded_read(table, path=None):
        assert table != "observations", "UI audit must not load raw feeds"
        return original(table, path)
    monkeypatch.setattr(receipts, "read", guarded_read)
    def forbidden(*args, **kwargs):
        raise AssertionError("UI audit must not build a full backup")
    monkeypatch.setattr(audit, "backup_bundle", forbidden)
    downloads = audit.audit_downloads(fixture[0])
    assert len(downloads) == 2
    assert all(isinstance(value, bytes) for _, _, value in downloads)
    assert json.loads(downloads[0][2])["receipts"] == 4
    assert json.loads(downloads[1][2]) == []
