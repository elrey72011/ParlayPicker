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


def _paired_rows(day_counts):
    from datetime import date, timedelta
    rows = []
    for day_text, count in day_counts.items():
        day = date.fromisoformat(day_text)
        for game in range(count):
            for family in ('spread', 'total'):
                for side in ('home', 'away'):
                    rows.append(dict(family=family, event=('mlb', f'{day_text}-{game}'),
                        x=[1, 1.5], outcome='WIN' if side == 'home' else 'LOSS',
                        reference_outcome='WIN', slate=day_text,
                        start=f'{day_text}T20:00:00+00:00',
                        cutoff=f'{day_text}T19:00:00+00:00',
                        outcome_at=f'{day + timedelta(days=1)}T02:00:00+00:00'))
    return rows


def test_chronological_capacity_exposes_missing_training_split():
    rows = _paired_rows({'2026-09-17': 2, '2026-09-18': 15,
                         '2026-09-19': 13, '2026-09-20': 15})
    report = audit.chronological_capacity(rows)
    assert report['max_training_units_with_later_evaluation_floors'] == {'spread': 17, 'total': 17}
    assert report['count_feasible_cutoff_pairs'] == 0
    assert report['chronology_feasible_cutoff_pairs'] == 0
    assert report['cutoff_selection_performed'] is False


def test_chronological_capacity_only_checks_counts_and_order():
    rows = _paired_rows({'2026-09-17': 20, '2026-09-18': 10, '2026-09-19': 10})
    report = audit.chronological_capacity(rows)
    assert report['count_feasible_cutoff_pairs'] == 1
    assert report['chronology_feasible_cutoff_pairs'] == 1
    assert report['cutoff_selection_performed'] is False


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
