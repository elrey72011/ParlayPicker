from copy import deepcopy
from io import BytesIO
import pytest
from app_core import mlb_receipt_remote as remote
from app_core import mlb_pregame_receipts as r
from app_core.mlb_receipt_audit import backup_bundle
from test_mlb_pregame_receipts import fixture, collect


class Store:
    def __init__(self): self.data = {}
    def put_object(self, **kw): self.data[kw['Key']] = kw['Body']
    def get_object(self, **kw): return {'Body': BytesIO(self.data[kw['Key']])}
    def read_objects(self, **kw): return [(k,v) for k,v in self.data.items() if k.startswith(kw['Prefix'])]


def test_empty_store_recovery_matches_every_hash(fixture, tmp_path):
    collect(fixture)
    store = Store()
    before = backup_bundle(fixture[0])
    assert remote.backup(store, 'folder', fixture[0])['remote_backup_verified']
    target = tmp_path / 'recovered.sqlite3'
    assert remote.recover(store, target) > 0
    assert backup_bundle(target) == before
    assert remote.recover(store, target) == 0


def test_corrupt_backup_rejected(fixture, tmp_path):
    collect(fixture)
    bundle = backup_bundle(fixture[0])
    bundle['sha256'] = 'wrong'
    with pytest.raises(ValueError): remote.restore(bundle, tmp_path/'bad.sqlite3')


def test_local_conflict_does_not_rewrite(fixture, tmp_path):
    collect(fixture)
    bundle = backup_bundle(fixture[0])
    target = tmp_path/'conflict.sqlite3'
    key, payload = next(iter(bundle['payload']['tables']['receipts'].items()))
    changed = deepcopy(payload)
    changed['sha256'] = 'conflicting'
    r.append('receipts', key, changed, target)
    with pytest.raises(ValueError, match='conflict'): remote.restore(bundle, target)
    assert r.read('observations', target) == {}
    assert r.read('receipts', target)[key] == changed


def test_collect_restores_then_backs_up_before_and_after_reconcile(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, 'connection', lambda: ('client','folder'))
    monkeypatch.setattr(remote, 'recover', lambda *a: calls.append('recover') or 2)
    monkeypatch.setattr(r, 'capture_live_games', lambda games, **kw: (calls.append('capture') or games, {}))
    monkeypatch.setattr(remote, 'backup', lambda *a: calls.append('backup') or {'remote_backup_verified':True})
    monkeypatch.setattr(r, 'reconcile', lambda **kw: calls.append('reconcile') or {'outcomes_created':1})
    _, health = remote.collect_durable([])
    assert calls == ['recover','capture','backup','reconcile','backup']
    assert health['remote_backup_verified'] is True
    assert health['records_restored'] == 2


def test_remote_corruption_fails_before_restore(fixture, tmp_path):
    collect(fixture)
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    key = next(iter(store.data))
    store.data[key] = store.data[key] + b'corrupt'
    with pytest.raises(ValueError): remote.recover(store, tmp_path/'tampered.sqlite3')


def test_total_store_can_exceed_object_limit(fixture, tmp_path, monkeypatch):
    collect(fixture)
    bundle = backup_bundle(fixture[0])
    largest = max(len(remote.canonical({'table':t,'id':k,'payload':v})) for t,rs in bundle['payload']['tables'].items() for k,v in rs.items())
    monkeypatch.setattr(remote, 'MAX_OBJECT_BYTES', largest + 1000)
    assert len(remote.canonical(bundle)) > remote.MAX_OBJECT_BYTES
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    target = tmp_path/'large.sqlite3'
    remote.recover(store, target)
    assert backup_bundle(target) == bundle


def test_legacy_backup_recovers(fixture, tmp_path):
    collect(fixture)
    bundle = backup_bundle(fixture[0])
    store = Store()
    store.data[remote.PREFIX + remote.digest(bundle) + '.json'] = remote.canonical(bundle)
    target = tmp_path/'legacy.sqlite3'
    remote.recover(store, target)
    assert backup_bundle(target) == bundle


def test_failed_backup_reports_stage_without_reconciling(monkeypatch):
    monkeypatch.setattr(remote, 'connection', lambda: ('client','folder'))
    monkeypatch.setattr(remote, 'recover', lambda *a: 0)
    monkeypatch.setattr(r, 'capture_live_games', lambda games, **kw: (games, {'receipts_created':4}))
    def fail(*a): raise ValueError('sensitive provider detail')
    monkeypatch.setattr(remote, 'backup', fail)
    monkeypatch.setattr(r, 'reconcile', lambda **kw: pytest.fail('must back up first'))
    _, health = remote.collect_durable([])
    assert health['failed_stage'] == 'backup_before_reconciliation'
    assert health['remote_backup_verified'] is False
    assert 'sensitive' not in str(health)


def test_missing_manifest_record_does_not_partially_restore(fixture, tmp_path):
    collect(fixture)
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    del store.data[next(k for k in store.data if k.startswith(remote.RECORD_PREFIX))]
    target = tmp_path/'missing.sqlite3'
    with pytest.raises(ValueError): remote.recover(store, target)
    assert r.read('receipts', target) == {}


def test_restore_diagnostic_distinguishes_conflicts_without_leaking_secrets():
    detail = remote.restore_diagnostic('restore_uploaded_backup', ValueError('Receipt recovery conflict'))
    assert detail['reason'] == 'LOCAL_RECORD_CONFLICT'
    detail = remote.restore_diagnostic('backup_and_verify_drive', ValueError('secret URL token=123'))
    assert detail['reason'] == 'RESTORE_OPERATION_FAILED'
    assert 'token' not in str(detail)
    assert detail['remote_backup_verified'] is False


def test_recovered_records_and_second_backup_make_no_remote_writes(fixture, tmp_path):
    collect(fixture)
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    target = tmp_path/'incremental.sqlite3'
    remote.recover(store, target)
    calls = []
    store.put_object = lambda **kw: calls.append('put')
    store.get_object = lambda **kw: pytest.fail('verified unchanged objects must not be reread')
    first = remote.backup(store, 'folder', target)
    second = remote.backup(store, 'folder', target)
    assert calls == []
    assert first['objects_uploaded_verified'] == second['objects_uploaded_verified'] == 0
    assert first['objects_reused'] > 0


def test_new_record_only_uploads_record_and_manifest(fixture):
    collect(fixture)
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    observation = {'source':'test', 'observed_at':'2026-09-15T12:00:00Z','payload':{'new':True}}
    r.append('observations', remote.digest(observation), observation, fixture[0])
    report = remote.backup(store, 'folder', fixture[0])
    assert report['objects_uploaded_verified'] == 2
    assert report['objects_reused'] > 0


def test_new_recovery_does_not_trust_old_client_verification(fixture, tmp_path):
    collect(fixture)
    store = Store()
    remote.backup(store, 'folder', fixture[0])
    key = next(k for k in store.data if k.startswith(remote.RECORD_PREFIX))
    store.data[key] += b'corrupt'
    with pytest.raises(ValueError): remote.recover(store, tmp_path/'fresh.sqlite3')
    assert store._receipt_verified == set()


def test_catchup_fetches_fresh_mlb_quotes_and_bounded_batch(monkeypatch):
    from app_core import odds_api
    import core.streamlit_pipeline as pipeline
    seen = []
    class Client:
        def __init__(self, key, **kw):
            assert key == 'test-key'
            assert kw['markets'] == 'spreads,totals'
        def get_odds(self, sport):
            assert sport == 'baseball_mlb'
            return [{'id':'fresh'}]
    monkeypatch.setattr(pipeline, '_get_odds_api_key', lambda: 'test-key')
    monkeypatch.setattr(odds_api, 'TheOddsAPIClient', Client)
    monkeypatch.setattr(remote, 'collect_durable', lambda games, **kw: (seen.append((games,kw)), {'ok':True}))
    assert remote.catch_up_history() == {'ok':True}
    assert seen == [([{'id':'fresh'}], {'max_feeds':100})]
