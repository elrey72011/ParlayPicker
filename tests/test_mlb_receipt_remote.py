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
    def read_objects(self, **kw): return list(self.data.items())


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
    monkeypatch.setattr(r, 'capture_live_games', lambda games: (calls.append('capture') or games, {}))
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
    store.data[key] = store.data[key].replace(b'mlb-receipt-backup-v1',b'mlb-receipt-backup-v2')
    with pytest.raises(ValueError): remote.recover(store, tmp_path/'tampered.sqlite3')
