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


def test_final_score_must_match_its_saved_mlb_observation(fixture, tmp_path, monkeypatch):
    from datetime import timedelta
    from test_mlb_pregame_receipts import START
    collect(fixture)
    monkeypatch.setattr(r, 'now', lambda: START + timedelta(hours=4))
    assert r.reconcile(fixture[0], fetch=fixture[2])['outcomes_created'] == 1
    bundle = backup_bundle(fixture[0])
    outcome = next(iter(bundle['payload']['tables']['outcomes'].values()))
    outcome['home_score'] += 1
    bundle['sha256'] = remote.digest(bundle['payload'])
    with pytest.raises(r.Rejected, match='outcome_source_mismatch'):
        remote.restore(bundle, tmp_path / 'tampered-result.sqlite3')


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


def test_scheduled_reconciliation_restores_before_outcomes_and_backs_up(monkeypatch, tmp_path):
    from app_core import evidence_config, mlb_receipt_audit
    calls = []
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', 'configured')
    monkeypatch.setattr(evidence_config, 'service_account_info', lambda: {})
    monkeypatch.setattr(remote, 'connection', lambda: ('client', 'folder'))
    monkeypatch.setattr(remote, 'recover', lambda client, path: calls.append('restore') or 4)
    monkeypatch.setattr(r, 'reconcile', lambda path, max_games: calls.append(('reconcile', max_games)) or {'outcomes_created': 2})
    monkeypatch.setattr(r, 'export_records', lambda path, settled_only: [])
    monkeypatch.setattr(remote, 'prepare_rows', lambda records: calls.append('grade') or [])
    monkeypatch.setattr(remote, 'backup', lambda client, folder, path: calls.append('backup') or {'remote_backup_verified': True, 'backup_id': 'a'*64})
    monkeypatch.setattr(remote, 'verify_backup', lambda client, folder, report: calls.append('verify_backup') or {'remote_backup_verified': True, 'backup_id': report['backup_id']})
    monkeypatch.setattr(mlb_receipt_audit, 'audit_store', lambda path: calls.append('audit') or {'blockers': []})
    result = remote.reconcile_durable(path=tmp_path / 'receipts.sqlite3', max_games=100)
    assert calls == ['restore', ('reconcile', 100), 'grade', 'backup', 'verify_backup', 'audit']
    assert result['records_restored'] == 4
    assert result['reconciliation']['outcomes_created'] == 2
    assert result['remote_backup_verified'] is True
    assert result['audit']['remote_backup_verified'] is True


def test_scheduled_reconciliation_missing_configuration_is_classified(monkeypatch):
    monkeypatch.delenv('PARLAYPICKER_DRIVE_FOLDER_ID', raising=False)
    monkeypatch.delenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', raising=False)
    monkeypatch.setattr(remote, 'connection', lambda: pytest.fail('must fail before connection'))
    with pytest.raises(remote.ReceiptWorkflowFailure) as caught:
        remote.reconcile_durable()
    assert caught.value.report()['reason_code'] == 'MISSING_CONFIGURATION'
    assert caught.value.report()['failed_stage'] == 'configure'


def test_failed_backup_verification_prevents_audit_and_success(monkeypatch, tmp_path):
    from app_core import evidence_config, mlb_receipt_audit
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', 'configured')
    monkeypatch.setattr(evidence_config, 'service_account_info', lambda: {})
    monkeypatch.setattr(remote, 'connection', lambda: ('client', 'folder'))
    monkeypatch.setattr(remote, 'recover', lambda *a: 0)
    monkeypatch.setattr(r, 'reconcile', lambda *a, **kw: {'outcomes_created': 0})
    monkeypatch.setattr(r, 'export_records', lambda *a, **kw: [])
    monkeypatch.setattr(remote, 'backup', lambda *a: {'remote_backup_verified': True, 'backup_id': 'a'*64})
    def failed_verify(*_args):
        raise ValueError('read-back failed')
    monkeypatch.setattr(remote, 'verify_backup', failed_verify)
    monkeypatch.setattr(mlb_receipt_audit, 'audit_store', lambda *a: pytest.fail('audit must await verification'))
    with pytest.raises(remote.ReceiptWorkflowFailure) as caught:
        remote.reconcile_durable(path=tmp_path / 'receipts.sqlite3')
    assert caught.value.report()['reason_code'] == 'BACKUP_VERIFICATION_FAILURE'
    assert caught.value.report()['remote_backup_verified'] is False


def test_scheduled_cli_reports_only_error_class(monkeypatch, capsys, tmp_path):
    import sys
    from scripts import capture_mlb_pregame_receipts as cli
    def fail(**_kwargs):
        raise ValueError('private provider URL token=secret')
    monkeypatch.setattr(remote, 'reconcile_durable', fail)
    output = tmp_path / 'status.json'
    monkeypatch.setattr(sys, 'argv', ['capture_mlb_pregame_receipts.py', 'reconcile-remote', '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert 'ValueError' in capsys.readouterr().err
    assert 'secret' not in str(exc.value)
    import json
    assert json.loads(output.read_text())['reason_code'] == 'UNKNOWN_FAILURE'
    assert 'secret' not in output.read_text()


def test_scheduled_cli_success_writes_verified_audit_artifact(monkeypatch, tmp_path):
    import json
    from scripts import capture_mlb_pregame_receipts as cli
    output = tmp_path / 'audit.json'
    expected = {'schema': 'mlb-receipt-audit-v1', 'training_authorized': False,
                'remote_backup_verified': True, 'blockers': ['no_chronological_split_meets_minimums']}
    monkeypatch.setattr(remote, 'reconcile_durable', lambda **_kwargs: {
        'remote_backup_verified': True, 'backup_id': 'a'*64, 'audit': expected})
    assert cli.main(['reconcile-remote', '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'succeeded'
    assert report['remote_backup_verified'] is True
    assert report['audit'] == expected


def test_backup_verification_requires_fresh_manifest_and_all_records(fixture):
    collect(fixture)
    store = Store()
    report = remote.backup(store, 'folder', fixture[0])
    assert remote.verify_backup(store, 'folder', report)['verified_manifest_records'] > 0
    key = remote.MANIFEST_PREFIX + report['backup_id'] + '.json'
    store.data[key] += b'corrupt'
    with pytest.raises(ValueError, match='manifest verification'):
        remote.verify_backup(store, 'folder', report)


def test_uncertain_upload_is_not_retried(fixture):
    collect(fixture)
    store = Store()
    calls = []
    def timeout(**_kw):
        calls.append('put')
        raise r.requests.Timeout('unknown upload state')
    store.put_object = timeout
    with pytest.raises(r.requests.Timeout):
        remote.backup(store, 'folder', fixture[0])
    assert calls == ['put']


def test_provider_error_reason_codes_are_stage_specific():
    response = type('Response', (), {'status_code': 429})()
    err = r.requests.HTTPError('hidden URL', response=response)
    assert remote._workflow_reason('reconcile', err) == 'PROVIDER_RATE_LIMIT'
    response.status_code = 401
    assert remote._workflow_reason('reconcile', err) == 'AUTH_FAILURE'
    assert remote._workflow_reason('restore', r.Rejected('stored_hash_mismatch')) == 'RECEIPT_INTEGRITY_FAILURE'


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

def test_live_refresh_defers_settlement_but_verifies_backup(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, 'connection', lambda: ('client', 'folder'))
    monkeypatch.setattr(remote, 'recover', lambda c: calls.append('recover') or 0)
    monkeypatch.setattr(r, 'capture_live_games', lambda games, **kw: (games, {'receipts_created':4}))
    monkeypatch.setattr(remote, 'backup', lambda *a: calls.append('backup') or {'remote_backup_verified':True})
    monkeypatch.setattr(r, 'reconcile', lambda **kw: pytest.fail('settlement is deferred'))
    _, health = remote.collect_durable([], reconcile_history=False)
    assert calls == ['recover', 'backup']
    assert health['remote_backup_verified'] and health['reconciliation_deferred']
    assert set(health['stage_timings_seconds']) == {'connect','restore','capture','backup_before_reconciliation'}


def test_backup_does_not_materialize_whole_store(fixture, monkeypatch):
    collect(fixture)
    expected = backup_bundle(fixture[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("backup must stream stored records")
    monkeypatch.setattr(r, "read", forbidden)
    store = Store()
    assert remote.backup(store, "folder", fixture[0])["remote_backup_verified"]
    import json
    manifests = [json.loads(v) for k,v in store.data.items() if k.startswith(remote.MANIFEST_PREFIX)]
    assert len(manifests) == 1
    assert {t: set(refs) for t,refs in manifests[0]["tables"].items()} == {
        t: set(values) for t,values in expected["payload"]["tables"].items()}


def test_backup_corrupt_local_record_never_publishes_manifest(fixture):
    collect(fixture)
    with r.connect(fixture[0]) as db:
        db.execute("DROP TRIGGER observations_UPDATE")
        db.execute("UPDATE observations SET sha256='invalid'")
    store = Store()
    with pytest.raises(r.Rejected, match="stored_hash_mismatch"):
        remote.backup(store, "folder", fixture[0])
    assert not any(k.startswith(remote.MANIFEST_PREFIX) for k in store.data)


def test_cumulative_manifests_restore_once(fixture, tmp_path, monkeypatch):
    collect(fixture)
    store = Store()
    remote.backup(store, "folder", fixture[0])
    observation = {"source": "test", "payload": {"new": True}}
    r.append("observations", remote.digest(observation), observation, fixture[0])
    remote.backup(store, "folder", fixture[0])
    original = remote.restore
    calls = []
    def tracked(bundle, path=None):
        calls.append(1)
        return original(bundle, path)
    monkeypatch.setattr(remote, "restore", tracked)
    target = tmp_path / "union.sqlite3"
    remote.recover(store, target)
    assert len(calls) == 1
    assert backup_bundle(target) == backup_bundle(fixture[0])


def test_invalid_manifest_dependency_cannot_be_hidden_by_union(fixture, tmp_path):
    import json
    collect(fixture)
    store = Store()
    remote.backup(store, "folder", fixture[0])
    manifest = next(json.loads(v) for k,v in store.data.items() if k.startswith(remote.MANIFEST_PREFIX))
    manifest["tables"]["observations"] = {}
    key = remote.MANIFEST_PREFIX + remote.digest(manifest) + ".json"
    store.data[key] = remote.canonical(manifest)
    with pytest.raises(ValueError, match="Receipt observation missing"):
        remote.recover(store, tmp_path / "bad-union.sqlite3")
