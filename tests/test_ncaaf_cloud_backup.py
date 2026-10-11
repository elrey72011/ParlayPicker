"""SYNTHETIC cloud backup drills only. External transports blocked by conftest."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from app_core import ncaaf_cloud_backup as cloud, ncaaf_pilot_setup as local
from app_core import prediction_evidence

SECRET = b'SYNTHETIC-owner-vault-passphrase-only'
BACKUP = 'SYNTHETIC-initial'


@pytest.fixture
def ready(tmp_path, monkeypatch):
    source = tmp_path / 'SYNTHETIC-original.sqlite3'
    with closing(prediction_evidence.connect(source)) as db:
        for table in local.REPLAY: db.execute(f'DROP TABLE {table}')
        db.execute("INSERT INTO bundles VALUES ('SYNTHETIC', '2026-10-10T00:00:00Z', '{}')")
        db.commit()
    root = tmp_path / 'SYNTHETIC-private'; root.mkdir()
    monkeypatch.setattr(local, 'inspect_security', lambda p: dict(path=str(p), status='VERIFIED',
        evidence='SYNTHETIC_OS_PROOF', private_acl=True, encrypted=True, no_reparse=True))
    scope = dict(version=cloud.SETUP_VERSION, source=str(source.resolve()), root=str(root.resolve()))
    monkeypatch.setattr(cloud, 'AUTHORIZED_SETUPS', {'SYNTHETIC-setup': cloud.sha(cloud.encode(scope))})
    original = source.read_bytes()
    receipt = cloud.setup(source, root, authorization_ref='SYNTHETIC-setup')
    assert source.read_bytes() == original
    assert receipt['status'] == 'LOCAL_PREPARED_CLOUD_BACKUP_PENDING'
    assert not receipt['collection_ready'] and not receipt['cloud_readback_verified']
    body = b'{"evidence_label":"SYNTHETIC","body":"PRIVATE_BODY_CANARY"}'
    (root / 'custody/SYNTHETIC-body.json').write_bytes(body)
    (root / 'custody/SYNTHETIC-attempts.jsonl').write_bytes(
        b'{"evidence_label":"SYNTHETIC","attempt":1,"spent":true}\n')
    return source, root, receipt


def prepared(ready):
    source, root, receipt = ready
    report = cloud.prepare(root, root, backup_id=BACKUP, secret=SECRET,
        writer_stop_reference='SYNTHETIC-stopped-all-writers',
        recovery_reference='SYNTHETIC-independent-vault-access')
    raw = (root / report['filename']).read_bytes()
    return root, report, raw


class SyntheticCloud:
    def __init__(self, folder, objects):
        self.folder = deepcopy(folder); self.objects = deepcopy(objects); self.calls = []
    def metadata(self, id):
        self.calls.append(('metadata',id))
        return deepcopy(self.folder if id == self.folder['id'] else self.objects[id][0])
    def download(self, id, *, max_bytes):
        self.calls.append(('download',id,max_bytes))
        assert max_bytes == cloud.MAX_BYTES + 4096
        return self.objects[id][1]


def remote(ready, monkeypatch):
    root, report, raw = prepared(ready)
    owner = 'SYNTHETIC-owner@example.invalid'
    perms = [dict(type='user', role='owner', emailAddress=owner)]
    folder = dict(id='SYNTHETIC-folder', name='SYNTHETIC-private-backups',
        mimeType='application/vnd.google-apps.folder', trashed=False,
        permissions_complete=True, permissions=perms)
    objects = {}
    scope = dict(version=cloud.VERSION, folder_id=folder['id'], folder_name=folder['name'],
        owner=owner, allowed_users=[owner], independent_key_recovery_reference='SYNTHETIC-owner-vault-review')
    for kind, name, body in [('bundle',report['filename'],raw),
        ('manifest',BACKUP+'.manifest.json',(root/(BACKUP+'.manifest.json')).read_bytes())]:
        id = 'SYNTHETIC-'+kind
        objects[id] = (dict(id=id,name=name,parents=[folder['id']],trashed=False,
            permissions_complete=True,permissions=perms),body)
        scope[kind] = dict(file_id=id,name=name,sha256=cloud.sha(body))
    monkeypatch.setattr(cloud, 'AUTHORIZED_READBACKS', {'SYNTHETIC-readback':scope})
    return root, report, SyntheticCloud(folder,objects)


def test_local_setup_remains_cloud_pending_and_original_preserved(ready):
    source, root, receipt = ready
    assert not receipt['accepted'] and not receipt['inference']
    assert not receipt['collection_ready'] and not receipt['recovery_verified']
    with closing(local._readonly(root/'prediction/evidence.sqlite3')) as db:
        assert local.contents(db) == receipt['core']
        assert len(local._replay_schema(db)['immutable_triggers']) == 6
    with pytest.raises(ValueError,match='ALREADY_ATTEMPTED'):
        cloud.setup(source,root,authorization_ref='SYNTHETIC-setup')
    assert local.AUTHORIZED_SETUPS == {}


def test_security_unknown_stops_before_creation(tmp_path, monkeypatch):
    source=tmp_path/'SYNTHETIC-source'; root=tmp_path/'SYNTHETIC-root'; root.mkdir()
    scope=dict(version=cloud.SETUP_VERSION,source=str(source.resolve()),root=str(root.resolve()))
    monkeypatch.setattr(cloud,'AUTHORIZED_SETUPS',{'SYNTHETIC':cloud.sha(cloud.encode(scope))})
    monkeypatch.setattr(local,'inspect_security',lambda p:dict(status='UNKNOWN'))
    with pytest.raises(ValueError,match='SECURITY_UNVERIFIED'):
        cloud.setup(source,root,authorization_ref='SYNTHETIC')
    assert list(root.iterdir()) == []


def test_ciphertext_has_no_private_bodies_keys_or_plain_manifest(ready):
    root, report, raw=prepared(ready)
    assert b'PRIVATE_BODY_CANARY' not in raw and SECRET not in raw
    side=(root/(BACKUP+'.manifest.json')).read_bytes()
    assert b'PRIVATE_BODY_CANARY' not in side and b'custody/' not in side and SECRET not in side
    assert report['status'] == 'ENCRYPTED_LOCAL_ONLY_UPLOAD_NOT_AUTHORIZED'
    manifest,files=cloud.decrypt(raw,SECRET,report['ciphertext_sha256'])
    assert files['custody/SYNTHETIC-body.json'] == (root/'custody/SYNTHETIC-body.json').read_bytes()
    assert 'custody/SYNTHETIC-attempts.jsonl' in manifest['journal_files']
    assert not report['cloud_readback_verified'] and not report['recovery_verified']


def test_committed_working_wal_included_without_checkpoint(ready):
    source,root,receipt=ready
    with closing(sqlite3.connect(root/'prediction/evidence.sqlite3')) as writer:
        writer.execute('PRAGMA journal_mode=WAL');writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute("INSERT INTO bundles VALUES ('SYNTHETIC-WAL', '2026-10-10T00:00:01Z', '{}')")
        writer.commit()
        main=root/'prediction/evidence.sqlite3'; wal=Path(str(main)+'-wal')
        main_before=main.read_bytes();wal_before=wal.read_bytes()
        _,report,raw=prepared(ready)
        _,files=cloud.decrypt(raw,SECRET,report['ciphertext_sha256'])
        with closing(sqlite3.connect(':memory:')) as db:
            db.deserialize(files['database/working-snapshot.sqlite3'])
            assert db.execute("SELECT COUNT(*) FROM bundles WHERE version='SYNTHETIC-WAL'").fetchone()[0] == 1
        assert main.read_bytes() == main_before and wal.read_bytes() == wal_before


@pytest.mark.parametrize('issue',['secret','corruption','expected_hash'])
def test_cipher_rejection_preserves_originals(ready,issue):
    root,report,raw=prepared(ready)
    original=(root/'prediction/evidence.sqlite3').read_bytes()
    secret=SECRET if issue!='secret' else b'SYNTHETIC-another-vault-secret'
    expected=report['ciphertext_sha256']
    if issue=='corruption': raw=raw[:-1]+bytes([raw[-1]^1])
    if issue=='expected_hash': expected='0'*64
    with pytest.raises(ValueError):
        cloud.recover(raw,secret,expected,root/'SYNTHETIC-recovery')
    assert not (root/'SYNTHETIC-recovery').exists()
    assert (root/'prediction/evidence.sqlite3').read_bytes() == original


def test_local_recovery_is_isolated_not_claimed_as_cloud(ready):
    root,report,raw=prepared(ready)
    record=cloud.recover(raw,SECRET,report['ciphertext_sha256'],root/'SYNTHETIC-isolated')
    assert record['exact_readback'] and record['spent_journals_preserved']
    assert not record['cloud_origin_verified'] and not record['collection_ready']
    assert (root/'SYNTHETIC-isolated/custody/SYNTHETIC-attempts.jsonl').read_bytes() == (root/'custody/SYNTHETIC-attempts.jsonl').read_bytes()
    with closing(sqlite3.connect(root/'SYNTHETIC-isolated/database/working-snapshot.sqlite3')) as db:
        db.execute("INSERT INTO snapshots VALUES ('SYNTHETIC-snapshot','SYNTHETIC','2026-10-10T00:00:01Z','[]','[]','{}','SYNTHETIC')")
        db.execute("INSERT INTO research_replay_sources VALUES ('SYNTHETIC-snapshot','SYNTHETIC','SYNTHETIC','{}','SYNTHETIC')")
        db.execute("INSERT INTO research_replay_exports VALUES ('SYNTHETIC','SYNTHETIC','{}')")
        db.execute("INSERT INTO research_source_intakes VALUES ('SYNTHETIC','{}','SYNTHETIC')")
        for table in local.REPLAY:
            for action in ('UPDATE','DELETE'):
                sql=f"{action} {table}" if action=='DELETE' else f"UPDATE {table} SET rowid=rowid"
                if action=='DELETE': sql=f'DELETE FROM {table}'
                with pytest.raises(sqlite3.IntegrityError,match='append-only'): db.execute(sql)
        db.rollback()
    with pytest.raises(ValueError,match='DESTINATION_EXISTS'):
        cloud.recover(raw,SECRET,report['ciphertext_sha256'],root/'SYNTHETIC-isolated')


def test_authenticated_synthetic_reader_requires_both_files_and_recovery(ready,monkeypatch):
    root,report,reader=remote(ready,monkeypatch)
    record=cloud.verify_cloud(reader,approval_ref='SYNTHETIC-readback',recovery_parent=root,secret=SECRET)
    assert record['cloud_readback_verified'] and record['spent_journals_preserved']
    assert not record['collection_ready'] and not record['accepted'] and not record['inference']
    assert [c[1] for c in reader.calls if c[0]=='download'] == ['SYNTHETIC-bundle','SYNTHETIC-manifest']
    assert cloud.AUTHORIZED_SETUPS.keys() == {'SYNTHETIC-setup'}
    assert local.AUTHORIZED_SETUPS == {}


@pytest.mark.parametrize('issue',['missing_permissions','public','domain','wrong_folder',
    'wrong_file','wrong_parent','extra_reader','corrupt_media','wrong_manifest'])
def test_remote_failure_never_reports_verified_or_overwrites(ready,monkeypatch,issue):
    root,report,reader=remote(ready,monkeypatch)
    if issue=='missing_permissions': reader.folder['permissions_complete']=False
    if issue=='public': reader.folder['permissions'].append(dict(type='anyone',role='reader'))
    if issue=='domain': reader.folder['permissions'].append(dict(type='domain',role='reader',domain='example.invalid'))
    if issue=='wrong_folder': reader.folder['id']='SYNTHETIC-unrelated-folder'
    meta,raw=reader.objects['SYNTHETIC-bundle']
    if issue=='wrong_file': meta['name']='SYNTHETIC-other.ppbackup'
    if issue=='wrong_parent': meta['parents']=['SYNTHETIC-other-folder']
    if issue=='extra_reader': meta['permissions'].append(dict(type='user',role='reader',emailAddress='SYNTHETIC-other@example.invalid'))
    if issue=='corrupt_media': reader.objects['SYNTHETIC-bundle']=(meta,b'SYNTHETIC-corrupt')
    if issue=='wrong_manifest':
        mm,_=reader.objects['SYNTHETIC-manifest'];reader.objects['SYNTHETIC-manifest']=(mm,b'{}')
    before=(root/'custody/SYNTHETIC-attempts.jsonl').read_bytes()
    with pytest.raises((ValueError,KeyError)):
        cloud.verify_cloud(reader,approval_ref='SYNTHETIC-readback',recovery_parent=root,secret=SECRET)
    assert not (root/(BACKUP+'-isolated')).exists()
    assert (root/'custody/SYNTHETIC-attempts.jsonl').read_bytes() == before


def test_unapproved_readback_makes_zero_cloud_calls(ready,monkeypatch):
    root,report,reader=remote(ready,monkeypatch)
    with pytest.raises(ValueError,match='AUTHORIZATION_UNTRUSTED'):
        cloud.verify_cloud(reader,approval_ref='not-approved',recovery_parent=root,secret=SECRET)
    assert reader.calls == []


@pytest.mark.parametrize('issue',['body_limit','credential_file','changed_journal','repeat'])
def test_preparation_stops_honestly(ready,monkeypatch,issue):
    source,root,receipt=ready
    if issue=='body_limit': monkeypatch.setattr(cloud,'MAX_BYTES',32)
    if issue=='credential_file': (root/'custody/secrets.toml').write_bytes(b'SYNTHETIC-not-a-secret')
    if issue=='changed_journal':
        original=cloud._snapshot
        def changed(path, secured_root):
            raw=original(path, secured_root)
            with (root/'custody/SYNTHETIC-attempts.jsonl').open('ab') as f: f.write(b'{"attempt":2}\n')
            return raw
        monkeypatch.setattr(cloud,'_snapshot',changed)
    if issue=='repeat': prepared(ready)
    with pytest.raises(ValueError):
        prepared(ready)


def test_unmodified_v1_and_empty_production_catalogs():
    assert cloud.AUTHORIZED_SETUPS == {} and cloud.AUTHORIZED_READBACKS == {}
    assert local.AUTHORIZED_SETUPS == {}
    with pytest.raises(TypeError): local.setup('SYNTHETIC','SYNTHETIC',authorization_ref='SYNTHETIC')


@pytest.mark.parametrize('raw',[b'{"api_key":"SYNTHETIC-secret"}',
    b'{"api_key":"SYNTHETIC-interrupted', b'{"url":"https://example.invalid?apiKey=SYNTHETIC"}'])
def test_credential_echo_rejects_whole_package_without_stripping(ready,raw):
    source,root,_=ready
    p=root/'custody/SYNTHETIC-credential.jsonl';p.write_bytes(raw)
    with pytest.raises(ValueError,match='CREDENTIAL'): prepared(ready)
    assert p.read_bytes() == raw and not (root/(BACKUP+'.ppbackup')).exists()


def test_cloud_timeout_has_no_retry_or_local_recovery(ready,monkeypatch):
    root,report,reader=remote(ready,monkeypatch)
    attempts=[]
    def timeout(id,*,max_bytes):
        attempts.append(id)
        raise TimeoutError('SYNTHETIC cloud timeout')
    reader.download=timeout
    with pytest.raises(TimeoutError):
        cloud.verify_cloud(reader,approval_ref='SYNTHETIC-readback',recovery_parent=root,secret=SECRET)
    assert attempts == ['SYNTHETIC-bundle']
    assert not (root/(BACKUP+'-isolated')).exists()


def test_unresolved_key_recovery_stops_before_cloud_access(ready,monkeypatch):
    root,report,reader=remote(ready,monkeypatch)
    cloud.AUTHORIZED_READBACKS['SYNTHETIC-readback'].pop('independent_key_recovery_reference')
    with pytest.raises(ValueError,match='KEY_RECOVERY_UNVERIFIED'):
        cloud.verify_cloud(reader,approval_ref='SYNTHETIC-readback',recovery_parent=root,secret=SECRET)
    assert reader.calls == []


def test_altered_original_recovery_snapshot_rejected_before_encryption(ready):
    source,root,_=ready
    backup=root/'backups/pre-migration.sqlite3'
    with backup.open('ab') as f: f.write(b'SYNTHETIC-original-altered')
    before=source.read_bytes()
    with pytest.raises(ValueError,match='ORIGINAL_BACKUP_CHANGED'): prepared(ready)
    assert source.read_bytes() == before
    assert not (root/(BACKUP+'.ppbackup')).exists()

