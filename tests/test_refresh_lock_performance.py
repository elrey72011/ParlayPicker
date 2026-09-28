from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
import json
import logging

import pytest

from app_core import evidence_remote
from app_core.evidence_drive import API, DriveStore
from app_core.performance_spans import PerformanceSpan, operation_ids
from app_core.public_history import History, LockSavePartial, digest
from test_public_history import pub


class Response:
    def __init__(self, data=None, content=b''):
        self._data, self.content = data, content

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class PagedDriveSession:
    """Deterministic fake provider with deliberately small listing pages."""
    def __init__(self, files=None):
        self.files = list(files or [])
        self.list_requests = 0
        self.media_requests = []

    def get(self, url, params, timeout):
        if url == API + '/folder':
            return Response({'id': 'folder', 'driveId': 'drive',
                             'mimeType': 'application/vnd.google-apps.folder', 'trashed': False})
        if params.get('alt') == 'media':
            file_id = url.rsplit('/', 1)[-1]
            self.media_requests.append(file_id)
            return Response(content=next(item['content'] for item in self.files if item['id'] == file_id))
        self.list_requests += 1
        offset = int(params.get('pageToken', 0))
        page = self.files[offset:offset + 2]
        data = {'files': [{key: item[key] for key in ('id', 'name', 'sha256Checksum') if key in item}
                          for item in page], 'incompleteSearch': False}
        if offset + 2 < len(self.files):
            data['nextPageToken'] = str(offset + 2)
        return Response(data)


def remote_file(file_id, name, raw, *, checksum=True):
    value = {'id': file_id, 'name': name, 'content': raw}
    if checksum:
        value['sha256Checksum'] = sha256(raw).hexdigest()
    return value


def test_one_complete_inventory_serves_six_prefixes_and_warm_cache(tmp_path):
    prefixes = [f'evidence/{name}/' for name in (
        'validation_plans', 'closing_observations', 'bundles', 'snapshots',
        'snapshot_runtime', 'score_revisions')]
    session = PagedDriveSession([
        remote_file(str(index), prefix + 'record.json', prefix.encode())
        for index, prefix in enumerate(prefixes)
    ] + [remote_file('other', 'unrelated/history.json', b'unrelated')])
    store = DriveStore('folder', session=session)

    first_inventory = store.discover_complete_inventory(namespace='evidence/')
    first = store.read_verified_prefixes(
        Prefixes=prefixes, inventory=first_inventory, cache_dir=tmp_path)
    assert all(len(first[prefix]) == 1 for prefix in prefixes)
    assert session.list_requests == 4  # seven items across four fake pages
    assert len(session.media_requests) == 6 and 'other' not in session.media_requests

    second_inventory = store.discover_complete_inventory(namespace='evidence/')
    second = store.read_verified_prefixes(
        Prefixes=prefixes, inventory=second_inventory, cache_dir=tmp_path)
    assert second == first
    assert session.list_requests == 8
    assert len(session.media_requests) == 6
    assert store.last_read_report.objects_reused == 6
    assert store.last_read_report.objects_downloaded == 0


def test_evidence_restore_uses_one_inventory_even_for_six_empty_tables(tmp_path, monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setenv('PARLAYPICKER_SHARED_INVENTORY', '1')

    class Client:
        def __init__(self):
            self.discoveries = 0
            self.requests = None
            self.last_read_report = SimpleNamespace(
                listing_traversals=1, listing_pages=3, metadata_items_seen=12,
                objects_downloaded=0, objects_reused=0, bytes_downloaded=0,
                cache_hits=0, cache_misses=0)

        def discover_complete_inventory(self, **kwargs):
            self.discoveries += 1
            return SimpleNamespace(scope_hash='scope', operation_id=kwargs['operation_id'])

        def read_verified_prefixes(self, *, Prefixes, **kwargs):
            self.requests = tuple(Prefixes)
            return {prefix: [] for prefix in Prefixes}

    client = Client()
    assert evidence_remote.restore(tmp_path / 'empty.sqlite3', client=client) == 0
    assert client.discoveries == 1
    assert len(client.requests) == len(evidence_remote.TABLES) == 6


def test_inventory_feature_flag_has_safe_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setenv('PARLAYPICKER_SHARED_INVENTORY', '0')

    class Client:
        def __init__(self):
            self.reads = []

        def discover_complete_inventory(self, **kwargs):
            pytest.fail('optimized inventory must be disabled')

        def read_verified_prefixes(self, **kwargs):
            pytest.fail('optimized reads must be disabled')

        def read_objects(self, *, Prefix):
            self.reads.append(Prefix)
            return []

    client = Client()
    evidence_remote.restore(tmp_path / 'legacy.sqlite3', client=client)
    assert len(client.reads) == 6


def test_restore_once_detects_database_replacement_at_same_path(tmp_path, monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setattr(evidence_remote, '_restored', set())
    path = tmp_path / 'evidence.sqlite3'
    calls = []

    def restore(target, **kwargs):
        calls.append(target)
        if not path.exists():
            path.write_bytes(b'first-generation')

    monkeypatch.setattr(evidence_remote, 'restore', restore)
    evidence_remote.restore_once(path)
    evidence_remote.restore_once(path)
    assert len(calls) == 1
    replacement = tmp_path / 'replacement.sqlite3'
    replacement.write_bytes(b'second-generation')
    path.unlink()
    replacement.rename(path)
    evidence_remote.restore_once(path)
    assert len(calls) == 2


def test_lock_snapshot_uses_one_fresh_inventory_for_locks_and_removals(tmp_path, monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path / 'cache'))
    session = PagedDriveSession()
    drive = DriveStore('folder', session=session)
    history = History('site-1234', 'folder', drive)
    first = {'id': 'one', 'date': '2026-09-28', 'legs': []}
    second = {'id': 'two', 'date': '2026-09-28', 'legs': []}
    removal = {'lock_hash': digest(first), 'lock_id': 'one', 'removed_at': '2026-09-28T12:00:00Z',
               'reason': 'owner correction', 'lock': first}
    session.files.extend([
        remote_file('l1', history.prefix + 'locks/one.json', json.dumps(first).encode()),
        remote_file('l2', history.prefix + 'locks/two.json', json.dumps(second).encode()),
        remote_file('r1', history.prefix + 'lock_removals/one.json', json.dumps(removal).encode()),
    ])
    snapshot = history.active_lock_snapshot()
    assert [row['id'] for row in snapshot.active] == ['two']
    assert session.list_requests == 2  # one traversal, two fake pages

    third = {'id': 'three', 'date': '2026-09-28', 'legs': []}
    session.files.append(remote_file('l3', history.prefix + 'locks/three.json', json.dumps(third).encode()))
    refreshed = history.active_lock_snapshot()
    assert {row['id'] for row in refreshed.active} == {'two', 'three'}
    assert session.list_requests == 4  # a distinct fresh post-write/retry phase


def test_verified_write_receipt_and_partial_batch_report(monkeypatch):
    class Exists(Exception):
        response = {'Error': {'Code': 'PreconditionFailed'}}

    class Memory:
        def __init__(self):
            self.data = {}
            self.lock_writes = 0

        def put_object(self, *, Key, Body, **kwargs):
            if Key in self.data:
                raise Exists()
            if '/locks/' in Key:
                self.lock_writes += 1
                if self.lock_writes == 2:
                    raise OSError('simulated provider failure')
            self.data[Key] = Body

        def get_object(self, *, Key):
            from io import BytesIO
            return {'Body': BytesIO(self.data[Key])}

        def get_paginator(self, name):
            return self

        def paginate(self, *, Prefix):
            return [{'Contents': [{'Key': key} for key in self.data if key.startswith(Prefix)]}]

    client = Memory()
    history = History('site-1234', 'folder', client)
    receipt = history.put_verified('checks/one.json', {'value': 1})
    assert receipt.created and receipt.verification_status == 'exact_readback_verified'
    assert receipt.content_sha256 == sha256(b'{"value":1}').hexdigest()

    rows = [{'id': 'a'}, {'id': 'b'}]
    monkeypatch.setattr('app_core.locked_picks.lock_candidates', lambda package, at: rows)
    monkeypatch.setattr(history, 'archive', lambda package: 'archived')
    with pytest.raises(LockSavePartial) as error:
        history.lock_picks({}, ['a', 'b'])
    assert len(error.value.saved_keys) == 1
    assert len(error.value.failed_keys) == 1
    assert len(history.last_write_receipts) == 1


def test_publication_retry_reconciles_without_lock_write(monkeypatch):
    from app.ui import lock_picks, public_results, sftp_publish

    class Status:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def update(self, **kwargs): pass

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {'lock_publication_pending': {'state': 'PUBLICATION_FAILED'}}
            self.errors = []
        def status(self, *args, **kwargs): return Status()
        def warning(self, value): pass
        def error(self, value): self.errors.append(value)
        def rerun(self): pass

    class Store:
        def active_lock_snapshot(self, **kwargs):
            return SimpleNamespace(active=(), removals=())
        def lock_picks(self, *args, **kwargs):
            pytest.fail('publication retry must never write locks')

    fake = FakeStreamlit()
    sent = []
    monkeypatch.setattr(lock_picks, 'st', fake)
    monkeypatch.setattr(public_results, 'history', lambda setting: Store())
    monkeypatch.setattr(sftp_publish, 'publish_action', lambda package, setting: sent.append(deepcopy(package)) or 'Published: verified')
    package = pub()['package']
    original = deepcopy(package)
    saved = {'publications': [], 'revisions': [], 'imports': [], 'locks': [], 'rows': []}
    lock_picks.retry_lock_publication(package, lambda key: 'site', saved)
    assert len(sent) == 1
    assert 'lock_publication_pending' not in fake.session_state
    assert package == original


def test_correlated_spans_keep_unknown_counts_null_and_parent_wall_time(caplog):
    ids = operation_ids(refresh_run_id='refresh')
    with caplog.at_level(logging.WARNING):
        with PerformanceSpan('parent', ids=ids):
            with PerformanceSpan('child', ids=ids) as child:
                child.set(records_returned=0)
    records = [json.loads(record.message.split(' ', 1)[1])
               for record in caplog.records if record.message.startswith('PERFORMANCE_SPAN ')]
    by_operation = {record['operation']: record for record in records}
    assert by_operation['child']['parent_span_id'] == by_operation['parent']['span_id']
    assert by_operation['parent']['records_returned'] is None
    assert by_operation['child']['records_returned'] == 0
    assert by_operation['parent']['elapsed_ms'] >= by_operation['child']['elapsed_ms']
