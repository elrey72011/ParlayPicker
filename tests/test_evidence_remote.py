from io import BytesIO
import json

import pytest

from app_core import evidence_remote as remote
from app_core import prediction_evidence as evidence
from test_prediction_evidence import frozen, fixture_frames


class PreconditionFailed(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class MemoryObjectStore:
    def __init__(self):
        self.objects = {}
        self.fail = False

    def put_object(self, *, Key, Body, IfNoneMatch, **kwargs):
        assert IfNoneMatch == '*'
        if self.fail:
            raise OSError('network unavailable')
        if Key in self.objects:
            raise PreconditionFailed()
        self.objects[Key] = Body

    def get_object(self, *, Key, **kwargs):
        return {'Body': BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        assert name == 'list_objects_v2'
        return self

    def paginate(self, *, Prefix, **kwargs):
        # One object per page ensures the restore does not silently stop at page 1.
        for key in sorted(self.objects, reverse=True):
            if key.startswith(Prefix):
                yield {'Contents': [{'Key': key}]}


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'test-shared-folder')
    monkeypatch.setattr(remote, '_status', {})
    monkeypatch.setattr(remote, '_restored', set())
    client = MemoryObjectStore()
    monkeypatch.setattr(remote, '_client', lambda: client)
    return client


def save_fixture(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    return evidence.capture_run(context, audit, final, audit, path=db)


def test_backup_restore_new_filesystem_and_grade(frozen, cloud, tmp_path):
    _, db, _ = frozen
    _, card = save_fixture(frozen)
    scores = card[['snapshot_id', 'matchup_id']].copy()
    scores['actual_home_score'], scores['actual_away_score'] = 6, 4
    evidence.record_scores(scores, path=db)
    scores['actual_home_score'], scores['actual_away_score'] = 2, 1
    evidence.record_scores(scores, path=db)
    assert remote.sync(db)
    assert remote.sync(db)  # All existing objects are verified, not overwritten.
    fresh = tmp_path / 'new-instance' / 'evidence.sqlite3'
    assert remote.restore(fresh) == 1
    assert remote.restore(fresh) == 0
    original, _ = evidence.materialize(db)
    restored, _ = evidence.materialize(fresh)
    assert original.to_csv(index=False) == restored.to_csv(index=False)
    assert restored.candidate_outcome.tolist() == ['LOSS', 'WIN']
    assert remote.remote_status()['restored_snapshots'] == 1


def test_outage_keeps_local_snapshot_and_retry_succeeds(frozen, cloud):
    _, db, _ = frozen
    save_fixture(frozen)
    cloud.fail = True
    assert remote.sync(db) is False
    assert len(evidence.load_snapshots(db)) == 1
    assert remote.remote_status()['status'] == 'error'
    cloud.fail = False
    assert remote.sync(db)
    assert remote.remote_status()['error'] is None


def test_corrupt_remote_payload_rejected_before_any_restore(frozen, cloud, tmp_path):
    _, db, _ = frozen
    save_fixture(frozen)
    remote.sync(db)
    key = next(k for k in cloud.objects if '/snapshots/' in k)
    payload = json.loads(cloud.objects[key])
    payload['row'][3] += 'tampered'
    cloud.objects[key] = json.dumps(payload).encode()
    destination = tmp_path / 'fresh.sqlite3'
    with pytest.raises(ValueError, match='hash mismatch'):
        remote.restore(destination)
    assert not destination.exists()


def test_existing_remote_identity_cannot_be_overwritten(frozen, cloud):
    _, db, _ = frozen
    save_fixture(frozen)
    assert remote.sync(db)
    key = next(k for k in cloud.objects if '/snapshot_runtime/' in k)
    item = json.loads(cloud.objects[key])
    item['row'][1] = 'another-runtime'
    cloud.objects[key] = json.dumps(item).encode()
    assert not remote.sync(db)
    assert 'conflicts with local evidence' in remote.remote_status()['error']
    assert remote.remote_status()['operation'] == 'upload_verify:snapshot_runtime'
    assert json.loads(cloud.objects[key])['row'][1] == 'another-runtime'


def test_concurrent_bundle_registration_keeps_first_freeze(frozen, cloud):
    context, db, _ = frozen
    assert remote.sync(db)
    encoded = json.dumps(context['manifest'], sort_keys=True, separators=(',', ':'))
    frozen_at = remote.register_bundle(context['model_version'], '2026-09-04T00:00:00Z', encoded)
    assert frozen_at == context['frozen_at']


def test_restore_conflict_rolls_back(frozen, cloud, tmp_path):
    _, db, _ = frozen
    save_fixture(frozen)
    assert remote.sync(db)
    fresh = tmp_path / 'fresh.sqlite3'
    remote.restore(fresh)
    key = next(k for k in cloud.objects if '/snapshot_runtime/' in k)
    item = json.loads(cloud.objects[key])
    item['row'][1] = 'conflict'
    cloud.objects[key] = json.dumps(item).encode()
    with pytest.raises(ValueError, match='conflicts'):
        remote.restore(fresh)
    assert len(evidence.load_snapshots(fresh)) == 1


def test_unconfigured_storage_never_contacts_drive(monkeypatch):
    monkeypatch.delenv('PARLAYPICKER_DRIVE_FOLDER_ID', raising=False)
    monkeypatch.setattr(remote, '_client', lambda: pytest.fail('Drive should not be used'))
    assert remote.sync() is False
    assert remote.restore() == 0
    assert remote.remote_status()['status'] == 'not_configured'


def test_default_capture_restores_and_preserves_existing_freeze(frozen, cloud, monkeypatch, tmp_path):
    context, db, root = frozen
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(db.parent))
    started = evidence.begin_run({"use_ml": False}, root=root)
    assert started["frozen_at"] == context["frozen_at"]
    audit, final = fixture_frames()
    saved, _ = evidence.capture_run(started, audit, final, audit)
    assert remote.remote_status()["status"] == "synced"
    fresh = tmp_path / "empty-host"
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(fresh))
    restored = evidence.load_snapshots()
    assert len(restored) == 1
    assert restored[0][0] == saved.snapshot_id.iloc[0]
    assert evidence.begin_run({"use_ml": False}, root=root)["frozen_at"] == context["frozen_at"]
