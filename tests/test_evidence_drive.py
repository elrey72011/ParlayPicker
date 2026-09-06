from email import message_from_bytes
import json

import pytest

from app_core.evidence_drive import API, AlreadyExists, DriveStore
from app_core import evidence_remote as remote
from test_prediction_evidence import frozen
from test_evidence_remote import save_fixture
from app_core import prediction_evidence as evidence


class Response:
    def __init__(self, data=None, content=b''):
        self.data, self.content = data, content

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class DriveSession:
    def __init__(self):
        self.files = []
        self.shared = True
        self.incomplete = False

    def get(self, url, params, timeout):
        assert timeout > 0
        assert params.get('supportsAllDrives') == 'true'
        if url == API + '/folder':
            return Response({'id': 'folder', 'driveId': 'shared' if self.shared else None,
                             'mimeType': 'application/vnd.google-apps.folder'})
        if params.get('alt') == 'media':
            return Response(content=next(f['content'] for f in self.files if url.endswith('/' + f['id'])))
        assert url == API
        assert params['corpora'] == 'drive' and params['driveId'] == 'shared'
        assert "'folder' in parents" in params['q']
        files = self.files
        if " and name = '" in params['q']:
            name = params['q'].split(" and name = '")[1][:-1]
            files = [f for f in files if f['name'] == name]
        offset = int(params.get('pageToken', 0))
        data = {'files': [{k: f[k] for k in ('id', 'name')} for f in files[offset:offset+1]],
                'incompleteSearch': self.incomplete}
        if offset + 1 < len(files):
            data['nextPageToken'] = str(offset + 1)
        return Response(data)

    def post(self, url, params, headers, data, timeout):
        assert url == 'https://www.googleapis.com/upload/drive/v3/files'
        assert params['uploadType'] == 'multipart' and params['supportsAllDrives'] == 'true'
        message = message_from_bytes(('Content-Type: ' + headers['Content-Type'] + '\r\n\r\n').encode() + data)
        parts = message.get_payload()
        metadata = json.loads(parts[0].get_payload(decode=True))
        assert metadata['parents'] == ['folder']
        item = {'id': str(len(self.files) + 1), 'name': metadata['name'],
                'content': parts[1].get_payload(decode=True)}
        self.files.append(item)
        return Response({'id': item['id']})


def test_drive_creation_readback_and_duplicate_conflict():
    session = DriveSession()
    store = DriveStore('folder', session=session)
    store.put_object(Key='prefix/snapshots/a.json', Body=b'{"x":1}', IfNoneMatch='*')
    assert store.get_object(Key='prefix/snapshots/a.json')['Body'].read() == b'{"x":1}'
    with pytest.raises(AlreadyExists):
        store.put_object(Key='prefix/snapshots/a.json', Body=b'changed', IfNoneMatch='*')
    session.files.append(dict(session.files[0], id='2'))
    assert store.get_object(Key='prefix/snapshots/a.json')['Body'].read() == b'{"x":1}'
    session.files[-1]['content'] = b'conflict'
    with pytest.raises(ValueError, match='conflicting duplicate'):
        store.get_object(Key='prefix/snapshots/a.json')


def test_personal_drive_folder_is_rejected():
    session = DriveSession()
    session.shared = False
    with pytest.raises(ValueError, match='Workspace Shared Drive'):
        DriveStore('folder', session=session)


def test_incomplete_listing_cannot_look_like_empty_storage():
    session = DriveSession()
    store = DriveStore('folder', session=session)
    session.incomplete = True
    with pytest.raises(ValueError, match='incomplete'):
        list(store.paginate(Prefix='prefix/'))


def test_full_replication_through_drive_adapter(frozen, tmp_path, monkeypatch):
    session = DriveSession()
    store = DriveStore('folder', session=session)
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setattr(remote, '_client', lambda: store)
    monkeypatch.setattr(remote, '_status', {})
    _, db, _ = frozen
    save_fixture(frozen)
    assert remote.sync(db)
    before = len(session.files)
    assert remote.sync(db)
    assert len(session.files) == before
    restored = tmp_path / 'fresh.sqlite3'
    assert remote.restore(restored) == 1
    assert evidence.load_snapshots(db)[0][1].to_csv(index=False) == evidence.load_snapshots(restored)[0][1].to_csv(index=False)
