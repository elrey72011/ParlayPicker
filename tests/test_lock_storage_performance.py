from copy import deepcopy
import json
from threading import Barrier, Lock, get_ident

import pytest

from app_core.evidence_drive import API, DriveStore
from app_core.public_history import History
from app_core.locked_picks import lock_candidates
from test_evidence_drive import DriveSession, Response
from test_public_history import pub
from test_locked_picks import AT


class CountingSession(DriveSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.closed = False

    def get(self, url, params, timeout):
        self.calls.append((url, dict(params)))
        return super().get(url, params, timeout)

    def close(self):
        self.closed = True


def test_bulk_read_reuses_listing_ids_and_checks_every_duplicate():
    session = CountingSession()
    session.files = [{'id': str(i), 'name': f'locks/{i}.json', 'content': b'{}'} for i in range(8)]
    store = DriveStore('folder', session=session)
    session.calls.clear()
    old = [(item['Key'], store.get_object(Key=item['Key'])['Body'].read())
           for page in store.paginate(Prefix='locks/') for item in page['Contents']]
    old_requests = len(session.calls)
    session.calls.clear()
    assert store.read_objects(Prefix='locks/') == old
    assert len(session.calls) == old_requests - 8
    assert not any(' and name = ' in params.get('q', '') for _, params in session.calls)
    session.files.append(dict(session.files[0], id='duplicate'))
    assert store.read_objects(Prefix='locks/') == old
    session.files[-1]['content'] = b'conflicting'
    with pytest.raises(ValueError, match='conflicting duplicate'):
        store.read_objects(Prefix='locks/')
    session.incomplete = True
    with pytest.raises(ValueError, match='incomplete'):
        store.read_objects(Prefix='locks/')


def test_bulk_read_includes_new_file_when_search_index_is_delayed():
    class Delayed(CountingSession):
        def get(self, url, params, timeout):
            if url == API:
                return Response({'files': []})
            return super().get(url, params, timeout)
    session = Delayed()
    store = DriveStore('folder', session=session)
    store.put_object(Key='locks/new', Body=b'original', IfNoneMatch='*')
    assert store.read_objects(Prefix='locks/') == [('locks/new', b'original')]
    session.files[0]['content'] = b'changed'
    assert store.read_objects(Prefix='locks/') == [('locks/new', b'changed')]


@pytest.mark.parametrize('fail', [False, True])
def test_parallel_calls_are_bounded_isolated_closed_and_never_retried(fail):
    store = DriveStore('folder', session=CountingSession())
    sessions, calls, progress = [], [], []
    guard, barrier = Lock(), Barrier(4)
    caller = get_ident()
    def factory():
        session = CountingSession()
        sessions.append(session)
        return session
    store._session_factory = factory
    def operation(worker, item):
        with guard:
            calls.append((item, get_ident(), id(worker.session)))
        barrier.wait(timeout=5)
        if fail and item == 0:
            raise RuntimeError('uncertain upload')
        return item * 2
    def report(done, total):
        progress.append((done, total, get_ident()))
    if fail:
        with pytest.raises(RuntimeError, match='uncertain upload'):
            store.run_parallel(operation, range(4), progress=report)
    else:
        assert store.run_parallel(operation, range(8), progress=report) == list(range(0, 16, 2))
        assert len(progress) == 8
    assert len(sessions) == 4 and all(session.closed for session in sessions)
    assert len({thread for _, thread, _ in calls}) == 4
    assert len({item for item, _, _ in calls}) == len(calls)
    assert all(len({session for _, thread, session in calls if thread == identity}) == 1
               for identity in {thread for _, thread, _ in calls})
    assert all(thread == caller for _, _, thread in progress)


def test_parallel_history_save_preserves_first_write_and_reads_corrections_once(monkeypatch):
    from app_core import public_history
    monkeypatch.setattr(public_history, 'now', lambda: AT)
    backend, mutex = CountingSession(), Lock()
    sessions = []
    class Shared(CountingSession):
        def __init__(self):
            super().__init__()
            self.files = backend.files
        def get(self, *args, **kwargs):
            with mutex:
                return super().get(*args, **kwargs)
        def post(self, *args, **kwargs):
            with mutex:
                return super().post(*args, **kwargs)
    drive = DriveStore('folder', session=Shared())
    def factory():
        session = Shared()
        sessions.append(session)
        return session
    drive._session_factory = factory
    store = History('site-1234', 'folder', drive)
    package = pub()['package']
    package['games']['overall'] = [dict(package['games']['overall'][0], game=f'Team {i} at Boston') for i in range(12)]
    ids = [row['id'] for row in lock_candidates(package, AT)]
    reads, updates = [], []
    read = store._all
    def counted(kind):
        reads.append(kind)
        return read(kind)
    monkeypatch.setattr(store, '_all', counted)
    original = store.lock_picks(package, ids, progress=lambda *args: updates.append(args))
    assert reads == ['locks', 'lock_removals']
    assert len(original) == 12
    assert updates[-1] == ('Saving and verifying locks', 12, 12)
    assert sorted(store.all('locks'), key=lambda row: row['id']) == original
    newer = deepcopy(package)
    for leg in newer['games']['overall']:
        leg['odds'] = 120
    assert store.lock_picks(newer, ids) == original
    assert all(session.closed for session in sessions)
    # A fresh operation must notice a concurrent correction, not cache history.
    lock = original[0]
    store.put('lock_removals/test.json', {'lock_hash': public_history.digest(lock), 'lock_id': lock['id']})
    assert lock not in store.all('locks')


def test_failed_save_does_not_publish_success_or_website(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import lock_picks, public_results, sftp_publish
    from test_public_history import Memory
    store = History('site-1234', 'folder', Memory())
    def fail(*args, **kwargs):
        raise RuntimeError('verification failed')
    monkeypatch.setattr(store, 'lock_picks', fail)
    monkeypatch.setattr(public_results, 'history', lambda setting: store)
    monkeypatch.setattr(lock_picks, 'now', lambda: AT)
    published = []
    monkeypatch.setattr(sftp_publish, 'publish_action', lambda *args: published.append(args))
    code = "import streamlit as st\nfrom app.ui.lock_picks import render_lock_picks\n"
    code += "st.session_state['public_results_site-1234']={'publications':[],'revisions':[],'locks':[]}\n"
    code += "render_lock_picks(" + repr(pub()['package']) + ", lambda key: 'site-1234')"
    app = AppTest.from_string(code).run()
    app.button(key='lock_picks_action').click().run()
    assert not app.exception and not published
    assert not any('Your locks are saved' in message.value for message in app.success)
    assert app.error
