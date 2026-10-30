"""SYNTHETIC local prototype qualification; no authentic review or execution."""
import hashlib
from pathlib import Path
import sqlite3

import pytest
from app.ui import ncaaf_saved_refresh as viewer
from app_core import ncaaf_private_refresh as refresh
from test_ncaaf_private_normal_refresh import selection
from test_ncaaf_model_compatibility import synthetic

TOKEN = 'SYNTHETIC-owner-token-for-offline-test'


class Stub:
    def __init__(self, state): self.session_state = state
    def subheader(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def checkbox(self, *a, **k): return False


@pytest.mark.parametrize('token', ['', 'wrong', TOKEN])
def test_wrong_or_inactive_owner_has_zero_storage_access(monkeypatch, token):
    state = {'publication_token': token, 'unrelated': 7}
    if token != TOKEN: state[viewer.PREFIX+'loaded'] = ['private']
    monkeypatch.setattr(viewer, 'st', Stub(state))
    monkeypatch.setattr(viewer, 'catalog', lambda *a: pytest.fail('INACTIVE STORAGE ACCESS'))
    setting = lambda name, default='': TOKEN if name == 'PARLAYPICKER_PUBLISH_TOKEN' else default
    viewer.render(setting)
    assert state['unrelated'] == 7
    assert viewer.PREFIX+'loaded' not in state


def test_missing_root_and_unsafe_name_reject_without_database(monkeypatch, tmp_path):
    from app_core import ncaaf_pilot_setup
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **k: pytest.fail('DATABASE ACCESS'))
    with pytest.raises(ValueError): viewer.catalog(tmp_path/'absent')
    monkeypatch.setattr(ncaaf_pilot_setup, 'inspect_security', lambda p: {'status':'VERIFIED'})
    (tmp_path/'not-a-native.private-refresh.json').write_bytes(b'{}')
    with pytest.raises(ValueError): viewer.catalog(tmp_path)


def _snapshot(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


@pytest.fixture(scope='module')
def saved_native(tmp_path_factory):
    # One owned SYNTHETIC native pair, reused only by these read-only checks.
    # Its patch context and private directory belong to this fixture and are
    # restored on failure/teardown. Assertions and all six identities remain.
    root = tmp_path_factory.mktemp('SYNTHETIC-ncaaf-saved-native')
    with pytest.MonkeyPatch.context() as patch:
        from app_core import ncaaf_pilot_setup
        patch.setattr(ncaaf_pilot_setup, 'inspect_security',
                      lambda p: {'status':'VERIFIED', 'evidence':'SYNTHETIC_TEST_ONLY'})
        source = synthetic.__wrapped__(patch)
        value = selection(source, patch, root, -3.5)
        _, exports, ranked = refresh.run(value)
        yield root, exports, ranked


def _readonly(monkeypatch):
    original = sqlite3.connect
    def connect(target, *a, **kw):
        assert str(target).endswith('?mode=ro') and kw.get('uri') is True
        db = original(target, *a, **kw)
        def authorizer(action, *rest):
            if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                          sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE,
                          sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_ALTER_TABLE}:
                pytest.fail('SAVED VIEW DATABASE WRITE')
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorizer)
        return db
    monkeypatch.setattr(sqlite3, 'connect', connect)
    from app_core import market_probability_model, ncaaf_research
    monkeypatch.setattr(market_probability_model, 'predict_market_probabilities',
                        lambda *a, **k: pytest.fail('SAVED VIEW PREDICTOR CALL'))
    monkeypatch.setattr(ncaaf_research, 'centers',
                        lambda *a, **k: pytest.fail('SAVED VIEW NUMERICAL CALL'))


def test_native_saved_read_and_changed_body_rejection(saved_native, monkeypatch):
    tmp_path, exports, ranked = saved_native
    name = Path(exports[0]['private_result']['path']).name
    before = _snapshot(tmp_path)
    _readonly(monkeypatch)
    assert viewer.read(tmp_path, name) == ranked
    assert viewer.read(tmp_path, name) == ranked
    assert _snapshot(tmp_path) == before
    path = tmp_path/name
    original = path.read_bytes()
    try:
        path.write_bytes(original.replace(b'UNVALIDATED', b'TAMPERED'))
        with pytest.raises(ValueError): viewer.read(tmp_path, name)
    finally:
        path.write_bytes(original)
    assert _snapshot(tmp_path) == before


def test_real_apptest_controls_fresh_sessions_and_authorization_loss(saved_native, monkeypatch, tmp_path):
    from tests.apptest_owned_script import app_from_owned_script
    _, exports, ranked = saved_native
    root = Path(exports[0]['private_result']['path']).parent
    name = Path(exports[0]['private_result']['path']).name
    before = _snapshot(root)
    _readonly(monkeypatch)
    code = ("import streamlit as st\nfrom app.ui.ncaaf_saved_refresh import render\n"
            "st.text_input('Owner token',type='password',key='publication_token')\n"
            f"settings={{'PARLAYPICKER_PUBLISH_TOKEN':{TOKEN!r},'PARLAYPICKER_NCAAF_CUSTODY_ROOT':{str(root)!r}}}\n"
            "render(lambda name,default='':settings.get(name,default))\n")
    for number in range(2):
        app = app_from_owned_script(tmp_path, code, default_timeout=180)
        app.run()
        assert not app.checkbox and not app.exception
        app.text_input[0].input('wrong').run()
        assert not app.checkbox and not app.exception
        app.text_input[0].input(TOKEN).run()
        assert len(app.checkbox) == 1 and not app.multiselect
        app.checkbox[0].check().run()
        app.multiselect[0].set_value([name]).run()
        next(b for b in app.button if b.label=='Load saved NCAAF results').click().run()
        assert not app.exception and not app.error
        assert len(app.dataframe) == 1
        assert app.dataframe[0].value['result_sha256'].tolist() == [x['result_sha256'] for x in ranked]
        assert app.dataframe[0].value['inference_time'].tolist() == [x['inference_time'] for x in ranked]
        app.run()
        assert len(app.dataframe) == 1 and not app.exception
        app.text_input[0].input('wrong').run()
        assert not app.dataframe and not app.checkbox
        assert not any(k.startswith(viewer.PREFIX) for k in app.session_state.filtered_state)
    assert _snapshot(root) == before
