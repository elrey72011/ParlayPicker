"""Linux/Windows synthetic renderer controls; authentic browser proof is separate."""
from io import BytesIO
import sqlite3

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app_core import nfl_scenario_zip as replay, nfl_saved_scenarios as saved
from test_nfl_scenario_zip import fixtures
from test_nfl_saved_scenarios import readonly

TOKEN = 'SYNTHETIC-private-zip-viewer-token'


def synthetic_app(policy):
    import hmac
    import os
    import streamlit as st
    from app.ui.nfl_saved_viewer_state import revoke
    supplied = st.text_input('Publishing token', type='password', key='publication_token')
    if not hmac.compare_digest(supplied, os.environ['PARLAYPICKER_PUBLISH_TOKEN']):
        revoke(st.session_state)
        return
    from app.ui.nfl_saved_scenarios import render
    render(synthetic_policy=policy)


def production_panel():
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(None, None, lazy_history=True)


class Uploaded(BytesIO):
    def __init__(self, raw, name):
        super().__init__(raw)
        self.name, self.size, self.file_id = name, len(raw), name


def no_database(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('ZIP viewing must never resolve or access a database')
    monkeypatch.setattr(sqlite3, 'connect', denied)
    for name in ('store_path', 'catalog', 'load', 'private_download'):
        monkeypatch.setattr(saved, name, denied)


def controls(at):
    at.text_input(key='publication_token').set_value(TOKEN).run()
    at.radio(key='nfl_saved_source').set_value('Upload saved scenario ZIPs').run()
    at.checkbox(key='nfl_saved_show').check().run()
    assert not at.exception
    return at


def test_zip_controls_both_switch_clear_reupload_and_session_isolation(monkeypatch, tmp_path):
    values, policy = fixtures(); originals = [x[0] for x in values]
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path / 'absent-hosted-style-store'))
    provided = [Uploaded(x[0], str(i) + '.private-scenario.zip') for i, x in enumerate(values)]
    current = {'files':provided}
    original_upload = st.file_uploader
    # AppTest has no file-uploader input driver. Only the file-input boundary is
    # supplied synthetically; no parser, trust check, model or verdict is patched.
    # Real authentic file-picker/browser coverage is performed privately.
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: current['files'] if k.get('key','').startswith('nfl_saved_zip_upload_') else original_upload(*a, **k))
    no_database(monkeypatch)
    with readonly(monkeypatch):
        at = controls(AppTest.from_function(synthetic_app, args=(policy,), default_timeout=30).run())
        assert not at.metric
        assert at.button(key='nfl_saved_zip_load').disabled
        at.multiselect[0].set_value(['0', '1']).run()
        at.button(key='nfl_saved_zip_load').click().run()
        assert not at.exception and [m.value for m in at.metric] == ['42.0000%', '58.0000%']
        assert len([m for m in at.markdown if 'Original inference time: 2026-10-06T19:25:05Z' in m.value]) == 2
        assert any('not an authentic model execution' in w.value for w in at.warning)
        at.multiselect[0].set_value(['1']).run()
        assert not at.metric
        at.button(key='nfl_saved_zip_load').click().run()
        assert [m.value for m in at.metric] == ['58.0000%']
        fresh = controls(AppTest.from_function(synthetic_app, args=(policy,), default_timeout=30).run())
        assert not fresh.metric and fresh.session_state.filtered_state.get('nfl_saved_zip_loaded') is None
        assert [m.value for m in at.metric] == ['58.0000%']
        current['files'] = []
        previous_selector = at.multiselect[0].key
        at.button(key='nfl_saved_zip_clear').click().run()
        assert not at.exception and not at.metric
        assert at.multiselect[0].key != previous_selector
        assert at.multiselect[0].value == []
        assert at.session_state.filtered_state.get('nfl_saved_zip_loaded') is None
        current['files'] = provided
        at.run()
        assert at.button(key='nfl_saved_zip_load').disabled
        at.multiselect[0].set_value(['0', '1']).run()
        at.button(key='nfl_saved_zip_load').click().run()
        assert [m.value for m in at.metric] == ['42.0000%', '58.0000%']
        at.text_input(key='publication_token').set_value('wrong').run()
        assert not at.exception and not at.metric and not at.radio and not at.multiselect
        assert not any(str(k).startswith('nfl_saved_') for k in at.session_state.filtered_state)
        at.text_input(key='publication_token').set_value(TOKEN).run()
        assert not at.checkbox(key='nfl_saved_show').value and not at.metric
    assert not (tmp_path / 'absent-hosted-style-store').exists()
    assert [x.getvalue() for x in provided] == originals


def test_wrong_token_and_inactive_actual_owner_panel_have_zero_archive_or_database_access(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
    no_database(monkeypatch)
    def denied(*a, **k):
        pytest.fail('Wrong-token/inactive viewer attempted archive processing')
    monkeypatch.setattr(replay, 'read_archive', denied)
    monkeypatch.setattr(replay, 'read_archives', denied)
    with readonly(monkeypatch):
        at = AppTest.from_function(production_panel, default_timeout=30).run()
        at.text_input(key='publication_token').set_value('wrong').run()
        assert not at.exception and not at.radio
        at.text_input(key='publication_token').set_value(TOKEN).run()
        assert not at.exception and not at.checkbox(key='nfl_saved_show').value
        at.radio(key='nfl_saved_source').set_value('Upload saved scenario ZIPs').run()
        assert not at.exception and not at.metric


def test_production_authorization_loss_revokes_all_private_viewer_state(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
    no_database(monkeypatch)
    with readonly(monkeypatch):
        at = AppTest.from_function(production_panel, default_timeout=30).run()
        at.text_input(key='publication_token').set_value(TOKEN).run()
        at.session_state['nfl_saved_zip_loaded'] = {'SYNTHETIC_PRIVATE_BODY_CANARY':True}
        at.session_state['nfl_saved_zip_upload_99'] = b'SYNTHETIC_PRIVATE_BODY_CANARY'
        at.text_input(key='publication_token').set_value('wrong').run()
        assert not at.exception and not at.metric
        assert not any(str(k).startswith('nfl_saved_') for k in at.session_state.filtered_state)
