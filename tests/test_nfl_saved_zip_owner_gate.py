"""Owner-gate compatibility and real-session revocation; synthetic inputs only."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import publish_panel, canonical_evidence_download
from app_core import nfl_scenario_zip as replay
from test_nfl_saved_scenarios import readonly, records
from test_nfl_scenario_zip_ui import no_database, production_panel, TOKEN


@pytest.mark.parametrize('state_kind', ['absent', 'empty', 'populated'])
@pytest.mark.parametrize('configured,supplied,authorized', [
    ('', TOKEN, False), ('short', TOKEN, False),
    (TOKEN, 'wrong-token', False), (TOKEN, TOKEN, True),
])
def test_owner_gate_state_contract(monkeypatch, state_kind, configured, supplied, authorized):
    state = {} if state_kind != 'populated' else {
        'nfl_saved_zip_upload_2': b'SYNTHETIC_PRIVATE_UPLOAD',
        'nfl_saved_zip_selected_2': ['SYNTHETIC'],
        'nfl_saved_zip_loaded': {'SYNTHETIC_RESULT': True},
        'nfl_saved_loaded': ['SYNTHETIC_RECEIPT'],
        'nfl_saved_source': 'Upload saved scenario ZIPs',
        'unrelated_feature': {'preserve': True},
        'publication_token': supplied,
    }
    before = dict(state); text_calls = []; authorized_calls = []
    def text_input(*args, **kwargs):
        text_calls.append(True)
        return supplied
    mock = SimpleNamespace(subheader=lambda *a, **k: None, caption=lambda *a, **k: None,
        info=lambda *a, **k: None, text_input=text_input)
    if state_kind != 'absent':
        mock.session_state = state
    monkeypatch.setattr(publish_panel, 'st', mock)
    monkeypatch.setattr(publish_panel, 'setting', lambda *a, **k: configured)
    class AuthorizedBoundary(Exception):
        pass
    def reached(*args, **kwargs):
        authorized_calls.append(True)
        raise AuthorizedBoundary
    monkeypatch.setattr(canonical_evidence_download, 'render', reached)
    no_database(monkeypatch)
    def forbidden(*a, **k):
        pytest.fail('Owner authorization attempted archive processing')
    monkeypatch.setattr(replay, 'read_archive', forbidden)
    monkeypatch.setattr(replay, 'read_archives', forbidden)
    with readonly(monkeypatch):
        if authorized:
            with pytest.raises(AuthorizedBoundary):
                publish_panel.render_publish_panel(None, None)
            assert state == before and authorized_calls == [True]
        else:
            publish_panel.render_publish_panel(None, None)
            assert not authorized_calls
            assert state == {k: v for k, v in before.items() if not k.startswith('nfl_saved_')}
    assert len(text_calls) == (0 if len(configured) < 16 else 1)
    assert hasattr(mock, 'session_state') == (state_kind != 'absent')


@pytest.mark.parametrize('configured,supplied', [('short', TOKEN), (TOKEN, 'wrong-token')])
def test_real_cleanup_errors_are_not_suppressed(monkeypatch, configured, supplied):
    class BrokenState(dict):
        def __delitem__(self, key):
            raise RuntimeError('SYNTHETIC genuine cleanup failure')
    mock = SimpleNamespace(session_state=BrokenState(nfl_saved_zip_loaded=True),
        subheader=lambda *a, **k: None, caption=lambda *a, **k: None,
        info=lambda *a, **k: None, text_input=lambda *a, **k: supplied)
    monkeypatch.setattr(publish_panel, 'st', mock)
    monkeypatch.setattr(publish_panel, 'setting', lambda *a, **k: configured)
    no_database(monkeypatch)
    with readonly(monkeypatch), pytest.raises(RuntimeError, match='genuine cleanup failure'):
        publish_panel.render_publish_panel(None, None)


@pytest.mark.parametrize('loss', ['wrong-token', 'missing-token', 'short-token'])
def test_actual_panel_revokes_after_loading_without_touching_unrelated_state(monkeypatch, tmp_path, loss):
    _, saved = records(monkeypatch, tmp_path)
    originals = [value['sha256'] for receipt, value in saved]
    clocks = [value['payload']['inference_time'] for receipt, value in saved]
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
    with readonly(monkeypatch):
        at = AppTest.from_function(production_panel, default_timeout=30).run()
        at.text_input(key='publication_token').set_value(TOKEN).run()
        at.checkbox(key='nfl_saved_show').check().run()
        ids = [receipt['export_id'] for receipt, value in saved]
        at.multiselect(key='nfl_saved_selected').set_value(ids).run()
        at.button(key='nfl_saved_load').click().run()
        assert not at.exception and len(at.metric) == 2
        at.session_state['unrelated_owner_feature'] = 'SYNTHETIC_PRESERVE_ME'
        if loss == 'wrong-token':
            at.text_input(key='publication_token').set_value('wrong-token').run()
        else:
            monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', '' if loss == 'missing-token' else 'short')
            at.run()
        assert not at.exception and not at.metric and not at.radio and not at.multiselect
        assert not any(str(k).startswith('nfl_saved_') for k in at.session_state.filtered_state)
        assert at.session_state['unrelated_owner_feature'] == 'SYNTHETIC_PRESERVE_ME'
        monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
        at.run()
        at.text_input(key='publication_token').set_value(TOKEN).run()
        assert not at.checkbox(key='nfl_saved_show').value and not at.metric
    assert [value['sha256'] for receipt, value in saved] == originals
    assert [value['payload']['inference_time'] for receipt, value in saved] == clocks
