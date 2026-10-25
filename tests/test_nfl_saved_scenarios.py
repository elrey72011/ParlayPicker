"""Read-only viewer regressions; synthetic saved results, no numerical calls."""
from contextlib import closing, contextmanager
from copy import deepcopy
import hashlib
import json
import socket
import sqlite3
import sys
import threading

import pytest
from streamlit.testing.v1 import AppTest

from app_core import nfl_saved_scenarios as saved, nfl_scenario_research as scenario, research_replay
from test_nfl_scenario_research import fixture, INFERENCE, FINISHED


def app():
    from app.ui.nfl_private_research import render_scenario
    render_scenario()


def records(monkeypatch, tmp_path):
    configured = tmp_path / 'evidence.sqlite3'
    with closing(sqlite3.connect(configured)) as db, db:
        db.execute('CREATE TABLE snapshots(snapshot_id TEXT PRIMARY KEY)')
        research_replay.setup(db)
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path))
    values = []
    for selected, point, probability in [('Dallas', -1.5, .42), ('Buffalo', 1.5, .58)]:
        packet = fixture(monkeypatch, selected=selected, line=point)
        result = scenario.seal(dict(version=scenario.RESULT, status='COMPLETE', labels=scenario.LABELS,
            input=packet, inference_time=INFERENCE, completed_at=FINISHED,
            reader=scenario.reader_binding(), model=packet['payload']['model'],
            native_output=dict(ml_probability=probability, ml_probability_source='score-distribution-v1:nfl',
                ml_target='spread_cover', ml_projection=-2.0 if selected == 'Dallas' else 2.0, ml_residual_scale=13.86),
            probability=probability, probability_stage='native_model_output', calibrated=False,
            probability_semantics='full_game_including_overtime_cover_conditional_on_completed_sporting_result',
            publisher_availability='UNKNOWN', source_acceptance=False, wagering_authority=False,
            qualification=False, wager_action='PASS', stake=0))
        values.append((research_replay.retain_scenario(result, path=configured), result))
    return configured, values


@contextmanager
def readonly(monkeypatch):
    """Fail on all predictor calls/network/DB writes without altering bindings."""
    from app_core import market_probability_model as model
    forbidden = {getattr(model, n).__code__ for n in ('predict_market_probabilities', '_normal_cdf')}
    forbidden.update(f.__code__ for f in (scenario.compute, scenario.run, research_replay.retain_scenario))
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in forbidden:
            raise AssertionError('Viewer attempted inference or storage')
    old_profile, old_thread = sys.getprofile(), threading.getprofile()
    def deny(*a, **k):
        raise AssertionError('Viewer attempted network access')
    original = sqlite3.connect
    def connect(database, *args, **kwargs):
        assert kwargs.get('uri') is True and '?mode=ro' in str(database), 'Viewer attempted writable DB connection'
        db = original(database, *args, **kwargs)
        def authorize(action, *args):
            assert action in {sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_PRAGMA,
                              sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}, 'Viewer attempted database write'
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize)
        return db
    with monkeypatch.context() as m:
        m.setattr(socket, 'create_connection', deny)
        m.setattr(socket, 'getaddrinfo', deny)
        m.setattr(socket.socket, 'connect', deny)
        m.setattr(sqlite3, 'connect', connect)
        sys.setprofile(profile); threading.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(old_profile); threading.setprofile(old_thread)


def test_real_controls_both_switch_and_fresh_session_are_static(monkeypatch, tmp_path):
    path, values = records(monkeypatch, tmp_path)
    before = path.read_bytes()
    with readonly(monkeypatch):
        at = AppTest.from_function(app, default_timeout=30).run()
        assert not at.exception
        assert at.button(key='nfl_scenario_run').disabled
        assert not at.checkbox(key='nfl_scenario_enabled').value
        assert not at.checkbox(key='nfl_saved_show').value
        at.checkbox(key='nfl_saved_show').check().run()
        keys = [v[0]['export_id'] for v in values]
        at.multiselect(key='nfl_saved_selected').set_value(keys).run()
        at.button(key='nfl_saved_load').click().run()
        assert not at.exception and not at.error
        assert [m.value for m in at.metric] == ['42.00%', '58.00%']
        assert all('saved native' in m.label for m in at.metric)
        assert any('Original inference time: ' + INFERENCE in m.value for m in at.markdown)
        assert any('selected-team final points minus opponent' in m.value for m in at.markdown)
        assert len([c for c in at.caption if 'PASS — zero stake' in c.value]) == 2
        at.multiselect(key='nfl_saved_selected').set_value([keys[1]]).run()
        assert not at.metric  # Selection change never shows old loaded data.
        at.button(key='nfl_saved_load').click().run()
        assert [m.value for m in at.metric] == ['58.00%']
        fresh = AppTest.from_function(app, default_timeout=30).run()
        assert not fresh.metric
        assert not fresh.checkbox(key='nfl_saved_show').value
        fresh.checkbox(key='nfl_saved_show').check().run()
        fresh.multiselect(key='nfl_saved_selected').set_value(keys).run()
        fresh.button(key='nfl_saved_load').click().run()
        assert not fresh.exception and [m.value for m in fresh.metric] == ['42.00%', '58.00%']
        for receipt, original in values:
            assert research_replay.read_scenario(receipt, path=path) == original
    assert path.read_bytes() == before


def test_postgame_viewing_uses_original_inference_clock(monkeypatch, tmp_path):
    path, values = records(monkeypatch, tmp_path)
    monkeypatch.setattr(scenario, 'now', lambda: '2099-01-01T00:00:00Z')
    with readonly(monkeypatch):
        index = saved.catalog()
        assert len(index['records']) == 2
        for receipt, original in values:
            assert saved.load(receipt)['inference_time'] == original['payload']['inference_time']
            assert saved.private_download(receipt)[0]


@pytest.mark.parametrize('attack', ['missing', 'corrupt', 'input_missing', 'oversize'])
def test_missing_corrupt_or_oversize_records_do_not_change_store(monkeypatch, tmp_path, attack):
    path, values = records(monkeypatch, tmp_path)
    receipt = values[0][0]
    # Construct labelled synthetic damage BEFORE enforcing viewer read-only mode.
    with sqlite3.connect(path) as db:
        for name in ('immutable_research_replay_exports_UPDATE', 'immutable_research_replay_exports_DELETE',
                     'immutable_research_source_intakes_DELETE'):
            db.execute('DROP TRIGGER ' + name)
        if attack == 'missing':
            db.execute('DELETE FROM research_replay_exports WHERE export_id=?', (receipt['export_id'],))
        elif attack == 'input_missing':
            db.execute('DELETE FROM research_source_intakes WHERE reference=?', ('nfl-scenario-input:' + receipt['input_sha256'],))
        else:
            db.execute('UPDATE research_replay_exports SET payload=? WHERE export_id=?',
                       ('{}' if attack == 'corrupt' else 'x' * (scenario.MAX_PACKET + 1), receipt['export_id']))
        research_replay.setup(db)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with readonly(monkeypatch):
        index = saved.catalog()
        assert len(index['records']) == 1
        assert len(index['rejected']) == (0 if attack == 'missing' else 1)
        with pytest.raises((ValueError, TypeError, KeyError)):
            saved.load(receipt)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_store_is_not_initialized(monkeypatch, tmp_path):
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path / 'absent'))
    with readonly(monkeypatch):
        at = AppTest.from_function(app, default_timeout=30).run()
        at.checkbox(key='nfl_saved_show').check().run()
        assert not at.exception and not at.metric
        assert any('NFL_SCENARIO_STORE_UNAVAILABLE' in x.value for x in at.info)
    assert not (tmp_path / 'absent').exists()


def test_unauthorized_owner_cannot_list_or_load(monkeypatch):
    def protected_app():
        from app.ui.publish_panel import render_publish_panel
        render_publish_panel(None, None)
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', 'SYNTHETIC-owner-token-12345')
    for name in ('store_path', 'catalog', 'load', 'private_download'):
        monkeypatch.setattr(saved, name, lambda *a, **k: pytest.fail('Unauthorized store read'))
    with readonly(monkeypatch):
        at = AppTest.from_function(protected_app, default_timeout=30).run()
        assert not at.exception and not at.metric
        at.text_input(key='publication_token').set_value('wrong').run()
        assert not at.exception and not at.multiselect


def test_changed_configured_store_clears_loaded_results(monkeypatch, tmp_path):
    path, values = records(monkeypatch, tmp_path)
    with readonly(monkeypatch):
        at = AppTest.from_function(app, default_timeout=30).run()
        at.checkbox(key='nfl_saved_show').check().run()
        at.multiselect(key='nfl_saved_selected').set_value([values[0][0]['export_id']]).run()
        at.button(key='nfl_saved_load').click().run()
        assert len(at.metric) == 1
        monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path / 'other-store'))
        at.run()
        assert not at.exception and not at.metric
        assert at.session_state['nfl_saved_loaded'] == []


def test_correct_token_and_remote_controls_leave_viewer_inactive(monkeypatch):
    from test_remote_canonical_continuation import test_owner_scope_changes_invalidate_zip_and_wrong_token_has_no_controls
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail('Inactive viewer must not access saved storage')
    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    for name in ('store_path', 'catalog', 'load', 'private_download'):
        monkeypatch.setattr(saved, name, forbidden)
    # Execute every original assertion and its original three-second deadline.
    # The new viewer itself is not mocked or removed from the owner panel.
    with readonly(monkeypatch):
        test_owner_scope_changes_invalidate_zip_and_wrong_token_has_no_controls(monkeypatch)
    assert calls == []


def test_deactivation_precedes_store_resolution_even_with_loaded_receipts(monkeypatch, tmp_path):
    _, values = records(monkeypatch, tmp_path)
    with readonly(monkeypatch):
        at = AppTest.from_function(app, default_timeout=30).run()
        at.checkbox(key='nfl_saved_show').check().run()
        at.multiselect(key='nfl_saved_selected').set_value([values[0][0]['export_id']]).run()
        at.button(key='nfl_saved_load').click().run()
        assert len(at.metric) == 1
        def forbidden(*args, **kwargs):
            pytest.fail('Deactivated viewer accessed storage')
        for name in ('store_path', 'catalog', 'load', 'private_download'):
            monkeypatch.setattr(saved, name, forbidden)
        at.checkbox(key='nfl_saved_show').uncheck().run()
        assert not at.exception and not at.metric
        assert at.session_state['nfl_saved_loaded'] == []
