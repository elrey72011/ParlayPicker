"""Synthetic button-to-receipt qualification; all external work fails closed."""
import copy
import json
import socket
import subprocess
import sys
import threading

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import nfl_storage_diagnostic as ui
from app.ui.nfl_workflow_state import revoke
from app_core import nfl_diagnostic_receipts as receipts

TOKEN = 'SYNTHETIC-owner-token-valid'
SECRET = 'SYNTHETIC-sensitive-value-do-not-emit'
HOST = 'synthetic.invalid'
APP = '''import streamlit as st
from app.ui.nfl_private_workflow import render
from app.ui.nfl_workflow_state import revoke
TOKEN='SYNTHETIC-owner-token-valid'
token=st.text_input('Owner',type='password',key='publication_token')
def setting(name,default=''):
    return TOKEN if name=='PARLAYPICKER_PUBLISH_TOKEN' else default
if token==TOKEN: render(setting)
else: revoke(st.session_state)
'''


def setting(name, default=''):
    return TOKEN if name == 'PARLAYPICKER_PUBLISH_TOKEN' else default


def state():
    return {'publication_token': TOKEN, 'nfl_workflow_diag_expected_host': HOST}


@pytest.fixture(autouse=True)
def blocked(monkeypatch):
    def fail(*a, **k):
        pytest.fail('Database, provider, prediction or mutation forbidden')
    monkeypatch.setattr(socket, 'create_connection', fail)
    monkeypatch.setattr(socket.socket, 'connect', fail)
    for name in ('psycopg.connect', 'requests.sessions.Session.request',
                 'app_core.nfl_workflow_store.Store.postgres', 'app_core.nfl_workflow_store.Store.verify',
                 'app_core.nfl_workflow_store.Store.put', 'app_core.nfl_workflow_store.Store.claim',
                 'app_core.nfl_workflow_acquisition.capture_scores',
                 'app_core.nfl_workflow_acquisition.capture_market', 'app_core.nfl_scenario_research.compute'):
        monkeypatch.setattr(name, fail)


@pytest.mark.parametrize('native,outcome', [
    ({'status': 'PASS', 'attempted_connections': 1, 'driver_attempts': 1}, 'COMPLETED'),
    ({'status': 'STOPPED', 'attempted_connections': 0,
      'failure': {'phase': 'SETTINGS', 'exception_class': None, 'sqlstate': None,
                  'category': 'SETTINGS_CHECK_FAILED'}}, 'FAILED')])
def test_start_receipt_and_external_evidence_precede_work(native, outcome, monkeypatch, capsys):
    s = state()
    def fake(*a):
        assert s[receipts.GUARD] and s['nfl_workflow_diag_spent']
        assert s['nfl_workflow_diag_result']['stage'] == 'DIAGNOSTIC_STARTED'
        start = capsys.readouterr().err
        assert 'HANDLER_ENTERED' in start and 'DIAGNOSTIC_STARTED' in start
        assert SECRET not in start and HOST not in start
        return native
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', fake)
    ui._invoke(setting, s)
    result = s['nfl_workflow_diag_result']
    assert result['outcome'] == outcome and result['stage'] == 'TERMINAL'
    assert result['started_at'] == result['invoked_at'] <= result['ended_at']
    assert len(result['invocation_id']) == 36
    assert result['attempted_connections'] == native['attempted_connections']
    assert result['invocation_id'] in capsys.readouterr().err


@pytest.mark.parametrize('error', [RuntimeError(SECRET + HOST), ValueError(SECRET + HOST)])
def test_unexpected_exception_never_invents_zero_or_emits_raw(error, monkeypatch, capsys):
    def fake(*a): raise error
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', fake)
    s = state(); ui._invoke(setting, s)
    result = s['nfl_workflow_diag_result']
    assert result['outcome'] == 'FAILED'
    assert result['attempted_connections'] == result['driver_attempts'] == 'UNKNOWN'
    assert result['failure']['category'] == 'DIAGNOSTIC_UNAVAILABLE'
    assert SECRET not in str(result) + capsys.readouterr().err


def test_interruption_retains_spent_unknown_and_never_resumes(monkeypatch, capsys):
    class Interrupt(BaseException): pass
    calls = []
    def fake(*a): calls.append(1); raise Interrupt(SECRET)
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', fake)
    s = state()
    with pytest.raises(Interrupt): ui._invoke(setting, s)
    result = copy.deepcopy(s['nfl_workflow_diag_result'])
    assert result['outcome'] == 'INTERRUPTED_UNKNOWN'
    assert result['attempted_connections'] == 'UNKNOWN'
    ui._invoke(setting, s)
    assert calls == [1] and s['nfl_workflow_diag_result'] == result
    assert SECRET not in capsys.readouterr().err


def test_duplicate_event_guard_is_independent_of_widget(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def fake(*a):
        calls.append(1); entered.set()
        assert release.wait(5)
        return {'status': 'PASS', 'attempted_connections': 1}
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', fake)
    s = state(); thread = threading.Thread(target=ui._invoke, args=(setting, s))
    thread.start()
    try:
        assert entered.wait(5)
        ui._invoke(setting, s)
    finally:
        release.set(); thread.join(5)
    assert not thread.is_alive() and calls == [1]


def test_auth_loss_clears_private_display_but_preserves_spent_guard(monkeypatch):
    calls = []
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose',
                        lambda *a: calls.append(1) or {'status': 'PASS', 'attempted_connections': 1})
    s = state(); s['unrelated'] = 3
    ui._invoke(setting, s); tombstone = copy.deepcopy(s[receipts.GUARD])
    revoke(s)
    assert s[receipts.GUARD] == tombstone and s['unrelated'] == 3
    assert 'nfl_workflow_diag_result' not in s and 'nfl_workflow_diag_expected_host' not in s
    s['publication_token'] = 'wrong'; ui._invoke(setting, s)
    s['publication_token'] = TOKEN; ui._invoke(setting, s)
    assert calls == [1] and 'nfl_workflow_diag_result' not in s


def test_wrong_owner_never_claims_or_imports_work(monkeypatch, capsys):
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', lambda *a: pytest.fail('Unauthorized'))
    s = state(); s['publication_token'] = 'wrong'; ui._invoke(setting, s)
    assert receipts.GUARD not in s and not capsys.readouterr().err


def test_projection_rejects_untrusted_fields_and_secret_exceptions():
    raw = {'status': 'PASS', 'attempted_connections': True, 'driver_attempts': SECRET,
           'raw': SECRET, 'checks': {SECRET: True, 'verify_full': SECRET, 'system_roots': True},
           'runtime': {'python': SECRET, 'psycopg': '3.2.10', 'checkout_head': SECRET, 'libpq': 170005},
           'failure': {'phase': SECRET, 'exception_class': SECRET, 'sqlstate': SECRET, 'category': SECRET}}
    safe = receipts.sanitize(raw)
    assert SECRET not in json.dumps(safe) and safe['attempted_connections'] == 'UNKNOWN'
    assert safe['checks'] == {'system_roots': True}
    assert receipts.sanitize(None)['attempted_connections'] == 'UNKNOWN'


def test_actual_widgets_terminal_receipt_disabled_immediately_reopen_and_auth_loss(tmp_path, monkeypatch):
    app = tmp_path / 'synthetic.py'; app.write_text(APP)
    calls = []
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose',
                        lambda *a: calls.append(1) or {'status': 'PASS', 'attempted_connections': 1})
    at = AppTest.from_file(app, default_timeout=20).run()
    at.text_input(key='publication_token').input('wrong').run()
    assert not at.checkbox and not calls
    at.text_input(key='publication_token').input(TOKEN).run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert at.button(key='nfl_workflow_diag_run').disabled
    at.text_input(key='nfl_workflow_diag_expected_host').input(HOST).run()
    at.button(key='nfl_workflow_diag_run').click().run()
    assert not at.exception and len(calls) == 1
    assert at.button(key='nfl_workflow_diag_run').disabled
    result = copy.deepcopy(at.session_state['nfl_workflow_diag_result'])
    at.run()
    at.checkbox(key='nfl_workflow_diag_enabled').uncheck().run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert at.session_state['nfl_workflow_diag_result'] == result and len(calls) == 1
    at.text_input(key='publication_token').input('wrong').run()
    assert not at.json and 'nfl_workflow_diag_result' not in at.session_state
    at.text_input(key='publication_token').input(TOKEN).run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    at.text_input(key='nfl_workflow_diag_expected_host').input(HOST).run()
    assert at.button(key='nfl_workflow_diag_run').disabled and len(calls) == 1
    fresh = AppTest.from_file(app, default_timeout=20).run()
    assert not fresh.checkbox and not fresh.json and len(calls) == 1


def test_log_failure_stops_before_diagnostic_and_keeps_tombstone(monkeypatch):
    monkeypatch.setattr(receipts, 'emit', lambda *a: (_ for _ in ()).throw(OSError(SECRET)))
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', lambda *a: pytest.fail('Must not run'))
    s = state()
    ui._invoke(setting, s)
    assert s[receipts.GUARD]
    assert s['nfl_workflow_diag_result']['outcome'] == 'FAILED'
    assert s['nfl_workflow_diag_result']['attempted_connections'] == 'UNKNOWN'
    assert s['nfl_workflow_diag_result']['server_log_recorded'] is False


def test_abrupt_owned_subprocess_retains_start_without_claiming_completion():
    code = '''import os,socket
from app.ui.nfl_storage_diagnostic import _invoke
import app_core.nfl_storage_diagnostic as diagnostic
def blocked(*a,**k): raise AssertionError('NETWORK_FORBIDDEN')
socket.create_connection=blocked; socket.socket.connect=blocked
def abrupt(*a,**k): os._exit(7)
diagnostic.diagnose=abrupt
token='SYNTHETIC-owner-token-valid'
_invoke(lambda name,default='': token if name=='PARLAYPICKER_PUBLISH_TOKEN' else default,
        {'publication_token':token,'nfl_workflow_diag_expected_host':'synthetic.invalid'})
'''
    child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, timeout=20)
    assert child.returncode == 7
    rows = [json.loads(line.split(' ', 1)[1]) for line in child.stderr.decode().splitlines()
            if line.startswith('NFL_STORAGE_DIAGNOSTIC ')]
    assert [r['stage'] for r in rows] == ['HANDLER_ENTERED', 'DIAGNOSTIC_STARTED']
    assert len({r['invocation_id'] for r in rows}) == 1
    assert all(r['attempted_connections'] == 'UNKNOWN' and r['outcome'] == 'STARTED' for r in rows)
    assert HOST not in child.stderr.decode()


def test_orphan_started_receipt_is_not_resumed_by_another_event(monkeypatch):
    s = state(); started = receipts.claim(s)
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', lambda *a: pytest.fail('No resume'))
    ui._invoke(setting, s)
    assert s['nfl_workflow_diag_result'] == started
    assert started['outcome'] == 'STARTED' and started['attempted_connections'] == 'UNKNOWN'


def test_terminal_log_precedes_interruptible_session_write(monkeypatch, capsys):
    class InterruptedWrite(BaseException): pass
    class State(dict):
        def __setitem__(self, key, value):
            if key == 'nfl_workflow_diag_result' and value.get('outcome') == 'INTERRUPTED_UNKNOWN':
                raise InterruptedWrite()
            super().__setitem__(key, value)
    s = State(state())
    def interrupted(*a): raise InterruptedWrite()
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', interrupted)
    with pytest.raises(InterruptedWrite): ui._invoke(setting, s)
    logs = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().err.splitlines()]
    assert logs[-1]['outcome'] == 'INTERRUPTED_UNKNOWN'
    assert logs[-1]['attempted_connections'] == 'UNKNOWN'
    assert s['nfl_workflow_diag_result']['outcome'] == 'STARTED' and s[receipts.GUARD]
