"""Synthetic diagnostics: no providers, predictions or real database connections."""
import io
import logging
import os
import socket
import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.publish_panel import setting
from app_core.nfl_private_workflow import authenticate
from app_core.nfl_storage_diagnostic import diagnose, safe_error
from app_core.nfl_workflow_store import Store

TOKEN = 'SYNTHETIC-owner-token-valid'
SECRET = 'SYNTHETIC-secret-never-display'
HOST = 'synthetic.invalid'
ENV = {'PARLAYPICKER_PUBLISH_TOKEN': TOKEN, 'PARLAYPICKER_NFL_DATABASE_URL': 'synthetic-url',
       'PARLAYPICKER_NFL_STORE_KEY': SECRET * 2, 'PARLAYPICKER_NFL_ACQUISITION_ENABLED': 'false',
       'PARLAYPICKER_NFL_INFERENCE_ENABLED': 'false'}
INFO = {'host': HOST, 'user': 'nfl_app', 'dbname': 'neondb', 'port': '5432',
        'sslmode': 'verify-full', 'sslrootcert': 'system', 'password': SECRET}


class Cursor:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return self.rows
    def fetchone(self): return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, fail_schema=False):
        self.calls, self.fail_schema, self.closed = [], fail_schema, False
        self.pgconn = types.SimpleNamespace(ssl_in_use=True)

    def execute(self, sql, args=()):
        self.calls.append(sql)
        if sql == 'BEGIN READ ONLY': return Cursor([])
        if sql.startswith("SELECT current_user = 'nfl_app'"): return Cursor([(True,) * 12])
        if sql == 'SELECT version FROM nfl_private.meta':
            if self.fail_schema:
                class UndefinedTable(Exception): sqlstate = '42P01'
                raise UndefinedTable(SECRET + HOST)
            return Cursor([('nfl-private-workflow-store-v1',)])
        if sql.startswith('SELECT r.rolsuper'): return Cursor([(False,) * 6])
        if sql.startswith('SELECT count(*)'): return Cursor([(0,)])
        if sql.startswith('SELECT t.tgname'):
            return Cursor([(args[0] + '_immutable', "BEGIN RAISE EXCEPTION 'NFL private records are append-only'; END;")])
        raise AssertionError('database write or unexpected query forbidden')

    def commit(self): self.calls.append('COMMIT')
    def rollback(self): self.calls.append('ROLLBACK')
    def close(self): self.closed = True


@pytest.fixture
def synthetic(monkeypatch):
    for key, value in ENV.items(): monkeypatch.setenv(key, value)
    fake = types.ModuleType('psycopg')
    fake.__version__ = '3.2.10'
    class SyntheticConnection:
        @staticmethod
        def _connect_gen():
            yield None
    fake.Connection = SyntheticConnection
    fake.pq = types.SimpleNamespace(version=lambda: 170002)
    conninfo = types.ModuleType('psycopg.conninfo')
    conninfo.conninfo_to_dict = lambda _: dict(INFO)
    monkeypatch.setitem(sys.modules, 'psycopg', fake)
    monkeypatch.setitem(sys.modules, 'psycopg.conninfo', conninfo)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('network forbidden'))
    for path in ('app_core.nfl_workflow_acquisition.capture_scores', 'app_core.nfl_workflow_acquisition.capture_market',
                 'app_core.nfl_scenario_research.compute', 'app_core.nfl_scenario_research.run'):
        monkeypatch.setattr(path, lambda *a, **k: pytest.fail('provider or prediction forbidden'))
    calls = []
    def connect(*args, **kwargs):
        list(fake.Connection._connect_gen())
        calls.append((args, kwargs))
        return FakeConnection()
    fake.connect = connect
    return fake, conninfo, calls


def one_attempt(fake, action):
    def connect(*args, **kwargs):
        list(fake.Connection._connect_gen())
        return action(*args, **kwargs)
    return connect


def checked(result):
    text = str(result)
    assert SECRET not in text and HOST not in text and 'synthetic-url' not in text
    return result


def invoke():
    return diagnose(setting, authenticate(TOKEN, TOKEN), HOST)


def test_real_released_verifier_once_with_existing_deadlines(synthetic):
    _, _, calls = synthetic
    result = checked(invoke())
    assert result['status'] == 'PASS', result
    assert result['attempted_connections'] == len(calls) == result['driver_attempts'] == 1
    assert result['checks']['client_tls_active'] and result['checks']['role_database_readonly_and_permissions']
    assert calls[0][1] == {'autocommit': True, 'connect_timeout': 5,
        'options': '-c statement_timeout=5000 -c lock_timeout=5000'}


def test_original_connection_exception_is_sanitized_before_translation(synthetic, caplog):
    fake, _, _ = synthetic
    class OperationalError(Exception): sqlstate = None
    def fail(*_a, **_k):
        logging.getLogger('psycopg').warning(SECRET + HOST)
        logging.getLogger('psycopg.pq').warning(SECRET + HOST)
        logging.getLogger('psycopg.debug.newchild').warning(SECRET + HOST)
        raise OperationalError(SECRET + HOST)
    fake.connect = one_attempt(fake, fail)
    with caplog.at_level(logging.DEBUG, logger='psycopg'):
        result = checked(invoke())
    assert result['failure'] == {'phase': 'CONNECT', 'exception_class': 'OperationalError',
                                  'sqlstate': None, 'category': 'CONNECTION_FAILURE_UNCLASSIFIED'}
    assert result['attempted_connections'] == 1 and SECRET not in caplog.text and HOST not in caplog.text


def test_original_schema_exception_is_sanitized_before_translation(synthetic):
    fake, _, _ = synthetic
    con = FakeConnection(fail_schema=True)
    fake.connect = one_attempt(fake, lambda *_a, **_k: con)
    result = checked(invoke())
    assert result['failure'] == {'phase': 'SCHEMA', 'exception_class': 'UndefinedTable',
                                  'sqlstate': '42P01', 'category': 'RELATION_MISSING'}
    assert con.closed and 'ROLLBACK' in con.calls


def test_disabled_flags_and_changed_settings_stop_before_connection(synthetic, monkeypatch):
    _, _, calls = synthetic
    monkeypatch.setenv('PARLAYPICKER_NFL_INFERENCE_ENABLED', 'true')
    result = checked(invoke())
    assert result['failure']['category'] == 'SETTINGS_CHECK_FAILED' and not calls
    monkeypatch.setenv('PARLAYPICKER_NFL_INFERENCE_ENABLED', 'false')
    wrong_host = checked(diagnose(setting, authenticate(TOKEN, TOKEN), 'other.invalid'))
    assert not wrong_host['checks']['production_host_matches'] and not calls
    _, conninfo, _ = synthetic
    conninfo.conninfo_to_dict = lambda _: {**INFO, 'user': 'different', 'sslrootcert': 'other'}
    wrong_identity = checked(invoke())
    assert not wrong_identity['checks']['restricted_user_matches']
    assert not wrong_identity['checks']['system_roots'] and not calls
    conninfo.conninfo_to_dict = lambda _: {**INFO, 'password': ''}
    assert not checked(invoke())['checks']['password_present'] and not calls
    conninfo.conninfo_to_dict = lambda _: dict(INFO)
    seen = []
    def changed(name, default=''):
        if name == 'PARLAYPICKER_NFL_DATABASE_URL':
            seen.append(1)
            if len(seen) > 1: return SECRET
        return setting(name, default)
    result = checked(diagnose(changed, authenticate(TOKEN, TOKEN), HOST))
    assert result['failure']['phase'] == 'PRODUCTION_SERVICE' and not calls


def test_handled_setting_exception_does_not_mask_connection_failure(synthetic):
    fake, _, _ = synthetic
    def handled(name, default=''):
        try: raise ValueError(SECRET + HOST)
        except ValueError: pass
        return setting(name, default)
    class OperationalError(Exception): sqlstate = None
    def fail(*_a, **_k): raise OperationalError(SECRET + HOST)
    fake.connect = one_attempt(fake, fail)
    assert checked(diagnose(handled, authenticate(TOKEN, TOKEN), HOST))['failure']['phase'] == 'CONNECT'


def test_hostile_sqlstate_property_cannot_escape():
    class Dangerous(Exception):
        @property
        def sqlstate(self): raise ValueError(SECRET + HOST)
    assert checked(safe_error('CONNECT', Dangerous(SECRET + HOST)))['sqlstate'] is None


def test_existing_trace_stops_and_remains_unchanged(synthetic):
    def oldtrace(frame, event, arg): return oldtrace
    sys.settrace(oldtrace)
    try:
        result = checked(invoke())
        assert result['failure']['category'] == 'EXISTING_TRACE_UNSUPPORTED'
        assert sys.gettrace() is oldtrace
    finally:
        sys.settrace(None)


def test_owner_ui_is_default_off_and_reruns_do_not_repeat(tmp_path, monkeypatch):
    app = tmp_path / 'SYNTHETIC-diagnostic-ui.py'
    app.write_text('''\nimport streamlit as st\nfrom app.ui.nfl_private_workflow import render\nfrom app.ui.nfl_workflow_state import revoke\nTOKEN="SYNTHETIC-owner-token-valid"\ntoken=st.text_input("Owner",type="password",key="publication_token")\ndef setting(name,default=""):\n    return TOKEN if name=="PARLAYPICKER_PUBLISH_TOKEN" else default\nif token != TOKEN:\n    revoke(st.session_state)\nelse:\n    render(setting)\n''')
    calls = []
    def fake(*_a, **_k):
        calls.append(1)
        return {'status': 'STOPPED', 'attempted_connections': 1}
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', fake)
    monkeypatch.setattr(Store, 'verify', lambda *_a, **_k: pytest.fail('normal store forbidden'))
    monkeypatch.setattr(Store, 'put', lambda *_a, **_k: pytest.fail('database write forbidden'))
    monkeypatch.setattr(Store, 'claim', lambda *_a, **_k: pytest.fail('database write forbidden'))
    for path in ('requests.sessions.Session.request', 'app_core.nfl_workflow_acquisition.capture_scores',
                 'app_core.nfl_workflow_acquisition.capture_market', 'app_core.nfl_scenario_research.compute'):
        monkeypatch.setattr(path, lambda *_a, **_k: pytest.fail('provider or prediction forbidden'))
    at = AppTest.from_file(app, default_timeout=20).run()
    assert not at.checkbox and not calls
    at.text_input(key='publication_token').input('wrong').run()
    assert not at.checkbox and not calls
    at.text_input(key='publication_token').input(TOKEN).run()
    assert not at.exception and not calls
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert not at.exception and not calls
    assert at.button(key='nfl_workflow_diag_run').disabled
    at.text_input(key='nfl_workflow_diag_expected_host').input(HOST).run()
    at.button(key='nfl_workflow_diag_run').click().run()
    assert not at.exception and len(calls) == 1
    assert 'invoked_at' in at.session_state['nfl_workflow_diag_result']
    at.run(); assert len(calls) == 1 and at.button(key='nfl_workflow_diag_run').disabled
    at.checkbox(key='nfl_workflow_diag_enabled').uncheck().run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert len(calls) == 1 and at.button(key='nfl_workflow_diag_run').disabled
    at.text_input(key='publication_token').input('wrong').run()
    assert 'nfl_workflow_diag_result' not in at.session_state


def test_failed_transaction_and_driver_import_are_sanitized(synthetic, monkeypatch):
    fake, _, calls = synthetic
    class OperationalError(Exception): sqlstate = '57014'
    class FailedBegin(FakeConnection):
        def execute(self, sql, args=()):
            if sql == 'BEGIN READ ONLY': raise OperationalError(SECRET + HOST)
            return super().execute(sql, args)
    fake.connect = one_attempt(fake, lambda *_a, **_k: FailedBegin())
    result = checked(invoke())
    assert result['failure'] == {'phase':'TRANSACTION', 'exception_class':'OperationalError',
                                 'sqlstate':'57014', 'category':'QUERY_CANCELLED'}
    monkeypatch.setitem(sys.modules, 'psycopg', None)
    result = checked(invoke())
    assert result['failure']['phase'] == 'DRIVER_IMPORT'
    assert result['failure']['exception_class'] == 'ModuleNotFoundError'
    assert result['attempted_connections'] == 0 and not calls


def test_missing_permission_or_inactive_tls_stops_safely(synthetic):
    fake, _, _ = synthetic
    class BadPermissions(FakeConnection):
        def execute(self, sql, args=()):
            if sql.startswith("SELECT current_user = 'nfl_app'"): return Cursor([(True,) * 11 + (False,)])
            return super().execute(sql, args)
    fake.connect = one_attempt(fake, lambda *_a, **_k: BadPermissions())
    result = checked(invoke())
    assert not result['checks']['role_database_readonly_and_permissions']
    assert result['failure']['phase'] == 'SCHEMA_PERMISSIONS'
    con = FakeConnection(); con.pgconn.ssl_in_use = False
    def inactive(*_a, **_k): return con
    fake.connect = one_attempt(fake, inactive)
    result = checked(invoke())
    assert not result['checks']['client_tls_active']
    assert result['failure']['phase'] == 'CONNECT_TLS_CHECK' and con.closed


def test_second_resolved_address_attempt_is_blocked_before_connection(synthetic):
    fake, _, _ = synthetic
    events=[]
    def connect(*_a, **_k):
        list(fake.Connection._connect_gen()); events.append('first_attempt')
        list(fake.Connection._connect_gen()); events.append('second_attempt')
        return FakeConnection()
    fake.connect=connect
    result=checked(invoke())
    assert events==['first_attempt']
    assert result['second_attempt_blocked'] and result['driver_attempts']==1
    assert result['failure']['category']=='SECOND_CONNECTION_ATTEMPT_BLOCKED'


def test_unsupported_driver_attempt_instrumentation_fails_closed(synthetic):
    fake, _, calls=synthetic
    fake.Connection=types.SimpleNamespace()
    result=checked(invoke())
    assert result['failure']['phase']=='INSTRUMENTATION'
    assert not calls and result['attempted_connections']==0


def test_real_psycopg_driver_loop_cannot_try_second_address_offline(monkeypatch):
    psycopg = pytest.importorskip('psycopg')
    import psycopg.connection as driver
    from app_core.nfl_storage_diagnostic import FailureTrace
    monkeypatch.setattr(socket,'socket',lambda *_a,**_k: pytest.fail('network forbidden'))
    monkeypatch.setattr(driver,'conninfo_attempts',lambda params: [dict(params),dict(params)])
    events=[]
    def synthetic_attempt(cls, conninfo='', *, timeout=0):
        events.append('first_attempt')
        raise psycopg.OperationalError('SYNTHETIC_ERROR')
        yield
    monkeypatch.setattr(psycopg.Connection,'_connect_gen',classmethod(synthetic_attempt))
    report={'checks':{}}
    trace=FailureTrace(setting,psycopg.connect,psycopg.Connection._connect_gen,report)
    sys.settrace(trace)
    try:
        with pytest.raises(RuntimeError,match='SECOND_CONNECTION_ATTEMPT_BLOCKED'):
            psycopg.connect('host=synthetic.invalid dbname=synthetic user=synthetic sslmode=verify-full')
    finally:
        sys.settrace(None)
    assert events==['first_attempt'] and report['second_attempt_blocked']
    assert len(trace.attempt_frames)==1
