"""Real pinned-driver loop, synthetic errors only; no external connections."""
import errno
import json
import socket
import ssl
import sys
from types import SimpleNamespace

import psycopg
import psycopg.connection as driver
import pytest

from app_core import nfl_diagnostic_receipts as receipts
from app_core.nfl_storage_diagnostic import FailureTrace, diagnose, driver_failure
from app_core.nfl_private_workflow import authenticate
from tests.test_nfl_storage_diagnostic import FakeConnection

TOKEN = 'SYNTHETIC-owner-token-valid'
SECRET = 'SYNTHETIC-password-key-url-never-retain'
HOST = 'synthetic.invalid'
ADDRESS = '192.0.2.73'
DSN = 'host=synthetic.invalid user=nfl_app password=' + SECRET + ' dbname=neondb port=5432 sslmode=verify-full sslrootcert=system'


def setting(name, default=''):
    return {'PARLAYPICKER_PUBLISH_TOKEN': TOKEN, 'PARLAYPICKER_NFL_DATABASE_URL': DSN,
            'PARLAYPICKER_NFL_STORE_KEY': SECRET * 2,
            'PARLAYPICKER_NFL_ACQUISITION_ENABLED': 'false',
            'PARLAYPICKER_NFL_INFERENCE_ENABLED': 'false'}.get(name, default)


@pytest.fixture(autouse=True)
def blocked(monkeypatch):
    assert psycopg.__version__ == '3.2.10'
    def forbidden(*a, **k): pytest.fail('external transport, mutation or prediction forbidden')
    for name in ('create_connection', 'getaddrinfo'):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ('connect', 'connect_ex', 'sendto'):
        monkeypatch.setattr(socket.socket, name, forbidden)
    import psycopg._connection_base as base
    monkeypatch.setattr(base.generators, 'connect', forbidden)
    for path in ('requests.sessions.Session.request', 'app_core.nfl_workflow_store.Store.put',
                 'app_core.nfl_workflow_store.Store.claim', 'app_core.nfl_workflow_acquisition.capture_scores',
                 'app_core.nfl_workflow_acquisition.capture_market', 'app_core.nfl_scenario_research.compute'):
        monkeypatch.setattr(path, forbidden)
    monkeypatch.setattr(driver, 'conninfo_attempts', lambda params: [dict(params, hostaddr=ADDRESS),
                                                                  dict(params, hostaddr='192.0.2.74')])


def checked(value):
    rendered = json.dumps(value, sort_keys=True)
    assert all(secret not in rendered for secret in (SECRET, HOST, ADDRESS, 'postgresql://'))
    return value


def install_attempt(monkeypatch, *, error=None, pauses=0):
    calls = []
    def attempt(cls, conninfo='', *, timeout=0):
        calls.append(timeout)
        for _ in range(pauses): yield None
        if error is not None: raise error
        result = FakeConnection()
        result.pgconn.used_gssapi = False
        return result
    monkeypatch.setattr(psycopg.Connection, '_connect_gen', classmethod(attempt))
    if pauses:
        def drain(gen, **kwargs):
            while True:
                try: next(gen)
                except StopIteration as done: return done.value
        monkeypatch.setattr(driver.waiting, 'wait_conn', drain)
    return calls


@pytest.mark.parametrize('pauses', [0, 1, 4])
def test_first_success_and_resumptions_are_not_failures(monkeypatch, pauses):
    calls = install_attempt(monkeypatch, pauses=pauses)
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert result['status'] == 'PASS' and calls == [5]
    assert result['attempted_connections'] == result['driver_attempts'] == 1
    assert 'first_connect_failure' not in result and 'blocked_attempt' not in result
    assert result['checks']['client_tls_active'] and result['checks']['role_database_readonly_and_permissions']


@pytest.mark.parametrize('error,category,state', [
    (psycopg.errors.InvalidPassword(SECRET + HOST + ADDRESS), 'AUTHENTICATION_REJECTED', '28P01'),
    (psycopg.OperationalError(SECRET + HOST + ADDRESS), 'CONNECTION_FAILURE_UNCLASSIFIED', None),
    (psycopg.errors.ConnectionTimeout('connection timeout expired\n' + SECRET), 'CONNECTION_TIMEOUT', None),
])
def test_real_loop_preserves_first_failure_before_blocking_second(monkeypatch, caplog, error, category, state):
    calls = install_attempt(monkeypatch, error=error)
    with caplog.at_level('DEBUG', logger='psycopg'):
        result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert result['status'] == 'STOPPED' and calls == [5]
    assert result['failure'] == result['first_connect_failure']
    assert result['failure']['category'] == category and result['failure']['sqlstate'] == state
    assert result['failure']['os_errno'] is result['failure']['os_winerror'] is None
    assert result['second_attempt_blocked'] and result['blocked_attempt']['category'] == 'SECOND_CONNECTION_ATTEMPT_BLOCKED'
    assert result['attempted_connections'] == result['driver_attempts'] == 1
    assert 'client_tls_active' not in result['checks']
    assert all(secret not in caplog.text for secret in (SECRET, HOST, ADDRESS))
    assert receipts.sanitize(result)['failure'] == result['failure']


def test_real_base_generator_error_is_observed_before_translation(monkeypatch):
    import psycopg._connection_base as base
    calls = []
    def fail(conninfo, *, timeout=0):
        calls.append(timeout)
        raise psycopg.errors.InvalidPassword(SECRET)
        yield
    monkeypatch.setattr(base.generators, 'connect', fail)
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert calls == [5] and result['failure']['sqlstate'] == '28P01'
    assert result['blocked_attempt']['category'] == 'SECOND_CONNECTION_ATTEMPT_BLOCKED'


def test_one_candidate_failure_remains_primary_without_guard_block(monkeypatch):
    monkeypatch.setattr(driver, 'conninfo_attempts', lambda params: [dict(params)])
    calls = install_attempt(monkeypatch, error=psycopg.OperationalError(SECRET))
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert calls == [5] and result['failure'] == result['first_connect_failure']
    assert 'blocked_attempt' not in result and 'second_attempt_blocked' not in result
    # A real driver error can also precede the first generator (for example,
    # address resolution). Preserve it without inventing an admitted attempt.
    def unresolved(params): raise psycopg.OperationalError(SECRET)
    monkeypatch.setattr(driver, 'conninfo_attempts', unresolved)
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert result['attempted_connections'] == 1 and result['driver_attempts'] == 0
    assert result['failure'] == result['first_connect_failure']
    assert result['failure']['category'] == 'CONNECTION_FAILURE_UNCLASSIFIED' and calls == [5]
    # An OS error escaping the pinned wait/driver path is genuine failure too;
    # it is not a normal generator event and must preserve its structured code.
    monkeypatch.setattr(driver, 'conninfo_attempts', lambda params: [dict(params)])
    calls = install_attempt(monkeypatch, error=TimeoutError(errno.ETIMEDOUT, SECRET))
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert result['failure'] == result['first_connect_failure']
    assert result['failure']['os_errno'] == errno.ETIMEDOUT
    assert result['failure']['category'] == 'CONNECTION_TIMEOUT' and calls == [5]
    assert result['driver_attempts'] == result['attempted_connections'] == 1
    assert 'blocked_attempt' not in result


@pytest.mark.parametrize('attribute,number,category', [
    ('errno', errno.ETIMEDOUT, 'CONNECTION_TIMEOUT'),
    ('winerror', 10061, 'CONNECTION_REFUSED'),
], ids=['posix-timeout', 'windows-refused'])
def test_real_loop_preserves_structured_os_code(monkeypatch, attribute, number, category):
    error = psycopg.OperationalError(SECRET + HOST)
    setattr(error, attribute, number)
    calls = install_attempt(monkeypatch, error=error)
    result = checked(diagnose(setting, authenticate(TOKEN, TOKEN), HOST))
    assert calls == [5] and result['failure']['os_' + attribute] == number
    assert result['failure']['category'] == category and result['second_attempt_blocked']


def test_interruption_is_not_reported_as_genuine_driver_error(monkeypatch):
    class Interrupt(BaseException): pass
    calls = install_attempt(monkeypatch, error=Interrupt(SECRET))
    report = {'checks': {}}
    trace = FailureTrace(setting, psycopg.connect, psycopg.Connection._connect_gen, report)
    sys.settrace(trace)
    try:
        with pytest.raises(Interrupt): psycopg.connect(DSN)
    finally:
        sys.settrace(None)
    assert calls == [130] and trace.connect_calls == len(trace.attempt_frames) == 1
    assert trace.record is None and 'first_connect_failure' not in report


@pytest.mark.parametrize('error,category,number,win', [
    (TimeoutError(errno.ETIMEDOUT, SECRET), 'CONNECTION_TIMEOUT', errno.ETIMEDOUT, None),
    (ConnectionRefusedError(errno.ECONNREFUSED, SECRET), 'CONNECTION_REFUSED', errno.ECONNREFUSED, None),
    (OSError(errno.ENETUNREACH, SECRET), 'NETWORK_UNREACHABLE', errno.ENETUNREACH, None),
    (socket.gaierror(socket.EAI_NONAME, SECRET), 'ADDRESS_RESOLUTION_FAILURE', socket.EAI_NONAME, None),
    (ssl.SSLCertVerificationError(SECRET), 'TLS_VERIFICATION_FAILED', None, None),
], ids=['timeout', 'refused', 'unreachable', 'dns', 'certificate'])
def test_structured_classification_never_retains_error_text(error, category, number, win):
    safe = checked(driver_failure(error))
    assert (safe['category'], safe['os_errno'], safe['os_winerror']) == (category, number, win)
    assert receipts.sanitize({'status': 'STOPPED', 'failure': safe})['failure'] == safe


@pytest.mark.parametrize('message,category', [
    ('connection failed: SSL error: certificate verify failed\n', 'TLS_VERIFICATION_FAILED'),
    ('connection failed: timeout expired\n', 'CONNECTION_TIMEOUT'),
    ('could not translate host name "' + HOST + '" to address: ' + SECRET, 'ADDRESS_RESOLUTION_FAILURE'),
    ('connection failed: Connection refused\n', 'CONNECTION_REFUSED'),
    ('connection failed: No route to host\n', 'NETWORK_UNREACHABLE'),
    ('connection failed: FATAL:  password authentication failed for user "' + SECRET + '"\n', 'AUTHENTICATION_REJECTED'),
    (SECRET + ' certificate verify failed', 'CONNECTION_FAILURE_UNCLASSIFIED'),
    ('x' * 8193 + 'timeout expired', 'CONNECTION_FAILURE_UNCLASSIFIED'),
], ids=['tls-libpq', 'timeout-libpq', 'dns-libpq', 'refused-libpq', 'network-libpq', 'auth-libpq', 'unrecognized', 'oversized'])
def test_only_allowlisted_libpq_clauses_leave_memory(message, category):
    result = checked(driver_failure(psycopg.OperationalError(message)))
    assert result['category'] == category and result['sqlstate'] is None


def test_hostile_error_attributes_and_stringification_fail_closed():
    class Dangerous(psycopg.OperationalError):
        def __init__(self): BaseException.__init__(self)
        @property
        def sqlstate(self): raise ValueError(SECRET)
        @property
        def errno(self): raise ValueError(SECRET)
        @property
        def winerror(self): return SECRET
        def __str__(self): raise ValueError(SECRET)
    result = checked(driver_failure(Dangerous()))
    assert result['sqlstate'] is result['os_errno'] is result['os_winerror'] is None
    assert result['category'] == 'CONNECTION_FAILURE_UNCLASSIFIED'
    bad = {'phase': [SECRET], 'exception_class': {SECRET: 1}, 'sqlstate': SECRET,
           'category': [SECRET], 'os_errno': True, 'os_winerror': 987654321, 'raw': SECRET}
    safe = receipts.sanitize({'status': 'STOPPED', 'first_connect_failure': bad, 'blocked_attempt': bad})
    checked(safe)
    assert safe['first_connect_failure']['os_errno'] is safe['blocked_attempt']['os_winerror'] is None


def test_first_failure_is_logged_displayed_reopened_and_never_reinvoked(monkeypatch, capsys):
    from app.ui import nfl_storage_diagnostic as ui
    calls = install_attempt(monkeypatch, error=psycopg.errors.InvalidPassword(SECRET))
    state = {'publication_token': TOKEN, 'nfl_workflow_diag_expected_host': HOST}
    ui._invoke(setting, state)
    original = dict(state['nfl_workflow_diag_result'])
    rows = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().err.splitlines()
            if line.startswith('NFL_STORAGE_DIAGNOSTIC ')]
    assert [row['stage'] for row in rows] == ['HANDLER_ENTERED', 'DIAGNOSTIC_STARTED', 'TERMINAL']
    checked(rows)
    assert rows[-1] == original and original['failure']['sqlstate'] == '28P01'
    ui._invoke(setting, state)
    assert state['nfl_workflow_diag_result'] == original and calls == [5]
    monkeypatch.setattr('app_core.nfl_storage_diagnostic.diagnose', lambda *a: pytest.fail('replay must not connect'))
    displayed, disabled = [], []
    def button(label, **kwargs):
        if kwargs['key'] == 'nfl_workflow_diag_run': disabled.append(kwargs['disabled'])
        return False
    # Exercise the unchanged renderer's receipt projection directly. Complete
    # real-widget authorization/reopen tests remain in the original receipt file.
    monkeypatch.setattr(ui, 'st', SimpleNamespace(caption=lambda *a: None, button=button,
        text_input=lambda *a, **k: HOST, json=displayed.append))
    ui.render(setting, state)
    ui.render(setting, state)
    assert displayed == [original, original] and disabled == [True, True] and calls == [5]
