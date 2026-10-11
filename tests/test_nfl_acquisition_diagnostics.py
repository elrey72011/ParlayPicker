"""Labelled synthetic failures through the real CLI/HTTPS/custody path."""
import errno
import hashlib
import http.client
import json
import ssl
import socket
from functools import partial
import pytest
from app_core import nfl_scoring_acquisition as lane
from test_nfl_scoring_acquisition import fixture, NOW
from scripts.benchmark_drive_history_loading import blocked_network

CANARY = 'SYNTHETIC-SECRET api_key=private https://example.invalid/?sig=secret Cookie: private'


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def fake_https(monkeypatch, phase, error, status=200):
    calls = []
    class Sock:
        def settimeout(self, value): pass
    class Response:
        def __init__(self): self.status = status
        def getheaders(self):
            if phase == 'RESPONSE_HEADERS': raise error
            return [('Content-Type', 'text/csv'), ('Set-Cookie', CANARY)]
        def read1(self, size):
            if phase == 'BODY_READ': raise error
            return b''
    class Connection:
        def __init__(self, host, timeout):
            calls.append(self); self.sock = Sock(); self.closed = False
        def connect(self):
            if phase == 'HTTPS_CONNECT': raise error
        def request(self, *args, **kwargs):
            if phase == 'REQUEST_SEND': raise error
        def getresponse(self):
            if phase == 'RESPONSE_RECEIVE': raise error
            return Response()
        def close(self): self.closed = True
    monkeypatch.setattr(lane.http.client, 'HTTPSConnection', Connection)
    return calls


def test_reproduced_connect_diagnostic_survives_real_cli_bundle_readback(monkeypatch, tmp_path, capsys):
    from scripts.nfl_scoring_acquisition import main
    from app_core import ncaaf_pilot_setup
    p, a = fixture(monkeypatch, tmp_path)
    value = dict(version=lane.CONFIG_VERSION, plan_sha256=p['sha256'], permission=p['payload']['permission'], authorization=a,
                 custody=dict(id=p['payload']['custody_id'], root=str(tmp_path), storage_receipt_sha256='0'*64))
    value['review'] = dict(reviewer='Robert Velarde', reviewed_at='2026-10-17T00:00:00Z', subject_sha256=lane.digest(value),
                           status='OWNER_REVIEWED', scope='NFL scoring acquisition bindings only; no source acceptance.')
    raw = lane.custody.encode(lane.seal(value)).encode()
    paths = [tmp_path / ('SYNTHETIC-' + name + '.json') for name in ('plan', 'authorization', 'configuration')]
    for path, body in zip(paths, (json.dumps(p).encode(), json.dumps(a).encode(), raw)): path.write_bytes(body)
    monkeypatch.setattr(ncaaf_pilot_setup, 'inspect_security', lambda root: dict(status='VERIFIED', private_acl=True, encrypted=True, no_reparse=True))
    monkeypatch.setattr(lane, 'controlled', partial(lane.controlled, wall=lambda: NOW))
    monkeypatch.setattr(lane, 'acquire', partial(lane.acquire, wall=lambda: NOW))
    calls = fake_https(monkeypatch, 'HTTPS_CONNECT', PermissionError(errno.EACCES, CANARY))
    assert main([str(paths[0]), '--execute', '--authorization', str(paths[1]), '--custody-root', str(tmp_path),
                 '--controlled-bindings', str(paths[2]), '--bindings-sha256', hashlib.sha256(raw).hexdigest()]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    stored = (tmp_path / (p['sha256'] + '.nfl-scoring-unreviewed.json')).read_bytes()
    bundle = json.loads(stored)
    assert lane.digest(bundle['payload']) == bundle['sha256']
    assert report['reason'] == bundle['payload']['reason'] == 'NFL_ACQUIRE_TRANSPORT_FAILURE'
    assert report['attempted'] == 1 and len(calls) == 1 and calls[0].closed
    assert 'failure_diagnostics' in bundle['payload']
    assert report['failure_diagnostics'] == bundle['payload']['failure_diagnostics']
    assert report['failure_diagnostics']['phase'] == 'HTTPS_CONNECT'
    assert report['failure_diagnostics']['os_errno'] == errno.EACCES
    assert report['failure_diagnostics']['http_status'] == 'UNKNOWN'
    assert CANARY not in output + stored.decode()
    journal = tmp_path / (p['sha256'] + '.nfl-scoring-attempts.jsonl')
    before = journal.read_bytes()
    assert len(before.splitlines()) == 2
    assert main([str(paths[0]), '--execute', '--authorization', str(paths[1]), '--custody-root', str(tmp_path),
                 '--controlled-bindings', str(paths[2]), '--bindings-sha256', hashlib.sha256(raw).hexdigest()]) == 1
    assert json.loads(capsys.readouterr().out)['reason'] == 'NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'
    assert len(calls) == 1 and journal.read_bytes() == before


def certificate_error():
    error = ssl.SSLCertVerificationError(1, CANARY)
    error.verify_code = 20
    error.verify_message = CANARY
    return error


@pytest.mark.parametrize('phase,error,category,code', [
    ('HTTPS_CONNECT', socket.gaierror(-2, CANARY), 'DNS', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('HTTPS_CONNECT', PermissionError(errno.EACCES, CANARY, CANARY), 'OS', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('HTTPS_CONNECT', certificate_error(), 'TLS_CERTIFICATE_VERIFICATION', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('HTTPS_CONNECT', ssl.SSLError(1, CANARY), 'TLS', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('HTTPS_CONNECT', TimeoutError(CANARY), 'TIMEOUT', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('REQUEST_SEND', BrokenPipeError(errno.EPIPE, CANARY), 'CONNECTION', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('RESPONSE_RECEIVE', http.client.BadStatusLine(CANARY), 'HTTP_PROTOCOL', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('RESPONSE_RECEIVE', http.client.RemoteDisconnected(CANARY), 'CONNECTION', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('RESPONSE_HEADERS', OSError(errno.EIO, CANARY), 'OS', 'NFL_ACQUIRE_TRANSPORT_FAILURE'),
    ('BODY_READ', http.client.IncompleteRead(CANARY.encode(), 100), 'HTTP_PROTOCOL', 'NFL_ACQUIRE_STOPPED'),
    ('BODY_READ', ConnectionResetError(errno.ECONNRESET, CANARY), 'CONNECTION', 'NFL_ACQUIRE_STOPPED'),
    ('BODY_READ', TimeoutError(CANARY), 'TIMEOUT', 'NFL_ACQUIRE_STOPPED'),
    ('BODY_READ', RuntimeError(CANARY), 'UNKNOWN', 'NFL_ACQUIRE_STOPPED'),
])
def test_actual_transport_failure_facts_and_secret_exclusion(monkeypatch, tmp_path, phase, error, category, code):
    p, a = fixture(monkeypatch, tmp_path)
    calls = fake_https(monkeypatch, phase, error)
    class Timer:
        value = 0
        def __call__(self):
            self.value += .125
            return self.value
    result = lane.acquire(p, a, root=tmp_path, transport=lane.HttpsTransport(), wall=lambda: NOW, monotonic=Timer())
    payload = result['payload']; d = payload['failure_diagnostics']
    assert payload['reason'] == code and payload['status'] == 'INCOMPLETE'
    assert payload['attempted'] == len(calls) == 1 and calls[0].closed
    assert d['phase'] == phase and d['exception_category'] == category
    assert d['http_status'] == (200 if phase in ('RESPONSE_HEADERS', 'BODY_READ') else 'UNKNOWN')
    assert 0 < d['elapsed_seconds'] < 60
    assert d['os_errno'] == (error.errno if isinstance(error, OSError) and type(error.errno) is int else 'UNKNOWN')
    assert d['tls_error_code'] == (error.errno if isinstance(error, ssl.SSLError) else 'UNKNOWN')
    assert d['tls_verify_code'] == (20 if isinstance(error, ssl.SSLCertVerificationError) else 'UNKNOWN')
    serialized = lane.custody.encode(result) + json.dumps(lane.execution_report(result))
    for canary in (CANARY, 'api_key', 'sig=secret', 'Cookie: private', 'example.invalid'):
        assert canary not in serialized
    assert payload['object'] is None
    assert all(payload[k] is False for k in ('owner_reviewed', 'inference', 'source_acceptance', 'wagering_authority'))
    stored = json.loads((tmp_path / (p['sha256'] + '.nfl-scoring-unreviewed.json')).read_bytes())
    assert stored == result and lane.digest(stored['payload']) == stored['sha256']


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_only_actually_received_status_is_retained(monkeypatch, tmp_path, status):
    p, a = fixture(monkeypatch, tmp_path)
    calls = fake_https(monkeypatch, None, None, status=status)
    result = lane.acquire(p, a, root=tmp_path, transport=lane.HttpsTransport(), wall=lambda: NOW)
    assert result['payload']['reason'] == 'NFL_ACQUIRE_HTTP_STATUS'
    d = result['payload']['failure_diagnostics']
    assert d['http_status'] == status and d['phase'] == 'RESPONSE_VALIDATION'
    assert len(calls) == 1 and calls[0].closed
    assert CANARY not in lane.custody.encode(result)


def test_numeric_fields_allow_no_string_or_sensitive_attributes():
    from app_core import nfl_acquisition_diagnostics as d
    error = OSError(errno.EIO, CANARY)
    error.winerror = 10054
    value = d.facts(error, phase='HTTPS_CONNECT', started=1, ended=1.25)
    assert value['os_winerror'] == 10054 and value['elapsed_seconds'] == .25
    error.winerror = CANARY; error.errno = CANARY
    value = d.facts(error, phase='SYNTHETIC-secret-phase', http_status=CANARY, started=1, ended=float('nan'))
    assert value['phase'] == value['os_errno'] == value['os_winerror'] == value['http_status'] == value['elapsed_seconds'] == 'UNKNOWN'
    assert CANARY not in json.dumps(value)
    assert d.checked(value) == value
    for changed in (dict(value, message=CANARY), dict(value, tls_error_code=CANARY), dict(value, exception_category=CANARY)):
        with pytest.raises(ValueError, match='NFL_ACQUIRE_DIAGNOSTICS_INVALID'): d.checked(changed)


def test_historical_report_not_rewritten_and_integrity_gate_preserved():
    old = lane.seal(dict(status='INCOMPLETE', reason='NFL_ACQUIRE_TRANSPORT_FAILURE', attempted=1))
    original = lane.custody.encode(old)
    assert lane.execution_report(old) == old['payload']
    assert lane.custody.encode(old) == original and 'failure_diagnostics' not in old['payload']
    old['payload']['attempted'] = 0
    with pytest.raises(ValueError, match='NFL_ACQUIRE_PRIVATE_READBACK'): lane.execution_report(old)


def test_connection_constructor_failure_is_sanitized(monkeypatch, tmp_path):
    p, a = fixture(monkeypatch, tmp_path)
    def fail(*args, **kwargs): raise OSError(errno.EACCES, CANARY)
    monkeypatch.setattr(lane.http.client, 'HTTPSConnection', fail)
    result = lane.acquire(p, a, root=tmp_path, transport=lane.HttpsTransport(), wall=lambda: NOW)
    assert result['payload']['reason'] == 'NFL_ACQUIRE_STOPPED'
    assert result['payload']['failure_diagnostics']['phase'] == 'CONNECTION_SETUP'
    assert result['payload']['failure_diagnostics']['http_status'] == 'UNKNOWN'


def test_deadline_retains_existing_code_and_default_tls_configuration(monkeypatch, tmp_path):
    p, a = fixture(monkeypatch, tmp_path)
    seen = []
    class Connection:
        sock = None
        def __init__(self, *args, **kwargs): seen.append((args, kwargs))
        def connect(self): pytest.fail('synthetic deadline must precede connect')
        def close(self): pass
    def timed(*args): raise ValueError('NCAAF_PILOT_DEADLINE')
    monkeypatch.setattr(lane.http.client, 'HTTPSConnection', Connection)
    monkeypatch.setattr(lane, '_timed', timed)
    result = lane.acquire(p, a, root=tmp_path, transport=lane.HttpsTransport(), wall=lambda: NOW)
    assert result['payload']['reason'] == 'NFL_ACQUIRE_TRANSPORT_FAILURE'
    assert result['payload']['failure_diagnostics']['exception_category'] == 'TIMEOUT'
    assert seen[0][0] == ('github.com',) and set(seen[0][1]) == {'timeout'}
    assert 0 < seen[0][1]['timeout'] <= 10


def test_no_diagnostic_extension_or_report_changes_on_success(monkeypatch, tmp_path):
    from test_nfl_bounded_acquisition import Stream
    from test_nfl_scoring_profile import sized_csv
    p, a = fixture(monkeypatch, tmp_path)
    result = lane.acquire(p, a, root=tmp_path, transport=lambda *args: Stream(sized_csv(1024)), wall=lambda: NOW)
    assert result['payload']['status'] == 'CAPTURED_UNREVIEWED'
    assert 'failure_diagnostics' not in result['payload']
    assert lane.execution_report(result) == dict(status='CAPTURED_UNREVIEWED', reason=None, attempted=1)
