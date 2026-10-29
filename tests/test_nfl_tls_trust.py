"""Synthetic public certificates; actual Linux inspection remains connection-free."""
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import socket
import ssl
import stat
import struct
import sys
import types

import pytest
from streamlit.testing.v1 import AppTest
from app_core import nfl_tls_trust as trust
from app_core.nfl_private_workflow import authenticate, OwnerAccess

TOKEN = 'SYNTHETIC-TLS-owner-token-valid'


@pytest.fixture(autouse=True)
def no_connections(monkeypatch):
    def blocked(*a, **k):
        pytest.fail('TLS trust inspection attempted a forbidden connection or action')
    for target in ('socket.socket', 'socket.create_connection', 'socket.getaddrinfo',
                   'requests.sessions.Session.request', 'psycopg.connect',
                   'sqlite3.connect', 'app_core.nfl_workflow_store.Store.postgres',
                   'app_core.nfl_workflow_store.Store.verify',
                   'app_core.nfl_scenario_research.run', 'app_core.nfl_scenario_research.compute',
                   'app_core.nfl_workflow_acquisition.capture_scores',
                   'app_core.nfl_workflow_acquisition.capture_market'):
        monkeypatch.setattr(target, blocked)


@pytest.fixture(scope='module')
def public_ca():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'SYNTHETIC public CA')])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(1)
        .not_valid_before(now-timedelta(days=1)).not_valid_after(now+timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256()))
    return certificate.public_bytes(serialization.Encoding.PEM)


def bundle_file(monkeypatch, tmp_path, raw):
    path = tmp_path/'SYNTHETIC-public-ca.pem'
    path.write_bytes(raw)
    real_stat, real_open = os.stat, os.open
    def fixed_stat(name, **kw):
        return real_stat(path if name == trust.BUNDLE else name, **kw)
    def fixed_open(name, flags, *args):
        return real_open(path if name == trust.BUNDLE else name, flags, *args)
    monkeypatch.setattr(os, 'stat', fixed_stat)
    monkeypatch.setattr(os, 'open', fixed_open)
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'O_NONBLOCK', 0, raising=False)  # Windows synthetic Linux fixture only.
    return path


def test_owner_rejected_before_environment_procfs_or_bundle(monkeypatch):
    monkeypatch.setattr(trust, '_environment', lambda: pytest.fail('Unauthorized inspection'))
    for access in (None, object(), OwnerAccess('Robert Velarde', object())):
        with pytest.raises(ValueError, match='OWNER_REQUIRED'):
            trust.inspect_trust(access, None)


def test_fixed_bundle_parses_and_hashes_only_public_bytes(monkeypatch, tmp_path, public_ca):
    import hashlib
    bundle_file(monkeypatch, tmp_path, public_ca)
    row = trust._bundle(ssl)
    assert row['exists'] is row['readable'] is row['parses_as_ca_bundle'] is True
    assert row['size_bytes'] == len(public_ca)
    assert row['sha256'] == hashlib.sha256(public_ca).hexdigest()
    assert row['certificate_count'] == row['ca_certificate_count'] == 1
    assert row['parser'] == 'python_ssl_explicit_empty_context'
    assert row['neon_certificate_validation'] == 'NOT_TESTED'
    assert 'BEGIN CERTIFICATE' not in json.dumps(row)


@pytest.mark.parametrize('failure,status,exists,readable', [
    (FileNotFoundError('SYNTHETIC-secret-host'), 'MISSING', False, False),
    (PermissionError('SYNTHETIC-secret-password'), 'UNREADABLE', 'UNKNOWN', False),
    (OSError('SYNTHETIC-private-path'), 'READ_UNAVAILABLE', 'UNKNOWN', 'UNKNOWN')])
def test_missing_unreadable_or_unknown_bundle_is_sanitized(monkeypatch, failure, status, exists, readable):
    monkeypatch.setattr(sys, 'platform', 'linux')
    original = os.stat
    def bad(path, **kw):
        if path == trust.BUNDLE:
            raise failure
        return original(path, **kw)
    monkeypatch.setattr(os, 'stat', bad)
    row = trust._bundle(ssl)
    assert (row['status'], row['exists'], row['readable']) == (status, exists, readable)
    assert row['sha256'] == row['certificate_count'] == 'UNKNOWN'
    assert 'SYNTHETIC' not in json.dumps(row)


def test_exists_but_open_permission_denied_remains_distinct(monkeypatch, tmp_path, public_ca):
    bundle_file(monkeypatch, tmp_path, public_ca)
    original = os.open
    def denied(path, flags, *args):
        if path == trust.BUNDLE:
            raise PermissionError('SYNTHETIC-secret')
        return original(path, flags, *args)
    monkeypatch.setattr(os, 'open', denied)
    row = trust._bundle(ssl)
    assert row['exists'] is True and row['readable'] is False
    assert row['status'] == 'UNREADABLE' and row['sha256'] == 'UNKNOWN'


@pytest.mark.parametrize('raw', [b'', b'not a certificate', b'\xffSYNTHETIC-private-certificate-text'],
                         ids=['empty', 'text', 'non-ascii'])
def test_malformed_bundle_keeps_readability_but_rejects_parsing(monkeypatch, tmp_path, raw):
    bundle_file(monkeypatch, tmp_path, raw)
    row = trust._bundle(ssl)
    assert row['readable'] is True and row['parses_as_ca_bundle'] is False
    assert row['status'] == 'PARSE_REJECTED' and row['certificate_count'] == 'UNKNOWN'
    assert row['sha256'] == 'UNKNOWN'  # Never hash malformed/private-looking input into the receipt.
    assert 'private-certificate-text' not in json.dumps(row)


@pytest.mark.parametrize('mode,size', [(stat.S_IFIFO, 10), (stat.S_IFREG, trust.MAX_BUNDLE+1)])
def test_nonregular_or_oversize_bundle_is_not_opened(monkeypatch, mode, size):
    monkeypatch.setattr(sys, 'platform', 'linux')
    original = os.stat
    def fake(path, **kw):
        return types.SimpleNamespace(st_mode=mode, st_size=size) if path == trust.BUNDLE else original(path, **kw)
    monkeypatch.setattr(os, 'stat', fake)
    monkeypatch.setattr(os, 'open', lambda *a: pytest.fail('Unsafe bundle opened'))
    row = trust._bundle(ssl)
    assert row['status'] == 'NONREGULAR_OR_OVERSIZE'
    assert row['readable'] == row['sha256'] == 'UNKNOWN'


def test_candidate_path_is_fixed_not_environment_selected(monkeypatch, tmp_path, public_ca):
    bundle_file(monkeypatch, tmp_path, public_ca)
    for name in trust.ENV_NAMES:
        monkeypatch.setenv(name, 'SYNTHETIC-secret-host-password')
    before = {n:os.environ[n] for n in trust.ENV_NAMES}
    environment = trust._environment()
    assert all(r['present'] for r in environment.values())
    assert environment['SSL_CERT_FILE']['value_class'] == 'OTHER_REDACTED'
    assert trust._bundle(ssl)['parses_as_ca_bundle'] is True
    assert {n:os.environ[n] for n in trust.ENV_NAMES} == before
    assert 'SYNTHETIC-secret' not in json.dumps(environment)
    assert trust._path(trust.BUNDLE) == trust.BUNDLE
    assert trust._path(None) == 'UNSET'


def test_procfs_filters_paths_and_deduplicates_segments(monkeypatch):
    import io
    raw = (b'100-200 r-xp 0000 08:02 42 /usr/lib/libssl.so.3\n'
           b'200-300 r--p 0000 08:02 42 /usr/lib/libssl.so.3\n'
           b'300-400 r-xp 0000 08:02 43 /SYNTHETIC-secret-host/password.txt\n')
    monkeypatch.setattr(sys, 'platform', 'linux')
    def opened(path, mode):
        assert path == '/proc/self/maps' and mode == 'rb'
        return io.BytesIO(raw)
    monkeypatch.setattr('builtins.open', opened)
    rows = trust._maps()
    assert len(rows) == 1 and rows[0]['name'] == 'libssl.so.3'
    assert trust._identity(rows[0]) == dict(basename='libssl.so.3', mapped_inode=42,
        mapped_device=[8,2], source='actual_process_procfs')
    assert 'SYNTHETIC' not in json.dumps(rows)


def test_procfs_limit_rejects_without_raw_error(monkeypatch):
    import io
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('builtins.open', lambda *a: io.BytesIO(b'x'*(trust.MAX_MAPS+1)))
    with pytest.raises(ValueError, match='MAP_LIMIT'):
        trust._maps()


def elf(needed):
    raw = bytearray(1024)
    header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01'+b'\0'*10,
        3, 62, 1, 0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    raw[:len(header)] = header
    raw[64:120] = struct.pack('<IIQQQQQQ', 1, 4, 0, 0x1000, 0, 1024, 1024, 4096)
    raw[120:176] = struct.pack('<IIQQQQQQ', 2, 4, 256, 0x1100, 0, 64, 64, 8)
    raw[256:272] = struct.pack('<qQ', 5, 0x1200)
    raw[272:288] = struct.pack('<qQ', 1, 0)
    raw[288:304] = struct.pack('<qQ', 0, 0)
    raw[512:512+len(needed)+1] = needed.encode()+b'\0'
    return bytes(raw)


def test_bounded_elf_associates_exact_needed_object_not_all_ssl_mappings(monkeypatch):
    anchor = {'path':'/public/anchor', 'name':'pq.so'}
    one = {'path':'/public/libpq.so.5', 'name':'libpq.so.5'}
    other = {'path':'/other/libpq-other.so.5', 'name':'libpq-other.so.5'}
    monkeypatch.setattr(trust, '_mapped_bytes', lambda r: elf('libpq.so.5'))
    assert trust._needed(anchor) == ['libpq.so.5']
    assert trust._dependency(anchor, [one, other], 'pq') is one
    assert trust._dependency(anchor, [one, dict(one)], 'pq') is None


@pytest.mark.parametrize('raw', [b'', b'not ELF', b'\x7fELF'+b'\0'*100],
                         ids=['empty', 'not-elf', 'unsupported-header'])
def test_unrecognized_mapped_object_does_not_become_driver_evidence(monkeypatch, raw):
    monkeypatch.setattr(trust, '_mapped_bytes', lambda r: raw)
    with pytest.raises(ValueError):
        trust._needed({})


def test_driver_getters_only_use_already_loaded_crypto(monkeypatch):
    import ctypes
    import _ctypes
    row = dict(path='/public/libcrypto.so.3', name='libcrypto.so.3', inode=1, device=[1,2])
    calls = []
    class Function:
        def __init__(self, value):
            self.value = value
        def __call__(self, *args):
            return self.value
    library = types.SimpleNamespace(_handle=123, OpenSSL_version=Function(b'OpenSSL 3.5.3 SYNTHETIC-never-emit'),
        X509_get_default_cert_file=Function(b'/SYNTHETIC-private-host/default.pem'),
        X509_get_default_cert_dir=Function(b'/etc/ssl/certs'),
        X509_get_default_cert_file_env=Function(b'SSL_CERT_FILE'),
        X509_get_default_cert_dir_env=Function(b'SSL_CERT_DIR'))
    def no_load(path, **kw):
        calls.append((path, kw))
        return library
    monkeypatch.setattr(trust, '_mapped_bytes', lambda r: b'public object already mapped')
    monkeypatch.setattr(os, 'RTLD_NOLOAD', 4, raising=False)
    monkeypatch.setattr(os, 'RTLD_NOW', 2, raising=False)
    monkeypatch.setattr(ctypes, 'CDLL', no_load)
    closed = []
    monkeypatch.setattr(_ctypes, 'dlclose', closed.append, raising=False)
    result = trust._crypto_defaults(row)
    assert calls == [(row['path'], {'mode':6})]
    assert result['openssl_version'] == '3.5.3'
    assert result['default_ca_file'] == 'OTHER_REDACTED'
    assert result['default_ca_directory'] == '/etc/ssl/certs'
    assert result['new_library_load_allowed'] is False
    assert closed == [123]
    assert result['ca_file_environment_supported'] is True
    assert result['default_ca_file_metadata']['exists'] == 'UNKNOWN'
    assert 'SYNTHETIC' not in json.dumps(result)


def test_no_noload_support_does_not_load_a_library(monkeypatch):
    import ctypes
    monkeypatch.delattr(os, 'RTLD_NOLOAD', raising=False)
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: pytest.fail('New library load'))
    assert trust._crypto_defaults({}) == {'status':'UNKNOWN'}


def test_missing_procfs_keeps_python_and_driver_findings_separate(monkeypatch):
    monkeypatch.setattr(trust, '_maps', lambda: None)
    result = trust.inspect_trust(authenticate(TOKEN,TOKEN), types.SimpleNamespace(pq=types.SimpleNamespace(__impl__='binary')))
    assert result['psycopg']['implementation'] == 'binary'
    assert result['psycopg']['ssl'] == result['loaded_libraries'] == 'UNKNOWN'
    assert result['python_ssl']['source'] == 'python_ssl_not_assumed_equal_to_psycopg'
    assert result['neon_certificate_validation'] == 'NOT_TESTED'
    assert result['network_calls'] == result['database_connections'] == 0


def test_real_process_inspection_without_any_connection(monkeypatch, record_testsuite_property):
    import psycopg
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Inspection executed a command'))
    result = trust.inspect_trust(authenticate(TOKEN,TOKEN), psycopg)
    record_testsuite_property('NFL_TLS_TRUST_QUALIFICATION', json.dumps({
        'environment':'synthetic qualification process, not the hosted app',
        'platform':sys.platform, 'python':sys.version.split()[0],
        'psycopg':psycopg.__version__, 'libpq':psycopg.pq.version(), 'report':result}, sort_keys=True))
    assert result['network_calls'] == result['database_connections'] == 0
    assert result['settings_changed'] is False
    assert result['neon_certificate_validation'] == 'NOT_TESTED'
    if sys.platform == 'linux':
        # Native Linux execution, including binary libpq dependency association.
        assert isinstance(result['loaded_libraries'], list)
        if psycopg.pq.__impl__ in ('binary','c'):
            assert result['psycopg']['libpq'] != 'UNKNOWN'
            assert result['psycopg']['ssl'] != 'UNKNOWN'
            assert result['psycopg']['trust_defaults']['status'] == 'OBSERVED'
        if result['ca_bundle']['exists'] is True:
            assert result['ca_bundle']['readable'] is True
            assert result['ca_bundle']['parses_as_ca_bundle'] is True
            assert result['ca_bundle']['certificate_count'] > 0
        else:
            assert result['ca_bundle']['status'] == 'MISSING'
    else:
        assert result['loaded_libraries'] == 'UNKNOWN'
        assert result['ca_bundle']['exists'] == result['ca_bundle']['readable'] == 'UNKNOWN'
        assert result['ca_bundle']['status'] == 'UNAVAILABLE_PLATFORM'


def test_owned_noload_handle_is_released_on_getter_failure(monkeypatch):
    import ctypes
    import _ctypes
    row = dict(path='/public/libcrypto.so.3', name='libcrypto.so.3', inode=1, device=[1,2])
    class Failed:
        def __call__(self, *args):
            raise OSError('SYNTHETIC-secret-host')
    monkeypatch.setattr(trust, '_mapped_bytes', lambda r: b'public already mapped object')
    monkeypatch.setattr(os, 'RTLD_NOLOAD', 4, raising=False)
    monkeypatch.setattr(os, 'RTLD_NOW', 2, raising=False)
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: types.SimpleNamespace(_handle=456, OpenSSL_version=Failed()))
    closed = []
    monkeypatch.setattr(_ctypes, 'dlclose', closed.append, raising=False)
    with pytest.raises(OSError):
        trust._crypto_defaults(row)
    assert closed == [456]


def test_default_metadata_never_opens_contents_or_environment_path(monkeypatch):
    monkeypatch.setattr('builtins.open', lambda *a, **k: pytest.fail('Default contents opened'))
    monkeypatch.setattr(os, 'open', lambda *a, **k: pytest.fail('Default contents opened'))
    original = os.stat
    calls = []
    def missing(path):
        calls.append(path)
        raise FileNotFoundError('SYNTHETIC-private-error')
    monkeypatch.setattr(os, 'stat', missing)
    assert trust._default_metadata('/opt/public/ssl/cert.pem', False) == dict(exists=False, readable=False)
    assert calls == ['/opt/public/ssl/cert.pem']
    for path in ('/SYNTHETIC-private/value', '/etc/../private/cert.pem', '/etc/private/password'):
        assert trust._default_metadata(path, False) == dict(exists='UNKNOWN', readable='UNKNOWN')
    assert len(calls) == 1


def test_driver_observation_failure_does_not_emit_raw_error(monkeypatch):
    def failed():
        raise OSError('SYNTHETIC-secret-password-host')
    monkeypatch.setattr(trust, '_maps', failed)
    result = trust.inspect_trust(authenticate(TOKEN,TOKEN), None)
    assert result['driver_detail_status'] == 'UNKNOWN'
    assert result['psycopg']['ssl'] == 'UNKNOWN'
    assert 'SYNTHETIC-secret' not in json.dumps(result)


def test_ui_owner_action_once_rerender_and_authentication_loss(monkeypatch):
    calls = []
    def inspect(access, psycopg):
        calls.append(access)
        return {'version':'SYNTHETIC-TLS', 'network_calls':0, 'database_connections':0}
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', TOKEN)
    monkeypatch.setattr(trust, 'inspect_trust', inspect)
    at = AppTest.from_string('from app.ui.publish_panel import render_publish_panel\nrender_publish_panel(None,None)').run()
    assert not calls and not at.exception
    at.text_input(key='publication_token').set_value('wrong').run()
    assert not calls and not at.button
    at.text_input(key='publication_token').set_value(TOKEN).run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert not calls
    at.session_state['nfl_workflow_diag_spent'] = True
    at.button(key='nfl_workflow_build_run').click().run()
    assert len(calls) == 1 and not at.exception
    assert at.session_state['nfl_workflow_build_result']['tls_trust']['version'] == 'SYNTHETIC-TLS'
    at.run()
    assert len(calls) == 1 and at.session_state['nfl_workflow_diag_spent'] is True
    at.text_input(key='publication_token').set_value('wrong').run()
    assert len(calls) == 1 and 'nfl_workflow_build_result' not in at.session_state.filtered_state
