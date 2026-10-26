"""Temporary diagnostic: one process-bound, credential-free-output storage check."""
import base64
import hmac
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading

from app_core.nfl_workflow_store import Store, WorkflowError

ONCE = Path(tempfile.gettempdir()) / 'parlaypicker-nfl-hosted-diagnostic-v2.once'
PUBLIC_KEY_DER = 'MCowBQYDK2VwAyEAqtX1Yl0Bm/NOKpXzf2jOyD93DA5MfZodCp25t3ERdPE='
_PID = os.getpid()
_CHALLENGE = secrets.token_hex(32)
_LOCK = threading.Lock()
_SPENT = False
SAFE_SQLSTATE = re.compile(r'^[0-9A-Z]{5}$')
SAFE_VERSION = re.compile(r'^[0-9]+(?:\.[0-9]+){1,3}(?:[a-zA-Z0-9.+-]*)?$')


def metadata(psycopg):
    version = getattr(psycopg, '__version__', '')
    result = {'python': '.'.join(str(x) for x in sys.version_info[:3]),
              'psycopg': version if isinstance(version, str) and SAFE_VERSION.fullmatch(version) else 'UNKNOWN',
              'libpq': int(psycopg.pq.version()), 'checkout_head': 'UNKNOWN'}
    try:
        root = str(Path(__file__).resolve().parents[2])
        p = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True,
                           text=True, check=True, timeout=2)
        v = p.stdout.strip()
        if re.fullmatch(r'[0-9a-f]{40}', v):
            result['checkout_head'] = v
    except Exception:
        pass
    return result


def classify(exc, phase):
    state = getattr(exc, 'sqlstate', None)
    state = state if isinstance(state, str) and SAFE_SQLSTATE.fullmatch(state) else None
    name = type(exc).__name__
    allowed = {'WorkflowError', 'OperationalError', 'InterfaceError', 'ProgrammingError',
               'DatabaseError', 'InsufficientPrivilege', 'UndefinedTable',
               'QueryCanceled', 'InvalidPassword', 'InvalidCatalogName', 'OSError',
               'FileExistsError', 'ImportError', 'ModuleNotFoundError', 'ValueError'}
    name = name if name in allowed else 'OTHER_EXCEPTION'
    if state and state.startswith('28'):
        category = 'AUTHENTICATION_REJECTED'
    elif state == '42501':
        category = 'PERMISSION_DENIED'
    elif state == '42P01':
        category = 'RELATION_MISSING'
    elif state == '57014':
        category = 'QUERY_CANCELLED'
    elif isinstance(exc, WorkflowError):
        category = 'RELEASED_STORE_CHECK_REJECTED'
    elif phase == 'CONNECT':
        category = 'CONNECTION_FAILURE_UNCLASSIFIED'
    else:
        category = 'FAILURE_UNCLASSIFIED'
    return {'phase': phase, 'exception_class': name, 'sqlstate': state,
            'category': category}


def claim(authorization, marker):
    """Consume exactly one authorization tied to this imported process."""
    global _SPENT
    from cryptography.hazmat.primitives.serialization import load_der_public_key
    with _LOCK:
        if os.getpid() != _PID or _SPENT:
            raise WorkflowError('PRIVATE_DIAGNOSTIC_ALREADY_SPENT')
        try:
            signature = base64.b64decode(authorization, validate=True)
            public = load_der_public_key(base64.b64decode(PUBLIC_KEY_DER, validate=True))
            public.verify(signature, ('parlaypicker-nfl-diagnostic-v2:' + _CHALLENGE).encode())
        except Exception:
            raise WorkflowError('PRIVATE_DIAGNOSTIC_AUTHORIZATION_INVALID') from None
        _SPENT = True
        fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)


def check(setting, confirmed_host, authorization, connect=None, marker=ONCE):
    """Run once. No secret values, exception text or host digests are returned."""
    phase = 'IMPORT'
    con = None
    began = False
    report = {'status': 'STOPPED', 'attempted_connections': 0,
              'database_writes': False, 'checks': {}}
    report['runtime'] = {'python': '.'.join(str(x) for x in sys.version_info[:3]),
                         'psycopg': 'UNKNOWN', 'libpq': 'UNKNOWN', 'checkout_head': 'UNKNOWN'}
    try:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict
        report['runtime'] = metadata(psycopg)
        phase = 'EFFECTIVE_SETTINGS'
        dsn = setting('PARLAYPICKER_NFL_DATABASE_URL', '')
        key = setting('PARLAYPICKER_NFL_STORE_KEY', '')
        acquisition = setting('PARLAYPICKER_NFL_ACQUISITION_ENABLED', 'false')
        inference = setting('PARLAYPICKER_NFL_INFERENCE_ENABLED', 'false')
        info = conninfo_to_dict(dsn)
        checks = report['checks']
        checks.update(dsn_present=isinstance(dsn, str) and bool(dsn),
            key_minimum_length=isinstance(key, str) and len(key.encode()) >= 32,
            restricted_user=info.get('user') == 'nfl_app', database=info.get('dbname') == 'neondb',
            host_matches_owner_verified_input=isinstance(confirmed_host, str) and bool(confirmed_host)
                and hmac.compare_digest(info.get('host', '').encode(), confirmed_host.encode()),
            explicit_port=info.get('port') == '5432', verify_full=info.get('sslmode') == 'verify-full',
            system_roots=info.get('sslrootcert') == 'system',
            only_expected_parameters=set(info).issubset(
                {'host', 'port', 'dbname', 'user', 'password', 'sslmode', 'sslrootcert'}),
            acquisition_disabled=acquisition == 'false', inference_disabled=inference == 'false')
        if not all(checks.values()):
            report['failure'] = {'phase': phase, 'exception_class': None,
                                 'sqlstate': None, 'category': 'SETTINGS_CHECK_FAILED'}
            return report
        phase = 'ONE_SHOT_CLAIM'
        claim(authorization, marker)
        phase = 'CONNECT'
        report['attempted_connections'] = 1
        con = (connect or psycopg.connect)(dsn, autocommit=True, connect_timeout=5,
            options='-c statement_timeout=5000 -c lock_timeout=5000')
        phase = 'READ_ONLY_TRANSACTION'
        con.execute('BEGIN READ ONLY')
        began = True
        phase = 'RELEASED_SCHEMA_CHECK'
        Store(lambda _readonly: None, key.encode())._schema(con)
        checks['released_schema_check'] = True
        phase = 'IDENTITY_AND_PERMISSIONS'
        row = con.execute("SELECT current_user = 'nfl_app', current_database() = 'neondb', "
            "current_setting('transaction_read_only') = 'on', "
            "has_schema_privilege(current_user,'nfl_private','USAGE'), "
            "NOT has_schema_privilege(current_user,'nfl_private','CREATE'), "
            "has_table_privilege(current_user,'nfl_private.meta','SELECT'), "
            "has_table_privilege(current_user,'nfl_private.records','SELECT'), "
            "has_table_privilege(current_user,'nfl_private.records','INSERT'), "
            "NOT has_table_privilege(current_user,'nfl_private.records','UPDATE,DELETE,TRUNCATE'), "
            "has_table_privilege(current_user,'nfl_private.claims','SELECT'), "
            "has_table_privilege(current_user,'nfl_private.claims','INSERT'), "
            "NOT has_table_privilege(current_user,'nfl_private.claims','UPDATE,DELETE,TRUNCATE')").fetchone()
        checks.update(client_tls_active=bool(con.pgconn.ssl_in_use),
                      role_database_readonly_and_permissions=bool(row) and all(row))
        if not checks['client_tls_active'] or not checks['role_database_readonly_and_permissions']:
            report['failure'] = {'phase': phase, 'exception_class': None, 'sqlstate': None,
                                 'category': 'IDENTITY_TLS_OR_PERMISSION_CHECK_FAILED'}
            return report
        report['status'] = 'PASS'
        return report
    except Exception as exc:
        report['failure'] = classify(exc, phase)
        return report
    finally:
        if con is not None:
            try:
                if began:
                    con.execute('ROLLBACK')
            except Exception:
                report['cleanup_warning'] = 'ROLLBACK_FAILED'
            try:
                con.close()
            except Exception:
                report['cleanup_warning'] = 'CLOSE_FAILED'


def render(setting):
    """Called in place of normal workflow, after the existing owner gate."""
    import streamlit as st
    from app_core.nfl_private_workflow import authenticate
    authenticate(st.session_state.get('publication_token', ''), str(setting('PARLAYPICKER_PUBLISH_TOKEN')))
    st.subheader('One-time private NFL storage diagnostic')
    st.caption('Read-only; one process-bound authorization and one connection maximum. Normal NFL workflow is unavailable while this diagnostic patch is deployed.')
    st.caption('Diagnostic process challenge (not a password or credential):')
    st.code(_CHALLENGE)
    host = st.text_input('Previously verified production hostname', type='password')
    authorization = st.text_input('One-use diagnostic authorization', type='password')
    if st.button('Run one-time read-only storage diagnostic', key='nfl_diag_run'):
        st.session_state['nfl_diag_result'] = check(setting, host, authorization)
    if 'nfl_diag_result' in st.session_state:
        st.json(st.session_state['nfl_diag_result'])
