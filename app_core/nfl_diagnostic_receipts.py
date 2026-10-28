"""Small sanitized lifecycle receipts; no settings, connections or private bodies."""
from datetime import datetime, timezone
import json
import re
import sys
import threading
import uuid

GUARD = 'nfl_storage_diagnostic_spent'
_CLAIM = threading.Lock()
_CHECKS = {'dsn_present', 'key_minimum_length', 'host_present', 'production_host_matches',
           'restricted_user_matches', 'password_present', 'database_matches', 'port_matches',
           'verify_full', 'system_roots', 'only_expected_parameters', 'acquisition_disabled',
           'inference_disabled', 'client_tls_active', 'role_database_readonly_and_permissions'}
_PHASES = {'SETTINGS', 'DRIVER_IMPORT', 'CONNECTION_SETTINGS_PARSE', 'INSTRUMENTATION',
           'RELEASED_VERIFICATION', 'PRODUCTION_SERVICE', 'STORE_FACTORY_SETUP', 'STORE_FACTORY',
           'STORE_VERIFY', 'CONNECT', 'CONNECT_TLS_CHECK', 'TRANSACTION', 'SCHEMA',
           'SCHEMA_PERMISSIONS', 'OWNER_OR_IMPORT', 'UI_DIAGNOSTIC'}
_CATEGORIES = {'AUTHENTICATION_REJECTED', 'PERMISSION_DENIED', 'RELATION_MISSING',
               'QUERY_CANCELLED', 'RELEASED_CHECK_REJECTED', 'CONNECTION_FAILURE_UNCLASSIFIED',
               'FAILURE_UNCLASSIFIED', 'SECOND_CONNECTION_ATTEMPT_BLOCKED', 'DIAGNOSTIC_BUSY',
               'SETTINGS_CHECK_FAILED', 'EXISTING_TRACE_UNSUPPORTED', 'DIAGNOSTIC_UNAVAILABLE'}
_CLASSES = {'OperationalError', 'InterfaceError', 'ProgrammingError', 'DatabaseError',
            'InsufficientPrivilege', 'UndefinedTable', 'QueryCanceled', 'InvalidPassword',
            'InvalidCatalogName', 'WorkflowError', 'ImportError', 'ModuleNotFoundError',
            'ValueError', 'TypeError', 'OSError', 'OTHER_EXCEPTION'}


def now():
    return datetime.now(timezone.utc).isoformat()


def unknown_failure(category, exc=None):
    name, sqlstate = 'UNKNOWN', None
    if exc is not None:
        name = type(exc).__name__
        name = name if name in _CLASSES else 'OTHER_EXCEPTION'
        try:
            candidate = getattr(exc, 'sqlstate', None)
        except Exception:
            candidate = None
        if type(candidate) is str and re.fullmatch('[0-9A-Z]{5}', candidate):
            sqlstate = candidate
    return {'status': 'STOPPED', 'attempted_connections': 'UNKNOWN', 'driver_attempts': 'UNKNOWN',
            'failure': {'phase': 'UI_DIAGNOSTIC', 'exception_class': name,
                        'sqlstate': sqlstate, 'category': category}}


def sanitize(result):
    """Project only the released nonsecret contract; never serialize arbitrary values."""
    if type(result) is not dict or result.get('status') not in ('PASS', 'STOPPED'):
        return unknown_failure('INVALID_DIAGNOSTIC_RECEIPT')
    clean = {'status': result['status']}
    for key in ('attempted_connections', 'driver_attempts'):
        count = result.get(key)
        clean[key] = count if type(count) is int and 0 <= count <= 100 else 'UNKNOWN'
    checks = result.get('checks')
    if type(checks) is dict:
        clean['checks'] = {k: v for k, v in checks.items() if k in _CHECKS and type(v) is bool}
    runtime = result.get('runtime')
    if type(runtime) is dict:
        clean['runtime'] = {}
        for key in ('python', 'psycopg'):
            value = runtime.get(key)
            clean['runtime'][key] = value if type(value) is str and re.fullmatch(
                r'[0-9]+(?:\.[0-9]+){1,3}[a-zA-Z0-9.+-]*', value) else 'UNKNOWN'
        value = runtime.get('libpq')
        clean['runtime']['libpq'] = value if type(value) is int and 0 <= value < 10000000 else 'UNKNOWN'
        value = runtime.get('checkout_head')
        clean['runtime']['checkout_head'] = value if type(value) is str and re.fullmatch('[0-9a-f]{40}', value) else 'UNKNOWN'
    failure = result.get('failure')
    if type(failure) is dict:
        state = failure.get('sqlstate')
        clean['failure'] = {
            'phase': failure.get('phase') if failure.get('phase') in _PHASES else 'UNKNOWN',
            'exception_class': failure.get('exception_class') if failure.get('exception_class') in _CLASSES else 'UNKNOWN',
            'sqlstate': state if type(state) is str and re.fullmatch('[0-9A-Z]{5}', state) else None,
            'category': failure.get('category') if failure.get('category') in _CATEGORIES else 'FAILURE_UNCLASSIFIED'}
    if type(result.get('second_attempt_blocked')) is bool:
        clean['second_attempt_blocked'] = result['second_attempt_blocked']
    return clean


def emit(receipt):
    # Only locally constructed/projected receipts reach here. This process log
    # survives a browser session; Cloud's retention is separate and not guaranteed.
    print('NFL_STORAGE_DIAGNOSTIC ' + json.dumps(receipt, sort_keys=True, separators=(',', ':')),
          file=sys.stderr, flush=True)


def claim(state):
    """The guard is checked independently of a stale enabled widget."""
    with _CLAIM:
        if state.get(GUARD) or state.get('nfl_workflow_diag_spent'):
            return None
        started = now()
        receipt = {**unknown_failure('DIAGNOSTIC_NOT_COMPLETED'),
                   'invocation_id': str(uuid.uuid4()), 'started_at': started, 'invoked_at': started,
                   'outcome': 'STARTED', 'status': 'STARTED', 'stage': 'HANDLER_ENTERED'}
        # Keep only a nonsecret spent tombstone outside revocable private viewer state.
        state[GUARD] = {'invocation_id': receipt['invocation_id'], 'started_at': started}
        state['nfl_workflow_diag_spent'] = True
        state['nfl_workflow_diag_result'] = receipt
        return receipt


def finish(state, started, result, outcome):
    receipt = {**result, 'invocation_id': started['invocation_id'],
               'started_at': started['started_at'], 'invoked_at': started['invoked_at'],
               'ended_at': now(), 'outcome': outcome, 'stage': 'TERMINAL'}
    try:
        emit(receipt)
    except Exception:
        receipt['server_log_recorded'] = False
    # Streamlit may interrupt a session-state write at a yield point. Retain
    # the terminal process receipt first, independently of display finalization.
    state['nfl_workflow_diag_result'] = receipt
    return receipt
