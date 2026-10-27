"""Owner-invoked, sanitized inspection of the released NFL store verifier.

No acquisition, inference, database initialization or retry is performed here.
"""
import ast
import hmac
import inspect
import logging
import os
import re
import subprocess
import sys
import threading

from app_core.nfl_private_workflow import production_service
from app_core.nfl_workflow_store import Store

_LOCK = threading.Lock()
_SETTINGS = {'PARLAYPICKER_NFL_DATABASE_URL', 'PARLAYPICKER_NFL_STORE_KEY',
             'PARLAYPICKER_NFL_ACQUISITION_ENABLED', 'PARLAYPICKER_NFL_INFERENCE_ENABLED'}
_CLASSES = {'OperationalError', 'InterfaceError', 'ProgrammingError', 'DatabaseError',
            'InsufficientPrivilege', 'UndefinedTable', 'QueryCanceled', 'InvalidPassword',
            'InvalidCatalogName', 'WorkflowError', 'ImportError', 'ModuleNotFoundError',
            'ValueError', 'TypeError', 'OSError'}
_CHILD_ENV = {'PATH', 'HOME', 'USERPROFILE', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'TMPDIR', 'LANG', 'LC_ALL'}
_IDENTITY_SQL = ("SELECT current_user = 'nfl_app', current_database() = 'neondb', "
    "current_setting('transaction_read_only') = 'on', "
    "has_schema_privilege(current_user,'nfl_private','USAGE'), "
    "NOT has_schema_privilege(current_user,'nfl_private','CREATE'), "
    "has_table_privilege(current_user,'nfl_private.meta','SELECT'), "
    "has_table_privilege(current_user,'nfl_private.records','SELECT'), "
    "has_table_privilege(current_user,'nfl_private.records','INSERT'), "
    "NOT has_table_privilege(current_user,'nfl_private.records','UPDATE,DELETE,TRUNCATE'), "
    "has_table_privilege(current_user,'nfl_private.claims','SELECT'), "
    "has_table_privilege(current_user,'nfl_private.claims','INSERT'), "
    "NOT has_table_privilege(current_user,'nfl_private.claims','UPDATE,DELETE,TRUNCATE')")


def safe_error(phase, exc):
    name = type(exc).__name__
    name = name if name in _CLASSES else 'OTHER_EXCEPTION'
    try:
        state = getattr(exc, 'sqlstate', None)
    except Exception:
        state = None
    state = state if isinstance(state, str) and re.fullmatch(r'[0-9A-Z]{5}', state) else None
    if state and state.startswith('28'):
        category = 'AUTHENTICATION_REJECTED'
    elif state == '42501':
        category = 'PERMISSION_DENIED'
    elif state == '42P01':
        category = 'RELATION_MISSING'
    elif state == '57014':
        category = 'QUERY_CANCELLED'
    elif name == 'WorkflowError':
        category = 'RELEASED_CHECK_REJECTED'
    elif phase == 'CONNECT':
        category = 'CONNECTION_FAILURE_UNCLASSIFIED'
    else:
        category = 'FAILURE_UNCLASSIFIED'
    return {'phase': phase, 'exception_class': name, 'sqlstate': state, 'category': category}


def runtime(psycopg):
    result = {'python': '.'.join(str(n) for n in sys.version_info[:3]),
              'psycopg': 'UNKNOWN', 'libpq': 'UNKNOWN', 'checkout_head': 'UNKNOWN'}
    version = getattr(psycopg, '__version__', '')
    if isinstance(version, str) and re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}[a-zA-Z0-9.+-]*', version):
        result['psycopg'] = version
    result['libpq'] = int(psycopg.pq.version())
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(Store))))
        env = {name: os.environ[name] for name in _CHILD_ENV if name in os.environ}
        value = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True,
                               text=True, check=True, timeout=2, env=env).stdout.strip()
        if re.fullmatch(r'[0-9a-f]{40}', value):
            result['checkout_head'] = value
    except Exception:
        pass
    return result


def _factory_lines():
    source = inspect.getsource(Store.postgres)
    tree = ast.parse(inspect.cleandoc(source))
    factory = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == 'factory')
    _, first = inspect.getsourcelines(Store.postgres)
    lines = {}
    for node in ast.walk(factory):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines[first + node.lineno - 1] = 'DRIVER_IMPORT'
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'conninfo_to_dict':
                lines[first + node.lineno - 1] = 'CONNECTION_SETTINGS_PARSE'
            elif isinstance(node.func, ast.Attribute) and node.func.attr == 'connect':
                lines[first + node.lineno - 1] = 'CONNECT'
    return lines


class FailureTrace:
    def __init__(self, setting, connect, attempt, report):
        self.report = report
        self.record = None
        self.connect_calls = 0
        self.connect_code = getattr(getattr(connect, '__func__', connect), '__code__', None)
        self.attempt_code = getattr(getattr(attempt, '__func__', attempt), '__code__', None)
        if (self.connect_code is None or self.attempt_code is None or
                '_connect_gen' not in self.connect_code.co_names or
                not self.attempt_code.co_flags & inspect.CO_GENERATOR):
            raise RuntimeError('CONNECT_INSTRUMENTATION_UNAVAILABLE')
        self.attempt_frames = set()
        self.pending, self.last_event = {}, {}
        self.codes = {setting.__code__: 'SETTINGS', production_service.__code__: 'PRODUCTION_SERVICE',
                      Store.postgres.__func__.__code__: 'STORE_FACTORY_SETUP', Store.verify.__code__: 'STORE_VERIFY',
                      Store._transaction.__wrapped__.__code__: 'TRANSACTION', Store._schema.__code__: 'SCHEMA',
                      Store._execute.__code__: 'SCHEMA'}
        self.factory_lines = _factory_lines()
        self.store_file = Store._schema.__code__.co_filename

    def __call__(self, frame, event, arg):
        code = frame.f_code
        if code.co_name == 'factory' and code.co_filename == self.store_file:
            phase = self.factory_lines.get(frame.f_lineno, 'STORE_FACTORY')
        else:
            phase = self.codes.get(code)
        if code is self.attempt_code and event == 'call' and frame not in self.attempt_frames:
            if self.attempt_frames:
                self.report['second_attempt_blocked'] = True
                self.record = {'phase': 'CONNECT', 'exception_class': None, 'sqlstate': None,
                               'category': 'SECOND_CONNECTION_ATTEMPT_BLOCKED'}
                raise RuntimeError('SECOND_CONNECTION_ATTEMPT_BLOCKED')
            self.attempt_frames.add(frame)
        if code is self.connect_code and event == 'call':
            self.connect_calls += 1
            if self.connect_calls > 1:
                raise RuntimeError('SECOND_CONNECTION_REJECTED')
        if code is self.connect_code and event == 'return' and arg is not None:
            try:
                if len(self.attempt_frames) != 1:
                    raise RuntimeError('CONNECT_ATTEMPT_NOT_OBSERVED')
                active = bool(arg.pgconn.ssl_in_use)
                self.report['checks']['client_tls_active'] = active
                if not active:
                    raise RuntimeError('TLS_INACTIVE')
            except Exception as exc:
                self.record = safe_error('CONNECT_TLS_CHECK', exc)
                try:
                    arg.close()
                except Exception:
                    pass
                raise
        if code is Store._schema.__code__ and event == 'return' and self.record is None:
            try:
                row = frame.f_locals['con'].execute(_IDENTITY_SQL).fetchone()
                good = isinstance(row, tuple) and len(row) == 12 and all(value is True for value in row)
                self.report['checks']['role_database_readonly_and_permissions'] = good
                if not good:
                    raise RuntimeError('IDENTITY_OR_PERMISSIONS_REJECTED')
            except Exception as exc:
                self.record = safe_error('SCHEMA_PERMISSIONS', exc)
                raise
        if phase and self.record is None:
            identity = id(frame)
            if event == 'exception':
                if phase != 'SETTINGS':
                    self.record = safe_error(phase, arg[1])
                else:
                    self.pending.setdefault(identity, safe_error(phase, arg[1]))
                    self.last_event[identity] = 'exception'
            elif event == 'line' and identity in self.pending:
                self.last_event[identity] = 'line'
            elif event == 'return' and identity in self.pending:
                if self.last_event.get(identity) == 'exception' and arg is None:
                    self.record = self.pending[identity]
                self.pending.pop(identity, None)
                self.last_event.pop(identity, None)
        return self


class _QuietDiagnosticThread(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record):
        return record.thread != self.thread_id


def diagnose(setting, access, expected_host):
    """Run once when explicitly invoked; no exception text is returned or logged."""
    report = {'status': 'STOPPED', 'checks': {}, 'attempted_connections': 0}
    if not _LOCK.acquire(blocking=False):
        report['failure'] = {'phase': 'INSTRUMENTATION', 'exception_class': None,
                             'sqlstate': None, 'category': 'DIAGNOSTIC_BUSY'}
        return report
    tracer, installed, quiet, filtered = None, False, None, []
    phase = 'SETTINGS'
    frozen = {}
    try:
        def restricted_setting(name, default=''):
            if name not in _SETTINGS:
                raise RuntimeError('UNAUTHORIZED_SETTING')
            value = setting(name, default)
            if name in frozen and (not isinstance(value, str) or
                    not hmac.compare_digest(value.encode(), frozen[name].encode())):
                raise RuntimeError('EFFECTIVE_SETTING_CHANGED')
            return value

        dsn = restricted_setting('PARLAYPICKER_NFL_DATABASE_URL')
        key = restricted_setting('PARLAYPICKER_NFL_STORE_KEY')
        acquire = restricted_setting('PARLAYPICKER_NFL_ACQUISITION_ENABLED', 'false')
        infer = restricted_setting('PARLAYPICKER_NFL_INFERENCE_ENABLED', 'false')
        phase = 'DRIVER_IMPORT'
        import psycopg
        from psycopg.conninfo import conninfo_to_dict
        report['runtime'] = runtime(psycopg)
        phase = 'CONNECTION_SETTINGS_PARSE'
        info = conninfo_to_dict(dsn)
        report['checks'] = {'dsn_present': isinstance(dsn, str) and bool(dsn),
                            'key_minimum_length': isinstance(key, str) and len(key.encode()) >= 32,
                            'host_present': bool(info.get('host')),
                            'production_host_matches': bool(expected_host) and hmac.compare_digest(
                                str(info.get('host', '')).casefold().encode(), str(expected_host).strip().casefold().encode()),
                            'restricted_user_matches': info.get('user') == 'nfl_app',
                            'password_present': bool(info.get('password')),
                            'database_matches': info.get('dbname') == 'neondb',
                            'port_matches': info.get('port') == '5432',
                            'verify_full': info.get('sslmode') == 'verify-full',
                            'system_roots': info.get('sslrootcert') == 'system',
                            'only_expected_parameters': set(info).issubset(
                                {'host', 'port', 'dbname', 'user', 'password', 'sslmode', 'sslrootcert'}),
                            'acquisition_disabled': acquire == 'false', 'inference_disabled': infer == 'false'}
        if not all(report['checks'].values()):
            report['failure'] = {'phase': 'SETTINGS', 'exception_class': None, 'sqlstate': None,
                                 'category': 'SETTINGS_CHECK_FAILED'}
            return report
        frozen.update({'PARLAYPICKER_NFL_DATABASE_URL': dsn, 'PARLAYPICKER_NFL_STORE_KEY': key,
                       'PARLAYPICKER_NFL_ACQUISITION_ENABLED': acquire,
                       'PARLAYPICKER_NFL_INFERENCE_ENABLED': infer})
        if sys.gettrace() is not None:
            report['failure'] = {'phase': 'INSTRUMENTATION', 'exception_class': None,
                                 'sqlstate': None, 'category': 'EXISTING_TRACE_UNSUPPORTED'}
            return report
        phase = 'INSTRUMENTATION'
        tracer = FailureTrace(setting, psycopg.connect, getattr(getattr(psycopg, 'Connection', None), '_connect_gen', None), report)
        quiet = _QuietDiagnosticThread()
        # Handler filters also cover records propagated from psycopg child loggers.
        # Drop only records from this invocation's thread, never other threads.
        loggers = [logging.getLogger(), logging.getLogger('psycopg')]
        loggers += [value for name, value in logging.root.manager.loggerDict.items()
                    if name.startswith('psycopg.') and isinstance(value, logging.Logger)]
        targets = list(loggers) + [handler for logger in loggers for handler in logger.handlers]
        if logging.lastResort is not None:
            targets.append(logging.lastResort)
        for target in dict.fromkeys(targets):
            target.addFilter(quiet)
            filtered.append(target)
        sys.settrace(tracer)
        installed = True
        phase = 'RELEASED_VERIFICATION'
        service = production_service(restricted_setting, access)
        service.verify()
        report['status'] = 'PASS'
    except Exception as exc:
        report['failure'] = tracer.record if tracer and tracer.record else safe_error(phase, exc)
    finally:
        if installed:
            sys.settrace(None)
        for target in filtered:
            target.removeFilter(quiet)
        if tracer is not None:
            report['attempted_connections'] = tracer.connect_calls
            report['driver_attempts'] = len(tracer.attempt_frames)
        _LOCK.release()
    return report
