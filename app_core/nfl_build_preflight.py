"""Connection-free owner preflight; observes Git and already loaded code independently."""
import ast
from datetime import datetime, timezone
import importlib
from importlib import metadata
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import types

MODULES = ('app_core.nfl_build_preflight', 'app_core.nfl_tls_trust', 'app.ui.nfl_storage_diagnostic',
           'app_core.nfl_diagnostic_receipts',
           'app_core.nfl_storage_diagnostic', 'app_core.nfl_workflow_store',
           'app_core.nfl_private_workflow', 'app.ui.publish_panel')
SOURCE_DEPENDENCIES = ('streamlit_app.py', 'app/ui/nfl_private_workflow.py',
    'scripts/ncaaf_filtered_history_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
    'scripts/check_launch_change_scope.py', 'docs/research/ncaaf-filtered-history-bindings-v1.json')
MAX_SOURCE = 2 * 1024 * 1024


def _git(root, *args):
    # Do not inherit Git tracing, credential helpers, external diff or pagers.
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'WINDIR', 'HOME',
                                    'USERPROFILE', 'TEMP', 'TMP') if k in os.environ}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0', GIT_CONFIG_NOSYSTEM='1',
               GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_LAZY_FETCH='1')
    result = subprocess.run(['git', '--no-pager', '-c', 'core.fsmonitor=false',
                             '-c', 'core.hooksPath=', '-c', 'protocol.allow=never', '-C', str(root), *args],
                            capture_output=True, timeout=5, env=env, check=True)
    if len(result.stdout) > MAX_SOURCE:
        raise ValueError('GIT_OUTPUT_LIMIT')
    return result.stdout


def _codes(code):
    values = {}
    for child in code.co_consts:
        if isinstance(child, types.CodeType):
            values[child.co_qualname] = child
            values.update(_codes(child))
    return values


def _literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        left, right = _literal(node.left), _literal(node.right)
        if type(left) is int and type(right) is int:
            value = left * right if isinstance(node.op, ast.Mult) else left + right
            if abs(value) <= 2**40:
                return value
        raise ValueError('UNSUPPORTED_LITERAL')
    return ast.literal_eval(node)


def _safe_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) in (str, bytes, int, float, bool, type(None)):
        return actual == expected
    if type(expected) in (tuple, list):
        return len(actual) == len(expected) and all(_safe_equal(a,b) for a,b in zip(actual,expected))
    if type(expected) in (set, frozenset):
        return all(type(v) in (str,int,bool) for v in actual) and actual == expected
    if type(expected) is dict:
        return (all(type(k) is str for k in actual) and set(actual)==set(expected)
                and all(_safe_equal(actual[k],expected[k]) for k in expected))
    return False


def _defaults_match(function, node, module):
    defaults = function.__defaults__ or ()
    if len(defaults) != len(node.args.defaults):
        return False
    pairs = list(zip(node.args.defaults,defaults))
    declared = {arg.arg:default for arg,default in zip(node.args.kwonlyargs,node.args.kw_defaults)
                if default is not None}
    actual = function.__kwdefaults__ or {}
    if set(actual) != set(declared):
        return False
    pairs += [(default,actual[name]) for name,default in declared.items()]
    for expression, value in pairs:
        if isinstance(expression, ast.Name):
            if expression.id not in vars(module) or value is not vars(module)[expression.id]:
                return False
        else:
            try:
                if not _safe_equal(value,_literal(expression)):
                    return False
            except (ValueError,TypeError,SyntaxError):
                return False
    return True


def _loaded_matches(module, raw, filename):
    """Never execute uploaded/source code. Compile for code-object comparison only."""
    expected = _codes(compile(raw, filename, 'exec', dont_inherit=True,
                              optimize=sys.flags.optimize))
    actual, callables = {}, {}
    for name, value in vars(module).items():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            actual[value.__qualname__] = value.__code__
            callables[value.__qualname__] = value
            for child_name, child in _codes(value.__code__).items():
                actual[child_name] = child
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for member in vars(value).values():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member) and member.__qualname__ in expected:
                    # contextmanager keeps the original generator in __wrapped__.
                    member = inspect.unwrap(member)
                    actual[member.__qualname__] = member.__code__
                    callables[member.__qualname__] = member
                    actual.update(_codes(member.__code__))
    # Class body objects have no callable counterpart; nested functions do.
    class_names = {n.name for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)}
    required = {k: v for k, v in expected.items() if k not in class_names}
    code_ok = bool(actual) and all(k in required and code == required[k] and code.co_filename == filename
                                   for k, code in actual.items())
    top_functions = {n.name for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    code_ok = code_ok and top_functions.issubset(actual)
    definitions = {n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    for cls in (n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)):
        code_ok = code_ok and all(cls.name + '.' + n.name in actual for n in cls.body
                                 if isinstance(n, ast.FunctionDef))
        definitions.update({cls.name+'.'+n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)})
    code_ok = code_ok and all(name in callables and _defaults_match(callables[name],node,module)
                             for name,node in definitions.items())
    constants_ok = True
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = _literal(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue  # Dynamic locks/import state are not claimed as verified constants.
            constants_ok = constants_ok and _safe_equal(vars(module).get(node.targets[0].id),value)
    return code_ok, constants_ok


def _process():
    result = {'pid': os.getpid(), 'started_at': 'UNKNOWN', 'start_ticks': 'UNKNOWN'}
    if sys.platform == 'linux':
        # Parenthesized comm may contain spaces; never disclose it.
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])
        boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime '))
        result.update(start_ticks=ticks, started_at=datetime.fromtimestamp(
            boot + ticks / os.sysconf('SC_CLK_TCK'), timezone.utc).isoformat())
    return result


def inspect_build(access):
    """No settings values, DSN parsing, Store construction, sockets or inference."""
    from app_core.nfl_private_workflow import OwnerAccess, _ISSUER, WorkflowError
    if type(access) is not OwnerAccess or access.proof is not _ISSUER or access.owner != 'Robert Velarde':
        raise WorkflowError('NFL_WORKFLOW_OWNER_REQUIRED')
    report = {'version': 'nfl-build-preflight-v1', 'status': 'UNVERIFIED',
              'observed_at': datetime.now(timezone.utc).isoformat(),
              'database_connections': 0, 'provider_requests': 0,
              'diagnostic_allowance_changed': False, 'checkout': {}, 'loaded_modules': []}
    phase = 'RUNTIME'
    try:
        report['process'] = _process()
        streamlit_version = metadata.version('streamlit')
        report['runtime'] = {'python': '.'.join(map(str, sys.version_info[:3])),
                             'psycopg': 'UNKNOWN', 'libpq': 'UNKNOWN',
                             'streamlit': streamlit_version if re.fullmatch(
                                 r'[0-9]+(?:\.[0-9]+){1,3}[a-zA-Z0-9.+-]*', streamlit_version) else 'UNKNOWN'}
        try:
            import psycopg
        except ImportError:
            pass
        else:
            from app_core.nfl_storage_diagnostic import runtime
            observed = runtime(psycopg)  # Released version sanitization; makes no connection.
            report['runtime'].update({k:observed[k] for k in ('python','psycopg','libpq')})
            report['runtime_checkout_head'] = observed['checkout_head']
            from app_core.nfl_tls_trust import inspect_trust
            report['tls_trust'] = inspect_trust(access, psycopg)
        phase = 'GIT'
        root = Path(__file__).resolve().parents[1]
        commit = _git(root, 'rev-parse', 'HEAD').decode().strip()
        tree = _git(root, 'rev-parse', commit + '^{tree}').decode().strip()
        if not all(re.fullmatch('[0-9a-f]{40}', v) for v in (commit, tree)):
            raise ValueError('INVALID_GIT_IDENTITY')
        report['checkout'].update(commit=commit, tree=tree)
        report['runtime_checkout_agrees'] = report.get('runtime_checkout_head', commit) == commit
        phase = 'LOADED_CODE'
        modules = {}
        for name in MODULES:
            module = importlib.import_module(name)
            modules[name] = module
            relative = name.replace('.', '/') + '.py'
            path = root / relative
            raw = path.read_bytes()
            if len(raw) > MAX_SOURCE:
                raise ValueError('SOURCE_LIMIT')
            location_ok = Path(module.__file__).resolve() == path.resolve()
            git_raw = _git(root, 'show', commit + ':' + relative)
            disk_matches = raw.replace(b'\r\n', b'\n') == git_raw.replace(b'\r\n', b'\n')
            code_ok, constants_ok = _loaded_matches(module, raw, str(path))
            report['loaded_modules'].append({'module': name, 'checkout_path_matches': location_ok,
                'source_matches_git': disk_matches, 'loaded_code_matches_source': code_ok,
                'literal_constants_match_source': constants_ok})
        diag, store, workflow = (modules[n] for n in (
            'app_core.nfl_storage_diagnostic', 'app_core.nfl_workflow_store',
            'app_core.nfl_private_workflow'))
        report['loaded_aliases_match'] = (diag.Store is store.Store and workflow.Store is store.Store
                                         and diag.production_service is workflow.production_service)
        paths = [name.replace('.', '/') + '.py' for name in MODULES] + list(SOURCE_DEPENDENCIES)
        changed = _git(root, 'diff', '--name-only', commit, '--', *paths).decode().splitlines()
        report['checkout']['relevant_changes'] = [p for p in changed if p in paths]
        last_commit = _git(root, 'rev-parse', 'HEAD').decode().strip()
        report['checkout_stable'] = last_commit == commit
        valid = report['loaded_aliases_match'] and report['runtime_checkout_agrees'] and report['checkout_stable'] and not changed and all(
            all(row[k] for k in ('checkout_path_matches', 'source_matches_git',
                                'loaded_code_matches_source', 'literal_constants_match_source'))
            for row in report['loaded_modules'])
        report['status'] = 'VERIFIED_RELEVANT_LOADED_CODE' if valid else 'IDENTITY_MISMATCH'
        report['scope'] = 'Relevant loaded Python code, supported defaults and literal constants; not all interpreter memory or packages'
    except Exception as exc:
        # No raw exceptions, paths, hostnames, environment values or exception attributes.
        report['failure'] = {'phase': phase, 'exception_class': type(exc).__name__
                             if type(exc).__name__ in ('OSError', 'ValueError', 'ImportError',
                                  'ModuleNotFoundError', 'CalledProcessError', 'TimeoutExpired')
                             else 'OTHER_EXCEPTION', 'category': 'BUILD_EVIDENCE_UNAVAILABLE'}
    return report
