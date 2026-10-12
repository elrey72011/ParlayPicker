"""Read-only per-validation Git object stream; no object/decision cache."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from pathlib import Path
import re
import subprocess
import threading
import os
import sys

_active = ContextVar('scope_git_object_stream', default=None)
_SPEC = re.compile(r'^(?:HEAD|[0-9a-f]{40}):[^\r\n\x00]+$')
MAX_OBJECT_BYTES = 16 * 1024 * 1024

class ObjectStream:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.process = None
        self.lock = threading.RLock()
    def _start(self):
        if self.process is None:
            self.process = subprocess.Popen(['git', 'cat-file', '--batch-command'], cwd=self.root,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def _header(self, command, spec):
        self._start()
        self.process.stdin.write((command+' '+spec+'\n').encode('utf-8'))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if line.endswith(b' missing\n'): return None
        fields = line.split()
        if len(fields) != 3 or not re.fullmatch(b'[0-9a-f]{40}', fields[0]):
            raise RuntimeError('SCOPE_GIT_BATCH_INVALID_HEADER')
        try: size = int(fields[2])
        except ValueError: raise RuntimeError('SCOPE_GIT_BATCH_INVALID_HEADER') from None
        if size < 0: raise RuntimeError('SCOPE_GIT_BATCH_INVALID_HEADER')
        return fields[0].decode('ascii'), fields[1], size
    def info(self, spec):
        with self.lock:
            return self._header('info', spec)
    def contents(self, spec):
        with self.lock:
            # Info avoids issuing an unsupported tree/oversized contents request.
            info = self._header('info', spec)
            if info is None or info[1] != b'blob' or info[2] > MAX_OBJECT_BYTES: return None
            result = self._header('contents', spec)
            if result is None or result[1] != b'blob' or result[2] > MAX_OBJECT_BYTES:
                raise RuntimeError('SCOPE_GIT_BATCH_OBJECT_CHANGED')
            raw = self.process.stdout.read(result[2])
            if len(raw) != result[2] or self.process.stdout.read(1) != b'\n':
                raise RuntimeError('SCOPE_GIT_BATCH_INCOMPLETE_OBJECT')
            oid = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if oid != result[0]: raise RuntimeError('SCOPE_GIT_BATCH_CORRUPT_OBJECT')
            return raw
    def close(self):
        if self.process is not None:
            self.process.stdin.close(); self.process.stdin = None
            process = self.process; self.process = None
            try: process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.communicate()
                raise RuntimeError('SCOPE_GIT_BATCH_CLOSE_TIMEOUT') from None
            if process.returncode: raise RuntimeError('SCOPE_GIT_BATCH_EXIT_FAILURE')

@contextmanager
def validation_stream(root):
    root = Path(root).resolve(); current = _active.get()
    if current is not None and current.root == root:
        yield current; return
    stream = ObjectStream(root); token = _active.set(stream)
    try: yield stream
    finally:
        _active.reset(token); stream.close()


class LiveRoot(type(Path())):
    """Fresh traversal on every call; only basename patterns are optimized."""
    def rglob(self, pattern, **options):
        pattern=str(pattern)
        if options or pattern in {'.','..'} or (pattern != '*' and not re.fullmatch(r'[A-Za-z0-9_.-]+',pattern)):
            yield from super().rglob(pattern,**options);return
        pending=[str(self)]
        match=os.path.normcase(pattern)
        while pending:
            directory=pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if pattern=='*' or os.path.normcase(entry.name)==match:
                            yield type(self)(entry.path)
                        if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
            except OSError as exc:
                if exc.errno not in (2,13,20,9,10062) and getattr(exc,'winerror',None) not in (21,123,1921):raise


POLICY_MODULE_BLOBS = {'canonical_download_scope': '9366753bb073c972df7eac80465a3cfa88d8e919', 'ci_scheduling_scope': 'faf317c99270118baa4d1da1010ec1d5d6fb9069', 'ci_timing_scope': 'c87f12becf84eef4aa1f73ba5a6227f91058acfb', 'drive_history_scope': '9f5e1864a55d9e103f22885e64d9d645812a7174', 'estimate_scope': '3736461db23d240f12e540cf749d7fef90c47235', 'football_catalog_scope': 'fba371100c2553f438bfcd06ace86bbe386b573f', 'football_research_scope': '87a40bcb70f4daf778165e245fceb811989afc77', 'guard_bootstrap_scope': '72dcd43832b331a8bda4b32bdb233bac082894d4', 'guard_performance_scope': 'a0d2b603a81a6bda33a98ad95ca11779d66dfc7f', 'guard_scan_scope': '58afb28ffd23e9e5d02c80df27cabe0feed59d03', 'home_runline_scope': '73f5df38e41505c05c062e419670862b5fcf5659', 'mlb_nhl_research_scope': 'c9b3ef62581ff6ea319e4bcb119e875701bfed6b', 'ncaaf_chronology_scope': 'fd6987db22db69b297c6ed8cca96df8173fd7fbd', 'ncaaf_cloud_backup_scope': 'd8e509831030e6ac4e2e892b1dce62a7a208cd02', 'ncaaf_compatibility_scope': '0a4ca827e6c45cbf31a56848196bafb2b1987337', 'ncaaf_custody_scope': '3c4a030cccb3e36127d6a8cae7b3a2a2e07bbd55', 'ncaaf_discovery_scope': '21dd00c4e87d79d21f77e6f1ef07965e83e2e2b9', 'ncaaf_normal_scope': 'ea978c9d039f454af8ace12285c99213fc5480fd', 'ncaaf_owner_scope': '2ce250090d795966ab50d7dfc52ad9a262501dce', 'ncaaf_pilot_scope': 'a143b17fedf19b78a1e54f3428ea240d9dbfbd7d', 'ncaaf_research_scope': '9a7868510cd23e1da714facd184ef5bc100a8f08', 'nfl_admission_scope': 'be7adeda34890ec1e21d3efbfd12cce864b05c4d', 'nfl_calibration_scope': '3df90eaa611197be6c639e3138de0094b6292bd9', 'nfl_diagnostics_scope': 'ab4ff7b0811b376e1c17d7cabe62ffc4a4639aa4', 'nfl_inputs_scope': '646f8e2a6c73dc578a1160a882563690580439bf', 'nfl_native_scope': 'de661c83e4961ad8f3b14b5db6e0718b803d52cb', 'nfl_observed_scope': 'f17c0741ca3fcb26d72ff7ea267a530070816884', 'nfl_odds_scope': 'c889098fa3f62f4cf599a4fb7a59f6eee440cecb', 'nfl_odds_timing_scope': '25ef6f8ca2e4b1c3111aa44605ad8860863d6c01', 'nfl_owner_scope': 'fcf566680917629ae8725548bc021daef96247c3', 'nfl_provenance_scope': 'd1aa16b6a7ba12ba176f5322b9241f1c201bd769', 'nfl_scenario_scope': 'cc2d4cd5c20d5b91858d61e08f4812f8542ff40c', 'nfl_scoring_scope': 'bc27fe162ce2a0769b6e32cdc6e16e86fb583bee', 'nfl_ui_scope': '481495accab4edea34c2205c9f3b0b76966e194d', 'nhl_puck_line_scope': 'e6affd5136a14d01b1924009bf7fca7935c710f6', 'pick_board_scope': 'a30f15c5313f56aee6e8f338123734eccdeef8aa', 'preview_binding_scope': '28b466742467965b5720b2245ee92430ee67f75a', 'producer_provenance_scope': '45b83121a640a1148bb766759bc4d61a9ab9c948', 'readiness_dashboard_scope': '03bb1a1a5241522d15e762b5daccad5d363d21d3', 'remote_canonical_scope': '4f4b32630bd4f9118a94fa7eae781db788098569', 'remote_continuation_scope': 'a0d7881520b194ac04405cb80c77dd9355b90e11', 'slate_audit_scope': 'f06b906cd0f8c2ec368407edab3d46087ff3eaa0', 'source_contract_scope': 'e4b95fd978c28a70c2f0d429def9f1d4fb026351', 'source_intake_scope': 'b88ef498b232b785e794799de5bfcec050fdb95c'}

def _policy_source(path):
    with path.open('rb') as f:raw=f.read(128*1024+1)
    if len(raw)>128*1024:raise RuntimeError('SCOPE_POLICY_READER_SOURCE_CHANGED')
    return raw.replace(b'\r\n',b'\n')

def install_policy_readers(g):
    if not getattr(g,'__file__',None):return
    origin=Path(g.__file__).resolve().parent
    for name,oid in POLICY_MODULE_BLOBS.items():
        module=sys.modules.get('scripts.'+name)
        if module is None or getattr(module,'_scope_policy_reads_installed',False):continue
        fn=getattr(module,'make_policy',None)
        path=origin/(name+'.py')
        if not callable(fn) or Path(module.__file__).resolve()!=path:continue
        raw=_policy_source(path)
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=oid:
            raise RuntimeError('SCOPE_POLICY_READER_SOURCE_CHANGED')
        def wrap(method,path,oid):
            @wraps(method)
            def wrapped(guard,*args,**kwargs):
                raw=_policy_source(path)
                if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=oid:
                    raise RuntimeError('SCOPE_POLICY_READER_SOURCE_CHANGED')
                with validation_stream(guard.ROOT):return method(guard,*args,**kwargs)
            return wrapped
        module.make_policy=wrap(fn,path,oid)
        module._scope_policy_reads_installed=True

def install(g):
    """Keep original Git operations; optimize only exact blob reads in a run."""
    install_policy_readers(g)
    if getattr(g, '_scope_live_reads_installed', False): return
    original_git, original_bytes, original_exists = g.git, g.git_bytes, g.exists_at
    def current():
        stream = _active.get()
        return stream if stream is not None and stream.root == Path(g.ROOT).resolve() else None
    @wraps(original_git)
    def git(*args, check=True):
        stream = current()
        if stream is not None and len(args) == 2 and args[0] == 'rev-parse' and _SPEC.fullmatch(args[1]):
            info = stream.info(args[1])
            if info is not None: return info[0]
        return original_git(*args, check=check)
    @wraps(original_bytes)
    def git_bytes(*args):
        stream = current()
        if stream is not None and len(args) == 2 and args[0] == 'show' and _SPEC.fullmatch(args[1]):
            raw = stream.contents(args[1])
            if raw is not None: return raw
        return original_bytes(*args)
    @wraps(original_exists)
    def exists_at(revision, path):
        stream = current(); spec = str(revision)+':'+str(path)
        if stream is not None and _SPEC.fullmatch(spec):
            return stream.info(spec) is not None
        return original_exists(revision, path)
    g.git, g.git_bytes, g.exists_at = git, git_bytes, exists_at
    def wrap(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            install_policy_readers(g)
            original_root=g.ROOT
            g.ROOT=LiveRoot(original_root)
            try:
                with validation_stream(g.ROOT):return method(*args, **kwargs)
            finally:g.ROOT=original_root
        return wrapped
    for name, method in list(vars(g).items()):
        if name.startswith('_run_') and name.endswith('_integrated') and callable(method):
            setattr(g, name, wrap(method))
    def predecessor(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            module=method(*args, **kwargs)
            install(module)
            return module
        return wrapped
    for name,method in list(vars(g).items()):
        if name.endswith('_predecessor') and callable(method):setattr(g,name,predecessor(method))
    g._scope_live_reads_installed = True
