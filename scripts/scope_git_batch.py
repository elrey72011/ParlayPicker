"""Read-only per-validation Git object stream; no object/decision cache."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from pathlib import Path
import re
import subprocess
import threading

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

def install(g):
    """Keep original Git operations; optimize only exact blob reads in a run."""
    if getattr(g, '_scope_object_stream_installed', False): return
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
            with validation_stream(g.ROOT): return method(*args, **kwargs)
        return wrapped
    for name, method in list(vars(g).items()):
        if name.startswith('_run_') and name.endswith('_integrated') and callable(method):
            setattr(g, name, wrap(method))
    g._scope_object_stream_installed = True
