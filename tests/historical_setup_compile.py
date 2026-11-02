"""Fixture-owned immutable helper code; every native authentication still runs."""
from collections import OrderedDict
import __future__
import builtins
import hashlib
import io
import marshal
from pathlib import Path
import sys
import types

MAX_SOURCE = 1024 * 1024
MAX_RETAINED = 8 * 1024 * 1024
MAX_ENTRIES = 2
FUTURE_FLAGS = sum(getattr(__future__, name).compiler_flag for name in __future__.all_feature_names)


def retained_size(value):
    pending, seen, size = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if size > MAX_RETAINED or len(seen) > 128000:
            return MAX_RETAINED + 1
        if isinstance(item, types.CodeType):
            pending.extend(getattr(item, name) for name in (
                'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_filename',
                'co_linetable', 'co_exceptiontable', 'co_freevars', 'co_cellvars'))
        elif type(item) in (tuple, frozenset):
            pending.extend(item)
        elif type(item) not in (str, bytes, int, float, complex, bool, type(None), type(Ellipsis)):
            return MAX_RETAINED + 1
    return size + 4096


class OwnedCompilation:
    """Reuse code, not modules, source bytes read from disk, or reader verdicts."""
    def __init__(self, root, callers):
        self.paths = frozenset(str(Path(root) / p) for p in (
            'scripts/ncaaf_filtered_history_scope.py', 'scripts/ci_fixture_stability_scope.py'))
        # Retain code objects, but never hash unrelated caller constants.
        self.callers = tuple(callers)
        self.caller_ids = frozenset(id(code) for code in self.callers)
        self.compiler = builtins.compile
        self.entries = OrderedDict()
        self.retained_bytes = 0
        self.hits = self.misses = 0
        self.closed = False

    def clear(self):
        self.closed = True
        self.entries.clear()
        self.retained_bytes = 0

    def __call__(self, source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
        caller = sys._getframe(1).f_code
        inherited = 0 if dont_inherit else caller.co_flags & FUTURE_FLAGS
        compiler = builtins.compile
        # Explicit inherited flags preserve the semantics of calling compile
        # from the original helper rather than from this small adapter.
        def native():
            return compiler(source, filename, mode, flags | inherited, True, optimize, **kwargs)
        if (self.closed or compiler is not self.compiler
                or not isinstance(compiler, types.BuiltinFunctionType)
                or str(filename) not in self.paths or id(caller) not in self.caller_ids
                or type(source) is not bytes or len(source) > MAX_SOURCE
                or mode != 'exec' or flags or dont_inherit or optimize != -1 or kwargs):
            return native()
        # Native readers have already checked their physical binding/source.
        # Read again here so a changed physical input cannot borrow old code.
        with io.FileIO(filename, 'r') as stream:
            physical = stream.read(MAX_SOURCE + 1)
        if physical.replace(b'\r\n', b'\n') != source:
            return native()
        key = (source, str(filename), inherited, sys.implementation.cache_tag, sys.flags.optimize)
        if key in self.entries:
            code, expected, cost = self.entries[key]
            if (not isinstance(code, types.CodeType)
                    or hashlib.sha256(marshal.dumps(code, 2)).hexdigest() != expected):
                raise ValueError('HISTORICAL_COMPILED_HELPER_CHANGED')
            self.hits += 1
            self.entries.move_to_end(key)
            return code
        self.misses += 1
        code = native()
        if not isinstance(code, types.CodeType):
            return code
        encoded = marshal.dumps(code, 2)
        cost = retained_size((key, code))
        while self.entries and (len(self.entries) >= MAX_ENTRIES or self.retained_bytes + cost > MAX_RETAINED):
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.retained_bytes -= removed
        if cost <= MAX_RETAINED:
            self.entries[key] = (code, hashlib.sha256(encoded).hexdigest(), cost)
            self.retained_bytes += cost
        return code


# Observed in the unchanged, authenticated native constructors on the retained
# local measurement seal. These are program identities, never approval facts.
PROGRAM_SOURCES = {
    '0772744a3b07e7a3aa67d889efff45743e4562e4fe8b781e2d2b3d5111424d9d': 1776896,
    '1ef895767373c06e537ab780bda679395d6a38eebcb7c2ee04330fa935cd3eba': 1911519,
    '1f56664efc9e8462cf0879430af78f27e0b530c4f6052477ad099d57d25e738e': 724467,
    '1f97ebc1b46de177d905b2a37b5c5414cd4fcefc5dffe78d344964fb5f9e0ff3': 1679131,
    '35ac29313e9509efd68894a86e4a98b109a58459605fe589cc6f86159e0dbc64': 1852999,
    '48c1cd7eb91a43ccc9b0467f1432ab3c827276b762cec13a1e6185f9f51cb711': 890021,
    '50867905878c73a124ff3e4f05fb6680c07c6bef13f7ddeaa6a569333dbb139e': 858161,
    '50bf44612636481124b9c7b80dc9eeaf6a6e4837ac65177ed6bade491caae9e0': 1896570,
    '522e938deb1114a7d3bfcfa09cf0b46a1f68b4961021abb88095263255a2da0f': 876400,
    '70741f61ae5b23173ee59f687c87e46f13d0e2034abc8284b5aec6aa468c6340': 1277721,
    '9197f27d8456a8cb161dc8e227d682daca482b28ba74e08c2ad6d63a5edcb42b': 1931437,
    'cb8e3f4aae84b1a6f31cc9842d0f47ec5b3203a589db6482d3eda59c62a70a0e': 845792,
}
PROGRAM_MAX_SOURCE = 8 * 1024 * 1024
PROGRAM_MAX_RETAINED = 32 * 1024 * 1024
PROGRAM_MAX_ENTRIES = 16


def program_size(value):
    """Account for the entire immutable graph, including code constants."""
    pending, seen, size = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if size > PROGRAM_MAX_RETAINED or len(seen) > 128000:
            return PROGRAM_MAX_RETAINED + 1
        if isinstance(item, types.CodeType):
            pending.extend(getattr(item, name) for name in (
                'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_filename',
                'co_linetable', 'co_exceptiontable', 'co_freevars', 'co_cellvars'))
        elif type(item) in (tuple, frozenset):
            pending.extend(item)
        elif type(item) not in (str, bytes, int, float, complex, bool, type(None), type(Ellipsis)):
            return PROGRAM_MAX_RETAINED + 1
    return size + 4096


class OwnedProgramExecution:
    """Run each native program afresh; retain only verified compilation output.

    The caller seeds this adapter with authenticated bootstrap code. Only code
    originating there, or in a catalogued program compiled from exact bytes,
    may introduce this adapter into a fresh predecessor namespace.
    """
    def __init__(self, filename, scope_key, pool, patch, active, seeds, catalog=None, proofs=None):
        self.filename = str(filename)
        self.scope_key = tuple(scope_key)
        self.catalog = frozenset((PROGRAM_SOURCES if catalog is None else catalog).items())
        self.pool = pool
        self.proofs = getattr(pool, 'proofs', {}) if proofs is None else proofs
        self.patch = patch
        self.active = active
        self.compiler = builtins.compile
        self.executor = builtins.exec
        self.digest = hashlib.sha256
        self.codes = {}
        self.closed = False
        self.hits = self.misses = self.uncached = 0
        self.compile_hook = self.compile
        self.exec_hook = self.execute
        for code in seeds:
            self.remember(code)
        if (pool.keys() != self.proofs.keys() or len(pool) > PROGRAM_MAX_ENTRIES
                or sum(value[2] for value in self.proofs.values()) > PROGRAM_MAX_RETAINED
                or any(self.proofs[key][0] is not value[0] or self.proofs[key][1] != value[1]
                       for key,value in pool.items())):
            raise ValueError('HISTORICAL_PROGRAM_POOL_CHANGED')

    def remember(self, code, verified=False):
        if (not isinstance(code, types.CodeType) or id(code) in self.codes
                or (not verified and program_size(code) > PROGRAM_MAX_RETAINED)):
            return
        pending = [code]
        while pending:
            current = pending.pop()
            if id(current) in self.codes:
                continue
            self.codes[id(current)] = current
            pending.extend(value for value in current.co_consts if isinstance(value, types.CodeType))

    def enabled(self, caller):
        return (not self.closed and self.active() and self.codes.get(id(caller)) is caller
                and builtins.compile is self.compiler and builtins.exec is self.executor
                and hashlib.sha256 is self.digest
                and isinstance(self.compiler, types.BuiltinFunctionType)
                and isinstance(self.executor, types.BuiltinFunctionType))

    def code(self, source, filename, flags, optimize, eligible):
        raw = source.encode('utf-8') if type(source) is str else source
        if not eligible or type(raw) is not bytes or len(raw) > PROGRAM_MAX_SOURCE:
            self.uncached += 1
            return self.compiler(source, filename, 'exec', flags, True, optimize)
        source_digest = self.digest(raw).hexdigest()
        if (source_digest,len(raw)) not in self.catalog:
            self.uncached += 1
            return self.compiler(source,filename,'exec',flags,True,optimize)
        key = (self.scope_key, source_digest, len(raw), type(source).__name__, str(filename),
               flags, optimize, sys.implementation.cache_tag, sys.flags.optimize, self.catalog)
        if key in self.pool:
            code, expected = self.pool[key]
            proof = self.proofs.get(key)
            if (not isinstance(code, types.CodeType) or code.co_filename != str(filename)
                    or proof is None or proof[0] is not code or proof[1] != expected):
                raise ValueError('HISTORICAL_PROGRAM_CODE_CHANGED')
            self.hits += 1
            self.remember(code,verified=True)
            return code
        self.misses += 1
        code = self.compiler(source, filename, 'exec', flags, True, optimize)
        cost=program_size((key,code))
        if cost > PROGRAM_MAX_RETAINED:
            return code
        self.remember(code,verified=True)
        digest=self.digest(marshal.dumps(code,2)).hexdigest()
        self.pool[key]=(code,digest)
        # The native compiler produced this exact code object. Its complete
        # constant graph was checked immutable above. Identity is a fresh proof
        # that neither it nor that graph changed; no approval result is retained.
        self.proofs[key]=(code,digest,cost)
        # Prefer larger repeated programs if the graph cap prevents retaining
        # the complete catalog. Do not churn a cyclic working set through LRU.
        while (len(self.pool) > PROGRAM_MAX_ENTRIES
                or sum(value[2] for value in self.proofs.values()) > PROGRAM_MAX_RETAINED):
            victim = min(self.pool, key=lambda item: (item[2], item[1], repr(item[:1])))
            del self.pool[victim]
            del self.proofs[victim]
        return code

    def compile(self, source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
        caller = sys._getframe(1).f_code
        inherited = 0 if dont_inherit else caller.co_flags & FUTURE_FLAGS
        if builtins.compile is not self.compiler:
            return builtins.compile(source, filename, mode, flags, dont_inherit, optimize, **kwargs)
        if (mode == 'exec' and not kwargs and not flags and not dont_inherit
                and str(filename) in (self.filename, '<string>')):
            return self.code(source, filename, inherited, optimize, self.enabled(caller))
        return self.compiler(source, filename, mode, flags | inherited, True, optimize, **kwargs)

    def install(self, namespace):
        if (self.closed or not self.active() or type(namespace) is not dict
                or namespace.get('__file__') != self.filename):
            return
        if any(name in namespace and namespace[name] is not hook
               for name, hook in (('compile', self.compile_hook), ('exec', self.exec_hook))):
            return  # Unrelated patches and explicit native names retain ownership.
        if 'compile' not in namespace:
            self.patch.setitem(namespace, 'compile', self.compile_hook)
        if 'exec' not in namespace:
            self.patch.setitem(namespace, 'exec', self.exec_hook)

    def execute(self, source, *args, **kwargs):
        caller = sys._getframe(1)
        if builtins.exec is not self.executor:
            return builtins.exec(source, *args, **kwargs)
        if len(args) > 2 or set(kwargs) - {'closure'}:
            return self.executor(source, *args, **kwargs)
        globals_ = args[0] if args else None
        locals_ = args[1] if len(args) == 2 else None
        if globals_ is None:
            globals_ = caller.f_globals
            if locals_ is None:
                locals_ = caller.f_locals
        if locals_ is None:
            locals_ = globals_
        closure = kwargs.get('closure')
        eligible = (self.enabled(caller.f_code) and type(globals_) is dict
                    and globals_.get('__file__') == self.filename)
        if type(source) in (str, bytes):
            if closure is not None:
                return self.executor(source, globals_, locals_, closure=closure)
            source = self.code(source, '<string>', caller.f_code.co_flags & FUTURE_FLAGS, -1, eligible)
        if eligible and self.codes.get(id(source)) is source:
            self.install(globals_)
        if closure is not None:
            return self.executor(source, globals_, locals_, closure=closure)
        return self.executor(source, globals_, locals_)

    def clear(self):
        self.closed = True
        self.codes.clear()
        self.pool = self.proofs = self.patch = self.active = None
