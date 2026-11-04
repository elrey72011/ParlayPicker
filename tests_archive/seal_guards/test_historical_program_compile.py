"""Synthetic fresh execution and native-bootstrap code-only reuse boundaries."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests import historical_setup_compile as reuse

BODY = '''if not allowed:
    raise ValueError('SYNTHETIC_PERMISSION_REQUIRED')
if observed >= permission_end:
    raise ValueError('SYNTHETIC_PERMISSION_EXPIRED')
if model != 'SYNTHETIC reviewed model':
    raise ValueError('SYNTHETIC_MODEL_CHANGED')
if claimed:
    raise ValueError('SYNTHETIC_DUPLICATE_CLAIM')
values = []
def result():
    return values
'''


class ProgramPool(dict):
    def __init__(self):
        super().__init__()
        self.proofs = {}


@contextmanager
def synthetic(tmp_path, pool=None, scope=('SYNTHETIC authenticated inputs',), body=BODY, future=False):
    filename = str(tmp_path / 'scripts/check_launch_change_scope.py')
    module = ModuleType('SYNTHETIC native program constructor')
    module.__file__ = filename
    module.ModuleType = ModuleType
    module.SOURCE = body
    text = ('from __future__ import annotations\n' if future else '') + '''def construct(**data):
    child = ModuleType('SYNTHETIC fresh predecessor')
    child.__file__ = __file__
    child.__dict__.update(allowed=True, observed=0, permission_end=1,
                          model='SYNTHETIC reviewed model', claimed=False)
    child.__dict__.update(data)
    exec(SOURCE, child.__dict__)
    return child
'''
    seed = compile(text, filename, 'exec', dont_inherit=True)
    exec(seed, module.__dict__)
    pool = ProgramPool() if pool is None else pool
    active = [True]
    catalog = {hashlib.sha256(body.encode()).hexdigest(): len(body.encode())}
    with pytest.MonkeyPatch.context() as patch:
        adapter = reuse.OwnedProgramExecution(filename, scope, pool, patch, lambda: active[0], [seed], catalog)
        adapter.install(module.__dict__)
        try:
            yield module, adapter, pool, active
        finally:
            adapter.clear()
    assert 'compile' not in vars(module) and 'exec' not in vars(module)


def test_program_code_reused_but_module_data_and_functions_remain_fresh(tmp_path):
    with synthetic(tmp_path) as (module, adapter, pool, _):
        first = module.construct()
        first.values.append('SYNTHETIC first-only value')
        second = module.construct()
        assert first is not second and vars(first) is not vars(second)
        assert first.result is not second.result and first.result.__code__ is second.result.__code__
        assert second.result() == [] and first.result() != second.result()
        assert adapter.hits == 1 and adapter.misses == 1
        assert pool and reuse.program_size(tuple(pool.items())) <= reuse.PROGRAM_MAX_RETAINED
    assert 'compile' not in vars(first) and 'exec' not in vars(first)
    assert 'compile' not in vars(second) and 'exec' not in vars(second)


@pytest.mark.parametrize('changed,reason', [
    ({'allowed': False}, 'SYNTHETIC_PERMISSION_REQUIRED'),
    ({'observed': 1}, 'SYNTHETIC_PERMISSION_EXPIRED'),
    ({'model': 'SYNTHETIC altered model'}, 'SYNTHETIC_MODEL_CHANGED'),
    ({'claimed': True}, 'SYNTHETIC_DUPLICATE_CLAIM'),
])
def test_warm_compilation_never_caches_permission_model_or_claim_outcomes(tmp_path, changed, reason):
    with synthetic(tmp_path) as (module, adapter, _, _):
        module.construct()
        with pytest.raises(ValueError, match=reason):
            module.construct(**changed)
        with pytest.raises(ValueError, match=reason):
            module.construct(**changed)
        assert adapter.hits == 2  # The same code executes and rejects twice.


def test_changed_source_is_not_served_by_an_old_compilation(tmp_path):
    with synthetic(tmp_path) as (module, adapter, pool, _):
        original = module.construct()
        saved = tuple(pool.items())
        module.SOURCE += "\nvalues.append('SYNTHETIC changed source')\n"
        changed = module.construct()
        assert changed.result() != original.result() and adapter.hits == 0
        assert tuple(pool.items()) == saved and adapter.uncached == 1
        assert 'compile' not in vars(changed) and 'exec' not in vars(changed)


def test_exact_input_context_is_required_across_fixture_owners(tmp_path):
    pool = ProgramPool()
    with synthetic(tmp_path, pool) as (first, a, _, _):
        one = first.construct()
    with synthetic(tmp_path, pool) as (second, b, _, _):
        two = second.construct()
        assert b.hits == 1 and one.result.__code__ is two.result.__code__
        assert one.result is not two.result and two.result() == []
    with synthetic(tmp_path, pool, scope=('SYNTHETIC changed binding',)) as (third, c, _, _):
        three = third.construct()
        assert c.hits == 0 and c.misses == 1 and three.result.__code__ is not two.result.__code__


@pytest.mark.parametrize('rehashed', [False, True])
def test_altered_cached_program_rejects_before_execution(tmp_path,rehashed):
    with synthetic(tmp_path) as (module, _, pool, _):
        module.construct()
        key = next(iter(pool))
        code, expected = pool[key]
        changed=compile("SYNTHETIC_UNAPPROVED = True", code.co_filename, 'exec')
        if rehashed:
            import marshal
            expected=hashlib.sha256(marshal.dumps(changed,2)).hexdigest()
        pool[key] = (changed, expected)
        with pytest.raises(ValueError, match='HISTORICAL_PROGRAM_CODE_CHANGED'):
            module.construct()


def test_inactive_owner_and_changed_compiler_cannot_borrow_cached_program(tmp_path, monkeypatch):
    with synthetic(tmp_path) as (module, adapter, _, active):
        module.construct()
        active[0] = False
        assert module.construct().result() == [] and adapter.hits == 0
        active[0] = True
        import builtins
        def changed(*args, **kwargs):
            raise ValueError('SYNTHETIC compiler patch')
        # Native exec strings do not call a monkeypatched Python compile name.
        # The adapter must nevertheless decline its cache while it is changed.
        with monkeypatch.context() as patch:
            patch.setattr(builtins, 'compile', changed)
            assert module.construct().result() == [] and adapter.hits == 0


def test_future_flags_are_inherited_from_the_original_native_caller(tmp_path):
    body = 'def result(value: SYNTHETIC_UNDEFINED): pass\n'
    with synthetic(tmp_path, body=body, future=True) as (module, adapter, _, _):
        first = module.construct(); second = module.construct()
        assert first.result.__annotations__ == second.result.__annotations__ == {'value': 'SYNTHETIC_UNDEFINED'}
        assert adapter.hits == 1


@pytest.mark.parametrize('failure', [None, ValueError, KeyboardInterrupt])
def test_actual_bootstrap_reuses_only_code_and_restores_every_namespace(failure):
    from tests import historical_scope_input_boundary as boundary
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    request = SimpleNamespace(module=SimpleNamespace(__file__=str(boundary.ROOT / 'tests/test_football_document_io_scope.py')),
        node=SimpleNamespace(originalname='test_exact_policy_and_ordered_ci_merge'))
    plan = boundary.resolve_context(request)
    boundary.clear_program_code_pool()
    first = second = None
    def operation():
        nonlocal first, second
        with boundary.InputOwner(plan):
            first, _ = _new_released_scope_inputs(boundary.canonical_guard())
            parent = first._ncaaf_pilot_seeded_custody_parent
            parent.ROOT = Path('SYNTHETIC first-only root')
            second, _ = _new_released_scope_inputs(boundary.canonical_guard())
            other = second._ncaaf_pilot_seeded_custody_parent
            assert parent is not other and vars(parent) is not vars(other)
            assert other.ROOT != parent.ROOT
            assert parent.run is not other.run and parent.run.__code__ is other.run.__code__
            assert boundary._PROGRAM_CODE_POOL
            assert boundary._bootstrap_immutable_size(tuple(boundary._PROGRAM_CODE_POOL.items())) <= boundary._BOOTSTRAP_LIMIT
            if failure is not None:
                raise failure('SYNTHETIC program owner exit')
    if failure is None:
        operation()
    else:
        with pytest.raises(failure, match='SYNTHETIC program owner exit'):
            operation()
    for module in (first, second, first._ncaaf_pilot_seeded_custody_parent, second._ncaaf_pilot_seeded_custody_parent):
        assert 'compile' not in vars(module) and 'exec' not in vars(module)
    boundary.clear_program_code_pool()
    assert not boundary._PROGRAM_CODE_POOL


def test_program_pool_clears_when_pytest_configuration_cleans_up():
    from tests import historical_scope_input_boundary as boundary
    callbacks = []
    config = SimpleNamespace(add_cleanup=callbacks.append)
    boundary.capture_generated_parent(config)
    assert callbacks == [boundary.clear_program_code_pool]
    boundary._PROGRAM_CODE_POOL[('SYNTHETIC cleanup marker',)] = (b'SYNTHETIC immutable data', 'SYNTHETIC digest')
    callbacks[0]()
    assert not boundary._PROGRAM_CODE_POOL
