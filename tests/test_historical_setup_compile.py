"""Synthetic code reuse, fresh authentication and fixture ownership boundaries."""
import builtins
import io
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests import historical_setup_compile as reuse


def synthetic(tmp_path, body=b"VALUE = 'SYNTHETIC original'\n", future=False):
    path = tmp_path / 'scripts/ncaaf_filtered_history_scope.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    module = ModuleType('SYNTHETIC_fresh_helper_loader')
    module.__file__ = str(tmp_path / 'SYNTHETIC_loader.py')
    module.path = path
    text = ("from __future__ import annotations\n" if future else '') + '''def load(source=None):
    source = path.read_bytes() if source is None else source
    code = compile(source, str(path), 'exec')
    result = {}
    exec(code, result)
    return result, code
'''
    exec(compile(text, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    compiler = reuse.OwnedCompilation(tmp_path, [module.load.__code__])
    module.compile = compiler
    return module, compiler, path


def test_cold_warm_code_reuse_keeps_fresh_mutable_namespaces(tmp_path):
    module, compiler, _ = synthetic(tmp_path)
    first, code = module.load()
    first['VALUE'] = 'SYNTHETIC first-only mutation'
    second, again = module.load()
    assert first is not second and code is again
    assert second['VALUE'] == 'SYNTHETIC original'
    assert compiler.misses == 1 and compiler.hits == 1
    assert compiler.retained_bytes <= reuse.MAX_RETAINED


def test_changed_complete_source_invalidates_code_and_physical_mismatch_cannot_hit(tmp_path):
    module, compiler, path = synthetic(tmp_path)
    before = path.read_bytes()
    first, _ = module.load()
    path.write_bytes(b"VALUE = 'SYNTHETIC changed'\n")
    second, _ = module.load()
    assert second['VALUE'] != first['VALUE'] and compiler.misses == 2
    hits = compiler.hits
    stale, _ = module.load(before)
    assert stale['VALUE'] == first['VALUE'] and compiler.hits == hits


def test_changed_compiler_is_called_instead_of_reusing_code(tmp_path, monkeypatch):
    module, compiler, _ = synthetic(tmp_path)
    module.load()
    def reject(*args, **kwargs):
        raise ValueError('SYNTHETIC_COMPILER_CHANGED')
    monkeypatch.setattr(builtins, 'compile', reject)
    with pytest.raises(ValueError, match='SYNTHETIC_COMPILER_CHANGED'):
        module.load()
    assert compiler.hits == 0


def test_mutated_cached_code_rejects(tmp_path):
    module, compiler, _ = synthetic(tmp_path)
    module.load()
    key = next(iter(compiler.entries))
    code, expected, cost = compiler.entries[key]
    compiler.entries[key] = (compile("VALUE='SYNTHETIC unapproved'", code.co_filename, 'exec'), expected, cost)
    with pytest.raises(ValueError, match='HISTORICAL_COMPILED_HELPER_CHANGED'):
        module.load()


def test_future_annotations_match_native_caller_semantics(tmp_path):
    module, compiler, _ = synthetic(tmp_path, b'def f(value: SYNTHETIC_UNDEFINED): pass\n', future=True)
    first, _ = module.load()
    second, _ = module.load()
    assert first['f'].__annotations__ == second['f'].__annotations__ == {'value': 'SYNTHETIC_UNDEFINED'}
    assert compiler.hits == 1


def test_undeclared_caller_and_nondefault_compile_modes_do_not_borrow_code(tmp_path):
    module, compiler, path = synthetic(tmp_path)
    module.load()
    assert isinstance(compiler(path.read_bytes(), str(path), 'exec'), type(compile('', '<SYNTHETIC>', 'exec')))
    assert compiler.hits == 0
    tree = compiler(path.read_bytes(), str(path), 'exec', flags=1024)
    import ast
    assert isinstance(tree, ast.Module) and compiler.hits == 0


def test_unrelated_caller_code_constants_are_not_hashed(tmp_path):
    _, compiler, path = synthetic(tmp_path)
    def unrelated():
        return compiler(b"VALUE='SYNTHETIC uncached'", str(path), 'exec')
    # Native compile does not inspect or hash the calling code's constants.
    unrelated.__code__ = unrelated.__code__.replace(co_consts=unrelated.__code__.co_consts + ([],))
    code = unrelated()
    assert isinstance(code, type(compile('', '<SYNTHETIC>', 'exec')))
    assert compiler.hits == compiler.misses == 0


def test_errors_and_oversized_sources_are_not_retained(tmp_path, monkeypatch):
    module, compiler, path = synthetic(tmp_path, b'def SYNTHETIC_broken(\n')
    for _ in range(2):
        with pytest.raises(SyntaxError):
            module.load()
    assert compiler.misses == 2 and not compiler.entries
    path.write_bytes(b"VALUE='SYNTHETIC correct'\n")
    monkeypatch.setattr(reuse, 'MAX_SOURCE', 1)
    assert module.load()[0]['VALUE'] == 'SYNTHETIC correct' and not compiler.entries


def test_independent_fixture_caches_do_not_share_code_or_state(tmp_path):
    one, first, _ = synthetic(tmp_path / 'one')
    two, second, _ = synthetic(tmp_path / 'two')
    a, acode = one.load()
    b, bcode = two.load()
    assert first is not second and acode is not bcode and first.hits == second.hits == 0
    a['VALUE'] = 'SYNTHETIC contamination'
    assert two.load()[0]['VALUE'] == b['VALUE'] == 'SYNTHETIC original'
    first.clear()
    assert first.closed and not first.entries and first.retained_bytes == 0
    assert one.load()[0]['VALUE'] == 'SYNTHETIC original' and first.hits == 0


def owner_plan(boundary):
    request = SimpleNamespace(module=SimpleNamespace(__file__=str(boundary.ROOT / 'tests/test_ncaaf_coverage_scope_policy.py')),
        node=SimpleNamespace(originalname='test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    return boundary.resolve_context(request)


@pytest.mark.parametrize('failure', [None, ValueError, KeyboardInterrupt])
def test_native_loads_reauthenticate_and_cleanup_on_every_exit(failure):
    from tests import historical_scope_input_boundary as boundary
    live = boundary.canonical_guard()
    original = vars(live).get('compile', boundary._ABSENT)
    compiler = None
    def operation():
        nonlocal compiler
        plan = owner_plan(boundary)
        with boundary.InputOwner(plan) as owner:
            compiler = boundary.install_owned_helper_compilation(plan, owner.patch, live)
            first, binding = live._ci_fixture_module()
            second, again = live._ci_fixture_module()
            assert first is not second and first.__dict__ is not second.__dict__
            assert binding == again and compiler.hits >= 2
            if failure is not None:
                raise failure('SYNTHETIC owner exit')
    if failure is None:
        operation()
    else:
        with pytest.raises(failure, match='SYNTHETIC owner exit'):
            operation()
    assert vars(live).get('compile', boundary._ABSENT) is original
    assert compiler.closed and not compiler.entries and compiler.retained_bytes == 0


def test_warm_native_authentication_rejects_altered_source_before_code_reuse(monkeypatch):
    from tests import historical_scope_input_boundary as boundary
    live = boundary.canonical_guard()
    plan = owner_plan(boundary)
    with boundary.InputOwner(plan) as owner:
        compiler = boundary.install_owned_helper_compilation(plan, owner.patch, live)
        live._ci_fixture_module(); live._ci_fixture_module()
        hits = compiler.hits
        physical = io.FileIO
        target = boundary.ROOT / 'scripts/ncaaf_filtered_history_scope.py'
        raw = boundary.physical_bytes(target)
        def changed(path, *args, **kwargs):
            if Path(path) == target:
                return io.BytesIO(raw + b'\n# SYNTHETIC altered authenticated helper\n')
            return physical(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(io, 'FileIO', changed)
            with pytest.raises(ValueError, match='NCAAF_FILTERED_SOURCE_CHANGED'):
                live._ci_fixture_module()
        assert compiler.hits == hits


def test_nested_native_context_suspends_only_owned_compiler():
    from tests import historical_scope_input_boundary as boundary
    live = boundary.canonical_guard()
    original = vars(live).get('compile', boundary._ABSENT)
    plan = owner_plan(boundary)
    with boundary.InputOwner(plan) as owner:
        compiler = boundary.install_owned_helper_compilation(plan, owner.patch, live)
        with boundary.InputOwner(plan):
            assert vars(live).get('compile', boundary._ABSENT) is original
        assert live.compile is compiler
    assert vars(live).get('compile', boundary._ABSENT) is original


def test_unrelated_compiler_patch_and_unowned_use_remain_untouched(monkeypatch):
    from tests import historical_scope_input_boundary as boundary
    live = boundary.canonical_guard()
    plan = owner_plan(boundary)
    def unrelated(*args, **kwargs):
        raise RuntimeError('SYNTHETIC unrelated compiler')
    monkeypatch.setattr(live, 'compile', unrelated, raising=False)
    with boundary.InputOwner(plan) as owner:
        assert boundary.install_owned_helper_compilation(plan, owner.patch, live) is None
        assert live.compile is unrelated
    assert live.compile is unrelated
    with pytest.raises(ValueError, match='HISTORICAL_COMPILE_OWNER_REQUIRED'):
        boundary.install_owned_helper_compilation(plan, None, live)
