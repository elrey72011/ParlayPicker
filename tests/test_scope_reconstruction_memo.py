"""Labelled synthetic immutable-source memo boundaries; no external services."""
import hashlib
import types
import pytest
from scripts import scope_reconstruction as h


def synthetic(path, body=b'SYNTHETIC source\n'):
    g=types.ModuleType('SYNTHETIC_reconstruction');g.__file__=str(path)
    g.REVIEWED_SHA256=hashlib.sha256(body).hexdigest()
    exec('''import hashlib
def verify(source):
    if hashlib.sha256(source).hexdigest()!=REVIEWED_SHA256:
        raise ValueError('SYNTHETIC_SOURCE_CHANGED')
def _nfl_odds_timing_base_source(source):
    verify(source)
    return source.replace(b'SYNTHETIC',b'predecessor')
''',g.__dict__)
    return g,body


def test_exact_bytes_context_and_source_change(tmp_path):
    g,body=synthetic(tmp_path/'one.py');h.install(g)
    method=g._nfl_odds_timing_base_source;memo=method.reconstruction_memo
    assert method(body)==method(body)==b'predecessor source\n'
    assert memo.hits==1 and memo.misses==1
    with pytest.raises(ValueError,match='SYNTHETIC_SOURCE_CHANGED'):method(body+b'altered')
    with pytest.raises(ValueError,match='SYNTHETIC_SOURCE_CHANGED'):method(body+b'altered')
    assert memo.misses==3
    g.REVIEWED_SHA256='0'*64
    with pytest.raises(ValueError,match='SYNTHETIC_SOURCE_CHANGED'):method(body)


def test_measured_capacity_leaf_reuse_preserves_warm_rejection(tmp_path):
    g,body=synthetic(tmp_path/'capacity.py')
    g._listing_capacity_base_source=g._nfl_odds_timing_base_source
    del g._nfl_odds_timing_base_source
    from tests.measured_scope_reconstruction import install
    install(g)
    method=g._listing_capacity_base_source
    assert method(body)==method(body)==b'predecessor source\n'
    assert method.reconstruction_memo.hits==1
    with pytest.raises(ValueError,match='SYNTHETIC_SOURCE_CHANGED'):method(body+b'changed')
    g.REVIEWED_SHA256='0'*64
    with pytest.raises(ValueError,match='SYNTHETIC_SOURCE_CHANGED'):method(body)


def test_measured_leaf_never_caches_live_repository_decisions(tmp_path):
    g=types.ModuleType('SYNTHETIC_capacity_live');g.__file__=str(tmp_path/'guard.py');g.ROOT=tmp_path
    exec('def _listing_capacity_base_source(source):\n    return (ROOT/"live").read_bytes()\n',g.__dict__)
    from tests.measured_scope_reconstruction import install
    install(g)
    assert not hasattr(g._listing_capacity_base_source,'reconstruction_memo')
    (tmp_path/'live').write_bytes(b'first')
    assert g._listing_capacity_base_source(b'input')==b'first'
    (tmp_path/'live').write_bytes(b'second')
    assert g._listing_capacity_base_source(b'input')==b'second'


@pytest.mark.parametrize('failure',[None,ValueError,KeyboardInterrupt])
def test_fixture_memos_are_released_on_success_failure_and_interruption(tmp_path,failure):
    from types import SimpleNamespace
    from tests import historical_scope_input_boundary as boundary
    g,body=synthetic(tmp_path/'owned.py');h.install(g)
    memo=g._nfl_odds_timing_base_source.reconstruction_memo
    request=SimpleNamespace(module=SimpleNamespace(__file__=str(boundary.ROOT/'tests/test_ncaaf_coverage_scope_policy.py')),
        node=SimpleNamespace(originalname='test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    plan=boundary.resolve_context(request)
    def run():
        with boundary.InputOwner(plan):
            boundary.own_reconstruction_memos(g)
            assert g._nfl_odds_timing_base_source(body)==b'predecessor source\n'
            assert memo.entries and memo.retained_bytes
            if failure is not None:raise failure('SYNTHETIC_INTERRUPTION')
    if failure is None:run()
    else:
        with pytest.raises(failure,match='SYNTHETIC_INTERRUPTION'):run()
    assert not memo.entries and memo.retained_bytes==0


def test_changed_delegate_does_not_borrow_cached_result(tmp_path):
    g,body=synthetic(tmp_path/'one.py');h.install(g)
    method=g._nfl_odds_timing_base_source;method(body)
    def reject(source): raise ValueError('SYNTHETIC_DELEGATE_CHANGED')
    g.verify=reject
    with pytest.raises(ValueError,match='SYNTHETIC_DELEGATE_CHANGED'):method(body)


def test_cross_repository_and_module_contexts_are_distinct(tmp_path):
    one,body=synthetic(tmp_path/'one/guard.py');two,_=synthetic(tmp_path/'two/guard.py')
    h.install(one);h.install(two)
    first=one._nfl_odds_timing_base_source;second=two._nfl_odds_timing_base_source
    assert first(body)==second(body)
    assert first.reconstruction_memo is not second.reconstruction_memo
    assert second.reconstruction_memo.hits==0
    one.__file__=str(tmp_path/'other/guard.py')
    first(body);assert first.reconstruction_memo.misses==2


def test_delegate_code_and_closure_mutation_cannot_borrow_result(tmp_path):
    g,body=synthetic(tmp_path/'code.py');h.install(g);fn=g._nfl_odds_timing_base_source;fn(body)
    def reject(source):raise ValueError('SYNTHETIC_CODE_CHANGED')
    g.verify.__code__=reject.__code__
    with pytest.raises(ValueError,match='SYNTHETIC_CODE_CHANGED'):fn(body)
    g,body=synthetic(tmp_path/'closure.py')
    def make(expected):
        def verify(source):
            if source!=expected:raise ValueError('SYNTHETIC_CLOSURE_CHANGED')
        return verify
    g.verify=make(body);h.install(g);fn=g._nfl_odds_timing_base_source;fn(body)
    g.verify.__closure__[0].cell_contents=b'changed'
    with pytest.raises(ValueError,match='SYNTHETIC_CLOSURE_CHANGED'):fn(body)


def test_mutable_reconstruction_declarations_are_snapshotted(tmp_path):
    g=types.ModuleType('SYNTHETIC_declarations');g.__file__=str(tmp_path/'guard.py')
    exec('''DECLARATIONS={'expected':b'SYNTHETIC'}
def _nfl_odds_timing_base_source(source):
    if source!=DECLARATIONS['expected']:raise ValueError('SYNTHETIC_DECLARATION_CHANGED')
    return source
''',g.__dict__)
    h.install(g);fn=g._nfl_odds_timing_base_source
    assert fn(b'SYNTHETIC')==fn(b'SYNTHETIC')
    g.DECLARATIONS['expected']=b'changed'
    with pytest.raises(ValueError,match='SYNTHETIC_DECLARATION_CHANGED'):fn(b'SYNTHETIC')


def test_io_or_mutable_decisions_cannot_be_memoized(tmp_path):
    g=types.ModuleType('SYNTHETIC_live');g.__file__=str(tmp_path/'guard.py');g.ROOT=tmp_path
    exec('def _nfl_odds_timing_base_source(source):\n    return (ROOT/"live").read_bytes()\n',g.__dict__)
    h.install(g)
    assert not hasattr(g._nfl_odds_timing_base_source,'reconstruction_memo')
    (tmp_path/'live').write_bytes(b'first')
    assert g._nfl_odds_timing_base_source(b'irrelevant')==b'first'
    (tmp_path/'live').write_bytes(b'second')
    assert g._nfl_odds_timing_base_source(b'irrelevant')==b'second'


def test_entry_and_byte_caps_unknown_types_and_errors(monkeypatch):
    monkeypatch.setattr(h,'MAX_SOURCE_BYTES',16);monkeypatch.setattr(h,'MAX_RETAINED_BYTES',32)
    monkeypatch.setattr(h,'MAX_ENTRIES',2)
    memo=h.ReconstructionMemo();calls=[]
    def method(source):calls.append(source);return source
    for source in (b'a'*8,b'b'*8,b'c'*8):memo.call(method,source,())
    assert len(memo.entries)==2 and memo.retained_bytes==32
    assert memo.call(method,b'a'*17,())==b'a'*17
    assert memo.call(method,b'a'*17,())==b'a'*17
    assert calls[-2:]==[b'a'*17,b'a'*17] and memo.retained_bytes==32
    assert memo.call(method,'UNKNOWN',())=='UNKNOWN' and len(memo.entries)==2
    def reject(source):raise ValueError('SYNTHETIC_REJECT')
    for _ in range(2):
        with pytest.raises(ValueError,match='SYNTHETIC_REJECT'):memo.call(reject,b'a',())
    assert all(key[0] is not reject for key in memo.entries)


def test_builtin_and_captured_io_are_not_pure(tmp_path):
    path=tmp_path/'live';path.write_bytes(b'first')
    g=types.ModuleType('SYNTHETIC_builtin_io');g.__file__=str(tmp_path/'guard.py');g.LOCATION=str(path)
    exec('def _nfl_odds_timing_base_source(source):\n    with open(LOCATION,"rb") as f:return f.read()\n',g.__dict__)
    h.install(g)
    assert not hasattr(g._nfl_odds_timing_base_source,'reconstruction_memo')
    assert g._nfl_odds_timing_base_source(b'unchanged')==b'first'
    path.write_bytes(b'second')
    assert g._nfl_odds_timing_base_source(b'unchanged')==b'second'
    def make(op):
        def captured(source):return op(path,'rb').read()
        return captured
    assert h._context_fields(make(open)) is None


def test_cyclic_or_oversized_context_falls_back():
    cyclic={};cyclic['self']=cyclic
    with pytest.raises(ValueError,match='CONTEXT_LIMIT'):h._freeze(cyclic)
    with pytest.raises(ValueError,match='CONTEXT_LIMIT'):h._freeze({'huge':'x'*(256*1024+1)})


def test_actual_reviewed_reconstruction_rejects_changed_source():
    from pathlib import Path
    from scripts import check_launch_change_scope as g
    runtime=g._scope_reconstruction_for_validation()
    source=Path(g.__file__).read_bytes().replace(b'\r\n',b'\n')
    fn=g._nfl_odds_timing_base_source;before=fn(source)
    assert fn(source)==before and fn.reconstruction_memo.hits>0
    with pytest.raises(ValueError,match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
        fn(source+b'\nSYNTHETIC_UNREVIEWED=True\n')
    assert runtime is not None
