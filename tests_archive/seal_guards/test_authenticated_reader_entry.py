"""Synthetic reader/caller regressions; no operational data or acceptance."""
import cProfile
import io
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import check_launch_change_scope as live
from tests import historical_scope_input_boundary as boundary

RELATIVE='scripts/ci_test_file_costs_v1.json'

def source():return boundary.physical_bytes(boundary.ROOT/RELATIVE)

def auth_count(operation):
    profiler=cProfile.Profile();profiler.enable()
    try:result=operation()
    finally:profiler.disable()
    count=sum(r.callcount for r in profiler.getstats()if not isinstance(r.code,str)
        and r.code.co_name=='authenticate'and r.code.co_filename.endswith('ncaaf_filtered_history_scope.py'))
    return result,count

def test_one_reader_entry_keeps_one_fresh_native_authentication():
    raw=source()
    expected,count=auth_count(lambda:live._filtered_history_module().predecessor_source(boundary.ROOT,RELATIVE,raw))
    assert count==2
    actual,count=auth_count(lambda:boundary._filtered_source_parent(RELATIVE,raw))
    assert count==1 and actual==expected
    again,count=auth_count(lambda:boundary._filtered_source_parent(RELATIVE,raw))
    assert count==1 and again==actual

def test_each_direct_read_physically_checks_the_complete_current_graph(monkeypatch):
    module=live._filtered_history_module();binding=module.authenticate(boundary.ROOT)
    expected={boundary.ROOT/p for p in binding['components']}|{boundary.ROOT/module.BINDING}
    native=io.FileIO;seen=set()
    def physical(path,*args,**kwargs):
        seen.add(Path(path));return native(path,*args,**kwargs)
    raw=source()
    with monkeypatch.context()as patch:
        patch.setattr(io,'FileIO',physical)
        for _ in range(2):
            seen.clear();live._filtered_history_module(read=(RELATIVE,raw))
            assert expected<=seen

def altered_reader(native,target):
    class Altered:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def fileno(self):return self.stream.fileno()
        def read(self,*args):
            raw=self.stream.read(*args)
            return bytes([raw[0]^1])+raw[1:]if raw else raw
    def physical(path,*args,**kwargs):
        stream=native(path,*args,**kwargs)
        return Altered(stream)if Path(path)==target else stream
    return physical

@pytest.mark.parametrize('relative',[
    'docs/research/ncaaf-filtered-history-bindings-v1.json',
    'scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py'])
@pytest.mark.parametrize('warm',[False,True])
def test_direct_entry_rejects_changed_current_inputs(monkeypatch,relative,warm):
    assert relative in live._filtered_history_module().authenticate(boundary.ROOT)['components'] or relative.endswith('bindings-v1.json')
    raw=source()
    if warm:live._filtered_history_module(read=(RELATIVE,raw))
    with monkeypatch.context()as patch:
        patch.setattr(io,'FileIO',altered_reader(io.FileIO,boundary.ROOT/relative))
        with pytest.raises(ValueError,match='NCAAF_FILTERED_(BINDING|SOURCE)_CHANGED'):
            live._filtered_history_module(read=(RELATIVE,raw))

def test_standalone_factory_still_rejects_changed_component(monkeypatch):
    with monkeypatch.context()as patch:
        patch.setattr(io,'FileIO',altered_reader(io.FileIO,boundary.ROOT/'scripts/ncaaf_private_refresh_scope.py'))
        with pytest.raises(ValueError,match='NCAAF_FILTERED_SOURCE_CHANGED'):
            live._filtered_history_module()

def test_later_public_reader_reauthenticates_after_factory_return(monkeypatch):
    raw=source();module=live._filtered_history_module()
    with monkeypatch.context()as patch:
        patch.setattr(io,'FileIO',altered_reader(io.FileIO,boundary.ROOT/'scripts/ncaaf_private_refresh_scope.py'))
        with pytest.raises(ValueError,match='NCAAF_FILTERED_SOURCE_CHANGED'):
            module.predecessor_source(boundary.ROOT,RELATIVE,raw)

@pytest.mark.parametrize('value',[(),('only-one',),[RELATIVE,b'not-a-tuple']])
def test_direct_entry_rejects_malformed_operation(value):
    with pytest.raises(ValueError,match='NCAAF_FILTERED_READER_ARGUMENT_CHANGED'):
        live._filtered_history_module(read=value)

def test_real_loader_code_remains_the_compiler_caller():
    request=SimpleNamespace(module=SimpleNamespace(__file__=str(boundary.ROOT/'tests/test_ncaaf_coverage_scope_policy.py')),
        node=SimpleNamespace(originalname='test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    plan=boundary.resolve_context(request);original=vars(live).get('compile',boundary._ABSENT)
    with boundary.InputOwner(plan)as owner:
        compiler=boundary.install_owned_helper_compilation(plan,owner.patch,live)
        raw=source();first=live._filtered_history_module(read=(RELATIVE,raw))
        second=live._filtered_history_module(read=(RELATIVE,raw))
        assert first==second and compiler.hits>=1
        assert live._filtered_history_module.__code__ in compiler.callers
    assert vars(live).get('compile',boundary._ABSENT)is original
    assert compiler.closed and not compiler.entries

def test_normalization_and_fresh_namespace_isolation(tmp_path):
    one=live._filtered_history_module();two=live._filtered_history_module()
    assert one is not two and one.__dict__ is not two.__dict__
    one.SYNTHETIC_MUTATION='isolated';assert not hasattr(two,'SYNTHETIC_MUTATION')
    path=tmp_path/'SYNTHETIC-input.py';path.write_bytes(b'one\r\ntwo\n')
    assert one.physical(path)==two.physical(path)==b'one\ntwo\n'
