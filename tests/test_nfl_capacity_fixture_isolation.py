"""Same-process fixture lifecycle and unchanged approved-report semantics."""
import ast
import builtins
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tests import nfl_capacity_profile_inputs as inputs
from tests.nfl_capacity_import_owner import MODULES

ROOT=Path(__file__).resolve().parents[1]
MISSING=object()


def snapshot():
    package=sys.modules['scripts']
    list(package.__path__)  # Resolve the incoming namespace before recording it.
    modules={name:sys.modules.get(name,MISSING) for name in MODULES}
    aliases={name:vars(package).get(name.split('.',1)[1],MISSING) for name in MODULES if name.startswith('scripts.')}
    attributes={name:dict(vars(module)) for name,module in modules.items() if module is not MISSING}
    caches={id(v):(v,dict(v)) for d in attributes.values() for k,v in d.items() if k.endswith('_IMAGES') and isinstance(v,dict)}
    return modules,aliases,attributes,caches,sys.path,list(sys.path)


def assert_restored(before):
    modules,aliases,attributes,caches,path,values=before
    assert sys.path is path and sys.path==values
    for name,module in modules.items():
        assert sys.modules.get(name,MISSING) is module
        if module is not MISSING:
            assert set(vars(module))==set(attributes[name])
            assert all(vars(module)[k] is v for k,v in attributes[name].items())
    for name,alias in aliases.items():assert vars(sys.modules['scripts']).get(name.split('.',1)[1],MISSING) is alias
    for cache,items in caches.values():assert cache==items


@pytest.mark.parametrize('preloaded',[False,True])
@pytest.mark.parametrize('phase',['normal','body_exception','setup_exception'])
def test_fixture_restores_transitive_imports_in_same_process(tmp_path,monkeypatch,preloaded,phase):
    package=importlib.import_module('scripts')
    for name in MODULES[:3]:
        if preloaded:importlib.import_module(name)
        else:
            monkeypatch.delitem(sys.modules,name,raising=False)
            monkeypatch.delattr(package,name.split('.',1)[1],raising=False)
    unrelated=SimpleNamespace(marker=object());monkeypatch.setitem(sys.modules,'SYNTHETIC_unrelated_capacity_module',unrelated)
    before=snapshot();operator=sys.modules.get(MODULES[2]);verifier=getattr(operator,'verify_checkout',None)
    request=SimpleNamespace(module=SimpleNamespace(__name__='test_ncaaf_filtered_history_scope',ROOT=ROOT,scope=object()),
        node=SimpleNamespace(name='test_legacy_fixture_adapter_is_exact_and_live_inputs_stay_current'),
        getfixturevalue=lambda name:tmp_path if name=='tmp_path' else pytest.fail('Unexpected eager fixture'))
    if phase=='setup_exception':
        def execute(code,namespace):
            builtins.exec(code,namespace)
            if namespace.get('__name__')=='exact_capacity_parent_guard':raise RuntimeError('SYNTHETIC_SETUP_EXCEPTION')
        monkeypatch.setattr(inputs,'exec',execute,raising=False)
    generator=inputs.historical_filtered_profile_generation.__wrapped__(request)
    if phase=='setup_exception':
        with pytest.raises(RuntimeError,match='^SYNTHETIC_SETUP_EXCEPTION$'):next(generator)
    else:
        next(generator)
        during=importlib.import_module(MODULES[2])
        if preloaded:assert during is operator and during.verify_checkout is verifier
        else:assert during.verify_checkout.__kwdefaults__['root']==tmp_path/'SYNTHETIC-authenticated-f6a7506-inputs'
        if phase=='body_exception':
            with pytest.raises(RuntimeError,match='^SYNTHETIC_BODY_EXCEPTION$'):generator.throw(RuntimeError('SYNTHETIC_BODY_EXCEPTION'))
        else:
            with pytest.raises(StopIteration):next(generator)
    assert_restored(before)
    assert sys.modules['SYNTHETIC_unrelated_capacity_module'] is unrelated
    live=importlib.import_module(MODULES[2])
    assert Path(live.__file__).resolve()==ROOT/'scripts/ncaaf_reviewed_target_operator.py'
    assert live.verify_checkout.__kwdefaults__['root']==ROOT
    if preloaded:assert live is operator and live.verify_checkout is verifier
    assert not any(str(tmp_path) in str(k) for k in sys.path_importer_cache)


@pytest.mark.parametrize('consumer,node',[('test_unrelated','test_unrelated'),
    ('test_ncaaf_filtered_history_scope','test_source_graph_and_exact_scope_are_bound')])
def test_inactive_fixture_performs_no_authentication_or_import_work(monkeypatch,consumer,node):
    def forbidden(*args,**kwargs):pytest.fail('Inactive predecessor fixture performed work')
    monkeypatch.setattr(inputs,'predecessor_imports',forbidden)
    request=SimpleNamespace(module=SimpleNamespace(__name__=consumer),node=SimpleNamespace(name=node),getfixturevalue=forbidden)
    generator=inputs.historical_filtered_profile_generation.__wrapped__(request)
    assert next(generator) is None
    with pytest.raises(StopIteration):next(generator)


def test_original_function_signatures_and_bodies_are_exact():
    from scripts import ncaaf_filtered_history_scope as history
    b=history.authenticate(ROOT);path='tests/test_ncaaf_filtered_history_scope.py'
    old=history.capacity_predecessor(b,path,history.physical(ROOT/path))
    functions=lambda raw:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    assert functions(old)==functions(history.physical(ROOT/path))
