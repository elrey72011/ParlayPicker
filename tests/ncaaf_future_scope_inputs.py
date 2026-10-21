"""Exact historical inputs selected before construction, with owned cleanup."""
import hashlib,json,sys,types
from pathlib import Path
import pytest


def _new_released_scope_inputs(live):
    from tests.historical_scope_input_boundary import physical_bytes, selected_reader_generation, historical_reader_source, authenticated_predecessor
    root=Path(__file__).resolve().parents[1]
    scope,binding=live._ci_fixture_module()
    reader_generation=selected_reader_generation(binding)
    if reader_generation=='released-future-v1':
        current=authenticated_predecessor(scope,binding)
        source=live._ncaaf_future_base_source(current)
        reviewed=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'))
        assert hashlib.sha256(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json')).hexdigest()==binding['historical_input_boundary']['source_generations']['docs/paid-launch/ncaaf-reviewed-target-binding-v1.json']
        assert hashlib.sha256(source).hexdigest()==reviewed['historical_future']['prior_base_sha256']
    else:
        source=historical_reader_source(live,binding,reader_generation)
    frozen=types.ModuleType('exact_ncaaf_future_released_scope');frozen.__file__=str(root/live.GUARD_PATH)
    from unittest.mock import patch
    read=Path.read_bytes
    with patch.dict(sys.modules,{frozen.__name__:frozen}), patch.object(Path,'read_bytes',lambda path:source if path.resolve()==(root/live.GUARD_PATH).resolve() else read(path)):
        if reader_generation=='released-future-v1':
            from tests.historical_scope_input_boundary import execute_released_bootstrap
            execute_released_bootstrap(source,frozen,live,binding)
            frozen._scope_reconstruction_for_validation()
        else:
            exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)
    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)

@pytest.fixture
def ncaaf_released_scope_inputs():
    # This compatibility object belongs to the authenticated requesting context.
    # A session-cached namespace would disagree with the per-context module view.
    from tests.historical_scope_input_boundary import canonical_guard, require_historical_fixture
    frozen = require_historical_fixture('ncaaf_released_scope_inputs')
    return frozen, json.loads(canonical_guard().NCAAF_FUTURE_RECONSTRUCTION_JSON)


def historical_test_consumers(module,live):
    pending=[module];seen=set();result=[]
    while pending:
        current=pending.pop()
        if id(current) in seen:continue
        seen.add(id(current));assert len(seen)<=128
        if current is not module and not current.__name__.split('.')[-1].startswith('test_'):continue
        result.extend((current,name) for name,value in vars(current).items() if value is live)
        pending.extend(value for value in vars(current).values() if isinstance(value,types.ModuleType) and value.__name__.split('.')[-1].startswith('test_'))
    return result


def _install_selected_future_inputs(request,patch,live,plan):
    # The authenticated generation decides before alias discovery or loading.
    # A consumer importing the guard later has the same namespace contract.
    if plan.generation not in {'future','transport-future'}:
        return
    # Selection is the authenticated physical contract, not alias identity or
    # whichever generation happened to populate a session fixture first.
    aliases=[];pending=[request.module];seen=set()
    target=Path(__file__).resolve().parents[1]/live.GUARD_PATH
    while pending:
        current=pending.pop()
        if id(current) in seen:continue
        seen.add(id(current));assert len(seen)<=128
        for name,value in tuple(vars(current).items()):
            if isinstance(value,types.ModuleType):
                file=getattr(value,'__file__',None)
                if file and Path(file).resolve()==target.resolve():
                    aliases.append((current,name))
                elif value.__name__.split('.')[-1].startswith('test_'):
                    pending.append(value)
    frozen,records=_new_released_scope_inputs(live)
    root=Path(__file__).resolve().parents[1];read=Path.read_bytes
    from tests.historical_scope_input_boundary import physical_bytes,generation_input
    scope,binding=live._ci_fixture_module()
    source_inputs=binding['historical_input_boundary']['consumers'][plan.consumer].get('source_inputs',{})
    assert len(source_inputs)<=64 and all(isinstance(p,str) and isinstance(h,str) and len(h)==64 for p,h in source_inputs.items())
    physical_guard=physical_bytes(root/live.GUARD_PATH).replace(b'\r\n',b'\n')
    from tests.historical_scope_input_boundary import selected_reader_generation, historical_reader_source, authenticated_predecessor
    reader_generation=selected_reader_generation(binding,plan)
    if reader_generation=='released-future-v1':
        main_guard=authenticated_predecessor(scope,binding)
        guard_source=live._ncaaf_future_base_source(main_guard)
        guard_inputs=(physical_guard,main_guard,guard_source)
    else:
        guard_source=historical_reader_source(live,binding,reader_generation)
        guard_inputs=(physical_guard,guard_source)
    allowed_guard_inputs={hashlib.sha256(v).hexdigest() for v in guard_inputs}
    from tests.historical_scope_input_boundary import _BOOTSTRAP_SOURCE
    allowed_guard_inputs.add(_BOOTSTRAP_SOURCE)
    original=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-future-spread-binding-v1.json'))
    for data in (original.get('ci_fixture_copy',{}),original.get('transport_diagnostics',{})):
        if 'previous_guard_sha256' in data:allowed_guard_inputs.add(data['previous_guard_sha256'])
    def historical(path):
        try:relative=path.resolve().relative_to(root).as_posix()
        except ValueError:return read(path)
        value=read(path)
        if relative==live.GUARD_PATH:
            return guard_source if hashlib.sha256(value.replace(b'\r\n',b'\n')).hexdigest() in allowed_guard_inputs else value
        exact=generation_input(plan,relative,value) if relative in source_inputs else None
        if exact is not None:return exact
        normalized=value.replace(b'\r\n',b'\n')
        record=records.get(relative)
        if record is None:return value
        return live._ncaaf_future_previous_source(relative,normalized)
    for consumer,name in aliases:patch.setattr(consumer,name,frozen)
    import scripts
    patch.setattr(scripts,'check_launch_change_scope',frozen)
    patch.setitem(sys.modules,'scripts.check_launch_change_scope',frozen)
    patch.setitem(sys.modules,frozen.__name__,frozen)
    # The shared boundary owns the final open/read_bytes agreement after these
    # exact transformations. Preserve unrelated incoming tampering here.
    patch.setattr(Path,'read_bytes',historical)


@pytest.fixture(autouse=True)
def exact_released_scope_contract_inputs(request,monkeypatch):
    from tests.historical_scope_input_boundary import resolve_context,InputOwner,apply_selected_inputs
    plan=resolve_context(request)
    with InputOwner(plan) as owner:
        apply_selected_inputs(plan,request,owner.patch)
        yield


class _HistoricalSourceSetup:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self,fixturedef,request):
        import inspect
        if request.config.getoption('setupplan',False):
            yield;return
        module=sys.modules.get(fixturedef.func.__module__)
        exact_module=(fixturedef.scope=='module' and fixturedef.argname=='predecessor'
            and module is not None and module.__name__.split('.')[-1]=='test_football_document_ci_performance'
            and Path(module.__file__).resolve()==Path(__file__).resolve().with_name('test_football_document_ci_performance.py'))
        if ((fixturedef.scope!='session' and not exact_module) or inspect.isgeneratorfunction(fixturedef.func)
            or module is None or not module.__name__.split('.')[-1].startswith('test_')
            or module.__name__.split('.')[-1]=='test_ncaaf_future_spread_scope'):
            yield;return
        from tests.historical_scope_input_boundary import resolve_context,InputOwner,apply_selected_inputs,Role
        proxy=types.SimpleNamespace(module=module,node=getattr(request,'node',None),function=fixturedef.func,
                                   getfixturevalue=request.getfixturevalue,config=request.config)
        item=getattr(request,'_pyfuncitem',None)
        proxy.fixture_consumer_path=getattr(item,'path',None)
        proxy.fixture_consumer_function=getattr(item,'obj',None)
        plan=resolve_context(proxy,phase=fixturedef.scope,fixture=fixturedef.argname)
        if plan.role is not Role.HISTORICAL:
            yield;return
        with InputOwner(plan) as owner:
            apply_selected_inputs(plan,proxy,owner.patch)
            yield


def pytest_configure(config):
    from tests.historical_scope_input_boundary import capture_generated_parent
    capture_generated_parent(config)
    name='ncaaf-exact-historical-setup-v1'
    if not config.pluginmanager.hasplugin(name):config.pluginmanager.register(_HistoricalSourceSetup(),name)
