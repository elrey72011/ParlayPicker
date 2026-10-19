"""Exact released scope-test inputs, never live validation decisions."""
import hashlib,json,sys,types
from pathlib import Path
import pytest

@pytest.fixture(scope='session')
def ncaaf_released_scope_inputs():
    from scripts import check_launch_change_scope as live
    root=Path(__file__).resolve().parents[1]
    source=live._ncaaf_future_base_source((root/live.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n'))
    frozen=types.ModuleType('exact_ncaaf_future_released_scope');frozen.__file__=str(root/live.GUARD_PATH)
    sys.modules[frozen.__name__]=frozen
    # Import-time predecessor reconstruction must see that same exact source,
    # rather than reopening the successor through the unchanged __file__ path.
    from unittest.mock import patch
    read=Path.read_bytes
    with patch.object(Path,'read_bytes',lambda path:source if path.resolve()==(root/live.GUARD_PATH).resolve() else read(path)):
        exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)
    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)

def historical_test_consumers(module,live):
    # Follow imported test helpers only. Every alias sharing the canonical guard
    # receives the same frozen object, so a tampering monkeypatch reaches the
    # function actually being exercised. Production modules are never traversed.
    pending=[module];seen=set();result=[]
    while pending:
        current=pending.pop()
        if id(current) in seen:continue
        seen.add(id(current));assert len(seen)<=128
        if current is not module and not current.__name__.split('.')[-1].startswith('test_'):continue
        result.extend((current,name) for name,value in vars(current).items() if value is live)
        pending.extend(value for value in vars(current).values() if isinstance(value,types.ModuleType) and value.__name__.split('.')[-1].startswith('test_'))
    return result

@pytest.fixture(autouse=True)
def exact_released_scope_contract_inputs(request,monkeypatch):
    # The current successor gate must inspect live inputs, never historical views.
    if (request.module.__name__.rsplit('.',1)[-1]=='test_ci_fixture_stability_scope'
        and Path(request.module.__file__).resolve()==Path(__file__).resolve().with_name('test_ci_fixture_stability_scope.py')):
        from tests.ci_fixture_stability_inputs import install_exact_predecessor_sources
        install_exact_predecessor_sources(request,monkeypatch)
        return
    from tests.ci_fixture_stability_inputs import install_exact_predecessor_sources
    install_exact_predecessor_sources(request,monkeypatch)
    from tests.ncaaf_staged_capacity_fixtures import install_reviewed_target_profile_inputs
    install_reviewed_target_profile_inputs(request,monkeypatch)
    from ncaaf_transport_scope_inputs import install
    install(request,monkeypatch)
    module=request.module
    # Explicit old source-policy consumers only; runtime/feature/current-scope
    # tests and real CLI/subprocess checks never use this fixture projection.
    names=[name for name,value in vars(module).items() if getattr(value,'__name__',None) in
        {'scripts.check_launch_change_scope','exact_ncaaf_transport_predecessor_scope'}]
    if not names or module.__name__=='test_ncaaf_future_spread_scope':
        return
    from scripts import check_launch_change_scope as live
    names=[name for name in names if getattr(module,name) is live]
    if not names:return
    frozen,records=request.getfixturevalue('ncaaf_released_scope_inputs');root=Path(__file__).resolve().parents[1]
    read=Path.read_bytes
    def historical(path):
        value=read(path)
        try:relative=path.resolve().relative_to(root).as_posix()
        except ValueError:return value
        normalized=value.replace(b'\r\n',b'\n')
        if relative==live.GUARD_PATH:return live._ncaaf_future_base_source(normalized)
        record=records.get(relative)
        if record is None or hashlib.sha256(normalized).hexdigest()!=record['after_sha256']:return value
        return live._ncaaf_future_previous_source(relative,normalized)
    for consumer,name in historical_test_consumers(module,live):monkeypatch.setattr(consumer,name,frozen)
    monkeypatch.setattr(Path,'read_bytes',historical)

class _HistoricalSourceSetup:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self,fixturedef,request):
        # A non-generator session template is built before function autouse
        # fixtures. Supply the same exact historical inputs during construction,
        # then restore immediately. Live modules and generator teardown stay alone.
        import inspect
        if request.config.getoption("setupplan",False):
            yield
            return
        module=sys.modules.get(fixturedef.func.__module__)
        exact_module_predecessor = (
            fixturedef.scope == 'module' and fixturedef.argname == 'predecessor'
            and module is not None
            and module.__name__.split('.')[-1] == 'test_football_document_ci_performance'
            and Path(module.__file__).resolve() ==
                Path(__file__).resolve().with_name('test_football_document_ci_performance.py'))
        if ((fixturedef.scope!='session' and not exact_module_predecessor)
            or inspect.isgeneratorfunction(fixturedef.func)
            or module is None or not module.__name__.split('.')[-1].startswith('test_')
            or module.__name__.split('.')[-1]=='test_ncaaf_future_spread_scope'):
            yield
            return
        from scripts import check_launch_change_scope as live
        if not any(value is live for value in vars(module).values()):
            yield
            return
        with pytest.MonkeyPatch.context() as patch:
            proxy=types.SimpleNamespace(module=module,getfixturevalue=request.getfixturevalue)
            exact_released_scope_contract_inputs.__wrapped__(proxy,patch)
            yield

def pytest_configure(config):
    # Session fixture hooks use the session node proxy, which excludes hooks
    # scoped only to tests/conftest.py. Register the narrowly filtered hook as a
    # global test plugin; retain the existing local fixture definitions once.
    name='ncaaf-exact-historical-setup-v1'
    if not config.pluginmanager.hasplugin(name):
        config.pluginmanager.register(_HistoricalSourceSetup(),name)
