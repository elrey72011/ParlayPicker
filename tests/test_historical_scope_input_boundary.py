"""Synthetic authenticated boundary regressions; no provider originals or calls."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import io
import json
import pytest
from tests import historical_scope_input_boundary as boundary


def request(name, node='', *, nested=False):
    relative=('tests/paid_launch/' if nested else 'tests/')+name+'.py'
    return SimpleNamespace(module=SimpleNamespace(__name__=name,__file__=str(boundary.ROOT/relative)),
                           node=SimpleNamespace(originalname=node))


@pytest.mark.parametrize('method',['read_bytes','open'])
def test_physical_reader_is_independent_of_both_path_views(monkeypatch,method):
    path=boundary.ROOT/boundary.BINDING
    expected=boundary.physical_bytes(path)
    def blocked(*args,**kwargs):raise AssertionError('SYNTHETIC_PROJECTED_READER')
    monkeypatch.setattr(Path,method,blocked)
    assert boundary.physical_bytes(path)==expected


def test_actual_nested_paid_module_is_live_before_any_predecessor_access(monkeypatch):
    def forbidden():raise AssertionError('SYNTHETIC_PREDECESSOR_MUST_NOT_BE_CONSULTED')
    monkeypatch.setattr(boundary,'canonical_guard',forbidden)
    req=request('case_isolation_and_scope',nested=True)
    plan=boundary.resolve_context(req)
    assert plan.role is boundary.Role.LIVE
    assert not boundary.apply_selected_inputs(plan,req,None)


@pytest.mark.parametrize('module,node',[
 ('test_ncaaf_transport_scope','test_external_fixture_children_keep_their_actual_live_context'),
 ('test_ci_scheduling_fixture_scope','test_unchanged_native_context_assertion_receives_no_projection'),
 ('test_ci_fixture_stability_scope','test_synthetic_exact_seal_and_ordered_merge'),
])
def test_native_decisions_precede_outer_historical_views(module,node):
    plan=boundary.resolve_context(request(module,node))
    assert plan.role is boundary.Role.NATIVE and plan.generation=='current'
    read,opened=Path.read_bytes,Path.open
    with boundary.InputOwner(plan) as owner:
        assert not boundary.apply_selected_inputs(plan,request(module,node),owner.patch)
        assert Path.read_bytes is read and Path.open is opened
    assert Path.read_bytes is read and Path.open is opened


def test_unknown_explicit_historical_consumer_rejects():
    with pytest.raises(ValueError,match='HISTORICAL_CONSUMER_NOT_AUTHENTICATED'):
        boundary.resolve_context(request('test_historical_scope_input_boundary'),historical=True)


def test_physical_consumer_and_binding_identity_are_reauthenticated(monkeypatch):
    req=request('test_ncaaf_chronology_scope');plan=boundary.resolve_context(req)
    assert plan.role is boundary.Role.HISTORICAL
    native=boundary.physical_bytes
    for relative,reason in [(plan.consumer,'HISTORICAL_CONSUMER_SOURCE_CHANGED'),
                            (boundary.BINDING,'HISTORICAL_BINDING_CHANGED')]:
        with monkeypatch.context() as patch:
            patch.setattr(boundary,'physical_bytes',lambda p:native(p)+b'\nUNREVIEWED' if Path(p)==boundary.ROOT/relative else native(p))
            with pytest.raises(ValueError,match=reason):boundary.authenticate_plan(plan)
    boundary.authenticate_plan(plan)


def test_chronology_input_is_the_exact_authenticated_generation():
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    value=boundary.generation_input(plan,'app_core/ncaaf_pipeline_evidence.py')
    assert boundary.sha(value)=='7fdb9d01149f2e9ee19ee4092ed342ad11107d6cbca570450f964b6df2f9f49a'
    assert boundary.sha(value) not in {'3171c6a9fd74244c86ed10a21b175623ced7ab10f899fdbca40c4d5aaa7d19ba',
                                    '8c14aa0e76ca6fe61daa5511bce55dbecb2ef1c165866a261a3a19521833b1da'}


@pytest.mark.parametrize('failure',[False,True])
def test_owned_live_historical_live_sequence_restores_on_every_exit(failure):
    historical=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    live=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    read,opened=Path.read_bytes,Path.open
    synthetic=lambda path:b'SYNTHETIC-OWNED-VIEW'
    try:
        with boundary.InputOwner(historical) as outer:
            outer.patch.setattr(Path,'read_bytes',synthetic)
            with boundary.InputOwner(live):
                assert Path.read_bytes is read and Path.open is opened
            assert Path.read_bytes is synthetic
            if failure:raise RuntimeError('SYNTHETIC_FAILURE_CLEANUP')
    except RuntimeError as exc:
        assert failure and str(exc)=='SYNTHETIC_FAILURE_CLEANUP'
    assert Path.read_bytes is read and Path.open is opened
    assert not boundary._ACTIVE[1:]


def test_exact_generation_and_binding_mutation_do_not_reuse_a_root_cache(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    first=boundary.generation_input(plan,'app_core/ncaaf_pipeline_evidence.py')
    native=boundary.physical_bytes
    binding=json.loads(native(boundary.ROOT/boundary.BINDING))
    edge=next(iter(binding['historical_input_boundary']['source_generations']))
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'physical_bytes',lambda p:native(p)+b'UNREVIEWED' if Path(p)==boundary.ROOT/edge else native(p))
        with pytest.raises(ValueError,match='HISTORICAL_GENERATION_BINDING_CHANGED'):
            boundary.generation_input(plan,'app_core/ncaaf_pipeline_evidence.py')
    assert boundary.generation_input(plan,'app_core/ncaaf_pipeline_evidence.py')==first


def test_shared_boundary_does_not_change_native_probability_or_operational_modules():
    plan=boundary.resolve_context(request('test_ncaaf_spread_boundaries'))
    assert plan.role is boundary.Role.LIVE
    assert plan.generation=='current'


@pytest.mark.parametrize('field,value',[
    ('generation','SYNTHETIC_WRONG_GENERATION'),
    ('role',boundary.Role.NATIVE),
    ('binding_sha256','0'*64),
])
def test_rehashed_or_substituted_context_cannot_select_another_generation(field,value):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    with pytest.raises(ValueError):boundary.authenticate_plan(replace(plan,**{field:value}))


def test_unrelated_path_patch_survives_owned_nested_native_operation(monkeypatch):
    historical=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    live=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    read=Path.read_bytes;owned=lambda path:b'SYNTHETIC-OWNED';unrelated=lambda path:b'SYNTHETIC-UNRELATED'
    with boundary.InputOwner(historical) as owner:
        owner.patch.setattr(Path,'read_bytes',owned)
        with monkeypatch.context() as patch:
            patch.setattr(Path,'read_bytes',unrelated)
            with boundary.InputOwner(live):assert Path.read_bytes is unrelated
            assert Path.read_bytes is unrelated
        assert Path.read_bytes is owned
    assert Path.read_bytes is read


def test_legacy_bridge_preserves_globals_and_authenticates_every_repeated_call():
    from scripts import ci_fixture_stability_scope as scope
    native=boundary.canonical_guard()
    physical,binding=native._ci_authenticated_source_input()
    predecessor=scope.previous_guard(physical,binding)
    state={'changed':False}
    def authenticate():
        if state['changed']:raise ValueError('SYNTHETIC_CHANGED_BINDING_AT_SAME_ROOT')
        return native._ci_authenticated_source_input()
    def original(value):
        if value!=predecessor:raise ValueError('SYNTHETIC_WRONG_GENERATION')
        return value
    namespace={'__file__':__file__,'_owner_base_source':original}
    read,opened=Path.read_bytes,Path.open
    scope.install_legacy_source_bridge(namespace,authenticate)
    function=namespace['_owner_base_source']
    assert function.__globals__ is original.__globals__
    assert function(physical)==function(physical)==predecessor
    with pytest.raises(ValueError,match='SYNTHETIC_WRONG_GENERATION'):function(physical+b'UNREVIEWED')
    state['changed']=True
    with pytest.raises(ValueError,match='SYNTHETIC_CHANGED_BINDING_AT_SAME_ROOT'):function(physical)
    assert Path.read_bytes is read and Path.open is opened


def test_predecessor_construction_leaves_no_module_entry_or_root_cache():
    import sys
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    native=boundary.canonical_guard();name='exact_ncaaf_future_released_scope'
    before=sys.modules.get(name);read,opened=Path.read_bytes,Path.open
    first,_=_new_released_scope_inputs(native);second,_=_new_released_scope_inputs(native)
    assert first is not second and first.__dict__ is not second.__dict__
    assert sys.modules.get(name) is before
    assert Path.read_bytes is read and Path.open is opened


def test_rehashed_consumer_metadata_cannot_replace_the_physical_contract(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    native=boundary.physical_bytes;path=boundary.ROOT/plan.consumer;altered=native(path)+b'UNREVIEWED'
    changed=replace(plan,source_sha256=boundary.sha(altered))
    monkeypatch.setattr(boundary,'physical_bytes',lambda p:altered if Path(p)==path else native(p))
    with pytest.raises(ValueError,match='HISTORICAL_CONSUMER_SOURCE_CHANGED'):boundary.authenticate_plan(changed)


def test_module_entries_and_unrelated_package_patches_restore_independently(monkeypatch):
    import sys
    historical=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    live=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    key='SYNTHETIC_boundary_module_ownership';before=sys.modules.get(key)
    owned=SimpleNamespace(marker='owned');unrelated=SimpleNamespace(marker='unrelated')
    with boundary.InputOwner(historical) as owner:
        owner.patch.setitem(sys.modules,key,owned)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules,key,unrelated)
            with boundary.InputOwner(live):assert sys.modules[key] is unrelated
            assert sys.modules[key] is unrelated
        assert sys.modules[key] is owned
    assert sys.modules.get(key) is before


def test_owner_exit_does_not_overwrite_an_unrelated_patch(monkeypatch):
    historical=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    read=Path.read_bytes;owned=lambda path:b'SYNTHETIC-OWNED';unrelated=lambda path:b'SYNTHETIC-UNRELATED'
    try:
        with monkeypatch.context() as patch:
            with boundary.InputOwner(historical) as owner:
                owner.patch.setattr(Path,'read_bytes',owned)
                patch.setattr(Path,'read_bytes',unrelated)
            assert Path.read_bytes is unrelated
        assert Path.read_bytes is owned
    finally:
        Path.read_bytes=read


@pytest.mark.parametrize('relative',[
    'tests/ci_fixture_scope_inputs.py','tests/ncaaf_transport_scope_inputs.py',
    'tests/ncaaf_staged_capacity_fixtures.py'])
def test_migrated_helpers_keep_exact_authenticated_historical_input(relative,monkeypatch):
    req=request('test_ci_scheduling_fixture_scope')
    plan=boundary.resolve_context(req)
    raw=boundary.physical_bytes(boundary.ROOT/relative)
    binding=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    record=binding['historical_input_boundary']['helper_preimages'][relative]
    original=boundary.historical_helper_input(plan,relative,raw)
    assert boundary.sha(original)==record['before_sha256']
    assert boundary.historical_helper_input(plan,relative,raw+b'UNREVIEWED')==raw+b'UNREVIEWED'
    assert boundary.historical_helper_input(plan,'tests/UNREGISTERED.py',raw)==raw
    opened,reader=Path.open,Path.read_bytes
    with boundary.InputOwner(plan) as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        assert (boundary.ROOT/relative).read_bytes()==original
        with (boundary.ROOT/relative).open('rb') as stream:assert stream.read()==original
    assert Path.open is opened and Path.read_bytes is reader
    physical=boundary.physical_bytes
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'physical_bytes',lambda path:physical(path)+b'UNREVIEWED' if Path(path)==boundary.ROOT/relative else physical(path))
        with pytest.raises(ValueError,match='HISTORICAL_HELPER_SOURCE_CHANGED'):
            boundary.authenticate_plan(plan)


def test_legacy_helper_module_entries_are_owned_and_cache_clear_remains_compatible():
    import sys
    from tests.ncaaf_transport_scope_inputs import frozen_predecessor
    plan=boundary.resolve_context(request('test_ci_scheduling_fixture_scope'))
    name='exact_ncaaf_transport_predecessor_scope';before=sys.modules.get(name)
    assert frozen_predecessor.cache_clear() is None
    with boundary.InputOwner(plan):
        module,_=frozen_predecessor(str(boundary.ROOT))
        assert sys.modules[name] is module
        assert module.main.__globals__ is vars(module)
    assert sys.modules.get(name) is before


def test_future_layer_reader_selects_transport_input_before_future_reconstruction():
    plan=boundary.resolve_context(request('test_ncaaf_future_spread_scope'))
    assert plan.role is boundary.Role.HISTORICAL and plan.generation=='transport'


@pytest.mark.parametrize('consumer',[
    'tests/test_ci_scheduling_fixture_copy.py',
    'tests/test_ncaaf_chronology_fixture_copy.py',
    'tests/test_ncaaf_future_fixture_copy.py',
    'tests/test_shared_git_fixture_lifecycle.py'])
def test_imported_fixture_alias_authenticates_constructor_and_physical_callsite(consumer,monkeypatch):
    import importlib
    data=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    row=next(r for r in data['historical_input_boundary']['fixture_aliases'] if r['consumer']==consumer)
    constructor=importlib.import_module(Path(row['constructor']).stem)
    caller=importlib.import_module(Path(consumer).stem)
    function=next(v for k,v in vars(caller).items() if k.startswith('test_') and hasattr(v,'__code__') and Path(v.__code__.co_filename).resolve()==Path(caller.__file__).resolve())
    req=SimpleNamespace(module=constructor,function=getattr(constructor,row['fixture']).__wrapped__,
        node=SimpleNamespace(originalname=function.__name__),fixture_consumer_path=caller.__file__,fixture_consumer_function=function)
    plan=boundary.resolve_context(req,phase=row['scope'],fixture=row['alias'])
    assert plan.fixture_definition==row['fixture'] and plan.fixture_consumer==consumer
    assert plan.fixture_consumer_sha256==row['consumer_sha256']
    boundary.authenticate_plan(plan)
    with pytest.raises(ValueError,match='HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED'):
        boundary.resolve_context(req,phase=row['scope'],fixture='UNREGISTERED_ALIAS')
    physical=boundary.physical_bytes
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'physical_bytes',lambda p:physical(p)+b'UNREVIEWED' if Path(p)==boundary.ROOT/consumer else physical(p))
        with pytest.raises(ValueError,match='HISTORICAL_FIXTURE_ALIAS_SOURCE_CHANGED'):
            boundary.authenticate_plan(plan)


def test_all_registered_fixture_sites_match_authenticated_physical_definitions():
    import ast
    data=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    for relative,record in data['historical_input_boundary']['consumers'].items():
        source=boundary.physical_bytes(boundary.ROOT/relative)
        assert boundary.sha(source)==record['sha256']
        expected={}
        for node in ast.parse(source).body:
            if not isinstance(node,ast.FunctionDef):continue
            for decorator in node.decorator_list:
                if isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute) and decorator.func.attr=='fixture':
                    scope=next((ast.literal_eval(k.value) for k in decorator.keywords if k.arg=='scope'),'function')
                    if scope in {'session','module'}:
                        expected[node.name]=dict(scope=scope,line=min([node.lineno]+[d.lineno for d in node.decorator_list]))
        assert record.get('fixtures',{})==expected,relative


@pytest.mark.parametrize('failure',[False,True])
def test_requested_compatibility_namespace_matches_its_owner_and_restores(failure,monkeypatch):
    import sys
    from types import ModuleType
    from tests import ncaaf_future_scope_inputs as inputs
    name='exact_ncaaf_future_released_scope';missing=object()
    prior=sys.modules.get(name,missing)
    module=ModuleType(name)
    exec('import sys\nROOT=None\ndef selected_root():\n    return sys.modules[__name__].ROOT\n',vars(module))
    def forbidden(live):raise AssertionError('SYNTHETIC_REDUNDANT_PREDECESSOR_CONSTRUCTION')
    monkeypatch.setattr(inputs,'_new_released_scope_inputs',forbidden)
    plan=boundary.resolve_context(request('test_ci_scheduling_fixture_copy'))
    native=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    try:
        with boundary.InputOwner(plan) as owner:
            owner.patch.setitem(sys.modules,name,module)
            selected,_=inputs.ncaaf_released_scope_inputs.__wrapped__()
            assert selected is module and sys.modules[name] is module
            module.ROOT='SYNTHETIC-COPIED-REPOSITORY'
            assert module.selected_root()==module.ROOT
            with boundary.InputOwner(native):
                assert sys.modules.get(name,missing) is prior
            assert sys.modules[name] is module
            if failure:raise RuntimeError('SYNTHETIC_FIXTURE_FAILURE')
    except RuntimeError as exc:
        assert failure and str(exc)=='SYNTHETIC_FIXTURE_FAILURE'
    assert sys.modules.get(name,missing) is prior


@pytest.mark.parametrize('consumer',[
    'test_historical_scope_input_boundary','test_ncaaf_chronology_scope'])
def test_unregistered_explicit_compatibility_request_rejects_before_construction(consumer,monkeypatch):
    from tests import ncaaf_future_scope_inputs as inputs
    def forbidden(live):raise AssertionError('SYNTHETIC_PREDECESSOR_CONSTRUCTION_FORBIDDEN')
    monkeypatch.setattr(inputs,'_new_released_scope_inputs',forbidden)
    plan=boundary.resolve_context(request(consumer))
    with boundary.InputOwner(plan):
        with pytest.raises(ValueError,match='HISTORICAL_FIXTURE_'):
            inputs.ncaaf_released_scope_inputs.__wrapped__()


def test_requested_compatibility_fixture_rejects_a_substituted_owned_entry(monkeypatch):
    import sys
    from types import ModuleType
    from tests import ncaaf_future_scope_inputs as inputs
    name='exact_ncaaf_future_released_scope'
    plan=boundary.resolve_context(request('test_ci_scheduling_fixture_copy'))
    owned,foreign=ModuleType(name),ModuleType(name)
    with boundary.InputOwner(plan) as owner:
        owner.patch.setitem(sys.modules,name,owned)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules,name,foreign)
            with pytest.raises(ValueError,match='HISTORICAL_FIXTURE_MODULE_CONFLICT'):
                inputs.ncaaf_released_scope_inputs.__wrapped__()
        assert sys.modules[name] is owned


@pytest.mark.parametrize('descriptor',['tests.test_football_document_ci_closure','UNREGISTERED'])
def test_legacy_descriptor_without_its_physical_callsite_cannot_project(descriptor):
    from tests import ncaaf_staged_capacity_fixtures as inputs
    class NoMutation:
        def __getattr__(self,name):raise AssertionError('SYNTHETIC_UNAUTHENTICATED_VIEW')
    read,opened=Path.read_bytes,Path.open
    req=SimpleNamespace(module=SimpleNamespace(__name__=descriptor))
    with pytest.raises(ValueError,match='HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED'):
        inputs.install_reviewed_target_profile_inputs(req,NoMutation())
    assert Path.read_bytes is read and Path.open is opened


@pytest.mark.parametrize('relative,reason',[
    (boundary.BINDING,'HISTORICAL_BINDING_CHANGED'),
    ('tests/ncaaf_staged_capacity_fixtures.py','HISTORICAL_API_SOURCE_CHANGED'),
    ('tests/test_ncaaf_reviewed_target_timing.py','HISTORICAL_API_CONSUMER_SOURCE_CHANGED')])
def test_legacy_callsite_authentication_uses_physical_sources_before_projection(relative,reason,monkeypatch):
    from tests import ncaaf_staged_capacity_fixtures as inputs
    physical=boundary.physical_bytes
    def altered(path):
        value=physical(path)
        return b' '+value if Path(path)==boundary.ROOT/relative else value
    req=SimpleNamespace(module=SimpleNamespace(__name__='tests.test_football_document_ci_closure'))
    read,opened=Path.read_bytes,Path.open
    class NoMutation:
        def __getattr__(self,name):raise AssertionError('SYNTHETIC_UNAUTHENTICATED_VIEW')
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'physical_bytes',altered)
        with pytest.raises(ValueError,match=reason):
            inputs.install_reviewed_target_profile_inputs(req,NoMutation())
    assert Path.read_bytes is read and Path.open is opened


@pytest.mark.parametrize('failure',[False,True])
def test_explicit_legacy_patch_is_owned_through_native_entry_and_exception_cleanup(failure,monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    native=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    read=Path.read_bytes;outer=lambda path:b'SYNTHETIC_OUTER';inner=lambda path:b'SYNTHETIC_INNER'
    try:
        with boundary.InputOwner(plan) as owner:
            owner.patch.setattr(Path,'read_bytes',outer)
            with monkeypatch.context() as explicit:
                view=boundary.owned_api_patch(explicit,plan)
                view.setattr(Path,'read_bytes',inner)
                assert Path.read_bytes is inner
                with boundary.InputOwner(native):
                    assert Path.read_bytes is read
                assert Path.read_bytes is inner
                if failure:raise RuntimeError('SYNTHETIC_LEGACY_EXCEPTION')
            assert Path.read_bytes is outer
    except RuntimeError as exc:
        assert failure and str(exc)=='SYNTHETIC_LEGACY_EXCEPTION'
    assert Path.read_bytes is read


def test_explicit_legacy_undo_preserves_an_unrelated_path_method(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    read,opened=Path.read_bytes,Path.open
    inner=lambda path:b'SYNTHETIC_INNER'
    unrelated=lambda *a,**k:opened(*a,**k)
    with boundary.InputOwner(plan):
        with monkeypatch.context() as other:
            other.setattr(Path,'open',unrelated)
            with monkeypatch.context() as explicit:
                boundary.owned_api_patch(explicit,plan).setattr(Path,'read_bytes',inner)
                assert Path.read_bytes is inner
            assert Path.read_bytes is read and Path.open is unrelated
    assert Path.read_bytes is read and Path.open is opened


def test_explicit_legacy_undo_inside_native_context_cannot_resurrect_a_view(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    native=boundary.resolve_context(request('test_historical_scope_input_boundary'))
    read=Path.read_bytes;outer=lambda path:b'SYNTHETIC_OUTER';inner=lambda path:b'SYNTHETIC_INNER'
    with boundary.InputOwner(plan) as owner:
        owner.patch.setattr(Path,'read_bytes',outer)
        with monkeypatch.context() as explicit:
            boundary.owned_api_patch(explicit,plan).setattr(Path,'read_bytes',inner)
            with boundary.InputOwner(native):
                explicit.undo()
                assert Path.read_bytes is read
            assert Path.read_bytes is outer
        assert Path.read_bytes is outer
    assert Path.read_bytes is read


def test_explicit_fixture_undo_after_outer_exit_does_not_restore_a_historical_view(monkeypatch):
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    plan=boundary.resolve_context(request('test_ncaaf_reviewed_target_timing'))
    read,opened=Path.read_bytes,Path.open
    with monkeypatch.context() as explicit:
        with boundary.InputOwner(plan) as owner:
            boundary.apply_selected_inputs(plan,request('test_ncaaf_reviewed_target_timing'),owner.patch)
            import sys
            selected=sys.modules['scripts.check_launch_change_scope']
            assert selected is not boundary.canonical_guard()
            assert not hasattr(selected,'_RT_BINDING_SHA256')
            fixtures.exact_three_cost_review_inputs.__wrapped__(explicit)
            assert Path.read_bytes is not read
        assert Path.read_bytes is read and Path.open is opened
        explicit.undo()
        assert Path.read_bytes is read and Path.open is opened


def test_unregistered_explicit_fixture_caller_cannot_own_a_view(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_chronology_scope'))
    with boundary.InputOwner(plan):
        with pytest.raises(ValueError,match='HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED'):
            boundary.owned_fixture_patch(monkeypatch)


def generated_session_request(tmp_path, *, companion=False):
    import ast,textwrap
    path='tests/test_ncaaf_future_spread_scope.py'
    parent=next(n for n in ast.parse(boundary.physical_bytes(boundary.ROOT/path)).body
                if isinstance(n,ast.FunctionDef) and n.name=='test_real_session_setup_gets_exact_inputs_and_restores_live_context')
    child=tmp_path/'SYNTHETIC-session-fixture';child.mkdir()
    (child/'conftest.py').write_bytes(boundary.physical_bytes(boundary.ROOT/'tests/conftest.py'))
    for call in ast.walk(parent):
        if not(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='write_text'):continue
        target=call.func.value
        if not(isinstance(target,ast.BinOp) and isinstance(target.right,ast.Constant)):continue
        (child/target.right.value).write_text(textwrap.dedent(ast.literal_eval(call.args[0].args[0])),encoding='utf-8',newline='\n')
    selected=child/('test_ncaaf_future_spread_scope.py' if companion else 'test_aaa_historical.py')
    namespace={'__file__':str(selected),'__name__':'SYNTHETIC_GENERATED_TEMPLATE'}
    if not companion:exec(compile(boundary.physical_bytes(selected),str(selected),'exec'),namespace)
    module=SimpleNamespace(__name__=selected.stem,__file__=str(selected))
    config=SimpleNamespace(_historical_generated_parent=path+'::'+parent.name)
    req=SimpleNamespace(module=module,config=config,
        node=SimpleNamespace(originalname='test_live_context_after_historical_test_and_setup' if companion else 'test_session_template_precedes_function_projection'),
        function=None if companion else namespace['immutable_template_input'].__wrapped__)
    return req,selected


@pytest.mark.parametrize('failure',[False,True])
def test_exact_generated_session_has_source_only_view_and_native_guard(tmp_path,failure):
    import sys
    req,path=generated_session_request(tmp_path)
    live=boundary.canonical_guard();before=boundary.physical_bytes(boundary.ROOT/boundary.GUARD)
    read,opened=Path.read_bytes,Path.open
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    assert plan.role is boundary.Role.HISTORICAL
    try:
        with boundary.InputOwner(plan) as owner:
            assert boundary.apply_selected_inputs(plan,req,owner.patch)
            selected,source=req.function()
            assert selected is live is sys.modules['scripts.check_launch_change_scope']
            assert boundary.sha(source)=='3720e55ec37baae7dcf53d5eb3b3436fcc69ae41df820a110900965945adb306'
            assert boundary.physical_bytes(boundary.ROOT/boundary.GUARD)==before
            if failure:raise RuntimeError('SYNTHETIC_GENERATED_CLEANUP')
    except RuntimeError as exc:
        assert failure and str(exc)=='SYNTHETIC_GENERATED_CLEANUP'
    assert Path.read_bytes is read and Path.open is opened
    assert live is sys.modules['scripts.check_launch_change_scope']
    assert (boundary.ROOT/boundary.GUARD).read_bytes()==before


def test_generated_companion_and_call_phase_stay_native_without_source_construction(tmp_path,monkeypatch):
    req,path=generated_session_request(tmp_path,companion=True)
    plan=boundary.resolve_context(req)
    assert plan.role is boundary.Role.NATIVE and plan.generation=='current'
    def forbidden(*a):raise AssertionError('SYNTHETIC_NATIVE_SOURCE_CONSTRUCTION')
    monkeypatch.setattr(boundary,'_install_generated_source_input',forbidden)
    read,opened=Path.read_bytes,Path.open
    with boundary.InputOwner(plan) as owner:
        assert not boundary.apply_selected_inputs(plan,req,owner.patch)
    assert Path.read_bytes is read and Path.open is opened
    with pytest.raises(ValueError,match='HISTORICAL_CONSUMER_NOT_AUTHENTICATED'):
        boundary.resolve_context(req,historical=True)


def test_same_named_external_program_has_no_authority_without_exact_parent(tmp_path):
    req,path=generated_session_request(tmp_path)
    req.config._historical_generated_parent='UNREGISTERED::test_unreviewed'
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    assert plan.role is boundary.Role.LIVE
    assert not boundary.apply_selected_inputs(plan,req,None)
    with pytest.raises(ValueError,match='HISTORICAL_CONSUMER_NOT_AUTHENTICATED'):
        boundary.resolve_context(req,phase='session',fixture='immutable_template_input',historical=True)


@pytest.mark.parametrize('name',['test_aaa_historical.py','test_ncaaf_future_spread_scope.py','conftest.py'])
def test_generated_session_reauthenticates_each_physical_program_and_conftest(tmp_path,name):
    req,path=generated_session_request(tmp_path)
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    altered=path.parent/name;original=boundary.physical_bytes(altered)
    altered.write_bytes(original+b'\n# SYNTHETIC ALTERED SOURCE\n')
    with pytest.raises(ValueError,match='HISTORICAL_GENERATED_(PROGRAM|CONFTEST)_CHANGED'):
        boundary.authenticate_plan(plan)


@pytest.mark.parametrize('field,value',[
    ('generation','SYNTHETIC_OTHER_GENERATION'),('fixture','different_fixture'),
    ('generated_parent','UNREGISTERED::test_same_name'),('binding_sha256','0'*64)])
def test_generated_session_rejects_substituted_generation_fixture_origin_or_binding(tmp_path,field,value):
    req,path=generated_session_request(tmp_path)
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    with pytest.raises(ValueError):boundary.authenticate_plan(replace(plan,**{field:value}))


def test_generated_session_authenticates_parent_before_source_view(tmp_path,monkeypatch):
    req,path=generated_session_request(tmp_path)
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    physical=boundary.physical_bytes
    def changed(p):
        raw=physical(p)
        return raw+b'\n# SYNTHETIC PARENT CHANGE' if Path(p)==boundary.ROOT/plan.consumer else raw
    read,opened=Path.read_bytes,Path.open
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'physical_bytes',changed)
        with pytest.raises(ValueError,match='HISTORICAL_GENERATED_PARENT_CHANGED'):
            boundary.authenticate_plan(plan)
    assert Path.read_bytes is read and Path.open is opened


def test_generated_source_view_cannot_hide_unrelated_source_substitution(tmp_path,monkeypatch):
    req,path=generated_session_request(tmp_path)
    plan=boundary.resolve_context(req,phase='session',fixture='immutable_template_input')
    opened=Path.open;target=(boundary.ROOT/boundary.GUARD).resolve()
    def altered(p,mode='r',*args,**kwargs):
        return io.BytesIO(b'SYNTHETIC WRONG SOURCE') if Path(p).resolve()==target and mode=='rb' else opened(p,mode,*args,**kwargs)
    with monkeypatch.context() as outside:
        outside.setattr(Path,'open',altered)
        with boundary.InputOwner(plan) as owner:
            boundary.apply_selected_inputs(plan,req,owner.patch)
            with pytest.raises(ValueError,match='HISTORICAL_GENERATED_SOURCE_INPUT_CHANGED'):req.function()
    assert Path.open is opened


@pytest.mark.parametrize('consumer,generation,reader',[
    ('test_ncaaf_postmerge_scope','transport','_ncaaf_future_base_source'),
    ('test_scope_git_batch','transport-future','_guard_performance_base_source'),
    ('test_scope_git_reads','transport-future','_guard_performance_base_source'),
    ('test_scope_reconstruction_memo','transport-future','_nfl_odds_timing_base_source')])
def test_late_guard_import_uses_selected_namespace_and_exact_source_generation(consumer,generation,reader):
    import importlib,sys
    import scripts
    req=request(consumer)
    assert not hasattr(req.module,'guard')
    plan=boundary.resolve_context(req)
    assert plan.role is boundary.Role.HISTORICAL and plan.generation==generation
    native=boundary.canonical_guard();opened,read=Path.open,Path.read_bytes
    original_entry=sys.modules['scripts.check_launch_change_scope']
    with boundary.InputOwner(plan) as owner:
        assert boundary.apply_selected_inputs(plan,req,owner.patch)
        selected=importlib.import_module('scripts.check_launch_change_scope')
        assert scripts.check_launch_change_scope is selected and selected is not native
        source=Path(selected.__file__).read_bytes().replace(b'\r\n',b'\n')
        first=getattr(selected,reader)(source)
        assert first==getattr(selected,reader)(source)
        with pytest.raises(ValueError):getattr(selected,reader)(source+b'\nSYNTHETIC_UNREVIEWED=True\n')
        if generation=='transport':
            assert selected._ncaaf_future_guard_matches(source)
            assert len(source)==8388490
    assert sys.modules['scripts.check_launch_change_scope'] is original_entry
    assert Path.open is opened and Path.read_bytes is read


@pytest.mark.parametrize('failure',[False,True])
def test_late_import_namespace_is_suspended_for_native_and_restores_on_exception(failure):
    import importlib,sys
    import scripts
    req=request('test_ncaaf_postmerge_scope');plan=boundary.resolve_context(req)
    native=boundary.canonical_guard();entry=sys.modules['scripts.check_launch_change_scope']
    opened,read=Path.open,Path.read_bytes
    try:
        with boundary.InputOwner(plan) as owner:
            boundary.apply_selected_inputs(plan,req,owner.patch)
            selected=importlib.import_module('scripts.check_launch_change_scope')
            assert selected is not native
            inner=boundary.resolve_context(request('test_historical_scope_input_boundary'))
            with boundary.InputOwner(inner):
                assert importlib.import_module('scripts.check_launch_change_scope') is native
                assert scripts.check_launch_change_scope is native
            assert importlib.import_module('scripts.check_launch_change_scope') is selected
            if failure:raise RuntimeError('SYNTHETIC_LATE_IMPORT_FAILURE')
    except RuntimeError as exc:
        assert failure and str(exc)=='SYNTHETIC_LATE_IMPORT_FAILURE'
    assert sys.modules['scripts.check_launch_change_scope'] is entry
    assert Path.open is opened and Path.read_bytes is read


@pytest.mark.parametrize('consumer',[
    'test_ci_scheduling_fixture_scope','test_ncaaf_chronology_fixture_scope',
    'test_ncaaf_chronology_capacity'])
def test_stability_generation_keeps_current_namespace_and_exact_prior_source(consumer,monkeypatch):
    import importlib,sys
    import scripts
    from tests import ncaaf_future_scope_inputs as future,ncaaf_transport_scope_inputs as transport
    req=request(consumer);plan=boundary.resolve_context(req)
    assert plan.role is boundary.Role.HISTORICAL and plan.generation=='stability'
    def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_EARLIER_NAMESPACE_MUST_NOT_BE_CONSTRUCTED')
    monkeypatch.setattr(future,'_new_released_scope_inputs',forbidden)
    monkeypatch.setattr(transport,'frozen_predecessor',forbidden)
    native=boundary.canonical_guard();entry=sys.modules['scripts.check_launch_change_scope']
    physical=boundary.physical_bytes(boundary.ROOT/boundary.GUARD)
    binding=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    opened,read=Path.open,Path.read_bytes
    with boundary.InputOwner(plan) as owner:
        assert boundary.apply_selected_inputs(plan,req,owner.patch)
        assert importlib.import_module('scripts.check_launch_change_scope') is native
        assert scripts.check_launch_change_scope is native
        with (boundary.ROOT/boundary.GUARD).open('rb') as stream:source=stream.read().replace(b'\r\n',b'\n')
        expected=binding['predecessor_guard']['sha256']
        if consumer=='test_ncaaf_chronology_fixture_scope':
            original=json.loads(boundary.physical_bytes(boundary.ROOT/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'))
            expected=original['chronology_capacity']['previous_guard_sha256']
        assert boundary.sha(source)==expected and len(source)==8388490
        assert boundary.physical_bytes(boundary.ROOT/boundary.GUARD)==physical
        native._ci_fixture_module()
    assert sys.modules['scripts.check_launch_change_scope'] is entry
    assert Path.open is opened and Path.read_bytes is read


def test_capacity_historical_profile_and_nested_current_gate():
    from tests import ci_fixture_stability_inputs as inputs
    profile=boundary.ROOT/inputs.PROFILE
    physical=boundary.physical_bytes(profile)
    prior=inputs.capacity_predecessor_source(inputs.PROFILE,physical)
    assert boundary.sha(prior)=='6ea7580a2d3098c6906fd111c8d0f32bda7e3daec321e4717fa58aa03ba42cba'
    historical=request('test_ci_fixture_stability_scope','test_complete_profile_refresh_preserves_all_prior_rows_and_measurements')
    current=request('test_ci_test_shards','test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate')
    old=boundary.resolve_context(historical);new=boundary.resolve_context(current)
    assert old.role is boundary.Role.HISTORICAL and new.role in {boundary.Role.NATIVE,boundary.Role.LIVE}
    opened,reader=Path.open,Path.read_bytes
    with boundary.InputOwner(old) as owner:
        assert boundary.apply_selected_inputs(old,historical,owner.patch)
        assert profile.read_bytes()==prior
        assert '--shards 11' in (boundary.ROOT/'.github/workflows/ci.yml').read_text()
        with boundary.InputOwner(new) as inner:
            assert not boundary.apply_selected_inputs(new,current,inner.patch)
            assert profile.read_bytes()==physical
            assert '--shards 12' in (boundary.ROOT/'.github/workflows/ci.yml').read_text()
        assert profile.read_bytes()==prior
    assert Path.open is opened and Path.read_bytes is reader
    assert profile.read_bytes()==physical


def test_capacity_records_and_binding_mutations_reject(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    req=request('test_ci_fixture_stability_scope','test_complete_profile_views_are_exact_reject_alteration_and_restore_current_reads')
    plan=boundary.resolve_context(req);boundary.authenticate_plan(plan)
    records=boundary.capacity_input_records(plan)
    assert set(records)==set(inputs.CAPACITY_SOURCE_RECORDS)
    changed=json.loads(json.dumps(inputs.CAPACITY_SOURCE_RECORDS))
    changed[inputs.PROFILE]['before_git_blob']='0'*40
    with monkeypatch.context() as patch:
        patch.setattr(inputs,'CAPACITY_SOURCE_RECORDS',changed)
        with pytest.raises(ValueError,match='CI_CAPACITY_INPUT_GENERATION_CHANGED'):
            boundary.capacity_input_records(plan)
        with pytest.raises(ValueError,match='CI_CAPACITY_AUTHENTICATED_PREDECESSOR_CHANGED'):
            inputs.capacity_predecessor_source(inputs.PROFILE,boundary.physical_bytes(boundary.ROOT/inputs.PROFILE))
    physical=boundary.physical_bytes
    for relative,code in [('tests/ci_fixture_stability_inputs.py','CI_CAPACITY_INPUT_HELPER_CHANGED'),(boundary.BINDING,'HISTORICAL_BINDING_CHANGED')]:
        with monkeypatch.context() as patch:
            patch.setattr(boundary,'physical_bytes',lambda path,_rel=relative:physical(path)+b'UNREVIEWED' if Path(path)==boundary.ROOT/_rel else physical(path))
            with pytest.raises(ValueError,match=code):boundary.capacity_input_records(plan)
    raw=physical(boundary.ROOT/inputs.PROFILE)
    assert inputs.capacity_predecessor_source(inputs.PROFILE,raw+b'UNREVIEWED')==raw+b'UNREVIEWED'
    assert boundary.capacity_input_records(plan)==records


def test_capacity_failure_cleanup_restores_live_inputs():
    from tests import ci_fixture_stability_inputs as inputs
    req=request('test_ci_fixture_stability_scope','test_complete_profile_refresh_preserves_all_prior_rows_and_measurements')
    plan=boundary.resolve_context(req);raw=boundary.physical_bytes(boundary.ROOT/inputs.PROFILE)
    opened,reader=Path.open,Path.read_bytes
    with pytest.raises(RuntimeError,match='SYNTHETIC_CAPACITY_FAILURE'):
        with boundary.InputOwner(plan) as owner:
            boundary.apply_selected_inputs(plan,req,owner.patch)
            assert (boundary.ROOT/inputs.PROFILE).read_bytes()!=raw
            raise RuntimeError('SYNTHETIC_CAPACITY_FAILURE')
    assert Path.open is opened and Path.read_bytes is reader
    assert (boundary.ROOT/inputs.PROFILE).read_bytes()==raw
    assert not boundary._ACTIVE[1:]
