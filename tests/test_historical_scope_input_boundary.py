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


@pytest.fixture
def released_bootstrap_owner():
    req=request('test_football_document_io_scope','test_exact_policy_and_ordered_ci_merge')
    with boundary.InputOwner(boundary.resolve_context(req)) as owner:
        yield owner


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


def test_fixture_initializer_preserves_exact_helper_generation(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    path='tests/ncaaf_future_scope_inputs.py'
    current=boundary.physical_bytes(boundary.ROOT/path)
    prior=inputs.fixture_initializer_predecessor_source(path,current)
    assert boundary.sha(prior)=='3a4644e5d638d2ab83dcd02aaf19240552d5923659f950c5af1e77f4dd3d0dc4'
    assert inputs.fixture_initializer_predecessor_source(path,prior)==prior
    assert inputs.fixture_initializer_predecessor_source(path,current+b'\n# SYNTHETIC ALTERATION')==current+b'\n# SYNTHETIC ALTERATION'
    assert boundary.sha(inputs.original_source(path,current))==inputs.RECORDS[path]['before_sha256']
    assert boundary.sha(inputs.original_source(path,prior))==inputs.RECORDS[path]['before_sha256']
    rows=json.loads(json.dumps(inputs.FIXTURE_INITIALIZER_SOURCE_RECORDS))
    rows[path]['before_sha256']='0'*64
    monkeypatch.setattr(inputs,'FIXTURE_INITIALIZER_SOURCE_RECORDS',rows)
    with pytest.raises(ValueError,match='CI_FIXTURE_INITIALIZER_PREDECESSOR_CHANGED'):
        inputs.fixture_initializer_predecessor_source(path,current)


def test_fresh_fixture_initializes_existing_memo_before_consumption_and_reauthenticates(tmp_path,monkeypatch):
    import importlib
    consumer=importlib.import_module('tests.test_football_document_io_scope')
    name='test_exact_policy_and_ordered_ci_merge'
    req=SimpleNamespace(module=consumer,node=SimpleNamespace(name=name,originalname=name))
    plan=boundary.resolve_context(req)
    native=consumer.guard;read,opened=Path.read_bytes,Path.open
    incoming=tuple(boundary._ACTIVE)
    with boundary.InputOwner(plan) as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        guard=consumer.guard
        assert guard is not native and guard._scope_pure_reconstruction_installed is True
        runtime=guard._scope_reconstruction_for_validation()
        assert (runtime.MAX_ENTRIES,runtime.MAX_RETAINED_BYTES,runtime.MAX_CONTEXT_FIELDS)==(8,32*1024*1024,512)
        for field in runtime.NAMES:
            fn=getattr(guard,field)
            assert fn.__wrapped__.__globals__ is guard.__dict__
            assert fn.reconstruction_context_fields<=runtime.MAX_CONTEXT_FIELDS
            assert fn.reconstruction_memo.retained_bytes<=runtime.MAX_RETAINED_BYTES
        directory=tmp_path/'scripts';directory.mkdir()
        (directory/'scope_reconstruction.py').write_bytes(boundary.physical_bytes(boundary.ROOT/'scripts/scope_reconstruction.py')+b'\n# SYNTHETIC TAMPER')
        with monkeypatch.context() as changed:
            changed.setattr(guard,'__file__',str(directory/'check_launch_change_scope.py'))
            with pytest.raises(ValueError,match='SUCCESSOR_SCOPE_MODULE_CHANGED'):
                guard._scope_reconstruction_for_validation()
    assert consumer.guard is native and tuple(boundary._ACTIVE)==incoming
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


def test_predecessor_construction_leaves_no_module_entry_or_root_cache(released_bootstrap_owner):
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
    # Both authenticated edges retain their exact inputs, not regenerated hashes.
    for relative,row in inputs.CAPACITY_SUCCESSOR_RECORDS.items():
        current_source=boundary.physical_bytes(boundary.ROOT/relative)
        twelve=inputs.capacity_successor_source(relative,current_source)
        assert boundary.sha(current_source)==row['after_sha256']
        assert boundary.sha(twelve)==row['before_sha256']
        assert inputs.capacity_successor_source(relative,twelve)==twelve
        eleven=inputs.capacity_predecessor_source(relative,current_source)
        assert boundary.sha(eleven)==inputs.CAPACITY_SOURCE_RECORDS[relative]['before_sha256']
    historical=request('test_ci_fixture_stability_scope','test_complete_profile_refresh_preserves_all_prior_rows_and_measurements')
    current=request('test_ci_test_shards','test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate')
    old=boundary.resolve_context(historical);new=boundary.resolve_context(current)
    assert old.role is boundary.Role.HISTORICAL and new.role in {boundary.Role.NATIVE,boundary.Role.LIVE}
    opened,reader=Path.open,Path.read_bytes
    with boundary.InputOwner(old) as owner:
        assert boundary.apply_selected_inputs(old,historical,owner.patch)
        assert profile.read_bytes()==prior
        assert '--shards 11' in (boundary.ROOT/'.github/workflows/ci.yml').read_text()
        from tests.ncaaf_staged_capacity_fixtures import install_chronology_capacity_inputs
        from scripts import ncaaf_chronology_capacity_scope as chronology
        live=boundary.canonical_guard()
        chronology_binding=json.loads(boundary.physical_bytes(boundary.ROOT/chronology.BINDING))
        target='tests/test_ncaaf_staged_ci_capacity.py'
        outer=(boundary.ROOT/target).read_bytes()
        with (boundary.ROOT/target).open('rb') as source:
            chronology_input=source.read()
        reconstruction=chronology_binding['chronology_capacity']['source_reconstructions'][target]
        assert boundary.sha(chronology_input)==reconstruction['after_sha256']
        expected=chronology.previous_source(target,chronology_input,live,chronology_binding)
        assert boundary.sha(expected)==reconstruction['before_sha256']
        req=request('test_ncaaf_reviewed_target_timing','test_timing_fixture_changes_preserve_all_original_assertions_without_history')
        req.node.name=req.node.originalname
        for fail in (False,True):
            try:
                with pytest.MonkeyPatch.context() as nested:
                    assert install_chronology_capacity_inputs(req,nested)
                    assert (boundary.ROOT/target).read_bytes()==expected!=outer
                    with (boundary.ROOT/target).open('rb') as stream:assert stream.read()==expected
                    if fail:raise RuntimeError('SYNTHETIC_NESTED_CAPACITY_FAILURE')
            except RuntimeError as exc:
                assert fail and str(exc)=='SYNTHETIC_NESTED_CAPACITY_FAILURE'
            assert (boundary.ROOT/target).read_bytes()==outer
        with boundary.InputOwner(new) as inner:
            assert not boundary.apply_selected_inputs(new,current,inner.patch)
            assert profile.read_bytes()==physical
            assert '--shards 13' in (boundary.ROOT/'.github/workflows/ci.yml').read_text()
        assert profile.read_bytes()==prior
    assert Path.open is opened and Path.read_bytes is reader
    assert profile.read_bytes()==physical


def test_capacity_records_and_binding_mutations_reject(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    req=request('test_ci_fixture_stability_scope','test_complete_profile_views_are_exact_reject_alteration_and_restore_current_reads')
    plan=boundary.resolve_context(req);boundary.authenticate_plan(plan)
    with pytest.raises(ValueError,match='HISTORICAL_CONSUMER_NOT_AUTHENTICATED'):
        boundary.install_capacity_input_view(replace(plan,consumer='tests/SYNTHETIC_UNREGISTERED.py'),monkeypatch)
    with pytest.raises(ValueError,match='HISTORICAL_API_OWNER_REQUIRED'):
        boundary.install_capacity_input_view(plan,monkeypatch)
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
    successor=json.loads(json.dumps(inputs.CAPACITY_SUCCESSOR_RECORDS))
    path='.github/workflows/ci.yml';successor[path]['before_git_blob']='0'*40
    with monkeypatch.context() as patch:
        patch.setattr(inputs,'CAPACITY_SUCCESSOR_RECORDS',successor)
        with pytest.raises(ValueError,match='CI_CAPACITY_INPUT_GENERATION_CHANGED'):
            boundary.capacity_input_records(plan)
        with pytest.raises(ValueError,match='CI_CAPACITY_AUTHENTICATED_PREDECESSOR_CHANGED'):
            inputs.capacity_successor_source(path,boundary.physical_bytes(boundary.ROOT/path))
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


def test_predecessor_image_reuses_only_immutable_transformation(monkeypatch):
    scope,binding=boundary.canonical_guard()._ci_fixture_module()
    monkeypatch.setattr(boundary,'_PREDECESSOR_IMAGE',None)
    decode=scope.base64.b85decode;calls=[]
    def observed(value):
        calls.append(1)
        return decode(value)
    monkeypatch.setattr(scope.base64,'b85decode',observed)
    first=boundary.authenticated_predecessor(scope,binding)
    assert boundary.authenticated_predecessor(scope,binding) is first
    assert len(calls)==1
    assert type(first) is bytes and boundary.sha(first)==binding['predecessor_guard']['sha256']
    key,image=boundary._PREDECESSOR_IMAGE
    monkeypatch.setattr(boundary,'_PREDECESSOR_IMAGE',(key,image+b'UNREVIEWED'))
    with pytest.raises(ValueError,match='PREVIOUS_GUARD_LOGIC_CHANGED'):
        boundary.authenticated_predecessor(scope,binding)


@pytest.mark.parametrize('attack',['source','binding','generation'])
def test_warm_predecessor_image_never_caches_authentication(monkeypatch,attack):
    from copy import deepcopy
    scope,binding=boundary.canonical_guard()._ci_fixture_module()
    original=boundary.authenticated_predecessor(scope,binding)
    changed=deepcopy(binding);physical=boundary.physical_bytes
    if attack=='source':
        monkeypatch.setattr(boundary,'physical_bytes',lambda path:physical(path)+b'UNREVIEWED' if Path(path)==boundary.ROOT/boundary.GUARD else physical(path))
    elif attack=='binding':changed['base']='0'*40
    else:changed['predecessor_guard']['sha256']='0'*64
    with pytest.raises(ValueError):boundary.authenticated_predecessor(scope,changed)
    assert type(original) is bytes


@pytest.mark.parametrize('exit_kind',['success','failure','interruption'])
def test_transitive_reader_generation_is_owned_and_restored(tmp_path,exit_kind):
    import importlib,sys
    package=importlib.import_module('tests')
    names=('tests.test_ci_test_shards','tests.test_ncaaf_staged_ci_capacity','tests.test_ncaaf_reviewed_target_timing')
    native={name:importlib.import_module(name) for name in names}
    req=request('test_ncaaf_chronology_capacity','test_current_eleven_way_gate_preserves_profile_limits_and_history')
    plan=boundary.resolve_context(req)
    read,opened=Path.read_bytes,Path.open
    try:
        with boundary.InputOwner(plan) as owner:
            boundary.apply_selected_inputs(plan,req,owner.patch)
            selected={name:sys.modules[name] for name in names}
            assert all(selected[name] is not native[name] for name in names)
            consumer=importlib.import_module('tests.test_ncaaf_chronology_capacity')
            consumer.test_current_eleven_way_gate_preserves_profile_limits_and_history()
            consumer.test_historical_input_projection_is_exact_and_restores_global_state(tmp_path)
            live=boundary.resolve_context(request('test_historical_scope_input_boundary'))
            with boundary.InputOwner(live):
                assert all(sys.modules[name] is native[name] for name in names)
                native[names[0]].test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate()
            assert all(sys.modules[name] is selected[name] for name in names)
            if exit_kind=='failure':raise RuntimeError('SYNTHETIC_TRANSITIVE_FAILURE')
            if exit_kind=='interruption':raise KeyboardInterrupt('SYNTHETIC_TRANSITIVE_INTERRUPTION')
    except (RuntimeError,KeyboardInterrupt):
        assert exit_kind!='success'
    assert all(sys.modules[name] is native[name] and getattr(package,name.split('.')[1]) is native[name] for name in names)
    assert Path.read_bytes is read and Path.open is opened


@pytest.mark.parametrize('exit_kind',['success','failure','interruption'])
def test_bootstrap_assignment_owns_only_its_exact_reader(exit_kind):
    import types
    module=types.ModuleType('SYNTHETIC_bootstrap_owner')
    calls=[]
    def reader(value):
        calls.append(value)
        assert value==b'SYNTHETIC exact generation'
        return b'SYNTHETIC unchanged parent'
    module.reader=reader
    module.__dict__['source']=b'SYNTHETIC later generation'
    action='result=reader(source)'
    if exit_kind=='failure':action+='\nraise RuntimeError("SYNTHETIC failure")'
    if exit_kind=='interruption':action+='\nraise KeyboardInterrupt("SYNTHETIC interruption")'
    opened,read=Path.open,Path.read_bytes
    try:
        boundary._run_bootstrap_assignment(module,compile(action,'<SYNTHETIC bootstrap>','exec'),
            'reader',module.source,b'SYNTHETIC exact generation',boundary.sha(b'SYNTHETIC unchanged parent'))
    except (RuntimeError,KeyboardInterrupt):
        assert exit_kind!='success'
    assert module.reader is reader and calls==[b'SYNTHETIC exact generation']
    assert Path.open is opened and Path.read_bytes is read


def test_actual_cold_and_warm_bootstraps_keep_fresh_namespaces(released_bootstrap_owner):
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    live=boundary.canonical_guard()
    opened,read=Path.open,Path.read_bytes
    first,_=_new_released_scope_inputs(live)
    image=boundary._BOOTSTRAP_IMAGES
    compiled=boundary._BOOTSTRAP_PLAN
    assert compiled is not None and compiled[5]<=boundary._BOOTSTRAP_LIMIT
    first.ROOT=Path('SYNTHETIC mutated first-only root')
    second,_=_new_released_scope_inputs(live)
    assert first is not second and first.__dict__ is not second.__dict__
    assert second.ROOT==boundary.ROOT and first.ROOT!=second.ROOT
    assert boundary._BOOTSTRAP_IMAGES[1] is image[1]
    assert boundary._BOOTSTRAP_IMAGES[0] is image[0]
    assert boundary._BOOTSTRAP_PLAN is compiled
    assert first._ncaaf_custody_predecessor is not second._ncaaf_custody_predecessor
    assert first._ncaaf_custody_predecessor.__code__ is second._ncaaf_custody_predecessor.__code__
    assert all(type(value)is bytes for value in image[0]+image[1])
    # The public legacy adapters still require their declared read view; do not
    # confuse an unowned direct call with a fixture-generation failure.
    req=request('test_football_document_io_scope','test_exact_policy_and_ordered_ci_merge')
    with boundary.InputOwner(boundary.resolve_context(req)) as owner:
        boundary.apply_selected_inputs(owner.plan,req,owner.patch)
        for row in boundary._BOOTSTRAP_GENERATIONS:
            fn=getattr(second,row[1])
            assert fn.__name__!='normalized'
            assert boundary.sha(fn(boundary._BOOTSTRAP_IMAGES[1][boundary._BOOTSTRAP_GENERATIONS.index(row)]))==row[4]
    assert Path.open is opened and Path.read_bytes is read
    assert boundary._bootstrap_immutable_size((boundary._BOOTSTRAP_IMAGES,boundary._BOOTSTRAP_PLAN))<=boundary._BOOTSTRAP_LIMIT


@pytest.mark.parametrize('attack',['source','binding','generation','helper','cached_image','catalog','cached_plan','compiler'])
def test_warm_bootstrap_reauthenticates_every_input(monkeypatch,attack,released_bootstrap_owner):
    from copy import deepcopy
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    live=boundary.canonical_guard();scope,binding=live._ci_fixture_module()
    scope_source=boundary.authenticated_predecessor(scope,binding)
    source=live._ncaaf_future_base_source(scope_source)
    if boundary._BOOTSTRAP_IMAGES is None:_new_released_scope_inputs(live)
    physical=boundary.physical_bytes
    if attack in {'source','helper'}:
        target=boundary.GUARD if attack=='source' else 'tests/ncaaf_future_scope_inputs.py'
        monkeypatch.setattr(boundary,'physical_bytes',lambda path:physical(path)+b'UNREVIEWED' if Path(path)==boundary.ROOT/target else physical(path))
    elif attack=='binding':binding=deepcopy(binding);binding['base']='0'*40
    elif attack=='generation':source+=b'UNREVIEWED'
    elif attack=='cached_image':
        key,images=boundary._BOOTSTRAP_IMAGES
        monkeypatch.setattr(boundary,'_BOOTSTRAP_IMAGES',(key,(images[0]+b'UNREVIEWED',)+images[1:]))
    elif attack=='catalog':monkeypatch.setattr(boundary,'_BOOTSTRAP_GENERATIONS',boundary._BOOTSTRAP_GENERATIONS[:-1])
    elif attack=='cached_plan':
        plan=boundary._BOOTSTRAP_PLAN
        assert plan is not None
        wrong=compile('SYNTHETIC_WRONG=True','<SYNTHETIC tampered plan>','exec')
        monkeypatch.setattr(boundary,'_BOOTSTRAP_PLAN',plan[:3]+(wrong,)+plan[4:])
    else:
        def reject(*args,**kwargs):raise ValueError('SYNTHETIC_COMPILER_REMAINS_LIVE')
        monkeypatch.setattr(boundary,'compile',reject,raising=False)
    import types
    module=types.ModuleType('SYNTHETIC no rejected bootstrap execution')
    module.__file__=str(boundary.ROOT/boundary.GUARD)
    with pytest.raises(ValueError):boundary.execute_released_bootstrap(source,module,live,binding)
    assert not hasattr(module,'_ncaaf_pilot_seeded_custody_parent')


def test_declared_legacy_reader_precedes_bootstrap_and_agrees_across_read_apis(monkeypatch):
    import importlib
    consumer=importlib.import_module('tests.test_dfs_scope_policy')
    req=SimpleNamespace(module=consumer,node=SimpleNamespace(originalname='test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    plan=boundary.resolve_context(req)
    native,read,opened=consumer.guard,Path.read_bytes,Path.open
    images=boundary._BOOTSTRAP_IMAGES
    def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_WRONG_READER_GENERATION')
    monkeypatch.setattr(boundary,'execute_released_bootstrap',forbidden)
    expected=boundary._LEGACY_READER_SOURCES['legacy-drive-after-v1']['sha256']
    with boundary.InputOwner(plan) as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        path=boundary.ROOT/boundary.GUARD
        assert boundary.sha(path.read_bytes())==expected
        with path.open('rb') as stream:assert boundary.sha(stream.read())==expected
        assert boundary.sha(path.read_text(encoding='utf-8').encode())==expected
        assert not hasattr(consumer.guard,'_ncaaf_custody_predecessor')
        assert hasattr(consumer.guard,'_drive_previous_guard_source')
        assert boundary._BOOTSTRAP_IMAGES is images
    assert consumer.guard is native and Path.read_bytes is read and Path.open is opened


@pytest.mark.parametrize('inner_native',[False,True])
def test_declared_guard_generations_restore_nested_views(inner_native):
    outer=request('test_dfs_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception')
    inner=(request('test_ci_scheduling_fixture_scope','test_unchanged_native_context_assertion_receives_no_projection')
           if inner_native else request('test_football_document_io_scope','test_exact_policy_and_ordered_ci_merge'))
    path=boundary.ROOT/boundary.GUARD
    physical=boundary.sha(boundary.physical_bytes(path));read,opened=Path.read_bytes,Path.open
    expected=boundary._LEGACY_READER_SOURCES['legacy-drive-after-v1']['sha256']
    with boundary.InputOwner(boundary.resolve_context(outer)) as owner:
        boundary.apply_selected_inputs(owner.plan,outer,owner.patch)
        assert boundary.sha(path.read_bytes())==expected
        with boundary.InputOwner(boundary.resolve_context(inner)) as child:
            boundary.apply_selected_inputs(child.plan,inner,child.patch)
            wanted=physical if inner_native else boundary._BOOTSTRAP_SOURCE
            assert boundary.sha(path.read_bytes())==wanted
            with path.open('rb') as stream:assert boundary.sha(stream.read())==wanted
        assert boundary.sha(path.read_bytes())==expected
    assert Path.read_bytes is read and Path.open is opened


@pytest.mark.parametrize('attack',['body','rehashed_body','generation','trailing_stream','blob','size'])
def test_complete_reader_preimages_reject_altered_or_rehashed_rows(attack):
    import base64,zlib,hashlib
    binding=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    row=binding['historical_reader_sources']['legacy-drive-after-v1']
    value=boundary.decode_reader_source(row)
    if attack in {'body','rehashed_body'}:
        changed=value+b'\n# SYNTHETIC source mutation\n'
        row['compressed_b85']=base64.b85encode(zlib.compress(changed)).decode()
        if attack=='rehashed_body':
            row.update(sha256=boundary.sha(changed),bytes=len(changed),git_blob=hashlib.sha1(b'blob '+str(len(changed)).encode()+b'\0'+changed).hexdigest())
    elif attack=='generation':row['source_generation']='SYNTHETIC_UNREGISTERED_GENERATION'
    elif attack=='trailing_stream':row['compressed_b85']=base64.b85encode(zlib.compress(value)+zlib.compress(value)).decode()
    elif attack=='blob':row['git_blob']='0'*40
    else:row['bytes']+=1
    with pytest.raises(ValueError,match='HISTORICAL_READER_SOURCE_CHANGED'):
        boundary.decode_reader_source(row)


def test_legacy_reader_construction_is_fresh_and_rechecks_warm_physical_inputs(monkeypatch):
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    req=request('test_dfs_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception')
    with boundary.InputOwner(boundary.resolve_context(req)):
        live=boundary.canonical_guard()
        first,_=_new_released_scope_inputs(live);first.ROOT=Path('SYNTHETIC MUTATED MODULE ONLY')
        second,_=_new_released_scope_inputs(live)
        assert first is not second and second.ROOT==boundary.ROOT
        native=boundary.physical_bytes
        for relative in (boundary.GUARD,boundary.BINDING):
            with monkeypatch.context() as patch:
                patch.setattr(boundary,'physical_bytes',lambda path:native(path)+b'UNREVIEWED' if Path(path)==boundary.ROOT/relative else native(path))
                with pytest.raises(ValueError):_new_released_scope_inputs(live)


def test_declared_reader_compile_failure_restores_owned_methods_and_modules(monkeypatch):
    import builtins,sys
    req=request('test_dfs_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception')
    native=boundary.canonical_guard();read,opened=Path.read_bytes,Path.open
    current=sys.modules.get('scripts.check_launch_change_scope');compile_native=builtins.compile
    def fail_selected(source,filename,*args,**kwargs):
        if isinstance(source,bytes) and len(source)==170111 and Path(filename)==boundary.ROOT/boundary.GUARD:
            raise RuntimeError('SYNTHETIC_DECLARED_READER_COMPILE')
        return compile_native(source,filename,*args,**kwargs)
    monkeypatch.setattr(builtins,'compile',fail_selected)
    with pytest.raises(RuntimeError,match='SYNTHETIC_DECLARED_READER_COMPILE'):
        with boundary.InputOwner(boundary.resolve_context(req)) as owner:
            boundary.apply_selected_inputs(owner.plan,req,owner.patch)
    assert Path.read_bytes is read and Path.open is opened
    assert sys.modules.get('scripts.check_launch_change_scope') is current
    assert boundary.canonical_guard() is native


@pytest.mark.parametrize('consumer',['test_ci_timing_profile','test_dfs_scope_policy','test_football_document_io_scope'])
def test_declared_configuration_views_agree_before_consumption_and_restore(consumer):
    req=request(consumer);plan=boundary.resolve_context(req)
    read,opened=Path.read_bytes,Path.open
    live=boundary.canonical_guard();records=json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)
    with boundary.InputOwner(plan) as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        for relative in ('.github/workflows/ci.yml','scripts/ci_test_file_costs_v1.json'):
            path=boundary.ROOT/relative;value=path.read_bytes()
            expected=(boundary.capacity_input_records(plan)[relative]['before_sha256']
                      if relative.endswith('ci_test_file_costs_v1.json') else records[relative]['before_sha256'])
            assert boundary.sha(value)==expected
            with path.open('rb') as stream:assert stream.read()==value
            assert path.read_text(encoding='utf-8').encode()==value
        assert boundary._DECLARED_READ_DEPTH.get()==0
    assert Path.read_bytes is read and Path.open is opened
    assert boundary._DECLARED_READ_DEPTH.get()==0


@pytest.mark.parametrize('failure',[ValueError,KeyboardInterrupt])
def test_consistent_read_ownership_restores_after_rejection_and_interruption(failure):
    req=request('test_ci_timing_profile');plan=boundary.resolve_context(req)
    read,opened=Path.read_bytes,Path.open;target=boundary.ROOT/'scripts/ci_test_file_costs_v1.json'
    def reject(path):
        if path==target:raise failure('SYNTHETIC_DECLARED_READ_FAILURE')
        return read(path)
    with boundary.InputOwner(plan) as owner:
        owner.patch.setattr(Path,'read_bytes',reject)
        boundary.install_consistent_read_apis(plan,owner.patch,boundary.canonical_guard())
        for call in (target.read_bytes,lambda:target.open('rb')):
            with pytest.raises(failure,match='SYNTHETIC_DECLARED_READ_FAILURE'):call()
            assert boundary._DECLARED_READ_DEPTH.get()==0
            assert boundary._DECLARED_OPEN_DEPTH.get()==0
    assert Path.read_bytes is read and Path.open is opened
    assert boundary._DECLARED_READ_DEPTH.get()==0
    assert boundary._DECLARED_OPEN_DEPTH.get()==0


def test_recorded_afterimage_operation_selects_before_any_fixture_namespace(monkeypatch):
    from tests import ncaaf_future_scope_inputs
    from scripts import ncaaf_reviewed_target_scope
    req=request('test_ncaaf_reviewed_target_scope','test_current_sources_have_exact_preimages_and_unknown_bytes_stay_unknown')
    plan=boundary.resolve_context(req)
    assert plan.generation=='recorded-source-afterimages'
    with pytest.raises(ValueError,match='HISTORICAL_CONTEXT_GENERATION_CHANGED'):
        boundary.authenticate_plan(replace(plan,generation='future'))
    def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_UNREQUESTED_FIXTURE_NAMESPACE')
    monkeypatch.setattr(ncaaf_future_scope_inputs,'_new_released_scope_inputs',forbidden)
    native=boundary.canonical_guard();read,opened=Path.read_bytes,Path.open
    binding=json.loads(boundary.physical_bytes(boundary.ROOT/ncaaf_reviewed_target_scope.BINDING))
    with boundary.InputOwner(plan)as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        path=boundary.ROOT/boundary.GUARD;value=path.read_bytes()
        with path.open('rb')as stream:assert stream.read()==value
        prior=ncaaf_reviewed_target_scope.previous_source(boundary.GUARD,value,binding)
        assert boundary.sha(prior)==binding['source_reconstructions'][boundary.GUARD]['before_sha256']
        assert boundary.canonical_guard()is native
    assert Path.read_bytes is read and Path.open is opened


def test_recorded_profile_afterimage_avoids_an_unrequested_fixture_namespace(monkeypatch):
    from tests import ncaaf_future_scope_inputs
    from tests import test_ncaaf_reviewed_target_timing as consumer
    req=request('test_ncaaf_reviewed_target_timing','test_five_live_costs_preserve_historical_measurement_meanings')
    plan=boundary.resolve_context(req)
    assert plan.generation=='recorded-source-afterimages'
    def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_UNREQUESTED_PROFILE_NAMESPACE')
    monkeypatch.setattr(ncaaf_future_scope_inputs,'_new_released_scope_inputs',forbidden)
    read,opened=Path.read_bytes,Path.open
    with boundary.InputOwner(plan)as owner:
        boundary.apply_selected_inputs(plan,req,owner.patch)
        path=boundary.ROOT/'scripts/ci_test_file_costs_v1.json';value=path.read_bytes()
        with path.open('rb')as stream:assert stream.read()==value
        consumer.test_five_live_costs_preserve_historical_measurement_meanings()
    assert Path.read_bytes is read and Path.open is opened


def test_complete_direct_reader_family_selects_authenticated_operations():
    binding=json.loads(boundary.physical_bytes(boundary.ROOT/boundary.BINDING))
    rows=binding['private_delivery_repair']['direct_recorded_reader_family']
    assert len(rows)==5 and len({(row['path'],row['function'])for row in rows})==5
    for row in rows:
        source=boundary.physical_bytes(boundary.ROOT/row['path'])
        assert boundary.sha(source)==row['source_sha256']
        req=request(Path(row['path']).stem,row['function'])
        plan=boundary.resolve_context(req)
        assert plan.generation==row['generation']


def _real_git_reader(repo):
    import subprocess
    def git(*args,check=True):
        result=subprocess.run(['git',*args],cwd=module.ROOT,capture_output=True,check=check)
        return result.stdout.decode().strip()
    def git_bytes(*args):return subprocess.check_output(['git',*args],cwd=module.ROOT)
    def exists_at(revision,path):
        return subprocess.run(['git','cat-file','-e',str(revision)+':'+str(path)],cwd=module.ROOT,capture_output=True).returncode==0
    module=SimpleNamespace(ROOT=repo,git=git,git_bytes=git_bytes,exists_at=exists_at)
    return module


def _prepare_stream_repository(path):
    import subprocess
    path.mkdir()
    for args in (('init','-q'),('config','user.name','Offline Test'),('config','user.email','offline@example.invalid'),('config','core.autocrlf','false')):
        subprocess.run(['git',*args],cwd=path,check=True,capture_output=True)
    (path/'original.txt').write_bytes(b'first\n')
    subprocess.run(['git','add','--all'],cwd=path,check=True,capture_output=True)
    subprocess.run(['git','commit','-qm','synthetic first'],cwd=path,check=True,capture_output=True)
    return _real_git_reader(path)


@pytest.mark.parametrize('failure',[None,ValueError,KeyboardInterrupt])
def test_owned_object_stream_observes_mutations_and_closes_after_every_exit(tmp_path,monkeypatch,failure):
    from scripts import scope_git_reads
    module=_prepare_stream_repository(tmp_path/'repository')
    native=module.git_bytes;observed=[]
    real=scope_git_reads.ObjectStream
    class Track(real):
        def _start(self):
            super()._start()
            if self.process not in observed:observed.append(self.process)
    monkeypatch.setattr(scope_git_reads,'ObjectStream',Track)
    plan=boundary.resolve_context(request('test_ncaaf_coverage_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    read,opened=Path.read_bytes,Path.open
    def run():
        with boundary.InputOwner(plan):
            boundary.install_owned_git_stream(module)
            assert module.git_bytes.__globals__ is native.__globals__
            assert module.git_bytes('show','HEAD:original.txt')==b'first\n'
            assert module.exists_at('HEAD','original.txt') and not module.exists_at('HEAD','missing.txt')
            assert module.git('rev-parse','HEAD:original.txt')
            assert len(observed)==1
            (module.ROOT/'original.txt').write_bytes(b'second\n')
            module.git('add','--all');module.git('commit','-qm','synthetic second')
            assert module.git_bytes('show','HEAD:original.txt')==b'second\n'
            assert len(observed)==1
            with pytest.raises(Exception):module.git_bytes('show','HEAD:missing.txt')
            if failure is not None:raise failure('SYNTHETIC_OWNER_INTERRUPTION')
    if failure is None:run()
    else:
        with pytest.raises(failure,match='SYNTHETIC_OWNER_INTERRUPTION'):run()
    assert module.git_bytes is native and Path.read_bytes is read and Path.open is opened
    assert observed and all(process.poll() is not None for process in observed)


def test_nested_object_stream_owners_do_not_share_processes_or_leave_views(tmp_path,monkeypatch):
    from scripts import scope_git_reads
    module=_prepare_stream_repository(tmp_path/'repository')
    real=scope_git_reads.ObjectStream;streams=[]
    class Track(real):
        def __init__(self,*args):super().__init__(*args);streams.append(self)
    monkeypatch.setattr(scope_git_reads,'ObjectStream',Track)
    plan=boundary.resolve_context(request('test_ncaaf_coverage_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    native=module.git_bytes
    with boundary.InputOwner(plan):
        boundary.install_owned_git_stream(module)
        assert module.git_bytes('show','HEAD:original.txt')==b'first\n'
        parent=module.git_bytes
        with boundary.InputOwner(plan):
            boundary.install_owned_git_stream(module)
            assert module.git_bytes('show','HEAD:original.txt')==b'first\n'
            assert len(streams)==2 and streams[0] is not streams[1]
        assert module.git_bytes is parent and streams[1].process is None
        assert module.git_bytes('show','HEAD:original.txt')==b'first\n'
    assert module.git_bytes is native and all(stream.process is None for stream in streams)


def test_owned_stream_rejects_changed_physical_reader_before_import_or_launch(monkeypatch):
    plan=boundary.resolve_context(request('test_ncaaf_coverage_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    actual=boundary.physical_bytes
    def altered(path):
        return b'changed helper' if Path(path)==boundary.ROOT/'scripts/scope_git_reads.py' else actual(path)
    with boundary.InputOwner(plan):
        monkeypatch.setattr(boundary,'physical_bytes',altered)
        with pytest.raises(ValueError,match='HISTORICAL_GIT_READER_SOURCE_CHANGED'):
            boundary.install_owned_git_stream(SimpleNamespace())


@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_warmed_object_stream_rejects_missing_or_corrupt_required_objects(tmp_path,damage):
    import subprocess
    import stat
    module=_prepare_stream_repository(tmp_path/'repository')
    identity=module.git('rev-parse','HEAD:original.txt')
    path=module.ROOT/'.git/objects'/identity[:2]/identity[2:]
    plan=boundary.resolve_context(request('test_ncaaf_coverage_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    native=module.git_bytes
    with pytest.raises((RuntimeError,subprocess.CalledProcessError,BaseExceptionGroup)):
        with boundary.InputOwner(plan):
            boundary.install_owned_git_stream(module)
            assert module.git_bytes('show','HEAD:original.txt')==b'first\n'
            path.chmod(path.stat().st_mode|stat.S_IWUSR)
            if damage=='missing':path.unlink()
            else:path.write_bytes(b'corrupt required object')
            module.git_bytes('show','HEAD:original.txt')
    assert module.git_bytes is native


@pytest.mark.parametrize('damage',[None,'workflow','binding'])
def test_current_ci_budget_uses_exact_authenticated_historical_edge(monkeypatch,damage):
    import base64,zlib
    from tests import ci_fixture_stability_inputs as inputs
    relative='.github/workflows/ci.yml';path=boundary.ROOT/relative
    plan=boundary.resolve_context(request('test_ncaaf_coverage_scope_policy','test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception'))
    current=boundary.physical_bytes(path)
    expected=zlib.decompress(base64.b85decode(inputs.CAPACITY_SOURCE_RECORDS[relative]['compressed']))
    opened,read=Path.open,Path.read_bytes
    with boundary.InputOwner(plan) as owner:
        boundary.install_capacity_input_view(plan,owner.patch)
        assert path.read_bytes()==expected
        assert boundary.physical_bytes(path)==current
        assert b'timeout-minutes: 40' in current and current!=expected
        if damage is not None:
            actual=boundary.physical_bytes
            def altered(p):
                value=actual(p)
                return value+b'UNREVIEWED' if Path(p)==(path if damage=='workflow' else boundary.ROOT/boundary.BINDING) else value
            monkeypatch.setattr(boundary,'physical_bytes',altered)
            with pytest.raises((ValueError,__import__('json').JSONDecodeError)):path.read_bytes()
    assert Path.open is opened and Path.read_bytes is read


def test_bootstrap_reuses_verified_bytes_through_live_predecessor_checks(monkeypatch,released_bootstrap_owner):
    import types
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    live=boundary.canonical_guard()
    first,_=_new_released_scope_inputs(live)
    scope,binding=live._ci_fixture_module()
    predecessor=boundary.authenticated_predecessor(scope,binding)
    source=live._ncaaf_future_base_source(predecessor)
    checked=[];original=scope.previous_guard
    def observed(value,current,*,predecessor=None):
        checked.append(predecessor)
        return original(value,current,predecessor=predecessor)
    monkeypatch.setattr(scope,'previous_guard',observed)
    monkeypatch.setattr(live,'_ci_fixture_module',lambda:(scope,binding))
    second,_=_new_released_scope_inputs(live)
    assert checked and all(value is predecessor for value in checked)
    assert first is not second and first.__dict__ is not second.__dict__
    assert first._ncaaf_custody_predecessor is not second._ncaaf_custody_predecessor
    key,image=boundary._PREDECESSOR_IMAGE
    monkeypatch.setattr(boundary,'_PREDECESSOR_IMAGE',(key,image+b'UNREVIEWED'))
    rejected=types.ModuleType('SYNTHETIC_REJECTED_BOOTSTRAP')
    rejected.__file__=first.__file__
    with pytest.raises(ValueError,match='PREVIOUS_GUARD_LOGIC_CHANGED'):
        boundary.execute_released_bootstrap(source,rejected,live,binding)
    assert not hasattr(rejected,'_ncaaf_custody_predecessor')
