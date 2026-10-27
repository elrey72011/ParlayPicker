"""Exact successor source protection and unchanged legacy/default rejection."""
from copy import deepcopy
import hashlib,json,subprocess,sys
from pathlib import Path
import pytest
from scripts import ncaaf_filtered_history_scope as scope
from app_core import ncaaf_filtered_history as filtered,ncaaf_pilot as pilot,ncaaf_response_custody as custody
from test_ncaaf_filtered_history import context

from tests.nfl_capacity_profile_inputs import historical_filtered_profile_generation

ROOT=Path(__file__).resolve().parents[1]


def copied_graph(tmp_path):
    binding=scope.authenticate(ROOT)
    for relative in set(binding['components'])|{scope.BINDING}:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(scope.physical(ROOT/relative))
    return binding


def test_source_graph_and_exact_scope_are_bound():
    b=scope.authenticate(ROOT)
    assert b['proposal_sha256']=='4ba2ecdeaadc156070e3b402182a7eaa6cd4f6d4ea015dabb6aa2028a154f5a0'
    assert b['entrypoint_amendment_sha256']=='4b8ca656b600f500ffef8dffc3d09a9063e35249c66a0803bb2a8bbeb14cf4cd'
    assert len(b['implementation_paths'])==29 and scope.POLICY not in b['implementation_paths']
    assert b['sqlite_amendment_sha256']=='03f4d72f0e78207ed4f9c7ee974234318cb6a6f3af1599bd5a2ed1b75dd8608a'
    assert b['fresh_scoring_design_sha256']=='600e9728ca5e5aa397b8d5faff59ead1a271178205ca20bdf2b92076d81a7f08'
    assert b['shared_boundary_amendment_sha256']=='162061ae1a7240680f915a9f16ea585cff70a619a69c49d99a0e46c14600a53b'
    assert b['base']==scope.BASE and b['base_tree']==scope.BASE_TREE


def test_legacy_image_reuse_keeps_original_rejections_and_fresh_physical_authentication(monkeypatch,tmp_path):
    from tests import historical_scope_input_boundary as boundary
    original,binding=boundary.canonical_guard()._ci_fixture_module()
    copied_graph(tmp_path)
    cache={};monkeypatch.setattr(scope,'_LEGACY_GUARD_IMAGES',cache)
    checked=scope.preserve_guard_reader(original.previous_guard.__wrapped__,tmp_path)
    source=scope.physical(tmp_path/boundary.GUARD)
    first=checked(source,binding)
    assert scope.sha(first)==binding['predecessor_guard']['sha256'] and cache
    assert checked(source,binding) is first
    assert len(cache)<=8 and sum(len(v)+4096 for v in cache.values())<=scope._LEGACY_IMAGE_LIMIT
    with pytest.raises(ValueError,match='^PREVIOUS_GUARD_LOGIC_CHANGED$'):
        checked(source,binding,predecessor=b'SYNTHETIC corrupt explicit predecessor')
    key=next(iter(cache));value=cache[key]
    try:
        cache[key]=b'SYNTHETIC corrupt cached predecessor'
        with pytest.raises(ValueError,match='^PREVIOUS_GUARD_LOGIC_CHANGED$'):checked(source,binding)
    finally:cache[key]=value
    wrong=deepcopy(binding);wrong['predecessor_guard']['source_b85']+='0'
    with pytest.raises(ValueError,match='^REVIEWED_BINDINGS_CHANGED$'):checked(source,wrong)
    def forbidden(*args,**kwargs):raise RuntimeError('SYNTHETIC_DECODER_CALLBACK')
    with monkeypatch.context()as changed_codec:
        changed_codec.setattr(scope.base64,'b85decode',forbidden)
        with pytest.raises(RuntimeError,match='^SYNTHETIC_DECODER_CALLBACK$'):checked(source,binding)
    changed=tmp_path/'app_core/ncaaf_pilot.py';before=changed.read_bytes()
    try:
        changed.write_bytes(before+b'\n# SYNTHETIC warm alteration\n')
        with pytest.raises(ValueError,match='^NCAAF_FILTERED_SOURCE_CHANGED$'):checked(source,binding)
    finally:changed.write_bytes(before)
    assert checked(source,binding)==first


def test_capacity_image_reuse_rechecks_cached_bytes_and_restores_views(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from tests import historical_scope_input_boundary as boundary
    module=importlib.import_module('test_ci_six_shards')
    request=SimpleNamespace(module=module,node=SimpleNamespace(originalname='test_two_completed_file_costs_are_advisory_and_attributable'))
    plan=boundary.resolve_context(request)
    assert plan.role is boundary.Role.HISTORICAL
    cache={};monkeypatch.setattr(boundary,'_CAPACITY_IMAGES',cache)
    read,opened=Path.read_bytes,Path.open;incoming=tuple(boundary._ACTIVE)
    with boundary.InputOwner(plan)as owner:
        records=boundary.capacity_input_records(plan)
        boundary.install_capacity_input_view(plan,owner.patch)
        path=ROOT/'scripts/ci_test_file_costs_v1.json';first=path.read_bytes()
        assert boundary.sha(first)==records['scripts/ci_test_file_costs_v1.json']['before_sha256']
        assert path.read_bytes()==first and cache
        assert len(cache)<=8 and sum(len(v)+4096 for v in cache.values())<=boundary._CAPACITY_IMAGE_LIMIT
        saved=dict(cache)
        try:
            for key in cache:cache[key]=b'SYNTHETIC corrupt cached capacity image'
            with pytest.raises(ValueError,match='^CI_CAPACITY_AUTHENTICATED_PREDECESSOR_CHANGED$'):path.read_bytes()
        finally:cache.clear();cache.update(saved)
        with path.open('rb')as stream:assert stream.read()==first
        def forbidden(*args,**kwargs):raise RuntimeError('SYNTHETIC_DECODER_CALLBACK')
        with monkeypatch.context()as changed_codec:
            changed_codec.setattr(boundary.base64,'b85decode',forbidden)
            with pytest.raises(RuntimeError,match='^SYNTHETIC_DECODER_CALLBACK$'):path.read_bytes()
    assert Path.read_bytes is read and Path.open is opened and tuple(boundary._ACTIVE)==incoming


@pytest.mark.parametrize('relative',['scripts/check_launch_change_scope.py',
    'app_core/ncaaf_pilot.py','scripts/ncaaf_staged_acquisition.py'])
def test_capacity_adapter_preserves_unregistered_source_identity(relative):
    from tests import ci_fixture_stability_inputs as inputs
    raw=scope.physical(ROOT/relative)
    assert inputs.capacity_predecessor_source(relative,raw)==raw
    assert inputs.original_source(relative,raw)==raw


def test_capacity_adapter_preserves_the_registered_profile_transition():
    from tests import ci_fixture_stability_inputs as inputs
    relative='scripts/ci_test_file_costs_v1.json';raw=scope.physical(ROOT/relative)
    parent=scope.predecessor_source(ROOT,relative,raw)
    assert parent!=raw
    assert inputs.capacity_predecessor_source(relative,raw)==inputs.capacity_predecessor_source.__wrapped__(relative,parent)


def test_parent_view_leaves_guard_selection_to_the_declared_reader():
    import importlib
    from types import SimpleNamespace
    from tests import historical_scope_input_boundary as boundary
    module=importlib.import_module('test_football_document_io_scope')
    request=SimpleNamespace(module=module,node=SimpleNamespace(originalname='test_exact_successor_and_ordered_ci_merge'))
    plan=boundary.resolve_context(request)
    assert plan.role is boundary.Role.HISTORICAL
    opened,read=Path.open,Path.read_bytes;incoming=tuple(boundary._ACTIVE)
    guard=scope.physical(ROOT/boundary.GUARD);profile='scripts/ci_test_file_costs_v1.json'
    physical_profile=scope.physical(ROOT/profile)
    with boundary.InputOwner(plan)as owner:
        boundary.install_filtered_successor_input_view(plan,owner.patch)
        assert(ROOT/boundary.GUARD).read_bytes()==guard
        with(ROOT/boundary.GUARD).open('rb')as stream:assert stream.read()==guard
        assert(ROOT/boundary.GUARD).read_text(encoding='utf-8').encode()==guard
        assert(ROOT/profile).read_bytes()==scope.predecessor_source(ROOT,profile,physical_profile)
        assert boundary.physical_bytes(ROOT/profile)==physical_profile
    assert Path.open is opened and Path.read_bytes is read and tuple(boundary._ACTIVE)==incoming


@pytest.mark.parametrize('path',['app_core/ncaaf_filtered_history.py','app_core/ncaaf_fresh_history.py','app_core/ncaaf_pilot.py',
    'app_core/ncaaf_staged_acquisition.py','scripts/ncaaf_reviewed_target_operator.py','scripts/check_launch_change_scope.py',
    'scripts/ncaaf_private_refresh_scope.py','scripts/ncaaf_filtered_history_scope.py',
    'tests/historical_scope_input_boundary.py','scripts/ci_test_file_costs_v1.json',scope.BINDING])
def test_warm_source_or_binding_mutation_rejects(tmp_path,path):
    copied_graph(tmp_path);scope.authenticate(tmp_path)
    p=tmp_path/path;p.write_bytes(p.read_bytes()+b'\n# SYNTHETIC mutation\n')
    with pytest.raises(ValueError):scope.authenticate(tmp_path)


def test_physical_authentication_independent_of_projected_path_reads(monkeypatch,tmp_path):
    copied_graph(tmp_path)
    path=tmp_path/'app_core/ncaaf_pilot.py';original=path.read_bytes();path.write_bytes(original+b'\n# tamper\n')
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:original if p==path else read(p))
    with pytest.raises(ValueError):scope.authenticate(tmp_path)


def test_authenticated_predecessor_and_unknown_supplied_generation():
    b=scope.authenticate(ROOT);path='app_core/ncaaf_staged_acquisition.py';current=scope.physical(ROOT/path)
    old=scope.predecessor_source(ROOT,path,current)
    assert hashlib.sha256(old).hexdigest()==b['component_predecessors'][path]['before_sha256']
    assert scope.predecessor_source(ROOT,path,old)==old
    assert scope.predecessor_source(ROOT,path,b'SYNTHETIC unknown')==b'SYNTHETIC unknown'
    assert scope.physical(ROOT/path)==current


def test_unselected_default_custody_still_rejects_filtered_queries(monkeypatch,tmp_path):
    p,_=context(monkeypatch,tmp_path)
    for request in p['payload']['requests']:
        meta=dict(provider='cfbd',endpoint='games/teams',request_scope=request['params'],status=200,
            representation=custody.REPRESENTATION,content_type='application/json',complete=True,
            received_at='2026-10-17T18:00:01Z',receipt_clock_meaning=custody.RECEIPT_MEANING)
        with pytest.raises(ValueError):custody._metadata(meta)
    assert pilot.HttpsTransport(lambda _:None).filtered_history_policy is None


def test_native_launch_help_does_not_request_inputs_or_create_claims(tmp_path):
    # Public isolated forms; no execute, key, provider or custody operation.
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/ncaaf_staged_acquisition.py'),'--help'],text=True,capture_output=True,timeout=30)
    assert result.returncode==0 and 'readback' in result.stdout
    assert not list(tmp_path.iterdir())


def test_standalone_module_plan_uses_exact_filtered_inventory(monkeypatch,tmp_path):
    p,_=context(monkeypatch,tmp_path);path=tmp_path/'SYNTHETIC-plan.json'
    path.write_bytes(filtered.encode(p));libs=str(Path(pytest.__file__).resolve().parent.parent)
    code='import sys,runpy;sys.path[:0]='+repr([str(ROOT),libs])+';sys.argv='+repr(['ncaaf_staged_acquisition','plan',str(path)])+';runpy.run_module("scripts.ncaaf_staged_acquisition",run_name="__main__")'
    result=subprocess.run([sys.executable,'-I','-B','-c',code],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    value=json.loads(result.stdout)
    assert value['requests']==p['payload']['requests'] and value['network_requests']==value['credential_loads']==value['storage_operations']==0


def test_exact_intervening_refresh_generation_and_wrong_chain_rejection():
    from app_core import ncaaf_reviewed_target as target
    path='app_core/ncaaf_owner_spread.py';b=scope.authenticate(ROOT)
    expected=target.HISTORICAL_COMPONENTS[path]['after_sha256']
    assert scope.predecessor_generation(ROOT,b,path,expected)==b['component_predecessors'][path]['before_sha256']
    assert target.source_hash(path)==hashlib.sha256(scope.physical(ROOT/path)).hexdigest()
    assert target.source_hash(path,historical=True)==target.HISTORICAL_COMPONENTS[path]['before_sha256']
    with pytest.raises(ValueError):scope.predecessor_generation(ROOT,b,path,'0'*64)


def test_private_public_manifest_path_copies_the_complete_selected_graph(tmp_path):
    from scripts import ncaaf_private_refresh_scope as native
    assert native.BINDING==scope.BINDING
    b=copied_graph(tmp_path)
    assert native.authenticate(tmp_path)==b
    p=tmp_path/native.BINDING;p.write_bytes(p.read_bytes()+b'\nSYNTHETIC altered binding\n')
    with pytest.raises(ValueError):native.authenticate(tmp_path)


def test_legacy_ci_input_generation_is_separate_from_current_physical_bytes():
    from scripts import check_launch_change_scope as native,ci_fixture_stability_scope as historical
    current,binding=native._ci_current_source_input()
    assert current==scope.physical(ROOT/'scripts/check_launch_change_scope.py')
    selected,old_binding=native._ci_authenticated_source_input()
    assert old_binding==binding and selected!=current
    assert selected==scope.predecessor_source(ROOT,'scripts/check_launch_change_scope.py',current)
    adapted,_=native._ci_fixture_module()
    assert historical.previous_guard(selected,binding)==adapted.previous_guard(current,binding)


def test_immutable_predecessor_reuse_reauthenticates_and_checks_cached_bytes(monkeypatch,tmp_path):
    monkeypatch.setattr(scope,'_PREDECESSOR_IMAGES',{})
    b=copied_graph(tmp_path);path='scripts/check_launch_change_scope.py';raw=scope.physical(tmp_path/path)
    native=scope.base64.b85decode;calls=[]
    def observed(value):calls.append(1);return native(value)
    monkeypatch.setattr(scope.base64,'b85decode',observed)
    first=scope.predecessor_source(tmp_path,path,raw)
    assert scope.predecessor_source(tmp_path,path,raw) is first and len(calls)==1
    assert hashlib.sha256(first).hexdigest()==b['component_predecessors'][path]['before_sha256']
    changed=tmp_path/'app_core/ncaaf_pilot.py';original=changed.read_bytes()
    changed.write_bytes(original+b'\n# SYNTHETIC warm mutation\n')
    with pytest.raises(ValueError):scope.predecessor_source(tmp_path,path,raw)
    changed.write_bytes(original)
    key=next(iter(scope._PREDECESSOR_IMAGES));scope._PREDECESSOR_IMAGES[key]=first+b'UNREVIEWED'
    with pytest.raises(ValueError):scope.predecessor_source(tmp_path,path,raw)


@pytest.mark.parametrize('path',sorted(scope.NORMALIZED))
def test_supplied_older_normalized_source_stays_with_its_strict_reader(path):
    scope.authenticate(ROOT)
    raw=b'SYNTHETIC older or unknown supplied representation without current stamp'
    before=dict(scope._PREDECESSOR_IMAGES)
    assert scope.predecessor_source(ROOT,path,raw)==raw
    assert scope._PREDECESSOR_IMAGES==before


def test_two_missing_costs_preserve_every_authenticated_prior_row():
    path='scripts/ci_test_file_costs_v1.json'
    current=scope.physical(ROOT/path)
    prior=scope.predecessor_source(ROOT,path,current)
    reviewed=scope.advisory_profile_before(ROOT,current)
    old,new=json.loads(prior),json.loads(reviewed)
    assert len(old['files'])==508 and len(new['files'])==510
    assert new['files'][:-2]==old['files']
    assert all(new['sources'][key]==value for key,value in old['sources'].items())
    assert all(new[key]==value for key,value in old.items()if key not in {'files','sources'})
    assert {row['path'] for row in new['files'][-2:]}=={
        'tests/test_ncaaf_filtered_history.py','tests/test_ncaaf_private_normal_refresh.py'}
    for row in new['files'][-2:]:
        assert row['variance_multiplier']==1.15
        assert row['seconds']==round(row['basis_seconds']*1.15,6)
        assert new['sources'][row['references'][0]]['complete_file'] is True
    # Frozen508 and the exact reviewed510 predecessor remain distinct from the
    # current nondecreasing534-row advisory estimate.
    assert scope.sha(reviewed)=='3facc49a21124b179b5546d829389381122ef36e3cc70ee90650af5b658d112a'
    live=json.loads(scope.fresh_advisory_profile_before(ROOT,current));assert len(live['files'])==534
    indexed={r['path']:r for r in live['files']}
    assert [r['path']for r in live['files'][:510]]==[r['path']for r in new['files']]
    assert all(live['sources'][k]==v for k,v in new['sources'].items())
    assert all(live[k]==v for k,v in new.items()if k not in{'files','sources'})
    for row in new['files']:
        actual=indexed[row['path']]
        if actual==row:continue
        added=max(0,actual['profiled_tests']-row['profiled_tests'])
        floor=(row['seconds']+added*new['fallback']['seconds_per_test'])if row['seconds']is not None else max(
            new['fallback']['minimum_file_seconds'],actual['profiled_tests']*new['fallback']['seconds_per_test'])
        assert actual['seconds']+1e-9>=floor
        assert actual['variance_multiplier']==1.15
        assert actual['seconds']+1e-9>=actual['basis_seconds']*1.15



def _boundary_request(module,node=''):
    from types import SimpleNamespace
    return SimpleNamespace(module=SimpleNamespace(__name__=module,
        __file__=str(ROOT/'tests'/(module+'.py'))),node=SimpleNamespace(originalname=node))


@pytest.mark.parametrize('failure',[None,RuntimeError,KeyboardInterrupt])
def test_filtered_parent_view_owns_reads_aliases_and_nested_restoration(failure):
    from tests import historical_scope_input_boundary as boundary,ci_fixture_stability_inputs as inputs
    request=_boundary_request('test_ncaaf_reviewed_target_scope',
        'test_current_sources_have_exact_preimages_and_unknown_bytes_stay_unknown')
    plan=boundary.resolve_context(request)
    native_request=_boundary_request('test_ncaaf_spread_boundaries')
    native=boundary.resolve_context(native_request)
    alternate=boundary.resolve_context(_boundary_request('test_ncaaf_chronology_scope'))
    assert plan.role is alternate.role is boundary.Role.HISTORICAL
    assert native.role is boundary.Role.LIVE
    read,opened,capacity=Path.read_bytes,Path.open,inputs.capacity_predecessor_source
    active=tuple(boundary._ACTIVE)
    paths=['app_core/ncaaf_owner_spread.py','scripts/ci_test_file_costs_v1.json']
    physical={path:boundary.physical_bytes(ROOT/path) for path in paths}
    expected={path:scope.predecessor_source(ROOT,path,physical[path]) for path in paths}
    assert all(physical[path]!=expected[path] for path in paths)
    try:
        with boundary.InputOwner(plan) as owner:
            assert boundary.install_filtered_successor_input_view(plan,owner.patch)
            owned_read,owned_open=Path.read_bytes,Path.open
            for path in paths:
                file=ROOT/path
                assert file.read_bytes()==expected[path]
                with file.open('rb') as stream:assert stream.read()==expected[path]
                assert file.read_text(encoding='utf-8').encode()==expected[path]
                assert boundary.physical_bytes(file)==physical[path]
            with boundary.InputOwner(native) as inner:
                assert not boundary.apply_selected_inputs(native,native_request,inner.patch)
                for path in paths:assert (ROOT/path).read_bytes()==physical[path]
            assert Path.read_bytes is owned_read and Path.open is owned_open
            with boundary.InputOwner(alternate) as inner:
                assert boundary.install_filtered_successor_input_view(alternate,inner.patch)
                for path in paths:assert (ROOT/path).read_bytes()==expected[path]
            assert Path.read_bytes is owned_read and Path.open is owned_open
            if failure:raise failure('SYNTHETIC owned-view exit')
    except BaseException as exc:
        assert failure is not None and type(exc) is failure
    assert Path.read_bytes is read and Path.open is opened
    assert inputs.capacity_predecessor_source is capacity
    assert tuple(boundary._ACTIVE)==active


def test_live_parent_route_never_consults_historical_successor(monkeypatch):
    from tests import historical_scope_input_boundary as boundary
    request=_boundary_request('test_ncaaf_spread_boundaries')
    plan=boundary.resolve_context(request)
    def forbidden():raise AssertionError('SYNTHETIC historical lookup in LIVE route')
    monkeypatch.setattr(boundary,'canonical_guard',forbidden)
    assert not boundary.install_filtered_successor_input_view(plan,None)


def test_bootstrap_selects_parent_self_bytes_and_rejects_warm_alteration(monkeypatch):
    from tests import historical_scope_input_boundary as boundary
    from tests.ncaaf_future_scope_inputs import _new_released_scope_inputs
    request=_boundary_request('test_football_document_io_scope','test_exact_policy_and_ordered_ci_merge')
    with boundary.InputOwner(boundary.resolve_context(request)) as owner:
        boundary.apply_selected_inputs(owner.plan,request,owner.patch)
        live=boundary.canonical_guard()
        first,_=_new_released_scope_inputs(live)
        assert first.ROOT==ROOT
        physical=boundary.physical_bytes
        relative='tests/historical_scope_input_boundary.py'
        with monkeypatch.context() as changed:
            changed.setattr(boundary,'physical_bytes',lambda path:
                physical(path)+b'\nSYNTHETIC unknown generation' if Path(path)==ROOT/relative else physical(path))
            with pytest.raises(ValueError):_new_released_scope_inputs(live)
        second,_=_new_released_scope_inputs(live)
        assert second is not first and second.ROOT==ROOT


def test_parent_layer_does_not_shadow_a_later_owned_open():
    import io
    from tests import historical_scope_input_boundary as boundary
    request=_boundary_request('test_ncaaf_reviewed_target_scope',
        'test_current_sources_have_exact_preimages_and_unknown_bytes_stay_unknown')
    plan=boundary.resolve_context(request)
    file=ROOT/'app_core/ncaaf_owner_spread.py'
    physical=boundary.physical_bytes(file);read,opened=Path.read_bytes,Path.open
    with boundary.InputOwner(plan) as owner:
        boundary.install_filtered_successor_input_view(plan,owner.patch)
        parent_open=Path.open
        def narrower(path,mode='r',*args,**kwargs):
            if path==file and mode=='rb':return io.BytesIO(b'SYNTHETIC exact inner generation')
            return parent_open(path,mode,*args,**kwargs)
        owner.patch.setattr(Path,'open',narrower)
        assert file.read_bytes()==b'SYNTHETIC exact inner generation'
        with file.open('rb') as stream:assert stream.read()==b'SYNTHETIC exact inner generation'
        assert boundary.physical_bytes(file)==physical
    assert Path.read_bytes is read and Path.open is opened


def test_legacy_fixture_adapter_is_exact_and_live_inputs_stay_current(historical_filtered_profile_generation):
    from tests import historical_scope_input_boundary as boundary
    from types import SimpleNamespace
    binding=scope.authenticate(ROOT);consumer='tests/test_ci_fixture_stability_scope.py'
    edge=binding['component_predecessors'][consumer]
    assert scope.sha(scope.physical(ROOT/consumer))==edge['after_sha256']
    assert scope.sha(boundary._authenticated_consumer_source(consumer))==edge['before_sha256']
    request=SimpleNamespace(module=SimpleNamespace(__name__='test_ci_fixture_stability_scope',__file__=str(ROOT/consumer)),node=SimpleNamespace(originalname='test_synthetic_exact_seal_and_ordered_merge'))
    plan=boundary.resolve_context(request)
    assert plan.role is boundary.Role.NATIVE and plan.generation=='current'
    assert plan.physical_consumer_sha256==edge['after_sha256']
    opened,read=Path.open,Path.read_bytes
    with boundary.InputOwner(plan)as owner:
        assert not boundary.apply_selected_inputs(plan,request,owner.patch)
        assert Path.open is opened and Path.read_bytes is read
        assert len(json.loads(scope.fresh_advisory_profile_before(ROOT,scope.physical(ROOT/'scripts/ci_test_file_costs_v1.json')))['files'])==534
        assert len(json.loads((ROOT/'scripts/ci_test_file_costs_v1.json').read_bytes())['files'])==535
        assert(ROOT/boundary.GUARD).read_bytes()==scope.physical(ROOT/boundary.GUARD)
    with pytest.raises(ValueError,match='HISTORICAL_CURRENT_INPUT_NOT_AUTHENTICATED'):
        boundary.reviewed_current_input('scripts/ci_test_file_costs_v1.json')


@pytest.mark.parametrize('relative',['tests/test_ci_fixture_stability_scope.py',scope.BINDING])
def test_warm_legacy_fixture_consumer_or_binding_mutation_rejects(tmp_path,relative):
    copied_graph(tmp_path);scope.authenticate(tmp_path)
    path=tmp_path/relative;path.write_bytes(path.read_bytes()+b'\n# SYNTHETIC MUTATION\n')
    with pytest.raises(ValueError,match='NCAAF_FILTERED_(SOURCE|BINDING)_CHANGED'):
        scope.authenticate(tmp_path)


def test_advisory_profile_before_rejects_unreviewed_supplied_bytes():
    raw=scope.physical(ROOT/'scripts/ci_test_file_costs_v1.json')
    assert scope.sha(scope.advisory_profile_before(ROOT,raw))=='3facc49a21124b179b5546d829389381122ef36e3cc70ee90650af5b658d112a'
    with pytest.raises(ValueError,match='NCAAF_FILTERED_PROFILE_PARENT_CHANGED'):
        scope.advisory_profile_before(ROOT,raw+b'\nUNREVIEWED')


def test_fresh_cost_normalization_authenticates_fixed_prior_rows_and_rejects_other_generations(historical_filtered_profile_generation):
    current=scope.physical(ROOT/'scripts/ci_test_file_costs_v1.json')
    prior=scope.fresh_advisory_profile_before(ROOT,current)
    assert scope.sha(prior)=='253a8551db604bd01c5304b7f74e61faaf9bb221ec5ab2b3ae6a2f05a1713bb4'
    old,new=json.loads(prior),json.loads(current)
    assert len(old['files'])==534 and len(new['files'])==535
    assert new['files'][:-1]==old['files']
    assert {k:v for k,v in new['sources'].items()if k not in new['files'][-1]['references']}==old['sources']
    assert all(new[k]==old[k]for k in old if k not in{'files','sources'})
    for unknown in (prior,current+b'\n',scope.advisory_profile_before(ROOT,current)):
        with pytest.raises(ValueError,match='PROFILE_PARENT_CHANGED'):
            scope.fresh_advisory_profile_before(ROOT,unknown)


@pytest.mark.parametrize('module',['test_ncaaf_staged_scope','test_ncaaf_terms_scope'])
def test_explicit_reader_api_normalizes_parent_before_original_mutable_record_authentication(module,monkeypatch):
    from tests import historical_scope_input_boundary as boundary,ci_fixture_stability_inputs as source
    node=module+'::test_exact_predecessor_source_reconstruction_and_unknown_edits'
    record=source.EXACT_READER_INPUTS[node];raw=boundary.physical_bytes(ROOT/record['path'])
    reader=source.reader_source # An alias retains the original mutable globals.
    active=tuple(boundary._ACTIVE);read,opened=Path.read_bytes,Path.open
    assert reader.__wrapped__.__globals__ is vars(source)
    first=reader(record['path'],raw,node)
    assert hashlib.sha256(first).hexdigest()==record['input_sha256']
    assert reader(record['path'],raw,node)==first
    for key,value,code in [('binding_sha256','0'*64,'HISTORICAL_READER_BINDING_CHANGED'),
                           ('input_sha256','0'*64,'HISTORICAL_READER_INPUT_CHANGED'),
                           ('input_git_blob','0'*40,'HISTORICAL_READER_INPUT_CHANGED'),
                           ('input_bytes',record['input_bytes']+1,'HISTORICAL_READER_INPUT_CHANGED')]:
        with monkeypatch.context()as patch:
            patch.setitem(source.EXACT_READER_INPUTS,node,dict(record,**{key:value}))
            with pytest.raises(ValueError,match=code):reader(record['path'],raw,node)
        assert source.EXACT_READER_INPUTS[node] is record
    assert reader(record['path'],raw+b'UNREVIEWED',node)==raw+b'UNREVIEWED'
    assert reader(record['path'],raw,'SYNTHETIC::current_gate')==raw
    assert Path.read_bytes is read and Path.open is opened and tuple(boundary._ACTIVE)==active
