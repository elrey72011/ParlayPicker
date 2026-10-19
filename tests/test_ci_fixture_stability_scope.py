"""Real synthetic seals and live successor inputs; no repository history required."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import zlib
import pytest
from scripts import ci_fixture_stability_scope as scope


ROOT = Path(__file__).resolve().parents[1]


def raw_git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE, timeout=30)


def git(root, *args):
    return raw_git(root, *args).decode().strip()


@pytest.fixture(scope='session')
def sealed_template(tmp_path_factory):
    root = tmp_path_factory.mktemp('SYNTHETIC-ci-stability')/'repo'; root.mkdir()
    binding = json.loads((ROOT/scope.BINDING).read_bytes())
    from tests import ci_fixture_stability_inputs as inputs
    def write(path, value):
        file=root/path; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(value)
    def commit(message):
        git(root, 'add', '--all'); git(root, 'commit', '-qm', message)
        return git(root, 'rev-parse', 'HEAD')
    git(root, 'init', '-q'); git(root, 'config', 'user.name', 'SYNTHETIC CI stability')
    git(root, 'config', 'user.email', 'synthetic@example.invalid')
    git(root, 'config', 'core.autocrlf', 'false'); git(root, 'config', 'core.longpaths', 'true')
    git(root, 'config', 'gc.auto', '0'); git(root, 'config', 'maintenance.auto', 'false')
    current_guard = (ROOT/scope.GUARD).read_bytes().replace(b'\r\n', b'\n')
    write(scope.GUARD, scope.previous_guard(current_guard, binding))
    manifest='docs/paid-launch/launch-baseline-manifest.json'
    write(manifest, (ROOT/manifest).read_bytes().replace(b'\r\n', b'\n'))
    write('core/frozen.py', b'# SYNTHETIC immutable dependency\n')
    for path, row in inputs.RECORDS.items():
        before = zlib.decompress(base64.b85decode(row['compressed']))
        assert scope.sha(before) == row['before_sha256']
        write(path, before)
    base = commit('SYNTHETIC exact predecessor inputs')
    tree = git(root, 'rev-parse', 'HEAD^{tree}')
    b = deepcopy(binding); b['base']=base; b['base_tree']=tree; b['ordered_base_parents']=[]
    for path in b['paths']:
        if path in (scope.GUARD, scope.BINDING):continue
        value=(ROOT/path).read_bytes().replace(b'\r\n', b'\n')
        if path=='tests/ci_fixture_stability_inputs.py':
            value=value.replace(inputs.BASE.encode(),base.encode())
        write(path,value)
    b['reviewed_blobs']={path:scope.blob((root/path).read_bytes()) for path in b['paths'] if path not in (scope.GUARD,scope.BINDING)}
    write(scope.BINDING,scope.canonical(b))
    identity=re.search(rb"_CI_STABILITY_GUARD_SHA256='([0-9a-f]{64})'",current_guard).group(1)
    stamp=re.search(rb"_CI_STABILITY_BINDING_SHA256='([0-9a-f]{64})'",current_guard).group(1)
    new=current_guard.replace(stamp,scope.sha(scope.canonical(b)).encode()).replace(identity,b'0'*64)
    new=new.replace(b"_CI_STABILITY_GUARD_SHA256='"+b'0'*64+b"'",b"_CI_STABILITY_GUARD_SHA256='"+scope.sha(new).encode()+b"'")
    write(scope.GUARD,new)
    implementation=commit('SYNTHETIC reviewed implementation')
    def exists(revision,path):
        return subprocess.run(['git','cat-file','-e',revision+':'+path],cwd=root,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=root,GUARD_PATH=scope.GUARD,MANIFEST_PATH=manifest,
        git=lambda *args:git(root,*args),git_bytes=lambda *args:raw_git(root,*args),exists_at=exists)
    write(scope.POLICY,scope.canonical(scope.make_policy(g,b,implementation)))
    seal=commit('SYNTHETIC policy-only seal')
    git(root,'gc','--prune=never')
    return root,b,base,tree,implementation,seal


@pytest.fixture
def case(sealed_template,tmp_path,monkeypatch):
    source,binding,base,tree,implementation,seal=sealed_template
    root=tmp_path/'repo';shutil.copytree(source,root)
    monkeypatch.setattr(scope,'BASE',base);monkeypatch.setattr(scope,'BASE_TREE',tree)
    def exists(revision,path):
        return subprocess.run(['git','cat-file','-e',revision+':'+path],cwd=root,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=root,GUARD_PATH=scope.GUARD,MANIFEST_PATH='docs/paid-launch/launch-baseline-manifest.json',
        git=lambda *args:git(root,*args),git_bytes=lambda *args:raw_git(root,*args),exists_at=exists,
        run=lambda *args:(1,{'status':'SYNTHETIC retained original report'}))
    def commit(message):
        git(root,'add','--all');git(root,'commit','-qm',message);return git(root,'rev-parse','HEAD')
    return SimpleNamespace(g=g,binding=deepcopy(binding),commit=commit,implementation=implementation,seal=seal)


def test_synthetic_exact_seal_and_ordered_merge(case):
    f=case; g=f.g
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,None,f.binding)[0]==0
    merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',scope.BASE,'-p',f.seal,'-m','SYNTHETIC ordered tested merge')
    g.git('checkout','-q',merge)
    code,report=scope.run(g,g.ROOT/g.MANIFEST_PATH,scope.BASE,f.binding)
    assert code==0,report
    assert report['original_guard_report']=={'status':'SYNTHETIC retained original report'}


@pytest.mark.parametrize('base',['HEAD','main','--help','0'*40,'5fb8e135',''])
def test_arbitrary_comparison_bases_reject(case,base):
    assert scope.run(case.g,case.g.ROOT/case.g.MANIFEST_PATH,base,case.binding)[0]!=0


@pytest.mark.parametrize('attack',['dirty','staged','committed','resealed','binding','preimage','manifest','tail','rehashed-tail','wrong-parents'])
def test_live_mutations_and_rehashed_claims_reject(case,attack):
    f=case;g=f.g;b=f.binding
    if attack=='binding':b['reviewed_blobs']['tests/ci_fixture_stability_inputs.py']='e'*40
    elif attack=='preimage':b['predecessor_guard']['tail_sha256']='e'*64
    elif attack=='manifest':(g.ROOT/g.MANIFEST_PATH).write_bytes(b'{"base_sha":"UNREVIEWED"}\n')
    elif attack in ('tail','rehashed-tail'):
        path=g.ROOT/scope.GUARD;raw=path.read_bytes()+b'\n# SYNTHETIC altered tail\n'
        if attack=='rehashed-tail':
            identity=re.search(rb"_CI_STABILITY_GUARD_SHA256='([0-9a-f]{64})'",raw).group(1)
            raw=raw.replace(identity,b'0'*64);raw=raw.replace(b"_CI_STABILITY_GUARD_SHA256='"+b'0'*64+b"'",b"_CI_STABILITY_GUARD_SHA256='"+scope.sha(raw).encode()+b"'")
        with pytest.raises(ValueError):scope.previous_guard(raw,b)
        path.write_bytes(raw)
    elif attack=='wrong-parents':
        merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',f.seal,'-p',scope.BASE,'-m','SYNTHETIC reversed parents')
        g.git('checkout','-q',merge)
    else:
        if attack=='resealed':g.git('checkout','-q',f.implementation)
        (g.ROOT/'core/frozen.py').write_bytes(b'# SYNTHETIC changed protected source\n')
        if attack=='staged':g.git('add','core/frozen.py')
        if attack in ('committed','resealed'):f.commit('SYNTHETIC unauthorized source')
        if attack=='resealed':
            impl=g.git('rev-parse','HEAD')
            (g.ROOT/scope.POLICY).write_bytes(scope.canonical(scope.make_policy(g,b,impl)))
            f.commit('SYNTHETIC attempted reseal')
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)[0]!=0


def test_exact_preimages_and_live_input_isolation(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    binding=json.loads((ROOT/scope.BINDING).read_bytes())
    source=(ROOT/scope.GUARD).read_bytes().replace(b'\r\n',b'\n')
    old=scope.previous_guard(source,binding)
    assert len(source)<=scope.MAX_SOURCE and scope.sha(old)==binding['predecessor_guard']['sha256']
    for path,row in inputs.RECORDS.items():
        value=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        assert scope.sha(inputs.original_source(path,value))==row['before_sha256']
        assert inputs.original_source(path,value+b'UNREVIEWED')==value+b'UNREVIEWED'
    native=Path.open
    request=SimpleNamespace(module=SimpleNamespace(__name__=__name__,__file__=__file__))
    assert inputs.install_exact_predecessor_sources(request,monkeypatch) is False
    assert Path.open is native
    from scripts import scope_reconstruction
    assert scope_reconstruction.MAX_CONTEXT_FIELDS==512
    assert scope_reconstruction.MAX_SOURCE_BYTES==scope.MAX_SOURCE


def test_coordinated_family_retains_exact_original_bodies():
    import ast
    from tests import ci_fixture_stability_inputs as inputs
    binding=json.loads((ROOT/scope.BINDING).read_bytes())
    for path in binding['fixture_paths']:
        after=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        before=inputs.original_source(path,after)
        def bodies(source):
            tree=ast.parse(source)
            return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
                if isinstance(node,ast.FunctionDef) and node.name.startswith('test_')}
        assert bodies(before)==bodies(after)
        assert after.count(b"'gc.auto','0'")+after.count(b"'gc.auto', '0'")>=1


def test_authenticated_bootstrap_restores_method_after_injected_compile_failure(monkeypatch):
    import builtins
    native_read=Path.read_bytes;native_compile=builtins.compile;seen=[]
    guard=ROOT/scope.GUARD;source=guard.read_bytes()
    def fail_old_compile(value,filename,mode,*args,**kwargs):
        if isinstance(value,bytes) and len(value)>8_000_000 and str(filename)==str(guard.resolve()):
            seen.append(Path.read_bytes is not native_read)
            raise ValueError('SYNTHETIC predecessor compile failure')
        return native_compile(value,filename,mode,*args,**kwargs)
    compiled=native_compile(source,str(guard),'exec')
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'compile',fail_old_compile)
        with pytest.raises(ValueError,match='SYNTHETIC predecessor compile failure'):
            exec(compiled,{'__file__':str(guard),'__name__':'SYNTHETIC_bootstrap_cleanup'})
    assert seen==[True]
    assert Path.read_bytes is native_read and builtins.compile is native_compile


def test_bootstrap_authenticates_helper_before_execution(tmp_path):
    root=tmp_path/'SYNTHETIC-bootstrap'
    for name in (scope.GUARD,scope.BINDING,'scripts/ci_fixture_stability_scope.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    marker=root/'SYNTHETIC_UNREVIEWED_EXECUTION'
    path=root/'scripts/ci_fixture_stability_scope.py'
    path.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("UNREVIEWED")\n')
    native=Path.read_bytes
    with pytest.raises(ValueError,match='SUCCESSOR_SCOPE_MODULE_CHANGED'):
        exec(compile((root/scope.GUARD).read_bytes(),str(root/scope.GUARD),'exec'),
             {'__file__':str(root/scope.GUARD),'__name__':'SYNTHETIC_helper_authentication'})
    assert not marker.exists() and Path.read_bytes is native


def test_historical_guard_view_keeps_verified_loader_across_alias_replacement(monkeypatch):
    import scripts,sys
    from tests import ci_fixture_stability_inputs as inputs
    binding=json.loads((ROOT/scope.BINDING).read_bytes())
    request=SimpleNamespace(module=SimpleNamespace(__name__='test_nfl_admission_scope',
        __file__=str(ROOT/'tests/test_nfl_admission_scope.py')))
    native=Path.open
    with monkeypatch.context() as patch:
        assert inputs.install_exact_predecessor_sources(request,patch)
        previous=SimpleNamespace(__name__='SYNTHETIC_frozen_predecessor_without_successor_loader')
        patch.setattr(scripts,'check_launch_change_scope',previous)
        patch.setitem(sys.modules,'scripts.check_launch_change_scope',previous)
        with (ROOT/scope.GUARD).open('rb') as stream:value=stream.read()
        assert scope.sha(value)==binding['predecessor_guard']['sha256']
        assert not hasattr(previous,'_ci_fixture_module')
    assert Path.open is native


def test_successor_adapter_preserves_authenticated_intermediate_source_forms():
    from scripts import check_launch_change_scope as live,ncaaf_chronology_capacity_scope as historical
    binding=json.loads((ROOT/scope.BINDING).read_bytes())
    prior=scope.previous_guard((ROOT/scope.GUARD).read_bytes().replace(b'\r\n',b'\n'),binding)
    inherited=json.loads((ROOT/historical.BINDING).read_bytes())
    intermediate=historical.predecessor_guard(prior,live,inherited)
    assert intermediate!=prior
    assert live._ci_fixture_previous_guard_match(intermediate)
    assert live._ncaaf_future_guard_matches(intermediate)
    assert live._ncaaf_future_base_source(intermediate)==live._ci_fixture_previous_base_source(intermediate)
    assert not live._ncaaf_future_guard_matches(intermediate+b'UNREVIEWED')


def test_single_advisory_amendment_preserves_all_other_rows_and_metadata():
    from tests import ci_fixture_stability_inputs as inputs
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes()
    previous=inputs.original_source(inputs.PROFILE,raw)
    assert scope.sha(previous)=='287444d8aae28cd41d30a8f8e62b48ccebafbe989481ee030970bbdc0336d2bd'
    old=json.loads(previous);current=json.loads(raw)
    before={r['path']:r for r in old['files']};after={r['path']:r for r in current['files']}
    assert {p for p in after if before.get(p)!=after[p]}=={'tests/test_mlb_nhl_research_scope.py'}
    assert all(current[k]==v for k,v in old.items() if k not in ('files','sources'))
    assert all(current['sources'][k]==v for k,v in old['sources'].items())
    row=after['tests/test_mlb_nhl_research_scope.py']
    assert row['seconds']==557.62465==round(max(r['seconds'] for r in row['completed_measurements'])*1.15,6)
    assert row['completed_measurements'][:-1]==before[row['path']]['completed_measurements']
    source=current['sources'][row['completed_measurements'][-1]['source']]
    assert source['completed']==source['identities']==56
    assert source['failures']==source['errors']==source['skips']==0
    assert source['checkout_commit']=='198e8ae509e3e790aed49f5529e0c9f38ddcd2b9'
    assert len(source['checkout_receipt_log_sha256'])==64
    assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)


def test_profile_historical_selector_is_exact_and_current_runner_stays_live(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes();native_open=Path.open;native_read=Path.read_bytes
    def request(module,test):
        return SimpleNamespace(module=SimpleNamespace(__name__='tests.'+module,__file__=str(ROOT/('tests/'+module+'.py'))),
            node=SimpleNamespace(originalname=test))
    current=request('test_ci_test_shards','test_retained_profile_distinguishes_branch_head_tested_checkout_and_estimates')
    with monkeypatch.context() as patch:
        assert not inputs.install_exact_predecessor_sources(current,patch)
        assert (ROOT/inputs.PROFILE).read_bytes()==raw
        assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)
    historical=request('test_ncaaf_reviewed_target_timing','test_five_live_costs_preserve_historical_measurement_meanings')
    with monkeypatch.context() as patch:
        assert inputs.install_exact_predecessor_sources(historical,patch)
        assert (ROOT/inputs.PROFILE).read_bytes()==inputs.original_source(inputs.PROFILE,raw)
        assert json.loads((ROOT/inputs.PROFILE).read_text())['files']!=json.loads(raw)['files']
    assert Path.open is native_open and Path.read_bytes is native_read
    assert (ROOT/inputs.PROFILE).read_bytes()==raw


def test_mixed_historical_helper_retains_live_reads_and_rejects_unknown_profile(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs,ncaaf_staged_capacity_fixtures as fixtures
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes();native=fixtures.reviewed_target_predecessor_profile
    request=SimpleNamespace(module=SimpleNamespace(__name__='tests.test_ncaaf_reviewed_target_timing',
        __file__=str(ROOT/'tests/test_ncaaf_reviewed_target_timing.py')),
        node=SimpleNamespace(originalname='test_timing_historical_snapshot_is_isolated_while_current_runner_stays_live'))
    with monkeypatch.context() as patch:
        assert inputs.install_exact_predecessor_sources(request,patch)
        assert (ROOT/inputs.PROFILE).read_bytes()==raw
        assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)
        assert scope.sha(fixtures.reviewed_target_predecessor_profile(raw))=='7474e2d7f22d4c5ad5dfa6dbc945d2e549a401250e295546303084c6116ad5a7'
        changed=json.loads(raw);changed['files'][0]['seconds']+=1
        for value in (raw+b'UNREVIEWED',json.dumps(changed).encode()):
            assert inputs.original_source(inputs.PROFILE,value)==value
            with pytest.raises(AssertionError):fixtures.reviewed_target_predecessor_profile(value)
    assert fixtures.reviewed_target_predecessor_profile is native
    assert (ROOT/inputs.PROFILE).read_bytes()==raw


def test_second_advisory_amendment_retains_exact_single_entry_predecessor():
    from tests import ci_fixture_stability_inputs as inputs
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes()
    single=inputs.single_advisory_profile(raw)
    assert scope.sha(single)=='055c2bb9e9a7b3054d9adeac4fae532f85736fa4cf95d25320dab0fb78583fad'
    assert scope.sha(inputs.original_source(inputs.PROFILE,single))=='287444d8aae28cd41d30a8f8e62b48ccebafbe989481ee030970bbdc0336d2bd'
    old=json.loads(single);current=json.loads(raw)
    before={r['path']:r for r in old['files']};after={r['path']:r for r in current['files']}
    assert {p for p in after if before.get(p)!=after[p]}=={'tests/test_source_evidence_intake.py'}
    assert all(current[k]==v for k,v in old.items() if k not in ('files','sources'))
    assert all(current['sources'][k]==v for k,v in old['sources'].items())
    row=after['tests/test_source_evidence_intake.py']
    assert row['seconds']==207.3427==round(max(r['seconds'] for r in row['completed_measurements'])*1.15,6)
    assert row['completed_measurements'][:-1]==before[row['path']]['completed_measurements']
    record=current['sources'][row['completed_measurements'][-1]['source']]
    assert record['completed']==record['identities']==78
    assert record['failures']==record['errors']==record['skips']==0
    assert record['checkout_commit']=='ddba4aa72397a396cb274e4cc95e101222b58e18'
    assert record['junit_sha256']=='5e21d0ea3756eb256897481fb1f7ce06a8c3ab0637b4e9ce331940b0eceb0a74'
    assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)


def test_second_advisory_preimage_is_narrow_authenticated_and_restored(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    raw=(ROOT/inputs.PROFILE).read_bytes();native_open=Path.open;native_read=Path.read_bytes
    request=SimpleNamespace(module=SimpleNamespace(__name__=__name__,__file__=__file__),
        node=SimpleNamespace(originalname='test_single_advisory_amendment_preserves_all_other_rows_and_metadata'))
    with monkeypatch.context() as patch:
        assert inputs.install_exact_predecessor_sources(request,patch)
        assert (ROOT/inputs.PROFILE).read_bytes()==inputs.single_advisory_profile(raw)
        assert (ROOT/inputs.PROFILE).read_text()==inputs.single_advisory_profile(raw).decode()
    assert Path.open is native_open and Path.read_bytes is native_read
    assert (ROOT/inputs.PROFILE).read_bytes()==raw
    changed=json.loads(raw);changed['files'][0]['seconds']+=1
    for altered in (raw+b'UNREVIEWED',scope.canonical(changed)):
        assert inputs.single_advisory_profile(altered)==altered
        assert inputs.original_source(inputs.PROFILE,altered)==altered
    with monkeypatch.context() as patch:
        corrupted=dict(inputs.SINGLE_ADVISORY_PROFILE,before_git_blob='0'*40)
        patch.setattr(inputs,'SINGLE_ADVISORY_PROFILE',corrupted)
        with pytest.raises(ValueError,match='CI_FIXTURE_AUTHENTICATED_PROFILE_PREDECESSOR'):
            inputs.single_advisory_profile(raw)
    assert inputs.single_advisory_profile(raw)!=raw


def test_complete_profile_refresh_preserves_all_prior_rows_and_measurements():
    from tests import ci_fixture_stability_inputs as inputs
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes()
    previous=inputs.two_advisory_profile(raw)
    assert scope.sha(previous)=='71bf37a82b32eb3c640b2e7bb0b13655bf1d5c11324a597a975b69a8f5e8290f'
    old=json.loads(previous);current=json.loads(raw)
    before={r['path']:r for r in old['files']};after={r['path']:r for r in current['files']}
    manifest=current['ci_fixture_stability_complete_profile_amendment']
    changed={p for p in after if before.get(p)!=after[p]}
    assert len(changed)==347 and len(changed & set(before))==345
    assert set(after)-set(before)=={'tests/test_ci_scheduling_fixture_copy.py','tests/test_ncaaf_observed_spread_pair.py'}
    assert changed==set(manifest['paths']) and manifest['multiplier']==1.15
    assert manifest['proposal_sha256']==scope.CONSOLIDATED_AMENDMENT
    assert manifest['ledger_sha256']==scope.COMPLETE_LEDGER_SHA256
    assert manifest['additional_fixture_setup_seconds'] is None
    assert all(current[k]==v for k,v in old.items() if k not in ('files','sources'))
    assert all(current['sources'][k]==v for k,v in old['sources'].items())
    for name in changed:
        row=after[name];prior=before.get(name,{})
        for field in ('completed_measurements','references'):
            assert row[field][:len(prior.get(field,[]))]==prior.get(field,[])
        for field,value in prior.items():
            if field not in {'seconds','advisory_basis_seconds','measurement_complete','basis','completed_measurements','references'}:
                assert row[field]==value
        measurements=[r for r in row['completed_measurements'] if r.get('kind','completed_junit_testcase_sum')=='completed_junit_testcase_sum'
            and r['count']==row['profiled_tests'] and r.get('measurement_complete') is not False]
        assert row['seconds']==round(max(r['seconds'] for r in measurements)*1.15,6)
        assert row['seconds']>=prior.get('seconds',0)
        for record in row['completed_measurements'][len(prior.get('completed_measurements',[])):]:
            source=current['sources'][record['source']]
            assert source['completed']==source['identities']==row['profiled_tests']
            assert source['failures']==source['errors']==source['skips']==0
            assert len(source['junit_sha256'])==len(source['ordered_identity_sha256'])==64
            assert len(source['checkout_commit'])==40
            assert source['checkout_commit']==source['checkout_provenance']['checkout_commit']
    assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)


def test_complete_profile_views_are_exact_reject_alteration_and_restore_current_reads(monkeypatch):
    from tests import ci_fixture_stability_inputs as inputs
    from scripts import run_ci_tests as runner
    raw=(ROOT/inputs.PROFILE).read_bytes();native_open=Path.open;native_read=Path.read_bytes
    assert scope.sha(inputs.two_advisory_profile(raw))=='71bf37a82b32eb3c640b2e7bb0b13655bf1d5c11324a597a975b69a8f5e8290f'
    assert scope.sha(inputs.single_advisory_profile(raw))=='055c2bb9e9a7b3054d9adeac4fae532f85736fa4cf95d25320dab0fb78583fad'
    assert scope.sha(inputs.original_source(inputs.PROFILE,raw))=='287444d8aae28cd41d30a8f8e62b48ccebafbe989481ee030970bbdc0336d2bd'
    request=SimpleNamespace(module=SimpleNamespace(__name__=__name__,__file__=__file__),
        node=SimpleNamespace(originalname='test_second_advisory_amendment_retains_exact_single_entry_predecessor'))
    with monkeypatch.context() as patch:
        assert inputs.install_exact_predecessor_sources(request,patch)
        assert (ROOT/inputs.PROFILE).read_bytes()==inputs.two_advisory_profile(raw)
        assert (ROOT/inputs.PROFILE).read_text()==inputs.two_advisory_profile(raw).decode()
    assert Path.open is native_open and Path.read_bytes is native_read
    assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==scope.sha(raw)
    changed=json.loads(raw);changed['files'][0]['seconds']+=1
    for altered in (raw+b'UNREVIEWED',scope.canonical(changed)):
        assert inputs.two_advisory_profile(altered)==altered
        assert inputs.single_advisory_profile(altered)==altered
        assert inputs.original_source(inputs.PROFILE,altered)==altered
    with monkeypatch.context() as patch:
        patch.setattr(inputs,'CONSOLIDATED_PROFILE',dict(inputs.CONSOLIDATED_PROFILE,before_git_blob='0'*40))
        with pytest.raises(ValueError,match='CI_FIXTURE_AUTHENTICATED_PROFILE_PREDECESSOR'):
            inputs.two_advisory_profile(raw)


@pytest.mark.parametrize('failure',[False,True])
def test_module_predecessor_projection_precedes_consumption_and_restores(request,failure):
    import importlib
    setup=importlib.import_module('ncaaf_future_scope_inputs')
    module=importlib.import_module('test_football_document_ci_performance')
    native_open=Path.open;native_read=Path.read_bytes;native_guard=module.guard
    fixture=SimpleNamespace(func=module.predecessor.__wrapped__,scope='module',argname='predecessor')
    hook=setup._HistoricalSourceSetup().pytest_fixture_setup(fixture,request)
    next(hook)
    try:
        value,_,binding=module.predecessor.__wrapped__()
        assert scope.sha(value)==binding['previous_guard_sha256']
        assert module.guard is not native_guard
        if failure:
            with pytest.raises(RuntimeError,match='SYNTHETIC_MODULE_FIXTURE_FAILURE'):
                raise RuntimeError('SYNTHETIC_MODULE_FIXTURE_FAILURE')
    finally:
        hook.close()
    assert Path.open is native_open and Path.read_bytes is native_read
    assert module.guard is native_guard


def test_module_predecessor_projection_is_exact_in_fixture_scope_name_and_source(request,monkeypatch):
    import importlib
    setup=importlib.import_module('ncaaf_future_scope_inputs')
    module=importlib.import_module('test_football_document_ci_performance')
    native_open=Path.open;native_read=Path.read_bytes;native_guard=module.guard
    for scope_name,fixture_name in [('function','predecessor'),('module','different_fixture')]:
        fixture=SimpleNamespace(func=module.predecessor.__wrapped__,scope=scope_name,argname=fixture_name)
        hook=setup._HistoricalSourceSetup().pytest_fixture_setup(fixture,request)
        next(hook)
        try:
            assert Path.open is native_open and Path.read_bytes is native_read
            assert module.guard is native_guard
        finally:hook.close()
    fixture=SimpleNamespace(func=module.predecessor.__wrapped__,scope='module',argname='predecessor')
    current=(ROOT/scope.GUARD).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_bytes',lambda path:current+b'UNREVIEWED' if path.resolve()==(ROOT/scope.GUARD).resolve() else native_read(path))
        hook=setup._HistoricalSourceSetup().pytest_fixture_setup(fixture,request)
        next(hook)
        try:
            with pytest.raises((AssertionError,ValueError)):
                module.predecessor.__wrapped__()
        finally:hook.close()
    assert Path.open is native_open and Path.read_bytes is native_read
    assert module.guard is native_guard
