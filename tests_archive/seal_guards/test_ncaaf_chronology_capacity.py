"""Real synthetic Git seals; no full-history/account/provider dependency."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import pytest
from scripts import ncaaf_chronology_capacity_scope as scope


def sha(b):return hashlib.sha256(b).hexdigest()
def require(ok,reason):
    if not ok:raise ValueError(reason)


@pytest.fixture
def sealed(tmp_path):
    root=tmp_path/'SYNTHETIC-repository';root.mkdir()
    def git(*a):return subprocess.check_output(['git',*a],cwd=root).decode().strip()
    def raw(*a):return subprocess.check_output(['git',*a],cwd=root)
    def write(p,b):
        path=root/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    def commit(text):
        git('add','--all');git('commit','-qm',text);return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC scope test');git('config','user.email','synthetic@example.invalid')
    git('config','gc.auto','0');git('config','maintenance.auto','false');git('config','core.autocrlf','false');git('config','core.longpaths','true')
    G='scripts/check_launch_change_scope.py';M='docs/paid-launch/launch-baseline-manifest.json'
    F='scripts/ncaaf_reviewed_target_scope.py';APP='tests/test_ncaaf_chronology_scope.py'
    old_binding=dict(reviewed_blobs={F:'a'*40},source_reconstructions={},base='SYNTHETIC inherited contract')
    prior=json.dumps(old_binding).encode();old_identity='7'*64
    guard=b"_RT_BINDING_SHA256='"+sha(prior).encode()+b"'\nNCAAF_FUTURE_REVIEWED_GUARD_SHA256='"+old_identity.encode()+b"'\n# mirror "+old_identity.encode()+b'\n'
    guard+=b' '*(8388490-len(guard))
    manifest=json.dumps(dict(base_sha='5'*40,protected_files=['core/frozen.py'])).encode()
    write(G,guard);write(scope.BINDING,prior);write(F,b'# SYNTHETIC predecessor\n');write(APP,b'SYNTHETIC before\n')
    write(M,manifest);write('core/frozen.py',b'SYNTHETIC protected bytes\n')
    base=commit('SYNTHETIC reviewed baseline')
    write(F,b'# SYNTHETIC successor loader\n');write(APP,b'SYNTHETIC after\n')
    paths=[G,scope.BINDING,F,APP]
    d=dict(proposal_sha256=scope.PROPOSAL,base=base,base_tree=git('rev-parse',base+'^{tree}'),
        ordered_base_parents=[],predecessor=base,predecessor_tree=git('rev-parse',base+'^{tree}'),predecessor_parents=[],manifest_sha256=sha(manifest),previous_binding_sha256=sha(prior),
        previous_binding_b64=base64.b64encode(prior).decode(),previous_guard_sha256=sha(guard),previous_guard_identity=old_identity,
        paths=paths,new_paths=paths,reviewed_blobs={F:git('hash-object',F),APP:git('hash-object',APP)},
        source_reconstructions={APP:dict(before_sha256=sha(b'SYNTHETIC before\n'),after_sha256=sha(b'SYNTHETIC after\n'),
            reverse_patches=[dict(start=0,end=len(b'SYNTHETIC after\n'),text='SYNTHETIC before\n')])})
    binding=deepcopy(old_binding);binding['chronology_capacity']=d;binding['reviewed_blobs'][F]=d['reviewed_blobs'][F]
    binding['source_reconstructions'][APP]=d['source_reconstructions'][APP]
    body=(json.dumps(binding,sort_keys=True,indent=2)+'\n').encode();write(scope.BINDING,body)
    successor=guard.replace(sha(prior).encode(),sha(body).encode()).replace(old_identity.encode(),b'0'*64)
    identity=sha(successor);successor=successor.replace(b'0'*64,identity.encode());write(G,successor)
    impl=commit('SYNTHETIC implementation')
    def exists(rev,path):return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=root,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=root,GUARD_PATH=G,MANIFEST_PATH=M,git=git,git_bytes=raw,exists_at=exists,_require=require,
        _RT_BINDING_SHA256=sha(body),NCAAF_FUTURE_REVIEWED_GUARD_SHA256=identity,
        _ncaaf_future_guard_matches=lambda b:b==successor,run=lambda *a:(1,dict(status='SYNTHETIC original report unchanged')))
    policy=scope.make_policy(g,binding,impl);write(scope.POLICY,json.dumps(policy).encode());seal=commit('SYNTHETIC policy-only seal')
    return SimpleNamespace(g=g,binding=binding,write=write,commit=commit,impl=impl,seal=seal,prior_guard=guard,successor=successor)


def test_exact_successor_and_ordered_tested_merge(sealed):
    f=sealed;g=f.g
    result=scope.run(g,g.ROOT/g.MANIFEST_PATH,None,f.binding)
    assert result[0]==0,result[1]
    merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',f.binding['chronology_capacity']['base'],'-p',f.seal,'-m','SYNTHETIC ordered CI merge')
    g.git('checkout','-q',merge)
    result=scope.run(g,g.ROOT/g.MANIFEST_PATH,None,f.binding)
    assert result[0]==0,result[1]


@pytest.mark.parametrize('attack',['dirty','staged','committed','resealed','changed-binding','changed-source','rehashed-before-source'])
def test_mutation_and_rehashed_provenance_still_reject(sealed,attack):
    f=sealed;g=f.g;b=deepcopy(f.binding)
    if attack=='changed-binding':b['base']='UNREVIEWED'
    elif attack=='rehashed-before-source':b['chronology_capacity']['source_reconstructions']['tests/test_ncaaf_chronology_scope.py']['before_sha256']='e'*64
    elif attack=='changed-source':f.write('tests/test_ncaaf_chronology_scope.py',b'UNREVIEWED')
    else:
        if attack=='resealed':g.git('checkout','-q',f.impl)
        f.write('core/frozen.py',b'SYNTHETIC unauthorized mutation\n')
        if attack=='staged':g.git('add','core/frozen.py')
        if attack in {'committed','resealed'}:f.commit('SYNTHETIC unauthorized bytes')
        if attack=='resealed':
            impl=g.git('rev-parse','HEAD');f.write(scope.POLICY,json.dumps(scope.make_policy(g,b,impl)).encode());f.commit('SYNTHETIC reseal')
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)[0]!=0


@pytest.mark.parametrize('base',['0'*40,'HEAD','main','--help','5fb8e135',''])
def test_arbitrary_comparison_base_rejects(sealed,base):
    assert scope.run(sealed.g,sealed.g.ROOT/sealed.g.MANIFEST_PATH,base,sealed.binding)[0]!=0


def test_exact_predecessor_source_and_altered_source(sealed):
    f=sealed;g=f.g
    assert scope.predecessor_guard(f.successor,g,f.binding)==f.prior_guard
    with pytest.raises(ValueError):scope.predecessor_guard(f.successor+b'UNREVIEWED',g,f.binding)
    assert scope.previous_source('tests/test_ncaaf_chronology_scope.py',b'SYNTHETIC after\n',g,f.binding)==b'SYNTHETIC before\n'
    assert scope.previous_source('tests/test_ncaaf_chronology_scope.py',b'SYNTHETIC altered\n',g,f.binding)==b'SYNTHETIC altered\n'



def test_current_successor_has_exact_preimages_without_relaxing_limits():
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/scope.BINDING).read_bytes())
    from scripts import check_launch_change_scope as live
    with (root/live.GUARD_PATH).open('rb') as stream:current=stream.read().replace(b'\r\n',b'\n')
    assert sha(scope.predecessor_guard(current,live,binding))==binding['chronology_capacity']['previous_guard_sha256']
    for path,row in binding['chronology_capacity']['source_reconstructions'].items():
        with (root/path).open('rb') as stream:source=stream.read().replace(b'\r\n',b'\n')
        assert sha(scope.previous_source(path,source,live,binding))==row['before_sha256']
        assert scope.previous_source(path,source+b'UNREVIEWED',live,binding)==source+b'UNREVIEWED'
    from scripts import scope_reconstruction
    assert scope_reconstruction.MAX_CONTEXT_FIELDS==512
    assert len(current)==8388490<=scope_reconstruction.MAX_SOURCE_BYTES


def test_new_helper_is_authenticated_before_execution(tmp_path):
    from scripts import ncaaf_reviewed_target_scope as parent
    root=tmp_path/'SYNTHETIC helper';(root/'scripts').mkdir(parents=True)
    marker=root/'SYNTHETIC_EXECUTED'
    (root/'scripts/ncaaf_chronology_capacity_scope.py').write_text(
        'from pathlib import Path\nPath('+repr(str(marker))+').write_text("UNREVIEWED")\n')
    g=SimpleNamespace(ROOT=root,_require=require)
    binding={'chronology_capacity':{'reviewed_blobs':{'scripts/ncaaf_chronology_capacity_scope.py':'0'*40}}}
    code,report=parent.run(g,root/'SYNTHETIC-manifest',None,binding)
    assert code!=0 and report['reason_codes']==['CAPACITY_SCOPE_SOURCE_CHANGED']
    assert not marker.exists()


def synthetic_reports(directory):
    from scripts import run_ci_tests as runner
    profile=dict(files={},fallback=runner.FALLBACK.copy(),sha256='a'*64,version=1)
    original=[SimpleNamespace(path=f'test_SYNTHETIC_{i}.py',nodeid=f'test_SYNTHETIC_{i}.py::test_{j}') for i in range(11) for j in range(2)]
    for index in range(11):
        plugin=runner.FileShard(index,11,profile,manifest=directory/f'full-suite-{index+1}-assignment.json')
        items=original.copy();plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items:None)),items)
        xml=ET.Element('testsuite')
        for item in items:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when='call',outcome='passed'))
            file,name=item.nodeid.split('::');ET.SubElement(xml,'testcase',classname=file[:-3],name=name)
        assert plugin.finish(0)==0
        ET.ElementTree(xml).write(directory/f'full-suite-{index+1}.xml')


@pytest.mark.parametrize('attack',['missing10','missing11','duplicate','cancelled','wrong_revision_profile','wrong_count','changed_assignment','failed_xml','missing_xml','reordered_xml','missing_outcome'])
def test_eleven_way_reconciliation_rejects_incomplete_or_altered_evidence(tmp_path,attack):
    from scripts import run_ci_tests as runner
    synthetic_reports(tmp_path);assert runner.reconcile(tmp_path,11)['completed_count']==22
    path=tmp_path/'full-suite-11-assignment.json';data=json.loads(path.read_bytes())
    if attack in ('missing10','missing11'):(tmp_path/f'full-suite-{attack[7:]}-assignment.json').unlink()
    elif attack=='duplicate':(tmp_path/'full-suite-12-assignment.json').write_text(json.dumps(data))
    elif attack=='missing_xml':(tmp_path/'full-suite-11.xml').unlink()
    elif attack in ('failed_xml','reordered_xml'):
        xml=tmp_path/'full-suite-11.xml';tree=ET.parse(xml)
        if attack=='failed_xml':ET.SubElement(tree.getroot()[0],'failure')
        else:tree.getroot()[:]=list(reversed(list(tree.getroot())))
        tree.write(xml)
    else:
        if attack=='cancelled':data['execution_complete']=False;data['exit_code']=2
        elif attack=='wrong_revision_profile':data['profile_sha256']='b'*64
        elif attack=='wrong_count':data['shards']=9
        elif attack=='changed_assignment':data['assignments'][0]['shard']=11
        elif attack=='missing_outcome':data['completed_nodeids'].pop()
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,OSError)):runner.reconcile(tmp_path,11)


def test_actual_eleven_way_cli_and_exact_once_readback(tmp_path):
    from scripts import run_ci_tests as runner
    import sys
    for i in range(11):(tmp_path/f'test_SYNTHETIC_{i}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1,12):
        result=subprocess.run([sys.executable,str(runner.ROOT/'scripts/run_ci_tests.py'),'--shard',str(index),'--shards','11','--assignment-manifest',str(tmp_path/f'full-suite-{index}-assignment.json'),'-q',str(tmp_path),'--junitxml='+str(tmp_path/f'full-suite-{index}.xml')],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert runner.reconcile(tmp_path,11)['completed_count']==11


def test_current_eleven_way_gate_preserves_profile_limits_and_history():
    import yaml
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.load((root/'.github/workflows/ci.yml').read_text(),Loader=yaml.BaseLoader)
    shard=workflow['jobs']['full-suite-tests'];gate=workflow['jobs']['full-suite']
    assert shard['strategy']['matrix']['shard']==[str(i) for i in range(1,12)]
    assert shard['timeout-minutes']=='30' and shard['strategy']['fail-fast']=='false'
    assert '--shards 11' in shard['steps'][-2]['run'] and '--shards 11' in gate['steps'][-1]['run']
    assert hashlib.sha256((root/'scripts/ci_test_file_costs_v1.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='287444d8aae28cd41d30a8f8e62b48ccebafbe989481ee030970bbdc0336d2bd'
    from tests import test_ci_test_shards as current
    current.test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate()
    from tests import test_ncaaf_staged_ci_capacity as historical
    assert 'range(9)' in __import__('inspect').getsource(historical.reports)


def test_historical_input_projection_is_exact_and_restores_global_state(tmp_path):
    from tests.ncaaf_staged_capacity_fixtures import install_chronology_capacity_inputs
    from scripts import check_launch_change_scope as live
    root=Path(__file__).resolve().parents[1];native=Path.open
    raw=(root/scope.BINDING).read_bytes();binding=json.loads(raw)
    request=SimpleNamespace(module=SimpleNamespace(__name__='test_ncaaf_chronology_fixture_scope'))
    with pytest.MonkeyPatch.context() as patch:
        assert install_chronology_capacity_inputs(request,patch)
        assert (root/scope.BINDING).read_bytes()==base64.b64decode(binding['chronology_capacity']['previous_binding_b64'])
        with (root/live.GUARD_PATH).open('rb') as handle:
            assert sha(handle.read())==binding['chronology_capacity']['previous_guard_sha256']
        other=tmp_path/'SYNTHETIC-unrelated';other.write_bytes(b'UNREVIEWED')
        assert other.read_bytes()==b'UNREVIEWED'
    assert Path.open is native and (root/scope.BINDING).read_bytes()==raw
    from tests import test_ncaaf_reviewed_target_timing as historical
    request=SimpleNamespace(module=historical,node=SimpleNamespace(name=
        'test_timing_fixture_changes_preserve_all_original_assertions_without_history'))
    path='tests/test_ncaaf_staged_ci_capacity.py'
    current=(root/path).read_bytes()
    with pytest.MonkeyPatch.context() as patch:
        assert install_chronology_capacity_inputs(request,patch)
        expected=scope.previous_source(path,current,live,binding)
        assert (root/path).read_bytes()==expected!=current
        historical.test_timing_fixture_changes_preserve_all_original_assertions_without_history()
        assert scope.previous_source(path,current+b'UNREVIEWED',live,binding)==current+b'UNREVIEWED'
    assert Path.open is native and (root/path).read_bytes()==current
    request.node.name='test_five_live_costs_preserve_historical_measurement_meanings'
    with pytest.MonkeyPatch.context() as patch:
        assert not install_chronology_capacity_inputs(request,patch)
        assert Path.open is native and (root/path).read_bytes()==current
    # These are frozen-source readers, not current workflow gates. Their
    # exact nine-way bytes must precede inherited six/five-way hash checks.
    for module,node in (
        ('test_football_document_predecessor_fixture','test_exact_five_shard_input_and_idempotence'),
        ('test_nfl_seven_predecessor_inputs','test_exact_six_input_is_idempotent_and_live_seven_source_is_unchanged'),
        ('test_nfl_pdf_timing_costs','test_original_assertions_preserved_except_approved_current_cardinality'),
    ):
        request=SimpleNamespace(module=SimpleNamespace(__name__=module),
            node=SimpleNamespace(name=node))
        for path in ('.github/workflows/ci.yml','tests/test_ci_test_shards.py'):
            current=(root/path).read_bytes()
            expected=scope.previous_source(path,current,live,binding)
            assert expected!=current
            with pytest.MonkeyPatch.context() as patch:
                assert install_chronology_capacity_inputs(request,patch)
                assert (root/path).read_bytes()==expected
                assert scope.previous_source(path,current+b'UNREVIEWED',live,binding)==current+b'UNREVIEWED'
            assert Path.open is native and (root/path).read_bytes()==current
    for module in ('test_ci_test_shards','test_ncaaf_chronology_capacity'):
        request=SimpleNamespace(module=SimpleNamespace(__name__=module),
            node=SimpleNamespace(name='test_current_eleven_way_gate_preserves_profile_limits_and_history'))
        with pytest.MonkeyPatch.context() as patch:
            assert not install_chronology_capacity_inputs(request,patch)
            assert Path.open is native
    test_current_eleven_way_gate_preserves_profile_limits_and_history()
