"""Synthetic nine-way execution, exact predecessor inputs and completed costs."""
from copy import deepcopy
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
import yaml
from scripts import run_ci_tests as runner
from scripts import ncaaf_staged_capacity_scope as scope
from tests.ncaaf_staged_capacity_fixtures import exact_predecessor_bytes,historical_ncaaf_staged_capacity_profile,historical_reviewed_target_timing_profile
ROOT=Path(__file__).resolve().parents[1]
PROFILE='scripts/ci_test_file_costs_v1.json'

def test_exact_43_material_upward_costs_and_retained_metadata(historical_reviewed_target_timing_profile):
    raw=(ROOT/PROFILE).read_bytes().replace(b'\r\n',b'\n');now=json.loads(raw)
    prior=json.loads(exact_predecessor_bytes(PROFILE,raw))
    assert now['revision']==prior['revision']==3 and now['fallback']==prior['fallback']
    assert all(now[k]==v for k,v in prior.items() if k not in ('files','sources'))
    assert all(now['sources'][k]==v for k,v in prior['sources'].items())
    old={r['path']:r for r in prior['files']};rows={r['path']:r for r in now['files']}
    changed={p for p in rows if rows[p]!=old.get(p)}
    assert len(changed)==now['ncaaf_staged_runtime_revision']['changed_rows']==43
    for path in changed:
        entry=rows[path];previous=old.get(path);count=entry['profiled_tests'];fallback=now['fallback']
        effective=(previous['seconds']+max(0,count-previous['profiled_tests'])*fallback['seconds_per_test']) if previous and previous['seconds'] is not None else max(fallback['minimum_file_seconds'],count*fallback['seconds_per_test'])
        observations=[m for m in entry['completed_measurements'] if m['source'].startswith('ncaaf-staged-runtime-')]
        assert entry['seconds']==round(max(o['seconds'] for o in observations)*1.15,6)
        assert entry['seconds']-effective>=fallback['seconds_per_test']
        assert entry['kind']=='estimated' and entry['latest_measurement_complete']
        for observation in observations:
            source=now['sources'][observation['source']]
            assert source['kind']=='completed_junit_testcase_sum' and source['run_attempt']==1
            assert len(source['checkout_commit'])==40 and len(source['xml_sha256'])==len(source['assignment_manifest_sha256'])==64
            assert observation['count']==count and len(entry['runtime_followup_test_blob'])==40
        if previous:
            for key in ('references','completed_measurements','incomplete_observations'):
                before=previous.get(key,[]);assert entry.get(key,[])[:len(before)]==before

def reports(directory):
    profile=dict(files={},fallback=runner.FALLBACK.copy(),sha256='a'*64,version=1)
    original=[SimpleNamespace(path=f'test_SYNTHETIC_{i}.py',nodeid=f'test_SYNTHETIC_{i}.py::test_{j}') for i in range(9) for j in range(2)]
    for index in range(9):
        plugin=runner.FileShard(index,9,profile,manifest=directory/f'full-suite-{index+1}-assignment.json')
        items=original.copy();plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items:None)),items)
        xml=ET.Element('testsuite')
        for item in items:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when='call',outcome='passed'))
            file,name=item.nodeid.split('::');ET.SubElement(xml,'testcase',classname=file[:-3],name=name)
        assert plugin.finish(0)==0
        ET.ElementTree(xml).write(directory/f'full-suite-{index+1}.xml')

@pytest.mark.parametrize('attack',['missing8','missing9','duplicate','cancelled','wrong_revision_profile','wrong_count','changed_assignment','failed_xml','missing_xml','reordered_xml','missing_outcome'])
def test_nine_way_reconciliation_preserves_fail_closed_outcomes(tmp_path,attack):
    reports(tmp_path);assert runner.reconcile(tmp_path,9)['completed_count']==18
    path=tmp_path/'full-suite-9-assignment.json';data=json.loads(path.read_bytes())
    if attack.startswith('missing') and attack[-1] in '89':(tmp_path/f'full-suite-{attack[-1]}-assignment.json').unlink()
    elif attack=='duplicate':(tmp_path/'full-suite-10-assignment.json').write_text(json.dumps(data))
    elif attack=='missing_xml':(tmp_path/'full-suite-9.xml').unlink()
    elif attack in ('failed_xml','reordered_xml'):
        xml=tmp_path/'full-suite-9.xml';tree=ET.parse(xml)
        if attack=='failed_xml':ET.SubElement(tree.getroot()[0],'failure')
        else:tree.getroot()[:]=list(reversed(list(tree.getroot())))
        tree.write(xml)
    else:
        if attack=='cancelled':data['execution_complete']=False;data['exit_code']=2
        elif attack=='wrong_revision_profile':data['profile_sha256']='b'*64
        elif attack=='wrong_count':data['shards']=7
        elif attack=='changed_assignment':data['assignments'][0]['shard']=9
        elif attack=='missing_outcome':data['completed_nodeids'].pop()
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,OSError)):runner.reconcile(tmp_path,9)

def test_actual_nine_way_cli_and_exact_once_readback(tmp_path):
    for i in range(9):(tmp_path/f'test_SYNTHETIC_{i}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1,10):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_ci_tests.py'),'--shard',str(index),'--shards','9','--assignment-manifest',str(tmp_path/f'full-suite-{index}-assignment.json'),'-q',str(tmp_path),'--junitxml='+str(tmp_path/f'full-suite-{index}.xml')],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert runner.reconcile(tmp_path,9)['completed_count']==9

@pytest.mark.parametrize('attack',['run_id','run_head','artifact_run','artifact_head','digest','expired','tree'])
def test_artifact_provenance_rejects_wrong_revision_and_hash(attack):
    run=dict(id=100,head_sha='a'*40);artifact=dict(workflow_run=dict(id=100,head_sha='a'*40),digest='sha256:'+'b'*64,expired=False)
    expected=dict(run_id=100,head='a'*40,tree='c'*40,tested_tree='c'*40)
    assert scope.verify_artifact_provenance(run,artifact,'b'*64,**expected)
    if attack=='run_id':run['id']=99
    elif attack=='run_head':run['head_sha']='d'*40
    elif attack=='artifact_run':artifact['workflow_run']['id']=99
    elif attack=='artifact_head':artifact['workflow_run']['head_sha']='d'*40
    elif attack=='digest':artifact['digest']='sha256:'+'d'*64
    elif attack=='expired':artifact['expired']=True
    elif attack=='tree':expected['tested_tree']='d'*40
    with pytest.raises(ValueError,match='REVISION_OR_HASH_MISMATCH'):scope.verify_artifact_provenance(run,artifact,'b'*64,**expected)

def test_current_nine_way_configuration_keeps_original_limits():
    workflow=yaml.load((ROOT/'.github/workflows/ci.yml').read_text(),Loader=yaml.BaseLoader)
    shard=workflow['jobs']['full-suite-tests'];gate=workflow['jobs']['full-suite']
    assert shard['strategy']['matrix']['shard']==[str(i) for i in range(1,10)]
    assert shard['timeout-minutes']=='30' and shard['strategy']['fail-fast']=='false'
    assert '--shards 9' in shard['steps'][-2]['run'] and '--shards 9' in gate['steps'][-1]['run']
    assert shard['steps'][-1]['with']['if-no-files-found']=='error'
    assert gate['steps'][0]['run']=='test "$SHARD_RESULT" = success'

def test_historical_profile_is_exact_isolated_and_needs_no_git(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('SYNTHETIC_GIT_HISTORY_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    first=historical_ncaaf_staged_capacity_profile.__wrapped__(tmp_path/'one');second=historical_ncaaf_staged_capacity_profile.__wrapped__(tmp_path/'two')
    expected=exact_predecessor_bytes(PROFILE,(ROOT/PROFILE).read_bytes().replace(b'\r\n',b'\n'))
    assert (first/PROFILE).read_bytes()==(second/PROFILE).read_bytes()==expected
    assert not (first/'.git').exists()
    (first/PROFILE).write_bytes(expected+b'UNREVIEWED')
    assert (second/PROFILE).read_bytes()==expected
    with pytest.raises(AssertionError):exact_predecessor_bytes(PROFILE,(first/PROFILE).read_bytes())

@pytest.mark.parametrize('attack',['body','binding'])
def test_predecessor_fixture_rejects_rehashed_or_changed_input(tmp_path,monkeypatch,attack):
    import tests.ncaaf_staged_capacity_fixtures as fixtures
    root=tmp_path/'SYNTHETIC-other-repository'
    for name in (PROFILE,fixtures.BINDING):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    monkeypatch.setattr(fixtures,'ROOT',root)
    assert fixtures.exact_predecessor_bytes(PROFILE,(root/PROFILE).read_bytes())
    if attack=='body':(root/PROFILE).write_bytes((root/PROFILE).read_bytes()+b' ')
    else:
        p=root/fixtures.BINDING;d=json.loads(p.read_bytes());d['source_reconstructions'][PROFILE]['after_sha256']='0'*64;p.write_text(json.dumps(d))
    with pytest.raises(AssertionError):fixtures.exact_predecessor_bytes(PROFILE,(root/PROFILE).read_bytes())

def test_original_assertions_preserved_except_exact_current_cardinality():
    binding=json.loads((ROOT/scope.BINDING).read_bytes())
    from scripts import check_launch_change_scope as g
    exempt={'test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate','test_five_shard_gate_uploads_and_reconciles_all_five','test_six_way_current_gate_preserves_limits_and_all_artifacts'}
    def assertions(raw):return {n.name:[ast.dump(a) for a in ast.walk(n) if isinstance(a,ast.Assert)] for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    for path in binding['approved_existing_test_changes']:
        raw=(ROOT/path).read_bytes().replace(b'\r\n',b'\n');prior=g._ncaaf_staged_capacity_previous_source(path,raw)
        before=assertions(prior);after=assertions(raw);assert set(before)==set(after)
        for name in before:
            if name not in exempt:assert before[name]==after[name]
