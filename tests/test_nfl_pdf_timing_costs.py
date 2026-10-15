"""Seven-way synthetic execution; measured costs do not qualify predictions."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
from scripts import run_ci_tests as runner
from tests.nfl_pdf_capacity_fixtures import predecessor_profile, PROFILE, ROOT, BINDING


def reports(directory):
    profile=dict(files={},fallback=runner.FALLBACK.copy(),sha256='a'*64,version=1)
    original=[SimpleNamespace(path=f'test_{i}.py',nodeid=f'test_{i}.py::test_SYNTHETIC_{j}') for i in range(7) for j in range(2)]
    for index in range(7):
        plugin=runner.FileShard(index,7,profile,manifest=directory/f'full-suite-{index+1}-assignment.json')
        items=original.copy()
        plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items:None)),items)
        xml=ET.Element('testsuite')
        for item in items:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when='call',outcome='passed'))
            file,name=item.nodeid.split('::');ET.SubElement(xml,'testcase',classname=file[:-3],name=name)
        assert plugin.finish(0)==0
        ET.ElementTree(xml).write(directory/f'full-suite-{index+1}.xml')


def test_actual_seven_shard_cli_and_all_artifacts_reconcile_once(tmp_path):
    for i in range(7):(tmp_path/f'test_{i}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1,8):
        result=subprocess.run([sys.executable,str(runner.ROOT/'scripts/run_ci_tests.py'),'--shard',str(index),'--shards','7','--assignment-manifest',str(tmp_path/f'full-suite-{index}-assignment.json'),'-q',str(tmp_path),'--junitxml='+str(tmp_path/f'full-suite-{index}.xml')],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert runner.reconcile(tmp_path,7)['completed_count']==7
    manifests=[json.loads(p.read_bytes()) for p in sorted(tmp_path.glob('*assignment.json'))]
    assert len(manifests)==7 and len({m['assignment_sha256'] for m in manifests})==1
    nodes=[n for m in manifests for n in m['completed_nodeids']]
    assert len(nodes)==len(set(nodes))==7


@pytest.mark.parametrize('attack',['missing_seven','duplicate','cancelled','empty','missing_xml','failed_xml','duplicate_outcome','wrong_count','changed_assignment','missing_outcome'])
def test_seven_way_reconciliation_fails_closed(tmp_path,attack):
    reports(tmp_path);assert runner.reconcile(tmp_path,7)['completed_count']==14
    path=tmp_path/'full-suite-7-assignment.json';data=json.loads(path.read_bytes())
    if attack=='missing_seven':path.unlink()
    elif attack=='duplicate':(tmp_path/'full-suite-8-assignment.json').write_text(json.dumps(data))
    elif attack=='missing_xml':(tmp_path/'full-suite-7.xml').unlink()
    elif attack=='failed_xml':
        p=tmp_path/'full-suite-7.xml';xml=ET.parse(p).getroot();ET.SubElement(xml.find('testcase'),'failure');ET.ElementTree(xml).write(p)
    else:
        if attack=='cancelled':data['exit_code']=2;data['execution_complete']=False
        elif attack=='empty':data['selected_count']=0
        elif attack=='duplicate_outcome':data['completed_nodeids'][1]=data['completed_nodeids'][0]
        elif attack=='wrong_count':data['shards']=6
        elif attack=='changed_assignment':data['assignments'][0]['shard']=7
        elif attack=='missing_outcome':data['completed_nodeids'].pop()
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,OSError)):runner.reconcile(tmp_path,7)


def test_seven_whole_file_assignment_is_deterministic_without_reordering():
    profile=runner.load_profile(ROOT/PROFILE)
    items=[SimpleNamespace(path=ROOT/f'tests/test_SYNTHETIC_{i}.py',nodeid=f'tests/test_SYNTHETIC_{i}.py::test_{j}') for i in range(21) for j in range(3)]
    first=runner.assignment(items,7,profile,ROOT);assert first==runner.assignment(items,7,profile,ROOT)
    rows=first[0];assert [r['nodeid'] for r in rows]==[i.nodeid for i in items]
    assert Counter(r['nodeid'] for r in rows)==Counter(i.nodeid for i in items)
    assert {r['shard'] for r in rows}==set(range(1,8))
    assert all(len({r['shard'] for r in rows if r['path']==path})==1 for path in {r['path'] for r in rows})


def test_exact_66_completed_xml_costs_and_separate_successor_metadata():
    before=json.loads(predecessor_profile());data=json.loads((ROOT/PROFILE).read_bytes())
    successor=data['nfl_pdf_seven_capacity_successor'];paths=successor['changed_paths']
    assert len(paths)==len(set(paths))==66 and successor['completed_measurements_only'] is True
    assert successor['predecessor_profile_sha256']==hashlib.sha256(predecessor_profile()).hexdigest()
    for key,value in before.items():
        if key not in {'files','sources'}:assert data[key]==value
    assert all(data['sources'][k]==v for k,v in before['sources'].items())
    old={r['path']:r for r in before['files']};rows={r['path']:r for r in data['files']}
    assert {p for p in set(old)|set(rows) if old.get(p)!=rows.get(p)}==set(paths)
    for path in paths:
        row=rows[path];observations=row['nfl_pdf_capacity_measurements']
        assert row['kind']=='estimated' and row['nfl_pdf_capacity_variance_multiplier']==1.15
        assert row['seconds']==round(max(v['seconds'] for v in observations)*1.15,6)
        assert row['profiled_tests']==max(v['count'] for v in observations)
        for value in observations:
            source=data['sources'][value['source']]
            assert value['measurement_complete'] is True and source['kind']=='completed_junit'
            assert len(source['checkout_commit'])==40 and len(source['junit_sha256'])==len(source['assignment_manifest_sha256'])==64
            assert len(value['test_git_blob'])==40 and len(value['source_sha256'])==64
    assert data['revision']==before['revision']==3


@pytest.mark.parametrize('attack',['body','binding','other_repository'])
def test_exact_predecessor_fixture_rejects_changed_or_cross_repository_input(tmp_path,attack):
    root=tmp_path/'SYNTHETIC-unreviewed-repository'
    for name in (PROFILE,BINDING):
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    assert predecessor_profile(root)==predecessor_profile()
    target=root/(BINDING if attack=='binding' else PROFILE)
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(AssertionError):predecessor_profile(root)
    assert predecessor_profile()==predecessor_profile()


def test_historical_fixture_is_isolated_and_needs_no_git_history(tmp_path,monkeypatch):
    from tests.nfl_pdf_capacity_fixtures import historical_nfl_pdf_capacity_profile
    def forbidden(*a,**kw):raise AssertionError('SYNTHETIC_GIT_HISTORY_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    first=historical_nfl_pdf_capacity_profile.__wrapped__(tmp_path/'one')
    second=historical_nfl_pdf_capacity_profile.__wrapped__(tmp_path/'two')
    original=predecessor_profile();assert (first/PROFILE).read_bytes()==(second/PROFILE).read_bytes()==original
    assert not (first/'.git').exists()
    (first/PROFILE).write_bytes(original+b' ')
    assert (second/PROFILE).read_bytes()==original and predecessor_profile()==original


def test_original_assertions_preserved_except_approved_current_cardinality():
    from scripts import check_launch_change_scope as guard
    binding=json.loads((ROOT/BINDING).read_bytes())
    changed={'tests/test_ci_test_shards.py','tests/test_ci_five_shards.py','tests/test_ci_six_shards.py','tests/test_ci_timing_profile.py','tests/test_ncaaf_socket_ci_costs.py','tests/test_football_document_ci_closure.py'}
    def assertions(raw):
        return {n.name:[ast.dump(a) for a in ast.walk(n) if isinstance(a,ast.Assert)] for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    exempt={'test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate','test_five_shard_gate_uploads_and_reconciles_all_five','test_six_way_current_gate_preserves_limits_and_all_artifacts'}
    for path in changed:
        raw=(ROOT/path).read_bytes().replace(b'\r\n',b'\n');previous=guard._nfl_pdf_timing_previous_source(path,raw)
        assert hashlib.sha256(previous).hexdigest()==binding['source_reconstructions'][path]['before_sha256']
        old=assertions(previous);current=assertions(raw);assert set(old)==set(current)
        for name in old:
            if name not in exempt:assert old[name]==current[name]
