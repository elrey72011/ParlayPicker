"""Six-shard SYNTHETIC execution and fail-closed reconciliation; no providers."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
from scripts import run_ci_tests as runner


def reports(directory):
    profile=dict(files={},fallback=runner.FALLBACK.copy(),sha256='a'*64,version=1)
    original=[SimpleNamespace(path=f'test_{i}.py',nodeid=f'test_{i}.py::test_SYNTHETIC_{j}')
              for i in range(6) for j in range(2)]
    for index in range(6):
        path=directory/f'full-suite-{index+1}-assignment.json'
        plugin=runner.FileShard(index,6,profile,manifest=path)
        selected=original.copy()
        plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items:None)),selected)
        xml=ET.Element('testsuite')
        for item in selected:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when='call',outcome='passed'))
            file,name=item.nodeid.split('::')
            ET.SubElement(xml,'testcase',classname=file[:-3],name=name)
        assert plugin.finish(0)==0
        ET.ElementTree(xml).write(directory/f'full-suite-{index+1}.xml')


def test_actual_six_shard_cli_executes_every_identity_once(tmp_path):
    for i in range(6):
        (tmp_path/f'test_{i}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1,7):
        result=subprocess.run([sys.executable,str(runner.ROOT/'scripts/run_ci_tests.py'),
            '--shard',str(index),'--shards','6','--assignment-manifest',str(tmp_path/f'full-suite-{index}-assignment.json'),
            '-q',str(tmp_path),'--junitxml='+str(tmp_path/f'full-suite-{index}.xml')],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert runner.reconcile(tmp_path,6)['completed_count']==6
    manifests=[json.loads(p.read_text()) for p in sorted(tmp_path.glob('*assignment.json'))]
    assert len({m['assignment_sha256'] for m in manifests})==1
    nodes=[node for m in manifests for node in m['completed_nodeids']]
    assert len(nodes)==len(set(nodes))==6


@pytest.mark.parametrize('attack',['missing_four','missing_five','missing_six','duplicate','cancelled','empty','missing_xml',
                                  'failed_xml','duplicate_outcome','wrong_count','changed_assignment','missing_outcome'])
def test_all_six_artifacts_and_exact_partition_required(tmp_path,attack):
    reports(tmp_path)
    assert runner.reconcile(tmp_path,6)['completed_count']==12
    path=tmp_path/'full-suite-6-assignment.json'
    data=json.loads(path.read_text())
    if attack.startswith('missing_') and attack in {'missing_four','missing_five','missing_six'}:
        missing={"missing_four":4,"missing_five":5,"missing_six":6}[attack]
        (tmp_path/f'full-suite-{missing}-assignment.json').unlink()
    elif attack=='duplicate':(tmp_path/'full-suite-7-assignment.json').write_text(json.dumps(data))
    elif attack=='missing_xml':(tmp_path/'full-suite-6.xml').unlink()
    elif attack=='failed_xml':
        xml=tmp_path/'full-suite-6.xml';root=ET.parse(xml).getroot();ET.SubElement(root.find('testcase'),'failure');ET.ElementTree(root).write(xml)
    else:
        if attack=='cancelled':data['exit_code']=2;data['execution_complete']=False
        elif attack=='empty':data['selected_count']=0
        elif attack=='duplicate_outcome':data['completed_nodeids'][1]=data['completed_nodeids'][0]
        elif attack=='wrong_count':data['shards']=3
        elif attack=='changed_assignment':data['assignments'][0]['shard']=6
        elif attack=='missing_outcome':data['completed_nodeids'].pop()
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,OSError)):runner.reconcile(tmp_path,6)


def test_six_way_current_gate_preserves_limits_and_all_artifacts():
    import yaml
    workflow=yaml.load((runner.ROOT/'.github/workflows/ci.yml').read_text(),Loader=yaml.BaseLoader)
    shard=workflow['jobs']['full-suite-tests'];gate=workflow['jobs']['full-suite']
    assert shard['strategy']['matrix']['shard']==[str(i) for i in range(1,14)]
    assert shard['strategy']['fail-fast']=='false' and shard['timeout-minutes']=='30'
    assert '--shards 13' in shard['steps'][-2]['run'] and '--shards 13' in gate['steps'][-1]['run']
    assert shard['steps'][-1]['if']=='${{ !cancelled() }}'
    assert shard['steps'][-1]['with']['if-no-files-found']=='error'
    assert gate['needs']=='full-suite-tests' and gate['steps'][0]['run']=='test "$SHARD_RESULT" = success'


def test_two_completed_file_costs_are_advisory_and_attributable(historical_nfl_pdf_capacity_profile, monkeypatch):
    monkeypatch.setattr(runner, 'DEFAULT_PROFILE', historical_nfl_pdf_capacity_profile/'scripts/ci_test_file_costs_v1.json')
    profile=json.loads(runner.DEFAULT_PROFILE.read_text())
    entries={e['path']:e for e in profile['files']}
    for path,count in (('tests/test_nfl_document_html_anchors.py',50),('tests/test_nfl_document_html_pipeline.py',8)):
        entry=entries[path]
        assert entry['kind']=='estimated' and entry['profiled_tests']==count
        assert entry['latest_measurement_complete'] and entry['incomplete_observations']==[]
        assert entry['variance_multiplier']==1.15
        assert entry['seconds']==pytest.approx(max(e['seconds'] for e in entry['completed_measurements'])*1.15)
        for reference in entry['references']:
            source=profile['sources'][reference]
            assert source['kind']=='completed_junit_testcase_sum' and len(source['checkout_commit'])==40
            assert len(source['xml_sha256'])==64 and source['run_attempt']==1




from tests.nfl_pdf_capacity_fixtures import historical_nfl_pdf_capacity_profile
