"""Five-shard SYNTHETIC execution and fail-closed reconciliation; no providers."""
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
              for i in range(5) for j in range(2)]
    for index in range(5):
        path=directory/f'full-suite-{index+1}-assignment.json'
        plugin=runner.FileShard(index,5,profile,manifest=path)
        selected=original.copy()
        plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items:None)),selected)
        xml=ET.Element('testsuite')
        for item in selected:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when='call',outcome='passed'))
            file,name=item.nodeid.split('::')
            ET.SubElement(xml,'testcase',classname=file[:-3],name=name)
        assert plugin.finish(0)==0
        ET.ElementTree(xml).write(directory/f'full-suite-{index+1}.xml')


def test_actual_five_shard_cli_executes_every_identity_once(tmp_path):
    for i in range(5):
        (tmp_path/f'test_{i}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1,6):
        result=subprocess.run([sys.executable,str(runner.ROOT/'scripts/run_ci_tests.py'),
            '--shard',str(index),'--shards','5','--assignment-manifest',str(tmp_path/f'full-suite-{index}-assignment.json'),
            '-q',str(tmp_path),'--junitxml='+str(tmp_path/f'full-suite-{index}.xml')],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert runner.reconcile(tmp_path,5)['completed_count']==5
    manifests=[json.loads(p.read_text()) for p in sorted(tmp_path.glob('*assignment.json'))]
    assert len({m['assignment_sha256'] for m in manifests})==1
    nodes=[node for m in manifests for node in m['completed_nodeids']]
    assert len(nodes)==len(set(nodes))==5


@pytest.mark.parametrize('attack',['missing_four','missing_five','duplicate','cancelled','empty','missing_xml',
                                  'failed_xml','duplicate_outcome','wrong_count','changed_assignment','missing_outcome'])
def test_all_five_artifacts_and_exact_partition_required(tmp_path,attack):
    reports(tmp_path)
    assert runner.reconcile(tmp_path,5)['completed_count']==10
    path=tmp_path/'full-suite-5-assignment.json'
    data=json.loads(path.read_text())
    if attack.startswith('missing_') and attack in {'missing_four','missing_five'}:
        (tmp_path/f'full-suite-{4 if attack=="missing_four" else 5}-assignment.json').unlink()
    elif attack=='duplicate':(tmp_path/'full-suite-6-assignment.json').write_text(json.dumps(data))
    elif attack=='missing_xml':(tmp_path/'full-suite-5.xml').unlink()
    elif attack=='failed_xml':
        xml=tmp_path/'full-suite-5.xml';root=ET.parse(xml).getroot();ET.SubElement(root.find('testcase'),'failure');ET.ElementTree(root).write(xml)
    else:
        if attack=='cancelled':data['exit_code']=2;data['execution_complete']=False
        elif attack=='empty':data['selected_count']=0
        elif attack=='duplicate_outcome':data['completed_nodeids'][1]=data['completed_nodeids'][0]
        elif attack=='wrong_count':data['shards']=3
        elif attack=='changed_assignment':data['assignments'][0]['shard']=5
        elif attack=='missing_outcome':data['completed_nodeids'].pop()
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError,OSError)):runner.reconcile(tmp_path,5)


def test_five_shard_gate_uploads_and_reconciles_all_five(tmp_path):
    import yaml
    workflow=yaml.load((runner.ROOT/'.github/workflows/ci.yml').read_text(),Loader=yaml.BaseLoader)
    shards=workflow['jobs']['full-suite-tests'];gate=workflow['jobs']['full-suite']
    assert shards['strategy']['matrix']['shard']==[str(i) for i in range(1,14)]
    assert shards['strategy']['fail-fast']=='false'
    assert shards['timeout-minutes']=='40'
    assert '--shards 13' in shards['steps'][-2]['run']
    upload=shards['steps'][-1]
    assert upload['if']=='${{ !cancelled() }}'
    assert upload['with']['if-no-files-found']=='error'
    assert upload['with']['name']=='full-suite-test-results-${{ matrix.shard }}'
    assert gate['needs']=='full-suite-tests' and gate['if']=='${{ always() }}'
    assert gate['steps'][0]['run']=='test "$SHARD_RESULT" = success'
    assert '--shards 13' in gate['steps'][-1]['run']
    reports(tmp_path)
    assert runner.reconcile(tmp_path,5)['shards']==5
