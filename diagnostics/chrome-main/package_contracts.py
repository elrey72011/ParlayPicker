"""Network-free package, selection and failure-report contracts."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,shutil,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('diagnostic_probe',ROOT/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def test_committed_manifest_and_frozen_contexts():
    for row in json.loads((ROOT/'manifest.json').read_bytes()):
        data=(ROOT/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    plans=json.loads((ROOT/'selection.json').read_bytes())['contexts']
    assert [p['name'] for p in plans]==['isolated','fullcollection-target','prefix128','phase-observed-prefix128','shared-complete']
    assert [len(p['expected_nodeids']) for p in plans]==[1,1,128,128,953]
    assert plans[2]['expected_nodeids']==plans[3]['expected_nodeids']
    assert plans[2]['expected_nodeids'][-1]==plans[0]['expected_nodeids'][0]
    assert len(set(plans[4]['expected_nodeids']))==953
    assert plans[4]['collected_count']==9194

@pytest.mark.parametrize('damage',['missing-selection','missing-observer','tampered-input'])
def test_startup_precedes_missing_or_tampered_input(tmp_path,damage):
    package=tmp_path/'package';shutil.copytree(ROOT,package,ignore=shutil.ignore_patterns('__pycache__'))
    if damage=='missing-selection':(package/'selection.json').unlink()
    elif damage=='missing-observer':(package/'chrome_main_observer.py').unlink()
    else:(package/'identity.json').write_text('unreviewed')
    env=os.environ.copy();env['CHROME_PACKAGE_MANIFEST_SHA256']=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    out=tmp_path/'evidence'
    result=subprocess.run([sys.executable,'-I','-B',str(package/'launch.py'),str(out),str(tmp_path/'absent-app')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert result.returncode==2 and not result.stdout and not result.stderr
    assert json.loads((out/'startup.json').read_bytes())['phase']=='STARTUP'
    failure=json.loads((out/'startup-failure.json').read_bytes())
    assert set(failure)=={'phase','exception_class'}

def fixture(folder):
    plan=dict(name='synthetic',phase_probe=False,expected_nodeids=['tests/test_synthetic.py::test_one'],collected_count=1,collected_sha256='frozen')
    (folder/'observation').mkdir()
    (folder/'selection-receipt.json').write_text(json.dumps(dict(collected_count=1,collected_sha256='frozen',selected_nodeids=plan['expected_nodeids'])))
    rows=[dict(kind='CASE_STARTED',nodeid=plan['expected_nodeids'][0])]
    rows.extend(dict(kind='CASE_PHASE_COMPLETED',nodeid=plan['expected_nodeids'][0],phase=p) for p in ('setup','call','teardown'))
    rows.append(dict(kind='OBSERVER_CLEANUP',threads_stopped=True,observation_errors=[]))
    (folder/'observation/events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    (folder/'observation/1-browser-process.json').write_text(json.dumps(dict(deadline_seconds=40,profile_removed=True,pipes_closed=True,cleanup=dict(direct_child_reaped=True,remaining_are_dead=True))))
    (folder/'tests.xml').write_text('<testsuites><testsuite><testcase classname="tests.test_synthetic" name="test_one" /></testsuite></testsuites>')
    return plan

def test_complete_is_separate_from_passing(tmp_path):
    plan=fixture(tmp_path)
    assert probe.assess(tmp_path,plan,0)['tests_passed']
    assert probe.assess(tmp_path,plan,1)['execution_complete']
    assert not probe.assess(tmp_path,plan,1)['tests_passed']

@pytest.mark.parametrize('damage',['missing-xml','malformed-xml','substitution','abrupt-exit','missing-phase','wrong-deadline'])
def test_partial_and_bad_evidence_blocks_success(tmp_path,damage):
    plan=fixture(tmp_path);code=0
    if damage=='missing-xml':(tmp_path/'tests.xml').unlink()
    elif damage=='malformed-xml':(tmp_path/'tests.xml').write_text('<bad')
    elif damage=='substitution':(tmp_path/'tests.xml').write_text('<testsuites><testcase classname="tests.test_synthetic" name="test_other" /></testsuites>')
    elif damage=='abrupt-exit':code=4
    elif damage=='missing-phase':
        path=tmp_path/'observation/events.jsonl';path.write_text('\n'.join(l for l in path.read_text().splitlines() if 'teardown' not in l)+'\n')
    else:
        path=tmp_path/'observation/1-browser-process.json';row=json.loads(path.read_bytes());row['deadline_seconds']=41;path.write_text(json.dumps(row))
    result=probe.assess(tmp_path,plan,code)
    assert not result['execution_complete'] and not result['tests_passed']
    assert result['started']==1

def test_selection_rejects_reorder_and_preserves_full_collection(tmp_path,monkeypatch):
    monkeypatch.setenv('CHROME_CONTEXT','prefix128');monkeypatch.setenv('CHROME_SELECTION_RECEIPT',str(tmp_path/'selection.json'))
    name='selection_test';spec=importlib.util.spec_from_file_location(name,ROOT/'chrome_main_selection.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Item:
        def __init__(self,nodeid):self.nodeid=nodeid
    class Hook:
        def pytest_deselected(self,items):self.removed=items
    class Config:hook=Hook()
    # Exact full list is deliberately not fabricated from the selected subset.
    with pytest.raises(AssertionError):module.pytest_collection_modifyitems(None,Config(),[Item(n) for n in module.CONTEXT['expected_nodeids']])
    assert not (tmp_path/'selection.json').exists()
