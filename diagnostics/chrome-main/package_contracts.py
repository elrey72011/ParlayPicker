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

def load_transport():
    spec=importlib.util.spec_from_file_location('short_transport',ROOT/'transport_temp.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_observed_long_socket_path_rejects_before_browser():
    module=load_transport()
    observed='/home/runner/work/ParlayPicker/ParlayPicker/chrome-evidence/isolated/tmp'
    with pytest.raises(ValueError,match='^CHROME_TRANSPORT_TMP_PATH_TOO_LONG$'):module.verify_budget(observed)
    module.verify_budget('/tmp/pc-12345678')

@pytest.mark.parametrize('exceptional',[False,True])
def test_short_owned_transport_root_cleans_and_preserves_unrelated(exceptional):
    import tempfile
    module=load_transport()
    parent=Path(os.environ.get('CHROME_LOCAL_SHORT_PARENT','/tmp'));parent.mkdir(parents=True,exist_ok=True)
    sentinel=tempfile.NamedTemporaryFile(prefix='unrelated-',dir=parent,delete=False);sentinel.close();sentinel=Path(sentinel.name)
    root=None
    try:
        try:
            with module.short_transport_tmp(parent) as root:
                assert root.is_dir() and sentinel.exists()
                (root/'owned.txt').write_text('synthetic')
                if exceptional:raise RuntimeError('SYNTHETIC_CONTEXT_FAILURE')
        except RuntimeError as error:
            assert exceptional and error.args==('SYNTHETIC_CONTEXT_FAILURE',)
        assert root is not None and not root.exists() and sentinel.exists()
    finally:sentinel.unlink()

def test_overlong_root_is_removed_on_pre_yield_failure(tmp_path):
    module=load_transport();parent=tmp_path/('owned-'+'x'*90);parent.mkdir()
    before=list(parent.iterdir())
    with pytest.raises(ValueError,match='^CHROME_TRANSPORT_TMP_PATH_TOO_LONG$'):
        with module.short_transport_tmp(parent):raise AssertionError('Must not yield')
    assert list(parent.iterdir())==before

def test_transport_and_evidence_locations_stay_separate():
    import ast
    tree=ast.parse((ROOT/'probe.py').read_bytes())
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='update']
    environments=[n for n in assignments if {k.arg for k in n.keywords}>={'TMP','TEMP','TMPDIR'}]
    assert len(environments)==1
    assert all(ast.unparse(k.value)=='str(transport_tmp)' for k in environments[0].keywords if k.arg in ('TMP','TEMP','TMPDIR'))
    assert "'--basetemp='+str(folder/'tmp/pytest')" in (ROOT/'probe.py').read_text()

def startup_fixture(folder):
    plan=fixture(folder)
    path=folder/'observation/1-browser-process.json';row=json.loads(path.read_bytes())
    row.update(spawned=True,direct_returncode=0);path.write_text(json.dumps(row))
    (folder/'observation/1-browser-process.stdout').write_bytes(b'<pre id="probability-test-result">{}</pre>')
    return probe.assess(folder,plan,0)

@pytest.mark.parametrize('failure',['socket','missing-executable','missing-library','unconfirmed','timeout'])
def test_startup_defect_or_unconfirmed_completion_stops_broader_dispatch(tmp_path,failure):
    folder=tmp_path/'isolated';folder.mkdir();result=startup_fixture(folder)
    result.update(tests_passed=False,counts={'failed':1});result['native'][0]['direct_returncode']=-6
    (folder/'observation/1-browser-process.stdout').write_bytes(b'')
    if failure=='socket':(folder/'observation/1-browser-process.stderr').write_bytes(b'FATAL: Socket path too long')
    elif failure=='missing-executable':result['native'][0]['exception_class']='FileNotFoundError'
    elif failure=='missing-library':(folder/'observation/1-browser-process.stderr').write_bytes(b'error while loading shared libraries')
    elif failure=='timeout':result['native'][0]['phase']='EXECUTION_TIMEOUT'
    plans=[dict(name='isolated',mode='isolated'),dict(name='prefix128',mode='prefix')]
    results=[];calls=[]
    for plan in probe.guarded_plans(plans,results,tmp_path):
        calls.append(plan['name']);results.append(result)
    assert calls==['isolated'] and results[0] is result and not results[0]['tests_passed']
    gate=json.loads((tmp_path/'browser-startup-gate.json').read_bytes())
    assert not gate['continue_broader_contexts'] and gate['status'].startswith('STOP_')
    assert gate['safe_category']=='SOCKET_PATH_TOO_LONG' if failure=='socket' else True

@pytest.mark.parametrize('substantive_pass',[False,True])
def test_observed_renderer_admits_contexts_without_erasing_assertion_failures(tmp_path,substantive_pass):
    folder=tmp_path/'isolated';folder.mkdir();result=startup_fixture(folder)
    result['tests_passed']=substantive_pass
    plans=[dict(name='isolated',mode='isolated'),dict(name='prefix128',mode='prefix'),dict(name='shared-complete',mode='shared')]
    calls=[];results=[]
    for plan in probe.guarded_plans(plans,results,tmp_path):calls.append(plan['name']);results.append(copy.deepcopy(result))
    assert calls==[p['name'] for p in plans]
    gate=json.loads((tmp_path/'browser-startup-gate.json').read_bytes())
    assert gate['status']=='PASS_BROWSER_STARTUP' and gate['substantive_test_passed'] is substantive_pass
    assert all(r['tests_passed'] is substantive_pass for r in results)

@pytest.mark.parametrize('mode,uid,accepted',[(0o700,41,True),(0o755,41,False),(0o700,42,False)])
def test_posix_private_mode_and_uid_required(tmp_path,monkeypatch,mode,uid,accepted):
    import stat
    from types import SimpleNamespace
    module=load_transport()
    class FakePath:
        def lstat(self):return SimpleNamespace(st_mode=stat.S_IFDIR|mode,st_uid=uid)
        def resolve(self):return Path('/tmp/pc-synthetic')
    monkeypatch.setattr(module,'Path',lambda value:FakePath())
    monkeypatch.setattr(module,'os',SimpleNamespace(name='posix',geteuid=lambda:41,fsencode=os.fsencode))
    if accepted:
        value=module.protection(tmp_path);assert value['posix_mode']=='0o700' and value['current_uid_owned'] is True
    else:
        with pytest.raises(ValueError,match='^CHROME_TRANSPORT_TMP_NOT_PRIVATE$'):module.protection(tmp_path)

def test_final_inventory_preserves_cleanup_receipt_and_excludes_itself_and_tmp(tmp_path):
    (tmp_path/'tmp').mkdir();(tmp_path/'tmp/private-fixture').write_text('synthetic')
    (tmp_path/'transport-temp-cleanup.json').write_text('{"owned_root_removed": true}')
    probe.inventory(tmp_path);probe.inventory(tmp_path)
    rows=json.loads((tmp_path/'file-identities.json').read_bytes())
    assert [r['path'] for r in rows]==['transport-temp-cleanup.json']
