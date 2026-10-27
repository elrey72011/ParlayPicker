"""Synthetic offline qualification contracts; never application execution."""
from pathlib import Path
import json,shutil,subprocess,sys
import pytest
import qualification as q

def test_owned_snapshot_excludes_unrelated_processes(tmp_path,monkeypatch):
    native=Path;proc=tmp_path/'proc';proc.mkdir();folder=tmp_path/'context';folder.mkdir()
    def entry(pid,parent,command):
        p=proc/str(pid);p.mkdir();fields=['S',str(parent),'0','0']+['0']*15+[str(pid*10)]
        (p/'stat').write_text(str(pid)+' (example) '+' '.join(fields));(p/'cmdline').write_bytes(command)
    entry(100,1,b'/usr/bin/google-chrome\0--user-data-dir='+str(folder).encode()+b'/profile\0')
    entry(101,100,b'/usr/bin/renderer\0');entry(200,1,b'/usr/bin/google-chrome\0--user-data-dir=/unrelated\0')
    def path(*parts):return proc.joinpath(*parts[1:]) if parts and parts[0]=='/proc' else native(*parts)
    monkeypatch.setattr(q,'Path',path)
    assert [p['pid'] for p in q.owned_processes(folder)]==[100,101]

@pytest.mark.parametrize('mismatch',[False,True])
def test_selector_validates_full_collection_before_fixed_order(tmp_path,monkeypatch,mismatch):
    import importlib.util
    from types import SimpleNamespace
    monkeypatch.setenv('BROWSER_PROBE_EVIDENCE',str(tmp_path/'evidence'))
    manifest=tmp_path/'manifest.json'
    q.save(manifest,dict(required_contexts=[dict(name='example',files=['tests/test_example.py'],expected_nodeids=['tests/test_example.py::test_one'],expected_count=1,expected_collected_count=3 if mismatch else 2)]))
    monkeypatch.setenv('BROWSER_PROBE_MANIFEST',str(manifest));monkeypatch.setenv('BROWSER_PROBE_CONTEXT','example')
    spec=importlib.util.spec_from_file_location('selector_contract_'+str(mismatch),Path(__file__).with_name('browser_probe_observer.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=SimpleNamespace(rootpath=tmp_path,hook=SimpleNamespace(pytest_deselected=lambda **kwargs:None))
    items=[SimpleNamespace(path=tmp_path/'tests/test_other.py',nodeid='tests/test_other.py::test_two'),SimpleNamespace(path=tmp_path/'tests/test_example.py',nodeid='tests/test_example.py::test_one')]
    if mismatch:
        with pytest.raises(pytest.UsageError):module.pytest_collection_modifyitems(config,items)
        assert (tmp_path/'evidence/selection-failure.json').exists()
    else:
        module.pytest_collection_modifyitems(config,items)
        assert [i.nodeid for i in items]==['tests/test_example.py::test_one']

def fixture(tmp_path,mode='candidate'):
    plan=dict(name=mode+'-synthetic',mode=mode,files=['tests/test_example.py'],expected_nodeids=['tests/test_example.py::test_one'],expected_count=1,expected_collected_count=100,expected_native_invocations=1)
    q.save(tmp_path/'selection-receipt.json',dict(selected_nodeids=plan['expected_nodeids'],selected_count=1,collected_count=100,files=plan['files']))
    (tmp_path/'tests.xml').write_text('<testsuites><testsuite><testcase classname="tests.test_example" name="test_one"/></testsuite></testsuites>')
    (tmp_path/'outcomes.jsonl').write_text(''.join(json.dumps(dict(nodeid=plan['expected_nodeids'][0],when=phase,outcome='passed',owned_browser_processes=[]))+'\n' for phase in ('setup','call','teardown')))
    (tmp_path/'launches.jsonl').write_text(json.dumps(dict(profile_under_owned_root=True))+'\n')
    q.save(tmp_path/'one.browser-process.json',dict(dom_marker_observed=True,deadline_seconds=40,direct_returncode=0,pipes_closed=True,profile_removed=True,cleanup=dict(direct_child_reaped=True,remaining_are_dead=True,members_after=[]),output=[dict(truncated=False,reader_exception_class=None)]*2))
    return plan

def test_complete_candidate_success(tmp_path):
    p=fixture(tmp_path);r=q.assess(tmp_path,p,0)
    assert r['execution_complete'] and r['tests_passed'] and r['qualified']

@pytest.mark.parametrize('change',['nonzero','failed','skipped','missing_xml','missing_teardown','duplicate','wrong_order','wrong_collection','missing_launch','resource_leak','open_pipe','timeout','io','interrupted'])
def test_candidate_rejections(tmp_path,change):
    p=fixture(tmp_path);code=0;kwargs={}
    if change=='nonzero':code=1
    elif change in ('failed','skipped'):
        tag='failure' if change=='failed' else 'skipped'
        (tmp_path/'tests.xml').write_text('<testsuites><testsuite><testcase classname="tests.test_example" name="test_one"><'+tag+'/></testcase></testsuite></testsuites>')
    elif change=='missing_xml':(tmp_path/'tests.xml').unlink()
    elif change=='missing_teardown':(tmp_path/'outcomes.jsonl').write_text('\n'.join((tmp_path/'outcomes.jsonl').read_text().splitlines()[:2])+'\n')
    elif change=='duplicate':
        with (tmp_path/'outcomes.jsonl').open('a') as f:f.write((tmp_path/'outcomes.jsonl').read_text().splitlines()[0]+'\n')
    elif change=='wrong_order':p['expected_nodeids']=['tests/test_example.py::test_other']
    elif change=='wrong_collection':p['expected_collected_count']=101
    elif change=='missing_launch':(tmp_path/'launches.jsonl').unlink()
    elif change in ('resource_leak','open_pipe'):
        r=json.loads((tmp_path/'one.browser-process.json').read_bytes())
        if change=='resource_leak':r['cleanup']['members_after']=[dict(pid=123,state='S')]
        else:r['pipes_closed']=False
        q.save(tmp_path/'one.browser-process.json',r)
    elif change=='timeout':kwargs['timeout']=True
    elif change=='io':kwargs['io_complete']=False
    elif change=='interrupted':code=2
    assert not q.assess(tmp_path,p,code,**kwargs)['qualified']

def test_baseline_failure_complete_but_never_candidate_pass(tmp_path):
    p=fixture(tmp_path,'baseline')
    (tmp_path/'tests.xml').write_text('<testsuites><testsuite><testcase classname="tests.test_example" name="test_one"><failure/></testcase></testsuite></testsuites>')
    r=q.assess(tmp_path,p,1)
    assert r['execution_complete'] and not r['tests_passed'] and not r['qualified']
    plans=[dict(name=n) for n in ('baseline-10','baseline-9','candidate-10','candidate-9','candidate-shared')]
    results=[dict(r,name=x['name']) for x in plans[:2]]+[dict(name=x['name'],role='candidate',execution_complete=True,tests_passed=True,qualified=True,outcome_counts={'passed':1}) for x in plans[2:]]
    s=q.summarize(plans,results)
    assert s['qualification_passed'] and s['baseline_failed_tests']==2 and s['baseline_passes_counted_as_candidate_passes']==0
    assert not q.summarize(plans,results[:-1])['qualification_passed']
    results[-1]['qualified']=False
    s=q.summarize(plans,results);assert s['candidate_tests_passed'] and not s['qualification_passed']

@pytest.mark.parametrize('failure',['import','missing_selection','missing_candidate','invalid_selection','missing_contexts'])
def test_startup_receipt_and_sanitized_failure(tmp_path,failure,monkeypatch):
    here=Path(__file__).parent;tools=tmp_path/'tools';tools.mkdir();workspace=tmp_path/'workspace';workspace.mkdir()
    shutil.copy(here/'browser_harness_diagnostic.py',tools)
    if failure=='import':(tools/'qualification.py').write_text('raise ImportError("SYNTHETIC_SECRET_PASSWORD_URL_HOST")\n')
    else:
        shutil.copy(here/'qualification.py',tools)
        (tools/'selection.json').write_text('{"required_contexts": []}')
        (tools/'candidate.json').write_text('{}')
        if failure=='missing_selection':(tools/'selection.json').unlink()
        elif failure=='missing_candidate':(tools/'candidate.json').unlink()
        elif failure=='invalid_selection':(tools/'selection.json').write_text('SYNTHETIC_SECRET_PASSWORD_URL_HOST')
        elif failure=='missing_contexts':(tools/'selection.json').write_text('{}')
    monkeypatch.setenv('GITHUB_WORKSPACE',str(workspace))
    result=subprocess.run([sys.executable,'-B',str(tools/'browser_harness_diagnostic.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert result.returncode==2 and not result.stderr
    out=workspace/'browser-evidence';assert (out/'startup.json').exists()
    value=json.loads((out/'startup-failure.json').read_bytes());assert value['raw_exception_retained'] is False
    assert b'SYNTHETIC_SECRET' not in result.stdout and b'SYNTHETIC_SECRET' not in b''.join(p.read_bytes() for p in out.iterdir())
