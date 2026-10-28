"""One-shot Linux investigation; completeness and passing tests are separate."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,importlib.metadata,json,os,platform,signal,subprocess,sys,threading,time
import xml.etree.ElementTree as ET

def save(path,value):
    with Path(path).open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL).decode().strip()

def identity(case,expected):
    classname=case.attrib['classname']
    for path in sorted({n.split('::')[0] for n in expected},key=len,reverse=True):
        module=path[:-3].replace('/','.')
        if classname==module or classname.startswith(module+'.'):
            rest=classname[len(module):].strip('.')
            return path+'::'+(rest.replace('.','::')+'::' if rest else '')+case.attrib['name']
    raise ValueError('UNEXPECTED_XML_IDENTITY')

def assess(folder,plan,code,timeout=False):
    issues=[];counts={};started=[];completed=[];native=[]
    try:
        expected=plan['expected_nodeids'];receipt=json.loads((folder/'selection-receipt.json').read_bytes())
        if receipt!=dict(collected_count=plan['collected_count'],collected_sha256=plan['collected_sha256'],selected_nodeids=expected):issues.append('SELECTION_MISMATCH')
        rows=[json.loads(line) for line in (folder/'observation/events.jsonl').read_text().splitlines()]
        started=[r['nodeid'] for r in rows if r['kind']=='CASE_STARTED']
        completed=[r['nodeid'] for r in rows if r['kind']=='CASE_PHASE_COMPLETED' and r['phase']=='teardown']
        if started!=expected or completed!=expected:issues.append('INCOMPLETE_OR_SUBSTITUTED_CASES')
        phases=Counter((r['nodeid'],r['phase']) for r in rows if r['kind']=='CASE_PHASE_COMPLETED')
        if any(n not in expected or count!=1 for (n,_),count in phases.items()):issues.append('DUPLICATE_OR_UNEXPECTED_PHASE')
        if any(phases[(n,'setup')]!=1 or phases[(n,'call')]!=1 or phases[(n,'teardown')]!=1 for n in expected):issues.append('MISSING_PHASE')
        cases=list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
        if [identity(c,expected) for c in cases]!=expected:issues.append('XML_IDENTITIES_MISMATCH')
        counts=dict(Counter('failed' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed' for c in cases))
        native=[json.loads(p.read_bytes()) for p in (folder/'observation').glob('*-browser-process.json')]
        if len(native)!=1:issues.append('MISSING_NATIVE_TARGET_EVIDENCE')
        for record in native:
            cleanup=record.get('cleanup',{})
            if record.get('deadline_seconds')!=40 or not record.get('profile_removed') or not record.get('pipes_closed') or not cleanup.get('direct_child_reaped') or not cleanup.get('remaining_are_dead'):issues.append('NATIVE_RESOURCE_OR_DEADLINE_FAILURE')
        if not any(r['kind']=='OBSERVER_CLEANUP' and r['threads_stopped'] and not r['observation_errors'] for r in rows):issues.append('OBSERVER_INCOMPLETE')
    except (OSError,ValueError,KeyError,TypeError,ET.ParseError) as error:
        issues.append('MISSING_OR_INVALID_EVIDENCE');issues.append(type(error).__name__)
    if timeout:issues.append('CONTEXT_TIMEOUT')
    if code not in (0,1):issues.append('ABNORMAL_PYTEST_EXIT')
    complete=not issues
    return dict(name=plan['name'],original_baseline=not plan['phase_probe'],exit_code=code,execution_complete=complete,
        tests_passed=complete and code==0 and counts=={'passed':len(plan['expected_nodeids'])},
        collected_count=plan['collected_count'] if (folder/'selection-receipt.json').exists() else None,
        started=len(started),completed=len(completed),unconfirmed=max(0,len(plan['expected_nodeids'])-len(completed)),
        counts=counts,issues=sorted(set(issues)),native= native)

def startup_gate(folder,result):
    """Only observed native renderer completion admits broader selections.

    A substantive assertion failure stays failed even if browser startup worked.
    An unknown startup, incomplete evidence or setup defect stops the plan.
    """
    records=result.get('native',[])
    stderr=b''.join(p.read_bytes()[:65536] for p in (folder/'observation').glob('*-browser-process.stderr')).lower()
    categories=[(b'socket path too long','SOCKET_PATH_TOO_LONG'),(b'no usable sandbox','BROWSER_SANDBOX_UNAVAILABLE'),
        (b'error while loading shared libraries','MISSING_BROWSER_LIBRARY'),(b'cannot open shared object file','MISSING_BROWSER_LIBRARY')]
    category=next((label for marker,label in categories if marker in stderr),None)
    if any(r.get('exception_class')== 'FileNotFoundError' for r in records):category='MISSING_EXECUTABLE'
    outputs=list((folder/'observation').glob('*-browser-process.stdout'))
    import re
    dom=bool(len(outputs)==1 and re.search(br'<pre\b[^>]*id="probability-test-result"[^>]*>.*?</pre>',outputs[0].read_bytes(),re.S))
    native_ok=len(records)==1 and records[0].get('spawned') is True and records[0].get('direct_returncode')==0 and dom
    admitted=category is None and native_ok and result.get('execution_complete') is True
    return dict(status='PASS_BROWSER_STARTUP' if admitted else 'STOP_DIAGNOSTIC_SETUP_FAILURE' if category else 'STOP_STARTUP_UNCONFIRMED',
        safe_category=category or ('NATIVE_RENDERER_COMPLETED' if admitted else 'NO_CONFIRMED_NATIVE_COMPLETION'),
        continue_broader_contexts=admitted,complete_dom_report_observed=dom,substantive_test_passed=result.get('tests_passed') is True,
        historical_timeout_cause='UNKNOWN',automatic_retries=0)

def guarded_plans(plans,results,out):
    assert plans and plans[0]['name']=='isolated' and plans[0]['mode']=='isolated'
    for index,plan in enumerate(plans):
        assert len(results)==index,'Missing context result; no broader dispatch'
        if index==1:
            gate=startup_gate(out/plans[0]['name'],results[0]);save(out/'browser-startup-gate.json',gate)
            if not gate['continue_broader_contexts']:return
        if index and 'CONTEXT_TIMEOUT' in results[-1]['issues']:return
        yield plan

def inventory(out):
    records=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in out.rglob('*') if p.is_file() and 'tmp' not in p.relative_to(out).parts and p.name!='file-identities.json']
    save(out/'file-identities.json',records)

def owned_processes(folder):
    rows={};owned=set();needle=str(folder.resolve()).encode()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split();command=(path/'cmdline').read_bytes().split(b'\0')
            pid=int(path.name);rows[pid]=dict(pid=pid,ppid=int(fields[1]),state=fields[0],start=fields[19])
            if any(needle in arg for arg in command) and any(b'chrome' in arg.lower() or b'chromium' in arg.lower() for arg in command[:1]):owned.add(pid)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    while True:
        more={pid for pid,r in rows.items() if r['ppid'] in owned}-owned
        if not more:break
        owned.update(more)
    return [rows[pid] for pid in sorted(owned) if rows[pid]['state'] not in ('Z','X')]

def cleanup_owned(folder):
    import select
    before=owned_processes(folder);handles=[]
    try:
        for row in before:
            try:
                fd=os.pidfd_open(row['pid']);handles.append(fd)
                fields=Path('/proc',str(row['pid']),'stat').read_text().rsplit(')',1)[1].split()
                if fields[19]==row['start']:signal.pidfd_send_signal(fd,signal.SIGKILL)
            except (FileNotFoundError,ProcessLookupError):continue
        pending=set(handles);until=time.monotonic()+5
        while pending and time.monotonic()<until:
            ready,_,_=select.select(list(pending),[],[],max(0,until-time.monotonic()));pending.difference_update(ready)
        return dict(running_before=before,running_after=owned_processes(folder),identity_checked=True)
    finally:
        for fd in handles:os.close(fd)

def runtime():
    import shutil
    binary=shutil.which('google-chrome') or shutil.which('chromium');assert binary
    version=subprocess.run([binary,'--version'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10,check=True)
    assert len(version.stdout)<4096
    proc={}
    for name in ('/proc/loadavg','/proc/pressure/cpu','/proc/pressure/memory','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/pids.max'):
        path=Path(name)
        if path.exists():proc[name]=path.read_text()[:4096]
    return dict(python=sys.version,platform=platform.platform(),browser_executable=binary,browser_version=version.stdout.decode().strip(),
        browser_launcher_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        runner={n:os.environ.get(n) for n in ('ImageOS','ImageVersion','RUNNER_OS','RUNNER_ARCH')},resources=proc,
        dependencies=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))

def run(app,out,*,transport_tmp):
    from transport_temp import verify_budget,protection
    verify_budget(transport_tmp);assert transport_tmp.is_dir()
    save(out/'transport-temp.json',protection(transport_tmp))
    root=Path(__file__).resolve().parent;spec=json.loads((root/'selection.json').read_bytes());frozen=json.loads((root/'identity.json').read_bytes())
    save(out/'phase.json',dict(phase='VERIFYING_CHECKOUTS',at=datetime.now(timezone.utc).isoformat()))
    assert git(app,'rev-parse','HEAD')==frozen['base'] and git(app,'rev-parse','HEAD^{tree}')==frozen['tree']
    assert git(app,'rev-parse','--is-shallow-repository')=='true' and not git(app,'status','--porcelain')
    assert subprocess.run(['git','cat-file','-e',frozen['unavailable_predecessor']+'^{commit}'],cwd=app,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    for name,row in frozen['sources'].items():
        data=(app/name).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    assert sys.platform=='linux' and sys.version_info[:3]==(3,12,15)
    assert not any(os.environ.get(n) for n in ('PARLAYPICKER_NFL_DATABASE_URL','PARLAYPICKER_NFL_STORE_KEY','ODDS_API_KEY','THE_ODDS_API_KEY','CFBD_API_KEY','CBBD_API_KEY','PARLAYPICKER_PUBLISH_TOKEN'))
    assert all(os.environ.get(n)=='false' for n in ('PARLAYPICKER_NFL_ACQUISITION_ENABLED','PARLAYPICKER_NFL_INFERENCE_ENABLED'))
    # Match the original CI import context: do not preload Streamlit through a
    # configuration plugin. This disposable runner must have no private secrets.
    assert not (Path.home()/'.streamlit/secrets.toml').exists()
    assert not (app/'.streamlit/secrets.toml').exists()
    save(out/'runtime.json',runtime());save(out/'checkouts.json',dict(application_commit=git(app,'rev-parse','HEAD'),application_tree=git(app,'rev-parse','HEAD^{tree}'),tooling_commit=git(root,'rev-parse','HEAD'),source_bindings_verified=True,shallow=True,predecessor_absent=True))
    deadline=time.monotonic()+min(2100,max(0,2700-(time.time()-float(os.environ['CHROME_JOB_STARTED']))-240));results=[]
    for plan in guarded_plans(spec['contexts'],results,out):
        folder=out/plan['name'];folder.mkdir();(folder/'tmp').mkdir()
        save(out/'phase.json',dict(phase='EXECUTING',context=plan['name'],at=datetime.now(timezone.utc).isoformat()))
        if deadline-time.monotonic()<60:break
        env=os.environ.copy();env.update(PYTHONPATH=str(root),CAPACITY_LOCAL_SESSION_ROOT=str(folder/'tmp'),CHROME_OBSERVATION_ROOT=str(folder/'observation'),
            CHROME_CONTEXT=plan['name'],CHROME_SELECTION_RECEIPT=str(folder/'selection-receipt.json'),CHROME_PHASE_PROBE='1' if plan['phase_probe'] else '0',TMP=str(transport_tmp),TEMP=str(transport_tmp),TMPDIR=str(transport_tmp))
        command=[sys.executable,'-B','-m','pytest','-q','-p','chrome_main_observer','-p','chrome_main_selection',
            plan['expected_nodeids'][0] if plan['mode']=='isolated' else 'tests','--basetemp='+str(folder/'tmp/pytest'),'--junitxml='+str(folder/'tests.xml'),'--durations=20']
        began=time.monotonic();save(folder/'command.json',dict(command=command,at=datetime.now(timezone.utc).isoformat(),automatic_retries=0));capture=dict(bytes=0,retained=0,error=None);timeout=False;code=None
        with (folder/'console.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=app,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            def drain():
                try:
                    while True:
                        block=child.stdout.read(8192)
                        if not block:break
                        capture['bytes']+=len(block);keep=block[:max(0,8*1024*1024-capture['retained'])];log.write(keep);log.flush();capture['retained']+=len(keep)
                except BaseException as error:capture['error']=type(error).__name__
                finally:child.stdout.close()
            reader=threading.Thread(target=drain,daemon=True);reader.start()
            try:code=child.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timeout=True
                try:os.killpg(child.pid,signal.SIGINT)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            finally:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=5);reader.join(timeout=5)
        cleanup=cleanup_owned(folder);save(folder/'fallback-cleanup.json',cleanup)
        result=assess(folder,plan,code,timeout);result.update(elapsed_seconds=time.monotonic()-began,at=datetime.now(timezone.utc).isoformat(),console=capture,fallback_cleanup=cleanup)
        if cleanup['running_before'] or cleanup['running_after'] or reader.is_alive() or capture['error'] or capture['bytes']!=capture['retained']:
            result.update(execution_complete=False,tests_passed=False);result['issues'].append('INCOMPLETE_CAPTURE_OR_OWNED_FALLBACK_REQUIRED')
        results.append(result);save(folder/'result.json',result);save(out/'progress.json',results);print(json.dumps(result),flush=True)
    complete=[r['name'] for r in results]==[p['name'] for p in spec['contexts']] and all(r['execution_complete'] for r in results)
    summary=dict(execution_complete=complete,all_tests_passed=complete and all(r['tests_passed'] for r in results),results=results,
        not_started_contexts=[p['name'] for p in spec['contexts'][len(results):]],
        historical_cause='UNKNOWN',product_correction='NOT_ESTABLISHED',release_qualification=False,automatic_retries=0)
    save(out/'summary.json',summary)
    inventory(out)
    return 0 if summary['all_tests_passed'] else 2
