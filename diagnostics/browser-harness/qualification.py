"""Diagnostic-only execution and fail-closed interpretation of frozen contexts."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,signal,subprocess,sys,time,threading
import xml.etree.ElementTree as ET

PHASE='LOADING_INPUTS'

def save(path,value):
    with Path(path).open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def assess(folder,plan,code,*,timeout=False,io_complete=True):
    folder=Path(folder);issues=[];counts={};resources=False
    result=dict(name=plan['name'],role=plan['mode'],exit_code=code)
    try:
        expected=plan['expected_nodeids'];receipt=json.loads((folder/'selection-receipt.json').read_bytes())
        if any(receipt.get(k)!=v for k,v in dict(selected_nodeids=expected,files=plan['files'],selected_count=len(expected),collected_count=plan['expected_collected_count']).items()):issues.append('SELECTION_MISMATCH')
        def identity(case):
            classname=case.attrib['classname']
            for path in sorted(plan['files'],key=len,reverse=True):
                module=path[:-3].replace('/','.')
                if classname==module or classname.startswith(module+'.'):
                    suffix=classname[len(module):].strip('.')
                    return path+'::'+(suffix.replace('.','::')+'::' if suffix else '')+case.attrib['name']
            raise ValueError('UNEXPECTED_XML_IDENTITY')
        cases=list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
        actual=[identity(c) for c in cases]
        if actual!=expected or len(actual)!=len(set(actual)):issues.append('XML_IDENTITIES_MISMATCH')
        counts=dict(Counter('failed' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed' for c in cases))
        rows=[json.loads(s) for s in (folder/'outcomes.jsonl').read_text().splitlines()]
        phases=Counter((r['nodeid'],r['when']) for r in rows)
        if any(n not in expected or count!=1 for (n,_),count in phases.items()):issues.append('INVALID_PHASE_OUTCOMES')
        for node in expected:
            if phases[(node,'setup')]!=1 or phases[(node,'teardown')]!=1:issues.append('MISSING_PHASE_OUTCOME');break
            setup=next(r for r in rows if r['nodeid']==node and r['when']=='setup')
            if setup['outcome']=='passed' and phases[(node,'call')]!=1:issues.append('MISSING_CALL_OUTCOME');break
        launches=[json.loads(s) for s in (folder/'launches.jsonl').read_text().splitlines()]
        if len(launches)!=plan['expected_native_invocations']:issues.append('NATIVE_INVOCATIONS_MISMATCH')
        if not all(r['profile_under_owned_root'] for r in launches):issues.append('UNOWNED_BROWSER_PROFILE')
        live=[p for r in rows for p in r.get('owned_browser_processes',[]) if p['state'] not in ('Z','X')]
        native=[r for p in folder.glob('*.browser-process.json') if (r:=json.loads(p.read_bytes())).get('dom_marker_observed')]
        def clean(r):
            return (r.get('deadline_seconds')==40 and r.get('direct_returncode')==0 and r.get('pipes_closed') is True and r.get('profile_removed') is True
                and r.get('cleanup',{}).get('direct_child_reaped') is True and r.get('cleanup',{}).get('remaining_are_dead') is True
                and all(p['state'] in ('Z','X') for p in r['cleanup'].get('members_after',[]))
                and len(r.get('output',[]))==2 and all(not c.get('truncated') and c.get('reader_exception_class') is None for c in r['output']))
        resources=not live and len(native)==plan['expected_native_invocations'] and all(clean(r) for r in native)
        result.update(native_invocations=len(launches),native_evidence_records=len(native),running_owned_process_observations=len(live),native_resources_verified=resources)
    except (OSError,ValueError,KeyError,TypeError,ET.ParseError) as error:
        issues.append('MISSING_OR_INVALID_REQUIRED_EVIDENCE');result['evidence_exception_class']=type(error).__name__
    if timeout:issues.append('CONTEXT_TIMEOUT')
    if not io_complete:issues.append('CONSOLE_CAPTURE_INCOMPLETE')
    if code not in (0,1):issues.append('PYTEST_DID_NOT_COMPLETE_NORMALLY')
    complete=not issues
    passed=complete and code==0 and counts.get('passed',0)==len(plan['expected_nodeids']) and not counts.get('failed') and not counts.get('skipped')
    result.update(execution_complete=complete,tests_passed=passed,outcome_counts=counts,issues=sorted(set(issues)),qualified=bool(plan['mode']=='candidate' and passed and resources))
    return result

def summarize(plans,results):
    complete=[r['name'] for r in results]==[p['name'] for p in plans] and all(r['execution_complete'] for r in results)
    candidates=[r for r in results if r['role']=='candidate']
    success=len(candidates)==3 and all(r['tests_passed'] for r in candidates)
    return dict(execution_complete=complete,candidate_tests_passed=success,qualification_passed=complete and success and all(r['qualified'] for r in candidates),
        baseline_failed_tests=sum(r.get('outcome_counts',{}).get('failed',0) for r in results if r['role']=='baseline'),baseline_passes_counted_as_candidate_passes=0,
        outcomes=results,automatic_retries=0,historical_failure_cause='UNKNOWN',note='Original behavioral assertions must pass; a DOM marker alone is insufficient. This is not release clearance.')

def owned_processes(folder):
    """Identify only current-context profile owners and their extant descendants."""
    rows={};owned=set();needle=str(Path(folder).resolve()).encode()
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
    # A fallback never turns a resource leak into qualification. pidfds and a
    # second start-time check prevent signalling a reused/unrelated process ID.
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
            ready,_,_=select.select(list(pending),[],[],max(0,until-time.monotonic()))
            pending.difference_update(ready)
        return dict(running_before=before,running_after=owned_processes(folder),scope='context-profile-owner-and-descendants',pidfd_identity_checked=True)
    finally:
        for fd in handles:os.close(fd)

def run(workspace,out):
    import shutil,platform,importlib.metadata
    global PHASE
    workspace=Path(workspace);out=Path(out)
    def phase(name):
        global PHASE
        PHASE=name;save(out/'phase.json',dict(phase=name,at=datetime.now(timezone.utc).isoformat()))
    def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL).decode().strip()
    spec=json.loads((Path(__file__).parent/'selection.json').read_bytes());identity=json.loads((Path(__file__).parent/'candidate.json').read_bytes())
    plans=spec['required_contexts']
    assert [p['name'] for p in plans]==['baseline-10','baseline-9','candidate-10','candidate-9','candidate-shared']
    for p in plans:assert len(p['expected_nodeids'])==p['expected_count']==len(set(p['expected_nodeids']))
    baseline=workspace/'baseline-app';candidate=workspace/'candidate-app';phase('VERIFYING_CHECKOUTS')
    assert git(baseline,'rev-parse','HEAD')==identity['failed_head']
    assert git(candidate,'rev-parse','HEAD')==identity['head']
    assert git(candidate,'rev-parse','HEAD^{tree}')==identity['tree']
    for root in (baseline,candidate):
        assert git(root,'rev-parse','--is-shallow-repository')=='true'
        assert not git(root,'status','--porcelain')
        assert subprocess.run(['git','cat-file','-e',identity['base']+'^{commit}'],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    assert sys.platform=='linux'
    assert not any(os.environ.get(n) for n in ('PARLAYPICKER_NFL_DATABASE_URL','PARLAYPICKER_NFL_STORE_KEY','ODDS_API_KEY','CFBD_API_KEY','CBBD_API_KEY'))
    assert all(os.environ.get(n)=='false' for n in ('PARLAYPICKER_NFL_ACQUISITION_ENABLED','PARLAYPICKER_NFL_INFERENCE_ENABLED'))
    phase('IMPORTING_BROWSER_OWNER');sys.path.insert(0,str(candidate))
    from tests.browser_process_owner import run_owned
    phase('BROWSER_VERSION_PROBE');binary=shutil.which('google-chrome') or shutil.which('chromium');assert binary
    probe={}
    try:version=run_owned([binary,'--version'],timeout=10,evidence=probe)
    finally:
        for field in ('stdout','stderr'):(out/('version-probe.'+field)).write_bytes(probe.pop('_'+field,b''))
        save(out/'version-probe.json',probe)
    assert version.returncode==0
    save(out/'runtime.json',dict(python=sys.version,platform=platform.platform(),browser_executable=binary,browser_version=version.stdout.decode().strip(),
        candidate=identity,baseline_tree=git(baseline,'rev-parse','HEAD^{tree}'),dependencies=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),
        settings_present=False,nfl_acquisition_enabled=False,nfl_inference_enabled=False,provider_requests=0,database_connections=0))
    save(out/'frozen-plan.json',plans)
    deadline=time.monotonic()+min(2280,max(0,2700-(time.time()-float(os.environ['BROWSER_DIAGNOSTIC_JOB_STARTED']))-180));results=[]
    for plan in plans:
        name=plan['name'];folder=out/name;folder.mkdir();phase('EXECUTING_'+name)
        if deadline-time.monotonic()<60:break
        env=os.environ.copy();env.update(BROWSER_PROBE_EVIDENCE=str(folder),PYTHONPATH=str(Path(__file__).parent),BROWSER_PROBE_MANIFEST=str(Path(__file__).parent/'selection.json'),BROWSER_PROBE_CONTEXT=name)
        command=[sys.executable,'-B','-m','pytest','-q','-p','browser_probe_observer','tests','--basetemp='+str(folder/'tmp'),'--junitxml='+str(folder/'tests.xml'),'--durations=20']
        root=candidate if plan['mode']=='candidate' else baseline
        save(folder/'command.json',dict(command=command,checkout=str(root),started_at=datetime.now(timezone.utc).isoformat()))
        began=time.monotonic();timeout=False;code=None;capture=dict(total_bytes=0,retained_bytes=0,error=None)
        with (folder/'console.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            def drain():
                try:
                    while True:
                        block=child.stdout.read(8192)
                        if not block:break
                        capture['total_bytes']+=len(block);room=max(0,8*1024*1024-capture['retained_bytes']);log.write(block[:room]);capture['retained_bytes']+=min(room,len(block))
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
        item=assess(folder,plan,code,timeout=timeout,io_complete=not reader.is_alive() and capture['error'] is None and capture['total_bytes']==capture['retained_bytes'])
        item['fallback_cleanup']=cleanup
        if cleanup['running_before'] or cleanup['running_after']:
            item['qualified']=False;item['issues'].append('OWNED_PROCESS_FALLBACK_REQUIRED')
        item.update(elapsed_seconds=time.monotonic()-began,completed_at=datetime.now(timezone.utc).isoformat(),console=capture)
        results.append(item);save(folder/'result.json',item);save(out/'progress.json',results);print(json.dumps(item),flush=True)
        if timeout:break
    phase('FINALIZING');summary=summarize(plans,results);save(out/'diagnostic-summary.json',summary)
    save(out/'file-identities.json',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.rglob('*') if p.is_file() and 'tmp' not in p.relative_to(out).parts])
    return 0 if summary['qualification_passed'] else 2
