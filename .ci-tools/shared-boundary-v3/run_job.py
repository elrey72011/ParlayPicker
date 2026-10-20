"""One approved diagnostic job: cold service first; exact application phase second."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,sys,time,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
def canonical(v):return (json.dumps(v,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def terminate_group(process):
    if process.poll() is not None:return
    if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
    else:process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
        else:process.kill()
        process.wait(timeout=5)
def execute(command,log,timeout,env,cwd):
    with Path(log).open('xb') as stream:
        p=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
        try:return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_group(p);return 124
        finally:
            if p.poll() is None:terminate_group(p)
def completed(root,spec,code):
    result=root/'result.json'
    if not result.exists():return False,{}
    expected=spec['nodeids'];xml=root/'actual.xml'
    try:
        data=json.loads(result.read_bytes())
        if not (root/'collection.json').exists() or not xml.exists():
            return False,dict(data,diagnostic_readback_code='INCOMPLETE_PHASE_ARTIFACTS')
        selected=json.loads((root/'collection.json').read_bytes())['selected']
        cases=list(ET.fromstring(xml.read_bytes()).iter('testcase'))
    except (ValueError,KeyError,ET.ParseError):
        return False,dict(diagnostic_readback_code='INVALID_PHASE_ARTIFACTS')
    good=(code==0 and data.get('exit_code')==0 and data.get('exact_once') is True
        and data.get('instrumentation_cleanup') is True and data.get('selected_count')==len(expected)
        and data.get('outcomes')=={'passed':len(expected)} and not data.get('failed_phases')
        and selected==expected and len(cases)==len(expected)
        and not any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')))
    return good,data
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--service-python',required=True)
    p.add_argument('--output',required=True);p.add_argument('--job-start',required=True);a=p.parse_args()
    root=Path(a.candidate).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    raw=(HERE/'contract.json').read_bytes();d=json.loads(raw)
    assert os.environ.get('GITHUB_RUN_ATTEMPT','1')=='1','RETRY_NOT_AUTHORIZED'
    with (out/'single-dispatch-claim.json').open('xb') as f:f.write(canonical(dict(at=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(raw),attempt=1,retry=False,diagnostic_only=True)))
    # Stdlib/Git only. No application import or predecessor warm-up.
    git=lambda *args:subprocess.check_output(['git',*args],cwd=root).decode().strip()
    assert git('rev-parse','HEAD')==d['seal'] and git('rev-parse','HEAD^{tree}')==d['tree']
    assert git('rev-parse','--is-shallow-repository')=='true' and not git('status','--porcelain')
    assert git('rev-parse',d['base']+'^{tree}')=='028d3daf56d71596663a10714495a7b48d53c7ab'
    assert all(sha((root/path).read_bytes())==expected for path,expected in d['source_identities'].items())
    job_start=json.loads(Path(a.job_start).read_bytes())['monotonic'];started=time.perf_counter()
    assert 0<=started-job_start<1800
    deadline=min(started+d['active_budget_seconds'],job_start+d['job_budget_seconds']-d['artifact_reserve_seconds'])
    allowed=('PATH','HOME','LANG','LC_ALL','TZ','SHELL','TMPDIR','TEMP','TMP','CI','GITHUB_ACTIONS','RUNNER_OS','RUNNER_ARCH','DISPLAY','XDG_RUNTIME_DIR','CHROME_BIN','PYTHONUNBUFFERED')
    base_env={k:os.environ[k] for k in allowed if k in os.environ}
    results=[];known=sum(d['phases'][x]['known_active_estimate_seconds'] for x in d['phases_in_order'])
    for phase in d['phases_in_order']:
        spec=d['phases'][phase];remaining=deadline-time.perf_counter()
        if remaining<=0 or spec['known_active_estimate_seconds']>remaining or (not results and known>remaining):
            results.append(dict(phase=phase,status='NOT_STARTED_KNOWN_BUDGET_EXCEEDED',remaining_seconds=max(remaining,0),exit_code=124));break
        # Service uses isolated pinned requirements, ordinary installed plugins.
        env=dict(base_env,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=os.pathsep.join((str(HERE),str(root))),
            BOUNDARY_DIAGNOSTIC_CONTRACT=str(HERE/'contract.json'),
            BOUNDARY_DIAGNOSTIC_PHASE=phase,BOUNDARY_DIAGNOSTIC_OUTPUT=str(out/phase))
        py=a.service_python if phase=='cold-service' else sys.executable
        if phase=='cold-service':
            env['PAID_TEST_DATABASE_URL']='postgresql://subscriber:test-only@localhost:5432/subscriber_test'
        log=out/(phase+'.log')
        command=[py,'-B','-m','pytest','-p','diag_probe','-q',*spec['arguments'],
            '--tb=short','--show-capture=no','--disable-warnings','--junitxml='+str(out/phase/'actual.xml')]
        cap=min(remaining,1200) if phase=='cold-service' else remaining
        code=execute(command,log,cap,env,root)
        good,data=completed(out/phase,spec,code)
        results.append(dict(phase=phase,exit_code=code,seconds=data.get('seconds'),identities=data.get('selected_count'),
            passed=good,exact_once=data.get('exact_once',False),instrumentation_cleanup=data.get('instrumentation_cleanup',False),
            log_sha256=sha(log.read_bytes()),status='PASS_DIAGNOSTIC_ONLY' if good else 'FAILED_OR_INCOMPLETE_STOP'))
        (out/'progress.json').write_bytes(canonical(results))
        if not good:break
    assert not git('status','--porcelain')
    good=len(results)==len(d['phases_in_order']) and all(r.get('passed') is True for r in results)
    summary=dict(at=datetime.now(timezone.utc).isoformat(),seal=d['seal'],tree=d['tree'],results=results,completed=good,
        active_seconds=time.perf_counter()-started,setup_elapsed_before_supervisor_seconds=started-job_start,
        first_step_elapsed_seconds=time.perf_counter()-job_start,job_hard_limit_seconds=1800,active_cap_seconds=1560,artifact_reserve_seconds=120,
        caveat='GitHub job duration additionally includes service-container startup before the first step. Query actual completed job timestamps.',
        retries=0,provider_requests=0,release_coverage=False)
    (out/'summary.json').write_bytes(canonical(summary))
    hashes=[dict(path=f.relative_to(out).as_posix(),sha256=sha(f.read_bytes()),bytes=f.stat().st_size) for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'artifact-hashes.json').write_bytes(canonical(hashes));print(json.dumps(summary,indent=2))
    return 0 if good else 1
if __name__=='__main__':raise SystemExit(main())

