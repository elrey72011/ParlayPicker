"""One-shot bounded diagnostic supervisor. No retries and no release coverage."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,sys,time

HERE=Path(__file__).resolve().parent
def canonical(value):return (json.dumps(value,indent=2,ensure_ascii=True)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def terminate_group(process):
    if process.poll() is not None:return
    if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
    else:process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
        else:process.kill()
        process.wait(timeout=5)
def execute(command,log,timeout,env):
    with Path(log).open('xb') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
        try:return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_group(process);return 124
        finally:
            if process.poll() is None:terminate_group(process)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',choices=['a','b'],required=True);p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);p.add_argument('--job-start',required=True)
    args=p.parse_args();raw=(HERE/'contract.json').read_bytes();contract=json.loads(raw)
    assert os.environ.get('GITHUB_RUN_ATTEMPT','1')=='1'
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    claim=out/'dispatch-claim.json'
    with claim.open('xb') as f:f.write(canonical(dict(at=datetime.now(timezone.utc).isoformat(),job=args.job,attempt=1,no_retry=True,contract_sha256=digest(raw))))
    job_started=json.loads(Path(args.job_start).read_bytes())['monotonic']
    started=time.perf_counter();deadline=min(started+1560,job_started+1800-contract['artifact_reserve_seconds'])
    assert 0<=started-job_started<1800, 'INVALID_JOB_START_CLOCK'
    known_total=sum(contract['phases'][name]['known_active_estimate_seconds'] for name in contract['jobs'][args.job])
    # Ordinary runtime/CI identity only. Provider/account/secret environment is not copied.
    allowed=('PATH','HOME','LANG','LC_ALL','TZ','SHELL','TMPDIR','TEMP','TMP','CI','GITHUB_ACTIONS','RUNNER_OS','RUNNER_ARCH','DISPLAY','XDG_RUNTIME_DIR','CHROME_BIN','PYTHONUNBUFFERED')
    env={k:os.environ[k] for k in allowed if k in os.environ};env['PYTHONUNBUFFERED']='1'
    results=[]
    for phase in contract['jobs'][args.job]:
        spec=contract['phases'][phase];root=args.baseline if spec['baseline'] else args.candidate
        remaining=deadline-time.perf_counter();estimate=spec['known_active_estimate_seconds']
        if remaining<=0 or estimate>remaining or (not results and known_total>remaining):
            results.append(dict(phase=phase,status='NOT_STARTED_KNOWN_BUDGET_EXCEEDED',remaining_seconds=max(remaining,0),known_estimate_seconds=estimate,exit_code=124));break
        log=out/(phase+'.log')
        code=execute([sys.executable,'-B',str(HERE/'probe.py'),'--contract',str(HERE/'contract.json'),'--phase',phase,'--root',root,'--output',str(out/phase)],log,remaining,env)
        path=out/phase/'result.json';data=json.loads(path.read_bytes()) if path.exists() else {}
        bad=data.get('failed_phases',[])
        baseline_observation=(spec['baseline'] and code==1 and data.get('executed_once',False)
            and len(bad)==1 and bad[0]['nodeid']=='tests/test_drive_history_loading.py::test_actual_completed_analysis_publish_caller_defers_history'
            and bad[0]['phase']=='call' and bad[0]['failure_code']=='APPTEST_HARNESS_TIMEOUT'
            and data.get('attribution_cleanup_pass',False) and data.get('capture_cleanup_pass',False))
        row=dict(phase=phase,exit_code=code,seconds=data.get('seconds'),executed_once=data.get('executed_once',False),
            attribution_cleanup_pass=data.get('attribution_cleanup_pass',False),baseline_observation=baseline_observation,
            capture_cleanup_pass=data.get('capture_cleanup_pass',False),
            log_sha256=digest(log.read_bytes()),passing_candidate=not spec['baseline'] and code==0 and data.get('executed_once',False))
        results.append(row);(out/'progress.json').write_bytes(canonical(results))
        if not (code==0 and data.get('executed_once',False) and data.get('attribution_cleanup_pass',False) and data.get('capture_cleanup_pass',False)) and not baseline_observation:break
    complete=len(results)==len(contract['jobs'][args.job]) and all((r['exit_code']==0 or r.get('baseline_observation')) and r.get('executed_once') for r in results)
    result=dict(at=datetime.now(timezone.utc).isoformat(),job=args.job,results=results,completed=complete,seconds=time.perf_counter()-started,
        setup_elapsed_before_supervisor_seconds=started-job_started,job_elapsed_seconds=time.perf_counter()-job_started,
        original_active_cap_seconds=1560,artifact_reserve_seconds=contract['artifact_reserve_seconds'],
        effective_active_budget_seconds=max(0,deadline-started),no_retries=True,historical_causes='UNKNOWN unless independent evidence establishes them',release_coverage=False)
    (out/'summary.json').write_bytes(canonical(result))
    artifacts=[dict(path=str(f.relative_to(out)),sha256=digest(f.read_bytes()),bytes=f.stat().st_size) for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'artifact-hashes.json').write_bytes(canonical(artifacts))
    print(json.dumps(result,indent=2));return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())

