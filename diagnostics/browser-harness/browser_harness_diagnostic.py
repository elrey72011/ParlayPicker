"""Exactly one fixed Ubuntu job; each predetermined context runs at most once."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,subprocess,sys,time

WORKSPACE=Path(os.environ['GITHUB_WORKSPACE']).resolve()
OUT=WORKSPACE/'browser-evidence';OUT.mkdir(exist_ok=False)
BASELINE=WORKSPACE/'baseline-app';CANDIDATE=WORKSPACE/'candidate-app'
PHASE='LOADING_INPUTS'
(OUT/'startup.json').write_text(json.dumps(dict(phase=PHASE,python=sys.version,started_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
def retain_startup_failure(kind,error,trace):
    (OUT/'startup-failure.json').write_text(json.dumps(dict(phase=PHASE,exception_class=kind.__name__,
        raw_exception_retained=False,automatic_retries=0),indent=2)+'\n')
    sys.__excepthook__(kind,error,trace)
sys.excepthook=retain_startup_failure
SPEC=json.loads((Path(__file__).parent/'selection.json').read_bytes())
IDENTITY=json.loads((Path(__file__).parent/'candidate.json').read_bytes())
PHASE='VERIFYING_CHECKOUTS'
sys.path.insert(0,str(CANDIDATE))
from tests.browser_process_owner import run_owned

def git(root,*args):return subprocess.check_output(['git',*args],cwd=root).decode().strip()
assert git(BASELINE,'rev-parse','HEAD')==IDENTITY['failed_head']
assert git(CANDIDATE,'rev-parse','HEAD')==IDENTITY['head']
assert git(CANDIDATE,'rev-parse','HEAD^{tree}')==IDENTITY['tree']
assert git(BASELINE,'rev-parse','--is-shallow-repository')==git(CANDIDATE,'rev-parse','--is-shallow-repository')=='true'
for root in (BASELINE,CANDIDATE):
    assert not git(root,'status','--porcelain')
    assert subprocess.run(['git','cat-file','-e',IDENTITY['base']+'^{commit}'],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0

import shutil,platform
binary=shutil.which('google-chrome') or shutil.which('chromium');assert binary
version_record={};version=run_owned([binary,'--version'],timeout=10,evidence=version_record)
assert version.returncode==0
version_record.pop('_stdout',None);version_record.pop('_stderr',None)
runtime=dict(python=sys.version,platform=platform.platform(),browser_executable=binary,browser_version=version.stdout.decode().strip(),
    candidate=IDENTITY,baseline_tree=git(BASELINE,'rev-parse','HEAD^{tree}'),version_probe=version_record,
    settings_present={name:bool(os.environ.get(name)) for name in ('PARLAYPICKER_NFL_DATABASE_URL','PARLAYPICKER_NFL_STORE_KEY','ODDS_API_KEY')},
    nfl_acquisition_enabled=False,nfl_inference_enabled=False,provider_requests=0,database_connections=0)
(OUT/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
plans=[]
for mode,root,group in [('baseline',BASELINE,'10'),('baseline',BASELINE,'9'),('candidate',CANDIDATE,'10'),('candidate',CANDIDATE,'9'),('candidate',CANDIDATE,'shared')]:
    files=SPEC['prefixes'][group]['files'] if group!='shared' else ['tests/test_research_probability_browser.py','tests/test_browser_process_owner.py']
    plans.append(dict(mode=mode,root=str(root),group=group,files=files))
(OUT/'frozen-plan.json').write_text(json.dumps(plans,indent=2)+'\n')
job_started=float(os.environ['BROWSER_DIAGNOSTIC_JOB_STARTED'])
whole_job_remaining=max(0,2700-(time.time()-job_started)-180)
deadline=time.monotonic()+min(2280,whole_job_remaining)
completed=[]
for plan in plans:
    name=plan['mode']+'-'+plan['group'];folder=OUT/name;folder.mkdir()
    remaining=deadline-time.monotonic()
    if remaining<60:
        completed.append(dict(name=name,status='NOT_STARTED_BUDGET_EXHAUSTED'));break
    env=os.environ.copy();env['BROWSER_PROBE_EVIDENCE']=str(folder)
    env['PYTHONPATH']=str(Path(__file__).parent)
    env['PARLAYPICKER_NFL_ACQUISITION_ENABLED']=env['PARLAYPICKER_NFL_INFERENCE_ENABLED']='false'
    command=[sys.executable,'-B','-m','pytest','-q','-p','browser_probe_observer',*plan['files'],
        '--basetemp='+str(folder/'tmp'),'--junitxml='+str(folder/'tests.xml'),'--durations=20']
    (folder/'command.json').write_text(json.dumps(dict(command=command,started_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    began=time.monotonic();status='COMPLETE';code=None
    with (folder/'console.log').open('wb') as log:
        child=subprocess.Popen(command,cwd=plan['root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=max(1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            status='STEP_BUDGET_EXHAUSTED'
            try:os.killpg(child.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=5)
    item=dict(name=name,status=status,exit_code=code,elapsed_seconds=time.monotonic()-began,
        completed_at=datetime.now(timezone.utc).isoformat(),phase_group_cleanup=True)
    completed.append(item)
    (OUT/'progress.json').write_text(json.dumps(completed,indent=2)+'\n')
    print(json.dumps(item),flush=True)
    if status!='COMPLETE':break

# Existing temporary pytest pages are synthetic; retain only bounded evidence,
# not whole per-test workspaces, model fixtures or databases.
index=[]
for path in OUT.rglob('*'):
    if path.is_file() and 'tmp' not in path.relative_to(OUT).parts:
        index.append(dict(path=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(OUT/'file-identities.json').write_text(json.dumps(index,indent=2)+'\n')
all_complete=len(completed)==len(plans) and all(x['status']=='COMPLETE' for x in completed)
(OUT/'diagnostic-summary.json').write_text(json.dumps(dict(all_contexts_complete=all_complete,outcomes=completed,automatic_retries=0,source_changes_applied=False,
    note='Diagnostic outcomes are not release clearance. Baseline and candidate each execute once. Historical failure cause is not assumed.'),indent=2)+'\n')
if not all_complete:sys.exit(2)
