"""Offline diagnostic-helper safety controls; synthetic repositories only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,tempfile
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--temp-parent',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);checks=[]
    def ok(name,value):assert value,name;checks.append(name)
    from attribution import Attribution
    from run_job import execute
    from failure_capture import FailureCapture
    original=(subprocess.run,subprocess.check_output,shutil.copytree,Path.open,Path.read_bytes)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-diag-',dir=args.temp_parent) as td:
        root=Path(td);repo=root/'repo';repo.mkdir()
        def git(*a):return subprocess.check_output(['git',*a],cwd=repo,stderr=subprocess.DEVNULL)
        git('init','-q');git('config','user.name','SYNTHETIC');git('config','user.email','synthetic@example.invalid')
        (repo/'SYNTHETIC.txt').write_bytes(b'SYNTHETIC immutable fixture\n')
        git('add','--all');git('commit','-qm','SYNTHETIC immutable fixture')
        src='tests/test_football_catalog_scope.py';attr=Attribution(root,out/'native-attribution',[dict(path=src)])
        ns=dict(subprocess=subprocess,shutil=shutil,repo=repo,root=root)
        body="""def native_calls():
    a=subprocess.check_output(['git','config','gc.auto','0'],cwd=repo)
    b=subprocess.run(['git','config','maintenance.auto','false'],cwd=repo,check=True,capture_output=True)
    c=subprocess.check_output(['git','gc','--prune=never'],cwd=repo,stderr=subprocess.DEVNULL)
    failed=False
    try:subprocess.check_output(['git','cat-file','-e','SYNTHETIC_NEVER_RETAIN_EXAMPLE'],cwd=repo,stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:failed=True
    copied=shutil.copytree(repo,root/'copied')
    copy_failed=False
    try:shutil.copytree(root/'SYNTHETIC_MISSING',root/'no-copy')
    except FileNotFoundError:copy_failed=True
    return a,b.returncode,c,failed,copied,copy_failed
"""
        exec(compile(body,str(root/src),'exec'),ns)
        try:
            attr.start();result=ns['native_calls']()
        finally:attr.stop()
        ok('native_results_preserved',result[0]==b'' and result[1]==0 and isinstance(result[2],bytes))
        ok('native_missing_object_rejection_propagates',result[3] is True)
        ok('native_copy_and_copy_failure_preserved',result[4]==root/'copied' and result[5] is True)
        data=attr.result()
        counts={r['category']:sum(v['count'] for v in data['rows'] if v['category']==r['category']) for r in data['rows']}
        ok('three_added_commands_measured_once',all(counts.get(k)==1 for k in ('ADDED_CONFIG_GC_AUTO_ZERO','ADDED_CONFIG_MAINTENANCE_FALSE','ADDED_SYNCHRONOUS_GC_PRUNE_NEVER')))
        ok('nested_run_check_output_not_double_counted',sum(r['count'] for r in data['rows'] if r['category'].startswith(('ADDED_','GIT_')))==4)
        ok('copy_recursion_not_double_counted',counts.get('SYNTHETIC_REPOSITORY_COPY')==2)
        ok('native_state_restored',(subprocess.run,subprocess.check_output,shutil.copytree,Path.open,Path.read_bytes)==original and data['native_restored'])
        ok('marker_and_raw_arguments_excluded','SYNTHETIC_NEVER_RETAIN_EXAMPLE' not in json.dumps(data) and not any(k in json.dumps(data) for k in ('stderr','stdout','exception_message')))
        env={k:v for k,v in __import__('os').environ.items() if k in ('PATH','SYSTEMROOT','WINDIR','TEMP','TMP','HOME','LANG')}
        code=execute([sys.executable,'-c','pass'],out/'child-normal.log',5,env)
        ok('supervisor_normal_child',code==0)
        code=execute([sys.executable,'-c','import time; time.sleep(5)'],out/'child-timeout.log',.1,env)
        ok('supervisor_timeout_terminates_and_waits',code==124)
        class SyntheticAppTest:
            def _run(self):raise RuntimeError('SYNTHETIC_NEVER_RETAIN_EXAMPLE')
        native=SyntheticAppTest._run
        watch=FailureCapture(root,out/'failure-watch',near_deadline=.1)
        try:
            watch.start(SyntheticAppTest)
            try:SyntheticAppTest()._run()
            except RuntimeError:pass
        finally:watch.stop()
        observed=watch.result()
        ok('failure_watch_restores_and_joins',observed['original_restored'] and observed['workers_stopped'] and not observed['observer_errors'] and SyntheticAppTest._run is native)
        ok('failure_watch_excludes_exception_values','SYNTHETIC_NEVER_RETAIN_EXAMPLE' not in json.dumps(observed))
    ok('temporary_fixture_cleanup',not root.exists())
    record=dict(at=datetime.now(timezone.utc).isoformat(),passed=len(checks),failed=0,checks=checks,synthetic_only=True,provider_requests=0,
        note='Safety controls only, not release or authentic acquisition coverage.')
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()

