"""One phase, synthetic inputs, exact collection and failure-only metadata capture."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--phase',required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--verify-selection-only',action='store_true');p.add_argument('--local-smoke-only',action='store_true');p.add_argument('--local-library',action='append',default=[]);args=p.parse_args()
contract=json.loads(Path(args.contract).read_bytes());phase=contract['phases'][args.phase]
root=Path(args.root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root).decode().strip()
assert git('rev-parse','HEAD')==phase['head'] and git('rev-parse','HEAD^{tree}')==phase['tree']
assert git('rev-parse','--is-shallow-repository')=='true'
assert args.verify_selection_only or args.local_smoke_only or platform.python_version()=='3.12.15'
assert args.verify_selection_only or args.local_smoke_only or not args.local_library
assert sha((root/'scripts/ci_test_file_costs_v1.json').read_bytes())==phase['profile_sha256']
os.chdir(root);sys.path[:0]=[str(root),str(root/'scripts'),str(root/'tests'),str(HERE),*args.local_library]
import pytest
from failure_capture import FailureCapture
from attribution import Attribution
TARGET='tests/test_drive_history_loading.py::test_actual_completed_analysis_publish_caller_defers_history'
if phase.get('verification_only'):assert args.verify_selection_only or args.local_smoke_only
if args.local_smoke_only:assert phase.get('local_smoke_only') and phase['verification_only'] and all(args.phase not in sequence for sequence in contract['jobs'].values())
controls=contract['fixture_controls']
wanted=phase['nodeids'];assert len(wanted)==len(set(wanted)) and json.loads(json.dumps(wanted,ensure_ascii=True))==wanted
class Probe:
    def __init__(self):self.rows=[];self.ids=[];self.observed=[];self.at=time.perf_counter();self.collection_seconds=None;self.active=False;self.capture=FailureCapture(root,out,near_deadline=phase['harness_seconds']-.25)
    def pytest_sessionstart(self,session):
        if not args.verify_selection_only:attribution.start()
        import streamlit
        streamlit.config.set_option('secrets.files',[str(out/'SYNTHETIC_NONEXISTENT_SECRETS.toml')])
    def pytest_collection_modifyitems(self,config,items):
        from scripts import run_ci_tests as runner
        profile=runner.load_profile(root/'scripts/ci_test_file_costs_v1.json')
        rows,files,loads=runner.assignment(items,11,profile,root)
        assert runner.digest(rows)==phase['assignment_sha256'] and len(rows)==phase['inventory_count']
        keep=[i for i in items if i.nodeid in set(wanted)];assert [i.nodeid for i in keep]==wanted
        config.hook.pytest_deselected(items=[i for i in items if i.nodeid not in set(wanted)]);items[:]=keep
        self.ids=[i.nodeid for i in items];self.collection_seconds=time.perf_counter()-self.at
        (out/'collection.json').write_text(json.dumps(dict(selected=self.ids,selected_sha256=sha(json.dumps(self.ids,ensure_ascii=True).encode()),inventory_count=len(rows),assignment_sha256=runner.digest(rows),profile_sha256=profile['sha256'],seconds=self.collection_seconds),indent=2)+'\n')
    def pytest_runtest_logstart(self,nodeid,location):
        self.observed.append(nodeid);attribution.current_file=nodeid.split('::')[0]

    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_fixture_setup(self,fixturedef,request):
        start=time.perf_counter();before_scope=attribution.fixture_scope;attribution.fixture_scope=fixturedef.scope
        outcome=yield
        attribution.fixture_scope=before_scope
        code=getattr(fixturedef.func,'__code__',None)
        filename=code.co_filename if code is not None else ''
        try:relative=Path(filename).resolve().relative_to(root).as_posix()
        except (ValueError,OSError):relative=None
        if relative is not None and relative.startswith('tests/'):
            name=fixturedef.argname
            if __import__('re').fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',name):
                with (out/'fixture-setup.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(source=relative,fixture=name,scope=fixturedef.scope,
                        seconds=time.perf_counter()-start,raised=bool(outcome.excinfo)))+'\n')

    def pytest_runtest_setup(self,item):
        attribution.stage='setup'
        if item.nodeid==TARGET:self.capture.start();self.active=True
    def pytest_runtest_call(self,item):attribution.stage='call'
    def pytest_runtest_logreport(self,report):
        code=None
        if report.outcome=='failed':
            code='APPTEST_HARNESS_TIMEOUT' if 'AppTest script run timed out after' in str(report.longrepr) else 'UNEXPECTED_FAILURE'
        row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration,failure_code=code)
        self.rows.append(row)
        with (out/'phases.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self,item):
        attribution.stage='teardown'
        yield
        if item.nodeid==TARGET and self.active:self.capture.stop();self.active=False
attribution=Attribution(root,out,controls)
probe=Probe();started=time.perf_counter();code=2
try:
    command=['-q','tests','--tb=short','--show-capture=no','--disable-warnings','--durations=30','--junitxml='+str(out/'actual.xml')]
    if args.verify_selection_only:command+=['--collect-only']
    code=int(pytest.main(command,plugins=[probe]))
finally:
    if probe.active:probe.capture.stop()
    if attribution.started:attribution.stop()
    capture_cleanup=(probe.capture.owner is None or (probe.capture.restored and not probe.capture.errors
        and all(not worker.is_alive() for worker,_ in probe.capture.workers)))
    if attribution.started and (not attribution.restored or attribution.errors or attribution.dropped):code=1
    if not capture_cleanup:code=1
    completed=[r for r in probe.rows if r['phase']=='call' or (r['phase']=='setup' and r['outcome']=='skipped')]
    bad=[r for r in probe.rows if r['outcome']=='failed']
    counts=Counter(r['outcome'] for r in completed)
    result=dict(at=datetime.now(timezone.utc).isoformat(),phase=args.phase,head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),
        ordered_parents=[l.split()[1] for l in git('cat-file','-p','HEAD').splitlines() if l.startswith('parent ')],
        exit_code=code,seconds=time.perf_counter()-started,collection_seconds=probe.collection_seconds,selected_count=len(wanted),
        executed_once=probe.observed==wanted and len(completed)==len(wanted) and len({r['nodeid'] for r in completed})==len(wanted),
        outcomes=dict(counts),failed_phases=bad,capture=probe.capture.result(),attribution=attribution.result(),diagnostic_only=True,selection_only=args.verify_selection_only,local_smoke_only=args.local_smoke_only,
        environment=dict(python=platform.python_version(),platform=platform.platform(),working_directory=str(root),default_pytest_tmp=True,
            versions={name:importlib.metadata.version(name) for name in ('pytest','streamlit','pandas','numpy','pypdf')}),
        attribution_cleanup_pass=not attribution.started or (attribution.restored and not attribution.errors and not attribution.dropped),
        capture_cleanup_pass=capture_cleanup,
        limitations=['Fixture hook durations can exclude outer projection hooks; test setup/file totals include them. Instrumentation can affect scheduling. Timings from test phases may overlap internal attribution; do not add them to whole-job elapsed.','Interrupted progress is incomplete evidence. Diagnostic results never substitute for release artifacts.'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failed_phases','capture','environment')},indent=2))
raise SystemExit(code)
