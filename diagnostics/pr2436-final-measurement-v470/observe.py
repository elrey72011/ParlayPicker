"""Bounded offline diagnostic, not release coverage. No application patches."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, io, json, os, subprocess, sys, tempfile, time
P=argparse.ArgumentParser()
P.add_argument('--root',required=True);P.add_argument('--out',required=True)
P.add_argument('--phase',choices=['native9'],required=True)
P.add_argument('--collect-only',action='store_true');P.add_argument('--local-libs',action='append',default=[])
a=P.parse_args();D=Path(__file__).resolve().parent;root=Path(a.root).resolve();out=Path(a.out).resolve()
out.mkdir(exist_ok=False)
FileIO=io.FileIO;dump=json.dumps;load=json.loads
def write(name,value):
    with FileIO(out/name,'w')as f:f.write((dump(value,indent=2,ensure_ascii=True)+'\n').encode())
def sha(value):return hashlib.sha256(value).hexdigest()
def raw(path):
    with FileIO(path,'r')as f:return f.read()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
m=load(raw(D/'manifest.json'))
for name,expected in m['data_sha256'].items():assert sha(raw(D/name))==expected,('TOOL_CHANGED',name)
assert git('rev-parse','HEAD')==m['tested_merge'] and git('rev-parse','HEAD^{tree}')==m['source_tree']
parents=[line[7:]for line in git('cat-file','-p','HEAD').splitlines()if line.startswith('parent ')]
assert parents==m['ordered_parents'] and not git('status','--porcelain')
proof=load(raw(D/'source-proof.json'))
for path,row in proof.items():assert sha(raw(root/path))==row['sha256'],('PHYSICAL_SOURCE_CHANGED',path)
ids=load(raw(D/(a.phase+'.json')));assert len(ids)==len(set(ids))==m['phase_counts'][a.phase]
os.chdir(root);sys.path[:0]=[str(root),str(root/'scripts'),str(root/'tests'),str(D),*a.local_libs]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
if a.local_libs:
    # The local interpreter receives read-only libraries through this explicit
    # path; unisolated synthetic child runners need the same environment.
    os.environ['PYTHONPATH']=os.pathsep.join(a.local_libs)
    child=subprocess.run([sys.executable,'-c','import pytest;print(pytest.__version__)'],capture_output=True,text=True)
    assert child.returncode==0 and child.stdout.strip(),'LOCAL_CHILD_PYTEST_UNAVAILABLE'
temporary=Path(tempfile.mkdtemp(prefix='pp2436-v342-'))
os.environ['PYTEST_DEBUG_TEMPROOT']=str(temporary)
import pytest
from fixture_profile import FixtureProfiler
def context():
    owners=[]
    for name in ('tests.historical_scope_input_boundary','historical_scope_input_boundary'):
        module=sys.modules.get(name)
        if module:
            owners.append(dict(module=name,identity=id(module),owners=[dict(consumer=x.plan.consumer,generation=x.plan.generation,role=x.plan.role.value,fixture=x.plan.fixture)for x in getattr(module,'_ACTIVE',[])]))
    def ident(f):
        code=getattr(f,'__code__',None)
        return dict(module=getattr(f,'__module__',None),name=getattr(f,'__qualname__',None),source=getattr(code,'co_filename',None))
    return dict(owners=owners,read_bytes=ident(Path.read_bytes),open=ident(Path.open))
class Observer:
    def __init__(self):
        self.rows=[];self.selected=[];self.observed=[];self.started=time.perf_counter();self.collection=None;self.first=False
        self.fixtures=FixtureProfiler(root,out,context)
    def pytest_sessionstart(self,session):
        import streamlit
        streamlit.config.set_option('secrets.files',[str(out/'SYNTHETIC_NONEXISTENT.toml')])
    def pytest_collection_modifyitems(self,config,items):
        full=list(items);by_id={item.nodeid:item for item in full}
        assert len(by_id)==len(full),'DUPLICATE_COLLECTION'
        assert [item.nodeid for item in full if item.nodeid in set(ids)]==ids,'IDENTITY_OR_ORDER_CHANGED'
        if True:
            from scripts import run_ci_tests
            profile=run_ci_tests.load_profile(root/'scripts/ci_test_file_costs_v1.json')
            rows,_,_=run_ci_tests.assignment(full,13,profile,root)
            assert len(full)==m['collected_count'] and [x.nodeid for x in full]==load(raw(D/'full-inventory.json'))
            assert run_ci_tests.digest(rows)==m['assignment_sha256'] and profile['sha256']==m['profile_sha256']
        deselected=[item for item in full if item.nodeid not in set(ids)]
        items[:]=[by_id[node]for node in ids];config.hook.pytest_deselected(items=deselected)
        self.selected=[item.nodeid for item in items];self.collection=time.perf_counter()-self.started
        write('collection.json',dict(full_count=len(full),selected=self.selected,seconds=self.collection))
    def pytest_runtest_logstart(self,nodeid,location):self.observed.append(nodeid)
    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_fixture_setup(self,fixturedef,request):
        token=self.fixtures.begin(fixturedef,request)
        try:yield
        finally:self.fixtures.end(token)
    @pytest.hookimpl(specname='pytest_fixture_setup',hookwrapper=True,trylast=True)
    def pytest_fixture_setup_body(self,fixturedef,request):
        token=self.fixtures.begin(fixturedef,request,'fixture_body')
        try:yield
        finally:self.fixtures.end(token)
    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_makereport(self,item,call):
        before=context();outcome=yield;report=outcome.get_result()
        if report.failed and not self.first:
            self.first=True;hashes=[]
            for name in ('scripts/check_launch_change_scope.py','docs/paid-launch/ci-fixture-stability-bindings-v1.json','.github/workflows/ci.yml','scripts/ci_test_file_costs_v1.json'):
                path=root/name;row=dict(path=name,physical=sha(raw(path)))
                for api,reader in [('read_bytes',path.read_bytes),('open',lambda:read_open(path))]:
                    try:row[api]=sha(reader())
                    except Exception as exc:row[api]=dict(error_type=type(exc).__name__)
                hashes.append(row)
            write('first-mismatch.json',dict(nodeid=item.nodeid,phase=report.when,exception_type=getattr(getattr(call,'excinfo',None),'typename',None),before=before,after=context(),hashes=hashes))
    def pytest_runtest_logreport(self,report):
        row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration);self.rows.append(row)
        with FileIO(out/'phases.jsonl','a')as f:f.write((dump(row)+'\n').encode())
        if report.when=='teardown':write('progress.json',dict(completed=sum(x['phase']=='teardown'for x in self.rows),last_node=report.nodeid,context=context()))
def read_open(path):
    with path.open('rb')as f:return f.read()
observer=Observer();started=time.perf_counter();code=1
write('start.json',dict(head=git('rev-parse','HEAD'),tree=m['source_tree'],phase=a.phase,ids=ids,before=context(),python=sys.version,temporary=str(temporary),release_coverage=False))
try:
    args=['tests']
    code=int(pytest.main([*args,'-q','-x','--tb=short','--show-capture=no','--junitxml='+str(out/'actual.xml'),*(['--collect-only']if a.collect_only else[])],plugins=[observer]))
finally:
    exact=all(Counter(x['nodeid']for x in observer.rows if x['phase']==phase)==Counter(ids)for phase in ('setup','call','teardown'))
    unchanged=not git('status','--porcelain') and git('rev-parse','HEAD^{tree}')==m['source_tree']
    write('result.json',dict(phase=a.phase,exit_code=code,seconds=time.perf_counter()-started,collection_seconds=observer.collection,selected=observer.selected,observed=observer.observed,phases=observer.rows,after=context(),fixture_frames_remaining=observer.fixtures.active_frames,source_unchanged=unchanged,collect_only=a.collect_only,complete_pass=not a.collect_only and code==0 and unchanged and exact and observer.observed==ids and all(x['outcome']=='passed'for x in observer.rows) and observer.fixtures.active_frames==0,release_coverage=False))
raise SystemExit(code)
