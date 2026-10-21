"""One bounded offline observation of real runner ordering; never release coverage."""
from pathlib import Path
import argparse,cProfile,hashlib,io,json,os,runpy,signal,subprocess,sys,time,types
P=argparse.ArgumentParser();P.add_argument('--root',required=True);P.add_argument('--out',required=True);P.add_argument('--mode',choices=['prefix','cold','files','platform'],default='prefix');P.add_argument('--cold-index',type=int);P.add_argument('--collect-only',action='store_true');P.add_argument('--local-seal-preflight',action='store_true');a=P.parse_args()
D=Path(__file__).resolve().parent;root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False)
manifest=json.loads((D/'manifest.json').read_bytes())
def sha(value):return hashlib.sha256(value).hexdigest()
def write(name,value):
 with io.FileIO(out/name,'w')as f:f.write((json.dumps(value,indent=2,ensure_ascii=True)+'\n').encode())
for name,expected in manifest['data_sha256'].items():assert sha((D/name).read_bytes())==expected,(name,'TOOL_INPUT_CHANGED')
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def parents():return [line.split()[1] for line in git('cat-file','-p','HEAD').splitlines() if line.startswith('parent ')]
head=git('rev-parse','HEAD');tree=git('rev-parse','HEAD^{tree}')
assert tree==manifest['source_tree'] and not git('status','--porcelain')
if a.local_seal_preflight:assert head==manifest['source_head']
else:
 assert head==manifest['tested_merge']
 assert parents()==manifest['ordered_parents']
ids=json.loads((D/('ordered304.json'if a.mode=='prefix'else'affected162.json'if a.mode=='files'else'platform5.json'if a.mode=='platform'else'cold6.json')).read_bytes())
if a.mode=='cold':ids=[ids[a.cold_index]]
assert len(ids)==len(set(ids)) and len(ids)==(304 if a.mode=='prefix' else 162 if a.mode=='files' else 5 if a.mode=='platform'else 1)
os.chdir(root);sys.path[:0]=[str(root/'scripts'),str(root),str(root/'tests'),str(D)]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pytest
from temp_observer import TempOwnerTrace
def identity(value):
 code=getattr(value,'__code__',None)
 return dict(module=getattr(value,'__module__',''),name=getattr(value,'__qualname__',''),file=getattr(code,'co_filename',None),line=getattr(code,'co_firstlineno',None))
def context():
 active=[]
 for name in ('tests.historical_scope_input_boundary','historical_scope_input_boundary'):
  m=sys.modules.get(name)
  if m:active.append(dict(name=name,identity=id(m),read_depth=getattr(m,'_DECLARED_READ_DEPTH',None).get()if hasattr(m,'_DECLARED_READ_DEPTH')else None,open_depth=getattr(m,'_DECLARED_OPEN_DEPTH',None).get()if hasattr(m,'_DECLARED_OPEN_DEPTH')else None,owners=[dict(consumer=x.plan.consumer,generation=x.plan.generation,role=x.plan.role.value,fixture=x.plan.fixture)for x in getattr(m,'_ACTIVE',[])]))
 return dict(boundaries=active,read_bytes=identity(Path.read_bytes),open=identity(Path.open),modules={name:dict(identity=id(m),file=getattr(m,'__file__',None))for name in ('scripts.check_launch_change_scope','check_launch_change_scope','__main__')if(m:=sys.modules.get(name))is not None})
prof=cProfile.Profile();profiled=[]
class Observer:
 def __init__(self):self.phases=[];self.observed=[];self.selected=[];self.started=time.perf_counter();self.collection=0;self.first=False
 def pytest_configure(self,config):
  if os.environ.get('PYTEST_PLUGINS')=='_pr2436_diagnostic_observer':os.environ.pop('PYTEST_PLUGINS')
 def pytest_sessionstart(self,session):
  import streamlit
  streamlit.config.set_option('secrets.files',[str(out/'SYNTHETIC_NONEXISTENT.toml')])
  self.trace=TempOwnerTrace(out);self.trace.start()
 @pytest.hookimpl(hookwrapper=True,tryfirst=True)
 def pytest_collection_modifyitems(self,config,items):
  full=list(items);yield
  if a.mode=='prefix':
   from scripts import run_ci_tests
   rows,files,loads=run_ci_tests.assignment(full,13,run_ci_tests.load_profile(root/'scripts/ci_test_file_costs_v1.json'),root)
   assert len(full)==manifest['collected_count']
   assert run_ci_tests.digest(rows)==manifest['assignment_sha256']
   assert run_ci_tests.load_profile(root/'scripts/ci_test_file_costs_v1.json')['sha256']==manifest['profile_sha256']
   by_id={item.nodeid:item for item in full};assert all(node in by_id for node in ids)
   items[:]=[by_id[node]for node in ids]
   write('full-collection.json',dict(count=len(full),assignment_sha256=run_ci_tests.digest(rows),collection_nodeids_sha256=sha(json.dumps([x.nodeid for x in full],ensure_ascii=True,separators=(',',':')).encode())))
  assert [item.nodeid for item in items]==ids
  self.selected=[item.nodeid for item in items];self.collection=time.perf_counter()-self.started;write('selected.json',self.selected)
 def pytest_runtest_logstart(self,nodeid,location):
  self.observed.append(nodeid);self.trace.node=nodeid;self.trace.watch('test_start')
 @pytest.hookimpl(hookwrapper=True,tryfirst=True)
 def pytest_runtest_setup(self,item):
  enabled=a.mode=='prefix'and item.nodeid in manifest['profile_setup_nodes']
  if enabled:profiled.append(item.nodeid);prof.enable()
  try:yield
  finally:
   if enabled:prof.disable()
 @pytest.hookimpl(hookwrapper=True,tryfirst=True)
 def pytest_runtest_makereport(self,item,call):
  before=context();outcome=yield;report=outcome.get_result()
  if report.failed and not self.first:
   self.first=True;hashes=[]
   for name in ('scripts/check_launch_change_scope.py','docs/paid-launch/ci-fixture-stability-bindings-v1.json','.github/workflows/ci.yml','scripts/ci_test_file_costs_v1.json'):
    path=root/name;row=dict(path=name)
    for kind,reader in [('physical',lambda:io.FileIO(path,'r').read()),('read_bytes',path.read_bytes),('open',lambda:path.open('rb').read())]:
     try:row[kind]=sha(reader().replace(b'\r\n',b'\n'))
     except Exception as exc:row[kind]=dict(error_type=type(exc).__name__)
    hashes.append(row)
   write('first-mismatch.json',dict(nodeid=item.nodeid,phase=report.when,exception_type=getattr(getattr(call,'excinfo',None),'typename',None),before=before,after=context(),hashes=hashes))
 def pytest_runtest_logreport(self,report):
  row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration);self.phases.append(row)
  with io.FileIO(out/'phases.jsonl','a')as f:f.write((json.dumps(row)+'\n').encode())
  self.trace.watch(report.when+'_'+report.outcome)
  if report.when=='teardown':write('restoration-last.json',dict(nodeid=report.nodeid,**context()))
 def pytest_sessionfinish(self,session,exitstatus):self.trace.stop()
observer=Observer();start=time.perf_counter();code=1
write('start.json',dict(head=head,tree=tree,parents=parents(),mode=a.mode,count=len(ids),python=sys.version,manifest_sha256=sha((D/'manifest.json').read_bytes()),before=context(),shallow=git('rev-parse','--is-shallow-repository'),release_coverage=False))
try:
 if a.mode=='prefix':
  module=types.ModuleType('_pr2436_diagnostic_observer')
  for name in dir(observer):
   if name.startswith('pytest_'):setattr(module,name,getattr(observer,name))
  sys.modules[module.__name__]=module;os.environ['PYTEST_PLUGINS']=module.__name__
  sys.argv=[str(root/'scripts/run_ci_tests.py'),'--shard','9','--shards','13','--timing-profile',str(root/'scripts/ci_test_file_costs_v1.json'),'--assignment-manifest',str(out/'CURRENT_ASSIGNMENT_INCOMPLETE_NOT_RELEASE.json'),'tests','-q','--tb=short','--show-capture=no','--junitxml='+str(out/'actual.xml')]
  if a.collect_only:sys.argv.append('--collect-only')
  try:runpy.run_path(str(root/'scripts/run_ci_tests.py'),run_name='__main__')
  except SystemExit as exc:code=int(exc.code or 0)
 else:
  code=int(pytest.main(['-q',*ids,'--tb=short','--show-capture=no','--junitxml='+str(out/'actual.xml'),*(['--collect-only']if a.collect_only else[])],plugins=[observer]))
finally:
 prof.disable();rows=[];stats=prof.getstats();index={id(x.code):i for i,x in enumerate(stats)}
 def ident(c):return dict(file=c.co_filename,line=c.co_firstlineno,name=c.co_name)if hasattr(c,'co_filename')else dict(file='<builtin>',line=0,name=str(c))
 for x in stats:rows.append(dict(index=index[id(x.code)],**ident(x.code),calls=x.callcount,exclusive_seconds=x.inlinetime,inclusive_seconds=x.totaltime,callees=[dict(index=index.get(id(c.code)),**ident(c.code),calls=c.callcount,exclusive_seconds=c.inlinetime,inclusive_seconds=c.totaltime)for c in x.calls or()]))
 write('setup-profile.json',dict(profiled_nodeids=profiled,rows=rows,limitation='Only two declared setup phases are instrumented. Inclusive times overlap. Local matched profiling overhead is reported separately; Ubuntu overhead remains measured only when a matched observation exists.'))
 complete=observer.observed==ids;failures=[x for x in observer.phases if x['outcome'] in {'failed','skipped'}]
 write('result.json',dict(head=head,tree=tree,mode=a.mode,pytest_or_native_runner_exit=code,collect_only=a.collect_only,selected=observer.selected,observed=observer.observed,exact_order=complete,expected_count=len(ids),failed_phases=failures,seconds=time.perf_counter()-start,collection_seconds=observer.collection,phases=observer.phases,after=context(),diagnostic_complete_pass=complete and not failures and code==(1 if a.mode=='prefix' else 0),release_coverage=False,native_assignment_note='The native current shard assignment remains deliberately incomplete. Its failure is preserved; diagnostic execution does not satisfy a release gate.'))
raise SystemExit(code)
