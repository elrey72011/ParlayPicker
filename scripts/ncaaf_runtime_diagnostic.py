"""Isolated diagnostic observation only; never release coverage or acquisition."""
import argparse,base64,cProfile,collections,hashlib,json,os,pstats,shutil,signal,subprocess,sys,time,zlib
from pathlib import Path
ENCODED='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'
PAYLOAD_SHA='2531294217f259576b31b152efd6b37cb5b1fb5c5584636b072b0a391918f7ba'
def payload():
 raw=zlib.decompress(base64.b64decode(ENCODED,validate=True));assert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA;return json.loads(raw)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--constraints',type=Path);ap.add_argument('--case',choices=['previous','current']);ap.add_argument('--output',type=Path);ap.add_argument('--artifact-directory',type=Path);ap.add_argument('--collect-only',action='store_true');ap.add_argument('--local-dependencies',type=Path);ap.add_argument('--local-temp',type=Path);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();p=payload()
 assert len(p['ids'])==len(set(p['ids'])) and digest(p['ids'])==p['ids_sha256'];canary=[r'C:\one\two',r'C:\\one\\two','literal\\n','real\nnewline','quote"Δ'];assert json.loads(json.dumps(canary,ensure_ascii=True))==canary and json.loads(json.dumps(p['ids'],ensure_ascii=True))==p['ids']
 if a.self_test:print(json.dumps(dict(status='PASS',ids=len(p['ids']),ids_sha256=p['ids_sha256'],escaping='EXACT',payload_sha256=PAYLOAD_SHA)));return 0
 if a.constraints:a.constraints.write_text('\n'.join(p['constraints'])+'\n',encoding='utf-8');return 0
 if a.artifact_directory:
  directory=a.artifact_directory.resolve();directory.mkdir(parents=True,exist_ok=True);files=[]
  for item in sorted(directory.rglob('*')):
   if item.is_dir() or item.name=='artifact-index.json':continue
   assert not item.is_symlink() and item.stat().st_size<64*1024*1024
   raw=item.read_bytes();files.append(dict(path=item.relative_to(directory).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  assert sum(x['bytes'] for x in files)<256*1024*1024
  save(directory/'artifact-index.json',dict(files=files,partial_retained=not (directory/'results/receipt.json').exists(),not_release_coverage=True));return 0
 root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(root))
 if a.local_dependencies:
  sys.path.insert(1,str(a.local_dependencies.resolve()))
  if os.name=='nt':
   index=int(os.environ.get('GIT_CONFIG_COUNT','0'));os.environ['GIT_CONFIG_KEY_'+str(index)]='core.longpaths';os.environ['GIT_CONFIG_VALUE_'+str(index)]='true';os.environ['GIT_CONFIG_COUNT']=str(index+1)
 actual=lambda *v:subprocess.check_output(['git',*v],text=True).strip()
 case=p['revisions'][a.case];preflight=dict(head=actual('rev-parse','HEAD'),tree=actual('rev-parse','HEAD^{tree}'),shallow=actual('rev-parse','--is-shallow-repository'),python=sys.version.split()[0],image=os.environ.get('ImageVersion'),expected=case,payload_sha256=PAYLOAD_SHA);save(out/'preflight.json',preflight)
 assert preflight['head']==case['head'] and preflight['tree']==case['tree'] and preflight['shallow']=='true','DIAGNOSTIC_CHECKOUT_CONTEXT_MISMATCH'
 if not a.local_dependencies:
  import importlib.metadata
  assert sys.version.split()[0]==p['python'],'DIAGNOSTIC_PYTHON_CONTEXT_MISMATCH'
  assert os.environ.get('ImageVersion')==p['image'] and not a.local_temp,'DIAGNOSTIC_IMAGE_OR_TEMP_CONTEXT_MISMATCH'
  for constraint in p['constraints']:
   package,version=constraint.split('==');assert importlib.metadata.version(package)==version,package
 import pytest
 clock=time.perf_counter();state=dict(phase='initialization',node='session');phases=[];fixtures=[];outcomes=[];git_events=[];copies=[];events=(out/'progress.jsonl').open('x',encoding='utf-8',buffering=1)
 def emit(kind,**v):events.write(json.dumps(dict(kind=kind,elapsed=time.perf_counter()-clock,**v))+'\n');events.flush();os.fsync(events.fileno())
 desired=p['ids'][:2] if a.smoke else p['ids'];files=['tests']
 real_popen=subprocess.Popen
 class TimedProcess(real_popen):
  def __init__(self,*args,**kwargs):
   self.diag_start=time.perf_counter();self.diag_done=False;self.diag_state=dict(state);cmd=args[0] if args else kwargs.get('args');self.diag_git=isinstance(cmd,(list,tuple)) and Path(str(cmd[0])).stem.lower()=='git';self.diag_operation=str(cmd[1]) if self.diag_git and len(cmd)>1 else 'other';super().__init__(*args,**kwargs)
  def record(self):
   if self.diag_git and not self.diag_done and self.returncode is not None:self.diag_done=True;git_events.append(dict(operation=self.diag_operation,seconds=time.perf_counter()-self.diag_start,returncode=self.returncode,**self.diag_state))
  def communicate(self,*args,**kwargs):
   result=super().communicate(*args,**kwargs);self.record();return result
  def wait(self,*args,**kwargs):
   result=super().wait(*args,**kwargs);self.record();return result
 real_copy=shutil.copytree;depth=0
 def copytree(*args,**kwargs):
  nonlocal depth
  outer=depth==0;depth+=1;t=time.perf_counter()
  try:return real_copy(*args,**kwargs)
  finally:
   depth-=1
   if outer:copies.append(dict(seconds=time.perf_counter()-t,**state))
 subprocess.Popen=TimedProcess;shutil.copytree=copytree
 class Observe:
  @pytest.hookimpl(trylast=True)
  def pytest_sessionstart(self,session):
   import socket,inspect,requests
   assert Path(inspect.getsourcefile(socket.getaddrinfo)).resolve()==root/'tests/conftest.py'
   assert Path(inspect.getsourcefile(requests.sessions.Session.request)).resolve()==root/'tests/conftest.py';emit('network_blockers_verified')
  @pytest.hookimpl(trylast=True)
  def pytest_collection_modifyitems(self,session,config,items):
   assert len(items)==case['collected_count'] and digest([i.nodeid for i in items])==case['collection_sha256'];emit('full_release_collection_verified',count=len(items),collection_sha256=case['collection_sha256'])
   expected=set(desired);selected=[i for i in items if i.nodeid in expected];assert [i.nodeid for i in selected]==desired;rejected=[i for i in items if i.nodeid not in expected];items[:]=selected;config.hook.pytest_deselected(items=rejected)
   assert (config.option.basetemp is None) if not a.local_temp else (config.option.basetemp is not None);emit('exact_selection_verified',count=len(items),ids_sha256=digest([i.nodeid for i in items]));save(out/'selection.json',dict(ids=[i.nodeid for i in items],ids_sha256=digest([i.nodeid for i in items]),collected_modules=files,default_temp=not bool(a.local_temp),not_release_coverage=True))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_collection(self,session):
   state.update(phase='collection',node='session');t=time.perf_counter();yield;phases.append(dict(phase='collection',seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_fixture_setup(self,fixturedef,request):
   t=time.perf_counter();yield;row=dict(name=fixturedef.argname,module=fixturedef.func.__module__,scope=fixturedef.scope,node=request.node.nodeid,active_test=state['node'],seconds=time.perf_counter()-t);fixtures.append(row);emit('fixture_complete',**row)
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_setup(self,item):
   state.update(phase='setup',node=item.nodeid);t=time.perf_counter();yield;phases.append(dict(phase='setup',node=item.nodeid,seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_call(self,item):
   state.update(phase='call',node=item.nodeid);t=time.perf_counter();yield;phases.append(dict(phase='call',node=item.nodeid,seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_teardown(self,item):
   state.update(phase='teardown',node=item.nodeid);t=time.perf_counter();yield;phases.append(dict(phase='teardown',node=item.nodeid,seconds=time.perf_counter()-t))
  def pytest_runtest_logreport(self,report):
   if report.when=='call' or (report.when=='setup' and report.outcome!='passed'):
    row=dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome);outcomes.append(row);emit('outcome',**row)
 class Budget(BaseException):pass
 def alarm(*_):raise Budget()
 active_budget=1200
 if not a.collect_only and hasattr(signal,'SIGALRM'):signal.signal(signal.SIGALRM,alarm);signal.alarm(active_budget)
 environment=dict(case=a.case,revision=case,python=sys.version,pytest=pytest.__version__,image=os.environ.get('ImageVersion'),cwd=str(root),shallow=True,default_temp=not bool(a.local_temp),payload_sha256=PAYLOAD_SHA,profile_sha256=hashlib.sha256((root/'scripts/ci_test_file_costs_v1.json').read_bytes()).hexdigest(),constraints=p['constraints'],dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),limits=dict(job_seconds=1800,active_seconds=active_budget,retries=0),local_dependencies=bool(a.local_dependencies),selection_limitation=p['selection_limits']);save(out/'environment.json',environment)
 prof=cProfile.Profile();interrupted=False;code=124;opts=['-v',*files,'--durations=30','--junitxml='+str(out/'tests.xml')]
 if a.local_temp:opts+=['--basetemp',str(a.local_temp)]
 if a.collect_only:opts.append('--collect-only')
 try:
  prof.enable();code=int(pytest.main(opts,plugins=[Observe()]))
 except Budget:interrupted=True;emit('INTERRUPTED_BUDGET')
 finally:
  prof.disable()
  if hasattr(signal,'SIGALRM'):signal.alarm(0)
  subprocess.Popen=real_popen;shutil.copytree=real_copy;report=time.perf_counter();prof.dump_stats(str(out/'calls.pstats'));stats=pstats.Stats(prof);rows=[dict(file=f,line=l,name=n,primitive=v[0],calls=v[1],self_seconds=v[2],inclusive_seconds=v[3]) for (f,l,n),v in stats.stats.items()];save(out/'profile.json',sorted(rows,key=lambda r:-r['inclusive_seconds']))
  caches={};g=sys.modules.get('scripts.check_launch_change_scope')
  if g:
   for name,fn in vars(g).items():
    memo=getattr(fn,'reconstruction_memo',None)
    if memo is not None and hasattr(memo,'retained_bytes'):caches[name]=dict(hits=memo.hits,misses=memo.misses,retained_bytes=memo.retained_bytes,fields=fn.reconstruction_context_fields)
  try:
   import resource
   max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  except ImportError:max_rss=None
  complete=not a.collect_only and not interrupted and len(outcomes)==len(desired) and all(o['outcome']=='passed' for o in outcomes)
  save(out/'receipt.json',dict(status='INTERRUPTED' if interrupted else 'COMPLETED_DIAGNOSTIC_ONLY',case=a.case,revision=case,exit_code=code,elapsed_seconds=time.perf_counter()-clock,phases=phases,fixtures=fixtures,outcomes=outcomes,git=git_events,copies=copies,caches=caches,max_rss_kib_linux=max_rss,complete=complete,reporting_seconds=time.perf_counter()-report,collect_only=a.collect_only,smoke=a.smoke,default_temp=not bool(a.local_temp),warning='Nested profiler/fixture/Git process lifetime timings overlap and must not be added. Instrumentation overhead included. Not release coverage.'))
  emit('diagnostic_finished',exit_code=code,complete=complete);events.close()
 return code
if __name__=='__main__':raise SystemExit(main())
