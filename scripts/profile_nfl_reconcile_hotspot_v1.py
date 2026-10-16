"""Bounded paired observation; cProfile observes calls without replacing code."""
import argparse,base64,cProfile,hashlib,json,os,pstats,signal,subprocess,sys,time,zlib
from pathlib import Path
ENCODED='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'
PAYLOAD_SHA='db0468683f171a5ba2aa2c60dc531d04436352450e3507372047e85c6fea1307'
def payload():
 raw=zlib.decompress(base64.b64decode(ENCODED,validate=True));assert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA
 return json.loads(raw)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--constraints',type=Path);ap.add_argument('--case',choices=['passing','reconciled']);ap.add_argument('--output',type=Path);ap.add_argument('--collect-only',action='store_true');a=ap.parse_args();p=payload()
 assert len(p['ids'])==len(set(p['ids'])) and digest(p['ids'])==p['ids_sha256']
 assert json.loads(json.dumps(p['ids'],ensure_ascii=True))==p['ids']
 canary=[r'C:\one\two',r'C:\\one\\two','literal\\n','real\nnewline','quote"Δ'];assert json.loads(json.dumps(canary))==canary
 if a.self_test:print(json.dumps(dict(status='PASS',payload_sha256=PAYLOAD_SHA,ids=len(p['ids']),files=p['files'])));return 0
 if a.constraints:a.constraints.write_text('\n'.join(p['constraints'])+'\n');return 0
 root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(root));case=p['revisions'][a.case]
 actual=lambda *v:subprocess.check_output(['git',*v],text=True).strip()
 assert actual('rev-parse','HEAD')==case['head'] and actual('rev-parse','HEAD^{tree}')==case['tree']
 assert actual('rev-parse','--is-shallow-repository')=='true'
 if not a.collect_only:assert sys.version.split()[0]==p['python'] and os.environ.get('ImageVersion')==p['image']
 import pytest
 clock=time.perf_counter();events=(out/'progress.jsonl').open('x',encoding='utf-8',buffering=1);phases=[];fixtures=[];outcomes=[]
 def emit(kind,**v):events.write(json.dumps(dict(kind=kind,elapsed=time.perf_counter()-clock,**v))+'\n');events.flush()
 save(out/'environment.json',dict(case=a.case,revision=case,python=sys.version,pytest=pytest.__version__,cwd=str(root),shallow=True,payload_sha256=PAYLOAD_SHA,ids_sha256=p['ids_sha256'],profile_sha256=hashlib.sha256((root/'scripts/ci_test_file_costs_v1.json').read_bytes()).hexdigest(),image=os.environ.get('ImageVersion'),dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),limits=dict(job=1800,test=1500,retries=0),measurement='cProfile plus pytest hooks; inclusive costs overlap and are not additive'))
 class Observe:
  @pytest.hookimpl(trylast=True)
  def pytest_collection_modifyitems(self,session,config,items):
   assert [i.nodeid for i in items]==p['ids'];assert config.option.basetemp is None;emit('exact_ids_verified',count=len(items))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_collection(self,session):
   t=time.perf_counter();yield;phases.append(dict(phase='collection',seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_fixture_setup(self,fixturedef,request):
   t=time.perf_counter();yield;fixtures.append(dict(name=fixturedef.argname,scope=fixturedef.scope,node=request.node.nodeid,seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_protocol(self,item,nextitem):
   t=time.perf_counter();yield;emit('test_complete',nodeid=item.nodeid,seconds=time.perf_counter()-t)
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_setup(self,item):
   t=time.perf_counter();yield;phases.append(dict(phase='setup',node=item.nodeid,seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_call(self,item):
   t=time.perf_counter();yield;phases.append(dict(phase='call',node=item.nodeid,seconds=time.perf_counter()-t))
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_teardown(self,item):
   t=time.perf_counter();yield;phases.append(dict(phase='teardown',node=item.nodeid,seconds=time.perf_counter()-t))
  def pytest_runtest_logreport(self,report):
   if report.when=='call' or (report.when=='setup' and report.outcome!='passed'):
    row=dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome);outcomes.append(row);emit('outcome',**row)
 class Budget(BaseException):pass
 def alarm(*_):raise Budget()
 if not a.collect_only:signal.signal(signal.SIGALRM,alarm);signal.alarm(1500)
 profiler=cProfile.Profile();interrupted=False;code=124
 opts=['-v',*p['files'],'--durations=30','--junitxml='+str(out/'tests.xml')]
 if a.collect_only:opts.append('--collect-only')
 try:
  profiler.enable();code=int(pytest.main(opts,plugins=[Observe()]))
 except Budget:interrupted=True;emit('INTERRUPTED_BUDGET')
 finally:
  profiler.disable()
  if not a.collect_only:signal.alarm(0)
  report=time.perf_counter();profiler.dump_stats(str(out/'calls.pstats'))
  stats=pstats.Stats(profiler);rows=[]
  for (file,line,name),(primitive,calls,self_s,inclusive,callers) in stats.stats.items():
   rows.append(dict(file=file,line=line,name=name,primitive=primitive,calls=calls,self_seconds=self_s,inclusive_seconds=inclusive))
  caches={}
  g=sys.modules.get('scripts.check_launch_change_scope')
  if g:
   for name,fn in vars(g).items():
    if hasattr(fn,'cache_info'):caches[name]=dict(fn.cache_info()._asdict())
    if hasattr(fn,'reconstruction_memo'):caches[name]=dict(hits=fn.reconstruction_memo.hits,misses=fn.reconstruction_memo.misses,retained_bytes=fn.reconstruction_memo.retained_bytes,fields=fn.reconstruction_context_fields)
  save(out/'profile.json',sorted(rows,key=lambda r:-r['inclusive_seconds']))
  save(out/'receipt.json',dict(status='INTERRUPTED' if interrupted else 'COMPLETED_DIAGNOSTIC_ONLY',case=a.case,revision=case,exit_code=code,elapsed_seconds=time.perf_counter()-clock,phases=phases,fixtures=fixtures,outcomes=outcomes,complete=not a.collect_only and len(outcomes)==len(p['ids']),caches=caches,default_temp=True,network='Unchanged repository conftest blocks external application transports',reporting_seconds=time.perf_counter()-report,warning='Inclusive profiler and fixture/test timings overlap; instrumentation overhead is included. Not release coverage.',collect_only=a.collect_only));events.close()
 return code
if __name__=='__main__':raise SystemExit(main())
