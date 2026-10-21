from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,os,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--collect-only',action='store_true');a=p.parse_args()
D=Path(__file__).resolve().parent;root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False);m=json.loads((D/'manifest.json').read_bytes());ids=json.loads((D/'paid23.json').read_bytes())
assert len(ids)==len(set(ids))==23
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==m['tested_merge'] and git('rev-parse','HEAD^{tree}')==m['source_tree'] and not git('status','--porcelain')
assert [line.split()[1]for line in git('cat-file','-p','HEAD').splitlines()if line.startswith('parent ')]==m['ordered_parents']
os.chdir(root);sys.path[:0]=[str(root),str(root/'scripts'),str(root/'tests')];os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pytest
phases=[];observed=[];selected=[];started=time.perf_counter()
def write(name,value):
 with io.FileIO(out/name,'w')as f:f.write((json.dumps(value,indent=2)+'\n').encode())
class Observer:
 def pytest_collection_modifyitems(self,items):
  selected.extend(item.nodeid for item in items);assert selected==ids;write('selected.json',selected)
 def pytest_runtest_logstart(self,nodeid,location):observed.append(nodeid)
 def pytest_runtest_logreport(self,report):
  row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration);phases.append(row)
  with io.FileIO(out/'phases.jsonl','a')as f:f.write((json.dumps(row)+'\n').encode())
code=int(pytest.main(['-q',*ids,'--tb=short','--show-capture=no','--junitxml='+str(out/'actual.xml'),*(['--collect-only']if a.collect_only else[])],plugins=[Observer()]))
failed=[r for r in phases if r['outcome']=='failed'];skipped=[r for r in phases if r['outcome']=='skipped'];complete=observed==ids and not failed and not skipped and code==0
write('result.json',dict(at=datetime.now(timezone.utc).isoformat(),source_head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),seconds=time.perf_counter()-started,selected=selected,observed=observed,phases=phases,diagnostic_complete_pass=complete,collect_only=a.collect_only,release_coverage=False))
raise SystemExit(code)
