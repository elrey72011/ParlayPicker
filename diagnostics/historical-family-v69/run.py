"""Four fresh processes under one deadline; incremental retention and no retry."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,os,signal,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--job-start',type=float,required=True);a=p.parse_args()
D=Path(__file__).resolve().parent;m=json.loads((D/'manifest.json').read_bytes());a.out.mkdir(exist_ok=False)
for name,digest in m['data_sha256'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,('TOOL_CHANGED',name)
deadline=min(a.job_start+2700-90,time.time()+2400);results=[];interrupted=False
def save():
    with io.FileIO(a.out/'controller-progress.json','w')as stream:stream.write((json.dumps(results,indent=2)+'\n').encode())
for subject,kind in [('baseline','views'),('baseline','cases'),('candidate','views'),('candidate','cases')]:
    remaining=deadline-time.time()
    if remaining<=15:interrupted=True;break
    root=getattr(a,subject);name=subject+'-'+kind
    command=([sys.executable,'-I','-B',str(D/'views.py'),'--checkout',str(root),'--subject',subject,'--out',str(a.out/(name+'.json'))]if kind=='views'else
             [sys.executable,'-I','-B',str(D/'observe.py'),'--root',str(root),'--phase',subject,'--out',str(a.out/name)])
    start=time.perf_counter()
    with io.FileIO(a.out/(name+'.log'),'w')as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=remaining-15)
        except subprocess.TimeoutExpired:
            interrupted=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    record_path=a.out/(name+'.json')if kind=='views'else a.out/name/'result.json'
    record=json.loads(record_path.read_bytes())if record_path.exists()else{}
    passed=(record.get('complete')and len(record.get('contexts',[]))==13 and len(record.get('negative_controls',[]))==5)if kind=='views'else record.get('complete_pass',False)
    results.append(dict(subject=subject,kind=kind,exit_code=code,seconds=time.perf_counter()-start,complete_pass=bool(passed)));save()
    if code or not passed or interrupted:break
complete=len(results)==4 and all(r['complete_pass']for r in results)and not interrupted
if complete:
    before=json.loads((a.out/'baseline-views.json').read_bytes());after=json.loads((a.out/'candidate-views.json').read_bytes())
    for old,new in zip(before['contexts'],after['contexts']):
        assert (old['path'],old['node'],old['generation'])==(new['path'],new['node'],new['generation'])
        assert {k:v['git_sha256']for k,v in old['view_hashes'].items()}=={k:v['git_sha256']for k,v in new['view_hashes'].items()},('VIEW_CHANGED',old['path'])
    for root,subject in [(a.baseline,'baseline'),(a.candidate,'candidate')]:
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
        assert subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip()==m['sources'][subject]['tree']
record=dict(at=datetime.now(timezone.utc).isoformat(),results=results,interrupted=interrupted,complete_pass=complete,release_coverage=False)
with io.FileIO(a.out/'controller.json','w')as stream:stream.write((json.dumps(record,indent=2)+'\n').encode())
raise SystemExit(0 if complete else 1)
