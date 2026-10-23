"""Two fresh diagnostic phases, one active deadline, no retry."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,os,signal,subprocess,sys,time
P=argparse.ArgumentParser();P.add_argument('--root',required=True);P.add_argument('--out',required=True);P.add_argument('--job-start',type=float,required=True);a=P.parse_args()
D=Path(__file__).resolve().parent;out=Path(a.out);out.mkdir(exist_ok=False)
m=json.loads((D/'manifest.json').read_bytes())
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==v for n,v in m['data_sha256'].items()),'TOOL_CHANGED'
deadline=min(a.job_start+2700-90,time.time()+2400);results=[];interrupted=False
for phase in ('native5',):
    remaining=deadline-time.time()
    if remaining<=15:interrupted=True;break
    command=[sys.executable,'-I','-B',str(D/'observe.py'),'--root',a.root,'--out',str(out/phase),'--phase',phase]
    started=time.perf_counter()
    with io.FileIO(out/(phase+'.log'),'w')as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=remaining-15)
        except subprocess.TimeoutExpired:
            interrupted=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    p=out/phase/'result.json';record=json.loads(p.read_bytes())if p.exists()else{}
    results.append(dict(phase=phase,exit_code=code,seconds=time.perf_counter()-started,complete_pass=record.get('complete_pass',False)))
    with io.FileIO(out/'controller-progress.json','w')as f:f.write((json.dumps(results,indent=2)+'\n').encode())
    if code or not record.get('complete_pass',False)or interrupted:break
record=dict(at=datetime.now(timezone.utc).isoformat(),results=results,interrupted=interrupted,complete_pass=len(results)==1 and all(x['complete_pass']for x in results)and not interrupted,release_coverage=False)
with io.FileIO(out/'controller.json','w')as f:f.write((json.dumps(record,indent=2)+'\n').encode())
raise SystemExit(0 if record['complete_pass']else 1)
