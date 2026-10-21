from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--job-start',type=float,required=True);a=p.parse_args()
D=Path(__file__).resolve().parent;out=Path(a.out);out.mkdir(exist_ok=False);m=json.loads((D/'manifest.json').read_bytes());deadline=a.job_start+m['job_cap_seconds']-m['retention_reserve_seconds'];results=[];interrupted=False
phases=[('original304',['--mode','prefix'])]+[('cold-'+str(i+1),['--mode','cold','--cold-index',str(i)])for i in range(6)]
def save():
 (out/'controller.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),results=results,remaining_phases=[name for name,_ in phases if name not in{r['phase']for r in results}],interrupted=interrupted,release_coverage=False),indent=2)+'\n')
for label,extra in phases:
 remaining=deadline-time.time()
 if remaining<30:interrupted=True;save();break
 command=[sys.executable,'-I','-B',str(D/'observe.py'),'--root',a.root,'--out',str(out/label),*extra]
 started=time.perf_counter()
 with (out/(label+'.log')).open('xb')as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=max(1,remaining-15))
  except subprocess.TimeoutExpired:
   interrupted=True;os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
 path=out/label/'result.json';r=json.loads(path.read_bytes())if path.exists()else{}
 results.append(dict(phase=label,process_exit=code,seconds=time.perf_counter()-started,result_retained=path.exists(),diagnostic_complete_pass=r.get('diagnostic_complete_pass',False)));save()
 if interrupted or not r.get('diagnostic_complete_pass'):break
hashes=[]
for path in sorted(out.rglob('*')):
 if path.is_file():hashes.append(dict(path=path.relative_to(out).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'artifact-file-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
raise SystemExit(0 if len(results)==7 and all(r['diagnostic_complete_pass']for r in results) else 1)
