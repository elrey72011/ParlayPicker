from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json,os,signal,subprocess,sys,time
P=argparse.ArgumentParser();P.add_argument('--root',required=True);P.add_argument('--out',required=True);P.add_argument('--job-start',type=float,required=True);a=P.parse_args()
D=Path(__file__).resolve().parent;out=Path(a.out);out.mkdir(exist_ok=False);m=json.loads((D/'manifest.json').read_bytes())
assert all(hashlib.sha256((D/name).read_bytes()).hexdigest()==value for name,value in m['data_sha256'].items()),'DIAGNOSTIC_INPUT_CHANGED'
deadline=min(a.job_start+m['job_cap_seconds']-m['retention_reserve_seconds'],time.time()+m['active_phase_cap_seconds']);remaining=deadline-time.time();results=[];interrupted=False
if remaining<30:interrupted=True
else:
 command=[sys.executable,'-I','-B',str(D/'observe.py'),'--root',a.root,'--out',str(out/'critical605'),'--mode','prefix'];started=time.perf_counter()
 with io.FileIO(out/'critical605.log','w')as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=max(1,remaining-15))
  except subprocess.TimeoutExpired:
   interrupted=True;os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
 p=out/'critical605/result.json';r=json.loads(p.read_bytes())if p.exists()else{}
 results.append(dict(phase='critical605',process_exit=code,seconds=time.perf_counter()-started,result_retained=p.exists(),diagnostic_complete_pass=r.get('diagnostic_complete_pass',False)))
record=dict(at=datetime.now(timezone.utc).isoformat(),manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),results=results,remaining_phases=[]if results else['critical605'],interrupted=interrupted,release_coverage=False)
with io.FileIO(out/'controller.json','w')as f:f.write((json.dumps(record,indent=2)+'\n').encode())
hashes=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(out.rglob('*'))if p.is_file()]
with io.FileIO(out/'artifact-file-hashes.json','w')as f:f.write((json.dumps(hashes,indent=2)+'\n').encode())
raise SystemExit(0 if len(results)==1 and results[0]['diagnostic_complete_pass']and not interrupted else 1)
