"""Passive, value-free fixture setup attribution; never changes application state."""
from pathlib import Path
import io,json,threading,time
_perf=time.perf_counter

class FixtureProfiler:
 def __init__(self,root,out,context):
  self.root=Path(root).resolve();self.out=Path(out);self.context=context;self.local=threading.local();self.events=0
 def begin(self,definition,request,layer='outer'):
  frames=getattr(self.local,'frames',None)
  if frames is None:frames=[];self.local.frames=frames
  code=getattr(definition.func,'__code__',None);filename=getattr(code,'co_filename','')
  try:relative=Path(filename).resolve().relative_to(self.root).as_posix()
  except ValueError:relative='<external>'
  node=getattr(getattr(request,'node',None),'nodeid','')
  assert len(node)<=2048 and len(definition.argname)<=128,'DIAGNOSTIC_LABEL_LIMIT'
  frame=dict(start=_perf(),child=0.0,nodeid=node,fixture=definition.argname,scope=definition.scope,source=relative,line=getattr(code,'co_firstlineno',None),layer=layer,before=self.context())
  frames.append(frame);return frame
 def end(self,frame):
  ended=_perf();frames=self.local.frames
  assert frames and frames[-1]is frame,'DIAGNOSTIC_FIXTURE_OWNER_CHANGED'
  frames.pop();elapsed=ended-frame['start'];exclusive=elapsed-frame['child']
  if frames:frames[-1]['child']+=elapsed
  self.events+=1;assert self.events<=25000,'DIAGNOSTIC_EVENT_BUDGET'
  row={key:value for key,value in frame.items()if key not in('start','child')}
  row.update(inclusive_seconds=elapsed,exclusive_seconds=exclusive,nested_seconds=frame['child'],depth=len(frames),after=self.context(),timing_valid=elapsed>=0 and exclusive>=-1e-7)
  with io.FileIO(self.out/'fixture-setups.jsonl','a')as stream:stream.write((json.dumps(row,sort_keys=True)+'\n').encode())
 @property
 def active_frames(self):return len(getattr(self.local,'frames',[]))
