"""Bounded observer only: native audit events, no filesystem/API replacement."""
import io,json,os,sys,time,traceback
class TempOwnerTrace:
 def __init__(self,out):
  self.out=out;self.active=False;self.targets={};self.last=None;self.node=None;self.events=0
 def write(self,name,value):
  with io.FileIO(self.out/name,'a')as f:f.write((json.dumps(value)+'\n').encode())
 def watch(self,phase):
  try:
   candidates=[]
   module=sys.modules.get('streamlit.testing.v1.app_test')
   if module is not None:candidates.append(('module',id(module),vars(module).get('TMP_DIR')))
   package=sys.modules.get('streamlit.testing.v1');cls=vars(package).get('AppTest')if package is not None else None
   fn=getattr(getattr(cls,'from_string',None),'__func__',None)
   if fn is not None:candidates.append(('class_globals',id(fn.__globals__),fn.__globals__.get('TMP_DIR')))
   rows=[]
   for owner,identity,temp in candidates:
    name=getattr(temp,'name',None)
    if not isinstance(name,str):continue
    path=os.path.normcase(os.path.abspath(name))
    self.targets[path]=True
    rows.append(dict(owner=owner,identity=identity,temp_identity=id(temp),path=path,exists=os.path.isdir(path),parent_exists=os.path.isdir(os.path.dirname(path))))
   state=json.dumps(rows,sort_keys=True)
   if state!=self.last:self.last=state;self.write('apptest-temp-state.jsonl',dict(at=time.monotonic(),node=self.node,phase=phase,owners=rows))
  except Exception as exc:self.write('apptest-observer-error.jsonl',dict(type=type(exc).__name__,phase=phase))
 def audit(self,event,args):
  if not self.active or event not in {'shutil.rmtree','os.rmdir'}:return
  try:
   self.watch('before_'+event)
   value=args[0]if args else None
   if not isinstance(value,(str,bytes)):return
   path=os.path.normcase(os.path.abspath(os.fsdecode(value)))
   matches=[target for target in self.targets if target==path or target.startswith(path+os.sep)]
   if not matches:return
   self.events+=1
   if self.events>128:return
   stack=traceback.StackSummary.extract(traceback.walk_stack(None),limit=14,lookup_lines=False)
   self.write('apptest-deletion-audit.jsonl',dict(event=event,path=path,targets=matches,node=self.node,
    at=time.monotonic(),stack=[dict(file=x.filename,line=x.lineno,function=x.name)for x in stack]))
  except Exception as exc:self.write('apptest-observer-error.jsonl',dict(type=type(exc).__name__,phase='audit'))
 def start(self):
  self.active=True;sys.addaudithook(self.audit);self.watch('session_start')
 def stop(self):
  self.watch('session_finish');self.active=False
  self.write('apptest-observer-final.jsonl',dict(events=self.events,overflow=self.events>128,hook_disabled=True,
    no_native_methods_replaced=True,no_temporary_directory_created_or_recreated=True))

