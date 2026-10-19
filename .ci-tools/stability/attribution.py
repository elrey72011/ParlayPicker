"""Diagnostic-only native attribution; no arguments, values or exception text retained."""
from pathlib import Path
from datetime import datetime,timezone
import functools,inspect,json,shutil,subprocess,threading,time

class Attribution:
    def __init__(self,root,out,controls):
        self.root=Path(root).resolve();self.out=Path(out)
        self.allowed={r['path'] for r in controls}
        self.rows=[];self.aggregates={};self.total_events=0;self.detail_capped=0;self.dropped=0;self.errors=0;self.local=threading.local();self.started=False;self.restored=False
        self.current_file='<collection>';self.stage='collection';self.fixture_scope=None;self.helper_attributes=[]
        self.original_open=Path.open;self.original_read=Path.read_bytes
        self.original_run=subprocess.run;self.original_output=subprocess.check_output;self.original_copy=shutil.copytree

    def source(self):
        frame=inspect.currentframe()
        try:
            frame=frame.f_back
            for _ in range(24):
                if frame is None:return None
                filename=frame.f_code.co_filename
                try:relative=Path(filename).resolve().relative_to(self.root).as_posix()
                except (ValueError,OSError):relative=None
                if relative in self.allowed:return relative
                frame=frame.f_back
            return None
        finally:
            del frame

    def category(self,args):
        if not isinstance(args,(list,tuple)) or len(args)<2:return None
        if not isinstance(args[0],str) or Path(args[0]).name.lower() not in ('git','git.exe'):return None
        words=tuple(args[1:])
        if words==('config','gc.auto','0'):return 'ADDED_CONFIG_GC_AUTO_ZERO'
        if words==('config','maintenance.auto','false'):return 'ADDED_CONFIG_MAINTENANCE_FALSE'
        if words==('gc','--prune=never'):return 'ADDED_SYNCHRONOUS_GC_PRUNE_NEVER'
        if isinstance(words[0],str) and words[0] in {'init','add','commit','commit-tree','read-tree','write-tree','rev-parse','cat-file','ls-tree','update-index','checkout','diff','show','hash-object','repack','gc','fsck','config'}:
            return 'GIT_'+words[0].upper().replace('-','_')
        return None

    def record(self,source,category,seconds,raised):
        key=(source,category,raised,self.stage,self.fixture_scope)
        if key not in self.aggregates:
            if len(self.aggregates)>=2048:self.dropped+=1;return
            self.aggregates[key]=dict(source=source,category=category,raised=raised,stage=self.stage,fixture_scope=self.fixture_scope,count=0,seconds=0.0,maximum_seconds=0.0)
        row=self.aggregates[key];row['count']+=1;row['seconds']+=seconds;row['maximum_seconds']=max(row['maximum_seconds'],seconds)
        self.total_events+=1
        if category.startswith('ADDED_'):
            if len(self.rows)<256:
                event=dict(source=source,category=category,seconds=seconds,raised=raised)
                self.rows.append(event)
                self.out.mkdir(parents=True,exist_ok=True)
                with (self.out/'added-command-detail.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
            else:self.detail_capped+=1
        if category.startswith('ADDED_') or self.total_events%100==0:
            self.out.mkdir(parents=True,exist_ok=True)
            (self.out/'attribution-progress.json').write_text(json.dumps(self.result(),indent=2)+'\n')

    def git_call(self,native,args,pargs,kwargs):
        if getattr(self.local,'git_depth',0):return native(args,*pargs,**kwargs)
        category=self.category(args)
        source=self.source() if category else None
        if source is None:return native(args,*pargs,**kwargs)
        self.local.git_depth=1;started=time.perf_counter();raised=False
        try:return native(args,*pargs,**kwargs)
        except BaseException:raised=True;raise
        finally:
            self.local.git_depth=0
            self.record(source,category,time.perf_counter()-started,raised)

    def helper_call(self,native,category,args,kwargs):
        if getattr(self.local,'helper_depth',0):return native(*args,**kwargs)
        self.local.helper_depth=1;started=time.perf_counter();raised=False
        try:return native(*args,**kwargs)
        except BaseException:raised=True;raise
        finally:
            self.local.helper_depth=0
            self.record(self.current_file,category,time.perf_counter()-started,raised)

    def start_helpers(self):
        if not (self.root/'tests/ci_fixture_stability_inputs.py').is_file():return
        import importlib
        module=importlib.import_module('tests.ci_fixture_stability_inputs')
        owner=self
        for name in ('original_source','reader_source'):
            native=getattr(module,name)
            @functools.wraps(native)
            def span(*args,_native=native,_name=name,**kwargs):
                return owner.helper_call(_native,'HELPER_'+_name.upper(),args,kwargs)
            setattr(module,name,span);self.helper_attributes.append((module,name,span,native))
        native=module.install_exact_predecessor_sources
        @functools.wraps(native)
        def installer(*args,**kwargs):
            started=time.perf_counter();raised=False;result=False
            owner.local.helper_depth=getattr(owner.local,'helper_depth',0)+1
            try:
                result=native(*args,**kwargs)
                if result:
                    patch=args[1] if len(args)>1 else kwargs['patch']
                    for attribute in ('open','read_bytes'):
                        actual=getattr(Path,attribute)
                        if actual.__module__!=module.__name__:continue
                        @functools.wraps(actual)
                        def view(*vargs,_actual=actual,_attribute=attribute,**vkwargs):
                            return owner.helper_call(_actual,'SOURCE_VIEW_'+_attribute.upper(),vargs,vkwargs)
                        patch.setattr(Path,attribute,view)
                return result
            except BaseException:raised=True;raise
            finally:
                owner.local.helper_depth-=1
                owner.record(owner.current_file,'HELPER_INSTALL_ACTIVE' if result else 'HELPER_INSTALL_INACTIVE',time.perf_counter()-started,raised)
        module.install_exact_predecessor_sources=installer
        self.helper_attributes.append((module,'install_exact_predecessor_sources',installer,native))

    def start(self):
        assert not self.started;self.started=True
        owner=self
        @functools.wraps(self.original_run)
        def run(args,*pargs,**kwargs):return owner.git_call(owner.original_run,args,pargs,kwargs)
        @functools.wraps(self.original_output)
        def output(args,*pargs,**kwargs):return owner.git_call(owner.original_output,args,pargs,kwargs)
        @functools.wraps(self.original_copy)
        def copy(*args,**kwargs):
            if getattr(owner.local,'copy_depth',0):return owner.original_copy(*args,**kwargs)
            source=owner.source()
            if source is None:return owner.original_copy(*args,**kwargs)
            owner.local.copy_depth=1;started=time.perf_counter();raised=False
            try:return owner.original_copy(*args,**kwargs)
            except BaseException:raised=True;raise
            finally:
                owner.local.copy_depth=0
                owner.record(source,'SYNTHETIC_REPOSITORY_COPY',time.perf_counter()-started,raised)
        self.wrapper_run=run;self.wrapper_output=output;self.wrapper_copy=copy
        subprocess.run=run;subprocess.check_output=output;shutil.copytree=copy
        self.start_helpers()

    def stop(self):
        if not self.started:return
        pairs=[(subprocess,'run',self.wrapper_run,self.original_run),(subprocess,'check_output',self.wrapper_output,self.original_output),(shutil,'copytree',self.wrapper_copy,self.original_copy)]
        for obj,name,wrapper,native in pairs:
            if getattr(obj,name) is wrapper:setattr(obj,name,native)
            else:self.errors+=1
        for obj,name,wrapper,native in self.helper_attributes:
            if getattr(obj,name) is wrapper:setattr(obj,name,native)
            else:self.errors+=1
        self.restored=(all(getattr(obj,name) is native for obj,name,wrapper,native in pairs+self.helper_attributes)
                       and Path.open is self.original_open and Path.read_bytes is self.original_read)
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/'attribution-summary.json').write_text(json.dumps(self.result(),indent=2)+'\n')

    def result(self):
        return dict(events=self.total_events,aggregate_keys=len(self.aggregates),detail_capped=self.detail_capped,dropped=self.dropped,errors=self.errors,native_restored=self.restored,
            source_files=sorted({r['source'] for r in self.aggregates.values()}),
            timing_method='Native return/error-preserving outer API calls only. Nested run/check_output and copytree calls are not double counted.',
            limitations=['Git/copy attribution overlaps fixture/test/file elapsed; do not add it to those elapsed totals.',
             'Source-view/installer spans are gross added-boundary upper bounds; native reads inside spans are included. Nested source views are not double counted.',
             'Attribution can affect scheduling; no historical cause inferred.','All events aggregated; only up to256 added-command details retained. No command arguments or input values.'],rows=list(self.aggregates.values()),added_detail=self.rows)

