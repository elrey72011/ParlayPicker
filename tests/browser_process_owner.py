"""Bounded test-browser output and invocation-owned native process resources."""
import json
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

CHROME_SECONDS=40
OUTPUT_LIMIT=2*1024*1024
ERROR_LIMIT=64*1024
_INVOCATION=ContextVar('owned_browser_invocation',default=None)


def retain_browser_failure(function):
    """Keep bounded evidence in JUnit too; preserve the original exception."""
    @wraps(function)
    def checked(*args,**kwargs):
        context={'evidence':None};token=_INVOCATION.set(context)
        try:return function(*args,**kwargs)
        except BaseException as error:
            evidence=context['evidence']
            if evidence is not None:
                def bounded(path,limit):
                    data=path.read_bytes()
                    if len(data)>limit:data=data[:limit//2]+b'\n[bounded middle omitted]\n'+data[-limit//2:]
                    return data.decode('utf-8',errors='replace')
                error.add_note('Browser process: '+bounded(evidence/'browser-process.json',4096))
                error.add_note('Browser stdout: '+bounded(evidence/'browser-process.stdout',32768))
                error.add_note('Browser stderr: '+bounded(evidence/'browser-process.stderr',16384))
            raise
        finally:_INVOCATION.reset(token)
    return checked


class PosixChild:
    def __init__(self,command):
        self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            start_new_session=True,close_fds=True,bufsize=0)
        self.pid=self.child.pid;self.stdout=self.child.stdout;self.stderr=self.child.stderr

    def wait(self,timeout):return self.child.wait(timeout=max(0,timeout))

    def members(self):
        found=[]
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():continue
            try:
                fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==self.pid:found.append(dict(pid=int(directory.name),state=fields[0]))
            except (FileNotFoundError,ProcessLookupError):continue
        return found

    def cleanup(self):
        before=self.members()
        try:os.killpg(self.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        self.child.wait(timeout=5)
        after=self.members()
        if any(p['state'] not in ('Z','X') for p in after):raise RuntimeError('OWNED_GROUP_STILL_RUNNING')
        return dict(kind='posix_process_group',members_before=before,members_after=after,
            remaining_are_dead=True,direct_child_reaped=True)

    def close(self):
        for stream in (self.stdout,self.stderr):
            if not stream.closed:stream.close()


class Capture:
    def __init__(self,stream,limit):
        self.stream=stream;self.limit=limit;self.data=bytearray();self.total=0;self.error=None;self.eof=False
        self.progress=threading.Event()
        self.thread=threading.Thread(target=self.read,name='owned-browser-output',daemon=True)

    def read(self):
        try:
            while True:
                block=os.read(self.stream.fileno(),8192)
                if not block:self.eof=True;break
                self.total+=len(block)
                if len(self.data)<self.limit:self.data.extend(block[:self.limit-len(self.data)])
                self.progress.set()
        except BaseException as error:self.error=type(error).__name__
        finally:self.stream.close()


def run_owned(command,*,timeout=CHROME_SECONDS,evidence=None):
    """One launch, no retry. A separate bounded cleanup is never execution time."""
    from tests.browser_process_windows import WindowsChild
    started=time.monotonic();child=None;captures=[];failure=None;cleanup_failure=None;code=None
    report=dict(schema='owned-browser-v1',deadline_seconds=timeout,phase='LAUNCH',spawned=False)
    try:
        child=(WindowsChild if os.name=='nt' else PosixChild)(command)
        report.update(spawned=True,pid=child.pid,phase='PROCESS_RUNNING')
        captures=[Capture(child.stdout,OUTPUT_LIMIT),Capture(child.stderr,ERROR_LIMIT)]
        for capture in captures:capture.thread.start()
        code=child.wait(max(0,timeout-(time.monotonic()-started)))
        report.update(phase='DIRECT_PROCESS_EXITED',direct_returncode=code)
    except BaseException as error:
        failure=error;report.update(phase='EXECUTION_TIMEOUT' if isinstance(error,subprocess.TimeoutExpired) else report['phase'],exception_class=type(error).__name__)
    finally:
        if child:
            try:report['cleanup']=child.cleanup()
            except BaseException as error:cleanup_failure=error;report['cleanup_exception_class']=type(error).__name__
            finally:
                until=time.monotonic()+5
                for capture in captures:
                    if capture.thread.ident is not None:capture.thread.join(max(0,until-time.monotonic()))
                    else:capture.stream.close()
                report['pipes_closed']=all(c.eof and not c.thread.is_alive() and c.stream.closed for c in captures)
                child.close()
        report['elapsed_seconds']=time.monotonic()-started
        output=[bytes(c.data) for c in captures]+[b'',b''];stdout,stderr=output[:2]
        report['output']=[dict(total_bytes=c.total,retained_bytes=len(c.data),truncated=c.total>c.limit,reader_exception_class=c.error) for c in captures]
        report['dom_marker_observed']=b'probability-test-result' in stdout
        report['page_or_shutdown_phase']='DOM_OUTPUT_OBSERVED' if report['dom_marker_observed'] else 'UNKNOWN'
        if evidence is not None:
            evidence.update(report)
            evidence['_stdout']=stdout;evidence['_stderr']=stderr
    if cleanup_failure:
        if failure:failure.add_note('Owned browser cleanup failed: '+type(cleanup_failure).__name__)
        else:failure=cleanup_failure
    if child and not report['pipes_closed'] and failure is None:failure=RuntimeError('OWNED_BROWSER_PIPES_NOT_CLOSED')
    if any(c.error or c.total>c.limit for c in captures) and failure is None:failure=RuntimeError('OWNED_BROWSER_CAPTURE_LIMIT_OR_READ_FAILURE')
    if failure:
        failure.add_note('Owned browser evidence: '+json.dumps({k:v for k,v in report.items() if k!='pid'},sort_keys=True))
        raise failure
    return subprocess.CompletedProcess(command,code,stdout,stderr)


def run_browser(command,directory):
    """Fresh owned profile; preserve the rendered input and every launch option."""
    directory=Path(directory);record={};evidence=Path(tempfile.mkdtemp(prefix='browser-evidence-',dir=directory))
    profile=tempfile.TemporaryDirectory(prefix='owned-profile-',dir=evidence);failure=None;result=None
    actual=[('--user-data-dir='+profile.name) if arg.startswith('--user-data-dir=') else arg for arg in command]
    assert sum(arg.startswith('--user-data-dir=') for arg in actual)==1
    try:
        result=run_owned(actual,timeout=CHROME_SECONDS,evidence=record)
    except BaseException as error:failure=error
    finally:
        try:profile.cleanup();record['profile_removed']=not Path(profile.name).exists()
        except BaseException as error:
            record['profile_removed']=False;record['profile_cleanup_exception_class']=type(error).__name__
            if failure:failure.add_note('Owned profile cleanup failed: '+type(error).__name__)
            else:failure=error
        finally:
            stdout=record.pop('_stdout',b'');stderr=record.pop('_stderr',b'')
            (evidence/'browser-process.stdout').write_bytes(stdout)
            (evidence/'browser-process.stderr').write_bytes(stderr)
            (evidence/'browser-process.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            context=_INVOCATION.get()
            if context is not None:context['evidence']=evidence
    if failure:raise failure
    return result
