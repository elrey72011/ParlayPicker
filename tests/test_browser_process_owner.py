"""Real synthetic children prove ownership; real Chrome assertions remain separate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
from tests import browser_process_owner as owner


def source(tmp_path,descendant,blocking):
    leaf=tmp_path/'leaf.py';leaf.write_text('import threading\nthreading.Event().wait(30)\n')
    parent=tmp_path/'parent.py'
    parent.write_text('import subprocess,sys,threading,json\n'+
        ('p=subprocess.Popen([sys.executable,"-B",sys.argv[1]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nprint(json.dumps({"pid":p.pid}),flush=True)\n' if descendant else 'print("owned-output",flush=True)\n')+
        ('threading.Event().wait(30)\n' if blocking else ''))
    return [sys.executable,'-B',str(parent),str(leaf)]


def alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes as W
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
        k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];k.WaitForSingleObject.restype=W.DWORD
        k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
        h=k.OpenProcess(0x100000,False,pid)
        if not h:return False
        try:return k.WaitForSingleObject(h,0)==258
        finally:k.CloseHandle(h)
    p=Path('/proc')/str(pid)/'stat'
    try:return p.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False


@pytest.mark.parametrize('descendant',[False,True])
@pytest.mark.parametrize('mode',['success','timeout','interrupt'])
def test_native_owned_children_cleanup_on_every_exit(tmp_path,monkeypatch,descendant,mode):
    command=source(tmp_path,descendant,mode!='success');record={}
    before={id(t) for t in threading.enumerate()}
    if mode=='interrupt':
        from tests.browser_process_windows import WindowsChild
        selected=WindowsChild if os.name=='nt' else owner.PosixChild
        native=selected.wait;calls=[]
        captures=[];capture_init=owner.Capture.__init__
        def capture(self,*args):capture_init(self,*args);captures.append(self)
        monkeypatch.setattr(owner.Capture,'__init__',capture)
        def interrupted(self,timeout):
            calls.append(timeout)
            if len(calls)==1:
                assert captures[0].progress.wait(5),'Synthetic child never became ready'
                raise KeyboardInterrupt('SYNTHETIC_OWNER_INTERRUPT')
            return native(self,timeout)
        monkeypatch.setattr(selected,'wait',interrupted)
    if mode=='success':
        result=owner.run_owned(command,timeout=5,evidence=record)
        assert result.returncode==0
    else:
        with pytest.raises(subprocess.TimeoutExpired if mode=='timeout' else KeyboardInterrupt):
            owner.run_owned(command,timeout=.4,evidence=record)
    assert record['spawned'] and record['pipes_closed']
    if os.name=='nt':assert record['cleanup']['active_after']==0
    else:assert record['cleanup']['direct_child_reaped'] and record['cleanup']['remaining_are_dead']
    if descendant and record['_stdout']:
        assert not alive(json.loads(record['_stdout'])['pid'])
    assert not any(t.name=='owned-browser-output' and id(t) not in before for t in threading.enumerate())


def test_owned_cleanup_preserves_unrelated_native_child(tmp_path):
    command=source(tmp_path,True,False)
    sentinel=subprocess.Popen([sys.executable,'-B',str(tmp_path/'leaf.py')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        result=owner.run_owned(command,timeout=5,evidence={})
        assert not alive(json.loads(result.stdout)['pid'])
        assert sentinel.poll() is None
    finally:
        sentinel.kill();sentinel.wait(timeout=5)


def test_output_capture_is_bounded_without_blocking_child(tmp_path):
    script=tmp_path/'large.py';script.write_text('import sys\nsys.stdout.buffer.write(b"x"*3000000)\nsys.stderr.buffer.write(b"e"*100000)\n')
    record={}
    with pytest.raises(RuntimeError) as caught:
        owner.run_owned([sys.executable,'-B',str(script)],timeout=5,evidence=record)
    assert caught.value.args==('OWNED_BROWSER_CAPTURE_LIMIT_OR_READ_FAILURE',)
    assert record['pipes_closed'] and record['direct_returncode']==0
    assert len(record['_stdout'])==owner.OUTPUT_LIMIT and len(record['_stderr'])==owner.ERROR_LIMIT
    assert all(r['truncated'] for r in record['output'])


def test_launch_failure_removes_only_owned_profile_and_retains_evidence(tmp_path):
    existing=tmp_path/'profile';existing.mkdir();marker=existing/'owner.txt';marker.write_text('unrelated-profile')
    with pytest.raises(FileNotFoundError):
        owner.run_browser([str(tmp_path/'absent-browser'),'--user-data-dir='+str(existing)],tmp_path)
    evidence=json.loads(next(tmp_path.glob('browser-evidence-*/browser-process.json')).read_bytes())
    assert evidence['deadline_seconds']==40 and evidence['profile_removed'] and not evidence['spawned']
    assert marker.read_text()=='unrelated-profile'
    assert not list(tmp_path.glob('owned-profile-*'))


def test_partial_reader_start_failure_preserves_error_and_cleans_children(tmp_path,monkeypatch):
    native=threading.Thread.start;started=[]
    def start(self):
        if self.name=='owned-browser-output':
            started.append(self)
            if len(started)==2:raise RuntimeError('SYNTHETIC_READER_START_FAILURE')
        return native(self)
    monkeypatch.setattr(threading.Thread,'start',start)
    record={}
    with pytest.raises(RuntimeError) as caught:
        owner.run_owned(source(tmp_path,True,True),timeout=5,evidence=record)
    assert caught.value.args==('SYNTHETIC_READER_START_FAILURE',)
    assert record['exception_class']=='RuntimeError'
    assert not any(t.is_alive() for t in started)
    if os.name=='nt':assert record['cleanup']['active_after']==0
    else:assert record['cleanup']['remaining_are_dead']


def test_failure_context_preserves_original_exception_and_is_invocation_local(tmp_path):
    command=source(tmp_path,False,False)+['--user-data-dir='+str(tmp_path/'unused')]
    failure=AssertionError('SYNTHETIC_ORIGINAL_BROWSER_ASSERTION')
    @owner.retain_browser_failure
    def run():
        owner.run_browser(command,tmp_path)
        raise failure
    with pytest.raises(AssertionError) as caught:run()
    assert caught.value is failure and caught.value.args==('SYNTHETIC_ORIGINAL_BROWSER_ASSERTION',)
    assert any('Browser stdout: owned-output' in note for note in failure.__notes__)
    assert any('"profile_removed": true' in note for note in failure.__notes__)
    assert owner._INVOCATION.get() is None


def test_original_browser_assertions_and_deadline_are_frozen():
    import ast,hashlib
    root=Path(__file__).resolve().parents[1]
    source=(root/'tests/test_research_probability_browser.py').read_bytes()
    nodes=[ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    assert len(nodes)==9
    assert hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()=='515cb2c6ae77d0568a077344086acf5d3e81744d1693b70a0d6dd413b860b814'
    assert owner.CHROME_SECONDS==40
