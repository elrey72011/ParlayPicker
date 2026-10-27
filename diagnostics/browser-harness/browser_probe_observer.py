"""Diagnostic-only observation, no replacement of browser execution or assertions."""
from pathlib import Path
import hashlib,json,os,sys,time
import pytest

ROOT=Path(os.environ['BROWSER_PROBE_EVIDENCE']).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
LAUNCHES=[]

def append(name,value):
    with (ROOT/name).open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,sort_keys=True)+'\n')

def bounded(value,limit):
    if value is None:return b''
    if isinstance(value,str):value=value.encode('utf-8',errors='replace')
    return value if len(value)<=limit else value[:limit//2]+b'\n[bounded middle omitted]\n'+value[-limit//2:]

def processes():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            command=(p/'cmdline').read_bytes().split(b'\0')
            if not any(str(ROOT).encode() in arg for arg in command):continue
            if not any(b'chrome' in arg.lower() or b'chromium' in arg.lower() for arg in command[:1]):continue
            rows.append(dict(pid=int(p.name),ppid=int(fields[1]),pgrp=int(fields[2]),session=int(fields[3]),state=fields[0],
                executable=Path(os.fsdecode(command[0])).name,renderer=any(arg==b'--type=renderer' for arg in command)))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return rows

def audit(event,args):
    if event!='subprocess.Popen':return
    executable,command,_,_=args
    if not isinstance(command,(list,tuple)) or '--dump-dom' not in command:return
    profile=next((x.split('=',1)[1] for x in command if x.startswith('--user-data-dir=')),None)
    event=dict(at=time.time(),executable=executable,flags=[x for x in command[1:] if not x.startswith(('--user-data-dir=','file:'))],
        profile_under_owned_root=bool(profile and Path(profile).resolve().is_relative_to(ROOT)),
        profile_exists_before_launch=bool(profile and Path(profile).exists()),
        invocation='cold' if not LAUNCHES else 'warm_python_process_fresh_profile',
        environment_presence={name:bool(os.environ.get(name)) for name in ('DISPLAY','DBUS_SESSION_BUS_ADDRESS','WAYLAND_DISPLAY')})
    LAUNCHES.append(event);append('launches.jsonl',event)

sys.addaudithook(audit)

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config,items):
    # Match release CI's full-suite import/collection state before selecting
    # the fixed original complete-file prefix. Do not narrow collection itself.
    try:
        manifest=json.loads(Path(os.environ['BROWSER_PROBE_MANIFEST']).read_bytes())
        plan=next(p for p in manifest['required_contexts'] if p['name']==os.environ['BROWSER_PROBE_CONTEXT'])
        files=plan['files'];order={p:n for n,p in enumerate(files)}
    except BaseException as error:
        (ROOT/'selection-failure.json').write_text(json.dumps(dict(phase='LOADING_SELECTION',exception_class=type(error).__name__,raw_exception_retained=False))+'\n')
        raise pytest.UsageError('FROZEN_SELECTION_INPUT_INVALID') from None
    selected=[];other=[]
    for index,item in enumerate(items):
        path=Path(item.path).resolve().relative_to(Path(config.rootpath).resolve()).as_posix()
        (selected if path in order else other).append((path,index,item))
    selected.sort(key=lambda row:(order[row[0]],row[1]))
    config.hook.pytest_deselected(items=[row[2] for row in other])
    record=dict(collected_count=len(items),selected_count=len(selected),files=files,
        selected_nodeids=[row[2].nodeid for row in selected],full_collection_before_selection=True)
    (ROOT/'selection-receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if record['collected_count']!=plan['expected_collected_count'] or record['selected_nodeids']!=plan['expected_nodeids'] or record['selected_count']!=plan['expected_count']:
        (ROOT/'selection-failure.json').write_text(json.dumps(dict(phase='VALIDATING_SELECTION',category='FROZEN_SELECTION_MISMATCH',raw_exception_retained=False))+'\n')
        raise pytest.UsageError('FROZEN_SELECTION_MISMATCH')
    items[:]=[row[2] for row in selected]

def pytest_runtest_setup(item):
    append('before-tests.jsonl',dict(nodeid=item.nodeid,at=time.time(),owned_browser_processes=processes(),native_launches=len(LAUNCHES)))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
    result=yield;report=result.get_result()
    if report.when not in ('call','setup','teardown'):return
    row=dict(nodeid=item.nodeid,when=report.when,outcome=report.outcome,duration=report.duration,at=time.time(),native_launches=len(LAUNCHES),owned_browser_processes=processes())
    if call.excinfo is not None:
        error=call.excinfo.value;row['exception_class']=type(error).__name__
        if hasattr(error,'timeout'):row['subprocess_deadline']=error.timeout
        name=hashlib.sha256((item.nodeid+report.when).encode()).hexdigest()[:20]
        for field,limit in [('stdout',65536),('stderr',32768)]:
            raw=getattr(error,field,None)
            if raw is not None:
                data=bounded(raw,limit);(ROOT/(name+'.'+field)).write_bytes(data)
                row[field]=dict(file=name+'.'+field,retained_bytes=len(data),original_bytes=len(raw))
        trace=error.__traceback__
        while trace:
            for key in ('process','self'):
                child=trace.tb_frame.f_locals.get(key)
                if hasattr(child,'pid') and hasattr(child,'args') and isinstance(child.args,(list,tuple)) and '--dump-dom' in child.args:
                    row['failed_direct_process']=dict(pid=child.pid,returncode=child.returncode)
            trace=trace.tb_next
        row['page_script_or_shutdown_cause']='UNKNOWN_UNLESS_DOM_EVIDENCE'
    tmp=item.funcargs.get('tmp_path')
    if tmp is not None:
        records=[]
        for path in Path(tmp).rglob('browser-process.json'):
            value=json.loads(path.read_bytes());records.append(value)
            identity=hashlib.sha256(str(path.relative_to(ROOT)).encode()).hexdigest()[:20]
            (ROOT/(identity+'.browser-process.json')).write_bytes(path.read_bytes())
            if report.failed:
                for ext,limit in [('stdout',65536),('stderr',32768)]:
                    source=path.with_suffix('.'+ext)
                    if source.exists():(ROOT/(identity+'.'+ext)).write_bytes(bounded(source.read_bytes(),limit))
        if records:row['browser_process_records']=records
    append('outcomes.jsonl',row)
