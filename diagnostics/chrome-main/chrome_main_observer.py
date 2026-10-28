"""Opt-in synthetic-test telemetry; original browser/page/options/deadline unchanged."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,re,sys,threading,time
import pytest

TARGET='tests/test_estimate_availability.py::test_captured_half_point_compatibility_reaches_actual_renderer'
OUT=Path(os.environ['CHROME_OBSERVATION_ROOT']);OUT.mkdir(parents=True,exist_ok=False)
LOCK=threading.Lock();START=time.monotonic();ERRORS=[]
def event(kind,**fields):
    row=dict(at=datetime.now(timezone.utc).isoformat(),elapsed=time.monotonic()-START,kind=kind,**fields)
    with LOCK:
        with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
def pytest_runtest_logstart(nodeid,location):event('CASE_STARTED',nodeid=nodeid)
def pytest_runtest_logreport(report):event('CASE_PHASE_COMPLETED',nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration)
def environment():
    names=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy',
           'DISPLAY','WAYLAND_DISPLAY','DBUS_SESSION_BUS_ADDRESS','XDG_RUNTIME_DIR','XDG_CONFIG_HOME',
           'XDG_CACHE_HOME','LD_PRELOAD','LD_LIBRARY_PATH','CHROME_LOG_FILE','CHROME_WRAPPER',
           'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']
    plain={n:os.environ.get(n,'') for n in ('LANG','LC_ALL','TZ')}
    return dict(present={n:bool(os.environ.get(n)) for n in names},locale=plain,
                python_threads=[t.name for t in threading.enumerate()],python=sys.version.split()[0])
def sample(child,captures):
    row=dict(pid=child.pid,output=[dict(bytes=c.total,eof=c.eof,alive=c.thread.is_alive()) for c in captures])
    if captures:
        row['complete_dom_report_tag_observed']=bool(re.search(br'<pre\b[^>]*id="probability-test-result"[^>]*>.*?</pre>',captures[0].data,re.S))
    if os.name=='nt':
        row['owned_active_processes']=child.members()
        row['direct_returncode']=child.api.GetExitCodeProcess(child.process)
        row['process_detail']='Windows job membership only; Linux /proc details unavailable'
    else:
        row['direct_returncode']=child.child.poll();details=[]
        for member in child.members():
            p=Path('/proc')/str(member['pid'])
            try:
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                arguments=(p/'cmdline').read_bytes().split(b'\0')
                role=next((a.decode('ascii','replace').split('=',1)[1] for a in arguments if a.startswith(b'--type=')),'BROWSER_OR_WRAPPER')
                details.append(dict(pid=member['pid'],ppid=int(fields[1]),start_ticks=int(fields[19]),role=role,state=fields[0],cpu_user=int(fields[11]),cpu_system=int(fields[12]),
                    threads=int(fields[17]),rss_pages=int(fields[21]),exe=Path(os.readlink(p/'exe')).name,
                    wait_channel=(p/'wchan').read_text().strip()))
            except (OSError,ValueError):details.append(dict(pid=member['pid'],state='UNAVAILABLE'))
        row['owned_processes']=details
    return row

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.nodeid!=TARGET:
        yield;return
    from tests import browser_process_owner as owner
    from tests.browser_process_windows import WindowsChild
    selected=WindowsChild if os.name=='nt' else owner.PosixChild
    native_init,native_wait,native_capture=(selected.__init__,selected.wait,owner.Capture.__init__)
    captures=[];stop=threading.Event();watchers=[];counter=[0]
    event('TARGET_ENVIRONMENT',**environment())
    def capture(self,*a,**k):native_capture(self,*a,**k);captures.append(self)
    def observe(child):
        while not stop.is_set():
            try:event('PROCESS_SAMPLE',**sample(child,captures))
            except Exception as error:
                ERRORS.append(type(error).__name__);event('SAMPLE_UNAVAILABLE',exception_class=type(error).__name__)
            if stop.wait(.2):break
    def init(self,command):
        # Observe the actual constructor, not a module function alias already
        # imported by full collection. Do not replace page bytes or launch args.
        from urllib.parse import urlsplit,unquote
        page_uri=urlsplit(command[-1]);assert page_uri.scheme=='file'
        file_path=unquote(page_uri.path)
        if os.name=='nt' and file_path.startswith('/') and file_path[2:3]==':':file_path=file_path[1:]
        page=Path(file_path).resolve();assert page.name=='index.html'
        root=Path(os.environ.get('CAPACITY_LOCAL_SESSION_ROOT',str(OUT))).resolve()
        assert page.is_relative_to(root)
        raw=page.read_bytes();assert len(raw)<=2*1024*1024
        counter[0]+=1;(OUT/('rendered-page-'+str(counter[0])+'.html')).write_bytes(raw)
        event('GENERATED_PAGE_RETAINED',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),command=command)
        if os.environ.get('CHROME_PHASE_PROBE')=='1':
            # Separate, explicitly labelled observability probe. Never count
            # this changed logging/page condition as an unmodified baseline.
            text=raw.decode('utf-8')
            head='<script>window.__ppObserve=(phase)=>console.error("PP_PHASE:"+JSON.stringify({phase,ms:performance.now()}));__ppObserve("HEAD_EXECUTION");addEventListener("DOMContentLoaded",()=>__ppObserve("DOM_CONTENT_LOADED"));addEventListener("load",()=>__ppObserve("LOAD"));addEventListener("error",()=>__ppObserve("SCRIPT_ERROR"));</script>'
            begin='const initialClock=window.__probabilityClock;'
            end='evidence.textContent=JSON.stringify(report);document.body.append(evidence);'
            assert text.count('<head>')==text.count(begin)==text.count(end)==1
            text=text.replace('<head>','<head>'+head,1).replace(begin,'__ppObserve("INSPECTOR_BEGIN");'+begin,1).replace(end,end+'__ppObserve("REPORT_APPENDED");',1)
            page.write_text(text,encoding='utf-8');(OUT/'phase-observed-page.html').write_text(text,encoding='utf-8')
            command.insert(-1,'--enable-logging=stderr')
            event('SEPARATE_PHASE_PROBE',logging_only_flag='--enable-logging=stderr',original_page_saved=True,
                  instrumented_page_sha256=hashlib.sha256(text.encode()).hexdigest(),not_baseline_qualification=True)
        event('SPAWN_ENTER',command=command)
        native_init(self,command);event('SPAWN_RETURN',pid=self.pid)
        thread=threading.Thread(target=observe,args=(self,),name='chrome-observation',daemon=True)
        watchers.append(thread);thread.start()
    def wait(self,timeout):
        event('DIRECT_WAIT_ENTER',remaining_execution_seconds=timeout)
        try:
            result=native_wait(self,timeout);event('DIRECT_WAIT_RETURN',returncode=result);return result
        except BaseException as error:
            event('DIRECT_WAIT_RAISED',exception_class=type(error).__name__);raise
    selected.__init__,selected.wait,owner.Capture.__init__=init,wait,capture
    try:yield
    finally:
        stop.set()
        for thread in watchers:thread.join(3)
        selected.__init__,selected.wait,owner.Capture.__init__=native_init,native_wait,native_capture
        tmp=item.funcargs.get('tmp_path')
        if tmp:
            for number,folder in enumerate(Path(tmp).rglob('browser-evidence-*'),1):
                for name in ('browser-process.json','browser-process.stdout','browser-process.stderr'):
                    source=folder/name
                    if source.exists():(OUT/(str(number)+'-'+name)).write_bytes(source.read_bytes())
            if os.environ.get('CHROME_PHASE_PROBE')=='1':
                for page in Path(tmp).rglob('browser/index.html'):
                    page.write_bytes((OUT/'rendered-page-1.html').read_bytes())
                messages=[]
                for stderr in OUT.glob('*-browser-process.stderr'):
                    for line in stderr.read_text(encoding='utf-8',errors='replace').splitlines():
                        if 'CONSOLE' in line and 'PP_PHASE:' in line:messages.append(line[:2000])
                (OUT/'phase-console.json').write_text(json.dumps(dict(messages=messages,
                    complete=any('REPORT_APPENDED' in line for line in messages)),indent=2)+'\n')
        event('OBSERVER_CLEANUP',threads_stopped=all(not t.is_alive() for t in watchers),observation_errors=ERRORS,
              page_execution='Only a DOM result marker establishes renderer completion; absent marker remains UNKNOWN')
