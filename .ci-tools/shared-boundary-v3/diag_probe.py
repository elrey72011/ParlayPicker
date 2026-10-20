"""Diagnostic observation only; no application import before ordinary pytest collection."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import functools,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,threading,time
import pytest
HERE=Path(__file__).resolve().parent
ROOT=Path.cwd().resolve()
CONTRACT=json.loads(Path(os.environ['BOUNDARY_DIAGNOSTIC_CONTRACT']).read_bytes())
PHASE=os.environ['BOUNDARY_DIAGNOSTIC_PHASE']
OUT=Path(os.environ['BOUNDARY_DIAGNOSTIC_OUTPUT']).resolve()
VERIFY=os.environ.get('BOUNDARY_DIAGNOSTIC_VERIFY_ONLY')=='1'
assert PHASE in {'cold-service','combined-application'}
SPEC=CONTRACT['phases'][PHASE]
OUT.mkdir(parents=True,exist_ok=False)
START=time.perf_counter()
ROWS=[];OBSERVED=[];SELECTED=[];COLLECTION_SECONDS=None
ORIGINAL_OPEN=Path.open;ORIGINAL_READ=Path.read_bytes
ORIGINAL_RUN=subprocess.run;ORIGINAL_OUTPUT=subprocess.check_output;ORIGINAL_COPY=shutil.copytree
ATTR={};LOCAL=threading.local();CURRENT='<collection>';STAGE='collection';WRAPPERS=[]
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def record(category,started,raised):
    key=(CURRENT,STAGE,category,bool(raised))
    if key not in ATTR:
        assert len(ATTR)<4096,'DIAGNOSTIC_ATTRIBUTION_BOUND'
        ATTR[key]=dict(file=CURRENT,stage=STAGE,category=category,raised=bool(raised),count=0,seconds=0.0)
    row=ATTR[key];row['count']+=1;row['seconds']+=time.perf_counter()-started
def git_category(args):
    if not isinstance(args,(tuple,list)) or len(args)<2 or not isinstance(args[0],str):return None
    if Path(args[0]).name.lower() not in {'git','git.exe'}:return None
    fixed={'init','add','commit','commit-tree','read-tree','write-tree','rev-parse','cat-file','ls-tree','update-index','checkout','diff','show','hash-object','repack','gc','fsck','config','status','clone','fetch'}
    words=args[1:]
    if words==['config','gc.auto','0'] or tuple(words)==('config','gc.auto','0'):return 'ADDED_GC_AUTO_ZERO'
    if tuple(words)==('config','maintenance.auto','false'):return 'ADDED_MAINTENANCE_FALSE'
    if tuple(words)==('gc','--prune=never'):return 'ADDED_GC_PRUNE_NEVER'
    return 'GIT_'+words[0].upper().replace('-','_') if words and isinstance(words[0],str) and words[0] in fixed else None
def timed(native,kind,args,kwargs):
    depth=getattr(LOCAL,kind,0)
    if depth:return native(*args,**kwargs)
    category='REPOSITORY_COPY' if kind=='copy' else git_category(args[0]) if args else None
    if category is None:return native(*args,**kwargs)
    setattr(LOCAL,kind,1);start=time.perf_counter();raised=False
    try:return native(*args,**kwargs)
    except BaseException:raised=True;raise
    finally:setattr(LOCAL,kind,0);record(category,start,raised)
def install():
    if VERIFY or PHASE!='combined-application':return
    for obj,name,native,kind in ((subprocess,'run',ORIGINAL_RUN,'git'),(subprocess,'check_output',ORIGINAL_OUTPUT,'git'),(shutil,'copytree',ORIGINAL_COPY,'copy')):
        @functools.wraps(native)
        def wrapper(*a,_native=native,_kind=kind,**k):return timed(_native,_kind,a,k)
        setattr(obj,name,wrapper);WRAPPERS.append((obj,name,wrapper,native))
def pytest_sessionstart(session):
    # No predecessor/helper import, no eager application warm-up.
    install()
def pytest_collection_modifyitems(config,items):
    global SELECTED,COLLECTION_SECONDS
    if PHASE=='combined-application':
        streamlit=sys.modules.get('streamlit')
        if streamlit is not None:
            streamlit.config.set_option('secrets.files',[str(OUT/'SYNTHETIC_NONEXISTENT_SECRETS.toml')])
        from scripts import run_ci_tests as runner
        profile=runner.load_profile(ROOT/'scripts/ci_test_file_costs_v1.json')
        rows,files,loads=runner.assignment(items,11,profile,ROOT)
        assert len(rows)==CONTRACT['inventory_count']
        assert runner.digest(rows)==CONTRACT['assignment_sha256']
        assert profile['sha256']==CONTRACT['profile_sha256']
    wanted=SPEC['nodeids']
    assert len(wanted)==len(set(wanted)) and json.loads(json.dumps(wanted,ensure_ascii=True))==wanted
    keep=[item for item in items if item.nodeid in set(wanted)]
    assert [item.nodeid for item in keep]==wanted,'DIAGNOSTIC_SELECTION_CHANGED'
    config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in set(wanted)])
    items[:]=keep;SELECTED=[item.nodeid for item in keep];COLLECTION_SECONDS=time.perf_counter()-START
    write('collection.json',dict(selected=SELECTED,selection_sha256=sha(json.dumps(SELECTED,ensure_ascii=True).encode()),
        collection_seconds=COLLECTION_SECONDS,phase=PHASE,verify_only=VERIFY))
@pytest.hookimpl(hookwrapper=True,tryfirst=True)
def pytest_fixture_setup(fixturedef,request):
    start=time.perf_counter();result=yield
    code=getattr(fixturedef.func,'__code__',None)
    try:relative=Path(code.co_filename).resolve().relative_to(ROOT).as_posix() if code else None
    except (ValueError,OSError):relative=None
    if relative and relative.startswith('tests/'):
        with (OUT/'fixture-setup.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(source=relative,fixture=fixturedef.argname,scope=fixturedef.scope,
                seconds=time.perf_counter()-start,raised=bool(result.excinfo)))+'\n')
def pytest_runtest_logstart(nodeid,location):
    global CURRENT
    OBSERVED.append(nodeid);CURRENT=nodeid.split('::')[0]
def pytest_runtest_setup(item):
    global STAGE
    STAGE='setup'
def pytest_runtest_call(item):
    global STAGE
    STAGE='call'
def pytest_runtest_teardown(item):
    global STAGE
    STAGE='teardown'
def pytest_runtest_logreport(report):
    row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration)
    ROWS.append(row)
    with (OUT/'phases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
def pytest_sessionfinish(session,exitstatus):
    errors=[]
    for obj,name,wrapper,native in reversed(WRAPPERS):
        if getattr(obj,name) is wrapper:setattr(obj,name,native)
        else:errors.append('DIAGNOSTIC_WRAPPER_NOT_RESTORED')
    restored=not errors and Path.open is ORIGINAL_OPEN and Path.read_bytes is ORIGINAL_READ
    completed=[r for r in ROWS if r['phase']=='call' or (r['phase']=='setup' and r['outcome']=='skipped')]
    exact=OBSERVED==SELECTED==SPEC['nodeids'] and len(completed)==len(SELECTED) and len({r['nodeid'] for r in completed})==len(SELECTED)
    bad=[r for r in ROWS if r['outcome']=='failed'];counts=Counter(r['outcome'] for r in completed)
    if VERIFY:
        assert not ROWS and not OBSERVED
    elif int(exitstatus)!=0 or bad or counts.get('skipped',0) or not exact or not restored:session.exitstatus=1
    write('result.json',dict(at=datetime.now(timezone.utc).isoformat(),phase=PHASE,exit_code=int(session.exitstatus),
        seconds=time.perf_counter()-START,collection_seconds=COLLECTION_SECONDS,selected_count=len(SELECTED),
        outcomes=dict(counts),exact_once=exact,verify_only=VERIFY,failed_phases=bad,
        instrumentation_cleanup=restored,instrumentation_errors=errors,
        attribute_totals=list(ATTR.values()),
        input_generation_boundary_warmed_by_instrumentation=False,
        source_hash_reconstruction_internal_time='NOT_SEPARATELY_INSTRUMENTED; included in fixture/setup/file totals',
        attribution_overlap='Git/copy elapsed overlaps setup and test elapsed. Never add them to phase/file/job totals.',
        default_pytest_temp=True,per_test_deadlines_changed=False,network_policy='Actual repository conftest loopback-only transports; synthetic fixtures',
        remaining_threads=[dict(main=t is threading.main_thread(),daemon=t.daemon,alive=t.is_alive()) for t in threading.enumerate()],
        python=__import__('platform').python_version(),diagnostic_only=True,release_coverage=False))

