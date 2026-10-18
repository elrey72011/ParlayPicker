"""Isolated synthetic diagnostic. Never dispatches or retries an application request."""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import types


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encode(value)+b'\n')


class Attribution:
    """PEP669 observes code boundaries without replacing functions/hash factories.

    Durations are inclusive and overlapping. They must never be added together.
    No arguments, values, exceptions, paths, commands or environment are retained.
    """
    def __init__(self):
        self.tool=5;self.codes={};self.local=set();self.active={};self.depth={}
        self.rows=collections.defaultdict(lambda:dict(calls=0,seconds=0.0))
        self.started=False;self.errors=0

    def category(self,code):
        if code in self.codes:return self.codes[code]
        file=code.co_filename.replace('\\','/').rsplit('/',1)[-1];name=code.co_name
        category=None
        if file=='subprocess.py' and name=='run':category='subprocess_run'
        elif file=='shutil.py' and name=='copytree':category='repository_copytree'
        elif file=='scope_reconstruction.py' and name in ('call','_context_fields','_freeze'):category='reconstruction_'+name
        elif file=='ncaaf_reviewed_target_operator.py' and name=='verify_checkout':category='live_checkout_authentication'
        elif file=='ncaaf_spread_pair.py' and name in ('admitted_analysis','capture_exports'):category='pair_'+name
        elif file=='research_replay.py' and name in ('retain_export','read_export'):category='static_'+name
        elif file=='app_test.py' and name=='run':category='private_ui_run'
        self.codes[code]=category;return category

    def enter(self,category,key):
        thread=threading.get_ident();depthkey=(thread,category)
        depth=self.depth.get(depthkey,0);self.depth[depthkey]=depth+1
        self.active.setdefault((thread,key),[]).append((category,time.perf_counter(),depth==0))

    def leave(self,key):
        thread=threading.get_ident();stack=self.active.get((thread,key))
        if not stack:self.errors+=1;return
        category,at,outer=stack.pop();depthkey=(thread,category);self.depth[depthkey]-=1
        if outer:
            row=self.rows[category];row['calls']+=1;row['seconds']+=time.perf_counter()-at

    def start_code(self,code,offset):
        category=self.category(code)
        names=set(code.co_names)
        hashing=bool({'sha256','sha1'}&names)
        if category is None and not hashing:return sys.monitoring.DISABLE
        if code not in self.local:
            events=0
            if category:events|=sys.monitoring.events.PY_RETURN
            if hashing:events|=sys.monitoring.events.CALL
            sys.monitoring.set_local_events(self.tool,code,events);self.local.add(code)
        if category:
            if category=='subprocess_run':
                frame=sys._getframe(1)
                if frame.f_code is code:
                    arguments=frame.f_locals.get('popenargs',())
                    command=arguments[0] if arguments else None
                    executable=command[0] if type(command) in (list,tuple) and command else None
                    if type(executable) is str and executable.replace('\\','/').rsplit('/',1)[-1].lower() in ('git','git.exe'):
                        category='git_subprocess_run'
            self.enter(category,code)

    def end_code(self,code,offset,value):
        if self.category(code):self.leave(code)

    def call(self,code,offset,function,self_arg):
        if type(function) is types.BuiltinFunctionType and function.__module__=='_hashlib' and function.__name__ in ('openssl_sha256','openssl_sha1'):
            self.enter('hash_constructor',(code,offset,id(function)))

    def return_call(self,code,offset,function,self_arg):
        if type(function) is types.BuiltinFunctionType and function.__module__=='_hashlib' and function.__name__ in ('openssl_sha256','openssl_sha1'):
            self.leave((code,offset,id(function)))

    def start(self):
        m=sys.monitoring
        assert m.get_tool(self.tool) is None,'DIAGNOSTIC_TOOL_IN_USE'
        m.use_tool_id(self.tool,'bounded-ncaaf-timing')
        for event,callback in ((m.events.PY_START,self.start_code),(m.events.PY_RETURN,self.end_code),
            (m.events.PY_UNWIND,self.end_code),(m.events.CALL,self.call),(m.events.C_RETURN,self.return_call),(m.events.C_RAISE,self.return_call)):
            m.register_callback(self.tool,event,callback)
        self.started=True;m.set_events(self.tool,m.events.PY_START|m.events.PY_UNWIND)

    def stop(self):
        if not self.started:return
        m=sys.monitoring;m.set_events(self.tool,0)
        for code in self.local:m.set_local_events(self.tool,code,0)
        for event in (m.events.PY_START,m.events.PY_RETURN,m.events.PY_UNWIND,m.events.CALL,m.events.C_RETURN,m.events.C_RAISE):m.register_callback(self.tool,event,None)
        m.free_tool_id(self.tool);self.started=False

    def result(self):
        return dict(measurements=dict(self.rows),observer_errors=self.errors,
            caveat='Inclusive overlapping categories; do not sum. Hash constructors only; incremental updates are not measured. Git classification inspects only the executable name and records no command or argument.',
            uncovered_active_calls=sum(len(v) for v in self.active.values()))


class Selection:
    def __init__(self,root,preview,selection,out):
        self.root=root;self.preview=preview;self.selection=selection;self.out=out
        self.started=time.perf_counter();self.collection_seconds=None;self.observed=[];self.outcomes=[];self.attribution=Attribution();self.tracing=False

    def pytest_sessionstart(self,session):self.collection_started=time.perf_counter()

    def pytest_collection_modifyitems(self,config,items):
        from scripts import run_ci_tests as ci
        profile=ci.load_profile(self.root/'scripts/ci_test_file_costs_v1.json')
        rows,files,loads=ci.assignment(items,9,profile,self.root)
        assert rows==self.preview['assignments'],'DIAGNOSTIC_COLLECTION_IDENTITY_CONFLICT'
        assert ci.digest(rows)==self.preview['assignment_sha256'] and profile['sha256']==self.preview['profile_sha256'],'DIAGNOSTIC_PROFILE_CONFLICT'
        nodes=self.selection['selected_nodeids'];assert len(nodes)==len(set(nodes))==374
        shard=[r['nodeid'] for r in rows if r['shard']==9]
        assert shard[:len(nodes)]==nodes,'DIAGNOSTIC_PREFIX_CONFLICT'
        pipeline=[r['nodeid'] for r in rows if r['path']=='tests/test_ncaaf_reviewed_target_pipeline.py']
        assert len(pipeline)==39 and [n for n in nodes if n in set(pipeline)]==pipeline,'DIAGNOSTIC_PIPELINE_CONFLICT'
        wanted=set(nodes);keep=[i for i in items if i.nodeid in wanted];drop=[i for i in items if i.nodeid not in wanted]
        assert [i.nodeid for i in keep]==nodes,'DIAGNOSTIC_ORDER_CONFLICT'
        items[:]=keep;config.hook.pytest_deselected(items=drop)
        self.collection_seconds=time.perf_counter()-self.collection_started
        write(self.out/'selection-verified.json',dict(collected=len(rows),selected=len(nodes),selected_nodeids=nodes,
            selected_sha256=hashlib.sha256(encode(nodes)).hexdigest(),assignment=ci.digest(rows),profile=profile['sha256'],collection_seconds=self.collection_seconds))

    def pytest_runtest_logstart(self,nodeid,location):
        self.observed.append(nodeid)
        if not self.tracing:self.attribution.start();self.tracing=True

    def pytest_runtest_logreport(self,report):
        row=dict(nodeid=report.nodeid,phase=report.when,seconds=report.duration,outcome=report.outcome)
        self.outcomes.append(row)
        with (self.out/'phase-progress.jsonl').open('ab') as f:f.write(encode(row)+b'\n')


def self_test():
    import shutil
    a=Attribution();original=hashlib.sha256
    namespace={'hashlib':hashlib}
    exec(compile('def _freeze(v):\n return hashlib.sha256(v).hexdigest()\n','scope_reconstruction.py','exec'),namespace)
    a.start()
    try:
        assert namespace['_freeze'](b'SYNTHETIC')==original(b'SYNTHETIC').hexdigest()
        failure={}
        exec(compile('def verify_checkout():\n raise ValueError("SYNTHETIC_EXPECTED_FAILURE")\n','ncaaf_reviewed_target_operator.py','exec'),failure)
        try:failure['verify_checkout']()
        except ValueError:pass
        else:raise AssertionError('DIAGNOSTIC_SUPPRESSED_EXCEPTION')
        subprocess.run([sys.executable,'-c','pass'],check=True)
        subprocess.run(['git','--version'],check=True,stdout=subprocess.DEVNULL)
        base=Path(__file__).resolve().parent/('SYNTHETIC-helper-selftest-'+str(time.time_ns()))
        (base/'a').mkdir(parents=True);(base/'a'/'sample').write_bytes(b'SYNTHETIC')
        shutil.copytree(base/'a',base/'b');assert (base/'b'/'sample').read_bytes()==b'SYNTHETIC'
        assert hashlib.sha256 is original
    finally:a.stop()
    assert sys.monitoring.get_tool(5) is None
    result=a.result();assert not result['observer_errors'] and not result['uncovered_active_calls']
    assert {'reconstruction__freeze','hash_constructor','subprocess_run','git_subprocess_run','repository_copytree'}<=set(result['measurements'])
    sample=['tests/test_x.py::test_a[SYNTHETIC\\r\\n]','tests/test_x.py::test_a[SYNTHETIC quoted \'value\']']
    assert json.loads(encode(sample))==sample
    print(json.dumps(dict(status='PASS',exact_hash_identity=True,monitor_cleanup=True,escaping_roundtrip=True,attribution=result),sort_keys=True))


def execute_pytest(root,preview,selection,out,collect_only=False):
    os.chdir(root);out.mkdir(parents=True,exist_ok=True)
    import pytest
    plugin=Selection(root,preview,selection,out);begin=time.perf_counter();result=1
    try:
        argv=['-v','tests','--durations=0','--junitxml='+str(out/'diagnostic.xml')]
        if collect_only:argv.append('--collect-only')
        result=int(pytest.main(argv,plugins=[plugin]));return result
    finally:
        plugin.attribution.stop();elapsed=time.perf_counter()-begin
        phases=collections.Counter()
        for r in plugin.outcomes:phases[r['phase']]+=r['seconds']
        complete=(result==0 and (collect_only or plugin.observed==selection['selected_nodeids']))
        write(out/'diagnostic.json',dict(status='PASS' if complete else 'INCOMPLETE_OR_FAILED',exit_code=result,collect_only=collect_only,expected=374,observed=len(plugin.observed),observed_nodeids=plugin.observed,collection_seconds=plugin.collection_seconds,pytest_total_seconds=elapsed,phase_seconds=dict(phases),unassigned_pytest_seconds=elapsed-(plugin.collection_seconds or 0)-sum(phases.values()),attribution=plugin.attribution.result(),operational_requests=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--preview',type=Path);parser.add_argument('--selection',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--collect-only',action='store_true');parser.add_argument('--self-test',action='store_true');parser.add_argument('--require-merge',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    root=args.root.resolve();os.chdir(root);sys.path[:0]=[str(root),str(root/'tests')];args.out.mkdir(parents=True,exist_ok=True)
    preview=json.loads(args.preview.read_bytes());selection=json.loads(args.selection.read_bytes())
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert head in ({selection['tested_merge']} if args.require_merge else {selection['tested_merge'],selection['release_head']}) and tree=='ead24adbc40a75a9bab78bfb504949d66eeeaac1'
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'DIAGNOSTIC_DIRTY_SOURCE'
    write(args.out/'context.json',dict(head=head,tree=tree,python=sys.version.split()[0],pytest=importlib.metadata.version('pytest'),cwd=str(root),default_pytest_temp=True,synthetic_only=True,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return execute_pytest(root,preview,selection,args.out,args.collect_only)


if __name__=='__main__':raise SystemExit(main())
