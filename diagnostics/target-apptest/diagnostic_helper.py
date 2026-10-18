"""One exact, network-free release-context prefix with sanitized observation."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from phase_probe import PhaseProbe

HEAD='cf8b62b5048fc4f70ced131edcaa148175ca2532'
MERGE='4df0975694226f23e0951ad5f24bf6b89bcb990d'
TREE='88e0f3936f0f7e3519326bf760bafc52e5e7b476'
TARGET='tests/test_drive_history_loading.py::test_actual_completed_analysis_publish_caller_defers_history'
SOURCE='6a9068743f14abd04bf6244bf5fef2c26a530ddf5eaee8f536c149f0a1753486'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(value)+b'\n')


def require(condition,code):
    if not condition:
        raise ValueError(code)


class Selection:
    def __init__(self,root,preview,selection,out):
        self.root=root;self.preview=preview;self.selection=selection;self.out=out
        self.started=time.perf_counter();self.observed=[];self.reports=[]
        self.error_codes=[]
        self.collection_seconds=None;self.probe=PhaseProbe(root,out);self.probe_setup_seconds=0.0

    def pytest_sessionstart(self,session):
        self.collection_started=time.perf_counter()

    def pytest_collection_modifyitems(self,config,items):
        from scripts import run_ci_tests as ci
        profile=ci.load_profile(self.root/'scripts/ci_test_file_costs_v1.json')
        rows,files,loads=ci.assignment(items,9,profile,self.root)
        require(rows==self.preview['assignments'],'COLLECTION_IDENTITY_CONFLICT')
        require(profile['sha256']==self.preview['profile_sha256'] and ci.digest(rows)==self.preview['assignment_sha256'],'PROFILE_OR_ASSIGNMENT_CONFLICT')
        nodes=self.selection['selected']
        require(len(nodes)==len(set(nodes))==105,'SELECTION_COUNT_CONFLICT')
        require(json.loads(canonical(nodes))==nodes,'SELECTION_ENCODING_CONFLICT')
        require(hashlib.sha256(canonical(nodes)).hexdigest()==self.selection['selection_sha256'],'SELECTION_HASH_CONFLICT')
        assigned=[r['nodeid'] for r in rows if r['shard']==8]
        require(assigned[:105]==nodes,'ASSIGNMENT_PREFIX_CONFLICT')
        require(TARGET in nodes and len([n for n in nodes if n.startswith('tests/test_drive_history_loading.py::')])==33,'AFFECTED_FILE_CONFLICT')
        keep=[item for item in items if item.nodeid in set(nodes)]
        require([item.nodeid for item in keep]==nodes,'ORDER_CONFLICT')
        config.hook.pytest_deselected(items=[item for item in items if item.nodeid not in set(nodes)])
        items[:]=keep
        self.collection_seconds=time.perf_counter()-self.collection_started
        write(self.out/'selection-verified.json',dict(collected=len(rows),selected=105,selected_nodeids=nodes,selection_sha256=self.selection['selection_sha256'],assignment_sha256=ci.digest(rows),profile_sha256=profile['sha256'],collection_seconds=self.collection_seconds))

    def pytest_runtest_logstart(self,nodeid,location):
        require(nodeid==self.selection['selected'][len(self.observed)],'EXECUTION_ORDER_CONFLICT')
        self.observed.append(nodeid)
        if nodeid==TARGET:
            at=time.perf_counter();self.probe.start();self.probe_setup_seconds=time.perf_counter()-at

    def pytest_runtest_logreport(self,report):
        # Do not access longrepr, exception values, captured text or user data.
        require(report.nodeid in self.selection['selected'],'REPORT_IDENTITY_CONFLICT')
        require(report.when in ('setup','call','teardown') and report.outcome in ('passed','failed','skipped'),'REPORT_CODE_CONFLICT')
        row=dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,seconds=report.duration)
        self.reports.append(row)
        with (self.out/'phase-progress.jsonl').open('ab') as f:f.write(canonical(row)+b'\n')
        self.safe_xml()

    def pytest_runtest_makereport(self,item,call):
        if call.excinfo is None:return
        allowed={'AssertionError','RuntimeError','ValueError','TypeError','KeyError','FileNotFoundError','PermissionError','OSError','TimeoutError','ConnectionError','ImportError','ModuleNotFoundError'}
        name=call.excinfo.type.__name__;locations=[];tb=call.excinfo.tb
        while tb is not None and len(locations)<12:
            location=self.probe.location(tb.tb_frame)
            if location is not None:locations.append(location)
            tb=tb.tb_next
        self.error_codes.append(dict(nodeid=item.nodeid,phase=call.when,error_code=name if name in allowed else 'OTHER_TEST_EXCEPTION',locations=locations))
        write(self.out/'test-error-codes.json',self.error_codes)

    def pytest_runtest_logfinish(self,nodeid,location):
        if nodeid==TARGET:self.probe.stop()

    def safe_xml(self):
        # Actual outcomes, fixed failure codes. This is not a rewritten release XML.
        suite=ET.Element('testsuite',name='sanitized_exact_prefix_diagnostic')
        for node in self.observed:
            phases=[r for r in self.reports if r['nodeid']==node]
            if not any(r['phase']=='teardown' for r in phases):continue
            parts=node.split('::');case=ET.SubElement(suite,'testcase',classname=parts[0][:-3].replace('/','.')+('.'+'.'.join(parts[1:-1]) if len(parts)>2 else ''),name=parts[-1],time=str(sum(r['seconds'] for r in phases)))
            if any(r['outcome']=='failed' and r['phase'] in ('setup','teardown') for r in phases):ET.SubElement(case,'error',type='ACTUAL_SETUP_OR_TEARDOWN_ERROR',message='ACTUAL_TEST_ERROR')
            elif any(r['outcome']=='failed' for r in phases):ET.SubElement(case,'failure',type='ACTUAL_TEST_FAILURE',message='ACTUAL_TEST_FAILED')
            elif any(r['outcome']=='skipped' for r in phases):ET.SubElement(case,'skipped',message='ACTUAL_TEST_SKIPPED')
        suite.set('tests',str(len(suite)))
        ET.ElementTree(suite).write(self.out/'diagnostic.xml',encoding='utf-8',xml_declaration=True)


def execute(root,preview,selection,out,collect_only=False):
    import pytest
    out.mkdir(parents=True,exist_ok=True)
    plugin=Selection(root,preview,selection,out);started=time.perf_counter();result=2
    try:
        argv=['-v','tests','--durations=30','--tb=no','--show-capture=no','--disable-warnings','-rN']
        if collect_only:argv.append('--collect-only')
        result=int(pytest.main(argv,plugins=[plugin]));return result
    finally:
        plugin.probe.stop();plugin.safe_xml()
        phases=Counter()
        for row in plugin.reports:phases[row['phase']]+=row['seconds']
        completed=[node for node in plugin.observed if any(r['nodeid']==node and r['phase']=='teardown' for r in plugin.reports)]
        all_executed=completed==selection['selected']
        write(out/'diagnostic.json',dict(exit_code=result,status='PASS' if result==0 and (collect_only or all_executed) else 'FAILED_OR_INCOMPLETE',collect_only=collect_only,expected=105,observed=plugin.observed,completed=completed,execution_complete=all_executed,collection_seconds=plugin.collection_seconds,pytest_seconds=time.perf_counter()-started,phase_seconds=dict(phases),target_reports=[r for r in plugin.reports if r['nodeid']==TARGET],observer_setup_seconds=plugin.probe_setup_seconds,probe=plugin.probe.result(),operational_requests=0,provider_content_retained=False,limitation='One instrumented execution; behavior may differ. Reporting is sanitized, no raw exception payloads. Safe XML preserves actual outcomes and identity order, not original exception text. No passing diagnostic establishes the historical cause.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--preview',type=Path,required=True);parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--collect-only',action='store_true');parser.add_argument('--allow-local-head',action='store_true');args=parser.parse_args()
    root=args.root.resolve();out=args.out.resolve();os.chdir(root)
    # Match the release script's import search roots; do not pre-import app code.
    sys.path[0]=str(root/'scripts');sys.path.insert(0,str(root))
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    require(head in ({HEAD,MERGE} if args.allow_local_head else {MERGE}) and tree==TREE,'CHECKOUT_IDENTITY_CONFLICT')
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'CHECKOUT_DIRTY')
    require(hashlib.sha256((root/'tests/test_drive_history_loading.py').read_bytes()).hexdigest()==SOURCE,'SOURCE_HASH_CONFLICT')
    preview=json.loads(args.preview.read_bytes());selection=json.loads(args.selection.read_bytes())
    require(preview['collected_count']==8471 and selection['universe']==[r['nodeid'] for r in preview['assignments']],'UNIVERSE_CONFLICT')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'helper-context.json',dict(head=head,tree=tree,shallow=subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip(),python=sys.version.split()[0],pytest=importlib.metadata.version('pytest'),streamlit=importlib.metadata.version('streamlit'),source_sha256=SOURCE,default_pytest_temp=True,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(Path(__file__).with_name('phase_probe.py').read_bytes()).hexdigest(),clock='real production/application clocks unchanged',test_timeout_seconds=3,reporting_only_differences=['Tracebacks/captured output/warning summaries suppressed to exclude raw exception payloads.','Sanitized XML emitted from unchanged pytest reports; original assertions and outcomes untouched.']))
    return execute(root,preview,selection,out,args.collect_only)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,AssertionError):
        print('DIAGNOSTIC_CONTEXT_OR_SELECTION_REJECTED');raise SystemExit(2)
