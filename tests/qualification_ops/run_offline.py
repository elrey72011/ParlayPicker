"""Collect the prior 97 acceptance cases plus review-closure regressions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from paths import DRIVER, SOURCE, verify_application

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def collect():
    import unittest
    records=[]
    for name,cls in [('functional_suite','FunctionalTests'),('auth_recovery_suite','Tests')]:
        module=load(name)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module,cls))
        records.extend({'suite':name,'test':test.id()} for test in suite)
    records.extend({'suite':'duration_suite','test':name} for name in ['full_24000_capture_27000_whole_envelope','expiry_faults_stop_at_20_without_completion_claim'])
    assert len(records)==82, ('ACCEPTANCE_COLLECTION_CHANGED',len(records))
    module=load('mirror_recovery_suite')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.Tests)
    records.extend({'suite':'mirror_recovery_suite','test':test.id()} for test in suite)
    assert len(records)==97, ('MIRROR_ACCEPTANCE_COLLECTION_CHANGED',len(records))
    module=load('review_closure_suite')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.Tests)
    records.extend({'suite':'review_closure_suite','test':test.id()} for test in suite)
    assert len(records)==104, ('REVIEW_CLOSURE_COLLECTION_CHANGED',len(records))
    module=load('runner_scheduling_suite')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.Tests)
    records.extend({'suite':'runner_scheduling_suite','test':test.id()} for test in suite)
    return records

FULL_CORPUS_TEST='test_A05_27580_objects_virtual_full_duration'

def select_collection(records,partition):
    if partition=='all':return records
    def full(item):return item['suite']=='auth_recovery_suite' and item['test'].rsplit('.',1)[-1]==FULL_CORPUS_TEST
    return [item for item in records if full(item)==(partition=='full-corpus')]

def run_in_two_lanes(full_corpus,other_cases):
    with ThreadPoolExecutor(max_workers=2) as executor:
        full_future=executor.submit(full_corpus)
        other_future=executor.submit(other_cases)
        return [full_future.result(),*other_future.result()]

def verify_case_identities(results,collection):
    observed=[(r['suite'],item['test'].rsplit('.',1)[-1]) for r in results for item in r['results']]
    expected=[(item['suite'],item['test'].rsplit('.',1)[-1]) for item in collection]
    assert len(observed)==len(set(observed)) and set(observed)==set(expected),'ACCEPTANCE_IDENTITIES_CHANGED'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--collect-only',action='store_true')
    parser.add_argument('--partition',choices=('all','standard','full-corpus'),default='all')
    options=parser.parse_args()
    verify_application()
    output=options.output_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    collection=select_collection(collect(),options.partition)
    (output/'collection.json').write_text(json.dumps(collection,indent=2)+'\n',encoding='utf-8')
    print('COLLECTED:',len(collection),flush=True)
    if options.collect_only:
        for item in collection: print(item['suite']+':'+item['test'])
        return 0
    env=dict(os.environ)
    for name in ('PARLAYPICKER_DRIVE_FOLDER_ID','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT','PARLAYPICKER_OPERATIONS_WORKER_TOKEN'):
        env.pop(name,None)
    env['PARLAYPICKER_QUALIFICATION_APPLICATION_CHECKOUT']=str(SOURCE)
    env['PYTHONDONTWRITEBYTECODE']='1'
    # Start the full 27,580-object lifecycle first; independent suites occupy
    # one other child-process slot. Test bodies and resource limits are unchanged.
    heavy_selector=FULL_CORPUS_TEST
    def execute(name,extra,result_file,label):
        command=[sys.executable,'-B','-X','utf8',str(HERE/(name+'.py')),*extra]
        begun=time.monotonic()
        print('STARTED:',label,flush=True)
        with (output/(label+'.log')).open('wb') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        result_path=output/result_file
        result=json.loads(result_path.read_bytes()) if result_path.is_file() else {'success':False,'tests_run':0,'results':[],'real_socket_attempts':'UNKNOWN'}
        elapsed=time.monotonic()-begun
        sanitized=['python','-B','-X','utf8','tests/qualification_ops/'+name+'.py',extra[0],'PRIVATE_TEST_OUTPUT',*extra[2:]]
        record={'suite':name,'partition':label,'command':sanitized,'exit_code':proc.returncode,'elapsed_seconds':elapsed}
        result['suite']=name;result['exit_code']=proc.returncode
        print(label,'tests=',result['tests_run'],'success=',result['success'],'exit=',proc.returncode,'seconds=',round(elapsed,3),flush=True)
        return result,record
    heavy=('auth_recovery_suite',['--run-directory',str(output/'auth-heavy'),'--select',heavy_selector],'auth-heavy/test-results.json','auth-full-corpus')
    jobs=[('functional_suite',['--run-directory',str(output/'functional')],'functional/test-results.json','functional_suite'),
          ('auth_recovery_suite',['--run-directory',str(output/'auth'),'--exclude',heavy_selector],'auth/test-results.json','auth-other-cases'),
          ('duration_suite',['--result',str(output/'duration.json')],'duration.json','duration_suite'),
          ('mirror_recovery_suite',['--run-directory',str(output/'mirror')],'mirror/test-results.json','mirror_recovery_suite'),
          ('review_closure_suite',['--run-directory',str(output/'review')],'review/test-results.json','review_closure_suite'),
          ('runner_scheduling_suite',['--result',str(output/'runner-scheduling.json')],'runner-scheduling.json','runner_scheduling_suite')]
    def serial_others():return [execute(*job) for job in jobs]
    if options.partition=='full-corpus':partitions=[execute(*heavy)]
    elif options.partition=='standard':partitions=serial_others()
    else:partitions=run_in_two_lanes(lambda:execute(*heavy),serial_others)
    results=[];commands=[]
    for result,record in partitions:
        commands.append(record)
        matching=next((r for r in results if r['suite']==result['suite']),None)
        if matching is None:results.append(result)
        else:
            matching['results']+=result['results'];matching['tests_run']+=result['tests_run']
            for field in ('failures','errors'):
                if field in matching and field in result:matching[field]+=result[field]
            matching['success']=matching['success'] and result['success']
            matching['exit_code']=matching['exit_code'] or result['exit_code']
            if isinstance(matching['real_socket_attempts'],int) and isinstance(result['real_socket_attempts'],int):matching['real_socket_attempts']+=result['real_socket_attempts']
            else:matching['real_socket_attempts']='UNKNOWN'
    verify_case_identities(results,collection)
    combined={'partition':options.partition,'tooling_revision':subprocess.check_output(['git','-C',str(HERE.parents[1]),'rev-parse','HEAD'],text=True).strip(),'application_revision':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'driver_sha256':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),'synthetic_only':True,'collection_count':len(collection),'tests_run':sum(r['tests_run'] for r in results),'real_socket_attempts':sum(r['real_socket_attempts'] for r in results if isinstance(r['real_socket_attempts'],int)),'success':all(r['success'] and r['exit_code']==0 and r['real_socket_attempts']==0 for r in results),'commands':commands,'suites':[]}
    report=ET.Element('testsuites')
    for result in results:
        entries=result['results']
        suite=ET.SubElement(report,'testsuite',name=result['suite'],tests=str(result['tests_run']),failures=str(sum(item['status'] in ('FAIL','ERROR') for item in entries)),skipped=str(sum(item['status']=='SKIP' for item in entries)))
        for item in entries:
            case=ET.SubElement(suite,'testcase',name=item['test'],classname=result['suite'],time=str(item.get('seconds',item.get('elapsed_seconds',0))))
            if item['status']=='SKIP':ET.SubElement(case,'skipped').text=item.get('reason',item.get('fault_or_result','Platform-specific test'))
            elif item['status']!='PASS':ET.SubElement(case,'failure').text=item.get('detail',item.get('fault_or_result','Test failed'))
        summary={k:v for k,v in result.items() if k!='results'};summary['results']=entries
        if result['suite'] in ('auth_recovery_suite','mirror_recovery_suite','review_closure_suite'):
            folders=('auth','auth-heavy') if result['suite']=='auth_recovery_suite' else ({'mirror_recovery_suite':'mirror','review_closure_suite':'review'}[result['suite']],)
            summary['measurements']={p.parent.name:json.loads(p.read_bytes()) for folder in folders for p in (output/folder).glob('*/measurements.json')}
        combined['suites'].append(summary)
    assert combined['tests_run']==len(collection), 'ACCEPTANCE_EXECUTION_CHANGED'
    (output/'combined.json').write_text(json.dumps(combined,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    ET.ElementTree(report).write(output/'combined.xml',encoding='utf-8',xml_declaration=True)
    print('ALL PRIOR 97 + REVIEW CLOSURE CASES:',combined['success'],'REAL SOCKET ATTEMPTS:',combined['real_socket_attempts'],flush=True)
    return 0 if combined['success'] else 1
if __name__=='__main__':raise SystemExit(main())
