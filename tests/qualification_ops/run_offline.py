"""Collect the prior 97 acceptance cases plus review-closure regressions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from failure_reports import execute as execute_child, combined_report, case_key
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
    module=load('reporting_regressions')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.ReportingTests)
    records.extend({'suite':'reporting_regressions','test':test.id()} for test in suite)
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
    observed=[case_key(r['suite'],item['test']) for r in results for item in r['results']]
    expected=[case_key(item['suite'],item['test']) for item in collection]
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
    for name in ('ODDS_API_KEY','THE_ODDS_API_KEY','CFBD_API_KEY','CBBD_API_KEY',
                 'PARLAYPICKER_NFL_DATABASE_URL','PARLAYPICKER_NFL_STORE_KEY','PARLAYPICKER_PUBLISH_TOKEN'):
        env.pop(name,None)
    env['PARLAYPICKER_NFL_ACQUISITION_ENABLED']='false'
    env['PARLAYPICKER_NFL_INFERENCE_ENABLED']='false'
    env['PARLAYPICKER_QUALIFICATION_APPLICATION_CHECKOUT']=str(SOURCE)
    env['PYTHONDONTWRITEBYTECODE']='1'
    # Start the full 27,580-object lifecycle first; independent suites occupy
    # one other child-process slot. Test bodies and resource limits are unchanged.
    heavy_selector=FULL_CORPUS_TEST
    completed=[];report_lock=threading.Lock()
    metadata={'partition':options.partition,'tooling_revision':subprocess.check_output(['git','-C',str(HERE.parents[1]),'rev-parse','HEAD'],text=True).strip(),'application_revision':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'driver_sha256':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),'synthetic_only':True}
    metadata['runtime']={'python':sys.version.split()[0],'dependencies':{name:importlib.metadata.version(name) for name in ('google-auth','requests','cryptography','pytest','numpy','pandas')}}
    metadata['tooling_checkout_clean']=not bool(subprocess.check_output(['git','-C',str(HERE.parents[1]),'status','--porcelain=v1','--untracked-files=no'],text=True).strip())
    # Initial incomplete report exists even before the first child starts.
    combined_report(output,collection,completed,metadata,verify_case_identities)
    def execute(name,extra,result_file,label):
        print('STARTED:',label,flush=True)
        pair=execute_child(HERE,output,name,extra,result_file,label,env,collection)
        with report_lock:
            completed.append(pair)
            combined_report(output,collection,completed,metadata,verify_case_identities)
        result,record=pair
        print(label,'tests=',result['tests_run'],'success=',result['success'],'exit=',record['exit_code'],'seconds=',round(record['elapsed_seconds'],3),flush=True)
        return pair
    heavy=('auth_recovery_suite',['--run-directory',str(output/'auth-heavy'),'--select',heavy_selector],'auth-heavy/test-results.json','auth-full-corpus')
    jobs=[('functional_suite',['--run-directory',str(output/'functional')],'functional/test-results.json','functional_suite'),
          ('auth_recovery_suite',['--run-directory',str(output/'auth'),'--exclude',heavy_selector],'auth/test-results.json','auth-other-cases'),
          ('duration_suite',['--result',str(output/'duration.json')],'duration.json','duration_suite'),
          ('mirror_recovery_suite',['--run-directory',str(output/'mirror')],'mirror/test-results.json','mirror_recovery_suite'),
          ('review_closure_suite',['--run-directory',str(output/'review')],'review/test-results.json','review_closure_suite'),
          ('runner_scheduling_suite',['--result',str(output/'runner-scheduling.json')],'runner-scheduling.json','runner_scheduling_suite'),
          ('reporting_regressions',['--result',str(output/'reporting-regressions.json')],'reporting-regressions.json','reporting_regressions')]
    def serial_others():return [execute(*job) for job in jobs]
    if options.partition=='full-corpus':partitions=[execute(*heavy)]
    elif options.partition=='standard':partitions=serial_others()
    else:partitions=run_in_two_lanes(lambda:execute(*heavy),serial_others)
    combined=combined_report(output,collection,partitions,metadata,verify_case_identities)
    print('COMPLETE:',combined['execution_complete'],'SUCCESS:',combined['success'],'CONFIRMED COMPLETED:',combined['completed_count'],'REAL SOCKET ATTEMPTS:',combined['real_socket_attempts'],flush=True)
    return 0 if combined['success'] else 1
if __name__=='__main__':raise SystemExit(main())
