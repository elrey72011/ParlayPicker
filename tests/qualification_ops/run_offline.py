"""Collect and execute all 82 portable regressions; retain sanitized summaries only."""
import argparse
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
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--collect-only',action='store_true')
    options=parser.parse_args()
    verify_application()
    output=options.output_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    collection=collect()
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
    results=[];commands=[]
    for name,extra in [('functional_suite',['--run-directory',str(output/'functional')]),('auth_recovery_suite',['--run-directory',str(output/'auth')]),('duration_suite',['--result',str(output/'duration.json')])]:
        command=[sys.executable,'-B','-X','utf8',str(HERE/(name+'.py')),*extra]
        begun=time.monotonic()
        with (output/(name+'.log')).open('wb') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        result_path=output/({'functional_suite':'functional/test-results.json','auth_recovery_suite':'auth/test-results.json','duration_suite':'duration.json'}[name])
        result=json.loads(result_path.read_bytes()) if result_path.is_file() else {'success':False,'tests_run':0,'results':[],'real_socket_attempts':'UNKNOWN'}
        elapsed=time.monotonic()-begun
        commands.append({'suite':name,'command':['python','-B','-X','utf8','tests/qualification_ops/'+name+'.py',*extra[:1],'PRIVATE_TEST_OUTPUT'],'exit_code':proc.returncode,'elapsed_seconds':elapsed})
        result['suite']=name;result['exit_code']=proc.returncode;results.append(result)
        print(name, 'tests=',result['tests_run'],'success=',result['success'],'exit=',proc.returncode,'seconds=',round(elapsed,3),flush=True)
    combined={'tooling_revision':subprocess.check_output(['git','-C',str(HERE.parents[1]),'rev-parse','HEAD'],text=True).strip(),'application_revision':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'driver_sha256':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),'synthetic_only':True,'collection_count':len(collection),'tests_run':sum(r['tests_run'] for r in results),'real_socket_attempts':sum(r['real_socket_attempts'] for r in results if isinstance(r['real_socket_attempts'],int)),'success':all(r['success'] and r['exit_code']==0 and r['real_socket_attempts']==0 for r in results),'commands':commands,'suites':[]}
    report=ET.Element('testsuites')
    for result in results:
        entries=result['results']
        suite=ET.SubElement(report,'testsuite',name=result['suite'],tests=str(result['tests_run']),failures=str(sum(item['status'] in ('FAIL','ERROR') for item in entries)),skipped=str(sum(item['status']=='SKIP' for item in entries)))
        for item in entries:
            case=ET.SubElement(suite,'testcase',name=item['test'],classname=result['suite'],time=str(item.get('seconds',item.get('elapsed_seconds',0))))
            if item['status']=='SKIP':ET.SubElement(case,'skipped').text=item.get('reason',item.get('fault_or_result','Platform-specific test'))
            elif item['status']!='PASS':ET.SubElement(case,'failure').text=item.get('detail',item.get('fault_or_result','Test failed'))
        summary={k:v for k,v in result.items() if k!='results'};summary['results']=entries
        if result['suite']=='auth_recovery_suite':
            summary['measurements']={p.parent.name:json.loads(p.read_bytes()) for p in (output/'auth').glob('*/measurements.json')}
        combined['suites'].append(summary)
    assert combined['tests_run']==82, 'ACCEPTANCE_EXECUTION_CHANGED'
    (output/'combined.json').write_text(json.dumps(combined,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    ET.ElementTree(report).write(output/'combined.xml',encoding='utf-8',xml_declaration=True)
    print('ALL 82:',combined['success'],'REAL SOCKET ATTEMPTS:',combined['real_socket_attempts'],flush=True)
    return 0 if combined['success'] else 1
if __name__=='__main__':raise SystemExit(main())
