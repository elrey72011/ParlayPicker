"""Offline regressions for bounded test scheduling, not acquisition evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import run_offline as runner

SOCKET_ATTEMPTS=0
RESULTS=[]
def deny_network(event,args):
    global SOCKET_ATTEMPTS
    if event in ('socket.connect','socket.getaddrinfo'):
        SOCKET_ATTEMPTS+=1
        raise AssertionError('REAL_NETWORK_FORBIDDEN')
sys.addaudithook(deny_network)

class Tests(unittest.TestCase):
    def test_bounded_lanes_overlap_actual_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            code='''import sys,time
from pathlib import Path
root=Path(sys.argv[1]); own=sys.argv[2]; peer=sys.argv[3]
(root/own).write_text('SYNTHETIC_SCHEDULER_ONLY')
deadline=time.monotonic()+10
while not (root/peer).exists():
 if time.monotonic()>deadline: raise RuntimeError('LANES_DID_NOT_OVERLAP')
 time.sleep(.01)
'''
            def heavy():
                subprocess.run([sys.executable,'-B','-c',code,str(root),'heavy','other'],check=True)
                return 'heavy'
            def other():
                subprocess.run([sys.executable,'-B','-c',code,str(root),'other','heavy'],check=True)
                return ['other']
            self.assertEqual(runner.run_in_two_lanes(heavy,other),['heavy','other'])
            self.assertTrue((root/'heavy').exists() and (root/'other').exists())
    def test_partitions_cover_every_case_once(self):
        records=[{'suite':'auth_recovery_suite','test':'Tests.'+runner.FULL_CORPUS_TEST},
                 {'suite':'auth_recovery_suite','test':'Tests.other'},
                 {'suite':'synthetic','test':'Tests.third'}]
        full=runner.select_collection(records,'full-corpus')
        standard=runner.select_collection(records,'standard')
        self.assertEqual(len(full),1);self.assertEqual(len(standard),2)
        self.assertEqual(runner.select_collection(records,'all'),records)
        self.assertEqual(sorted(full+standard,key=lambda x:x['test']),sorted(records,key=lambda x:x['test']))
        self.assertFalse(any(x in standard for x in full))
    def test_worker_failure_is_not_hidden(self):
        def fail():raise RuntimeError('SYNTHETIC_SCHEDULER_FAILURE')
        with self.assertRaisesRegex(RuntimeError,'^SYNTHETIC_SCHEDULER_FAILURE$'):
            runner.run_in_two_lanes(fail,lambda:['other'])
    def test_missing_duplicate_or_substituted_cases_rejected(self):
        collection=[{'suite':'synthetic','test':'Tests.one'},{'suite':'synthetic','test':'Tests.two'}]
        def result(names):return [{'suite':'synthetic','results':[{'test':'__main__.Tests.'+name} for name in names]}]
        runner.verify_case_identities(result(['one','two']),collection)
        for names in (['one'],['one','one','two'],['one','other']):
            with self.subTest(names=names),self.assertRaisesRegex(AssertionError,'ACCEPTANCE_IDENTITIES_CHANGED'):
                runner.verify_case_identities(result(names),collection)

class Recorded(unittest.TextTestResult):
    def startTest(self,test):self.begun=time.monotonic();super().startTest(test)
    def addSuccess(self,test):RESULTS.append({'test':test.id(),'status':'PASS','seconds':time.monotonic()-self.begun});super().addSuccess(test)
    def addFailure(self,test,error):RESULTS.append({'test':test.id(),'status':'FAIL','detail':self._exc_info_to_string(error,test)});super().addFailure(test,error)
    def addError(self,test,error):RESULTS.append({'test':test.id(),'status':'ERROR','detail':self._exc_info_to_string(error,test)});super().addError(test,error)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);a=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Recorded).run(suite)
    a.result.write_text(json.dumps({'tests_run':result.testsRun,'success':result.wasSuccessful(),'results':RESULTS,'real_socket_attempts':SOCKET_ATTEMPTS,'synthetic_only':True},indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)