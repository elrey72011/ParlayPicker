"""Network-free failure-report and actual-driver subprocess regressions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import failure_reports as reports
from paths import DRIVER

HERE = Path(__file__).resolve().parent


class ReportingTests(unittest.TestCase):
    def scenario(self, mode):
        with tempfile.TemporaryDirectory(prefix='ParlayPicker-reporting-') as temporary:
            root = Path(temporary); output = root / 'out'; output.mkdir()
            # The exact same child launcher is used; the test owns all paths.
            (root / 'child_runner.py').write_bytes((HERE / 'child_runner.py').read_bytes())
            (root / 'synthetic.py').write_text(textwrap.dedent('''
                import json,os,sys,unittest
                from pathlib import Path
                result=Path(sys.argv[2]); mode=sys.argv[3]
                print('postgresql://private-user:DO_NOT_RETAIN_PASSWORD@private-host.invalid/private-db',flush=True)
                if mode=='startup-error': raise ImportError('DO_NOT_RETAIN_PASSWORD')
                class Tests(unittest.TestCase):
                    def test_one(self): pass
                    def test_two(self):
                        if mode=='abrupt': os._exit(4)
                ran=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
                entries=[dict(test='Tests.'+n,status='PASS') for n in ('test_one','test_two')]
                if mode=='duplicate': entries[1]['test']='Tests.test_one'
                if mode=='substitution': entries[1]['test']='Tests.other'
                if mode=='wrong-class': entries[1]['test']='OtherTests.test_two'
                if mode=='conflict': entries[1]['status']='FAIL'
                if mode=='malformed': result.write_text('{DO_NOT_RETAIN_PASSWORD');sys.exit(0)
                if mode!='missing': result.write_text(json.dumps(dict(synthetic_only=True,success=True,tests_run=2,real_socket_attempts=0,results=entries)))
            '''), encoding='utf-8')
            collection = [dict(suite='synthetic', test='Tests.' + n) for n in ('test_one', 'test_two')]
            pair = reports.execute(root, output, 'synthetic', ['--result', str(output / 'result.json'), mode],
                                   'result.json', 'synthetic', dict(os.environ), collection)
            # Import lazily: this module is also collected by run_offline.
            from run_offline import verify_case_identities
            combined = reports.combined_report(output, collection, [pair], dict(synthetic_only=True), verify_case_identities)
            self.assertTrue((output / 'combined.json').is_file())
            self.assertTrue((output / 'combined.xml').is_file())
            import xml.etree.ElementTree as ET
            ET.parse(output / 'combined.xml')
            retained = ''.join(p.read_text() for p in (output / 'diagnostics').iterdir()) + (output / 'combined.json').read_text()
            for secret in ('DO_NOT_RETAIN_PASSWORD', 'private-host.invalid', 'postgresql://', 'private-user'):
                self.assertNotIn(secret, retained)
            self.assertFalse(list(output.glob('.private-*')))
            return combined, pair

    def test_abrupt_exit_retains_started_and_completed_without_zero_fabrication(self):
        result, pair = self.scenario('abrupt')
        self.assertEqual(pair[1]['exit_code'], 4)
        self.assertEqual(result['completed_count'], 1)
        self.assertEqual(result['started_confirmed_count'], 2)
        self.assertEqual(result['tests_run'], 'UNKNOWN')
        self.assertEqual(result['real_socket_attempts'], 'UNKNOWN')
        self.assertEqual(len(result['unconfirmed']), 1)
        self.assertFalse(result['execution_complete'] or result['success'])

    def test_missing_final_result_preserves_journal_but_still_fails(self):
        result, pair = self.scenario('missing')
        self.assertEqual(pair[1]['result_status'], 'RESULT_MISSING')
        self.assertEqual(result['completed_count'], 2)
        self.assertEqual(result['tests_run'], 'UNKNOWN')
        self.assertFalse(result['execution_complete'] or result['success'])

    def test_malformed_final_result_preserves_journal_but_still_fails(self):
        result, pair = self.scenario('malformed')
        self.assertEqual(pair[1]['result_status'], 'RESULT_MALFORMED')
        self.assertEqual(result['completed_count'], 2)
        self.assertFalse(result['execution_complete'] or result['success'])

    def test_duplicate_substituted_and_conflicting_results_still_fail(self):
        for mode in ('duplicate', 'substitution', 'wrong-class', 'conflict'):
            with self.subTest(mode=mode):
                result, pair = self.scenario(mode)
                self.assertFalse(result['success'])
                self.assertEqual(pair[1]['result_status'], 'RESULT_JOURNAL_CONFLICT')
                if mode != 'conflict':
                    self.assertIn('ACCEPTANCE_IDENTITIES_CHANGED', result['identity_errors'])

    def test_valid_results_are_complete_without_changing_existing_identity_check(self):
        result, _ = self.scenario('valid')
        self.assertTrue(result['execution_complete'] and result['success'])
        self.assertEqual(result['tests_run'], 2)
        self.assertEqual(result['real_socket_attempts'], 0)

    def test_start_failure_still_writes_partial_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(reports.subprocess, 'run', side_effect=OSError('DO_NOT_RETAIN_PASSWORD')):
                pair = reports.execute(root, root, 'synthetic', ['--result', str(root / 'result')], 'result', 'synthetic', {}, [])
            self.assertEqual(pair[1]['phase'], 'CHILD_START_FAILED')
            self.assertEqual(pair[1]['exit_code'], 'UNKNOWN')
            self.assertNotIn('DO_NOT_RETAIN_PASSWORD', ''.join(p.read_text() for p in (root / 'diagnostics').iterdir()))
        result, pair = self.scenario('startup-error')
        self.assertEqual(pair[1]['exit_code'], 1)
        self.assertEqual(result['tests_run'], 'UNKNOWN')
        self.assertFalse(result['execution_complete'] or result['success'])

    def test_actual_frozen_clock_case_survives_controlled_slow_synthetic_work(self):
        import functional_suite as functional
        with tempfile.TemporaryDirectory(prefix='ParlayPicker-slow-case-') as temporary:
            root = Path(temporary)
            code = '''import sys,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1]);import functional_suite as f
f.RUN_DIR=Path(sys.argv[2]);f.DRIVER=f.RUNTIME_DRIVER
original=f.FunctionalTests.setUp
def setup(self):
 original(self);capture=self.m.capture
 def slow(spec,number):
  assert self.m.time.monotonic()==0
  time.sleep(3)
  return capture(spec,number)
 hook=patch.object(self.m,'capture',slow);hook.start();self.addCleanup(hook.stop)
f.FunctionalTests.setUp=setup
r=unittest.TextTestRunner().run(unittest.TestSuite([f.FunctionalTests('test_V06_capture_total_wall_stops_without_next_slice')]))
assert f.REAL_REQUESTS==0
raise SystemExit(0 if r.wasSuccessful() else 1)
'''
            child = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code, str(HERE), str(root)], capture_output=True, timeout=30)
            self.assertEqual(child.returncode, 0, 'Controlled logical-clock child failed; raw details withheld')
            self.assertEqual(functional.REAL_REQUESTS, 0)

    def test_real_watchdog_still_exits_owned_frozen_clock_process_with_four(self):
        code = '''import importlib.util,sys,time
from types import SimpleNamespace
def deny(event,args):
 if event in ('socket.connect','socket.getaddrinfo'): raise AssertionError('OFFLINE_SOCKET_DENIED')
sys.addaudithook(deny)
s=importlib.util.spec_from_file_location('unchanged_driver',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.time=SimpleNamespace(monotonic=lambda:0)
m.capture=lambda *a:time.sleep(5)
m.capture_block(dict(destination='SYNTHETIC_UNUSED',max_slices=1,capture_total_wall_seconds=.2,slice_wall_seconds=.2))
raise AssertionError('WATCHDOG_DID_NOT_STOP_PROCESS')
'''
        child = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(DRIVER)], capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 4)
        self.assertEqual(child.stderr.strip(), b'STAGE_WALL_LIMIT')


if __name__ == '__main__':
    import argparse
    import runner_scheduling_suite as recorded
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    options = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReportingTests)
    result = unittest.TextTestRunner(verbosity=2, resultclass=recorded.Recorded).run(suite)
    reports.write_json(options.result, dict(tests_run=result.testsRun, success=result.wasSuccessful(),
                       results=recorded.RESULTS, real_socket_attempts=recorded.SOCKET_ATTEMPTS, synthetic_only=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
