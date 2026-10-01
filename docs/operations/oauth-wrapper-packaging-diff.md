# Local-to-tracked packaging diff

Driver and baseline fixture bytes are unchanged. Test/source-only paths below redact the original machine-specific application path; hashes bind the exact original artifacts. No assertion is weakened. The former private baseline/spec preservation case now verifies preserved v1/v2 source bytes and every real resource cap; authentic private-file preservation is checked locally outside CI.

## tools/qualification/snapshot_acquire_and_assess.py

Original SHA256: `8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93`. Tracked SHA256: `8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93`.

Empty byte/content diff; relocation only.

## tests/qualification_ops/auth_recovery_suite.py

Original SHA256: `d04cfdd0188f183add8b5b93ae3d86c374e874dc2d8d2c332a80f03a25a79716`. Tracked SHA256: `6a52cd4ccd57874215b84bbb26ad3a446dab1d3773857bb18b34ee8c2168b050`.

```diff
--- local/offline-auth-recovery-tests.py
+++ tests/qualification_ops/auth_recovery_suite.py
@@ -30,9 +30,11 @@
 from urllib.parse import urlparse, parse_qs

 HERE=Path(__file__).resolve().parent
-SOURCE=Path('<PRIVATE_PINNED_APPLICATION_CHECKOUT>')
+sys.path.insert(0,str(HERE))
+from paths import SOURCE, DRIVER, LEGACY, verify_application
+verify_application()
 sys.path.insert(0,str(SOURCE))
-spec=importlib.util.spec_from_file_location('prior_validation',HERE.parent/'offline-functional-validation.py')
+spec=importlib.util.spec_from_file_location('prior_validation',HERE/'functional_suite.py')
 prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
 from app_core import evidence_config, evidence_drive, prospective_evidence as evidence, prospective_remote as codec
 from google.auth import _helpers
@@ -156,7 +158,7 @@
             self.temporary.cleanup()
         self.addCleanup(retain)
         self.fx=fixture(self.tmp/'synthetic',40)
-        self.m=prior.load_driver(HERE/'snapshot-acquire-and-assess-v2.py');self.m.source_check(self.fx.spec)
+        self.m=prior.load_driver(DRIVER);self.m.source_check(self.fx.spec)
         self.backend=AuthWire(self.fx.files,self.fx.wire)
         self.measure={};self.faults=[]
         self.addCleanup(lambda:write(self.case/'measurements.json',dict(self.measure,faults=self.faults,real_socket_attempts=REAL_SOCKET_ATTEMPTS,synthetic_only=True)))
@@ -208,7 +210,7 @@
     def hashes(self): return {str(p.relative_to(self.fx.root)):sha(p) for p in self.fx.root.rglob('*') if p.is_file()}

     def test_A02_old_auth_churn_actual_reader(self):
-        self.fresh();self.m=prior.load_driver(HERE.parent/'snapshot-acquire-and-assess.py');self.m.source_check(self.fx.spec)
+        self.fresh();self.m=prior.load_driver(LEGACY);self.m.source_check(self.fx.spec)
         self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=sha(self.fx.path))
         with RealAuthRuntime(self.backend): self.fails(lambda:self.m.capture(self.fx.spec,1),'OAUTH_REQUEST_LIMIT')
         state=self.m.load_state(self.fx.root,self.fx.spec);counts=[self.m.check_seal(self.fx.root,ref)['transport']['oauth_attempts'] for ref in state['accepted_batches']]
@@ -348,7 +350,7 @@
         proposed=self.m.recovery_policy(self.fx.spec,h,20,7)
         self.assertEqual(proposed['max_oauth_token_attempts'],40)
         proposed['destination']=str(self.tmp/'SYNTHETIC-SUCCESSOR');proposed['max_new_objects_per_slice']=5000
-        recovery={'predecessor_destination':str(self.fx.root),'original_driver_sha256':sha(HERE.parent/'snapshot-acquire-and-assess.py'),'additional_oauth_allowance':20}
+        recovery={'predecessor_destination':str(self.fx.root),'original_driver_sha256':sha(LEGACY),'additional_oauth_allowance':20}
         self.m.args.recovery_spec=self.case/'synthetic-addendum.json';write(self.m.args.recovery_spec,{'synthetic_only':True,'additional_oauth':20})
         self.m.initialize_successor(proposed,recovery,h,'SYNTHETIC-WORKER-TOKEN')
         successor=Path(proposed['destination']);self.assertEqual(len(list((successor/'raw').iterdir())),32)
@@ -386,10 +388,10 @@
         successor=self.fx.root.with_name('acquire-'+str(self.fx.spec['anchor_run_id'])+'-recovery-v2')
         expected={key:h[key] for key in ('original_operation_id','accepted_count','incurred_transport','state_sha256','journal_sha256','journal_tail_sha256')}
         expected['file_register_canonical_sha256']=hashlib.sha256(self.m.canonical(h['files'])).hexdigest()
-        addendum={'original_driver_path':str(HERE.parent/'snapshot-acquire-and-assess.py'),'original_driver_sha256':sha(HERE.parent/'snapshot-acquire-and-assess.py'),'replacement_driver_sha256':sha(HERE/'snapshot-acquire-and-assess-v2.py'),'original_spec_sha256':sha(self.fx.path),'predecessor_destination':str(self.fx.root),'successor_destination':str(successor),'expected_predecessor':expected,'additional_oauth_allowance':20,'successor_max_slices':7,'historical_elapsed_seconds':54.171,'status':'SYNTHETIC_APPROVAL_ONLY'}
+        addendum={'original_driver_path':str(LEGACY),'original_driver_sha256':sha(LEGACY),'replacement_driver_sha256':sha(DRIVER),'original_spec_sha256':sha(self.fx.path),'predecessor_destination':str(self.fx.root),'successor_destination':str(successor),'expected_predecessor':expected,'additional_oauth_allowance':20,'successor_max_slices':7,'historical_elapsed_seconds':54.171,'status':'SYNTHETIC_APPROVAL_ONLY'}
         addendum_path=self.case/'SYNTHETIC-recovery-spec.json';write(addendum_path,addendum)
         self.fx.backend_file()
-        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=sha(self.fx.path),recovery_spec=addendum_path,approved_recovery_sha256=sha(addendum_path),approved_driver_sha256=sha(HERE/'snapshot-acquire-and-assess-v2.py'))
+        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=sha(self.fx.path),recovery_spec=addendum_path,approved_recovery_sha256=sha(addendum_path),approved_driver_sha256=sha(DRIVER))
         original=subprocess.Popen
         def child(command,**kwargs):
             if '--worker' not in command:return original(command,**kwargs)
@@ -473,5 +475,5 @@
         suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
         if a.select:suite=unittest.TestSuite(t for t in suite if a.select in t.id())
         result=unittest.TextTestRunner(verbosity=2,resultclass=Recorded).run(suite)
-        write(RUN/'test-results.json',{'driver_sha256':sha(HERE/'snapshot-acquire-and-assess-v2.py'),'tests_run':result.testsRun,'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'results':RESULTS,'real_socket_attempts':REAL_SOCKET_ATTEMPTS,'dependency_versions':{n:importlib.metadata.version(n) for n in ('google-auth','requests','cryptography','pytest')},'synthetic_only':True})
+        write(RUN/'test-results.json',{'driver_sha256':sha(DRIVER),'tests_run':result.testsRun,'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'results':RESULTS,'real_socket_attempts':REAL_SOCKET_ATTEMPTS,'dependency_versions':{n:importlib.metadata.version(n) for n in ('google-auth','requests','cryptography','pytest')},'synthetic_only':True})
         sys.exit(0 if result.wasSuccessful() else 1)
```

## tests/qualification_ops/functional_suite.py

Original SHA256: `93b1be4c2721ebd9b6419d159e9faa3e83fa212febe2a0c33b7fbf2940174486`. Tracked SHA256: `78b5fb5858f1fcedfc254a0b384785d7116c752ba511f887c699aecb21d3fd76`.

```diff
--- local/offline-functional-validation-v2.py
+++ tests/qualification_ops/functional_suite.py
@@ -31,8 +31,10 @@
 from urllib.parse import urlparse, parse_qs
 import zipfile

-HERE=Path(__file__).resolve().parent.parent
-SOURCE=Path('<PRIVATE_PINNED_APPLICATION_CHECKOUT>')
+HERE=Path(__file__).resolve().parent
+sys.path.insert(0,str(HERE))
+from paths import SOURCE, DRIVER as TRACKED_DRIVER, LEGACY, TEMPLATE, verify_application
+verify_application()
 sys.path.insert(0,str(SOURCE))
 from app_core import prospective_remote as codec, prospective_evidence as evidence
 from app_core import evidence_drive, read_only_census as census, prospective_validation_plans as policy
@@ -63,7 +65,8 @@
     def __init__(self,base,number=16):
         self.base=Path(base);self.base.mkdir(parents=True)
         self.retained=self.base/'synthetic-anchor';(self.retained/'slice-01').mkdir(parents=True)
-        self.spec=copy.deepcopy(json.loads((HERE/'snapshot-acquisition-spec.json').read_bytes()))
+        self.spec=copy.deepcopy(json.loads(TEMPLATE.read_bytes()))
+        self.spec['source_checkout']=str(SOURCE)
         self.spec['retained_directory']=str(self.retained)
         self.spec['destination']=str(self.base/'local-app-data'/'ParlayPicker'/'qualification-canonical-snapshot'/self.spec['source_revision']/f"acquire-{self.spec['anchor_run_id']}-v1")
         self.spec['required_initial_free_disk_bytes']=1 # Scaled fixture limit; REAL SPEC IS NEVER EDITED.
@@ -346,10 +349,11 @@
         self.m.materialize(self.fx.spec)
         self.assertTrue((self.fx.root/'canonical.sqlite3').is_file())
     def test_V01_original_hashes_and_unchanged_envelope(self):
-        baseline=HERE/'offline-validation-20260930'/'baseline'
-        self.assertEqual(sha((baseline/'snapshot-acquire-and-assess.py').read_bytes()),'ec815e40cb3aabec61a90fac8a26b6d2add38a338070a023e8a3074ad150929d')
-        self.assertEqual(sha((HERE/'snapshot-acquisition-spec.json').read_bytes()),'b1f6681d9dcfacc92c55fa68de60259423a1e501823236b211c249f0b07c1892')
-        self.assertEqual((baseline/'snapshot-acquisition-spec.json').read_bytes(),(HERE/'snapshot-acquisition-spec.json').read_bytes())
+        self.assertEqual(sha(LEGACY.read_bytes()),'6c01b00da684956f4319017c3b7f08b78eafc5133b74de98697d69fad287de67')
+        self.assertEqual(sha(TRACKED_DRIVER.read_bytes()),'8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93')
+        envelope=json.loads(TEMPLATE.read_bytes())
+        for key,value in {'max_slices':8,'max_new_objects_per_slice':5000,'soft_bytes_per_slice':500000000,'slice_processing_seconds':2700,'slice_wall_seconds':3000,'soft_bytes_total':4000000000,'transport_observed_body_bytes_stop':5000000000,'max_drive_get_attempts':90000,'max_oauth_token_attempts':20,'capture_total_wall_seconds':24000,'materialization_wall_seconds':900,'assessment_wall_seconds':900,'total_wall_seconds':27000,'max_working_disk_bytes':12000000000,'required_initial_free_disk_bytes':14000000000,'max_snapshot_input_bytes':2000000000}.items():
+            self.assertEqual(envelope[key],value,key)
     def test_V03_all_mutations_and_unapproved_endpoints_denied(self):
         session,meter=self.transport();folder='https://www.googleapis.com/drive/v3/files/fixture-folder'
         attempts=[(method,folder) for method in ('POST','PUT','PATCH','DELETE','HEAD')]
@@ -678,6 +682,7 @@
     def startTest(self,test): self.starts[test.id()]=time.monotonic();super().startTest(test)
     def record(self,test,status,detail=None): self.records.append({'test':test.id(),'status':status,'elapsed_seconds':time.monotonic()-self.starts[test.id()],'fault_or_result':detail})
     def addSuccess(self,test): self.record(test,'PASS');super().addSuccess(test)
+    def addSkip(self,test,reason): self.record(test,'SKIP',reason);super().addSkip(test,reason)
     def addFailure(self,test,error): self.record(test,'FAIL',self._exc_info_to_string(error,test));super().addFailure(test,error)
     def addError(self,test,error): self.record(test,'ERROR',self._exc_info_to_string(error,test));super().addError(test,error)

@@ -698,10 +703,10 @@
             finally:
                 write_json(backend_path.with_name('fake-child-'+argv[argv.index('--worker')+1]+'-attempts.json'),{'real_socket_attempts':REAL_REQUESTS,'fake_requests':backend.calls,'synthetic_only':True})
     else:
-        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--driver',type=Path,default=HERE/'snapshot-acquire-and-assess.py');parser.add_argument('--run-directory',type=Path,required=True);parser.add_argument('--select');arguments=parser.parse_args()
+        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--driver',type=Path,default=TRACKED_DRIVER);parser.add_argument('--run-directory',type=Path,required=True);parser.add_argument('--select');arguments=parser.parse_args()
         DRIVER=arguments.driver.resolve();RUN_DIR=arguments.run_directory.resolve();RUN_DIR.mkdir(parents=True,exist_ok=False)
         suite=unittest.defaultTestLoader.loadTestsFromTestCase(FunctionalTests)
         if arguments.select: suite=unittest.TestSuite(test for test in suite if arguments.select in test.id())
         result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
-        write_json(RUN_DIR/'test-results.json',{'driver_bytes':DRIVER.stat().st_size,'driver_sha256':sha(DRIVER.read_bytes()),'spec_sha256':sha((HERE/'snapshot-acquisition-spec.json').read_bytes()),'synthetic_only':True,'real_socket_attempts':REAL_REQUESTS,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'results':result.records,'success':result.wasSuccessful()})
+        write_json(RUN_DIR/'test-results.json',{'driver_bytes':DRIVER.stat().st_size,'driver_sha256':sha(DRIVER.read_bytes()),'spec_sha256':sha(TEMPLATE.read_bytes()),'synthetic_only':True,'real_socket_attempts':REAL_REQUESTS,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'results':result.records,'success':result.wasSuccessful()})
         sys.exit(0 if result.wasSuccessful() else 1)
```

## tests/qualification_ops/duration_suite.py

Original SHA256: `d56125613f5f0ecd2d313582d9fa5235f422fed36828135e9d0df58f69376a86`. Tracked SHA256: `b3fb8209c1db3714549872cf171d80ebec6e9e0b5a9db52deec8277455087943`.

```diff
--- local/offline-duration-and-failure-tests.py
+++ tests/qualification_ops/duration_suite.py
@@ -8,13 +8,16 @@
 from concurrent.futures import ThreadPoolExecutor

 HERE=Path(__file__).resolve().parent
-i=importlib.util.spec_from_file_location('auth_tests',HERE/'offline-auth-recovery-tests.py');a=importlib.util.module_from_spec(i);i.loader.exec_module(a)
-out=HERE/'duration-and-failure-results.json'
+i=importlib.util.spec_from_file_location('auth_tests',HERE/'auth_recovery_suite.py');a=importlib.util.module_from_spec(i);i.loader.exec_module(a)
+import argparse
+parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);arguments=parser.parse_args()
+out=arguments.result
+assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 records=[]
 with tempfile.TemporaryDirectory(prefix='ParlayPicker-offline-duration-') as temp:
     root=Path(temp)
     fx=a.fixture(root/'full-envelope',40);fx.spec['max_new_objects_per_slice']=5;fx.save_spec()
-    m=a.prior.load_driver(HERE/'snapshot-acquire-and-assess-v2.py');m.source_check(fx.spec);fx.initialize(m)
+    m=a.prior.load_driver(a.DRIVER);m.source_check(fx.spec);fx.initialize(m)
     backend=a.AuthWire(fx.files,fx.wire);clock=backend.clock
     verified=m.verified_record
     charged=set()
@@ -43,7 +46,7 @@
         assert clock.seconds==27000 and backend.oauth_posts==at_capture_end
         assert backend.oauth_posts==5
     records.append({'test':'full_24000_capture_27000_whole_envelope','status':'PASS','actual_oauth_posts':backend.oauth_posts,'slice_oauth_totals':[verified(fx.root/f'slice-{n:02d}-result.json')['transport']['oauth_attempts'] for n in range(1,9)],'actual_get_attempts':meter['drive_get_attempts'],'simulated_capture_seconds':24000,'simulated_whole_seconds':27000,'credential_constructions':backend.actual_session_constructions,'assumption':'3000s boundary intervals, 3600s tokens, final concurrent expiry probe at23999; no authentication needed in local-only phases','throughput_claim':False})
-    fx=a.fixture(root/'fault-budget',160);m=a.prior.load_driver(HERE/'snapshot-acquire-and-assess-v2.py');m.source_check(fx.spec);fx.initialize(m)
+    fx=a.fixture(root/'fault-budget',160);m=a.prior.load_driver(a.DRIVER);m.source_check(fx.spec);fx.initialize(m)
     backend=a.AuthWire(fx.files,fx.wire);backend.media_seconds=3600
     with a.RealAuthRuntime(backend):
         try:m.capture(fx.spec,1);raise AssertionError('BUDGET_FAILURE_EXPECTED')
@@ -53,5 +56,5 @@
     records.append({'test':'expiry_faults_stop_at_20_without_completion_claim','status':'PASS','actual_oauth_posts':20,'counted_oauth_attempts':20,'reason':'OAUTH_REQUEST_LIMIT','accepted_objects':sum(len(m.check_seal(fx.root,ref)['objects']) for ref in m.load_state(fx.root,fx.spec)['accepted_batches']),'actual_get_attempts':meter['drive_get_attempts'],'real_wait_for_expiry':False})
     # Only test-owned temporary files are removed by this context manager.
     assert root.resolve().parent==Path(tempfile.gettempdir()).resolve() and root.name.startswith('ParlayPicker-offline-duration-')
-a.write(out,{'driver_sha256':a.sha(HERE/'snapshot-acquire-and-assess-v2.py'),'test_sha256':a.sha(__file__),'success':True,'tests_run':2,'real_socket_attempts':a.REAL_SOCKET_ATTEMPTS,'results':records,'synthetic_only':True})
+a.write(out,{'driver_sha256':a.sha(a.DRIVER),'test_sha256':a.sha(__file__),'success':True,'tests_run':2,'real_socket_attempts':a.REAL_SOCKET_ATTEMPTS,'results':records,'synthetic_only':True})
 print(json.dumps(records,indent=2))
```
