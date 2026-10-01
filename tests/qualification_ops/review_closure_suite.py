"""Offline closure of review comments 4158597559 and 4158597568."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import mirror_recovery_suite as mirror
import auth_recovery_suite as auth
from paths import DRIVER, LEGACY
import requests
from requests.adapters import HTTPAdapter

RUN=None
RESULTS=[]


class FakeResponse:
    def __init__(self, chunks=(b"x",), body_error=None, close_error=None):
        self.chunks=chunks
        self.body_error=body_error
        self.close_error=close_error
        self.close_calls=0
    def iter_content(self, size):
        yield from self.chunks
        if self.body_error is not None:
            raise self.body_error
    def close(self):
        self.close_calls+=1
        if self.close_error is not None:
            raise self.close_error


class Tests(unittest.TestCase):
    setUp=mirror.Tests.setUp
    fresh=auth.Tests.fresh
    meter=auth.Tests.meter
    failed_fixture=auth.Tests.failed_fixture
    history=auth.Tests.history
    actual_capture=auth.Tests.actual_capture
    fails=auth.Tests.fails

    def adapter(self, response, factory_module=None):
        self.fresh()
        meter=self.meter()
        module=factory_module or self.m
        session=requests.Session()
        token_session=requests.Session()
        session._auth_request_session=token_session
        self.addCleanup(session.close)
        self.addCleanup(token_session.close)
        with patch.dict(os.environ,{"PARLAYPICKER_DRIVE_FOLDER_ID":"fixture-folder"}):
            wrapped=module.guarded_session_factory(
                self.fx.spec,meter,{"fixture-id"},lambda:session)()
        url="https://www.googleapis.com/drive/v3/files/fixture-id?alt=media"
        adapter=wrapped.get_adapter(url)
        request=requests.Request("GET",url).prepare()
        sends=[]
        def wire(adapter, request, **kwargs):
            sends.append(request.method)
            return response
        return meter,adapter,request,wire,sends

    def exhaust_body_publication(self, calls):
        replace=Path.replace
        def operation(path,target):
            if Path(target).name=="transport-current.json":
                value=json.loads(path.read_bytes())
                if value["counters"]["observed_body_bytes"]>0:
                    calls.append("body_replace")
                    raise mirror.denied(32)
            return replace(path,target)
        return operation

    def assert_incurred_once(self, meter, sends, response):
        self.assertEqual(sends,["GET"])
        self.assertEqual(response.close_calls,1)
        self.assertEqual(dict(meter),{"drive_get_attempts":1,
                         "oauth_attempts":0,"observed_body_bytes":1})
        journal=self.m.read_transport_journal(self.fx.root,dict.fromkeys(meter.keys,0))
        self.assertEqual(journal["sequence"],2)
        self.assertEqual(journal["counters"],dict(meter))
        self.assertFalse(self.m.verified_record(self.fx.root/"state.json")["accepted_batches"])
        return journal

    def test_C02_actual_merged_adapter_restarts_publication(self):
        # This immutable, actual adapter is byte-identical to merged #2369.
        baseline=auth.prior.load_driver(mirror.OLD)
        source=mirror.OLD.read_text("utf-8")
        node=next(n for n in ast.parse(source).body
                  if isinstance(n,ast.FunctionDef) and n.name=="guarded_session_factory")
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),
             "b5b3cd7e461393388f085940c02da68918f1da1cb7c05fc0fe688f4e272b89ac")
        response=FakeResponse()
        meter,adapter,request,wire,sends=self.adapter(response,baseline)
        calls=[]
        with patch.object(HTTPAdapter,"send",wire),patch.object(Path,"replace",
                         self.exhaust_body_publication(calls)):
            with self.assertRaises(self.m.PublicationError) as caught:
                adapter.send(request)
        self.assertEqual(len(calls),16)
        self.assertNotIn("journal_sequence",caught.exception.first_error)
        context=caught.exception.__context__;seen=set();primaries=[]
        while context is not None and id(context) not in seen:
            seen.add(id(context))
            if isinstance(context,self.m.PublicationError):primaries.append(context)
            context=context.__context__
        self.assertTrue(any(error.first_error.get("journal_sequence")==2
                            for error in primaries))
        self.assert_incurred_once(meter,sends,response)
        self.measure.update(before_body_replace_attempts=len(calls),
             publication_cycles=2,http_sends=1,durable_events=2,
             primary_fields_replaced=True,response_closed=True)

    def body_failure(self, close_error=None):
        response=FakeResponse(close_error=close_error)
        meter,adapter,request,wire,sends=self.adapter(response)
        calls=[]
        with patch.object(HTTPAdapter,"send",wire),patch.object(Path,"replace",
                         self.exhaust_body_publication(calls)):
            with self.assertRaises(self.m.PublicationError) as caught:
                adapter.send(request)
        exc=caught.exception
        self.assertEqual(len(calls),8)
        journal=self.assert_incurred_once(meter,sends,response)
        self.assertEqual(exc.first_error["journal_sequence"],2)
        self.assertEqual(exc.first_error["journal_tail_sha256"],journal["tail"])
        self.assertEqual(exc.first_error["incurred_transport"],dict(meter))
        self.assertEqual(exc.first_error["local_retry_count"],7)
        prefix=self.m.validated_transport(self.fx.root,dict.fromkeys(meter.keys,0),
                                          allow_one_event_prefix=True)
        self.assertEqual(prefix["publication_gap_events"],1)
        report=self.m.retain_worker_failure(self.fx.root,self.fx.spec,"capture",
                                            self.m.sanitized_reason(exc),exc)
        self.assertEqual(report["first_error"],dict(exc.first_error,stage="capture"))
        self.assertEqual(report["transport"],dict(meter))
        self.measure.update(after_body_replace_attempts=len(calls),
             publication_cycles=1,http_sends=1,durable_events=2,
             primary_fields_preserved=True,response_closed=True)
        return exc

    def test_C03_actual_adapter_exhaustion_preserves_primary(self):
        self.body_failure()

    def test_C03_actual_adapter_primary_survives_close_error(self):
        exc=self.body_failure(OSError(5,"PRIVATE_CLOSE_TEXT_MUST_NOT_ESCAPE"))
        secondary=exc.first_error["secondary_cleanup_error"]
        self.assertEqual(secondary["exception_class"],"OSError")
        self.assertEqual(secondary["attempted_operation"],"close_response")
        self.assertNotIn("PRIVATE_CLOSE_TEXT",json.dumps(exc.first_error))
        self.measure["secondary_close_error_retained"]=True

    def test_C04_actual_adapter_success_publishes_only_charges(self):
        response=FakeResponse((b"abc",b"de"))
        meter,adapter,request,wire,sends=self.adapter(response)
        calls=[]
        replace=Path.replace
        def publication(path,target):
            if Path(target).name=="transport-current.json":calls.append(1)
            return replace(path,target)
        with patch.object(HTTPAdapter,"send",wire),patch.object(Path,"replace",publication):
            result=adapter.send(request)
        self.assertIs(result,response)
        self.assertEqual(result._content,b"abcde")
        self.assertEqual(sends,["GET"])
        self.assertEqual(response.close_calls,1)
        self.assertEqual(len(calls),3)  # GET + two body chunks, no fourth cleanup cycle.
        self.assertEqual(meter.sequence,3)
        self.assertEqual(meter["observed_body_bytes"],5)
        self.m.validated_transport(self.fx.root,dict.fromkeys(meter.keys,0))
        self.measure.update(http_sends=1,publication_cycles=3,durable_events=3,
                            response_closed=True)

    def test_C04_unrelated_body_error_is_not_replaced(self):
        primary=requests.ReadTimeout("PRIVATE_BODY_TEXT")
        response=FakeResponse(body_error=primary,
                              close_error=OSError(5,"PRIVATE_CLOSE_TEXT"))
        meter,adapter,request,wire,sends=self.adapter(response)
        with patch.object(HTTPAdapter,"send",wire):
            with self.assertRaises(requests.ReadTimeout) as caught:
                adapter.send(request)
        self.assertIs(caught.exception,primary)
        self.assert_incurred_once(meter,sends,response)
        self.assertEqual(primary.first_error["secondary_cleanup_error"]["errno"],5)
        self.assertNotIn("PRIVATE_",json.dumps(primary.first_error))
        self.m.validated_transport(self.fx.root,dict.fromkeys(meter.keys,0))
        self.measure.update(http_sends=1,response_closed=True,
                            unrelated_primary_preserved=True)

    def test_C04_close_error_without_primary_is_reported(self):
        cleanup=OSError(5,"PRIVATE_CLOSE_TEXT")
        response=FakeResponse(close_error=cleanup)
        meter,adapter,request,wire,sends=self.adapter(response)
        with patch.object(HTTPAdapter,"send",wire):
            with self.assertRaises(OSError) as caught:
                adapter.send(request)
        self.assertIs(caught.exception,cleanup)
        self.assertEqual(cleanup.first_error["attempted_operation"],"close_response")
        self.assertEqual(cleanup.first_error["errno"],5)
        self.assertNotIn("PRIVATE_",json.dumps(cleanup.first_error))
        self.assert_incurred_once(meter,sends,response)
        self.m.validated_transport(self.fx.root,dict.fromkeys(meter.keys,0))
        self.measure.update(http_sends=1,response_closed=True,close_error_not_swallowed=True)

    def synthetic_chain(self):
        self.fx=auth.fixture(self.tmp/"lineage",160)
        self.backend=auth.AuthWire(self.fx.files,self.fx.wire)
        self.failed_fixture()
        h1=self.history()
        v2root=self.fx.root.with_name("acquire-"+str(self.fx.spec["anchor_run_id"])+"-recovery-v2")
        expected={k:h1[k] for k in ("original_operation_id","accepted_count","incurred_transport",
                      "state_sha256","journal_sha256","journal_tail_sha256")}
        expected["file_register_canonical_sha256"]=hashlib.sha256(
            self.m.canonical(h1["files"])).hexdigest()
        v2={"original_driver_path":str(LEGACY),"original_driver_sha256":auth.sha(LEGACY),
            "replacement_driver_sha256":auth.sha(mirror.OLD),
            "original_spec_sha256":auth.sha(self.fx.path),
            "predecessor_destination":str(self.fx.root),"successor_destination":str(v2root),
            "expected_predecessor":expected,"additional_oauth_allowance":20,
            "successor_max_slices":7,"historical_elapsed_seconds":54.171}
        v2path=self.tmp/"v2-synthetic.json";auth.write(v2path,v2)
        effective=self.m.recovery_policy(self.fx.spec,h1,20,7)
        effective.update(destination=str(v2root),recovery_mode=True,
             working_disk_predecessor_bytes=sum(x["bytes"] for x in h1["files"].values()))
        for key in ("capture_total_wall_seconds","total_wall_seconds"):effective[key]-=54.171
        self.m.args.recovery_spec=v2path
        self.m.initialize_successor(effective,v2,h1,"SYNTHETIC-WORKER")
        link=self.m.verified_record(v2root/"recovery-link.json")
        link["replacement_driver_sha256"]=auth.sha(mirror.OLD)
        link["canonical_sha256"]=hashlib.sha256(self.m.canonical(
            {k:v for k,v in link.items() if k!="canonical_sha256"})).hexdigest()
        auth.write(v2root/"recovery-link.json",link)
        self.m.save(v2root/"effective-spec.json",effective)
        state=self.m.verified_record(v2root/"state.json")
        state["spec_sha256"]=auth.sha(v2root/"effective-spec.json")
        self.m.persist_state(v2root,state)
        self.m.args=SimpleNamespace(spec=v2root/"effective-spec.json",
             approved_spec_sha256=auth.sha(v2root/"effective-spec.json"))
        with auth.RealAuthRuntime(self.backend):
            for number in (1,2,3):self.m.capture(effective,number)
        state=self.m.verified_record(v2root/"state.json")
        self.m.seal(v2root,"slice-04-attempt.json",
             {"status":"ATTEMPT_STARTED_NO_AUTOMATIC_RETRY",
              "source_revision":effective["source_revision"],
              "storage_scope_hash":effective["storage_scope_hash"]})
        meter=self.m.TransportBudget(v2root,effective,state["transport"])
        for key,value in [("drive_get_attempts",15204),("oauth_attempts",22),
                          ("observed_body_bytes",140811320)]:
            meter.increment(key,value-meter[key])
        with patch.object(Path,"replace",side_effect=mirror.denied(5)):
            with self.assertRaises(self.m.PublicationError):
                meter.increment("observed_body_bytes",1)
        self.m.seal(v2root,"operation-blocked.json",
             {"status":"BLOCKED","reason":"WORKER_BLOCKED_NO_RETRY","elapsed_seconds":3641.547})
        recovery={"recovery_generation":3,"additional_oauth_allowance":0,
            "original_driver_sha256":auth.sha(LEGACY),
            "original_spec_sha256":auth.sha(self.fx.path),
            "v2_driver_path":str(mirror.OLD),"v2_recovery_spec_path":str(v2path),
            "v2_recovery_spec_sha256":auth.sha(v2path),"predecessor_destination":str(v2root),
            "successor_destination":str(v2root.with_name(
                       "acquire-"+str(self.fx.spec["anchor_run_id"])+"-recovery-v3")),
            "historical_elapsed_seconds":3695.718,"successor_max_slices":3}
        self.m.args.spec=self.fx.path
        h2=self.m.inspect_linked_predecessor(self.fx.spec,recovery)
        recovery["expected_predecessor"]={k:h2[k] for k in ("accepted_count","incurred_transport",
            "state_sha256","journal_sha256","journal_tail_sha256","effective_spec_sha256",
            "recovery_link_sha256","blocked_record_sha256","file_register_canonical_sha256",
            "attempts_used")}
        recovery["replacement_driver_sha256"]=auth.sha(DRIVER)
        return recovery

    def test_C05_C06_outer_hash_approved_lineage_and_clean_rejections(self):
        recovery=self.synthetic_chain()
        before={label:{str(p.relative_to(root)):auth.sha(p)
                       for p in root.rglob("*") if p.is_file()}
                for label,root in [("v1",self.fx.root),
                  ("v2",Path(recovery["predecessor_destination"]))]}
        path=self.tmp/"v3-synthetic.json"
        destination=Path(recovery["successor_destination"])
        http_before=len(self.backend.calls)
        variants=[("mismatch","0"*64,"RECOVERY_ORIGINAL_DRIVER_ANCESTRY_CONFLICT"),
                  ("missing",None,"RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID"),
                  ("null",None,"RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID"),
                  ("short","0"*63,"RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID"),
                  ("nonhex","Z"*64,"RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID"),
                  ("wrong_type",True,"RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID")]
        def approve(value):
            auth.write(path,value)
            self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=auth.sha(self.fx.path),
                recovery_spec=path,approved_recovery_sha256=auth.sha(path),
                approved_driver_sha256=auth.sha(DRIVER))
            self.assertEqual(self.m.args.approved_recovery_sha256,auth.sha(path))
        with patch.dict(os.environ,{"LOCALAPPDATA":str(self.fx.base/"local-app-data"),
             "PARLAYPICKER_DRIVE_FOLDER_ID":"fixture-folder",
             "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT":"SYNTHETIC_ONLY"}):
            for label,value,reason in variants:
                with self.subTest(label=label):
                    bad=dict(recovery)
                    if label=="missing":bad.pop("original_driver_sha256")
                    else:bad["original_driver_sha256"]=value
                    approve(bad)  # Outer approval is valid, not the cause of rejection.
                    with patch.object(self.m,"initialize_successor",wraps=self.m.initialize_successor) as creation,patch.object(
                          self.m,"run_approved_workers",side_effect=AssertionError("NO_WORKER_ALLOWED")):
                        with self.assertRaisesRegex(RuntimeError,"^"+reason+"$"):
                            self.m.supervise(self.fx.spec)
                        creation.assert_not_called()
                    self.assertFalse(destination.exists())
                    self.assertEqual(len(self.backend.calls),http_before)
            approve(recovery)
            # The full capture/assembly/assessment path remains covered by existing M12.
            with patch.object(self.m,"run_approved_workers",return_value=0) as worker:
                self.assertEqual(self.m.supervise(self.fx.spec),0)
                worker.assert_called_once()
        link=self.m.verified_record(destination/"recovery-link.json")
        self.assertEqual(link["original_driver_sha256"],auth.sha(LEGACY))
        self.assertEqual(link["original_driver_sha256"],recovery["original_driver_sha256"])
        self.assertEqual(len(self.backend.calls),http_before)
        for label,root in [("v1",self.fx.root),("v2",Path(recovery["predecessor_destination"]))]:
            self.assertEqual(before[label],{str(p.relative_to(root)):auth.sha(p)
                                            for p in root.rglob("*") if p.is_file()})
        self.measure.update(approved_outer_hash_recalculated=True,early_rejections=len(variants),
             copied_bytes_on_rejection=0,worker_starts_on_rejection=0,
             new_fake_http_calls=0,valid_link_ancestral_driver_sha256=auth.sha(LEGACY),
             predecessors_unchanged=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-directory",type=Path,required=True)
    parser.add_argument("--select")
    options=parser.parse_args()
    RUN=options.run_directory.resolve();RUN.mkdir(parents=True,exist_ok=False)
    mirror.RUN=RUN;auth.RUN=RUN
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if options.select:suite=unittest.TestSuite(t for t in suite if options.select in t.id())
    mirror.RESULTS.clear()
    result=unittest.TextTestRunner(verbosity=2,resultclass=mirror.Recorded).run(suite)
    auth.write(RUN/"test-results.json",{"driver_sha256":auth.sha(DRIVER),"tests_run":result.testsRun,
         "success":result.wasSuccessful(),"failures":len(result.failures),"errors":len(result.errors),
         "results":mirror.RESULTS,"real_socket_attempts":auth.REAL_SOCKET_ATTEMPTS+auth.prior.REAL_REQUESTS,
         "synthetic_only":True})
    raise SystemExit(0 if result.wasSuccessful() else 1)
