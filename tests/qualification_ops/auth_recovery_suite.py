"""Actual Google Auth + writer + parallel reader, at an offline HTTP boundary.

Synthetic signing keys are generated in memory and never retained. Real socket
connections are trapped in this process and all synthetic child workers.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, redirect_stdout
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import threading
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlparse, parse_qs

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from paths import SOURCE, DRIVER, LEGACY, verify_application
verify_application()
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('prior_validation',HERE/'functional_suite.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from app_core import evidence_config, evidence_drive, prospective_evidence as evidence, prospective_remote as codec
from google.auth import _helpers
from requests.adapters import HTTPAdapter
import requests
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

REAL_SOCKET_ATTEMPTS=0
def deny_network(event,arguments):
    global REAL_SOCKET_ATTEMPTS
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        REAL_SOCKET_ATTEMPTS+=1
        raise RuntimeError('OFFLINE_NETWORK_DENIED')
sys.addaudithook(deny_network)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')

class Clock:
    def __init__(self): self.seconds=0.;self.lock=threading.RLock()
    def advance(self,value):
        with self.lock: self.seconds+=value
    def monotonic(self):
        with self.lock: return self.seconds
    def utcnow(self): return datetime(2026,9,30,15,25)+timedelta(seconds=self.monotonic())


class AuthWire(prior.FakeWire):
    def __init__(self,files,wire,clock=None):
        super().__init__(files,wire)
        self.page_size=1000;self.clock=clock or Clock();self.oauth_posts=0
        self.oauth_failures=[];self.media_seconds=0.;self.unauthorized_barrier=None;self.unauthorized_left=0
        self.actual_session_constructions=0;self.token_lifetime=3600;self.interrupt_media=None
        self.media_seen=0;self.expire_next_media=False
        self.repeat_issuer_token=False
    def synthetic_info(self):
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        return {'type':'service_account','project_id':'offline-only','private_key_id':'SYNTHETIC','private_key':key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()).decode(),'client_email':'fixture-principal@example.invalid','client_id':'100000000','token_uri':'https://oauth2.googleapis.com/token'}
    def send(self,adapter,request,**kwargs):
        parsed=urlparse(request.url);query=parse_qs(parsed.query)
        if parsed.netloc=='oauth2.googleapis.com':
            with self.lock:
                self.oauth_posts+=1
                self.calls.append({'method':request.method,'host':parsed.netloc,'path':parsed.path,'media':False})
                fault=self.oauth_failures.pop(0) if self.oauth_failures else None
                count=self.oauth_posts
            if fault:
                return self.response(request,b'{"error":"temporarily_unavailable"}',fault)
            return self.response(request,json.dumps({'access_token':'SYNTHETIC-NONSECRET-TOKEN-'+str(1 if self.repeat_issuer_token else count),'expires_in':self.token_lifetime,'token_type':'Bearer'}).encode())
        if query.get('alt')==['media']:
            barrier=None
            with self.lock:
                self.media_seen+=1
                if self.expire_next_media:
                    self.expire_next_media=False;self.clock.advance(3600)
                if self.interrupt_media is not None and self.media_seen>=self.interrupt_media:
                    raise RuntimeError('SYNTHETIC_INTERRUPTION')
                if self.unauthorized_left:
                    self.unauthorized_left-=1;barrier=self.unauthorized_barrier
            if barrier:
                barrier.wait(timeout=10)
                with self.lock: self.calls.append({'method':request.method,'host':parsed.netloc,'path':parsed.path,'media':True})
                return self.response(request,b'{"synthetic_401":true}',401)
            # Let the real executor construct four worker sessions, rather than
            # replacing that construction with a high-level mocked reader.
            time.sleep(.001)
            self.clock.advance(self.media_seconds)
        return super().send(adapter,request,**kwargs)


class RealAuthRuntime:
    def __init__(self,backend): self.backend=backend;self.stack=ExitStack()
    def __enter__(self):
        info=self.backend.synthetic_info()
        self.stack.enter_context(patch.object(evidence_config,'service_account_info',lambda:info))
        self.stack.enter_context(patch.object(HTTPAdapter,'send',lambda adapter,request,**kw:self.backend.send(adapter,request,**kw)))
        self.stack.enter_context(patch.object(_helpers,'utcnow',self.backend.clock.utcnow))
        self.stack.enter_context(patch.dict(os.environ,{'PARLAYPICKER_DRIVE_FOLDER_ID':'fixture-folder','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT':'SYNTHETIC_ONLY'}))
        for module,names in [(codec,['sync']),(evidence,['register_model','register_calibration','freeze_validation_plan','create_validation_artifact','record_deployment_review'])]:
            for name in names:
                if hasattr(module,name): self.stack.enter_context(patch.object(module,name,prior.forbidden))
        original=evidence_drive._authorized_session
        def construct():
            self.backend.actual_session_constructions+=1
            session=original();session.trust_env=False;session._auth_request_session.trust_env=False
            return session
        self.stack.enter_context(patch.object(evidence_drive,'_authorized_session',construct))
        return self
    def __exit__(self,*args): return self.stack.__exit__(*args)


def fixture(path,count=40):
    fx=prior.Fixture(path)
    if len(fx.wire)<count:
        with closing(evidence.connect(fx.writer_db)) as db,db:
            source=db.execute('SELECT source_key FROM prospective_reconciled_source').fetchone()[0]
            for i in range(count-len(fx.wire)):
                ident=hashlib.sha256(('SYNTHETIC-'+str(i)).encode()).hexdigest()
                item={'fact_id':ident,'sport':'NFL','market_family':'SPREAD','canonical_identity':'SYNTHETIC-'+str(i),'capture_signature':ident,'source_key':source}
                evidence._insert(db,'prospective_reconciled_fact','fact_id',item,dict(item,synthetic_only=True))
            fx.wire={}
            for table,(columns,primary) in codec._schema(db).items():
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                    name,body=codec._encode(table,columns,primary,tuple(row));fx.wire[name]=body
        fx.files=[{'id':'fixture-'+str(i),'name':name,'sha256Checksum':hashlib.sha256(body).hexdigest()} for i,(name,body) in enumerate(sorted(fx.wire.items()))]
        fx.publish()
    assert len(fx.wire)==count
    return fx


RUN=None;RESULTS=[]
class Tests(unittest.TestCase):
    def setUp(self):
        self.case=RUN/self._testMethodName;self.case.mkdir()
        self.temporary=tempfile.TemporaryDirectory(prefix='ParlayPicker-offline-auth-v2-')
        self.tmp=Path(self.temporary.name).resolve()
        def retain():
            shutil.copytree(self.tmp,self.case/'retained-synthetic-fixtures')
            assert self.tmp.parent==Path(tempfile.gettempdir()).resolve() and self.tmp.name.startswith('ParlayPicker-offline-auth-v2-')
            self.temporary.cleanup()
        self.addCleanup(retain)
        self.fx=fixture(self.tmp/'synthetic',40)
        self.m=prior.load_driver(DRIVER);self.m.source_check(self.fx.spec)
        self.backend=AuthWire(self.fx.files,self.fx.wire)
        self.measure={};self.faults=[]
        self.addCleanup(lambda:write(self.case/'measurements.json',dict(self.measure,faults=self.faults,real_socket_attempts=REAL_SOCKET_ATTEMPTS,synthetic_only=True)))
        self.trace=io.StringIO();ctx=redirect_stdout(self.trace);ctx.__enter__()
        self.addCleanup(lambda:(ctx.__exit__(None,None,None),(self.case/'sanitized-driver-trace.log').write_text(self.trace.getvalue(),encoding='utf-8')))
    def fresh(self,count=40,**changes):
        if count!=40:
            self.fx=fixture(self.tmp/('synthetic-'+str(count)),count)
            self.backend=AuthWire(self.fx.files,self.fx.wire)
        self.fx.spec.update(changes);self.fx.save_spec()
        self.fx.initialize(self.m)
    def meter(self):
        return self.m.TransportBudget(self.fx.root,self.fx.spec,self.m.load_state(self.fx.root,self.fx.spec)['transport'])
    def report(self,n=1): return self.m.verified_record(self.fx.root/f'slice-{n:02d}-result.json')
    def fails(self,fn,reason):
        with self.assertRaises(Exception) as caught: fn()
        self.assertIn(reason,str(caught.exception));self.faults.append({'exception':type(caught.exception).__name__,'reason':reason,'actual_oauth_posts':self.backend.oauth_posts})
    def actual_capture(self,n=1):
        with RealAuthRuntime(self.backend): self.m.capture(self.fx.spec,n)
    def failed_fixture(self):
        self.fresh();self.fx.spec['max_new_objects_per_slice']=32;self.fx.save_spec()
        # Reinitialize the synthetic state binding to the changed fixture spec.
        state=self.m.verified_record(self.fx.root/'state.json');state['spec_sha256']=sha(self.fx.path);self.m.persist_state(self.fx.root,state)
        self.actual_capture()
        state=self.m.load_state(self.fx.root,self.fx.spec)
        # Last accepted-state usage deliberately lags durable interrupted work.
        state['transport']={'drive_get_attempts':66,'oauth_attempts':17,'observed_body_bytes':9919085}
        # Replace only SYNTHETIC journal with a valid historical usage chain.
        for name in ['transport-events.jsonl','transport-current.json']: (self.fx.root/name).unlink()
        meter=self.m.TransportBudget(self.fx.root,self.fx.spec,dict.fromkeys(self.m.TransportBudget.keys,0))
        with meter.lock:
            for k,v in state['transport'].items(): meter.increment(k,v)
        self.m.persist_state(self.fx.root,state)
        with meter.lock:
            meter.increment('drive_get_attempts',3);meter.increment('oauth_attempts',3);meter.increment('observed_body_bytes',10035)
        state['next_slice']=1;self.m.persist_state(self.fx.root,state)
        # Historical failed slice has no successful terminal slice result.
        (self.fx.root/'slice-01-result.json').unlink()
        inventory=self.m.verified_record(self.fx.root/'slice-01-inventory.json')
        completed=self.m.accepted_objects(self.fx.root,state,self.fx.spec,self.m.anchor(self.fx.spec)[2])
        pending=sorted(set(self.fx.wire)-set(completed))
        for name in pending[:3]:
            body=self.fx.wire[name];(self.fx.root/'raw'/hashlib.sha256(body).hexdigest()).write_bytes(body)
        self.m.seal(self.fx.root,'operation-blocked.json',{'status':'BLOCKED','reason':'WORKER_BLOCKED_NO_RETRY','elapsed_seconds':54.171})
        (self.fx.root/'capture-1.log').write_text('SYNTHETIC OAUTH_REQUEST_LIMIT\n')
        self.backend.oauth_posts=0;self.backend.calls=[];self.backend.media_seen=0
        return inventory
    def history(self): return self.m.inspect_predecessor(self.fx.root,self.fx.path,self.fx.spec)
    def hashes(self): return {str(p.relative_to(self.fx.root)):sha(p) for p in self.fx.root.rglob('*') if p.is_file()}

    def test_A02_old_auth_churn_actual_reader(self):
        self.fresh();self.m=prior.load_driver(LEGACY);self.m.source_check(self.fx.spec)
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=sha(self.fx.path))
        with RealAuthRuntime(self.backend): self.fails(lambda:self.m.capture(self.fx.spec,1),'OAUTH_REQUEST_LIMIT')
        state=self.m.load_state(self.fx.root,self.fx.spec);counts=[self.m.check_seal(self.fx.root,ref)['transport']['oauth_attempts'] for ref in state['accepted_batches']]
        self.assertEqual(counts,[5,9,13,17]);self.assertEqual(self.backend.oauth_posts,20)
        self.assertEqual(len(self.m.accepted_objects(self.fx.root,state,self.fx.spec,self.m.anchor(self.fx.spec)[2])),32)
        self.measure.update(batch_oauth_totals=counts,actual_oauth_posts=20,actual_session_constructions=self.backend.actual_session_constructions,stopped_batch=5)

    def test_A03_5000_actual_objects_625_batches(self):
        self.fresh(5000);self.actual_capture()
        r=self.report();self.assertEqual(r['cumulative_objects'],5000);self.assertEqual(self.backend.oauth_posts,1)
        self.assertEqual(len(self.m.load_state(self.fx.root,self.fx.spec)['accepted_batches']),625)
        self.assertEqual(r['transport']['drive_get_attempts'],5006)
        self.measure.update(objects=5000,batches=625,actual_oauth_posts=self.backend.oauth_posts,actual_drive_gets=r['transport']['drive_get_attempts'],credential_constructions=self.backend.actual_session_constructions)

    def child_supervisor(self,fx=None):
        fx=fx or self.fx;fx.backend_file();self.m.args=SimpleNamespace(spec=fx.path,approved_spec_sha256=sha(fx.path))
        original=subprocess.Popen
        def child(command,**kwargs):
            if '--worker' not in command: return original(command,**kwargs)
            rewritten=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--offline-child',str(fx.base/'fake-wire.json'),*command[3:]]
            return original(rewritten,**kwargs)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(fx.base/'local-app-data'),'PARLAYPICKER_DRIVE_FOLDER_ID':'fixture-folder','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT':'SYNTHETIC_ONLY'}),patch.object(subprocess,'Popen',child):
            code=self.m.supervise(fx.spec)
        return code

    def test_A04_actual_supervisor_eight_slices_child(self):
        self.fx.spec['max_new_objects_per_slice']=5;self.fx.save_spec()
        code=self.child_supervisor();self.assertEqual(code,0)
        slices=[self.report(i) for i in range(1,9)]
        self.assertEqual([r['cumulative_objects'] for r in slices],[5,10,15,20,25,30,35,40])
        self.assertEqual([r['transport']['oauth_attempts'] for r in slices],[1]*8)
        metrics=json.loads((self.fx.base/'fake-wire-child-capture-block.json').read_bytes())
        self.assertEqual(metrics['actual_oauth_posts'],1);self.assertEqual(metrics['real_socket_attempts'],0)
        self.measure.update(slice_oauth_totals=[r['transport']['oauth_attempts'] for r in slices],worker_metrics=metrics,supervisor_exit=code)

    def test_A05_27580_objects_virtual_full_duration(self):
        # Lower SYNTHETIC object cap forces eight boundaries. The real unchanged
        # cap is proven separately by the 5,000-object test. Advance Google Auth
        # expiry time independently of physical disk-poll/processing time: using
        # virtual seconds for disk polling caused an irrelevant scan per batch.
        self.fresh(27580,max_new_objects_per_slice=3448);self.backend.media_seconds=.75
        clock=self.backend.clock
        actual_verified=self.m.verified_record
        charged=set()
        def verify(path):
            value=actual_verified(path)
            key=str(path)
            if key.endswith('-result.json') and 'slice-' in key and key not in charged:
                charged.add(key);clock.advance(300)
            return value
        with RealAuthRuntime(self.backend),patch.object(self.m,'verified_record',verify):
            self.m.capture_block(self.fx.spec)
        result=actual_verified(self.fx.root/'capture-block-result.json')
        self.assertEqual(result['status'],'RAW_COMPLETE');self.assertEqual(result['slices_attempted'],8)
        self.assertLessEqual(self.backend.oauth_posts,20);self.assertGreater(self.backend.oauth_posts,1)
        self.assertGreater(clock.seconds,23000)
        self.measure.update(objects=27580,simulated_auth_timeline_seconds=clock.seconds,actual_oauth_posts=self.backend.oauth_posts,slice_oauth_totals=[actual_verified(self.fx.root/f'slice-{i:02d}-result.json')['transport']['oauth_attempts'] for i in range(1,9)],actual_drive_gets=result['transport']['drive_get_attempts'],clock_assumption='Auth clock:0.75s/object +300s per boundary;3600s tokens;physical processing/disk poll clock unchanged;synthetic cap3448 forces8 slices',production_throughput_claim=False)

    def sessions(self):
        self.fresh();meter=self.meter();ctx=self.m.CaptureAuthState()
        factory=self.m.guarded_session_factory(self.fx.spec,meter,{x['id'] for x in self.fx.files},evidence_drive._authorized_session,ctx)
        sessions=[factory() for _ in range(4)]
        self.addCleanup(lambda:[s.close() for s in sessions])
        return meter,ctx,sessions

    def test_A06_concurrent_expiry_and_401(self):
        with RealAuthRuntime(self.backend):
            meter,ctx,sessions=self.sessions()
            sessions[0].get('https://www.googleapis.com/drive/v3/files/fixture-folder')
            self.backend.clock.advance(3600)
            urls=['https://www.googleapis.com/drive/v3/files/'+x['id']+'?alt=media' for x in self.fx.files[:4]]
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda pair:pair[0].get(pair[1]).raise_for_status(),zip(sessions,urls)))
            self.assertEqual(self.backend.oauth_posts,2)
            self.backend.unauthorized_barrier=threading.Barrier(4);self.backend.unauthorized_left=4
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda pair:pair[0].get(pair[1]).raise_for_status(),zip(sessions,urls)))
            self.assertEqual(self.backend.oauth_posts,3);self.assertEqual(meter['drive_get_attempts'],13)
            self.measure.update(expiry_oauth_posts=2,after_concurrent_401_oauth_posts=3,drive_get_attempts=meter['drive_get_attempts'])

    def test_A06_refresh_retries_fail_closed(self):
        with RealAuthRuntime(self.backend),patch('google.auth._exponential_backoff.time.sleep',lambda n:None):
            meter,ctx,sessions=self.sessions();self.backend.oauth_failures=[503,503]
            sessions[0].get('https://www.googleapis.com/drive/v3/files/fixture-folder').raise_for_status()
            self.assertEqual(self.backend.oauth_posts,3);self.assertEqual(meter['oauth_attempts'],3)
            self.backend.clock.advance(3600);self.backend.oauth_failures=[503]*10
            self.fails(lambda:sessions[0].get('https://www.googleapis.com/drive/v3/files/fixture-folder'),'temporarily_unavailable')
            self.assertEqual(self.backend.oauth_posts,6);self.assertEqual(meter['drive_get_attempts'],1)
            self.measure.update(actual_oauth_posts=6,drive_get_attempts=1,no_get_after_failed_refresh=True)

    def test_A06_concurrent_refresh_failure_counts_all_retries(self):
        with RealAuthRuntime(self.backend),patch('google.auth._exponential_backoff.time.sleep',lambda n:None):
            meter,ctx,sessions=self.sessions();self.backend.oauth_failures=[503]*30
            def attempt(session):
                try:session.get('https://www.googleapis.com/drive/v3/files/fixture-folder');return 'UNEXPECTED_SUCCESS'
                except Exception:return 'OAUTH_REFRESH_FAILED'
            with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(attempt,sessions))
            self.assertEqual(outcomes,['OAUTH_REFRESH_FAILED']*4)
            self.assertEqual(meter['oauth_attempts'],12);self.assertEqual(self.backend.oauth_posts,12);self.assertEqual(meter['drive_get_attempts'],0)
            self.measure.update(actual_oauth_posts=12,drive_gets=0,failed_callers=4,retries_per_refresh=3)

    def test_A06_same_token_text_concurrent_401_generation(self):
        self.backend.repeat_issuer_token=True
        with RealAuthRuntime(self.backend):
            meter,ctx,sessions=self.sessions()
            sessions[0].get('https://www.googleapis.com/drive/v3/files/fixture-folder')
            self.backend.clock.advance(3600)
            urls=['https://www.googleapis.com/drive/v3/files/'+x['id']+'?alt=media' for x in self.fx.files[:4]]
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda pair:pair[0].get(pair[1]).raise_for_status(),zip(sessions,urls)))
            self.assertEqual(self.backend.oauth_posts,2)
            self.backend.unauthorized_barrier=threading.Barrier(4);self.backend.unauthorized_left=4
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda pair:pair[0].get(pair[1]).raise_for_status(),zip(sessions,urls)))
            self.assertEqual(self.backend.oauth_posts,3)
            self.assertEqual(ctx.credentials.generation,3)
            self.measure.update(actual_oauth_posts=3,refresh_generation=3,issuer_token_text_unchanged=True,concurrent_401_requests=4)

    def test_A07_nineteen_concurrent_limit(self):
        with RealAuthRuntime(self.backend):
            meter,ctx,sessions=self.sessions()
            with meter.lock: meter.increment('oauth_attempts',19)
            # Four independent real credentials force competing refresh requests
            # so the transport's atomic cap, rather than reuse, is tested.
            independent=[self.m.guarded_session_factory(self.fx.spec,meter,set(),evidence_drive._authorized_session)() for _ in range(4)]
            def request(session):
                try: session.get('https://www.googleapis.com/drive/v3/files/fixture-folder');return 'OK'
                except Exception as exc: return str(exc)
            with ThreadPoolExecutor(max_workers=4) as pool: reasons=list(pool.map(request,independent))
            for s in independent:s.close()
            self.assertEqual(self.backend.oauth_posts,1);self.assertEqual(meter['oauth_attempts'],20)
            self.assertEqual(reasons.count('OAUTH_REQUEST_LIMIT'),3)
            self.measure.update(historical_oauth=19,actual_new_oauth_posts=1,combined_oauth=20,denied_before_wire=3)

    def test_A08_A09_recovery_keeps_usage_and_orphans(self):
        self.failed_fixture();before=self.hashes();h=self.history()
        self.assertEqual(h['accepted_count'],32);self.assertEqual(len(h['unledgered_cache']),3)
        self.assertEqual(h['incurred_transport'],{'drive_get_attempts':69,'oauth_attempts':20,'observed_body_bytes':9929120})
        self.fails(lambda:self.m.recovery_policy(self.fx.spec,h,0,7),'BLOCKED_AUTHORIZATION_EXHAUSTED')
        self.assertEqual(self.backend.oauth_posts,0)
        proposed=self.m.recovery_policy(self.fx.spec,h,20,7)
        self.assertEqual(proposed['max_oauth_token_attempts'],40)
        proposed['destination']=str(self.tmp/'SYNTHETIC-SUCCESSOR');proposed['max_new_objects_per_slice']=5000
        recovery={'predecessor_destination':str(self.fx.root),'original_driver_sha256':sha(LEGACY),'additional_oauth_allowance':20}
        self.m.args.recovery_spec=self.case/'synthetic-addendum.json';write(self.m.args.recovery_spec,{'synthetic_only':True,'additional_oauth':20})
        self.m.initialize_successor(proposed,recovery,h,'SYNTHETIC-WORKER-TOKEN')
        successor=Path(proposed['destination']);self.assertEqual(len(list((successor/'raw').iterdir())),32)
        self.assertEqual(len(list((successor/'predecessor-evidence'/'raw').iterdir())),35)
        with RealAuthRuntime(self.backend):self.m.capture(proposed,1)
        result=self.m.verified_record(successor/'slice-01-result.json')
        self.assertEqual(result['accepted_reused_logical_objects'],32);self.assertEqual(result['newly_retained_logical_objects'],8)
        self.assertEqual(result['transport']['oauth_attempts'],21)
        self.assertEqual(result['transport']['drive_get_attempts'],79)
        self.assertEqual(self.backend.media_seen,8)
        self.assertEqual(before,self.hashes())
        self.measure.update(accepted=32,unledgered=3,historical=h['incurred_transport'],new_oauth=1,combined_oauth=21,combined_gets=79,new_downloads=8,orphan_admission='REDOWNLOAD_AND_NEW_LEDGER',prior_bytes_unchanged=True)

    def test_A10_recovery_faults_no_requests(self):
        self.failed_fixture();before=self.hashes()
        # Modify copies of the synthetic predecessor, never the live incident.
        for fault in ('missing-journal','bad-mirror','bad-commit','bad-cache','bad-source','bad-scope','bad-membership','interrupted-journal'):
            copied=self.tmp/fault;shutil.copytree(self.fx.root,copied)
            reason=None
            if fault=='missing-journal':(copied/'transport-events.jsonl').unlink();reason='RECOVERY_DURABLE_TRANSPORT_MISSING'
            elif fault=='bad-mirror':(copied/'transport-current.json').write_text('{}');reason='SEALED_RECORD_DIGEST_CONFLICT'
            elif fault=='bad-commit':(copied/'state-commit-000001.json').write_text('{}');reason='SEALED_RECORD_DIGEST_CONFLICT'
            elif fault=='bad-cache':next((copied/'raw').iterdir()).write_bytes(b'corrupt');reason='CACHE'
            elif fault=='interrupted-journal':
                with (copied/'transport-events.jsonl').open('ab') as f:f.write(b'{')
                reason='TRANSPORT_JOURNAL_INVALID'
            else:
                state=self.m.verified_record(copied/'state.json');key={'bad-source':'source_revision','bad-scope':'storage_scope_hash','bad-membership':'canonical_membership_sha256'}[fault];state[key]='ALTERED'
                self.m.persist_state(copied,state);reason='RAW_STATE_BINDING_CONFLICT'
            self.fails(lambda:self.m.inspect_predecessor(copied,self.fx.path,self.fx.spec),reason)
        self.assertEqual(before,self.hashes());self.assertEqual(self.backend.oauth_posts,0)

    def test_A09_actual_linked_recovery_supervisor(self):
        self.failed_fixture();before=self.hashes();h=self.history()
        successor=self.fx.root.with_name('acquire-'+str(self.fx.spec['anchor_run_id'])+'-recovery-v2')
        expected={key:h[key] for key in ('original_operation_id','accepted_count','incurred_transport','state_sha256','journal_sha256','journal_tail_sha256')}
        expected['file_register_canonical_sha256']=hashlib.sha256(self.m.canonical(h['files'])).hexdigest()
        addendum={'original_driver_path':str(LEGACY),'original_driver_sha256':sha(LEGACY),'replacement_driver_sha256':sha(DRIVER),'original_spec_sha256':sha(self.fx.path),'predecessor_destination':str(self.fx.root),'successor_destination':str(successor),'expected_predecessor':expected,'additional_oauth_allowance':20,'successor_max_slices':7,'historical_elapsed_seconds':54.171,'status':'SYNTHETIC_APPROVAL_ONLY'}
        addendum_path=self.case/'SYNTHETIC-recovery-spec.json';write(addendum_path,addendum)
        self.fx.backend_file()
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=sha(self.fx.path),recovery_spec=addendum_path,approved_recovery_sha256=sha(addendum_path),approved_driver_sha256=sha(DRIVER))
        original=subprocess.Popen
        def child(command,**kwargs):
            if '--worker' not in command:return original(command,**kwargs)
            return original([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--offline-child',str(self.fx.base/'fake-wire.json'),*command[3:]],**kwargs)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data'),'PARLAYPICKER_DRIVE_FOLDER_ID':'fixture-folder','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT':'SYNTHETIC_ONLY'}),patch.object(subprocess,'Popen',child):code=self.m.supervise(self.fx.spec)
        self.assertEqual(code,0);self.assertEqual(before,self.hashes())
        assessment=self.m.verified_record(successor/'local-assessment.json')
        result=self.m.verified_record(successor/'slice-01-result.json')
        self.assertEqual(result['transport']['oauth_attempts'],21);self.assertEqual(result['accepted_reused_logical_objects'],32)
        self.assertEqual(assessment['snapshot_sha256_before'],assessment['snapshot_sha256_after'])
        self.measure.update(supervisor_exit=code,accepted_reused=32,new_objects=8,combined_oauth=21,predecessor_unchanged=True,synthetic_snapshot_accepted=True,synthetic_assessment_ran=True)

    def test_A11_incremental_limit_and_denied_endpoint(self):
        self.failed_fixture();h=self.history();effective=self.m.recovery_policy(self.fx.spec,h,1,7)
        meter=self.m.TransportBudget(self.fx.root,effective,h['incurred_transport'])
        with RealAuthRuntime(self.backend):
            session=self.m.guarded_session_factory(effective,meter,set(),evidence_drive._authorized_session,self.m.CaptureAuthState())()
            try:
                session.get('https://www.googleapis.com/drive/v3/files/fixture-folder').raise_for_status()
                for method,url,reason in [('POST','https://www.googleapis.com/drive/v3/files/fixture-folder','REMOTE_WRITE_OR_ENDPOINT_DENIED'),('GET','https://www.googleapis.com/drive/v3/files/unpinned?alt=media','UNPINNED_MEDIA_DENIED')]:
                    self.fails(lambda:session.request(method,url),reason)
                self.backend.clock.advance(3600)
                self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/fixture-folder'),'OAUTH_REQUEST_LIMIT')
                self.assertEqual(meter['oauth_attempts'],21);self.assertEqual(self.backend.oauth_posts,1)
            finally:session.close()
        self.measure.update(incremental_oauth_limit=1,combined_limit=21,actual_new_oauth=1)

    def test_A11_startup_time_charged_before_capture_worker(self):
        self.fresh();self.fx.spec['capture_total_wall_seconds']=1
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:3.,sleep=lambda n:None)),patch.object(self.m.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('UNAUTHORIZED_WORKER_AFTER_BUDGET'))):
            code=self.m.run_approved_workers(self.fx.spec,self.fx.root,'SYNTHETIC-WORKER',0.)
        self.assertEqual(code,2);self.assertEqual(self.backend.oauth_posts,0)
        self.assertEqual(self.m.verified_record(self.fx.root/'operation-result.json')['status'],'PARTIAL')
        self.measure.update(startup_seconds_charged=3,configured_capture_wall_seconds=1,worker_launched=False,actual_oauth_posts=0)

    def test_A12_end_to_end_real_auth_empty_science(self):
        self.fx.add_football();self.backend=AuthWire(self.fx.files,self.fx.wire)
        code=self.child_supervisor();self.assertEqual(code,0)
        report=self.m.verified_record(self.fx.root/'local-assessment.json')
        for table in ('prospective_model','prospective_calibration','prospective_prediction','prospective_validation_artifact','prospective_deployment_review'): self.assertEqual(report['canonical_counts'][table],0)
        self.assertEqual(report['snapshot_sha256_before'],report['snapshot_sha256_after'])
        self.assertEqual(len(report['scopes']),12);self.assertEqual(len(report['plans']),16)
        self.assertTrue(all(p['status']=='BLOCKED_MISSING_BINDING' for p in report['plan_assessments']))
        self.assertEqual(report['eight_non_football_manifest_calculations'],'UNKNOWN_READER_UNSUPPORTED')
        self.measure.update(snapshot_accepted=True,assessment_ran=True,readiness='BLOCKED_MISSING_CANONICAL_MODEL_OR_CALIBRATION',snapshot_input_unchanged=True,supervisor_exit=code)

    def test_A15_failure_record_is_sanitized_and_retains_progress(self):
        self.fresh();self.backend.interrupt_media=35
        with RealAuthRuntime(self.backend):
            self.fails(lambda:self.m.capture(self.fx.spec,1),'SYNTHETIC_INTERRUPTION')
            self.m.retain_worker_failure(self.fx.root,self.fx.spec,'capture-block','SYNTHETIC_INTERRUPTION')
        report=self.m.verified_record(self.fx.root/'worker-blocked-capture-block.json')
        self.assertEqual(report['accepted_objects'],32);self.assertEqual(report['transport']['oauth_attempts'],1)
        self.assertEqual(self.m.sanitized_reason(RuntimeError('credential-value-must-not-escape')),'WORKER_RUNTIMEERROR_FAILED')
        for path in self.fx.root.rglob('*'):
            if path.is_file() and path.parent.name!='raw':
                body=path.read_bytes()
                for marker in (b'SYNTHETIC-NONSECRET-TOKEN-',b'BEGIN PRIVATE KEY',b'Authorization',b'credential-value-must-not-escape'):self.assertNotIn(marker,body)
        self.measure.update(accepted_objects=32,reason=report['reason'],actual_oauth_posts=1,sanitized_journals=True)


class Recorded(unittest.TextTestResult):
    def startTest(self,test):self.started=time.monotonic();super().startTest(test)
    def addSuccess(self,test):RESULTS.append({'test':test.id(),'status':'PASS','seconds':time.monotonic()-self.started});super().addSuccess(test)
    def addFailure(self,test,error):RESULTS.append({'test':test.id(),'status':'FAIL','detail':self._exc_info_to_string(error,test)});super().addFailure(test,error)
    def addError(self,test,error):RESULTS.append({'test':test.id(),'status':'ERROR','detail':self._exc_info_to_string(error,test)});super().addError(test,error)


if __name__=='__main__':
    if '--offline-child' in sys.argv:
        i=sys.argv.index('--offline-child');path=Path(sys.argv[i+1]);argv=sys.argv[i+2:]
        data=json.loads(path.read_bytes());assert data['synthetic_only'] is True
        backend=AuthWire(data['files'],{k:base64.b64decode(v) for k,v in data['wire'].items()})
        with RealAuthRuntime(backend):
            sys.argv=argv
            try:runpy.run_path(argv[0],run_name='__main__')
            finally:write(path.with_name(path.stem+'-child-'+argv[argv.index('--worker')+1]+'.json'),{'actual_oauth_posts':backend.oauth_posts,'actual_drive_gets':sum(c['method']=='GET' for c in backend.calls),'credential_constructions':backend.actual_session_constructions,'real_socket_attempts':REAL_SOCKET_ATTEMPTS,'synthetic_only':True})
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--run-directory',type=Path,required=True);parser.add_argument('--select');parser.add_argument('--exclude');a=parser.parse_args()
        RUN=a.run_directory.resolve();RUN.mkdir(parents=True,exist_ok=False)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        if a.select:suite=unittest.TestSuite(t for t in suite if a.select in t.id())
        if a.exclude:suite=unittest.TestSuite(t for t in suite if a.exclude not in t.id())
        result=unittest.TextTestRunner(verbosity=2,resultclass=Recorded).run(suite)
        write(RUN/'test-results.json',{'driver_sha256':sha(DRIVER),'tests_run':result.testsRun,'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'results':RESULTS,'real_socket_attempts':REAL_SOCKET_ATTEMPTS,'dependency_versions':{n:importlib.metadata.version(n) for n in ('google-auth','requests','cryptography','pytest')},'synthetic_only':True})
        sys.exit(0 if result.wasSuccessful() else 1)
