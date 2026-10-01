"""Synthetic-only functional verification of the actual operations wrapper.

No --execute-approved-operation command is invoked. Socket connections are
denied in parent and child. The existing writer codec generates every fixture
object; production model/calibration/prediction/validation/review tables stay
empty. Nothing here is hosted/model/real-evidence proof.
"""
import argparse
import base64
from collections import defaultdict
from contextlib import ExitStack, closing, redirect_stdout
import copy
import hashlib
import gc
import importlib.util
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
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from paths import SOURCE, DRIVER as RUNTIME_DRIVER, LEGACY, TEMPLATE, verify_application
# Preserve the exact prior v2 hash assertion; execute runtime cases on the repair.
TRACKED_DRIVER = HERE / 'fixtures/previous_oauth_snapshot_acquire.py'
verify_application()
sys.path.insert(0,str(SOURCE))
from app_core import prospective_remote as codec, prospective_evidence as evidence
from app_core import evidence_drive, read_only_census as census, prospective_validation_plans as policy
from app_core.prospective_reconciliation import ensure_reconciliation_schema
from app_core.performance_spans import opaque_hash
import requests
from requests.adapters import HTTPAdapter

REAL_REQUESTS=0
def deny_socket(event, arguments):
    global REAL_REQUESTS
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        REAL_REQUESTS+=1
        raise RuntimeError('OFFLINE_SOCKET_DENIED')
sys.addaudithook(deny_socket)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def write_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def load_driver(path):
    spec=importlib.util.spec_from_file_location('tested_acquisition_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self,base,number=16):
        self.base=Path(base);self.base.mkdir(parents=True)
        self.retained=self.base/'synthetic-anchor';(self.retained/'slice-01').mkdir(parents=True)
        self.spec=copy.deepcopy(json.loads(TEMPLATE.read_bytes()))
        self.spec['source_checkout']=str(SOURCE)
        self.spec['retained_directory']=str(self.retained)
        self.spec['destination']=str(self.base/'local-app-data'/'ParlayPicker'/'qualification-canonical-snapshot'/self.spec['source_revision']/f"acquire-{self.spec['anchor_run_id']}-v1")
        self.spec['required_initial_free_disk_bytes']=1 # Scaled fixture limit; REAL SPEC IS NEVER EDITED.
        self.spec['storage_scope_hash']=opaque_hash('google_workspace_shared_drive','fixture-principal@example.invalid','fixture-drive','fixture-folder')
        self.writer_db=self.base/'SYNTHETIC-writer.sqlite3'
        ensure_reconciliation_schema(self.writer_db)
        with closing(evidence.connect(self.writer_db)) as db,db:
            raw=b'{"synthetic_only":true,"observation_time":"2026-09-24T02:00:00+00:00"}'
            source_key=sha(b'fixture-source')
            payload={'source_key':source_key,'sport':'NFL','source_table':'fixture','source_record_id':sha(raw),'source_hash':sha(raw)}
            evidence._insert(db,'prospective_reconciled_source','source_key',payload,payload,raw_source=raw)
            fact={'fact_id':sha(b'fixture-fact'),'sport':'NFL','market_family':'SPREAD','canonical_identity':'synthetic-only','capture_signature':sha(b'fixture-signature'),'source_key':source_key}
            evidence._insert(db,'prospective_reconciled_fact','fact_id',fact,dict(fact,synthetic_only=True,observed_at='2026-09-24T02:00:00+00:00'))
            templates=policy.plan_specs(source_commit=self.spec['source_revision'])
            from app_core.football_validation_v2 import _plan_spec
            from datetime import datetime,timezone
            templates += [_plan_spec(sport,market,source_commit=self.spec['source_revision'],validation_start=datetime(2026,9,25,tzinfo=timezone.utc)) for sport,market in [('NFL','SPREAD'),('NFL','TOTAL'),('NCAAF','SPREAD'),('NCAAF','TOTAL')]]
            for template in templates[:number]:
                data=copy.deepcopy(template)
                data['validation_plan_id']='offline-test-'+data['validation_plan_id']
                if data.get('supersedes_plan_id'): data['supersedes_plan_id']='offline-test-'+data['supersedes_plan_id']
                data['frozen_at']='2026-09-24T01:00:00+00:00'
                data['synthetic_only']=True
                artifact=sha(codec._json(data));data['artifact_hash']=artifact
                fields=['validation_plan_id','sport','market_family','version','supersedes_plan_id','model_id','calibration_id','training_cutoff','validation_start','validation_end','holdout_start','holdout_end','minimum_independent_sample','minimum_effective_sample','frozen_at','artifact_hash']
                # Low-level writer only, in test DB. No freeze/registration/fitting entrypoint.
                evidence._insert(db,'prospective_validation_plan','validation_plan_id',{key:data.get(key) for key in fields},data)
            schema=codec._schema(db)
            self.wire={}
            for table,(columns,primary) in schema.items():
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                    key,body=codec._encode(table,columns,primary,tuple(row));self.wire[key]=body
        self.files=[{'id':'fixture-'+str(i),'name':name,'sha256Checksum':sha(body)} for i,(name,body) in enumerate(sorted(self.wire.items()))]
        self.path=self.base/'fixture-spec.json'
        self.publish()

    @property
    def root(self): return Path(self.spec['destination'])
    def publish(self):
        processed={}
        for name,body in self.wire.items():
            items=[x for x in self.files if x['name']==name]
            processed[name]={'namespace':self.spec['namespace'],'metadata_token':census._metadata_token(items),'content_sha256':sha(body),'facts':census._facts(name,body,self.spec['namespace'])}
        membership=census._membership_digest({name:[x for x in self.files if x['name']==name] for name in self.wire})
        cp={'schema':census.CHECKPOINT_SCHEMA,'source_revision':self.spec['source_revision'],'storage_scope_hash':self.spec['storage_scope_hash'],'operation_id':'SYNTHETIC-OFFLINE-OPERATION','inventory_membership':{self.spec['namespace']:membership},'processed':processed,'requested_scopes':list(census.SCOPES),'updated_at':self.spec['as_of']}
        cp['checkpoint_sha256']=census._checkpoint_digest(cp)
        with closing(evidence._reader(self.writer_db)) as db:
            manifests={scope:[row[0] for row in db.execute('SELECT manifest_id FROM prospective_football_training_manifest WHERE sport=? AND market_family=? ORDER BY training_row_id',scope.split('/'))] if scope.startswith(('NFL/','NCAAF/')) else [] for scope in census.SCOPES}
        rp={'status':'COMPLETE','source_revision':self.spec['source_revision'],'scopes':[{'scope':scope,'eligibility':{'manifest_ids':manifests[scope]}} for scope in census.SCOPES]}
        write_json(self.retained/'slice-01/checkpoint.json',cp);write_json(self.retained/'slice-01/report.json',rp)
        archive=self.retained/'synthetic-archive.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.write(self.retained/'slice-01/checkpoint.json','checkpoint.json');z.write(self.retained/'slice-01/report.json','report.json')
        write_json(self.retained/'artifact-register.json',{'files':[{'path':str(archive),'sha256':sha(archive.read_bytes())}]})
        self.spec.update(anchor_archive_sha256=sha(archive.read_bytes()),anchor_checkpoint_file_sha256=sha((self.retained/'slice-01/checkpoint.json').read_bytes()),anchor_checkpoint_canonical_sha256=cp['checkpoint_sha256'],anchor_report_sha256=sha((self.retained/'slice-01/report.json').read_bytes()),canonical_membership_sha256=membership,canonical_objects=len(processed))
        self.save_spec()
    def save_spec(self): write_json(self.path,self.spec)
    def add_football(self):
        from tests.test_football_stage2 import one_game
        one_game(self.writer_db,two_books=True) # Existing synthetic writer fixture; never build_reports.
        with closing(evidence._reader(self.writer_db)) as db:
            self.wire={}
            for table,(columns,primary) in codec._schema(db).items():
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                    key,body=codec._encode(table,columns,primary,tuple(row));self.wire[key]=body
        self.files=[{'id':'fixture-'+str(i),'name':name,'sha256Checksum':sha(body)} for i,(name,body) in enumerate(sorted(self.wire.items()))]
        self.publish()
    def repin_invalid(self,name,body,renamed=None):
        # Fault-specific synthetic anchor: preserve old facts to exercise the
        # independent import verifier, rather than its census parser.
        cp=json.loads((self.retained/'slice-01/checkpoint.json').read_bytes())
        self.wire[name]=body
        for item in self.files:
            if item['name']==name: item['sha256Checksum']=sha(body)
        if renamed:
            self.wire[renamed]=self.wire.pop(name);cp['processed'][renamed]=cp['processed'].pop(name)
            for item in self.files:
                if item['name']==name: item['name']=renamed
            name=renamed
        cp['processed'][name]['content_sha256']=sha(body)
        cp['processed'][name]['metadata_token']=census._metadata_token([item for item in self.files if item['name']==name])
        cp['inventory_membership'][self.spec['namespace']]=sha(codec._json(sorted((key,value['metadata_token']) for key,value in cp['processed'].items())))
        cp['checkpoint_sha256']=census._checkpoint_digest(cp)
        write_json(self.retained/'slice-01/checkpoint.json',cp)
        archive=self.retained/'synthetic-archive.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.write(self.retained/'slice-01/checkpoint.json','checkpoint.json');z.write(self.retained/'slice-01/report.json','report.json')
        write_json(self.retained/'artifact-register.json',{'files':[{'path':str(archive),'sha256':sha(archive.read_bytes())}]})
        self.spec.update(anchor_archive_sha256=sha(archive.read_bytes()),anchor_checkpoint_file_sha256=sha((self.retained/'slice-01/checkpoint.json').read_bytes()),anchor_checkpoint_canonical_sha256=cp['checkpoint_sha256'],canonical_membership_sha256=cp['inventory_membership'][self.spec['namespace']]);self.save_spec()
    def backend_file(self):
        path=self.base/'fake-wire.json'
        write_json(path,{'files':self.files,'wire':{name:base64.b64encode(raw).decode() for name,raw in self.wire.items()},'synthetic_only':True})
        return path
    def initialize(self,m):
        self.root.mkdir(parents=True);(self.root/'raw').mkdir()
        m.args=SimpleNamespace(spec=self.path,approved_spec_sha256=m.digest(self.path))
        state={'spec_sha256':m.digest(self.path),'supervisor_pid':os.getpid(),'source_revision':self.spec['source_revision'],'storage_scope_hash':self.spec['storage_scope_hash'],'canonical_membership_sha256':self.spec['canonical_membership_sha256'],'next_slice':1,'accepted_batches':[],'transport':{'drive_get_attempts':0,'oauth_attempts':0,'observed_body_bytes':0}}
        if hasattr(m,'persist_state'): m.persist_state(self.root,state)
        else: m.save(self.root/'state.json',state)
        m.args=SimpleNamespace(spec=self.path,approved_spec_sha256=m.digest(self.path))


class FakeWire:
    def __init__(self,files,wire):
        self.files=copy.deepcopy(files);self.wire=wire;self.sessions=[];self.calls=[];self.lock=threading.RLock()
        self.failures={};self.probe=None;self.probe_done=False;self.page_size=5;self.delay=None;self.raw_factory=None
    def session(self):
        session=requests.Session()
        session.trust_env=False
        session.credentials=SimpleNamespace(service_account_email='fixture-principal@example.invalid')
        session._auth_request_session=requests.Session()
        session._auth_request_session.trust_env=False
        self.sessions.append(session)
        return session
    def response(self,request,body,status=200):
        response=requests.Response();response.status_code=status;response.url=request.url;response.request=request
        response.raw=self.raw_factory(body) if self.raw_factory else io.BytesIO(body)
        return response
    def send(self,adapter,request,**kwargs):
        parsed=urlparse(request.url);params=parse_qs(parsed.query)
        with self.lock:
            self.calls.append({'method':request.method,'host':parsed.netloc,'path':parsed.path,'media':params.get('alt')==['media']})
            if self.delay: self.delay(request)
            remaining=self.failures.get(parsed.path,[])
            if remaining:
                fault=remaining.pop(0)
                if isinstance(fault,Exception): raise fault
                return self.response(request,b'{"fixture_retry":true}',fault)
            if parsed.netloc=='oauth2.googleapis.com':
                return self.response(request,b'{"synthetic_token_only":true}')
            if parsed.path=='/drive/v3/files/fixture-folder':
                if self.probe and not self.probe_done:
                    self.probe_done=True;self.probe(self.sessions[0])
                return self.response(request,b'{"id":"fixture-folder","driveId":"fixture-drive","mimeType":"application/vnd.google-apps.folder","trashed":false}')
            if parsed.path=='/drive/v3/files' and params.get('alt')!=['media']:
                index=int(params.get('pageToken',['0'])[0]);page=self.files[index:index+self.page_size]
                data={'files':page}
                if index+self.page_size<len(self.files): data['nextPageToken']=str(index+self.page_size)
                return self.response(request,json.dumps(data).encode())
            if params.get('alt')==['media']:
                ident=parsed.path.rsplit('/',1)[-1]
                item=next((x for x in self.files if x['id']==ident),None)
                if item is None: return self.response(request,b'{"missing":true}',404)
                body=self.wire[item['name']]
                if isinstance(body,dict): body=body[item['id']]
                return self.response(request,body)
            return self.response(request,b'{"unapproved_endpoint_reached_fake_wire":true}')


def forbidden(*args,**kwargs): raise AssertionError('FORBIDDEN_SCIENTIFIC_OR_UPLOAD_ENTRYPOINT')
class OfflineRuntime:
    def __init__(self,wire): self.wire=wire;self.stack=ExitStack()
    def __enter__(self):
        self.stack.enter_context(patch.object(evidence_drive,'_authorized_session',self.wire.session))
        wire=self.wire
        self.stack.enter_context(patch.object(HTTPAdapter,'send',lambda adapter,request,**kw:wire.send(adapter,request,**kw)))
        from app_core import football_stage2,football_validation_v2
        for module,names in [(codec,['sync']),(football_stage2,['build_reports']),(evidence,['register_model','register_calibration','freeze_validation_plan','create_validation_artifact','record_deployment_review']),(policy,['freeze_plans']),(football_validation_v2,['freeze_plans'])]:
            for name in names:
                if hasattr(module,name): self.stack.enter_context(patch.object(module,name,forbidden))
        self.stack.enter_context(patch.dict(os.environ,{'PARLAYPICKER_DRIVE_FOLDER_ID':'fixture-folder','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT':'OFFLINE_SYNTHETIC_PLACEHOLDER_NOT_A_CREDENTIAL'}))
        return self
    def __exit__(self,*args): self.stack.__exit__(*args)


RUN_DIR=None;DRIVER=None
class FunctionalTests(unittest.TestCase):
    def setUp(self):
        self.case=RUN_DIR/self._testMethodName
        self.case.mkdir()
        temporary=tempfile.TemporaryDirectory(prefix='ParlayPicker-offline-validation-')
        temporary_root=Path(temporary.name).resolve()
        def retain_then_clean():
            # Synthetic fixtures only. Keep fault evidence before disposing of
            # this specific test's temporary directory; never touch owner data.
            shutil.copytree(temporary_root,self.case/'synthetic-fixture-retained')
            assert temporary_root.parent==Path(tempfile.gettempdir()).resolve() and temporary_root.name.startswith('ParlayPicker-offline-validation-')
            gc.collect()
            temporary.cleanup()
        self.addCleanup(retain_then_clean)
        self.fx=Fixture(temporary_root/'fixture')
        self.m=load_driver(DRIVER);self.m.source_check(self.fx.spec)
        self.backend=FakeWire(self.fx.files,self.fx.wire)
        self.runtime=OfflineRuntime(self.backend);self.runtime.__enter__();self.addCleanup(self.runtime.__exit__,None,None,None)
        self.trace=io.StringIO();redirect=redirect_stdout(self.trace);redirect.__enter__()
        self.addCleanup(lambda:(redirect.__exit__(None,None,None),(self.case/'synthetic-driver-trace.log').write_text(self.trace.getvalue(),encoding='utf-8')))
        self.faults=[]
        self.addCleanup(lambda:write_json(self.case/'fault-evidence.json',self.faults))
    def fails(self,call,contains=None):
        with self.assertRaises(Exception) as caught: call()
        reason=str(caught.exception)
        self.faults.append({'exception':type(caught.exception).__name__,'reason':reason,'fake_requests_after':len(self.backend.calls)})
        if contains: self.assertIn(contains,reason)
        return reason
    def configured(self,**changes):
        self.fx.spec.update(changes);self.fx.save_spec()
    def report(self,number=1): return self.m.verified_record(self.fx.root/f'slice-{number:02d}-result.json')
    def transport(self):
        self.initialized()
        meter=self.m.TransportBudget(self.fx.root,self.fx.spec,{'drive_get_attempts':0,'oauth_attempts':0,'observed_body_bytes':0})
        session=self.m.guarded_session_factory(self.fx.spec,meter,{x['id'] for x in self.fx.files},evidence_drive._authorized_session)()
        return session,meter
    def inline_supervisor(self,hook=None):
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=self.m.digest(self.fx.path))
        original=subprocess.Popen;phases=[]
        def worker(command,**kwargs):
            if '--worker' not in command: return original(command,**kwargs)
            phase=command[command.index('--worker')+1];phases.append(phase)
            proc=SimpleNamespace(returncode=0,poll=lambda:0,kill=lambda:None,wait=lambda:0)
            try:
                if hook: hook(phase,command,proc)
                elif phase=='capture-block': self.m.capture_block(self.fx.spec)
                elif phase=='materialize': self.m.materialize(self.fx.spec)
                else: self.m.assess(self.fx.spec)
            except Exception as exc:
                self.faults.append({'phase':phase,'worker_exception':str(exc)});proc.returncode=1;proc.poll=lambda:1
            return proc
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data')}),patch.object(self.m.subprocess,'Popen',worker):
            result=self.m.supervise(self.fx.spec)
        return result,phases
    def initialized(self): self.fx.initialize(self.m)
    def full_raw(self):
        self.initialized();self.m.capture(self.fx.spec,1)
    def test_V02_actual_end_to_end_supervisor(self):
        self.fx.add_football();self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire
        self.fx.backend_file();self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=self.m.digest(self.fx.path))
        original=subprocess.Popen
        def offline_child(command,**kwargs):
            if '--worker' not in command: return original(command,**kwargs)
            transformed=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--isolated-child',str(self.fx.base/'fake-wire.json'),*command[3:]]
            return original(transformed,**kwargs)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data')}),patch.object(self.m.subprocess,'Popen',offline_child):
            result=self.m.supervise(self.fx.spec)
        self.assertEqual(result,0,(self.fx.root/'operation-blocked.json').read_text() if (self.fx.root/'operation-blocked.json').exists() else 'no failure report')
        report=json.loads((self.fx.root/'local-assessment.json').read_bytes())
        self.assertEqual(report['snapshot_sha256_before'],report['snapshot_sha256_after'])
        self.assertEqual(len(report['scopes']),12)
        self.assertEqual(len(report['plans']),16)
        self.assertTrue(all(p['status']=='BLOCKED_MISSING_BINDING' for p in report['plan_assessments']))
        self.assertEqual(report['football_manifests']['NFL/SPREAD']['independent_games'],1)
        self.assertEqual(report['football_manifests']['NFL/TOTAL']['independent_games'],1)
        for table in ['prospective_model','prospective_calibration','prospective_prediction','prospective_validation_artifact','prospective_deployment_review']:
            self.assertEqual(report['canonical_counts'][table],0)
    def test_V03_http_endpoint_is_denied(self):
        self.initialized()
        reached=[]
        def probe(session):
            try: session.get('http://unapproved.invalid/escape')
            except RuntimeError: return
            reached.append(True)
        self.backend.probe=probe
        self.m.capture(self.fx.spec,1)
        self.assertFalse(reached,'HTTP scheme bypassed the guard')
    def test_V05_checkpoint_reset_rejected(self):
        self.fx.spec['max_new_objects_per_slice']=8;self.fx.save_spec();self.initialized();self.m.capture(self.fx.spec,1)
        state=json.loads((self.fx.root/'state.json').read_bytes());state['accepted_batches']=[];state['transport']={'drive_get_attempts':0,'oauth_attempts':0,'observed_body_bytes':0}
        if 'canonical_sha256' in state: state['canonical_sha256']=sha(self.m.canonical({k:v for k,v in state.items() if k!='canonical_sha256'}))
        write_json(self.fx.root/'state.json',state)
        with self.assertRaises(RuntimeError): self.m.capture(self.fx.spec,2)
    def test_V05_interruption_cannot_reset_or_retry(self):
        self.fx.spec['max_new_objects_per_slice']=8;self.fx.save_spec();self.initialized();self.m.capture(self.fx.spec,1)
        done=set()
        state=json.loads((self.fx.root/'state.json').read_bytes())
        for ref in state['accepted_batches']: done.update(self.m.check_seal(self.fx.root,ref)['objects'])
        missing=sorted(set(self.fx.wire)-done)[0];ident=next(x['id'] for x in self.backend.files if x['name']==missing)
        self.backend.failures['/drive/v3/files/'+ident]=[RuntimeError('CONTROLLED_INTERRUPT')]
        with self.assertRaises(RuntimeError): self.m.capture(self.fx.spec,2)
        before=len(self.backend.calls);meter_before=json.loads((self.fx.root/'transport-current.json').read_bytes())
        with self.assertRaises(RuntimeError): self.m.capture(self.fx.spec,2)
        self.assertEqual(len(self.backend.calls),before,'interrupted slice was re-entered and incurred attempts')
        self.assertEqual(json.loads((self.fx.root/'transport-current.json').read_bytes()),meter_before)
    def test_V10_snapshot_acceptance_digest_rejected(self):
        self.initialized();target=self.fx.root/'canonical.sqlite3';shutil.copyfile(self.fx.writer_db,target)
        # Deliberately invalid acceptance, while database bytes themselves are valid.
        self.m.seal(self.fx.root,'snapshot-acceptance.json',{'source_revision':self.fx.spec['source_revision'],'storage_scope_hash':self.fx.spec['storage_scope_hash'],'snapshot_sha256':self.m.digest(target),'canonical_membership_sha256':self.fx.spec['canonical_membership_sha256']})
        record=json.loads((self.fx.root/'snapshot-acceptance.json').read_bytes());record['canonical_sha256']='0'*64;write_json(self.fx.root/'snapshot-acceptance.json',record)
        with self.assertRaises(RuntimeError): self.m.assess(self.fx.spec)
    def test_V07_actual_writer_schema_assembly(self):
        self.full_raw()
        self.m.materialize(self.fx.spec)
        self.assertTrue((self.fx.root/'canonical.sqlite3').is_file())
    def test_V01_original_hashes_and_unchanged_envelope(self):
        self.assertEqual(sha(LEGACY.read_bytes()),'6c01b00da684956f4319017c3b7f08b78eafc5133b74de98697d69fad287de67')
        self.assertEqual(sha(TRACKED_DRIVER.read_bytes()),'8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93')
        envelope=json.loads(TEMPLATE.read_bytes())
        for key,value in {'max_slices':8,'max_new_objects_per_slice':5000,'soft_bytes_per_slice':500000000,'slice_processing_seconds':2700,'slice_wall_seconds':3000,'soft_bytes_total':4000000000,'transport_observed_body_bytes_stop':5000000000,'max_drive_get_attempts':90000,'max_oauth_token_attempts':20,'capture_total_wall_seconds':24000,'materialization_wall_seconds':900,'assessment_wall_seconds':900,'total_wall_seconds':27000,'max_working_disk_bytes':12000000000,'required_initial_free_disk_bytes':14000000000,'max_snapshot_input_bytes':2000000000}.items():
            self.assertEqual(envelope[key],value,key)
    def test_V03_all_mutations_and_unapproved_endpoints_denied(self):
        session,meter=self.transport();folder='https://www.googleapis.com/drive/v3/files/fixture-folder'
        attempts=[(method,folder) for method in ('POST','PUT','PATCH','DELETE','HEAD')]
        attempts += [('GET',url) for url in ['http://www.googleapis.com/drive/v3/files/fixture-folder','https://unapproved.invalid/drive/v3/files/fixture-folder','https://www.googleapis.com/drive/v3/files/another-folder','https://www.googleapis.com/drive/v3/files','https://www.googleapis.com/drive/v3/files?alt=media','https://www.googleapis.com/drive/v3/files/unpinned?alt=media']]
        for method,url in attempts:
            with self.subTest(method=method,url=url): self.fails(lambda:session.request(method,url))
        auth=session._auth_request_session
        for method,url in [('GET','https://oauth2.googleapis.com/token'),('POST','http://oauth2.googleapis.com/token'),('POST','https://unapproved.invalid/token'),('POST','https://oauth2.googleapis.com/other'),('POST','https://oauth2.googleapis.com/token?x=1')]:
            with self.subTest(method=method,url=url): self.fails(lambda:auth.request(method,url))
        self.assertEqual(self.backend.calls,[]);self.assertEqual(meter['drive_get_attempts'],0);self.assertEqual(meter['oauth_attempts'],0)
    def test_V03_request_and_oauth_attempt_caps(self):
        self.configured(max_drive_get_attempts=2,max_oauth_token_attempts=2)
        session,meter=self.transport()
        for _ in range(2): session.get('https://www.googleapis.com/drive/v3/files/fixture-folder')
        self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/fixture-folder'),'DRIVE_REQUEST_LIMIT')
        for _ in range(2): session._auth_request_session.post('https://oauth2.googleapis.com/token')
        self.fails(lambda:session._auth_request_session.post('https://oauth2.googleapis.com/token'),'OAUTH_REQUEST_LIMIT')
        self.assertEqual(len(self.backend.calls),4);self.assertEqual(meter['drive_get_attempts'],2);self.assertEqual(meter['oauth_attempts'],2)
    def test_V03_retry_attempts_and_response_bodies_counted(self):
        self.initialized();ident=self.fx.files[0]['id'];path='/drive/v3/files/'+ident
        self.backend.failures[path]=[429,503]
        with patch.object(evidence_drive.time,'sleep',lambda _:None): self.m.capture(self.fx.spec,1)
        calls=[x for x in self.backend.calls if x['path']==path and x['media']]
        self.assertEqual(len(calls),3);self.assertEqual(self.report()['physical_metrics']['retries'],2)
        self.assertEqual(self.report()['transport']['drive_get_attempts'],len(self.backend.calls))
        self.assertGreaterEqual(self.report()['transport']['observed_body_bytes'],sum(len(x) for x in self.fx.wire.values())+2*len(b'{"fixture_retry":true}'))
    def test_V03_exhausted_retries_preserve_attempt_budget(self):
        self.initialized();ident=self.fx.files[0]['id'];path='/drive/v3/files/'+ident;self.backend.failures[path]=[503]*10
        with patch.object(evidence_drive.time,'sleep',lambda _:None): self.fails(lambda:self.m.capture(self.fx.spec,1))
        self.assertEqual(len([x for x in self.backend.calls if x['path']==path and x['media']]),3)
        mirror=self.m.verified_record(self.fx.root/'transport-current.json');self.assertEqual(mirror['counters']['drive_get_attempts'],len(self.backend.calls))
        self.assertGreater(mirror['counters']['observed_body_bytes'],0)
    def test_V04_missing_pin_blocks_before_media(self):
        self.initialized();self.backend.files.pop(0)
        self.fails(lambda:self.m.capture(self.fx.spec,1),'PINNED_OBJECT_MISSING')
        self.assertFalse(any(x['media'] for x in self.backend.calls))
    def test_V04_changed_pin_blocks_before_media(self):
        self.initialized();self.backend.files[0]['sha256Checksum']='0'*64
        self.fails(lambda:self.m.capture(self.fx.spec,1),'PINNED_METADATA_CHANGED')
        self.assertFalse(any(x['media'] for x in self.backend.calls))
    def test_V04_changed_content_blocks_no_acceptance(self):
        self.initialized();name=self.fx.files[0]['name'];self.backend.wire=dict(self.fx.wire);self.backend.wire[name]=b'changed raw bytes'
        self.fails(lambda:self.m.capture(self.fx.spec,1))
        self.assertFalse((self.fx.root/'slice-01-result.json').exists())
    def test_V04_new_names_excluded(self):
        self.initialized();new=self.fx.spec['namespace']+'new-unapproved.json';self.backend.files.append({'id':'new-id','name':new,'sha256Checksum':sha(b'new')});self.backend.wire=dict(self.fx.wire,**{new:b'new'})
        self.m.capture(self.fx.spec,1)
        inventory=self.m.verified_record(self.fx.root/'slice-01-inventory.json')
        self.assertEqual(inventory['added_names_excluded'],[new]);self.assertEqual(self.report()['cumulative_objects'],len(self.fx.wire))
        self.assertFalse(any(x['path'].endswith('/new-id') for x in self.backend.calls))
    def test_V04_conflicting_duplicates_block(self):
        self.fx.files.append(dict(self.fx.files[0],id='duplicate-id'))
        for item in self.fx.files:
            if item['name']==self.fx.files[0]['name']: item.pop('sha256Checksum',None)
        self.fx.publish();self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=copy.deepcopy(self.fx.wire)
        name=self.fx.files[0]['name'];self.backend.wire[name]={self.fx.files[0]['id']:self.fx.wire[name],'duplicate-id':b'conflicting bytes'}
        self.full_raw_failure()
    def full_raw_failure(self):
        self.initialized();self.fails(lambda:self.m.capture(self.fx.spec,1));self.assertFalse((self.fx.root/'slice-01-result.json').exists())
    def test_V04_source_storage_and_actual_membership_guards(self):
        wrong=copy.deepcopy(self.fx.spec);wrong['source_revision']='0'*40;self.fails(lambda:self.m.source_check(wrong),'SOURCE_SHA_CONFLICT')
        cp=json.loads((self.fx.retained/'slice-01/checkpoint.json').read_bytes());cp['processed'][next(iter(cp['processed']))]['metadata_token']='altered'
        cp['checkpoint_sha256']=census._checkpoint_digest(cp);write_json(self.fx.retained/'slice-01/checkpoint.json',cp)
        self.configured(anchor_checkpoint_file_sha256=sha((self.fx.retained/'slice-01/checkpoint.json').read_bytes()),anchor_checkpoint_canonical_sha256=cp['checkpoint_sha256'])
        self.fails(lambda:self.m.anchor(self.fx.spec),'PROCESSED_MEMBERSHIP_CONFLICT')
    def test_V04_secure_principal_binding_blocks(self):
        self.initialized();original=self.backend.session
        def wrong():
            session=original();session.credentials.service_account_email='different-fixture@example.invalid';return session
        with patch.object(evidence_drive,'_authorized_session',wrong): self.fails(lambda:self.m.capture(self.fx.spec,1),'SECURE_STORAGE_BINDING_CONFLICT')
        self.assertFalse(any(x['media'] for x in self.backend.calls))
    def test_V05_census_facts_do_not_replace_raw(self):
        self.full_raw();self.assertEqual(self.report()['physical_metrics']['objects_downloaded'],len(self.fx.wire))
        self.assertEqual(len(list((self.fx.root/'raw').iterdir())),len({sha(x) for x in self.fx.wire.values()}))
    def test_V05_verified_full_byte_cache_reused(self):
        self.initialized()
        for raw in self.fx.wire.values(): (self.fx.root/'raw'/sha(raw)).write_bytes(raw)
        self.m.capture(self.fx.spec,1)
        self.assertEqual(self.report()['physical_metrics'].get('objects_downloaded',0),0)
        self.assertEqual(self.report()['physical_metrics']['objects_reused'],len(self.fx.wire))
        self.assertFalse(any(x['media'] for x in self.backend.calls))
    def test_V05_two_slices_preserve_progress_and_aggregate_budget(self):
        self.configured(max_new_objects_per_slice=8);self.initialized();self.m.capture(self.fx.spec,1);first=self.report()
        self.m.capture(self.fx.spec,2);second=self.report(2)
        self.assertEqual(second['accepted_reused_logical_objects'],8);self.assertEqual(second['newly_retained_logical_objects'],8)
        self.assertEqual(second['cumulative_objects'],16);self.assertGreater(second['transport']['drive_get_attempts'],first['transport']['drive_get_attempts'])
        self.assertEqual(second['transport']['drive_get_attempts'],len(self.backend.calls))
        self.assertEqual(self.m.check_seal(self.fx.root,{'file':'slice-01-result.json','sha256':self.m.digest(self.fx.root/'slice-01-result.json')})['transport'],first['transport'])
    def test_V05_cache_corruption_rejects_resume_before_requests(self):
        self.configured(max_new_objects_per_slice=8);self.full_raw();cached=next((self.fx.root/'raw').iterdir());cached.write_bytes(b'corrupt')
        before=len(self.backend.calls);self.fails(lambda:self.m.capture(self.fx.spec,2),'RAW_CACHE_REJECTED');self.assertEqual(len(self.backend.calls),before)
    def test_V05_batch_corruption_rejects_resume_before_requests(self):
        self.configured(max_new_objects_per_slice=8);self.full_raw();batch=next(self.fx.root.glob('slice-*-batch-*.json'));batch.write_bytes(b'corrupt')
        before=len(self.backend.calls);self.fails(lambda:self.m.capture(self.fx.spec,2),'RAW_LEDGER_FILE_CONFLICT');self.assertEqual(len(self.backend.calls),before)
    def test_V05_transport_journal_corruption_rejects(self):
        self.configured(max_new_objects_per_slice=8);self.full_raw();(self.fx.root/'transport-events.jsonl').write_bytes(b'{corrupt}\n')
        before=len(self.backend.calls);self.fails(lambda:self.m.capture(self.fx.spec,2),'TRANSPORT_JOURNAL_INVALID');self.assertEqual(len(self.backend.calls),before)
    def test_V05_transport_mirror_corruption_rejects(self):
        self.configured(max_new_objects_per_slice=8);self.full_raw();(self.fx.root/'transport-current.json').write_bytes(b'corrupt')
        before=len(self.backend.calls);self.fails(lambda:self.m.capture(self.fx.spec,2),'SEALED_RECORD_MISSING_OR_INVALID');self.assertEqual(len(self.backend.calls),before)
    def test_V06_object_limit_exact(self):
        self.configured(max_new_objects_per_slice=5);self.full_raw();r=self.report()
        self.assertEqual(r['newly_retained_logical_objects'],5);self.assertEqual(r['terminal_reason'],'OBJECT_LIMIT');self.assertEqual(r['status'],'PARTIAL')
    def test_V06_soft_media_byte_limit_retains_one_batch_overshoot(self):
        self.configured(soft_bytes_per_slice=1);self.full_raw();r=self.report()
        self.assertEqual(r['newly_retained_logical_objects'],8);self.assertEqual(r['terminal_reason'],'BYTE_LIMIT')
        self.assertGreater(r['soft_byte_overshoot'],0);self.assertEqual(r['soft_byte_overshoot'],r['physical_metrics']['bytes_downloaded']-1)
    def test_V06_aggregate_threshold_stops_before_next_batch(self):
        self.configured(soft_bytes_total=5000);self.full_raw();r=self.report()
        self.assertEqual(r['terminal_reason'],'BLOCK_BYTE_LIMIT');self.assertEqual(r['newly_retained_logical_objects'],8)
        self.fails(lambda:self.m.capture(self.fx.spec,2),'PREVIOUS_SLICE_NOT_RESUMABLE')
    def test_V06_observed_body_safety_stop_counts_crossing_chunk(self):
        self.configured(transport_observed_body_bytes_stop=10);session,meter=self.transport()
        self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/fixture-folder'),'BODY_BYTE_STOP')
        self.assertGreater(meter['observed_body_bytes'],10);self.assertEqual(meter['drive_get_attempts'],1)
        self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/fixture-folder'),'BODY_BYTE_STOP');self.assertEqual(len(self.backend.calls),1)
    def test_V06_oversized_media_counts_observed_bytes(self):
        self.configured(max_object_bytes=1);session,meter=self.transport();ident=self.fx.files[0]['id']
        self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/'+ident+'?alt=media'),'REMOTE_OBJECT_SIZE_LIMIT')
        self.assertEqual(meter['observed_body_bytes'],len(self.fx.wire[self.fx.files[0]['name']]))
    def test_V06_processing_deadline_is_batch_soft_stop(self):
        self.configured(slice_processing_seconds=1);self.initialized();clock=[0.0]
        self.backend.delay=lambda request:clock.__setitem__(0,clock[0]+2) if 'alt=media' in request.url else None
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): self.m.capture(self.fx.spec,1)
        r=self.report();self.assertEqual(r['terminal_reason'],'PROCESSING_DEADLINE');self.assertEqual(r['newly_retained_logical_objects'],8);self.assertGreater(r['processing_deadline_overshoot'],0)
    def test_V06_initial_free_disk_prerequisite(self):
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=self.m.digest(self.fx.path))
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data')}),patch.object(self.m.shutil,'disk_usage',lambda _:SimpleNamespace(free=0)):
            self.fails(lambda:self.m.supervise(self.fx.spec),'INITIAL_FREE_DISK_LIMIT')
        self.assertFalse(self.fx.root.exists());self.assertEqual(self.backend.calls,[])
    def test_V06_working_disk_monitor_and_forced_check(self):
        self.initialized();self.configured(max_working_disk_bytes=1)
        self.fails(lambda:self.m.budget_disk(self.fx.root,self.fx.spec,force=True),'WORKING_DISK_LIMIT')
    def test_V06_standalone_database_size_limit(self):
        self.configured(max_snapshot_input_bytes=4096);self.full_raw();self.fails(lambda:self.m.materialize(self.fx.spec))
        self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists());self.assertFalse((self.fx.root/'canonical.sqlite3').exists())
    def test_V06_eight_slices_no_ninth_dispatch_or_import(self):
        self.configured(max_new_objects_per_slice=2);result,phases=self.inline_supervisor()
        self.assertEqual(result,2);self.assertEqual(phases,['capture-block']);self.assertEqual(self.report(8)['cumulative_objects'],16)
        self.assertFalse((self.fx.root/'canonical.sqlite3').exists());self.assertEqual(self.m.verified_record(self.fx.root/'operation-result.json')['status'],'PARTIAL')
    def test_V06_worker_failure_no_automatic_retry(self):
        def failure(phase,command,proc): raise RuntimeError('SYNTHETIC_FAILURE')
        result,phases=self.inline_supervisor(failure);self.assertEqual(result,3);self.assertEqual(phases,['capture-block'])
        self.assertTrue(self.m.verified_record(self.fx.root/'operation-blocked.json')['no_cold_restart_authorized'])
    def test_V06_post_exit_stage_wall_guard(self):
        clock=[0.0]
        def late(phase,command,proc):
            proc.poll=lambda:(clock.__setitem__(0,clock[0]+3001) or 0)
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): result,phases=self.inline_supervisor(late)
        self.assertEqual(result,3);self.assertEqual(phases,['capture-block']);self.assertEqual(self.m.verified_record(self.fx.root/'operation-blocked.json')['reason'],'STAGE_WALL_LIMIT')
    def test_V06_post_exit_whole_wall_guard(self):
        self.configured(total_wall_seconds=1);clock=[0.0]
        def late(phase,command,proc): proc.poll=lambda:(clock.__setitem__(0,2) or 0)
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): result,phases=self.inline_supervisor(late)
        self.assertEqual(result,3);self.assertEqual(self.m.verified_record(self.fx.root/'operation-blocked.json')['reason'],'TOTAL_WALL_LIMIT')
    def test_V07_payload_hash_corruption_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_fact/' in x);payload=json.loads(self.fx.wire[name]);payload['row'][payload['columns'].index('payload_hash')]='0'*64
        self.fx.repin_invalid(name,codec._json(payload));self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire
        self.full_raw();self.fails(lambda:self.m.materialize(self.fx.spec));self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V07_source_hash_corruption_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_source/' in x);payload=json.loads(self.fx.wire[name]);payload['row'][payload['columns'].index('source_hash')]='0'*64
        self.fx.repin_invalid(name,codec._json(payload));self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire
        self.full_raw();self.fails(lambda:self.m.materialize(self.fx.spec));self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V07_wrong_primary_key_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_fact/' in x)
        renamed=name.rsplit('/',1)[0]+'/wrong-primary-key.json'
        self.fx.repin_invalid(name,self.fx.wire[name],renamed);self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire
        self.full_raw();self.fails(lambda:self.m.materialize(self.fx.spec));self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V07_dependency_closure_failure_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_source/' in x);del self.fx.wire[name];self.fx.files=[x for x in self.fx.files if x['name']!=name];self.fx.publish()
        self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire;self.full_raw()
        self.fails(lambda:self.m.materialize(self.fx.spec),'DEPENDENCY_CLOSURE_FAILED');self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V07_database_reencoding_failure_rejected(self):
        self.full_raw();original=codec._encode;original_decode=codec._decode;decoding=[False]
        def decoded(*args):
            decoding[0]=True
            try: return original_decode(*args)
            finally: decoding[0]=False
        def changed(*args):
            key,raw=original(*args);return key,raw if decoding[0] else raw+b' '
        with patch.object(codec,'_encode',changed),patch.object(codec,'_decode',decoded): self.fails(lambda:self.m.materialize(self.fx.spec),'DATABASE_WIRE_REPLAY_FAILED')
        self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V08_empty_production_tables_complete_blocked_readiness(self):
        self.full_raw();self.m.materialize(self.fx.spec);self.m.assess(self.fx.spec)
        r=self.m.verified_record(self.fx.root/'local-assessment.json')
        for table in ['prospective_model','prospective_model_training_result','prospective_calibration','prospective_calibration_result','prospective_prediction','prospective_validation_artifact','prospective_deployment_review']: self.assertEqual(r['canonical_counts'][table],0)
        self.assertFalse(r['production_eligible']);self.assertEqual(r['recommended_stake'],0)
        self.assertEqual(len(r['plans']),16);self.assertTrue(all(x['status']=='BLOCKED_MISSING_BINDING' for x in r['plan_assessments']))
    def test_V09_historical_timestamps_and_forbidden_entrypoints(self):
        self.fx.add_football();self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire
        self.full_raw();self.m.materialize(self.fx.spec);self.m.assess(self.fx.spec)
        with closing(evidence._reader(self.fx.root/'canonical.sqlite3')) as db:
            for table,(columns,primary) in codec._schema(db).items():
                for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                    key,raw=codec._encode(table,columns,primary,tuple(row));self.assertEqual(raw,self.fx.wire[key])
        # Every upload/build/register/freeze entrypoint is trapped by OfflineRuntime.
        self.assertTrue(all(x['method']=='GET' for x in self.backend.calls))
    def test_V10_read_only_readers_and_input_hashes(self):
        self.full_raw();self.m.materialize(self.fx.spec);before=sha((self.fx.root/'canonical.sqlite3').read_bytes());calls=[]
        original=self.m.sqlite3.connect
        def observed(*args,**kwargs):
            calls.append((str(args[0]),kwargs.get('uri',False)));return original(*args,**kwargs)
        with patch.object(self.m.sqlite3,'connect',observed): self.m.assess(self.fx.spec)
        self.assertTrue(calls);self.assertTrue(all(uri and 'mode=ro' in path for path,uri in calls),calls)
        self.assertEqual(sha((self.fx.root/'canonical.sqlite3').read_bytes()),before)
    def test_V10_changed_input_bytes_rejected(self):
        self.full_raw();self.m.materialize(self.fx.spec);path=self.fx.root/'canonical.sqlite3'
        with path.open('ab') as stream: stream.write(b'changed')
        self.fails(lambda:self.m.assess(self.fx.spec),'ASSESSMENT_INPUT_REJECTED');self.assertFalse((self.fx.root/'local-assessment.json').exists())
    def test_V10_sidecar_rejected(self):
        self.full_raw();self.m.materialize(self.fx.spec);(self.fx.root/'canonical.sqlite3-wal').write_bytes(b'sidecar')
        self.fails(lambda:self.m.assess(self.fx.spec),'ASSESSMENT_SIDECAR_CONFLICT')
    def test_V11_supported_zero_and_eight_unsupported_unknown(self):
        self.full_raw();self.m.materialize(self.fx.spec);self.m.assess(self.fx.spec);r=self.m.verified_record(self.fx.root/'local-assessment.json')
        self.assertEqual(len(r['scopes']),12)
        unsupported=[x for x in r['scopes'] if not x['scope'].startswith(('NFL/','NCAAF/'))]
        self.assertEqual(len(unsupported),8);self.assertTrue(all(x['football_manifest']['independent_games'] is None and x['football_manifest']['status']=='UNKNOWN_READER_UNSUPPORTED' for x in unsupported))
        self.assertTrue(all(x['governing_release_route']=='UNKNOWN' for x in r['scopes']))
        self.assertTrue(all(x['model_id'] is None and x['calibration_id'] is None and x['cohorts']=='UNKNOWN' for x in r['plan_assessments']))
        self.assertEqual({p['validation_plan_id'] for p in r['plans']},{f['validation_plan_id'] for entry in json.loads((self.fx.retained/'slice-01/checkpoint.json').read_bytes())['processed'].values() for f in entry['facts'] if f['record_type']=='prospective_validation_plan'})
    def test_V03_exact_media_path_required(self):
        session,meter=self.transport();ident=self.fx.files[0]['id']
        self.fails(lambda:session.get('https://www.googleapis.com/drive/v3/files/unapproved-subpath/'+ident+'?alt=media'))
        self.assertEqual(self.backend.calls,[])
    def test_V04_anchor_file_archive_and_embedded_digest_faults(self):
        original=self.fx.spec
        for field in ['anchor_archive_sha256','anchor_checkpoint_file_sha256','anchor_report_sha256','anchor_checkpoint_canonical_sha256','canonical_membership_sha256']:
            spec=copy.deepcopy(original);spec[field]='0'*64
            with self.subTest(field=field): self.fails(lambda:self.m.anchor(spec))
        self.assertEqual(self.backend.calls,[])
    def test_V05_historical_state_commit_corruption_rejected(self):
        self.configured(max_new_objects_per_slice=8);self.full_raw();first=sorted(self.fx.root.glob('state-commit-*.json'))[0];first.write_bytes(b'corrupt')
        before=len(self.backend.calls);self.fails(lambda:self.m.capture(self.fx.spec,2));self.assertEqual(len(self.backend.calls),before)
    def test_V05_attempt_marker_survives_disk_failure(self):
        self.initialized();self.configured(max_working_disk_bytes=1)
        # Spec updates invalidate the state before any transport: expected, no restart.
        self.fails(lambda:self.m.capture(self.fx.spec,1),'RAW_STATE_SPEC_CONFLICT');self.assertFalse(any(x['media'] for x in self.backend.calls))
    def test_V06_post_exit_disk_growth_rejected(self):
        self.configured(max_working_disk_bytes=10000)
        def growing(phase,command,proc): (self.fx.root/'synthetic-growth').write_bytes(b'x'*20000)
        result,phases=self.inline_supervisor(growing);self.assertEqual(result,3);self.assertEqual(phases,['capture-block'])
        self.assertEqual(self.m.verified_record(self.fx.root/'operation-blocked.json')['reason'],'WORKING_DISK_LIMIT')
    def test_V06_capture_total_wall_stops_without_next_slice(self):
        self.configured(max_new_objects_per_slice=8,capture_total_wall_seconds=2);clock=[0.0]
        def consumed(phase,command,proc):
            original_verify=self.m.verified_record
            def consumed_result(path):
                result=original_verify(path)
                if str(path).endswith('slice-01-result.json'): clock[0]=2
                return result
            with patch.object(self.m,'verified_record',consumed_result): self.m.capture_block(self.fx.spec)
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): result,phases=self.inline_supervisor(consumed)
        self.assertEqual(result,2);self.assertEqual(phases,['capture-block']);self.assertEqual(self.report()['cumulative_objects'],8)
    def test_V06_materialization_and_assessment_stage_wall_guards(self):
        self.configured(materialization_wall_seconds=1,assessment_wall_seconds=1);clock=[0.0]
        def late(phase,command,proc):
            if phase=='capture-block': self.m.capture_block(self.fx.spec)
            elif phase=='materialize': proc.poll=lambda:(clock.__setitem__(0,clock[0]+2) or 0)
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): result,phases=self.inline_supervisor(late)
        self.assertEqual(result,3);self.assertEqual(phases,['capture-block','materialize']);self.assertEqual(self.m.verified_record(self.fx.root/'operation-blocked.json')['reason'],'STAGE_WALL_LIMIT')
    def test_V06_windows_timeout_stops_entire_synthetic_worker_tree(self):
        if os.name!='nt': self.skipTest('Windows launcher process-tree verification')
        self.configured(slice_wall_seconds=2);backend_path=self.fx.backend_file();data=json.loads(backend_path.read_bytes());data['slow_worker_seconds']=4;write_json(backend_path,data)
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=self.m.digest(self.fx.path));original=subprocess.Popen
        def child(command,**kwargs):
            if '--worker' not in command: return original(command,**kwargs)
            return original([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--isolated-child',str(backend_path),*command[3:]],**kwargs)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data')}),patch.object(self.m.subprocess,'Popen',child): result=self.m.supervise(self.fx.spec)
        self.assertEqual(result,3);self.assertTrue((self.fx.base/'slow-worker-started.json').exists())
        time.sleep(4.2)
        marker=self.fx.base/'UNSTOPPED-SYNTHETIC-WORKER.json'
        self.faults.append({'synthetic_descendant_survived':marker.exists(),'reason':self.m.verified_record(self.fx.root/'operation-blocked.json')['reason']})
        self.assertFalse(marker.exists(),'Windows launcher killed but its synthetic worker survived the deadline')
    def test_V07_invalid_json_and_unknown_schema_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_fact/' in x)
        self.fx.repin_invalid(name,b'not-json');self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire;self.full_raw()
        self.fails(lambda:self.m.materialize(self.fx.spec),'canonical_remote_json_invalid');self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V10_input_changed_during_assessment_rejected(self):
        self.full_raw();self.m.materialize(self.fx.spec);original=evidence.read_records;changed=[]
        def tamper(*args,**kwargs):
            result=original(*args,**kwargs)
            if not changed:
                changed.append(True)
                with (self.fx.root/'canonical.sqlite3').open('ab') as stream: stream.write(b'tamper')
            return result
        with patch.object(evidence,'read_records',tamper): self.fails(lambda:self.m.assess(self.fx.spec),'READ_ONLY_SNAPSHOT_CHANGED')
        self.assertFalse((self.fx.root/'local-assessment.json').exists())
    def test_V05_retention_failure_keeps_raw_and_forbids_retry(self):
        self.initialized();original=self.m.seal
        def disk_failure(root,name,value):
            if '-batch-' in name: raise OSError('SYNTHETIC_LEDGER_RETENTION_FAILURE')
            return original(root,name,value)
        with patch.object(self.m,'seal',disk_failure): self.fails(lambda:self.m.capture(self.fx.spec,1),'SYNTHETIC_LEDGER_RETENTION_FAILURE')
        self.assertGreater(len(list((self.fx.root/'raw').iterdir())),0);before=len(self.backend.calls)
        self.fails(lambda:self.m.capture(self.fx.spec,1),'SLICE_RETRY_NOT_AUTHORIZED');self.assertEqual(len(self.backend.calls),before)
    def test_V06_no_forward_progress_blocks_and_keeps_attempt(self):
        self.configured(slice_processing_seconds=0);self.initialized();self.fails(lambda:self.m.capture(self.fx.spec,1),'NO_FORWARD_PROGRESS')
        self.assertTrue((self.fx.root/'slice-01-attempt.json').exists());before=len(self.backend.calls)
        self.fails(lambda:self.m.capture(self.fx.spec,1),'SLICE_RETRY_NOT_AUTHORIZED');self.assertEqual(len(self.backend.calls),before)
    def test_V06_assessment_stage_wall_guard(self):
        self.configured(assessment_wall_seconds=1);clock=[0.0]
        def late(phase,command,proc):
            if phase=='capture-block': self.m.capture_block(self.fx.spec)
            elif phase=='materialize': self.m.materialize(self.fx.spec)
            else: proc.poll=lambda:(clock.__setitem__(0,clock[0]+2) or 0)
        with patch.object(self.m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None)): result,phases=self.inline_supervisor(late)
        self.assertEqual(result,3);self.assertEqual(phases,['capture-block','materialize','assess']);self.assertEqual(self.m.verified_record(self.fx.root/'operation-blocked.json')['reason'],'STAGE_WALL_LIMIT')
    def test_V07_unknown_schema_rejected(self):
        name=next(x for x in self.fx.wire if '/prospective_reconciled_fact/' in x);payload=json.loads(self.fx.wire[name]);payload['schema']=999
        self.fx.repin_invalid(name,codec._json(payload));self.backend.files=copy.deepcopy(self.fx.files);self.backend.wire=self.fx.wire;self.full_raw()
        self.fails(lambda:self.m.materialize(self.fx.spec),'canonical_remote_schema_invalid');self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())
    def test_V07_sqlite_integrity_failure_no_acceptance(self):
        import sqlite3
        self.full_raw();original=sqlite3.connect
        class Faulty(sqlite3.Connection):
            def execute(inner,sql,*args,**kwargs):
                if sql=='PRAGMA integrity_check': return [('SYNTHETIC_CORRUPTION',)]
                return super().execute(sql,*args,**kwargs)
        def connection(*args,**kwargs): return original(*args,**dict(kwargs,factory=Faulty))
        with patch.object(sqlite3,'connect',connection): self.fails(lambda:self.m.materialize(self.fx.spec),'SQLITE_INTEGRITY_FAILED')
        self.assertFalse((self.fx.root/'snapshot-acceptance.json').exists())


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs);self.records=[];self.starts={}
    def startTest(self,test): self.starts[test.id()]=time.monotonic();super().startTest(test)
    def record(self,test,status,detail=None): self.records.append({'test':test.id(),'status':status,'elapsed_seconds':time.monotonic()-self.starts[test.id()],'fault_or_result':detail})
    def addSuccess(self,test): self.record(test,'PASS');super().addSuccess(test)
    def addSkip(self,test,reason): self.record(test,'SKIP',reason);super().addSkip(test,reason)
    def addFailure(self,test,error): self.record(test,'FAIL',self._exc_info_to_string(error,test));super().addFailure(test,error)
    def addError(self,test,error): self.record(test,'ERROR',self._exc_info_to_string(error,test));super().addError(test,error)


if __name__=='__main__':
    if '--isolated-child' in sys.argv:
        index=sys.argv.index('--isolated-child');backend_path=Path(sys.argv[index+1]);argv=sys.argv[index+2:]
        require_synthetic=json.loads(backend_path.read_bytes());assert require_synthetic['synthetic_only'] is True
        wire={k:base64.b64decode(v) for k,v in require_synthetic['wire'].items()};backend=FakeWire(require_synthetic['files'],wire)
        write_json(backend_path.with_name('synthetic-child-process.json'),{'pid':os.getpid(),'parent_pid':os.getppid(),'synthetic_only':True})
        if require_synthetic.get('slow_worker_seconds'):
            write_json(backend_path.with_name('slow-worker-started.json'),{'synthetic_only':True,'pid':os.getpid()})
            time.sleep(require_synthetic['slow_worker_seconds'])
            write_json(backend_path.with_name('UNSTOPPED-SYNTHETIC-WORKER.json'),{'synthetic_only':True})
        with OfflineRuntime(backend):
            sys.argv=argv
            try: runpy.run_path(argv[0],run_name='__main__')
            finally:
                write_json(backend_path.with_name('fake-child-'+argv[argv.index('--worker')+1]+'-attempts.json'),{'real_socket_attempts':REAL_REQUESTS,'fake_requests':backend.calls,'synthetic_only':True})
    else:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--driver',type=Path,default=RUNTIME_DRIVER);parser.add_argument('--run-directory',type=Path,required=True);parser.add_argument('--select');arguments=parser.parse_args()
        DRIVER=arguments.driver.resolve();RUN_DIR=arguments.run_directory.resolve();RUN_DIR.mkdir(parents=True,exist_ok=False)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(FunctionalTests)
        if arguments.select: suite=unittest.TestSuite(test for test in suite if arguments.select in test.id())
        result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
        write_json(RUN_DIR/'test-results.json',{'driver_bytes':DRIVER.stat().st_size,'driver_sha256':sha(DRIVER.read_bytes()),'spec_sha256':sha(TEMPLATE.read_bytes()),'synthetic_only':True,'real_socket_attempts':REAL_REQUESTS,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'results':result.records,'success':result.wasSuccessful()})
        sys.exit(0 if result.wasSuccessful() else 1)
