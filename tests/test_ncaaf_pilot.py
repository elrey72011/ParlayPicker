"""SYNTHETIC execution only; no real sports/Drive/model-service transport."""
import base64
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pandas as pd
from app_core import ncaaf_pilot as pilot, ncaaf_response_custody as custody, ncaaf_model_compatibility as model
from app_core import ncaaf_pipeline_evidence as adapter,ncaaf_research as research
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
import test_ncaaf_response_custody as previous


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


class Clock:
    value='2026-10-09T11:59:59Z'
    elapsed=0.
    def wall(self):return self.value
    def monotonic(self):return self.elapsed


class Stream:
    def __init__(self,raw,status=200,headers=None,chunks=None):
        self.raw=raw;self.status=status;self.headers=headers or {'content-type':'application/json','content-length':str(len(raw))}
        self.blocks=[raw] if chunks is None else chunks;self.closed=False
    def chunks(self):yield from self.blocks
    def close(self):self.closed=True


def fixture(synthetic,monkeypatch,kind='spread_home',line=-3.5,root=None):
    packet,row,approval=previous.fixture(synthetic,monkeypatch,kind,line)
    objs=packet['payload']['response_objects'];q=packet['payload']['native_packet']['payload']['observation']['payload']['quote']
    requests=[dict(id='SYNTHETIC-'+str(i),provider=o['payload']['metadata']['provider'],host=pilot.HOSTS[o['payload']['metadata']['provider']],
        endpoint=o['payload']['metadata']['endpoint'],params=deepcopy(o['payload']['metadata']['request_scope']),max_attempts=1) for i,o in enumerate(objs)]
    p=dict(version=pilot.VERSION,evidence_label='SYNTHETIC',target=dict(canonical_event_id='cfbd:9',provider_event_id=q['provider_event_id'],
        home_team=q['event_home_team'],away_team=q['event_away_team'],start_utc=q['event_start_utc'],neutral_site=False),
        offer=dict(market=q['market_type'],side=kind.split('_')[1],signed_line=q['point'],price=q['price'],bookmaker=q['book'],product=q['product'],period=q['period'],
            settlement_review_sha256=model.digest(packet['payload']['native_packet']['payload']['observation']['payload']['source_review']['terms_review'])),
        requests=requests,limits=dict(max_attempts=len(requests),max_objects=16,max_body_bytes=512*1024,max_aggregate_bytes=2*1024*1024,connect_seconds=5,read_seconds=5,total_seconds=30),
        execution_window=dict(not_before='2026-10-09T11:59:58Z',not_after='2026-10-09T12:00:00.750000Z',authorization_expires_at='2026-10-09T12:00:01Z'),
        advance_permissions=[],collection_authorization_ref='SYNTHETIC_OWNER_AUTH',custody_id='SYNTHETIC_LOCAL_CUSTODY',unresolved=[])
    for provider in ('cfbd','odds_api'):
        r=dict(reference='SYNTHETIC-advance-'+provider,provider=provider,reviewed_at='2026-10-09T11:00:00Z',effective_from='2026-10-01T00:00:00Z',
            effective_until='2026-11-01T00:00:00Z',permitted_use='private_prospective_capture',requests_sha256=model.digest([r for r in requests if r['provider']==provider]))
        p['advance_permissions'].append(r)
    plan=pilot.seal(p);auth=authorize(plan,monkeypatch)
    if root is not None:monkeypatch.setattr(pilot,'AUTHORIZED_CUSTODY_ROOTS',{p['custody_id']:str(root.resolve())})
    clock=Clock();calls=[]
    def transport(request,*args):
        i=len(calls);calls.append(deepcopy(request));clock.value=objs[i]['payload']['metadata']['received_at']
        return Stream(base64.b64decode(objs[i]['payload']['body_b64']))
    return plan,auth,packet,row,approval,clock,calls,transport


def authorize(plan,monkeypatch):
    p=plan['payload'];a=pilot.seal(dict(version=pilot.AUTH_VERSION,reference=p['collection_authorization_ref'],plan_sha256=plan['sha256'],custody_id=p['custody_id'],
        authorized_at='2026-10-09T11:30:00Z',expires_at='2026-10-09T12:00:01Z',owner='SYNTHETIC-owner'))
    monkeypatch.setattr(pilot,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(pilot,'ACCEPTED_ADVANCE_PERMISSIONS',{r['reference']:model.digest(r) for r in p['advance_permissions']})
    return a


def forbidden(monkeypatch):
    from app_core import football_stage1_cycle as stage,prospective_remote,odds_api,college_novig,ncaaf_schedule
    from core import streamlit_pipeline as live
    guards=[]
    for obj,name in [(stage,'run_cycle'),(prospective_remote,'sync'),(live,'run_analysis_pipeline'),(live,'fetch_live_odds_dataframe'),
        (odds_api.TheOddsAPIClient,'get_odds'),(college_novig,'recover_college_novig'),(ncaaf_schedule,'fetch_schedule')]:
        f=Mock(side_effect=AssertionError('Forbidden pilot entrypoint'));monkeypatch.setattr(obj,name,f);guards.append(f)
    monkeypatch.setattr('app_core.external_data_fetcher.enrich_with_external_data',Mock(side_effect=AssertionError('Enrichment forbidden')))
    return guards


def test_default_planning_no_credentials_or_storage(synthetic,monkeypatch,tmp_path):
    plan,*_=fixture(synthetic,monkeypatch,root=tmp_path);guards=forbidden(monkeypatch)
    before=list(tmp_path.iterdir())
    result=pilot.planning(plan)
    assert result['network_requests']==result['credential_loads']==result['storage_operations']==0
    assert list(tmp_path.iterdir())==before and not result['accepted'] and not result['inference']
    assert all(not g.called for g in guards)


@pytest.mark.parametrize('change,code',[
    ('missing','AUTHORIZATION_MISSING'),('expired','AUTHORIZATION_EXPIRED'),('mismatch','AUTHORIZATION_CONFLICT'),
    ('untrusted','AUTHORIZATION_UNTRUSTED'),('window','WINDOW_EXPIRED'),('permissions','PERMISSION_UNTRUSTED'),
    ('late_review','PERMISSION_CLOCK'),('scope','PERMISSION_SCOPE'),('unresolved','PREREQUISITES'),('host','HOST')])
def test_authorization_before_first_request(synthetic,monkeypatch,tmp_path,change,code):
    plan,auth,_,_,_,clock,calls,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    if change=='missing':auth=None
    elif change=='expired':auth['payload']['expires_at']='2026-10-09T11:59:58Z';auth=pilot.seal(auth['payload']);monkeypatch.setattr(pilot,'AUTHORIZED_COLLECTIONS',{'SYNTHETIC_OWNER_AUTH':auth['sha256']})
    elif change=='mismatch':auth['payload']['plan_sha256']='a'*64;auth=pilot.seal(auth['payload'])
    elif change=='untrusted':monkeypatch.setattr(pilot,'AUTHORIZED_COLLECTIONS',{})
    elif change=='window':clock.value='2026-10-09T12:00:02Z'
    elif change=='permissions':monkeypatch.setattr(pilot,'ACCEPTED_ADVANCE_PERMISSIONS',{})
    else:
        p=plan['payload']
        if change=='late_review':p['advance_permissions'][0]['reviewed_at']=clock.value
        elif change=='scope':p['advance_permissions'][0]['requests_sha256']='a'*64
        elif change=='unresolved':p['unresolved']=['SOURCE_ACCEPTANCE_MISSING']
        elif change=='host':p['requests'][0]['host']='evil.example'
        plan=pilot.seal(p);auth=authorize(plan,monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_PILOT_'+code):pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert calls==[] and list(tmp_path.iterdir())==[]


def test_exact_requests_full_bytes_unaccepted_and_no_resume(synthetic,monkeypatch,tmp_path):
    plan,auth,packet,_,_,clock,calls,transport=fixture(synthetic,monkeypatch,root=tmp_path);guards=forbidden(monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Acquisition cannot infer')))
    before=deepcopy(plan)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert calls==plan['payload']['requests'] and bundle['payload']['attempted_requests']==len(calls)
    assert bundle['payload']['status']=='CAPTURED_UNACCEPTED' and not bundle['payload']['accepted'] and not bundle['payload']['inference']
    assert plan==before and all(not g.called for g in guards)
    for obj,original in zip(bundle['payload']['objects'],packet['payload']['response_objects']):
        assert obj['body_b64']==original['payload']['body_b64'] and obj['metadata']==original['payload']['metadata']
    assert (tmp_path/(plan['sha256']+'.unaccepted.json')).read_bytes()==pilot.encode(bundle)
    with pytest.raises(ValueError,match='ALREADY_ATTEMPTED_NO_RESUME'):pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert len(calls)==len(plan['payload']['requests'])


@pytest.mark.parametrize('failure,code',[
    ('redirect','HTTP_STATUS'),('pagination','REDIRECT_OR_PAGINATION'),('link','REDIRECT_OR_PAGINATION'),('auth','AUTH_OR_RATE_LIMIT'),
    ('rate','AUTH_OR_RATE_LIMIT'),('timeout','TRANSPORT_FAILURE'),('slow','DEADLINE'),('incomplete','INCOMPLETE_BODY'),
    ('huge','BODY_LIMIT'),('gzip_bomb','BODY_LIMIT'),('gzip_incomplete','INCOMPLETE_BODY'),('encoding','CONTENT_ENCODING'),
    ('malformed','NCAAF_CUSTODY_BODY_SCHEMA'),('credential','NCAAF_CUSTODY_CREDENTIAL_FIELD'),('aggregate','AGGREGATE_LIMIT')])
def test_bounded_partial_exports_no_retry(synthetic,monkeypatch,tmp_path,failure,code):
    plan,auth,_,_,_,clock,calls,_=fixture(synthetic,monkeypatch,root=tmp_path)
    if failure=='aggregate':plan['payload']['limits']['max_aggregate_bytes']=2;plan=pilot.seal(plan['payload']);auth=authorize(plan,monkeypatch)
    def transport(request,*args):
        calls.append(request);raw=b'[{}]';s=Stream(raw)
        if failure=='redirect':s.status=302;s.headers['location']='https://evil.example/?apiKey=SYNTHETIC_SECRET'
        elif failure=='pagination':s.headers['x-next-page']='page-two'
        elif failure=='link':s.headers['link']='<https://evil.example>; rel="next"'
        elif failure=='auth':s.status=401
        elif failure=='rate':s.status=429
        elif failure=='timeout':raise TimeoutError('SYNTHETIC_SECRET_CANARY')
        elif failure=='slow':clock.elapsed=31
        elif failure=='incomplete':s.headers['content-length']='100'
        elif failure=='huge':s=Stream(b' '* (custody.MAX_BODY_BYTES+1))
        elif failure=='gzip_bomb':raw=gzip.compress(b' '* (custody.MAX_BODY_BYTES+1));s=Stream(raw);s.headers['content-encoding']='gzip'
        elif failure=='gzip_incomplete':raw=gzip.compress(b'[{}]')[:-3];s=Stream(raw);s.headers['content-encoding']='gzip'
        elif failure=='encoding':s.headers['content-encoding']='deflate'
        elif failure=='malformed':s=Stream(b'{"next_page":"two"}')
        elif failure=='credential':s=Stream(b'[{"apiKey":"SYNTHETIC_SECRET_CANARY"}]')
        return s
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert bundle['payload']['status']=='INCOMPLETE' and bundle['payload']['reason'].endswith(code)
    assert bundle['payload']['attempted_requests']==len(calls)==1 and bundle['payload']['objects']==[]
    assert 'SYNTHETIC_SECRET' not in pilot.encode(bundle).decode()
    with pytest.raises(ValueError,match='ALREADY_ATTEMPTED'):pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert len(calls)==1


def test_interruption_keeps_spent_attempt_no_silent_resume(synthetic,monkeypatch,tmp_path):
    plan,auth,_,_,_,clock,calls,_=fixture(synthetic,monkeypatch,root=tmp_path)
    def interrupted(request,*args):calls.append(request);raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):pilot.acquire(plan,auth,root=tmp_path,transport=interrupted,wall=clock.wall,monotonic=clock.monotonic)
    with pytest.raises(ValueError,match='ALREADY_ATTEMPTED'):pilot.acquire(plan,auth,root=tmp_path,transport=interrupted,wall=clock.wall,monotonic=clock.monotonic)
    journal=(tmp_path/(plan['sha256']+'.attempts.jsonl')).read_text()
    assert len(calls)==1 and '"attempt":1' in journal


def test_changed_custody_directory_cannot_repeat_authorized_plan(synthetic,monkeypatch,tmp_path):
    plan,auth,_,_,_,clock,calls,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    def interrupted(request,*args):calls.append(request);raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):pilot.acquire(plan,auth,root=tmp_path,transport=interrupted,wall=clock.wall,monotonic=clock.monotonic)
    other=tmp_path/'different-directory';other.mkdir()
    with pytest.raises(ValueError,match='CUSTODY_IDENTITY_CONFLICT'):
        pilot.acquire(plan,auth,root=other,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert len(calls)==1 and list(other.iterdir())==[]


def test_projection_passes_exact_bytes_into_existing_custody(synthetic,monkeypatch,tmp_path):
    plan,auth,packet,_,_,clock,_,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    batches={r['id']:json.loads(base64.b64decode(b['bytes_b64'])) for r,b in zip(plan['payload']['requests'],packet['payload']['native_packet']['payload']['dependency_objects'])}
    o=packet['payload']['native_packet']['payload']['observation']['payload']
    objects=pilot.project_capture(bundle,native_batches=batches,quote=o['quote'],terms=o['source_review']['terms_review'])
    assert objects==packet['payload']['response_objects']
    assert pilot.AUTHORIZED_COLLECTIONS and not bundle['payload']['accepted']


@pytest.mark.parametrize('problem', ['unaccepted','corrupt','mismatched','stale','future'])
def test_admitted_rejects_before_numeric(synthetic,monkeypatch,tmp_path,problem):
    plan,auth,packet,row,approval,clock,_,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    if problem=='unaccepted':monkeypatch.setattr(adapter,'ACCEPTED_PACKETS',{})
    elif problem=='corrupt':packet['payload']['response_objects'][0]['payload']['body_b64']=base64.b64encode(b'[]').decode()
    elif problem=='mismatched':row['spread_line']=-4.5
    elif problem=='stale':monkeypatch.setattr(adapter,'generated_time',lambda:'2026-10-09T13:00:00Z')
    elif problem=='future':monkeypatch.setattr(adapter,'generated_time',lambda:'2026-10-09T11:00:00Z')
    if problem=='corrupt':packet['sha256']=model.digest(packet['payload']);monkeypatch.setattr(adapter,'ACCEPTED_PACKETS',{packet['sha256']:approval})
    numeric=Mock(side_effect=AssertionError('Must reject before numerical inference'));monkeypatch.setattr(research,'centers',numeric)
    guards=forbidden(monkeypatch)
    frame=pilot.admitted_analysis(packet,row,inventory=None,bundle=bundle)
    assert frame.iloc[0].ml_inference_status=='unavailable' and frame.iloc[0].ml_unavailable_reason
    assert not numeric.called and all(not g.called for g in guards)


@pytest.mark.parametrize('kind,line',[('spread_home',-3.5),('spread_away',3.5),('total_over',50.5),('total_under',50.5)])
def test_accepted_synthetic_caller_then_existing_capture_export_display(synthetic,monkeypatch,tmp_path,kind,line):
    plan,auth,packet,row,_,clock,_,transport=fixture(synthetic,monkeypatch,kind,line,root=tmp_path);guards=forbidden(monkeypatch)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    from app_core import ncaaf_schedule as ns
    from test_ncaaf_schedule_coverage import event
    inv=ns.inventory_from_events([('FBS',[event('9',row['game_start_utc'],'Alabama','Georgia')]),
        ('FCS',[event('10',row['game_start_utc'],'McNeese Cowboys','Other FCS')])],'2026-10-10','2026-10-10',complete=True,observed_at=previous.NOW.isoformat())
    frame=pilot.admitted_analysis(packet,row,inventory=inv,bundle=bundle)
    assert frame.iloc[0].ml_inference_status=='success',frame.iloc[0].ml_unavailable_reason
    assert frame.iloc[0].ml_probability==previous.native.infer(packet['payload']['native_packet'],previous.NOW.isoformat())[2]
    assert frame.iloc[0].ml_probability!=.99
    # Existing full export consumer, same synthetic clocks. No native packet is
    # upgraded, blend fabricated, source acceptance registered or stake created.
    import test_source_contract_pipeline as ef
    monkeypatch.setattr(ef,'NOW',previous.NOW);monkeypatch.setattr(ef,'CAPTURE',(previous.NOW+timedelta(seconds=1)).isoformat())
    from app_core import prediction_evidence as evidence
    db=tmp_path/'SYNTHETIC-existing.sqlite3';evidence.connect(db).close()
    monkeypatch.setattr(evidence,'now_utc',lambda:ef.CAPTURE)
    monkeypatch.setattr(pilot,'datetime',ef.FrozenDateTime)
    monkeypatch.setattr('app_core.public_board.datetime',ef.FrozenDateTime)
    # The selector evaluates pregame status independently. Keep this labelled
    # synthetic instant aligned with the caller/export clocks; production
    # chronology and the stale/started rejection fixtures remain unchanged.
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:pd.Timestamp(previous.NOW))
    result=pilot.retain_analysis(frame,path=db)
    import hashlib,sqlite3
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as stored:
        manifest=json.loads(stored.execute('SELECT manifest FROM bundles LIMIT 1').fetchone()[0])
    assert manifest['artifacts']['app_core/ncaaf_pilot.py']==hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()
    assert manifest['artifacts']['app_core/ncaaf_research.py']==hashlib.sha256(Path(research.__file__).read_bytes()).hexdigest()
    assert manifest['controls']['bankroll']==1
    from app_core.public_board import validate_package
    from app_core import research_replay
    package=result['package'];validate_package(package)
    family='sides' if kind.startswith('spread') else 'totals'
    shown=next(r for r in package['games'][family] if r.get('research_display',{}).get('probability') is not None)
    assert shown['research_display']['probability']==frame.iloc[0].ml_probability
    assert shown['research_display']['ev'] is None
    assert len(result['coverage']['decisions'])==2 and result['coverage']['counts']['states']['UNVERIFIED']==2
    for f in ('overall','sides','totals'):
        assert len(package['games'][f])==2
        assert len({r['coverage_decision']['canonical_event_id'] for r in package['games'][f]})==2
    receipt=result['export']
    _,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    producer=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    assert json.loads(producer.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['original_packet']==packet
    retained_binding=json.loads(producer.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['pilot_capture']['payload']
    assert retained_binding['bundle_sha256']==bundle['sha256'] and retained_binding['request_receipts'][0]['requested_at']=='2026-10-09T11:59:59Z'
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    public=json.dumps(package)
    assert all(k not in public for k in ('body_b64','SYNTHETIC_PRIVATE_BODY_CANARY','custody_admission','dependency_objects','SYNTHETIC_SECRET'))
    assert all(not g.called for g in guards)
    if kind=='spread_home':
        from test_research_probability_browser import inspect_browser
        browser=inspect_browser(package,tmp_path/'browser',previous.NOW)
        assert browser['initial']['current']==0 and browser['initial']['shown'][0]['probability']==frame.iloc[0].ml_probability


def test_missing_capture_store_is_not_initialized(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='EXISTING_PREDICTION_STORE_REQUIRED'):
        pilot.retain_analysis(None,path=tmp_path/'missing.sqlite3')
    assert not (tmp_path/'missing.sqlite3').exists()


def test_normal_analysis_is_not_an_isolated_acquisition_lane(synthetic,monkeypatch):
    packet,_,_=previous.fixture(synthetic,monkeypatch)
    calls=[]
    names={'fetch_schedule','recover_college_novig','fetch_espn_ncaaf_fcs_odds',
        'enrich_with_model_features','enrich_with_external_data'}
    class SpiedPatches:
        def setattr(self,target,*args,**kwargs):
            name=target.rsplit('.',1)[-1] if isinstance(target,str) else args[0]
            if name in names:
                value=args[0] if isinstance(target,str) else args[1]
                def wrapped(*a,**k):calls.append(name);return value(*a,**k)
                args=(wrapped,) if isinstance(target,str) else (name,wrapped)
            return monkeypatch.setattr(target,*args,**kwargs)
    result,_=previous.actual(SpiedPatches(),packet)
    assert not result.empty and names<=set(calls)


@pytest.mark.parametrize('attack',['label','clock_meaning','body','count','authorization','target'])
def test_rehashed_bundle_attacks_reject_before_inference(synthetic,monkeypatch,tmp_path,attack):
    plan,auth,packet,row,_,clock,_,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    b=bundle['payload']
    if attack=='label':b['evidence_label']='RETAINED'
    elif attack=='clock_meaning':b['objects'][0]['request_clock_meaning']='unknown'
    elif attack=='body':b['objects'][0]['body_b64']=base64.b64encode(b'[]').decode()
    elif attack=='count':b['attempted_requests']=1
    elif attack=='authorization':b['authorization']['payload']['owner']='attacker'
    else:b['plan']['payload']['target']['neutral_site']=True;b['plan']=pilot.seal(b['plan']['payload'])
    bundle=pilot.seal(b)
    numeric=Mock(side_effect=AssertionError('Rejected before inference'));monkeypatch.setattr(research,'centers',numeric)
    result=pilot.admitted_analysis(packet,row,inventory=None,bundle=bundle)
    assert result.iloc[0].ml_unavailable_reason=='NCAAF_PACKET_INTEGRITY' and not numeric.called


def test_transport_one_https_attempt_no_redirect_and_sanitized_failure(synthetic,monkeypatch,tmp_path):
    plan,auth,_,_,_,clock,_,_=fixture(synthetic,monkeypatch,root=tmp_path);calls=[];loads=[]
    class Connection:
        def __init__(self,host,timeout):calls.append(('connect',host,timeout));self.sock=self
        def connect(self):pass
        def settimeout(self,t):assert 0<t<=5
        def request(self,method,url,headers):
            calls.append(('request',method,url,headers));assert method=='GET' and self.auto_open is False
        def getresponse(self):return SimpleNamespace(status=302,getheaders=lambda:[('location','https://evil.test/?apiKey=SYNTHETIC_SECRET')])
        def close(self):calls.append(('close',))
    monkeypatch.setattr(pilot.http.client,'HTTPSConnection',Connection)
    transport=pilot.HttpsTransport(lambda p:loads.append(p) or 'SYNTHETIC_SECRET')
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert bundle['payload']['reason']=='NCAAF_PILOT_HTTP_STATUS' and loads==['cfbd']
    assert sum(c[0]=='request' for c in calls)==1
    assert 'SYNTHETIC_SECRET' not in pilot.encode(bundle).decode()
    assert all('SYNTHETIC_SECRET' not in p.read_text() for p in tmp_path.iterdir())


@pytest.mark.parametrize('echo',[b'SYNTHETIC_SECRET',b'\\u0053YNTHETIC_SECRET',b'%53YNTHETIC_SECRET'])
def test_credential_echo_is_rejected_before_private_persistence(synthetic,monkeypatch,tmp_path,echo):
    plan,auth,_,_,_,clock,_,_=fixture(synthetic,monkeypatch,root=tmp_path)
    raw=b'[{"echo":"'+echo+b'"}]';sent=[]
    class Response:
        status=200
        def getheaders(self):return [('content-type','application/json'),('content-length',str(len(raw)))]
        def read1(self,n):
            if hasattr(self,'read'):return b''
            self.read=True;return raw
    class Connection:
        def __init__(self,*a,**k):self.sock=self
        def connect(self):pass
        def settimeout(self,*a):pass
        def request(self,*a,**k):sent.append(a)
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setattr(pilot.http.client,'HTTPSConnection',Connection)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=pilot.HttpsTransport(lambda p:'SYNTHETIC_SECRET'),wall=clock.wall,monotonic=clock.monotonic)
    assert bundle['payload']['reason']=='NCAAF_PILOT_CREDENTIAL_IN_BODY'
    assert bundle['payload']['objects']==[] and bundle['payload']['attempted_requests']==len(sent)==1
    assert all('SYNTHETIC_SECRET' not in p.read_text() and 'echo' not in p.read_text() for p in tmp_path.iterdir())


def test_expired_total_deadline_during_connect_never_sends_get(synthetic,monkeypatch,tmp_path):
    plan,auth,_,_,_,clock,_,_=fixture(synthetic,monkeypatch,root=tmp_path);sent=[]
    class Connection:
        def __init__(self,*a,**k):self.sock=self
        def connect(self):clock.elapsed=31
        def request(self,*a,**k):sent.append(a)
        def close(self):pass
    monkeypatch.setattr(pilot.http.client,'HTTPSConnection',Connection)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=pilot.HttpsTransport(lambda p:'SYNTHETIC_SECRET'),wall=clock.wall,monotonic=clock.monotonic)
    assert not sent and bundle['payload']['status']=='INCOMPLETE' and bundle['payload']['attempted_requests']==1


@pytest.mark.parametrize('stage',['connect','request','headers','read'])
@pytest.mark.parametrize('deadline_kind',['total','phase'])
def test_blocking_os_operation_returns_before_release_without_late_get(synthetic,monkeypatch,tmp_path,stage,deadline_kind):
    import threading,time
    plan,_,_,_,_,clock,_,_=fixture(synthetic,monkeypatch,root=tmp_path)
    plan['payload']['limits'].update(connect_seconds=.15,read_seconds=.15,total_seconds=.15)
    if deadline_kind=='phase':plan['payload']['limits'].update(connect_seconds=.02,read_seconds=.02,total_seconds=2)
    plan=pilot.seal(plan['payload']);auth=authorize(plan,monkeypatch)
    release=threading.Event();began=threading.Event();sent=[];finished=threading.Event();values={}
    def stall():began.set();release.wait(5)
    class Response:
        status=200
        def getheaders(self):return [('content-type','application/json'),('content-length','2')]
        def read1(self,n):
            if stage=='read':stall()
            return b'[]'
    class Connection:
        def __init__(self,*a,**k):self.sock=self;self.closed=False
        def connect(self):
            if stage=='connect':stall()
        def settimeout(self,*a):pass
        def request(self,*a,**k):
            if stage=='request':stall()
            if self.closed:
                assert self.auto_open is False
                raise ConnectionError('Closed synthetic socket must not reconnect')
            sent.append(a)
        def getresponse(self):
            if stage=='headers':stall()
            return Response()
        def close(self):self.closed=True
    monkeypatch.setattr(pilot.http.client,'HTTPSConnection',Connection)
    def run():
        try:values['bundle']=pilot.acquire(plan,auth,root=tmp_path,transport=pilot.HttpsTransport(lambda p:'SYNTHETIC_SECRET'),wall=clock.wall,monotonic=time.monotonic)
        finally:finished.set()
    worker=threading.Thread(target=run,daemon=True);worker.start()
    try:
        assert began.wait(2) and finished.wait(.6) and not release.is_set()
        assert values['bundle']['payload']['reason']=='NCAAF_PILOT_DEADLINE'
        assert values['bundle']['payload']['attempted_requests']==1 and values['bundle']['payload']['objects']==[]
        assert len(sent)==(0 if stage in {'connect','request'} else 1)
    finally:release.set();worker.join(2)
    assert len(sent)==(0 if stage in {'connect','request'} else 1)


def test_half_point_scope_without_invented_prior_line_or_price(synthetic,monkeypatch,tmp_path):
    plan,_,packet,_,_,clock,_,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    plan['payload']['offer'].update(signed_line=None,price=None);plan=pilot.seal(plan['payload']);auth=authorize(plan,monkeypatch)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    batches={r['id']:json.loads(base64.b64decode(b['bytes_b64'])) for r,b in zip(plan['payload']['requests'],packet['payload']['native_packet']['payload']['dependency_objects'])}
    o=packet['payload']['native_packet']['payload']['observation']['payload']
    assert pilot.project_capture(bundle,native_batches=batches,quote=o['quote'],terms=o['source_review']['terms_review'])==packet['payload']['response_objects']
    wrong=deepcopy(o['quote']);wrong['point']=-3
    with pytest.raises(ValueError,match='TARGET_OFFER_CONFLICT'):pilot.project_capture(bundle,native_batches=batches,quote=wrong,terms=o['source_review']['terms_review'])


@pytest.mark.parametrize('problem',['missing_bundle','broken_packet','oversize_bundle'])
def test_rejected_inputs_keep_schedule_coverage_without_numeric(synthetic,monkeypatch,tmp_path,problem):
    plan,auth,packet,row,_,clock,_,transport=fixture(synthetic,monkeypatch,root=tmp_path)
    bundle=pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    if problem=='missing_bundle':bundle=None
    elif problem=='broken_packet':packet['payload']['version']='unrecognized'
    else:bundle['payload']['objects'][0]['body_bytes']=custody.MAX_BODY_BYTES+1;bundle=pilot.seal(bundle['payload'])
    from app_core.ncaaf_schedule import inventory_from_events
    from test_ncaaf_schedule_coverage import event
    inv=inventory_from_events([('FBS',[event('9',row['game_start_utc'],'Alabama','Georgia')])],'2026-10-10','2026-10-10',complete=True,observed_at=previous.NOW.isoformat())
    numeric=Mock(side_effect=AssertionError('No numerical call'));monkeypatch.setattr(research,'centers',numeric)
    result=pilot.admitted_analysis(packet,row,inventory=inv,bundle=bundle)
    assert result.iloc[0].ml_inference_status=='unavailable' and not numeric.called
    retained=json.loads(result.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['pilot_capture']['payload']
    assert retained['status']=='rejected' and retained['reason'].startswith('NCAAF_')
    if problem=='missing_bundle':assert retained['reason']=='NCAAF_PILOT_BUNDLE_REQUIRED_OR_CORRUPT'
    assert len(result.attrs['slate_coverage']['decisions'])==1
    assert result.attrs['slate_coverage']['counts']['states']['UNVERIFIED']==1


@pytest.mark.parametrize('change',['budget','objects','attempts','endpoint','season','integer','credential'])
def test_invalid_plan_stops_without_request(synthetic,monkeypatch,tmp_path,change):
    plan,_,_,_,_,clock,calls,transport=fixture(synthetic,monkeypatch,root=tmp_path);p=plan['payload']
    if change=='budget':p['limits']['max_attempts']=1
    elif change=='objects':p['limits']['max_objects']=17
    elif change=='attempts':p['requests'][0]['max_attempts']=2
    elif change=='endpoint':p['requests'][0]['endpoint']='games/999/odds'
    elif change=='season':p['requests'][0]['params']['year']=2025
    elif change=='integer':p['offer']['signed_line']=-3
    else:p['requests'][0]['params']['apiKey']='SYNTHETIC_SECRET'
    plan=pilot.seal(p);auth=authorize(plan,monkeypatch)
    with pytest.raises(ValueError):pilot.acquire(plan,auth,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.monotonic)
    assert calls==[] and list(tmp_path.iterdir())==[]


def test_production_catalogs_empty():
    assert pilot.AUTHORIZED_COLLECTIONS==pilot.ACCEPTED_ADVANCE_PERMISSIONS=={}
    assert pilot.AUTHORIZED_CUSTODY_ROOTS=={}
    assert custody.ACCEPTED_ADMISSIONS==adapter.ACCEPTED_PACKETS=={}


def test_existing_entrypoints_are_not_isolated(monkeypatch,tmp_path):
    from app_core import football_stage1_cycle as stage
    calls=[]
    monkeypatch.setattr(stage.prospective_remote,'sync',lambda *a,**k:calls.append('DRIVE_SYNC') or {})
    monkeypatch.setattr(stage,'_nfl_schedule',lambda *a,**k:(calls.append('NFL_SCHEDULE') or [],[]))
    monkeypatch.setattr(stage,'_ncaaf_schedule',lambda *a,**k:(calls.append('NCAAF_SCHEDULE') or [],None,None,None,{}))
    monkeypatch.setattr(stage,'_odds',lambda sport,*a,**k:calls.append('ODDS_'+sport) or [])
    # Stop at the coverage boundary after each provider, rather than initialize
    # a real canonical store. The runner retains its failure, then next sport.
    monkeypatch.setattr(stage.foundation,'coverage',Mock(side_effect=stage.ProviderFailure('SYNTHETIC_STOP_BEFORE_STORE')))
    monkeypatch.setattr(stage,'_readiness',lambda *a,**k:{})
    monkeypatch.setattr(stage,'_lifecycle',lambda *a,**k:{})
    try:stage.run_cycle(tmp_path/'never-created.sqlite3','SYNTHETIC',object(),'SYNTHETIC','SYNTHETIC',now=previous.NOW,get=Mock())
    except (ValueError,KeyError,TypeError):pass
    assert calls[:3]==['DRIVE_SYNC','NFL_SCHEDULE','ODDS_NFL']
    assert 'NCAAF_SCHEDULE' in calls and 'ODDS_NCAAF' in calls
    assert not (tmp_path/'never-created.sqlite3').exists()


def test_generic_odds_client_retries_and_paginates_offline(monkeypatch):
    from app_core import odds_api
    calls=[]
    def get(url,**kwargs):
        calls.append(deepcopy(kwargs.get('params')))
        if len(calls)==1:return SimpleNamespace(status_code=429,text='SYNTHETIC',headers={},raise_for_status=lambda:None)
        return SimpleNamespace(status_code=200,headers={'x-next-page':'two'} if len(calls)==2 else {},text='[]',json=lambda:[],raise_for_status=lambda:None)
    monkeypatch.setattr(odds_api.requests,'get',get)
    monkeypatch.setattr('time.sleep',lambda *a:None)
    import builtins,io
    original_open=builtins.open
    monkeypatch.setattr(builtins,'open',lambda name,*a,**k:io.StringIO() if str(name).replace('\\','/').endswith('data/live_odds_debug.json') else original_open(name,*a,**k))
    client=odds_api.TheOddsAPIClient(api_key='SYNTHETIC')
    client.get_odds('americanfootball_ncaaf')
    assert len(calls)>=3 and any(c.get('cursor')=='two' for c in calls)
