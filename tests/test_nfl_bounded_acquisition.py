"""Labelled SYNTHETIC transport; no real acquisition or credential access."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pytest
from app_core import nfl_bounded_acquisition as lane
from scripts.benchmark_drive_history_loading import blocked_network

NOW='2026-10-18T15:30:00Z'
BODY=b'game_id,season,gameday,home_team,away_team,home_score,away_score,result\nSYNTHETIC,2026,2026-10-01,DAL,BUF,21,14,7\n'


@pytest.fixture(autouse=True)
def offline():
    with blocked_network(): yield


def fixture(monkeypatch,tmp_path):
    permission=dict(reference='SYNTHETIC-advance-permission',reviewer='Robert Velarde',reviewed_at='2026-10-17T00:00:00Z',
        effective_from='2026-10-17T00:00:00Z',effective_until='2026-10-19T00:00:00Z',endpoint=lane.ENDPOINT,
        private_retention=True,private_research_use=True,basis='SYNTHETIC reviewed scope, not authentic permission',requests_sha256=lane.digest([lane.REQUEST]))
    p=dict(version=lane.VERSION,evidence_label='SYNTHETIC',target=dict(canonical_event_id='SYNTHETIC-event',season=2026),requests=[deepcopy(lane.REQUEST)],
        limits=dict(max_attempts=1,max_objects=1,max_body_bytes=512*1024,max_aggregate_bytes=2*1024*1024,connect_seconds=10,read_seconds=10,total_seconds=30),
        window=dict(not_before=NOW,not_after='2026-10-18T15:31:00Z',expires_at='2026-10-18T15:31:30Z'),permission=permission,
        authorization_ref='SYNTHETIC-owner-authorization',custody_id='SYNTHETIC-private-root',unresolved=[])
    plan=lane.seal(p)
    authorization=lane.seal(dict(version=lane.AUTH_VERSION,reference=p['authorization_ref'],plan_sha256=plan['sha256'],custody_id=p['custody_id'],
        owner='Robert Velarde',authorized_at='2026-10-17T00:00:00Z',expires_at=p['window']['expires_at'],maximum_additional_spend=0))
    monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{p['authorization_ref']:authorization['sha256']})
    monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{permission['reference']:lane.digest(permission)})
    monkeypatch.setattr(lane,'AUTHORIZED_CUSTODY_ROOTS',{p['custody_id']:str(tmp_path)})
    return plan,authorization


class Stream:
    def __init__(self,body=BODY,status=200,headers=None):
        self.body=body;self.status=status;self.headers=headers or {'content-type':'text/csv','content-length':str(len(body))};self.closed=False
    def chunks(self): yield self.body
    def close(self):self.closed=True


def test_planning_is_network_storage_and_credential_free(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(lane,'HttpsTransport',lambda:pytest.fail('transport during planning'))
    before=list(tmp_path.iterdir());status=lane.planning(p)
    assert status['network_requests']==status['credentials']==status['storage_operations']==0
    assert not status['accepted'] and not status['inference'] and list(tmp_path.iterdir())==before


def test_one_exact_attempt_original_bytes_and_immutable_readback(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path);calls=[];stream=Stream()
    def transport(request,limits,deadline,monotonic):calls.append(request);return stream
    bundle=lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW)
    assert calls==[lane.REQUEST] and stream.closed
    assert bundle['payload']['status']=='CAPTURED_UNREVIEWED' and bundle['payload']['attempted']==1
    obj=bundle['payload']['object']
    assert lane.custody.read(obj)[0]==BODY and obj['receipt']['observed_at']==NOW
    assert obj['receipt']['provider_clock_meaning']=='publisher_availability_unknown'
    assert json.loads((tmp_path/(p['sha256']+'.nfl-unreviewed.json')).read_text(encoding='utf-8'))==bundle
    journal=(tmp_path/(p['sha256']+'.nfl-attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):
        lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW)
    assert calls==[lane.REQUEST] and journal==(tmp_path/(p['sha256']+'.nfl-attempts.jsonl')).read_bytes()
    assert not bundle['payload']['owner_reviewed'] and not bundle['payload']['inference'] and not bundle['payload']['wagering_authority']


@pytest.mark.parametrize('failure',['authorization','window','permission','custody','plan'])
def test_pre_request_gates(monkeypatch,tmp_path,failure):
    p,a=fixture(monkeypatch,tmp_path)
    if failure=='authorization':monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{})
    elif failure=='window':p['payload']['window']['not_before']='2026-10-19T00:00:00Z';p['sha256']=lane.digest(p['payload'])
    elif failure=='permission':monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{})
    elif failure=='custody':monkeypatch.setattr(lane,'AUTHORIZED_CUSTODY_ROOTS',{})
    elif failure=='plan':p['payload']['requests'][0]['host']='evil.invalid'
    with pytest.raises(ValueError):lane.acquire(p,a,root=tmp_path,transport=lambda *args:pytest.fail('request forbidden'),wall=lambda:NOW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure',['redirect','pagination','rate','oversize','incomplete','content','secret','timeout','interrupted','gzip','slow'])
def test_single_spent_attempt_stops_without_retry(monkeypatch,tmp_path,failure):
    p,a=fixture(monkeypatch,tmp_path);calls=[]
    stream=Stream()
    if failure=='redirect':stream.status=302;stream.headers['location']='https://evil.invalid'
    elif failure=='pagination':stream.headers['x-next-page']='2'
    elif failure=='rate':stream.status=429
    elif failure=='oversize':stream=Stream(b'x'*(512*1024+1))
    elif failure=='incomplete':stream.headers['content-length']=str(len(BODY)+1)
    elif failure=='content':stream.headers['content-type']='text/html'
    elif failure=='secret':stream=Stream(b'api_key="SYNTHETIC-CANARY"\n')
    elif failure=='gzip':stream=Stream(gzip.compress(b'x'*(512*1024+1)),headers={'content-type':'text/csv','content-encoding':'gzip'})
    ticks=iter([0,0,0,31,31]) if failure=='slow' else None
    def transport(*args):
        calls.append(args[0])
        if failure in {'timeout','interrupted'}:raise TimeoutError('SYNTHETIC secret exception never retained')
        return stream
    bundle=lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW,monotonic=(lambda:next(ticks,31)) if ticks else lane.time.monotonic)
    assert bundle['payload']['status']=='INCOMPLETE' and bundle['payload']['attempted']==1 and len(calls)==1
    assert bundle['payload']['object'] is None
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW)
    assert len(calls)==1 and 'SYNTHETIC-CANARY' not in lane.custody.encode(bundle)


def test_immutable_ncaaf_limits_and_empty_production_catalogs():
    from app_core import ncaaf_pilot, nfl_response_custody
    assert lane.custody.MAX_BODY==nfl_response_custody.MAX_BODY==ncaaf_pilot.MAX_WIRE==512*1024
    assert lane.custody.MAX_OBJECTS==16 and lane.custody.MAX_TOTAL==2*1024*1024
    assert lane.AUTHORIZED_COLLECTIONS==lane.REVIEWED_PERMISSIONS==lane.AUTHORIZED_CUSTODY_ROOTS=={}


def test_actual_https_seam_fixed_request_no_library_download(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path);calls=[];instances=[]
    class Socket:
        def settimeout(self,n):assert 0<n<=10
    class Response:
        status=200
        def __init__(self):self.read=False
        def getheaders(self):return [('Content-Type','text/plain'),('Content-Length',str(len(BODY)))]
        def read1(self,n):
            assert n==8192
            if self.read:return b''
            self.read=True;return BODY
    class Connection:
        def __init__(self,host,timeout):assert host==lane.REQUEST['host'];self.sock=Socket();instances.append(self);self.closed=False
        def connect(self):calls.append('connect')
        def request(self,method,path,headers):
            calls.append((method,path,headers));assert self.auto_open is False
        def getresponse(self):return Response()
        def close(self):self.closed=True
    monkeypatch.setattr(lane.http.client,'HTTPSConnection',Connection)
    bundle=lane.acquire(p,a,root=tmp_path,transport=lane.HttpsTransport(),wall=lambda:NOW)
    assert bundle['payload']['status']=='CAPTURED_UNREVIEWED'
    assert calls==['connect',('GET',lane.REQUEST['endpoint'],{'Accept':'text/csv','Accept-Encoding':'identity'})]
    assert len(instances)==1 and instances[0].closed


def test_permission_secret_is_rejected_without_retention(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path)
    p['payload']['permission']['basis']='Bearer SYNTHETIC-CANARY'
    p['sha256']=lane.digest(p['payload'])
    with pytest.raises(ValueError,match='NFL_ACQUIRE_CREDENTIAL_FORBIDDEN'):lane.planning(p)
    assert not list(tmp_path.iterdir())


def test_existing_store_required_no_initialization(tmp_path):
    with pytest.raises(ValueError,match='NFL_ACQUIRE_EXISTING_PREDICTION_STORE_REQUIRED'):
        lane.retain_analysis(None,path=tmp_path/'never-created.sqlite3')
    assert not list(tmp_path.iterdir())
