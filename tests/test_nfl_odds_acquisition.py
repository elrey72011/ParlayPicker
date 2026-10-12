"""Labelled synthetic bodies and clocks. All external transports remain blocked."""
from copy import deepcopy
from functools import partial
import gzip
import hashlib
import http.client
import io
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
import pytest
from app_core import nfl_odds_acquisition as lane
from scripts import nfl_odds_acquisition as cli
from scripts.benchmark_drive_history_loading import blocked_network

NOW='2026-10-19T12:00:00Z'
SECRET='SYNTHETIC-PRIVATE-KEY'
CANARY='apiKey=SECRET https://example.invalid/?sig=secret Cookie: private'


@pytest.fixture(autouse=True)
def offline():
    with blocked_network(): yield


def fixture(root):
    request=dict(lane.SOURCE,params=dict(lane.PARAMS,commenceTimeFrom='2026-10-19T20:00:00Z',commenceTimeTo='2026-10-19T20:00:01Z'))
    permission=dict(reference='SYNTHETIC-permission',reviewer='Robert Velarde',reviewed_at='2026-10-18T12:00:00Z',effective_from='2026-10-18T12:00:00Z',
                    effective_until='2026-10-20T12:00:00Z',request_sha256=lane.digest(request),private_acquisition=True,private_retention=True,
                    basis='SYNTHETIC owner advance private acquisition/retention review, not authentic permission')
    target=dict(canonical_event_id='SYNTHETIC-2026-BUF-LAR',home='Los Angeles Rams',away='Buffalo Bills',kickoff='2026-10-19T20:00:00Z',season=2026,
                neutral_site=False,schedule_evidence_sha256='a'*64)
    plan=lane.seal(dict(version=lane.VERSION,evidence_label='SYNTHETIC',target=target,request=request,limits=deepcopy(lane.LIMITS),
        window=dict(not_before=NOW,not_after='2026-10-19T12:15:00Z',expires_at='2026-10-19T12:15:30Z'),permission=permission,
        authorization_ref='SYNTHETIC-collection',custody_id='SYNTHETIC-private',unresolved=[]))
    auth=lane.seal(dict(version=lane.AUTH_VERSION,reference='SYNTHETIC-collection',plan_sha256=plan['sha256'],custody_id='SYNTHETIC-private',owner='Robert Velarde',
        authorized_at='2026-10-18T12:00:00Z',expires_at=plan['payload']['window']['expires_at'],maximum_additional_spend=0,maximum_odds_credits=1))
    value=dict(version=lane.CONFIG_VERSION,plan_sha256=plan['sha256'],permission=permission,authorization=auth,
        custody=dict(id='SYNTHETIC-private',root=str(root),storage_receipt_sha256='b'*64),
        account=dict(license_holder='Robert Velarde',reviewer='Robert Velarde',verified_at='2026-10-19T11:00:00Z',valid_until='2026-10-19T13:00:00Z',
                     resets_at='2026-11-01T00:00:00Z',remaining_included_credits=1,entitlement_sha256='c'*64,no_paid_overage=True),
        target_review=dict(reviewer='Robert Velarde',reviewed_at='2026-10-18T12:00:00Z',target_sha256=lane.digest(target),conclusion='VERIFIED_NAMED_ORIENTATION_AND_KICKOFF',evidence_sha256='a'*64))
    return plan,auth,configuration(value)


def configuration(value):
    value=deepcopy(value); value.pop('review',None)
    value['review']=dict(reviewer='Robert Velarde',reviewed_at='2026-10-19T11:30:00Z',subject_sha256=lane.digest(value),status='OWNER_REVIEWED',scope='NFL odds acquisition only; no offer admission.')
    return lane.custody.encode(lane.seal(value)).encode()


def security(root): return dict(status='VERIFIED',private_acl=True,encrypted=True,no_reparse=True)


def body():
    return [dict(id='SYNTHETIC-ODDS-EVENT',sport_key='americanfootball_nfl',home_team='Los Angeles Rams',away_team='Buffalo Bills',commence_time='2026-10-19T20:00:00Z',
        bookmakers=[dict(key='novig',markets=[dict(key='spreads',last_update=NOW,outcomes=[dict(name='Los Angeles Rams',point=1.5,price=-110),dict(name='Buffalo Bills',point=-1.5,price=-110)])])])]


class Transport:
    def __init__(self,raw=None,status=200,headers=None,error=None):
        self.raw=json.dumps(body()).encode() if raw is None else raw;self.status=status;self.headers=dict(headers or {'content-type':'application/json'});self.error=error;self.calls=[];self.closed=False
    def __call__(self,*args):
        self.calls.append(args)
        if self.error: raise self.error
        return self
    def chunks(self):
        for start in range(0,len(self.raw),8192):yield self.raw[start:start+8192]
    def close(self):self.closed=True
    def validate_body(self,raw):
        t=lane.HttpsTransport(lambda:SECRET);t._secret=SECRET;t.validate_body(raw)


def run(root,transport,*,plan=None,raw=None,wall=lambda:NOW):
    p,a,c=fixture(root)
    if plan is not None:p=plan
    if raw is not None:c=raw
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=root,wall=wall,security=security) as approved:
        return lane.acquire(p,approved,root=root,transport=transport,wall=wall)


def test_prior_runners_reject_nfl_odds_intentionally(monkeypatch,tmp_path):
    from app_core import nfl_scoring_acquisition as scoring,ncaaf_pilot
    from test_nfl_scoring_acquisition import fixture as old_fixture
    p,a=old_fixture(monkeypatch,tmp_path)
    p['payload']['requests']=[dict(lane.SOURCE,params=lane.PARAMS)]
    p=lane.seal(p['payload'])
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ENDPOINT_NOT_APPROVED'):scoring.validate(p)
    requested=[]
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_SCOPE_CONFLICT'):
        ncaaf_pilot.HttpsTransport(lambda *a:requested.append(a))(dict(id='SYNTHETIC',provider='odds_api',host=lane.SOURCE['host'],endpoint=lane.SOURCE['endpoint'].lstrip('/'),params=lane.PARAMS),lane.LIMITS,time.monotonic()+60,time.monotonic)
    assert not requested


def test_capture_original_bytes_unreviewed_and_spent_readback(tmp_path):
    p,_,_=fixture(tmp_path);raw=b'  '+json.dumps(body(),indent=2).encode()+b'\n';t=Transport(raw)
    result=run(tmp_path,t);v=result['payload']
    assert v['status']=='CAPTURED_UNREVIEWED' and v['attempted']==len(t.calls)==1 and t.closed
    assert lane.custody.read(v['object'])[0]==raw and v['object']['body_sha256']==hashlib.sha256(raw).hexdigest()
    assert v['object']['receipt']['observed_at']==NOW and v['findings']['status']=='OBSERVED_UNREVIEWED'
    assert all(v[k] is False for k in ('owner_reviewed','source_acceptance','inference','public_export','qualified')) and v['action']=='PASS' and v['stake']==0
    journal=tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl');out=tmp_path/(p['sha256']+'.nfl-odds-unreviewed.json')
    assert lane.read_back(out,journal)==result
    before=journal.read_bytes()
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):run(tmp_path,t)
    assert journal.read_bytes()==before and len(t.calls)==1
    journal.write_bytes(before+b'{}\n')
    with pytest.raises(ValueError,match='NFL_ACQUIRE_PRIVATE_READBACK'):lane.read_back(out,journal)


@pytest.mark.parametrize('case,code',[
    ('missing','TARGET_MISSING'),('ambiguous','TARGET_AMBIGUOUS'),('sport','WRONG_SPORT'),('orientation','ORIENTATION_CONFLICT'),
    ('kickoff','KICKOFF_CONFLICT'),('no_book','NOVIG_SPREADS_ABSENT'),('no_spreads','NOVIG_SPREADS_ABSENT'),('duplicate_id','IDENTITY_CONFLICT')])
def test_unresolved_target_retains_original_response_and_spends_once(tmp_path,case,code):
    b=body()
    if case=='missing':b=[]
    elif case=='ambiguous':b.append(dict(deepcopy(b[0]),id='SYNTHETIC-SECOND'))
    elif case=='sport':b[0]['sport_key']='americanfootball_ncaaf'
    elif case=='orientation':b[0]['home_team'],b[0]['away_team']=b[0]['away_team'],b[0]['home_team']
    elif case=='kickoff':b[0]['commence_time']='2026-10-19T20:01:00Z'
    elif case=='no_book':b[0]['bookmakers']=[]
    elif case=='no_spreads':b[0]['bookmakers'][0]['markets']=[]
    elif case=='duplicate_id':b.append(deepcopy(b[0]))
    raw=json.dumps(b).encode();t=Transport(raw);v=run(tmp_path,t)['payload']
    assert v['status']=='INCOMPLETE' and v['reason']=='NFL_ACQUIRE_'+code and len(t.calls)==v['attempted']==1
    assert lane.custody.read(v['object'])[0]==raw


@pytest.mark.parametrize('case',['malformed','over','gzip_over','gzip_incomplete','incomplete','redirect','pagination','auth','rate','tls','timeout','interrupt','secret','escaped_secret','encoded_secret'])
def test_failures_stop_without_retry_or_sensitive_text(tmp_path,case):
    t=Transport()
    if case=='malformed':t.raw=b'not json'
    elif case=='over':t.raw=b' '* (512*1024+1)
    elif case.startswith('gzip'):
        t.raw=gzip.compress(b' '* (512*1024+1) if case=='gzip_over' else json.dumps(body()).encode())
        if case=='gzip_incomplete':t.raw=t.raw[:-1]
        t.headers['content-encoding']='gzip'
    elif case=='incomplete':t.headers['content-length']='123456'
    elif case=='redirect':t.status=302;t.headers['location']='https://example.invalid/?apiKey='+SECRET
    elif case=='pagination':t.headers['x-next-page']='2'
    elif case in {'auth','rate'}:t.status=401 if case=='auth' else 429
    elif case in {'tls','timeout','interrupt'}:t.error={'tls':ssl.SSLError(1,CANARY),'timeout':TimeoutError(CANARY),'interrupt':KeyboardInterrupt(CANARY)}[case]
    elif case=='secret':t.raw=json.dumps([SECRET]).encode()
    elif case=='escaped_secret':t.raw=b'["\\u0053YNTHETIC-PRIVATE-KEY"]'
    elif case=='encoded_secret':t.raw=b'["SYNTHETIC%2dPRIVATE%2dKEY"]'
    v=run(tmp_path,t)['payload']
    assert v['status']=='INCOMPLETE' and v['object'] is None and v['attempted']==len(t.calls)==1
    serialized=''.join(x.read_text() for x in tmp_path.glob('*.json*'))
    assert SECRET not in serialized and CANARY not in serialized and 'sig=secret' not in serialized and 'Cookie: private' not in serialized


@pytest.mark.parametrize('case',['auth','auth_expired','permission','permission_expired','account','balance','account_expired','target','storage','configuration','credential','proposal','window'])
def test_failed_gates_precede_first_request(tmp_path,case):
    p,a,c=fixture(tmp_path);v=json.loads(c)['payload'];t=Transport();wall=lambda:NOW
    if case=='auth':v['authorization']['payload']['owner']='Another owner';v['authorization']=lane.seal(v['authorization']['payload'])
    elif case=='auth_expired':v['authorization']['payload']['expires_at']=NOW;v['authorization']=lane.seal(v['authorization']['payload'])
    elif case=='permission':v['permission']['request_sha256']='0'*64;p['payload']['permission']=v['permission'];p=lane.seal(p['payload']);v['plan_sha256']=p['sha256']
    elif case=='permission_expired':v['permission']['effective_until']=NOW;p['payload']['permission']=v['permission'];p=lane.seal(p['payload']);v['plan_sha256']=p['sha256']
    elif case=='account':v['account']['no_paid_overage']=False
    elif case=='balance':v['account']['remaining_included_credits']=0
    elif case=='account_expired':v['account']['valid_until']=NOW
    elif case=='target':v['target_review']['target_sha256']='0'*64
    elif case=='storage':v['custody']['root']=str(tmp_path/'other')
    elif case=='configuration':v['version']='nfl-scoring-controlled-bindings-v1'
    elif case=='credential':v['account']['entitlement_sha256']='https://example.invalid/?apiKey='+SECRET
    elif case=='proposal':p['payload']['evidence_label']='PROPOSAL';p=lane.seal(p['payload']);v['plan_sha256']=p['sha256']
    elif case=='window':wall=lambda:'2026-10-19T12:16:00Z'
    c=configuration(v)
    with pytest.raises((ValueError,KeyError)):run(tmp_path,t,plan=p,raw=c,wall=wall)
    assert not t.calls and not list(tmp_path.glob('*.json*')) and lane.CONTROLLED.get() is None


@pytest.mark.parametrize('case',['extra_params','other_book','other_region','other_market','other_host','other_sport','long_limit','bad_orientation','unknown_team','no_schedule'])
def test_rehashed_plan_cannot_expand_scope(tmp_path,case):
    p,_,_=fixture(tmp_path);v=p['payload']
    if case=='extra_params':v['request']['params']['includeLinks']='true'
    elif case=='other_book':v['request']['params']['bookmakers']='novig,draftkings'
    elif case=='other_region':v['request']['params']['regions']='us'
    elif case=='other_market':v['request']['params']['markets']='totals'
    elif case=='other_host':v['request']['host']='example.invalid'
    elif case=='other_sport':v['request']['endpoint']=v['request']['endpoint'].replace('nfl','ncaaf')
    elif case=='long_limit':v['limits']['total_seconds']=61
    elif case=='bad_orientation':v['target']['home']=v['target']['away']
    elif case=='unknown_team':v['target']['home']='Los Angeles'
    elif case=='no_schedule':v['target']['schedule_evidence_sha256']='UNKNOWN'
    with pytest.raises(ValueError):lane.validate(lane.seal(v))


def test_exact_body_limit_and_compressed_success(tmp_path):
    for compressed in (False,True):
        root=tmp_path/str(compressed);root.mkdir();raw=json.dumps(body()).encode();raw+=b' '*(512*1024-len(raw))
        t=Transport(gzip.compress(raw) if compressed else raw,headers={'content-type':'application/json','content-encoding':'gzip' if compressed else 'identity'})
        v=run(root,t)['payload']
        assert v['status']=='CAPTURED_UNREVIEWED' and lane.custody.read(v['object'])[0]==raw


def synthetic_socket_boundary(monkeypatch,raw,*,phase=None,error=None,status=200):
    """Keep real HTTPSConnection.connect/request/getresponse/read1 and _timed."""
    calls=[]
    class Sock:
        def settimeout(self,value):pass
        def setsockopt(self,*args):pass
        def sendall(self,data):
            if phase=='REQUEST_SEND':raise error
            calls.append(data)
        def makefile(self,*args):
            if phase=='RESPONSE_RECEIVE':raise error
            class File(io.BytesIO):
                def read1(self,*a):
                    if phase=='BODY_READ':raise error
                    return super().read1(*a)
            return File(('HTTP/1.1 '+str(status)+' SYNTHETIC\r\nContent-Type: application/json\r\nContent-Length: '+str(len(raw))+'\r\nSet-Cookie: '+CANARY+'\r\n\r\n').encode()+raw)
        def close(self):pass
    def connection(*args,**kwargs):
        if phase=='HTTPS_CONNECT':raise error
        return Sock()
    monkeypatch.setattr(socket,'create_connection',connection)
    monkeypatch.setattr(ssl.SSLContext,'wrap_socket',lambda self,sock,**kw:sock)
    return calls


@pytest.mark.parametrize('phase,error',[(None,None),('HTTPS_CONNECT',ssl.SSLError(1,CANARY)),('HTTPS_CONNECT',TimeoutError(CANARY)),('REQUEST_SEND',BrokenPipeError(CANARY)),('RESPONSE_RECEIVE',http.client.RemoteDisconnected(CANARY)),('BODY_READ',TimeoutError(CANARY)),('HTTPS_CONNECT',ValueError('NFL_ACQUIRE_PRIVATESECRET'))])
def test_real_cli_transport_journal_custody_readback(monkeypatch,tmp_path,capsys,phase,error):
    from app_core import ncaaf_pilot_setup
    p,a,c=fixture(tmp_path)
    paths=[tmp_path/(k+'.json') for k in ('SYNTHETIC-plan','SYNTHETIC-auth','SYNTHETIC-config')]
    for path,raw in zip(paths,(json.dumps(p).encode(),json.dumps(a).encode(),c)):path.write_bytes(raw)
    calls=synthetic_socket_boundary(monkeypatch,json.dumps(body()).encode(),phase=phase,error=error)
    monkeypatch.setattr(cli,'verify_runtime',lambda:None) # Labelled synthetic CLI injection only.
    monkeypatch.setattr(cli,'private_credential',lambda:SECRET)
    monkeypatch.setattr(ncaaf_pilot_setup,'inspect_security',security)
    monkeypatch.setattr(lane,'controlled',partial(lane.controlled,wall=lambda:NOW))
    monkeypatch.setattr(lane,'acquire',partial(lane.acquire,wall=lambda:NOW))
    args=[str(paths[0]),'--execute','--authorization',str(paths[1]),'--custody-root',str(tmp_path),'--controlled-bindings',str(paths[2]),'--bindings-sha256',hashlib.sha256(c).hexdigest()]
    assert cli.main(args)==(1 if error else 0)
    report=json.loads(capsys.readouterr().out)
    bundle=lane.read_back(tmp_path/(p['sha256']+'.nfl-odds-unreviewed.json'),tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl'))
    assert report==lane.execution_report(bundle) and report['attempted']==1
    if error:assert report['failure_diagnostics']['phase']==phase
    else:
        assert len(calls)==1 and b'GET /v4/sports/americanfootball_nfl/odds?' in calls[0]
        assert lane.custody.read(bundle['payload']['object'])[0]==json.dumps(body()).encode()
    saved=(tmp_path/(p['sha256']+'.nfl-odds-unreviewed.json')).read_text()
    assert SECRET not in saved and CANARY not in saved
    assert 'PRIVATESECRET' not in saved and 'PRIVATESECRET' not in json.dumps(report)
    assert cli.main(args)==1 and json.loads(capsys.readouterr().out)['reason']=='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'


def test_offline_runtime_rejects_without_disabling_network_blockers(tmp_path,capsys):
    before=socket.create_connection
    assert cli.main([str(tmp_path/'not_loaded.json'),'--execute'])==1
    assert json.loads(capsys.readouterr().out)['reason']=='NFL_ACQUIRE_RUNTIME_UNSUPPORTED'
    assert socket.create_connection is before
    v=run(tmp_path,lane.HttpsTransport(lambda:SECRET))['payload']
    assert v['status']=='INCOMPLETE' and v['attempted']==1 and v['failure_diagnostics']['phase']=='HTTPS_CONNECT'
    assert v['failure_diagnostics']['http_status']=='UNKNOWN'
    command="import runpy,sys; m=runpy.run_path(sys.argv[1],run_name='SYNTHETIC_runtime_only'); m['verify_runtime'](); print('ISOLATED_RUNTIME_PASS_NO_CONNECTION')"
    out=subprocess.check_output([sys.executable,'-I','-B','-c',command,str(Path(cli.__file__).resolve())],text=True)
    assert out.strip()=='ISOLATED_RUNTIME_PASS_NO_CONNECTION'


def test_planning_loads_no_credentials_or_store(monkeypatch,tmp_path,capsys):
    p,_,_=fixture(tmp_path);p['payload']['evidence_label']='PROPOSAL';p['payload']['unresolved']=['No acquisition authorization'];p=lane.seal(p['payload'])
    path=tmp_path/'proposal.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(cli,'private_credential',lambda:pytest.fail('credentials in planning'))
    monkeypatch.setattr(lane,'controlled',lambda *a,**k:pytest.fail('storage in planning'))
    monkeypatch.setattr(lane,'HttpsTransport',lambda *a:pytest.fail('network in planning'))
    assert cli.main([str(path)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='NON_EXECUTABLE_PROPOSAL' and report['network_requests']==report['credentials']==report['storage_operations']==0
    assert list(tmp_path.iterdir())==[path]


def test_journal_precedes_dispatch_and_expiry_consumes_attempt_without_request(tmp_path):
    p,_,_=fixture(tmp_path)
    class AtJournal(Transport):
        def __call__(self,*args):
            journal=tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl')
            rows=[json.loads(line) for line in journal.read_bytes().splitlines()]
            assert rows[-1]['attempt']==1 and rows[-1]['request']==p['payload']['request']
            return super().__call__(*args)
    assert run(tmp_path,AtJournal())['payload']['status']=='CAPTURED_UNREVIEWED'
    other=tmp_path/'expiry';other.mkdir();ticks=iter([NOW]*4+['2026-10-19T12:16:00Z'])
    t=Transport();v=run(other,t,wall=lambda:next(ticks,'2026-10-19T12:16:00Z'))['payload']
    assert v['status']=='INCOMPLETE' and v['attempted']==1 and not t.calls


def test_deadline_in_real_timed_worker_stops_and_closes(monkeypatch,tmp_path):
    # Millisecond synthetic deadlines exercise the existing worker, not 60s sleeps.
    from app_core.ncaaf_pilot import _timed
    import threading
    cancelled=threading.Event();closed=[]
    with pytest.raises(ValueError,match='NCAAF_PILOT_DEADLINE'):
        _timed(lambda:time.sleep(.03),time.monotonic()+.005,time.monotonic,cancelled,lambda:closed.append(True))
    assert cancelled.is_set() and closed


def test_unaccepted_bundle_cannot_borrow_offer_or_scenario_admission(tmp_path):
    from app_core import nfl_scoring_research,nfl_scenario_research
    result=run(tmp_path,Transport())
    assert lane.AUTHORIZED_COLLECTIONS==lane.REVIEWED_PERMISSIONS==lane.AUTHORIZED_CUSTODY_ROOTS=={}
    with pytest.raises((ValueError,KeyError,TypeError)):
        nfl_scoring_research.checked(result,NOW)
    with pytest.raises((ValueError,KeyError,TypeError)):
        nfl_scenario_research.checked(result,NOW)


@pytest.mark.parametrize('phase',['validation','matching'])
def test_shared_deadline_also_bounds_post_read_validation(monkeypatch,tmp_path,phase):
    class Timer:
        value=0
        def __call__(self):return self.value
    timer=Timer();t=Transport();p,a,c=fixture(tmp_path)
    if phase=='validation':
        old=t.validate_body
        def slow(raw):old(raw);timer.value=61
        t.validate_body=slow
    else:
        old=lane.target_findings
        def slow(*a):value=old(*a);timer.value=61;return value
        monkeypatch.setattr(lane,'target_findings',slow)
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security) as approved:
        result=lane.acquire(p,approved,root=tmp_path,transport=t,wall=lambda:NOW,monotonic=timer)
    assert result['payload']['status']=='INCOMPLETE' and result['payload']['reason']=='NFL_ACQUIRE_DEADLINE'
    assert result['payload']['attempted']==len(t.calls)==1
