"""Bounded fake release/asset transport. No scoring download or real sockets."""
from copy import deepcopy
import gzip
import hashlib
import json
import pytest
from app_core import nfl_scoring_acquisition as lane,nfl_scoring_profile as profile
from test_nfl_bounded_acquisition import Stream,NOW
from test_nfl_scoring_profile import sized_csv
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def fixture(monkeypatch,root):
    permission=dict(reference='SYNTHETIC-NFL-advance',reviewer='Robert Velarde',reviewed_at='2026-10-17T00:00:00Z',
        effective_from='2026-10-17T00:00:00Z',effective_until='2026-10-19T00:00:00Z',source_sha256=lane.digest(profile.SOURCE),
        private_retention=True,private_research_use=True,basis='SYNTHETIC reviewed CC BY private scope; not authentic permission',
        requests_sha256=lane.digest(dict(primary=[lane.PRIMARY],redirect=lane.REDIRECT)))
    plan=lane.seal(dict(version=lane.VERSION,profile=profile.PROFILE,evidence_label='SYNTHETIC',source=deepcopy(profile.SOURCE),
        target=dict(season=2026,purpose='prospective_original_scoring_snapshot'),requests=[deepcopy(lane.PRIMARY)],redirect=deepcopy(lane.REDIRECT),
        limits=dict(max_attempts=2,max_objects=1,max_wire_bytes=profile.MAX_WIRE,max_body_bytes=profile.MAX_BODY,max_aggregate_bytes=profile.MAX_TOTAL,
            max_packet_bytes=profile.MAX_PACKET,max_header_bytes=profile.MAX_HEADER_BYTES,connect_seconds=10,read_seconds=10,total_seconds=60),
        window=dict(not_before=NOW,not_after='2026-10-18T15:31:00Z',expires_at='2026-10-18T15:31:30Z'),permission=permission,
        authorization_ref='SYNTHETIC-NFL-collection',custody_id='SYNTHETIC-NFL-private',unresolved=[]))
    auth=lane.seal(dict(version=lane.AUTH_VERSION,reference=plan['payload']['authorization_ref'],plan_sha256=plan['sha256'],custody_id=plan['payload']['custody_id'],
        owner='Robert Velarde',authorized_at='2026-10-17T00:00:00Z',expires_at=plan['payload']['window']['expires_at'],maximum_additional_spend=0))
    monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{auth['payload']['reference']:auth['sha256']})
    monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{permission['reference']:lane.digest(permission)})
    monkeypatch.setattr(lane,'AUTHORIZED_CUSTODY_ROOTS',{plan['payload']['custody_id']:str(root)})
    return plan,auth


def location(path=None,host=None,filename='games.csv'):
    return 'https://'+(host or lane.REDIRECT['host'])+(path or lane.REDIRECT['repository_path']+'12345678-1234-1234-1234-123456789abc')+'?sig=SYNTHETIC-SIGNED-URL&response-content-disposition=attachment%3B%20filename%3D'+filename


def test_planning_does_not_inspect_credentials_security_or_stores(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(lane,'authorize',lambda *a,**k:pytest.fail('authorization/storage inspection during planning'))
    monkeypatch.setattr(lane,'HttpsTransport',lambda:pytest.fail('transport during planning'))
    report=lane.planning(p)
    assert report['network_requests']==report['storage_operations']==report['credentials']==0
    assert report['requests_min']==1 and report['requests_max']==2
    assert report['odds_credits']==report['cfbd_requests']==report['maximum_additional_spend']==0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('redirect',[False,True])
def test_exact_requests_custody_and_spent_attempt_readback(monkeypatch,tmp_path,redirect):
    p,a=fixture(monkeypatch,tmp_path);raw=sized_csv(2183999);calls=[];streams=[]
    def transport(request,*args):
        calls.append(request)
        stream=Stream(b'',status=302,headers={'location':location()}) if redirect and len(calls)==1 else Stream(raw)
        streams.append(stream);return stream
    result=lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW)
    assert result['payload']['status']=='CAPTURED_UNREVIEWED' and result['payload']['attempted']==len(calls)==(2 if redirect else 1)
    obj=result['payload']['object'];assert lane.custody.read_score(obj)[0]==raw
    assert all(s.closed for s in streams) and obj['receipt']['observed_at']==NOW
    assert obj['transfer']['requests'][0]['request']==lane.PRIMARY
    if redirect:lane.validate_hop_descriptor(obj['transfer']['requests'][1]['request'])
    serialized=lane.custody.encode(result)
    assert 'SYNTHETIC-SIGNED-URL' not in serialized and 'sig=' not in serialized
    assert not result['payload']['owner_reviewed'] and not result['payload']['source_acceptance'] and not result['payload']['inference']
    assert json.loads((tmp_path/(p['sha256']+'.nfl-scoring-unreviewed.json')).read_bytes())==result
    journal=(tmp_path/(p['sha256']+'.nfl-scoring-attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):
        lane.acquire(p,a,root=tmp_path,transport=transport,wall=lambda:NOW)
    assert journal==(tmp_path/(p['sha256']+'.nfl-scoring-attempts.jsonl')).read_bytes() and len(calls)==(2 if redirect else 1)


@pytest.mark.parametrize('mutation',['host','path','port','user','query','filename','nested','missing_signature'])
def test_arbitrary_redirect_stops_before_second_attempt(monkeypatch,tmp_path,mutation):
    p,a=fixture(monkeypatch,tmp_path);value=location();calls=[]
    if mutation=='host':value=location(host='release-assets.githubusercontent.com.evil.invalid')
    elif mutation=='path':value=location(path='/github-production-release-asset/123/12345678-1234-1234-1234-123456789abc')
    elif mutation=='port':value=value.replace('.com/','.com:444/')
    elif mutation=='user':value=value.replace('https://','https://user@')
    elif mutation=='query':value+='&api_key=SYNTHETIC-CANARY'
    elif mutation=='filename':value=location(filename='other.csv')
    elif mutation=='nested':value=location(path='/github-production-release-asset/452908115/../other')
    elif mutation=='missing_signature':value=value.replace('sig=SYNTHETIC-SIGNED-URL&','')
    def fake(*args):calls.append(args);return Stream(b'',302,{'location':value})
    result=lane.acquire(p,a,root=tmp_path,transport=fake,wall=lambda:NOW)
    assert result['payload']['status']=='INCOMPLETE' and result['payload']['attempted']==len(calls)==1 and result['payload']['object'] is None
    assert 'SYNTHETIC-CANARY' not in lane.custody.encode(result)


@pytest.mark.parametrize('failure',['wire_over','decoded_over','incomplete','gzip_incomplete','second_redirect','pagination','rate','auth','timeout','interrupt','secret','headers','slow'])
def test_stops_bounded_without_retry(monkeypatch,tmp_path,failure):
    p,a=fixture(monkeypatch,tmp_path);calls=[];stream=Stream(sized_csv(1024))
    if failure=='wire_over':stream=Stream(sized_csv(profile.MAX_WIRE+1),headers={'content-type':'text/csv'})
    elif failure=='decoded_over':stream=Stream(gzip.compress(sized_csv(profile.MAX_BODY+1)),headers={'content-type':'text/csv','content-encoding':'gzip'})
    elif failure=='incomplete':stream.headers['content-length']='2048'
    elif failure=='gzip_incomplete':stream=Stream(gzip.compress(sized_csv(1024))[:-1],headers={'content-type':'text/csv','content-encoding':'gzip'})
    elif failure=='pagination':stream.headers['x-next-page']='2'
    elif failure=='rate':stream.status=429
    elif failure=='auth':stream.status=401
    elif failure=='secret':stream=Stream(b'api_key="SYNTHETIC-CANARY"\n')
    elif failure=='headers':stream.headers['x-large']='x'*(profile.MAX_HEADER_BYTES+1)
    ticks=iter([0,0,61,61]) if failure=='slow' else None
    def fake(request,*args):
        calls.append(request)
        if failure in {'timeout','interrupt'}:raise TimeoutError('SYNTHETIC private failure never logged')
        if failure=='second_redirect':return Stream(b'',302,{'location':location()})
        return stream
    result=lane.acquire(p,a,root=tmp_path,transport=fake,wall=lambda:NOW,monotonic=(lambda:next(ticks,61)) if ticks else lane.time.monotonic)
    assert result['payload']['status']=='INCOMPLETE' and result['payload']['attempted']==len(calls)<=2
    assert result['payload']['object'] is None and 'SYNTHETIC-CANARY' not in lane.custody.encode(result)
    before=len(calls)
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):lane.acquire(p,a,root=tmp_path,transport=fake,wall=lambda:NOW)
    assert len(calls)==before


@pytest.mark.parametrize('encoding',['identity','gzip'])
def test_exact_wire_decoded_limits(monkeypatch,tmp_path,encoding):
    p,a=fixture(monkeypatch,tmp_path);raw=sized_csv(profile.MAX_BODY)
    wire=gzip.compress(raw) if encoding=='gzip' else raw
    decoded,measured=lane.read(Stream(wire,headers={'content-type':'application/octet-stream','content-encoding':encoding,'content-length':str(len(wire))}),p['payload']['limits'],60,lambda:0)
    assert decoded==raw and measured['wire_bytes']==len(wire) and measured['wire_sha256']==hashlib.sha256(wire).hexdigest()


@pytest.mark.parametrize('failure',['plan','source','authorization','expired','permission','custody'])
def test_pre_request_boundaries(monkeypatch,tmp_path,failure):
    p,a=fixture(monkeypatch,tmp_path)
    if failure=='plan':p['payload']['requests'][0]['host']='evil.invalid'
    elif failure=='source':p['payload']['source']['repository']='nflverse/nfldata';p['sha256']=lane.digest(p['payload'])
    elif failure=='authorization':monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{})
    elif failure=='expired':p['payload']['window']['expires_at']='2026-10-17T00:00:00Z';p['sha256']=lane.digest(p['payload'])
    elif failure=='permission':monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{})
    elif failure=='custody':monkeypatch.setattr(lane,'AUTHORIZED_CUSTODY_ROOTS',{})
    with pytest.raises(ValueError):lane.acquire(p,a,root=tmp_path,transport=lambda *a:pytest.fail('request forbidden'),wall=lambda:NOW)
    assert not list(tmp_path.iterdir())


def test_controlled_bindings_explicit_private_and_nonpersistent(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{});monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{});monkeypatch.setattr(lane,'AUTHORIZED_CUSTODY_ROOTS',{})
    value=dict(version=lane.CONFIG_VERSION,plan_sha256=p['sha256'],permission=p['payload']['permission'],authorization=a,
        custody=dict(id=p['payload']['custody_id'],root=str(tmp_path),storage_receipt_sha256='0'*64))
    value['review']=dict(reviewer='Robert Velarde',reviewed_at='2026-10-17T00:00:00Z',subject_sha256=lane.digest(value),status='OWNER_REVIEWED',scope='NFL scoring acquisition bindings only; no source acceptance.')
    raw=lane.custody.encode(lane.seal(value)).encode();expected=hashlib.sha256(raw).hexdigest()
    secure=lambda root:dict(status='VERIFIED',private_acl=True,encrypted=True,no_reparse=True)
    with lane.controlled(p,raw,expected,root=tmp_path,wall=lambda:NOW,security=secure):assert lane.authorize(p,a,NOW,tmp_path)
    assert lane.CONTROLLED.get() is None and lane.AUTHORIZED_COLLECTIONS==lane.REVIEWED_PERMISSIONS==lane.AUTHORIZED_CUSTODY_ROOTS=={}
    with pytest.raises(ValueError,match='NFL_ACQUIRE_CONFIGURATION_INTEGRITY'):
        with lane.controlled(p,raw+b' ',expected,root=tmp_path,wall=lambda:NOW,security=secure):pytest.fail('tampering accepted')
    with pytest.raises(ValueError,match='NFL_ACQUIRE_CUSTODY_UNVERIFIED'):
        with lane.controlled(p,raw,expected,root=tmp_path,wall=lambda:NOW,security=lambda r:{'status':'UNKNOWN'}):pytest.fail('security unknown accepted')


def test_all_production_catalogs_and_old_limits_remain_empty_unchanged():
    from app_core import nfl_bounded_acquisition as old,nfl_observed_custody as old_custody,ncaaf_pilot
    assert lane.AUTHORIZED_COLLECTIONS==lane.REVIEWED_PERMISSIONS==lane.AUTHORIZED_CUSTODY_ROOTS=={}
    assert old.AUTHORIZED_COLLECTIONS==old.REVIEWED_PERMISSIONS==old.AUTHORIZED_CUSTODY_ROOTS=={}
    assert old_custody.MAX_BODY==ncaaf_pilot.MAX_WIRE==512*1024 and old_custody.MAX_TOTAL==2*1024*1024


@pytest.mark.parametrize('failure',[None,'duplicate','header_limit','sas_https','cookies'])
def test_actual_https_seam_manual_asset_hop_and_header_conflicts(monkeypatch,tmp_path,failure):
    p,a=fixture(monkeypatch,tmp_path);calls=[];raw=sized_csv(1024)
    class Response:
        def __init__(self,status,headers,body):self.status=status;self._headers=headers;self.body=body
        def getheaders(self):return self._headers
        def read1(self,size):block=self.body[:size];self.body=self.body[size:];return block
    responses=[Response(302,[('Location',location())],b''),Response(200,[('Content-Type','text/csv'),('Content-Length',str(len(raw)))],raw)]
    if failure=='duplicate':responses[0]._headers.append(('location',location(host='evil.invalid')))
    if failure=='header_limit':responses[0]._headers.append(('X-Large','x'*(profile.MAX_HEADER_BYTES+1)))
    if failure=='sas_https':responses[0]._headers=[('Location',location()+'&spr=https')]
    if failure=='cookies':responses[0]._headers.extend([('Set-Cookie','SYNTHETIC-COOKIE-1'),('Set-Cookie','SYNTHETIC-COOKIE-2')])
    class Sock:
        def settimeout(self,value):assert 0<value<=10
    class Connection:
        def __init__(self,host,timeout):self.host=host;self.sock=Sock();self.auto_open=True;self.closed=False
        def connect(self):assert self.auto_open is False
        def request(self,method,url,headers):calls.append((self.host,method,url,headers))
        def getresponse(self):return responses.pop(0)
        def close(self):self.closed=True
    connections=[]
    def connect(*a,**k):c=Connection(*a,**k);connections.append(c);return c
    monkeypatch.setattr(lane.http.client,'HTTPSConnection',connect)
    result=lane.acquire(p,a,root=tmp_path,transport=lane.HttpsTransport(),wall=lambda:NOW)
    if failure in {'duplicate','header_limit'}:
        assert result['payload']['status']=='INCOMPLETE' and result['payload']['attempted']==len(calls)==1
    else:
        assert result['payload']['status']=='CAPTURED_UNREVIEWED' and result['payload']['attempted']==len(calls)==2
        assert calls[0][:3]==('github.com','GET',lane.PRIMARY['endpoint'])
        assert calls[1][0]=='release-assets.githubusercontent.com'
        assert lane.custody.read_score(result['payload']['object'])[0]==raw
        assert 'SYNTHETIC-SIGNED-URL' not in lane.custody.encode(result)
        assert 'SYNTHETIC-COOKIE' not in lane.custody.encode(result)
        assert all(c[3]=={'Accept':'text/csv','Accept-Encoding':'identity'} for c in calls)
    assert all(c.closed for c in connections)


def test_proposal_and_unreviewed_configuration_cannot_execute(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path);p['payload']['evidence_label']='PROPOSAL';p=lane.seal(p['payload'])
    with pytest.raises(ValueError,match='NFL_ACQUIRE_PREREQUISITES'):
        lane.acquire(p,a,root=tmp_path,transport=lambda *a:pytest.fail('proposal request'),wall=lambda:NOW)
    assert not list(tmp_path.iterdir())


def test_cli_planning_never_constructs_transport(monkeypatch,tmp_path,capsys):
    from scripts.nfl_scoring_acquisition import main
    p,a=fixture(monkeypatch,tmp_path);path=tmp_path/'SYNTHETIC-plan.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(lane,'HttpsTransport',lambda:pytest.fail('planning created transport'))
    monkeypatch.setattr(lane,'controlled',lambda *a,**k:pytest.fail('planning loaded runtime configuration'))
    assert main([str(path)])==0 and json.loads(capsys.readouterr().out)['network_requests']==0
    assert list(tmp_path.iterdir())==[path]


def test_process_interruption_keeps_spent_attempt_without_resume(monkeypatch,tmp_path):
    p,a=fixture(monkeypatch,tmp_path);calls=[]
    def interrupted(*args):calls.append(args);raise KeyboardInterrupt('SYNTHETIC process interruption')
    with pytest.raises(KeyboardInterrupt):lane.acquire(p,a,root=tmp_path,transport=interrupted,wall=lambda:NOW)
    journal=tmp_path/(p['sha256']+'.nfl-scoring-attempts.jsonl')
    before=journal.read_bytes();assert len(calls)==1 and json.loads(before.splitlines()[-1])['attempt']==1
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):
        lane.acquire(p,a,root=tmp_path,transport=interrupted,wall=lambda:NOW)
    assert before==journal.read_bytes() and len(calls)==1


@pytest.mark.parametrize('protocol',['http','https,http'])
def test_sas_protocol_and_framing_cannot_be_downgraded(protocol):
    with pytest.raises(ValueError,match='NFL_ACQUIRE_REDIRECT_NOT_APPROVED'):lane.redirect_request(location()+'&spr='+protocol)
    with pytest.raises(ValueError,match='NFL_ACQUIRE_HEADER_CONFLICT'):
        lane.checked_headers([('Content-Length','1024'),('Transfer-Encoding','chunked')])
    with pytest.raises(ValueError,match='NFL_ACQUIRE_HEADER_CONFLICT'):
        lane.checked_headers([('Content-Length','1024'),('content-length','1023')])
