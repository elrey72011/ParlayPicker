"""Labelled synthetic transport boundaries only; application network is blocked."""
from copy import deepcopy
from functools import partial
import gzip
import hashlib
import http.client
import io
import json
import socket
import ssl
import pytest
from app_core import nfl_document_acquisition as lane
from scripts import nfl_document_acquisition as cli
from scripts.benchmark_drive_history_loading import blocked_network

NOW='2026-10-06T19:25:02Z'
URL='https://ludlow-filings.s3.us-east-1.amazonaws.com/SYNTHETIC.pdf'
HOP='https://novig.intercom-attachments-1.com/SYNTHETIC.pdf'
CANARY='Cookie: SECRET apiKey=SECRET https://example.invalid/?sig=SECRET'

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def fixture(root,*,format='pdf',redirect=False):
    docs=[dict(id='rules',source_id='SYNTHETIC-rules',url=URL,redirects=[HOP] if redirect else [],format=format,
        purpose='SYNTHETIC original governing document, no listing applicability inferred',destination='rules.original')]
    permission=dict(reviewer='Robert Velarde',reviewed_at='2026-10-05T12:00:00Z',effective_from='2026-10-05T00:00:00Z',effective_until='2026-10-07T00:00:00Z',
        documents_sha256=lane.digest(docs),private_acquisition=True,private_retention=True,basis='SYNTHETIC advance terms and permissions reviewed; fixture only')
    plan=lane.seal(dict(version=lane.VERSION,evidence_label='SYNTHETIC',documents=docs,limits=deepcopy(lane.LIMITS),
        window=dict(not_before=NOW,not_after='2026-10-06T19:25:02.100Z',expires_at='2026-10-06T19:25:02.150Z'),permission=permission,
        authorization_ref='SYNTHETIC-auth',custody_id='SYNTHETIC-private',unresolved=[]))
    auth=lane.seal(dict(version=lane.AUTH,reference='SYNTHETIC-auth',plan_sha256=plan['sha256'],custody_id='SYNTHETIC-private',owner='Robert Velarde',
        authorized_at='2026-10-06T19:00:00Z',expires_at=plan['payload']['window']['expires_at'],maximum_attempts=2 if redirect else 1,maximum_additional_spend=0))
    c=dict(version=lane.CONFIG,plan_sha256=plan['sha256'],permission=permission,authorization=auth,
        custody=dict(id='SYNTHETIC-private',root=str(root),storage_receipt_sha256='a'*64))
    c['review']=dict(reviewer='Robert Velarde',reviewed_at='2026-10-06T19:24:00Z',subject_sha256=lane.digest(c),status='OWNER_REVIEWED',basis='SYNTHETIC actual human configuration-review fixture only')
    return plan,auth,lane.custody.encode(lane.seal(c)).encode()


def security(root):return dict(status='VERIFIED',private_acl=True,encrypted=True,no_reparse=True)


class Stream:
    def __init__(self,raw,status=200,headers=None,error=None):
        self.raw=raw;self.status=status;self.headers=headers or [('content-type','application/pdf'),('content-length',str(len(raw)))];self.error=error;self.closed=False
    def chunks(self):
        if self.error:raise self.error
        yield self.raw
    def close(self):self.closed=True


def run(root,stream,*,redirect=False,format='pdf'):
    p,a,c=fixture(root,redirect=redirect,format=format);calls=[]
    def transport(url,*args):
        journal=root/(p['sha256']+'.documents-attempts.jsonl')
        assert len(journal.read_bytes().splitlines())==len(calls)+1
        calls.append(url)
        if redirect and len(calls)==1:return Stream(b'',302,[('location',HOP)])
        return stream
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=root,wall=lambda:NOW,security=security):
        result=lane.acquire(p,a,root=root,transport=transport,wall=lambda:NOW)
    return result,p,calls


def pdf():
    from test_nfl_pdf_evidence import make_pdf
    return make_pdf('SYNTHETIC realistic original terms: full game includes overtime unless the listing overrides it.')


def test_old_runner_rejects_document_urls_before_transport():
    from app_core import nfl_odds_acquisition as old
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ENDPOINT_NOT_APPROVED'):
        old.check_request(dict(old.SOURCE,host='ludlow-filings.s3.us-east-1.amazonaws.com',endpoint='/NFL-spread-anchor-2.0-a.pdf'))


@pytest.mark.parametrize('format,raw,mime',[('pdf',None,'application/pdf'),('html',b'<!DOCTYPE html><html><body>SYNTHETIC authoritative prose</body></html>','text/html'),('text',b'SYNTHETIC full game includes overtime.','text/plain')])
@pytest.mark.parametrize('compressed,redirect',[(False,False),(True,False),(False,True)])
def test_complete_originals_redirects_and_readback(tmp_path,format,raw,mime,compressed,redirect):
    raw=pdf() if raw is None else raw;wire=gzip.compress(raw) if compressed else raw
    headers=[('content-type',mime),('content-length',str(len(wire)))]+([('content-encoding','gzip')] if compressed else [])
    stream=Stream(wire,headers=headers);result,p,calls=run(tmp_path,stream,redirect=redirect,format=format)
    assert result['payload']['status']=='CAPTURED_UNREVIEWED' and len(calls)==result['payload']['attempted']==1+int(redirect) and stream.closed
    assert lane.read_bundle(result)['rules'][0]==raw
    obj=result['payload']['objects'][0];assert obj['wire_sha256']==hashlib.sha256(wire).hexdigest() and obj['body_sha256']==hashlib.sha256(raw).hexdigest()
    assert obj['receipt']['observed_at']==NOW and obj['response_metadata']['publisher_available_at']=='UNKNOWN'
    saved=lane.read_back(tmp_path/(p['sha256']+'.documents-unreviewed.json'),tmp_path/(p['sha256']+'.documents-attempts.jsonl'));assert saved==result
    projected=lane.project(result,'rules',prepared_at='2026-10-06T19:25:02.200Z')
    if format=='pdf':
        from app_core.nfl_pdf_evidence import read_pdf
        assert read_pdf(projected)[0]==raw
    else:assert lane.custody.read(projected)[0]==raw
    before=(tmp_path/(p['sha256']+'.documents-attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='DOCUMENT_SPENT_NO_RESUME'):run(tmp_path,stream,redirect=redirect,format=format)
    assert (tmp_path/(p['sha256']+'.documents-attempts.jsonl')).read_bytes()==before


@pytest.mark.parametrize('case',['redirect','private_redirect','size','decoded_size','truncated','length','mime','malformed','rate','auth','timeout','interrupt','secret','pagination','duplicate_header'])
def test_failures_spend_once_preserve_diagnostics_and_no_secret(tmp_path,case):
    s=Stream(pdf())
    if case in {'redirect','private_redirect'}:s.status=302;s.headers=[('location','https://127.0.0.1/private' if case=='private_redirect' else 'https://example.invalid/?sig=SECRET')]
    elif case=='size':s.raw=b'X'*(512*1024+1);s.headers=[('content-type','application/pdf')]
    elif case=='decoded_size':s.raw=gzip.compress(b'X'*(512*1024+1));s.headers=[('content-type','application/pdf'),('content-encoding','gzip')]
    elif case=='truncated':s.raw=s.raw[:-7];s.headers=[('content-type','application/pdf')]
    elif case=='length':s.headers=[('content-type','application/pdf'),('content-length','99999')]
    elif case=='mime':s.headers=[('content-type','application/json')]
    elif case=='malformed':s.raw=b'not PDF';s.headers=[('content-type','application/pdf')]
    elif case in {'rate','auth'}:s.status=429 if case=='rate' else 401
    elif case=='timeout':s.error=TimeoutError(CANARY)
    elif case=='interrupt':s.error=KeyboardInterrupt(CANARY)
    elif case=='secret':s.raw=b'%PDF-1.3 api_key="SECRET" %%EOF';s.headers=[('content-type','application/pdf')]
    elif case=='pagination':s.headers.append(('x-next-page','2'))
    else:s.headers.append(('content-type','application/pdf'))
    result,p,calls=run(tmp_path,s);assert result['payload']['status']=='INCOMPLETE' and len(calls)==result['payload']['attempted']==1
    assert not result['payload']['objects'] and result['payload']['diagnostics']['exception_category'] in lane.diagnostics.CATEGORIES
    assert 'SECRET' not in json.dumps(result) and s.closed
    assert not result['payload']['inference'] and not result['payload']['source_acceptance'] and result['payload']['stake']==0


@pytest.mark.parametrize('attack',['url','auth','permission','future_permission','expired','config','reparse'])
def test_gates_before_first_attempt(tmp_path,attack):
    p,a,c=fixture(tmp_path)
    if attack=='url':p['payload']['documents'][0]['url']='https://127.0.0.1/';p=lane.seal(p['payload'])
    elif attack=='auth':a['payload']['plan_sha256']='0'*64;a=lane.seal(a['payload'])
    elif attack=='permission':p['payload']['permission']['private_retention']=False;p=lane.seal(p['payload'])
    elif attack=='future_permission':p['payload']['permission']['reviewed_at']='2026-10-06T19:26:00Z';p=lane.seal(p['payload'])
    def check(root):return dict(security(root),no_reparse=attack!='reparse')
    wall=lambda:'2026-10-07T00:00:00Z' if attack=='expired' else NOW
    with pytest.raises(ValueError):
        with lane.controlled(p,c,('0'*64 if attack=='config' else hashlib.sha256(c).hexdigest()),root=tmp_path,wall=wall,security=check):
            lane.acquire(p,a,root=tmp_path,transport=lambda *a:pytest.fail('gate reached transport'),wall=wall)
    assert list(tmp_path.iterdir())==[]


def socket_boundary(monkeypatch,raw):
    """Use real HTTPSConnection and worker; fake only DNS/TCP/TLS boundary."""
    sent=[]
    class Sock:
        def settimeout(self,*a):pass
        def setsockopt(self,*a):pass
        def connect(self,address):assert address[0]=='93.184.216.34'
        def sendall(self,data):sent.append(data)
        def makefile(self,*a,**k):return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw)
        def close(self):pass
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
    monkeypatch.setattr(socket,'socket',lambda *a,**k:Sock())
    monkeypatch.setattr(ssl.SSLContext,'wrap_socket',lambda self,sock,**kw:sock)
    return sent


def test_actual_cli_real_tls_worker_journal_custody(monkeypatch,tmp_path,capsys):
    from app_core import ncaaf_pilot_setup
    p,a,c=fixture(tmp_path);raw=pdf();sent=socket_boundary(monkeypatch,raw)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));conf=tmp_path/'configuration.json';conf.write_bytes(c)
    monkeypatch.setattr(cli,'verify_runtime',lambda:None) # Only this labelled fixture injects synthetic boundaries.
    monkeypatch.setattr(ncaaf_pilot_setup,'inspect_security',security)
    monkeypatch.setattr(lane,'controlled',partial(lane.controlled,wall=lambda:NOW));monkeypatch.setattr(lane,'acquire',partial(lane.acquire,wall=lambda:NOW))
    assert cli.main([str(path),'--execute','--controlled-bindings',str(conf),'--bindings-sha256',hashlib.sha256(c).hexdigest(),'--custody-root',str(tmp_path)])==0
    r=json.loads(capsys.readouterr().out);assert r['attempts']==1 and len(sent)==1 and b'GET /SYNTHETIC.pdf ' in sent[0]
    assert b'Authorization:' not in sent[0] and b'Cookie:' not in sent[0]
    assert lane.read_back(tmp_path/(p['sha256']+'.documents-unreviewed.json'),tmp_path/(p['sha256']+'.documents-attempts.jsonl'))['sha256']==r['bundle_sha256']


@pytest.mark.parametrize('address',['127.0.0.1','10.0.0.1','169.254.169.254','::1','fe80::1','224.0.0.1'])
def test_dns_private_destinations_reject_before_connect(monkeypatch,address):
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,443))])
    monkeypatch.setattr(http.client.HTTPSConnection,'connect',lambda *a:pytest.fail('private target reached connect'))
    import time
    with pytest.raises(ValueError,match='DOCUMENT_PRIVATE_ADDRESS'):lane.HttpsTransport()(URL,lane.LIMITS,time.monotonic()+10,time.monotonic)


def test_planning_networkfree_and_test_runtime_not_disabled(monkeypatch,tmp_path,capsys):
    p,a,c=fixture(tmp_path);path=tmp_path/'proposal.json';path.write_text(json.dumps(p));before=socket.create_connection
    assert cli.main([str(path)])==0;assert json.loads(capsys.readouterr().out)['network_requests']==0
    assert cli.main([str(path),'--execute'])==1;assert json.loads(capsys.readouterr().out)['reason']=='NFL_ACQUIRE_RUNTIME_UNSUPPORTED'
    assert socket.create_connection is before and list(tmp_path.iterdir())==[path]


def test_original_bundle_tampering_and_no_admission(tmp_path):
    from app_core import nfl_pdf_research,nfl_scenario_research
    result,p,_=run(tmp_path,Stream(pdf()))
    for reader in (nfl_pdf_research,nfl_scenario_research):
        with pytest.raises((ValueError,KeyError,TypeError)):reader.checked(result,NOW)
    changed=deepcopy(result);changed['payload']['objects'][0]['receipt']['observed_at']='2026-10-05T00:00:00Z';changed=lane.seal(changed['payload'])
    with pytest.raises(ValueError):lane.read_bundle(changed)
    assert lane.CONTROLLED.get() is None


def test_exact_body_limit_and_aggregate_stop(tmp_path):
    raw=b'<html>SYNTHETIC original complete body</html>'
    raw+=b' '*(512*1024-len(raw))
    result,_,_=run(tmp_path,Stream(raw,headers=[('content-type','text/html')]),format='html')
    assert result['payload']['status']=='CAPTURED_UNREVIEWED' and len(lane.read_bundle(result)['rules'][0])==512*1024
    other=tmp_path/'aggregate';other.mkdir();p,a,c=fixture(other,format='html')
    p['payload']['limits']['aggregate_decoded_bytes']=len(raw)-1;p=lane.seal(p['payload'])
    value=json.loads(c)['payload'];value['plan_sha256']=p['sha256'];a['payload']['plan_sha256']=p['sha256'];a=lane.seal(a['payload']);value['authorization']=a
    value['review']['subject_sha256']=lane.digest({k:v for k,v in value.items() if k!='review'});c=lane.custody.encode(lane.seal(value)).encode()
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=other,wall=lambda:NOW,security=security):
        stopped=lane.acquire(p,a,root=other,transport=lambda *a:Stream(raw,headers=[('content-type','text/html')]),wall=lambda:NOW)
    assert stopped['payload']['reason']=='DOCUMENT_AGGREGATE_SIZE' and stopped['payload']['attempted']==1


def test_shared_deadline_includes_postread_work_and_future_receipts(tmp_path):
    p,a,c=fixture(tmp_path);elapsed=[0.]
    class Slow(Stream):
        def chunks(self):
            yield self.raw
            elapsed[0]=61
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):
        stopped=lane.acquire(p,a,root=tmp_path,transport=lambda *a:Slow(pdf()),wall=lambda:NOW,monotonic=lambda:elapsed[0])
    assert stopped['payload']['status']=='INCOMPLETE' and stopped['payload']['reason']=='DOCUMENT_DEADLINE'
    assert stopped['payload']['attempted']==1


def test_bounded_real_worker_timeout_and_untrusted_source(monkeypatch):
    import time
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:(time.sleep(.025) or [(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))]))
    monkeypatch.setattr(http.client.HTTPSConnection,'connect',lambda *a:pytest.fail('late resolver reached connection'))
    with pytest.raises(ValueError):lane.HttpsTransport()(URL,lane.LIMITS,time.monotonic()+.005,time.monotonic)
    with pytest.raises(ValueError,match='DOCUMENT_URL'):lane.url('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds')


def test_failed_http_status_is_observed_not_invented(tmp_path):
    result,_,_=run(tmp_path,Stream(b'',status=403))
    assert result['payload']['diagnostics']['http_status']==403 and result['payload']['diagnostics']['phase']=='RESPONSE_HEADERS'


def test_embedded_links_and_headers_remain_inert(tmp_path):
    raw=b'<html>SYNTHETIC original page <script>fetch("http://127.0.0.1/private")</script><a href="file:///private">link</a></html>'
    result,p,calls=run(tmp_path,Stream(raw,headers=[('content-type','text/html'),('set-cookie',CANARY),('etag',CANARY)]),format='html')
    assert result['payload']['status']=='CAPTURED_UNREVIEWED' and calls==[URL]
    assert lane.read_bundle(result)['rules'][0]==raw and CANARY not in json.dumps(result)
    assert result['payload']['objects'][0]['response_metadata']['http_date']=='UNKNOWN'
