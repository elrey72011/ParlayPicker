"""Labelled synthetic HTML via real HTTPSConnection/HTTPResponse; no external calls."""
from functools import partial
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
from test_nfl_document_acquisition import guide_fixture, security, NOW

ANCHOR=b'<!DOCTYPE html><html><section id="api-key">SYNTHETIC public parameter heading</section></html>'

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield

def boundary(monkeypatch,raw,*,wire=None,declared=None):
    sent=[];wire=raw if wire is None else wire
    class Sock:
        def settimeout(self,*a):pass
        def setsockopt(self,*a):pass
        def connect(self,address):assert address[0]=='93.184.216.34'
        def sendall(self,data):sent.append(data)
        def makefile(self,*a,**k):
            headers=b'HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/html\r\nContent-Length: '+str(len(wire) if declared is None else declared).encode()+b'\r\n'
            if wire is not raw:headers+=b'Content-Encoding: gzip\r\n'
            return io.BytesIO(headers+b'\r\n'+wire)
        def shutdown(self,*a):pass
        def close(self):pass
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))])
    monkeypatch.setattr(socket,'socket',lambda *a,**k:Sock())
    monkeypatch.setattr(ssl.SSLContext,'wrap_socket',lambda self,sock,**kw:sock)
    return sent

def actual_cli(monkeypatch,root,capsys,raw,*,version=lane.VERSION_V2,wire=None,declared=None):
    from app_core import ncaaf_pilot_setup
    plan,auth,config=guide_fixture(root,version=version)
    planpath=root/'SYNTHETIC-plan.json';planpath.write_text(json.dumps(plan))
    configpath=root/'SYNTHETIC-config.json';configpath.write_bytes(config)
    sent=boundary(monkeypatch,raw,wire=wire,declared=declared)
    # Synthetic transport/security/clock boundaries only; actual HTTP worker,
    # CLI, authorization, durable journal, body validator and read-back remain.
    monkeypatch.setattr(cli,'verify_runtime',lambda:None)
    monkeypatch.setattr(ncaaf_pilot_setup,'inspect_security',security)
    monkeypatch.setattr(lane,'controlled',partial(lane.controlled,wall=lambda:NOW))
    monkeypatch.setattr(lane,'acquire',partial(lane.acquire,wall=lambda:NOW))
    exit_code=cli.main([str(planpath),'--execute','--controlled-bindings',str(configpath),'--bindings-sha256',hashlib.sha256(config).hexdigest(),'--custody-root',str(root)])
    report=json.loads(capsys.readouterr().out)
    bundle=lane.read_back(root/(plan['sha256']+'.documents-unreviewed.json'),root/(plan['sha256']+'.documents-attempts.jsonl'))
    assert len(sent)==report['attempts']==bundle['payload']['attempted']==1
    assert b'Authorization:' not in sent[0] and b'Cookie:' not in sent[0]
    assert bundle['payload']['inference'] is bundle['payload']['owner_reviewed'] is bundle['payload']['source_acceptance'] is False
    assert bundle['payload']['stake']==0 and bundle['payload']['wager_action']=='PASS'
    return exit_code,report,bundle,plan,sent

def test_original_v2_actual_transport_anchor_rejection(monkeypatch,tmp_path,capsys):
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,ANCHOR)
    assert code==1 and report['reason']=='DOCUMENT_CREDENTIAL_BODY'
    assert report['diagnostics']['phase']=='BODY_VALIDATION' and report['diagnostics']['http_status']==200
    assert bundle['payload']['objects']==[]

@pytest.mark.parametrize('markup',[
 '<section id="api-key">Public heading</section>',
 "<h2 ID = 'APIKEY'>Public heading</h2>",
 '<a name="access_token"></a>',
 '<a href="#api-key">Heading link</a>',
 '<section id="api&#45;key">Encoded public anchor</section>',
 '<a href="&#35;api-key">Encoded fragment</a>',
 '<div id="password"><h3>Parameter heading</h3></div>',
 '<html><section id="api-key">apiKey=YOUR_API_KEY &amp; apiKey={apiKey}</section></html>'
])
@pytest.mark.parametrize('compressed',[False,True])
def test_v3_complete_anchor_bytes_cli_transport_and_readback(monkeypatch,tmp_path,capsys,markup,compressed):
    import gzip
    from app_core import nfl_html_guide_custody as guide, nfl_public_guide_custody as old
    raw=('<!DOCTYPE html><html><body>SYNTHETIC '+markup+'</body></html>').encode()
    code,report,bundle,plan,sent=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V3,wire=gzip.compress(raw) if compressed else None)
    assert code==0 and report['status']=='CAPTURED_UNREVIEWED' and bundle['payload']['version']==lane.BUNDLE_V3
    assert lane.read_bundle(bundle)['clock'][0]==raw and report['bodies'][0]['body_sha256']==hashlib.sha256(raw).hexdigest()
    projected=lane.project(bundle,'clock')
    assert projected['version']==guide.VERSION and guide.read(projected)[0]==raw
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_MISSING'):old.read(projected)
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_MISSING'):lane.custody.read(projected)
    with guide.selected():assert lane.custody.read(projected)[0]==raw
    journal=tmp_path/(plan['sha256']+'.documents-attempts.jsonl');saved=journal.read_bytes()
    p,a,c=guide_fixture(tmp_path,version=lane.VERSION_V3)
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):
        with pytest.raises(ValueError,match='DOCUMENT_SPENT_NO_RESUME'):lane.acquire(p,a,root=tmp_path,wall=lambda:NOW)
    assert journal.read_bytes()==saved and len(sent)==1

@pytest.mark.parametrize('sensitive',[
 'apiKey=SECRET_CANARY', 'api_key="SECRET_CANARY"', 'access_token: SECRET_CANARY',
 'authorization: Bearer SECRET_CANARY', 'password=SECRET_CANARY', 'client_secret="SECRET_CANARY"',
 'apiKey=YOUR_API_KEY_SECRET_CANARY', 'apiKey={apiKey}SECRET_CANARY',
 'apiKey=YOUR_API_KEY&amp;apiKey=SECRET_CANARY',
 '<section id="api-key:SECRET_CANARY">Value</section>',
 '<a href="#api-key=SECRET_CANARY">Value</a>',
 '<input name="api-key" value="SECRET_CANARY">',
 '<div id="api-key" data-value="SECRET_CANARY">Value</div>',
 '<div id="api-key" id="other">Ambiguous syntax</div>',
 '<section data-key="api-key">Unrecognized attribute</section>',
 '<!-- <section id="api-key">Not an actual anchor</section> -->',
 '<script>const token = "<section id=\'api-key\'>";apiKey="SECRET_CANARY"</script>',
 'apiKey&#61;SECRET_CANARY', 'api&#75;ey=SECRET_CANARY',
 '%61piKey%3DSECRET_CANARY', '%2561piKey%253DSECRET_CANARY',
 'apiKey&amp;#61;SECRET_CANARY', r'\u0061piKey=SECRET_CANARY',
 r'{\"apiKey\":\"SECRET_CANARY\"}'
])
def test_v3_sensitive_mixed_encoded_and_unrecognized_content_rejects(monkeypatch,tmp_path,capsys,sensitive):
    raw=('<!DOCTYPE html><html><section id="api-key">SYNTHETIC public anchor</section>'+sensitive+'</html>').encode()
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V3)
    assert code==1 and report['reason']=='DOCUMENT_CREDENTIAL_BODY'
    assert not bundle['payload']['objects'] and 'SECRET_CANARY' not in json.dumps(bundle)
    assert 'SECRET_CANARY' not in json.dumps(report)

@pytest.mark.parametrize('case',['exact','wire_over','decoded_over','truncated','deadline'])
def test_v3_bounds_unchanged(monkeypatch,tmp_path,capsys,case):
    import gzip
    raw=ANCHOR+b' '*(512*1024-len(ANCHOR));wire=None;declared=None
    if case=='wire_over':raw+=b' '
    elif case=='decoded_over':raw+=b' ';wire=gzip.compress(raw)
    elif case=='truncated':declared=len(raw)+1
    elif case=='deadline':
        def late_chunks(self):
            yield raw
            raise TimeoutError('SYNTHETIC SECRET_CANARY')
        monkeypatch.setattr(lane.Response,'chunks',late_chunks)
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V3,wire=wire,declared=declared)
    assert code==int(case!='exact') and 'SECRET_CANARY' not in json.dumps(bundle)
    if case=='exact':assert len(lane.read_bundle(bundle)['clock'][0])==512*1024
    else:assert report['reason'] in {'DOCUMENT_SIZE','DOCUMENT_INCOMPLETE','DOCUMENT_TRANSPORT_FAILURE'}

def test_v2_authorization_cannot_authorize_v3_before_transport(tmp_path):
    p,a,c=guide_fixture(tmp_path,version=lane.VERSION_V3)
    old,old_auth,_=guide_fixture(tmp_path,version=lane.VERSION_V2)
    assert p['sha256']!=old['sha256']
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):
        with pytest.raises(ValueError,match='DOCUMENT_AUTHORIZATION'):
            lane.acquire(p,old_auth,root=tmp_path,wall=lambda:NOW,transport=lambda *a:pytest.fail('OLD AUTHORIZATION REACHED TRANSPORT'))
    assert list(tmp_path.iterdir())==[]

def test_v3_cannot_rehash_bundle_as_an_old_version(monkeypatch,tmp_path,capsys):
    from copy import deepcopy
    _,_,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,ANCHOR,version=lane.VERSION_V3)
    changed=deepcopy(bundle['payload']);changed['version']=lane.BUNDLE_V2
    with pytest.raises(ValueError,match='DOCUMENT_READBACK'):lane.read_bundle(lane.seal(changed))

def test_v3_does_not_whitelist_domains_or_non_html(tmp_path):
    from test_nfl_document_acquisition import capture_guide
    result,_,_=capture_guide(tmp_path,b'<html>apiKey=YOUR_API_KEY</html>',version=lane.VERSION_V3,url='https://support.novig.com/en/SYNTHETIC')
    assert result['payload']['reason']=='DOCUMENT_CREDENTIAL_BODY'
    with pytest.raises(ValueError,match='DOCUMENT_CREDENTIAL_BODY'):lane.body_ok(b'apiKey=SECRET_CANARY','text',html_syntax=True)

def test_v3_deeply_encoded_unknown_content_is_unavailable(monkeypatch,tmp_path,capsys):
    raw=b'<html><section id="api-key">SYNTHETIC</section>%252561piKey%25253DSECRET_CANARY</html>'
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V3)
    assert code==1 and report['reason']=='DOCUMENT_FORMAT'
    assert not bundle['payload']['objects'] and 'SECRET_CANARY' not in json.dumps(bundle)
