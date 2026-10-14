"""Explicit original-document acquisition. No crawling, credentials or admission.

Wire bytes and decoded entity bytes are distinct originals. PDF extraction uses
only the complete decoded entity, never a rewritten PDF or a gzip wrapper.
"""
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import re
from pathlib import Path
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit
import zlib
from app_core import nfl_observed_custody as custody, nfl_acquisition_diagnostics as diagnostics
from app_core.ncaaf_pilot import _timed
from app_core.producer_provenance import clock

VERSION = 'football-original-document-plan-v1'
VERSION_V2 = 'football-original-document-plan-v2'
VERSION_V3 = 'football-original-document-plan-v3'
BUNDLE = 'football-original-document-unreviewed-v1'
BUNDLE_V2 = 'football-original-document-unreviewed-v2'
BUNDLE_V3 = 'football-original-document-unreviewed-v3'
GUIDE_URL = 'https://the-odds-api.com/liveapi/guides/v4/'
CONFIG = 'football-original-document-bindings-v1'
AUTH = 'football-original-document-authorization-v1'
CONTROLLED = ContextVar('football_original_document_controlled', default=None)
HOSTS = frozenset(('www.denverbroncos.com', 'www.seahawks.com', 'the-odds-api.com',
    'support.novig.com', 'novig.intercom-attachments-1.com',
    'ludlow-filings.s3.us-east-1.amazonaws.com', 'www.ludlowexchange.com',
    'sportsbook.draftkings.com', 'www.goldennugget.com', 'lb.goldennugget.com',
    'bluehens.com', 'goblueraiders.com', 'conferenceusa.com', 'wkusports.com', 'gobearkats.com'))
LIMITS = dict(max_attempts=8, max_documents=4, wire_body_bytes=512*1024,
    decoded_body_bytes=512*1024, aggregate_wire_bytes=2*1024*1024,
    aggregate_decoded_bytes=2*1024*1024, serialized_bundle_bytes=6*1024*1024,
    connect_seconds=10, read_seconds=10, total_seconds=60)
require = custody.require
digest = custody.digest
seal = lambda p: dict(payload=p, sha256=digest(p))


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()


def envelope(v):
    require(isinstance(v,dict) and set(v)=={'payload','sha256'} and digest(v['payload'])==v['sha256'], 'DOCUMENT_INTEGRITY')
    return v['payload']


def url(value):
    require(isinstance(value,str) and 0<len(value)<=2048 and not re.search(r'[\s\\\x00-\x1f]',value), 'DOCUMENT_URL')
    u=urlsplit(value)
    require(u.scheme=='https' and u.hostname in HOSTS and u.netloc==u.hostname
            and not u.username and not u.password and not u.query and not u.fragment
            and u.path.startswith('/') and '..' not in u.path.split('/'), 'DOCUMENT_URL')
    return u


def checked_clock(v):
    require(isinstance(v,str) and re.search(r'(Z|[+-]\d\d:\d\d)$',v) and clock(v), 'DOCUMENT_CLOCK')
    return clock(v)


def validate(plan, executable=False):
    p=envelope(plan)
    require(len(custody.encode(plan).encode())<=128*1024,'DOCUMENT_PLAN')
    require(set(p)==set('version evidence_label documents limits window permission authorization_ref custody_id unresolved'.split())
        and p['version'] in {VERSION,VERSION_V2,VERSION_V3} and p['evidence_label'] in {'PROPOSAL','SYNTHETIC','PROSPECTIVE'}, 'DOCUMENT_PLAN')
    docs=p['documents'];require(isinstance(docs,list) and 0<len(docs)<=4, 'DOCUMENT_PLAN')
    ids=set();urls=set();destinations=set();attempts=0
    for d in docs:
        require(isinstance(d,dict) and set(d)==set('id source_id url redirects format purpose destination'.split())
            and isinstance(d['id'],str) and re.fullmatch('[a-zA-Z0-9_-]{1,64}',d['id']) and d['id'] not in ids
            and isinstance(d['source_id'],str) and d['source_id'] and d['format'] in {'html','text','pdf'}
            and isinstance(d['purpose'],str) and len(d['purpose'])>=16
            and d['destination']==d['id']+'.original' and d['destination'] not in destinations
            and isinstance(d['redirects'],list) and len(d['redirects'])<=1, 'DOCUMENT_PLAN')
        ids.add(d['id']);destinations.add(d['destination'])
        for v in [d['url'],*d['redirects']]:
            url(v);require(v not in urls,'DOCUMENT_DUPLICATE');urls.add(v);attempts+=1
    lim=p['limits'];require(isinstance(lim,dict) and set(lim)==set(LIMITS), 'DOCUMENT_LIMITS')
    require(all(type(lim[k]) is int and 0<lim[k]<=v for k,v in LIMITS.items())
        and attempts<=lim['max_attempts'] and len(docs)<=lim['max_documents'], 'DOCUMENT_LIMITS')
    w=p['window'];require(set(w)=={'not_before','not_after','expires_at'} and checked_clock(w['not_before'])<checked_clock(w['not_after'])<=checked_clock(w['expires_at']), 'DOCUMENT_WINDOW')
    require(isinstance(p['unresolved'],list) and isinstance(p['custody_id'],str) and p['custody_id']
        and isinstance(p['authorization_ref'],str) and p['authorization_ref'], 'DOCUMENT_PLAN')
    if executable:require(p['evidence_label']!='PROPOSAL' and not p['unresolved'],'DOCUMENT_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p=validate(plan)
    return dict(version=p['version'],plan_sha256=plan['sha256'],status='NON_EXECUTABLE' if p['unresolved'] or p['evidence_label']=='PROPOSAL' else 'AWAITING_AUTHORIZATION',
        maximum_attempts=sum(1+len(d['redirects']) for d in p['documents']),documents=p['documents'],limits=p['limits'],
        network_requests=0,credential_loads=0,storage_operations=0,odds_credits=0,cfbd_requests=0,additional_spend=0)


def authorize(plan, authorization, at, root):
    p=validate(plan,True);c=CONTROLLED.get();require(c is not None,'DOCUMENT_UNTRUSTED_CONFIGURATION')
    a=envelope(authorization);now=checked_clock(at);w=p['window']
    require(set(a)==set('version reference plan_sha256 custody_id owner authorized_at expires_at maximum_attempts maximum_additional_spend'.split())
        and a['version']==AUTH and a['owner']=='Robert Velarde' and a['reference']==p['authorization_ref']
        and a['plan_sha256']==plan['sha256'] and a['custody_id']==p['custody_id']
        and type(a['maximum_attempts']) is int and a['maximum_attempts']==sum(1+len(d['redirects']) for d in p['documents'])
        and type(a['maximum_additional_spend']) is int and a['maximum_additional_spend']==0
        and authorization==c['authorization'] and c['plan_sha256']==plan['sha256'], 'DOCUMENT_AUTHORIZATION')
    require(checked_clock(w['not_before'])<=now<checked_clock(w['not_after']) and checked_clock(a['authorized_at'])<=now
        <checked_clock(a['expires_at'])==checked_clock(w['expires_at']), 'DOCUMENT_WINDOW_EXPIRED')
    permission=p['permission']
    require(isinstance(permission,dict) and set(permission)==set('reviewer reviewed_at effective_from effective_until documents_sha256 private_acquisition private_retention basis'.split())
        and permission['reviewer']=='Robert Velarde' and permission['documents_sha256']==digest(p['documents'])
        and permission['private_acquisition'] is True and permission['private_retention'] is True
        and isinstance(permission['basis'],str) and len(permission['basis'])>=16 and permission==c['permission'], 'DOCUMENT_PERMISSION')
    require(checked_clock(permission['reviewed_at'])<now and checked_clock(permission['effective_from'])<=now
        <checked_clock(w['expires_at'])<=checked_clock(permission['effective_until']), 'DOCUMENT_PERMISSION_CLOCK')
    require(c['custody']['id']==p['custody_id'] and Path(c['custody']['root']).resolve()==Path(root).resolve() and Path(root).is_dir(), 'DOCUMENT_CUSTODY')
    return p


@contextmanager
def controlled(plan, raw, expected_hash, *, root, wall=utc, security=None):
    validate(plan);require(isinstance(raw,bytes) and 0<len(raw)<=128*1024 and sha(raw)==expected_hash,'DOCUMENT_CONFIGURATION')
    c=envelope(json.loads(raw,object_pairs_hook=custody.unique))
    from app_core.ncaaf_response_custody import _credentials
    try:
        _credentials({k:v for k,v in c.items() if k!='authorization'})
        _credentials(c.get('authorization',{}))
    except ValueError:raise ValueError('DOCUMENT_CONFIGURATION') from None
    require(set(c)==set('version plan_sha256 permission authorization custody review'.split()) and c['version']==CONFIG
        and c['plan_sha256']==plan['sha256'] and c['permission']==plan['payload']['permission'],'DOCUMENT_CONFIGURATION')
    d=c['custody'];require(set(d)=={'id','root','storage_receipt_sha256'} and re.fullmatch('[0-9a-f]{64}',d['storage_receipt_sha256']), 'DOCUMENT_CUSTODY')
    r=c['review'];require(set(r)=={'reviewer','reviewed_at','subject_sha256','status','basis'} and r['reviewer']=='Robert Velarde'
        and r['status']=='OWNER_REVIEWED' and r['subject_sha256']==digest({k:v for k,v in c.items() if k!='review'})
        and isinstance(r['basis'],str) and len(r['basis'])>=16 and checked_clock(r['reviewed_at'])<checked_clock(wall()),'DOCUMENT_CONFIGURATION_REVIEW')
    if security is None:
        from app_core.ncaaf_pilot_setup import inspect_security
        security=inspect_security
    found=security(root);require(found.get('status')=='VERIFIED' and all(found.get(k) is True for k in ('private_acl','encrypted','no_reparse')),'DOCUMENT_CUSTODY')
    token=CONTROLLED.set(dict(deepcopy(c),configuration_bytes_base64=base64.b64encode(raw).decode(),configuration_file_sha256=expected_hash))
    try:
        authorize(plan,c['authorization'],wall(),root);yield c['authorization']
    finally:CONTROLLED.reset(token)


class HttpsTransport:
    """Real TLS path with validated public DNS pinned to one address; no fallback."""
    def __call__(self, target, limits, deadline, monotonic):
        u=url(target);started=monotonic();phase='CONNECTION_SETUP';cancel=threading.Event();conn=None
        try:
            def close():
                if conn is not None:conn.close()
            addresses=_timed(lambda:socket.getaddrinfo(u.hostname,443,type=socket.SOCK_STREAM),
                min(deadline,started+limits['connect_seconds']),monotonic,cancel,close)
            require(addresses and all(ipaddress.ip_address(a[4][0]).is_global and not ipaddress.ip_address(a[4][0]).is_multicast for a in addresses),'DOCUMENT_PRIVATE_ADDRESS')
            selected=sorted(addresses,key=lambda a:(a[0],a[4][0]))[0]
            conn=http.client.HTTPSConnection(u.hostname,timeout=limits['connect_seconds'],context=ssl.create_default_context());conn.auto_open=False
            def pinned(address, timeout=None, source_address=None):
                require(not cancel.is_set() and monotonic()<deadline,'DOCUMENT_DEADLINE')
                sock=socket.socket(selected[0],socket.SOCK_STREAM)
                try:sock.settimeout(timeout);sock.connect(selected[4]);return sock
                except BaseException:sock.close();raise
            conn._create_connection=pinned
            phase='HTTPS_CONNECT';_timed(conn.connect,min(deadline,started+limits['connect_seconds']),monotonic,cancel,close)
            conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
            phase='REQUEST_SEND'
            _timed(lambda:conn.request('GET',u.path,headers={'Accept':'application/pdf,text/html,text/plain','Accept-Encoding':'identity'}),deadline,monotonic,cancel,close)
            # HTTPConnection detaches its socket on Connection: close while
            # HTTPResponse still owns the readable body. Retain that exact socket.
            read_socket=conn.sock
            phase='RESPONSE_RECEIVE';response=_timed(conn.getresponse,min(deadline,monotonic()+limits['read_seconds']),monotonic,cancel,close)
            return Response(conn,response,read_socket,cancel,deadline,monotonic,limits,started)
        except BaseException as exc:
            if conn is not None:conn.close()
            code=str(exc) if isinstance(exc,ValueError) and str(exc) in {'DOCUMENT_PRIVATE_ADDRESS','DOCUMENT_DEADLINE','DOCUMENT_URL'} else 'DOCUMENT_TRANSPORT_FAILURE'
            raise diagnostics.Failure(code,diagnostics.facts(exc,phase=phase,started=started,ended=monotonic())) from None


class Response:
    def __init__(self,conn,response,read_socket,cancel,deadline,monotonic,limits,started):
        self.conn=conn;self.response=response;self.read_socket=read_socket;self.cancel=cancel;self.deadline=deadline;self.monotonic=monotonic;self.limits=limits;self.started=started
        self.status=response.status;self.headers=response.getheaders()
    def chunks(self):
        while True:
            try:
                require(not self.cancel.is_set() and self.monotonic()<self.deadline,'NCAAF_PILOT_DEADLINE')
                # HTTPResponse closes its file (and the detached socket) after
                # the final framed read. No further socket operation is needed;
                # acquire still verifies the exact declared body length.
                if self.response.isclosed():return
                self.read_socket.settimeout(min(self.limits['read_seconds'],max(.001,self.deadline-self.monotonic())))
                block=_timed(lambda:self.response.read(8192),min(self.deadline,self.monotonic()+self.limits['read_seconds']),self.monotonic,self.cancel,self.close)
            except BaseException as exc:
                raise diagnostics.Failure('DOCUMENT_READ_FAILURE',diagnostics.facts(exc,phase='BODY_READ',http_status=self.status,started=self.started,ended=self.monotonic())) from None
            if not block:return
            yield block
    def close(self):
        # Interrupt a detached body's pending read before closing its file; do
        # not wait on a buffered-reader lock with an otherwise live socket.
        shutdown=getattr(self.read_socket,'shutdown',None)
        if shutdown is not None:
            try:shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.response.close();self.conn.close()


def headers(items):
    require(isinstance(items,(list,tuple)) and len(items)<=64,'DOCUMENT_HEADERS');out={}
    for k,v in items:
        require(isinstance(k,str) and isinstance(v,str) and len(k)+len(v)<=4096 and not re.search(r'[\r\n\x00]',v),'DOCUMENT_HEADERS')
        k=k.lower();require(k not in out,'DOCUMENT_HEADERS');out[k]=v
    require(sum(len(k)+len(v) for k,v in out.items())<=16384,'DOCUMENT_HEADERS')
    require(not any(k in out for k in ('x-next-page','x-next-cursor')) and not ('rel="next"' in out.get('link','')),'DOCUMENT_PAGINATION')
    return out


def decode(wire, encoding, maximum):
    require(encoding in {'identity','gzip'},'DOCUMENT_ENCODING')
    if encoding=='identity':raw=wire
    else:
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        try:
            raw=decoder.decompress(wire,maximum+1)
            require(len(raw)<=maximum and not decoder.unconsumed_tail,'DOCUMENT_SIZE')
            require(decoder.eof and not decoder.unused_data,'DOCUMENT_INCOMPLETE')
        except zlib.error:raise ValueError('DOCUMENT_INCOMPLETE') from None
    require(0<len(raw)<=maximum,'DOCUMENT_SIZE');return raw


def response_metadata(h, mime, encoding):
    def date(key):
        value=h.get(key)
        return value if value and re.fullmatch(r'[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT',value) else 'UNKNOWN'
    return dict(content_type=mime,content_encoding=encoding,content_length=h.get('content-length','UNKNOWN'),
        http_date=date('date'),last_modified=date('last-modified'),etag_sha256=sha(h['etag'].encode()) if 'etag' in h else 'UNKNOWN',publisher_available_at='UNKNOWN')


def public_examples(plan, document):
    return (plan['version'] in {VERSION_V2,VERSION_V3} and document['id']==document['source_id']=='clock'
        and document['url']==GUIDE_URL and document['format']=='html' and document['redirects']==[])


def body_ok(raw, format, *, public_guide_examples=False, html_syntax=False):
    if html_syntax and format=='html':
        from app_core.nfl_html_document_scan import checked
        checked(raw,public_guide_examples=public_guide_examples)
        return
    scan=raw
    if public_guide_examples:
        # Validation-only recognition of two exact published noncredential
        # literals. Originals are never replaced, filtered or reserialized.
        for literal in (b'apiKey=YOUR_API_KEY',b'apiKey={apiKey}'):
            scan=re.sub(rb'(?<![A-Za-z0-9_])'+re.escape(literal)+rb'(?=$|[&\s"\x27<>])',b'PUBLIC_DOCUMENTATION_LITERAL',scan)
    require(not re.search(rb'(?i)(api[_-]?key|access[_-]?token|authorization|client_secret|password)\s*["\x27:=]',scan),'DOCUMENT_CREDENTIAL_BODY')
    if format=='pdf':require(raw.startswith(b'%PDF-') and raw.rstrip().endswith(b'%%EOF'),'DOCUMENT_FORMAT')
    else:
        try:text=raw.decode('utf-8')
        except UnicodeError:raise ValueError('DOCUMENT_FORMAT') from None
        require(text.strip() and (format!='html' or re.search(r'(?i)<(?:!doctype\s+html|html)(?:\s|>)',text[:4096])),'DOCUMENT_FORMAT')


def write_exclusive(path, raw):
    import os
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    require(path.read_bytes()==raw,'DOCUMENT_READBACK')


def acquire(plan, authorization, *, root, transport=None, wall=utc, monotonic=time.monotonic):
    root=Path(root);p=authorize(plan,authorization,wall(),root);stem=plan['sha256']
    journal=root/(stem+'.documents-attempts.jsonl');output=root/(stem+'.documents-unreviewed.json')
    require(not journal.exists() and not output.exists(),'DOCUMENT_SPENT_NO_RESUME')
    start=monotonic()
    # A timed connection worker cannot outlive the exact collection expiry.
    expires=datetime.fromisoformat(checked_clock(authorization['payload']['expires_at']))
    current=datetime.fromisoformat(checked_clock(wall()))
    remaining=(expires-current).total_seconds();require(remaining>0,'DOCUMENT_WINDOW_EXPIRED')
    deadline=start+min(p['limits']['total_seconds'],remaining);transport=transport or HttpsTransport()
    entries=[];objects=[];reason=None;diagnostic=None;aggregate_wire=aggregate_decoded=0;status=diagnostics.UNKNOWN;phase='JOURNAL'
    # Reserve even a zero-attempt crash; reruns never spend silently again.
    write_exclusive(journal,b'')
    try:
        for d in p['documents']:
            chain=[]
            for hop,target in enumerate([d['url'],*d['redirects']]):
                authorize(plan,authorization,wall(),root);require(monotonic()<deadline,'DOCUMENT_DEADLINE')
                entry=dict(attempt=len(entries)+1,document=d['id'],url=target,request_started_at=wall(),plan_sha256=stem)
                require(entry['attempt']<=p['limits']['max_attempts'],'DOCUMENT_BUDGET')
                import os
                with journal.open('ab') as f:f.write((custody.encode(entry)+'\n').encode());f.flush();os.fsync(f.fileno())
                entries.append(entry);response=None
                try:
                    phase='TRANSPORT';status=diagnostics.UNKNOWN
                    response=transport(target,p['limits'],deadline,monotonic);status=response.status;phase='RESPONSE_HEADERS';h=headers(response.headers)
                    require(type(status) is int,'DOCUMENT_STATUS')
                    if status in {301,302,303,307,308}:
                        require(hop==0 and len(d['redirects'])==1 and h.get('location')==d['redirects'][0],'DOCUMENT_REDIRECT')
                        url(h['location']);chain.append(dict(url=target,status=status,request_started_at=entry['request_started_at'],received_at=wall()))
                        continue
                    require(status==200,'DOCUMENT_HTTP_STATUS')
                    mime=h.get('content-type','').split(';')[0].strip().lower()
                    require(mime in {'pdf':{'application/pdf'},'html':{'text/html','application/xhtml+xml'},'text':{'text/plain'}}[d['format']],'DOCUMENT_FORMAT')
                    parts=[];size=0
                    phase='BODY_READ'
                    for block in response.chunks():
                        require(monotonic()<deadline,'DOCUMENT_DEADLINE');authorize(plan,authorization,wall(),root)
                        require(isinstance(block,bytes),'DOCUMENT_INCOMPLETE');size+=len(block)
                        require(size<=p['limits']['wire_body_bytes'],'DOCUMENT_SIZE');parts.append(block)
                    wire=b''.join(parts)
                    require('content-length' not in h or (h['content-length'].isdigit() and int(h['content-length'])==len(wire)),'DOCUMENT_INCOMPLETE')
                    phase='BODY_VALIDATION';encoding=h.get('content-encoding','identity').lower();raw=decode(wire,encoding,p['limits']['decoded_body_bytes']);body_ok(raw,d['format'],public_guide_examples=public_examples(p,d),html_syntax=p['version']==VERSION_V3)
                    aggregate_wire+=len(wire);aggregate_decoded+=len(raw)
                    require(aggregate_wire<=p['limits']['aggregate_wire_bytes'] and aggregate_decoded<=p['limits']['aggregate_decoded_bytes'],'DOCUMENT_AGGREGATE_SIZE')
                    received=wall();observed=wall();require(checked_clock(entry['request_started_at'])<=checked_clock(received)<=checked_clock(observed),'DOCUMENT_CLOCK')
                    receipt=dict(source_id=d['source_id'],endpoint=target,format='text' if d['format']=='html' else d['format'],
                        evidence_label='SYNTHETIC' if p['evidence_label']=='SYNTHETIC' else 'RETAINED',request_started_at=entry['request_started_at'],received_at=received,observed_at=observed,
                        observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='attributed_document_no_publication_clock')
                    objects.append(dict(id=d['id'],source_url=d['url'],final_url=target,redirects=chain,receipt=receipt,status=status,
                        response_metadata=response_metadata(h,mime,encoding),
                        wire_bytes_base64=base64.b64encode(wire).decode(),wire_sha256=sha(wire),body_bytes_base64=base64.b64encode(raw).decode(),body_sha256=sha(raw)))
                    require(monotonic()<deadline,'DOCUMENT_DEADLINE');authorize(plan,authorization,wall(),root)
                    break
                finally:
                    if response is not None:response.close()
            else:raise ValueError('DOCUMENT_REDIRECT')
    except BaseException as exc:
        reason=str(exc) if isinstance(exc,ValueError) and str(exc) in CODES else 'DOCUMENT_TRANSPORT_FAILURE'
        diagnostic=getattr(exc,'diagnostic',diagnostics.facts(exc,phase=phase,http_status=status,started=start,ended=monotonic()))
    result=seal(dict(version={VERSION:BUNDLE,VERSION_V2:BUNDLE_V2,VERSION_V3:BUNDLE_V3}[p['version']],plan=deepcopy(plan),authorization=deepcopy(authorization),status='CAPTURED_UNREVIEWED' if reason is None else 'INCOMPLETE',reason=reason,
        attempted=len(entries),journal_sha256=sha(journal.read_bytes()),entries=entries,objects=objects,diagnostics=diagnostic,
        configuration_bytes_base64=CONTROLLED.get()['configuration_bytes_base64'],configuration_file_sha256=CONTROLLED.get()['configuration_file_sha256'],
        source_acceptance=False,owner_reviewed=False,inference=False,publication=False,wager_action='PASS',stake=0))
    encoded=custody.encode(result).encode();require(len(encoded)<=p['limits']['serialized_bundle_bytes'],'DOCUMENT_SERIALIZED_SIZE')
    write_exclusive(output,encoded);read_back(output,journal);return result


CODES=frozenset('DOCUMENT_'+v for v in '''PRIVATE_ADDRESS DEADLINE URL TRANSPORT_FAILURE READ_FAILURE HEADERS PAGINATION ENCODING INCOMPLETE SIZE FORMAT CREDENTIAL_BODY REDIRECT HTTP_STATUS STATUS CLOCK BUDGET AGGREGATE_SIZE WINDOW_EXPIRED PERMISSION_CLOCK AUTHORIZATION PERMISSION'''.split())


def read_bundle(bundle):
    p=envelope(bundle);require(set(p)==set('version plan authorization status reason attempted journal_sha256 entries objects diagnostics configuration_bytes_base64 configuration_file_sha256 source_acceptance owner_reviewed inference publication wager_action stake'.split()) and p['version'] in {BUNDLE,BUNDLE_V2,BUNDLE_V3} and p['status'] in {'CAPTURED_UNREVIEWED','INCOMPLETE'},'DOCUMENT_READBACK')
    plan=validate(p['plan']);require(envelope(p['authorization'])['plan_sha256']==p['plan']['sha256'],'DOCUMENT_READBACK')
    require(p['version']=={VERSION:BUNDLE,VERSION_V2:BUNDLE_V2,VERSION_V3:BUNDLE_V3}[plan['version']],'DOCUMENT_READBACK')
    config_raw=base64.b64decode(p['configuration_bytes_base64'],validate=True)
    require(0<len(config_raw)<=128*1024 and sha(config_raw)==p['configuration_file_sha256'],'DOCUMENT_READBACK')
    c=envelope(json.loads(config_raw,object_pairs_hook=custody.unique))
    require(set(c)==set('version plan_sha256 permission authorization custody review'.split()) and c['version']==CONFIG and c['plan_sha256']==p['plan']['sha256']
        and c['authorization']==p['authorization'] and c['permission']==plan['permission']
        and c['review']['subject_sha256']==digest({k:v for k,v in c.items() if k!='review'}),'DOCUMENT_READBACK')
    a=envelope(p['authorization']);permission=plan['permission'];review=c['review']
    require(set(a)==set('version reference plan_sha256 custody_id owner authorized_at expires_at maximum_attempts maximum_additional_spend'.split())
        and a['version']==AUTH and a['owner']=='Robert Velarde' and a['reference']==plan['authorization_ref'] and a['custody_id']==plan['custody_id']
        and a['maximum_attempts']==sum(1+len(d['redirects']) for d in plan['documents']) and a['maximum_additional_spend']==0
        and checked_clock(a['expires_at'])==checked_clock(plan['window']['expires_at']),'DOCUMENT_READBACK')
    require(set(permission)==set('reviewer reviewed_at effective_from effective_until documents_sha256 private_acquisition private_retention basis'.split())
        and permission['reviewer']=='Robert Velarde' and permission['private_acquisition'] is True and permission['private_retention'] is True
        and permission['documents_sha256']==digest(plan['documents']) and isinstance(permission['basis'],str) and len(permission['basis'])>=16
        and set(review)=={'reviewer','reviewed_at','subject_sha256','status','basis'} and review['reviewer']=='Robert Velarde'
        and review['status']=='OWNER_REVIEWED' and isinstance(review['basis'],str) and len(review['basis'])>=16,'DOCUMENT_READBACK')
    require(all(p[k] is False for k in ('source_acceptance','owner_reviewed','inference','publication')) and p['wager_action']=='PASS' and p['stake']==0,'DOCUMENT_AUTHORITY')
    require(len(p['entries'])==p['attempted']<=plan['limits']['max_attempts'],'DOCUMENT_READBACK')
    for i,e in enumerate(p['entries']):
        require(set(e)=={'attempt','document','url','request_started_at','plan_sha256'} and e['attempt']==i+1 and e['plan_sha256']==p['plan']['sha256'],'DOCUMENT_READBACK')
        at=checked_clock(e['request_started_at']);a=envelope(p['authorization']);permission=plan['permission']
        require(any(e['document']==d['id'] and e['url'] in [d['url'],*d['redirects']] for d in plan['documents']),'DOCUMENT_READBACK')
        require(checked_clock(plan['window']['not_before'])<=at<checked_clock(plan['window']['not_after']) and checked_clock(a['authorized_at'])<=at<checked_clock(a['expires_at'])
            and checked_clock(c['review']['reviewed_at'])<at and checked_clock(permission['reviewed_at'])<at
            and checked_clock(permission['effective_from'])<=at<checked_clock(permission['effective_until']),'DOCUMENT_CLOCK')
    require(sha((''.join(custody.encode(e)+'\n' for e in p['entries'])).encode())==p['journal_sha256'],'DOCUMENT_READBACK')
    if p['diagnostics'] is not None:diagnostics.checked(p['diagnostics'])
    docs={d['id']:d for d in plan['documents']};out={};total_wire=total_raw=0
    for obj in p['objects']:
        require(set(obj)==set('id source_url final_url redirects receipt status response_metadata wire_bytes_base64 wire_sha256 body_bytes_base64 body_sha256'.split()),'DOCUMENT_READBACK')
        require(obj['id'] in docs and obj['id'] not in out,'DOCUMENT_READBACK');d=docs[obj['id']]
        wire=base64.b64decode(obj['wire_bytes_base64'],validate=True);raw=base64.b64decode(obj['body_bytes_base64'],validate=True)
        require(sha(wire)==obj['wire_sha256'] and sha(raw)==obj['body_sha256'] and len(wire)<=plan['limits']['wire_body_bytes']
            and decode(wire,obj['response_metadata']['content_encoding'],plan['limits']['decoded_body_bytes'])==raw,'DOCUMENT_READBACK')
        body_ok(raw,d['format'],public_guide_examples=public_examples(plan,d),html_syntax=plan['version']==VERSION_V3);total_wire+=len(wire);total_raw+=len(raw)
        r=obj['receipt'];times=[checked_clock(r[k]) for k in ('request_started_at','received_at','observed_at')]
        require(set(r)==set('source_id endpoint format evidence_label request_started_at received_at observed_at observation_meaning provider_clock_meaning'.split())
            and r['format']==('text' if d['format']=='html' else d['format']) and r['evidence_label']==('SYNTHETIC' if plan['evidence_label']=='SYNTHETIC' else 'RETAINED')
            and r['observation_meaning']=='actual_local_first_observation_of_complete_decoded_body' and r['provider_clock_meaning']=='attributed_document_no_publication_clock'
            and set(obj['response_metadata'])=={'content_type','content_encoding','content_length','http_date','last_modified','etag_sha256','publisher_available_at'}
            and obj['response_metadata']['publisher_available_at']=='UNKNOWN','DOCUMENT_READBACK')
        require(times[-1]<checked_clock(plan['window']['expires_at']),'DOCUMENT_CLOCK')
        require(times==sorted(times) and obj['status']==200 and obj['source_url']==d['url']
            and obj['final_url'] in [d['url'],*d['redirects']] and r['endpoint']==obj['final_url']
            and r['source_id']==d['source_id'],'DOCUMENT_READBACK')
        matching=[e for e in p['entries'] if e['document']==d['id']]
        require([e['url'] for e in matching]==[d['url'],*([obj['final_url']] if obj['final_url']!=d['url'] else [])]
            and matching[-1]['request_started_at']==r['request_started_at'],'DOCUMENT_READBACK')
        if len(matching)==2:
            require(len(obj['redirects'])==1,'DOCUMENT_READBACK');hop=obj['redirects'][0]
            require(set(hop)=={'url','status','request_started_at','received_at'} and hop['url']==d['url'] and hop['status'] in {301,302,303,307,308}
                and hop['request_started_at']==matching[0]['request_started_at']
                and checked_clock(hop['request_started_at'])<=checked_clock(hop['received_at'])<=checked_clock(r['request_started_at']),'DOCUMENT_READBACK')
        else:require(obj['redirects']==[],'DOCUMENT_READBACK')
        out[obj['id']]=(raw,obj)
    require(total_wire<=plan['limits']['aggregate_wire_bytes'] and total_raw<=plan['limits']['aggregate_decoded_bytes'],'DOCUMENT_AGGREGATE_SIZE')
    require(p['status']!='CAPTURED_UNREVIEWED' or set(out)==set(docs),'DOCUMENT_READBACK')
    require(len(custody.encode(bundle).encode())<=plan['limits']['serialized_bundle_bytes'],'DOCUMENT_SERIALIZED_SIZE')
    return out


def read_back(path,journal):
    raw=Path(path).read_bytes();require(len(raw)<=LIMITS['serialized_bundle_bytes'],'DOCUMENT_SERIALIZED_SIZE')
    bundle=json.loads(raw,object_pairs_hook=custody.unique);read_bundle(bundle)
    require(sha(Path(journal).read_bytes())==bundle['payload']['journal_sha256'],'DOCUMENT_READBACK');return bundle


def project(bundle, identifier, *, prepared_at=None):
    """Caller supplies extraction time; this operation creates no review."""
    raw,obj=read_bundle(bundle)[identifier];receipt=deepcopy(obj['receipt'])
    plan=bundle['payload']['plan']['payload'];document=next(d for d in plan['documents'] if d['id']==identifier)
    if public_examples(plan,document):
        if plan['version']==VERSION_V3:
            from app_core.nfl_html_guide_custody import retained
            return retained(raw,receipt=receipt)
        from app_core.nfl_public_guide_custody import retained
        return retained(raw,receipt=receipt)
    if receipt['format']=='pdf':
        from app_core.nfl_pdf_evidence import retained_pdf
        require(prepared_at is not None and checked_clock(receipt['observed_at'])<=checked_clock(prepared_at),'DOCUMENT_CLOCK')
        return retained_pdf(raw,receipt=receipt,prepared_at=prepared_at)
    return custody.retained(raw,receipt=receipt)


def report(bundle):
    p=envelope(bundle)
    return dict(status=p['status'],reason=p['reason'],attempts=p['attempted'],bundle_sha256=bundle['sha256'],journal_sha256=p['journal_sha256'],
        bodies=[dict(id=o['id'],bytes=len(base64.b64decode(o['body_bytes_base64'])),body_sha256=o['body_sha256'],wire_sha256=o['wire_sha256'],receipt=o['receipt']) for o in p['objects']],diagnostics=p['diagnostics'],wager_action='PASS',stake=0)
