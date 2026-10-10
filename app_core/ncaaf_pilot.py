"""Fixed-plan NCAAF acquisition seam. Default planning is transport/storage free.

Acquisition stops at unaccepted decoded bytes. Admission, new inference and
existing private retention are explicitly separate; no general live runner.
"""
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlencode, unquote
import zlib

from app_core import ncaaf_response_custody as custody
from app_core import ncaaf_model_compatibility as model
from app_core.ncaaf_history import timestamp

VERSION = 'ncaaf-pilot-plan-v1'
BUNDLE_VERSION = 'ncaaf-pilot-unaccepted-bodies-v1'
AUTH_VERSION = 'ncaaf-pilot-collection-authorization-v1'
HOSTS = {'cfbd':'api.collegefootballdata.com','odds_api':'api.the-odds-api.com'}
# Separate owner authorization, not independent source/model acceptance.
# No UI upload, plan, execution or test populates production catalogs.
AUTHORIZED_COLLECTIONS = {}
ACCEPTED_ADVANCE_PERMISSIONS = {}
AUTHORIZED_CUSTODY_ROOTS = {}
MAX_REQUESTS = 16
MAX_WIRE = 512*1024


def require(ok, code):
    if not ok: raise ValueError(code)


def encode(value): return model.encode(value)
def seal(payload): return dict(payload=deepcopy(payload),sha256=model.digest(payload))
def utc(): return datetime.now(timezone.utc).isoformat()


def validate_plan(plan, *, executable=False):
    require(isinstance(plan,dict) and set(plan)=={'payload','sha256'},'NCAAF_PILOT_PLAN_SCHEMA')
    p=plan['payload']
    require(isinstance(p,dict) and model.digest(p)==plan['sha256'],'NCAAF_PILOT_PLAN_INTEGRITY')
    require(set(p)==set('version evidence_label target offer requests limits execution_window advance_permissions collection_authorization_ref custody_id unresolved'.split())
        and p['version']==VERSION and p['evidence_label'] in {'SYNTHETIC','RETAINED','PROPOSAL'},'NCAAF_PILOT_PLAN_SCHEMA')
    custody._credentials(p)
    require(set(p['target'])==set('canonical_event_id provider_event_id home_team away_team start_utc neutral_site'.split()),'NCAAF_PILOT_TARGET')
    require(set(p['offer'])==set('market side signed_line price bookmaker product period settlement_review_sha256'.split()),'NCAAF_PILOT_OFFER')
    require(p['offer']['market'] in {'spread_home','spread_away','total_over','total_under'} and p['offer']['bookmaker']!='novig','NCAAF_PILOT_OFFER')
    limits=p['limits']
    require(set(limits)==set('max_attempts max_objects max_body_bytes max_aggregate_bytes connect_seconds read_seconds total_seconds'.split()),'NCAAF_PILOT_LIMITS')
    require(type(limits['max_attempts']) is int and 1<=limits['max_attempts']<=MAX_REQUESTS
        and type(limits['max_objects']) is int and 1<=limits['max_objects']<=custody.MAX_OBJECTS
        and type(limits['max_body_bytes']) is int and 1<=limits['max_body_bytes']<=custody.MAX_BODY_BYTES
        and type(limits['max_aggregate_bytes']) is int and 1<=limits['max_aggregate_bytes']<=custody.MAX_TOTAL_BODY_BYTES,'NCAAF_PILOT_LIMITS')
    require(all(type(limits[k]) in (int,float) and 0<limits[k]<=120 for k in ('connect_seconds','read_seconds','total_seconds'))
        and limits['connect_seconds']<=limits['total_seconds'] and limits['read_seconds']<=limits['total_seconds'],'NCAAF_PILOT_LIMITS')
    reqs=p['requests']
    require(isinstance(reqs,list) and 1<=len(reqs)<=limits['max_attempts'] and len(reqs)<=limits['max_objects'],'NCAAF_PILOT_BUDGET')
    ids=[]; scopes=[]; odds=[]
    for r in reqs:
        require(set(r)=={'id','provider','host','endpoint','params','max_attempts'} and r['max_attempts']==1,'NCAAF_PILOT_REQUEST')
        require(r['provider'] in HOSTS and r['host']==HOSTS[r['provider']],'NCAAF_PILOT_HOST')
        meta=dict(provider=r['provider'],endpoint=r['endpoint'],request_scope=r['params'],status=200,
            representation=custody.REPRESENTATION,content_type='application/json',complete=True,received_at='2000-01-01T00:00:00Z',receipt_clock_meaning=custody.RECEIPT_MEANING)
        custody._metadata(meta)
        ids.append(r['id']); scopes.append(model.digest(meta['request_scope']|{'provider':r['provider'],'endpoint':r['endpoint']}))
        if r['provider']=='odds_api': odds.append(r)
    require(all(isinstance(i,str) and i and len(i)<=64 for i in ids) and len(ids)==len(set(ids)) and len(scopes)==len(set(scopes)),'NCAAF_PILOT_DUPLICATE_REQUEST')
    require(len(odds)==1 and reqs[-1]==odds[0] and reqs[0]['provider']=='cfbd' and reqs[0]['endpoint']=='games'
        and all(r['provider']=='cfbd' and r['endpoint']=='games/teams' for r in reqs[1:-1]),'NCAAF_PILOT_REQUEST_ORDER')
    op=odds[0]['params'];market='spreads' if p['offer']['market'].startswith('spread') else 'totals'
    require(op['markets']==market and op['bookmakers']==p['offer']['bookmaker'] and ',' not in op['bookmakers'],'NCAAF_PILOT_OFFER')
    w=p['execution_window']
    require(set(w)=={'not_before','not_after','authorization_expires_at'} and all(timestamp(v) is not None for v in w.values())
        and timestamp(w['not_before'])<timestamp(w['not_after'])<=timestamp(w['authorization_expires_at']),'NCAAF_PILOT_WINDOW')
    start=timestamp(p['target']['start_utc'])
    if start is not None:
        require(all(r['params']['year']==start.year for r in reqs if r['provider']=='cfbd'),'NCAAF_PILOT_SEASON_SCOPE')
    require(isinstance(p['unresolved'],list) and isinstance(p['advance_permissions'],list) and isinstance(p['custody_id'],str) and p['custody_id'],'NCAAF_PILOT_PLAN_SCHEMA')
    if executable:
        require(not p['unresolved'] and p['evidence_label']!='PROPOSAL','NCAAF_PILOT_PREREQUISITES')
        t=p['target'];o=p['offer'];start=timestamp(t['start_utc'])
        require(all(isinstance(t[k],str) and t[k] for k in ('canonical_event_id','provider_event_id','home_team','away_team'))
            and t['home_team']!=t['away_team'] and type(t['neutral_site']) is bool and start is not None,'NCAAF_PILOT_TARGET')
        require(timestamp(w['not_after'])<start and timestamp(op['commenceTimeFrom'])<=start<timestamp(op['commenceTimeTo']),'NCAAF_PILOT_WINDOW')
        # Null values mean a prospectively observed half-point offer within this
        # fixed side/book/product scope, never a fabricated current offer.
        require((o['signed_line'] is None or (type(o['signed_line']) in (int,float) and abs(o['signed_line']%1)==.5))
            and (o['price'] is None or (type(o['price']) in (int,float) and abs(o['price'])>=100)) and o['side']==o['market'].split('_')[1]
            and o['period']=='full_game' and isinstance(o['product'],str) and o['product']
            and isinstance(o['settlement_review_sha256'],str) and len(o['settlement_review_sha256'])==64,'NCAAF_PILOT_OFFER')
    return deepcopy(p)


def planning(plan):
    p=validate_plan(plan)
    return dict(version=VERSION,plan_sha256=plan['sha256'],status='BLOCKED' if p['unresolved'] else 'AWAITING_SEPARATE_COLLECTION_AUTHORIZATION',
        requests=len(p['requests']),max_attempts=p['limits']['max_attempts'],unresolved=deepcopy(p['unresolved']),
        network_requests=0,credential_loads=0,storage_operations=0,accepted=False,inference=False)


def authorize(plan, authorization, at):
    p=validate_plan(plan,executable=True);now=timestamp(at)
    require(now is not None,'NCAAF_PILOT_CLOCK')
    w=p['execution_window']
    require(timestamp(w['not_before'])<=now<=timestamp(w['not_after']) and now<timestamp(w['authorization_expires_at']),'NCAAF_PILOT_WINDOW_EXPIRED')
    require(isinstance(authorization,dict) and set(authorization)=={'payload','sha256'}
        and model.digest(authorization['payload'])==authorization['sha256'],'NCAAF_PILOT_AUTHORIZATION_MISSING')
    a=authorization['payload']
    require(set(a)==set('version reference plan_sha256 custody_id authorized_at expires_at owner'.split()) and a['version']==AUTH_VERSION
        and a['reference']==p['collection_authorization_ref'] and a['plan_sha256']==plan['sha256'] and a['custody_id']==p['custody_id'],'NCAAF_PILOT_AUTHORIZATION_CONFLICT')
    require(AUTHORIZED_COLLECTIONS.get(a['reference'])==authorization['sha256'],'NCAAF_PILOT_AUTHORIZATION_UNTRUSTED')
    require(timestamp(a['authorized_at']) is not None and timestamp(a['expires_at']) is not None
        and timestamp(a['authorized_at'])<=now<timestamp(a['expires_at'])<=timestamp(w['authorization_expires_at']),'NCAAF_PILOT_AUTHORIZATION_EXPIRED')
    required={'cfbd','odds_api'}
    seen=set()
    for receipt in p['advance_permissions']:
        require(isinstance(receipt,dict) and set(receipt)==set('reference provider reviewed_at effective_from effective_until permitted_use requests_sha256'.split()),'NCAAF_PILOT_PERMISSION')
        provider=receipt['provider']
        require(provider in required and provider not in seen and ACCEPTED_ADVANCE_PERMISSIONS.get(receipt['reference'])==model.digest(receipt),'NCAAF_PILOT_PERMISSION_UNTRUSTED')
        req=[r for r in p['requests'] if r['provider']==provider]
        require(receipt['requests_sha256']==model.digest(req) and receipt['permitted_use']=='private_prospective_capture','NCAAF_PILOT_PERMISSION_SCOPE')
        clocks=[timestamp(receipt[k]) for k in ('reviewed_at','effective_from','effective_until')]
        require(all(clocks) and clocks[0]<now and clocks[1]<=now<timestamp(w['not_after'])<clocks[2],'NCAAF_PILOT_PERMISSION_CLOCK')
        seen.add(provider)
    require(seen==required,'NCAAF_PILOT_PERMISSION_MISSING')
    return p


def _timed(operation, deadline, monotonic, cancelled, close):
    """Bound caller wait even when OS DNS/connect/read ignores its timeout.

    A late worker cannot start another request: cancellation/deadline checks
    precede GET, and execution stops at the first timeout. No retry or resume.
    """
    require(not cancelled.is_set() and monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
    done=threading.Event();result={}
    def work():
        try:
            require(not cancelled.is_set() and monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
            result['value']=operation()
        except BaseException as exc:result['error']=exc
        finally:
            if cancelled.is_set():close()
            done.set()
    threading.Thread(target=work,daemon=True).start()
    if not done.wait(max(0,deadline-monotonic())):
        cancelled.set();close();raise ValueError('NCAAF_PILOT_DEADLINE')
    require(not cancelled.is_set() and monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
    if 'error' in result:raise result['error']
    return result['value']


class HttpsTransport:
    """One stdlib GET, no redirects/retries. Credentials resolved only on call.

    Runner retains only a credential-free request. HTTP exceptions are never
    logged; caller exposes stable codes only. Stream reads honor remaining time.
    """
    def __init__(self, credentials): self.credentials=credentials

    def __call__(self, request, limits, deadline, monotonic):
        require(request['host']==HOSTS.get(request['provider']),'NCAAF_PILOT_HOST')
        custody._metadata(dict(provider=request['provider'],endpoint=request['endpoint'],request_scope=request['params'],status=200,
            representation=custody.REPRESENTATION,content_type='application/json',complete=True,received_at='2000-01-01T00:00:00Z',receipt_clock_meaning=custody.RECEIPT_MEANING))
        require(monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
        cancelled=threading.Event()
        key=_timed(lambda:self.credentials(request['provider']),deadline,monotonic,cancelled,lambda:None)
        require(isinstance(key,str) and 0<len(key)<=4096 and '\r' not in key and '\n' not in key,'NCAAF_PILOT_CREDENTIAL_UNAVAILABLE')
        self._secret=key
        params=deepcopy(request['params']);headers={'Accept':'application/json','Accept-Encoding':'identity'}
        if request['provider']=='cfbd':headers['Authorization']='Bearer '+key
        else:params['apiKey']=key
        conn=http.client.HTTPSConnection(request['host'],timeout=min(limits['connect_seconds'],max(.001,deadline-monotonic())))
        # Explicit connect only. A cancelled writer must not reopen a socket
        # through HTTPConnection.send's default auto_open behavior.
        conn.auto_open=False
        try:
            _timed(conn.connect,min(deadline,monotonic()+limits['connect_seconds']),monotonic,cancelled,conn.close)
            require(monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
            conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
            _timed(lambda:conn.request('GET','/'+request['endpoint']+'?'+urlencode(params),headers=headers),min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
            require(monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
            conn.sock.settimeout(min(limits['read_seconds'],deadline-monotonic()))
            response=_timed(conn.getresponse,min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
            require(monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
        except Exception as exc:
            conn.close()
            if isinstance(exc,ValueError) and str(exc)=='NCAAF_PILOT_DEADLINE':raise
            raise ValueError('NCAAF_PILOT_TRANSPORT_FAILURE') from None
        class Stream:
            status=response.status
            headers={k.lower():v for k,v in response.getheaders()}
            def chunks(self):
                while True:
                    remaining=deadline-monotonic()
                    require(remaining>0,'NCAAF_PILOT_DEADLINE')
                    conn.sock.settimeout(min(limits['read_seconds'],remaining))
                    block=_timed(lambda:response.read1(8192),min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
                    if not block:break
                    yield block
            def close(self):conn.close()
        return Stream()

    def validate_body(self, raw):
        require(self._secret.encode() not in raw,'NCAAF_PILOT_CREDENTIAL_IN_BODY')
        # JSON escaping and URL encoding must not conceal an echoed credential.
        # Reject the original body; never redact bytes and call it complete.
        pending=[custody._parse(raw)]
        while pending:
            item=pending.pop()
            if isinstance(item,str):require(self._secret not in unquote(item),'NCAAF_PILOT_CREDENTIAL_IN_BODY')
            elif isinstance(item,dict):pending.extend(item.keys());pending.extend(item.values())
            elif isinstance(item,list):pending.extend(item)


def _append(journal, value):
    journal.write(encode(value)+b'\n');journal.flush();os.fsync(journal.fileno())


def _read(stream, limit, deadline, monotonic):
    h={k.lower():str(v) for k,v in stream.headers.items()}
    require(stream.status==200,'NCAAF_PILOT_AUTH_OR_RATE_LIMIT' if stream.status in {401,403,429} else 'NCAAF_PILOT_HTTP_STATUS')
    require(not any(h.get(k) for k in ('location','x-next-page','x-next-page-token')) and 'next' not in h.get('link','').lower(),'NCAAF_PILOT_REDIRECT_OR_PAGINATION')
    require(h.get('content-type','').split(';')[0].strip()=='application/json','NCAAF_PILOT_CONTENT_TYPE')
    encoding=h.get('content-encoding','identity').lower()
    require(encoding in {'identity','gzip'},'NCAAF_PILOT_CONTENT_ENCODING')
    declared=h.get('content-length')
    require(declared is None or (declared.isdigit() and 0<int(declared)<=MAX_WIRE),'NCAAF_PILOT_BODY_LIMIT')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS) if encoding=='gzip' else None
    body=bytearray();wire=0
    for chunk in stream.chunks():
        require(monotonic()<=deadline,'NCAAF_PILOT_DEADLINE')
        require(isinstance(chunk,bytes) and 0<len(chunk)<=MAX_WIRE,'NCAAF_PILOT_INCOMPLETE_BODY')
        wire+=len(chunk);require(wire<=MAX_WIRE,'NCAAF_PILOT_BODY_LIMIT')
        block=decoder.decompress(chunk,limit-len(body)+1) if decoder else chunk
        body.extend(block)
        require(len(body)<=limit and (decoder is None or not decoder.unconsumed_tail),'NCAAF_PILOT_BODY_LIMIT')
    require(monotonic()<=deadline,'NCAAF_PILOT_DEADLINE')
    require(declared is None or wire==int(declared),'NCAAF_PILOT_INCOMPLETE_BODY')
    require(decoder is None or (decoder.eof and not decoder.unused_data),'NCAAF_PILOT_INCOMPLETE_BODY')
    raw=bytes(body);custody._parse(raw)
    return raw


def acquire(plan, authorization, *, root, transport, wall=utc, monotonic=time.monotonic):
    """Explicit execution only. Durable attempt before GET; never resumes/repeats.

    No inference, storage connector or acceptance operation. Failure retains a
    labelled incomplete bundle and spent-attempt journal; never truncates bodies.
    """
    p=authorize(plan,authorization,wall())
    root=Path(root);require(root.is_dir(),'NCAAF_PILOT_CUSTODY_DIRECTORY_REQUIRED')
    configured=AUTHORIZED_CUSTODY_ROOTS.get(p['custody_id'])
    require(configured is not None and Path(configured).resolve()==root.resolve(),'NCAAF_PILOT_CUSTODY_IDENTITY_CONFLICT')
    journal_path=root/(plan['sha256']+'.attempts.jsonl')
    try:journal=journal_path.open('xb')
    except FileExistsError:raise ValueError('NCAAF_PILOT_ALREADY_ATTEMPTED_NO_RESUME') from None
    deadline=monotonic()+p['limits']['total_seconds'];objects=[];attempts=0;total=0;code=None
    try:
        with journal:
            _append(journal,dict(plan_sha256=plan['sha256'],authorization_sha256=authorization['sha256'],custody_id=p['custody_id']))
            for request in p['requests']:
                authorize(plan,authorization,wall())
                require(monotonic()<deadline,'NCAAF_PILOT_DEADLINE')
                require(attempts<p['limits']['max_attempts'],'NCAAF_PILOT_BUDGET')
                require(total<p['limits']['max_aggregate_bytes'],'NCAAF_PILOT_AGGREGATE_LIMIT')
                started=wall();attempts+=1
                _append(journal,dict(request=request,attempt=attempts,requested_at=started,request_clock_meaning='actual_local_request_start'))
                stream=None
                try:
                    stream=transport(deepcopy(request),deepcopy(p['limits']),deadline,monotonic)
                    raw=_read(stream,p['limits']['max_body_bytes'],deadline,monotonic)
                    if isinstance(transport,HttpsTransport):transport.validate_body(raw)
                except ValueError:raise
                except Exception:raise ValueError('NCAAF_PILOT_TRANSPORT_FAILURE') from None
                finally:
                    if stream is not None:stream.close()
                received=wall();authorize(plan,authorization,received)
                require(timestamp(received)>=timestamp(started),'NCAAF_PILOT_CLOCK')
                total+=len(raw);require(total<=p['limits']['max_aggregate_bytes'],'NCAAF_PILOT_AGGREGATE_LIMIT')
                meta=dict(provider=request['provider'],endpoint=request['endpoint'],request_scope=request['params'],status=200,
                    representation=custody.REPRESENTATION,content_type='application/json',complete=True,received_at=received,receipt_clock_meaning=custody.RECEIPT_MEANING)
                custody._metadata(meta)
                objects.append(dict(request_id=request['id'],requested_at=started,request_clock_meaning='actual_local_request_start',
                    metadata=meta,body_b64=base64.b64encode(raw).decode(),body_bytes=len(raw),body_sha256=hashlib.sha256(raw).hexdigest()))
                _append(journal,dict(request_id=request['id'],received_at=received,body_sha256=objects[-1]['body_sha256']))
    except Exception as exc:
        reason=str(exc);code=reason if reason.startswith(('NCAAF_PILOT_','NCAAF_CUSTODY_')) and reason.replace('_','').isalnum() else 'NCAAF_PILOT_STOPPED'
    bundle=seal(dict(version=BUNDLE_VERSION,evidence_label=p['evidence_label'],plan=deepcopy(plan),authorization_sha256=authorization['sha256'],
        authorization=deepcopy(authorization),
        status='INCOMPLETE' if code else 'CAPTURED_UNACCEPTED',reason=code,attempted_requests=attempts,completed_objects=len(objects),
        objects=objects,accepted=False,inference=False,scientific_acceptance=False,wagering_authority=False))
    path=root/(plan['sha256']+'.unaccepted.json')
    with path.open('xb') as f:f.write(encode(bundle));f.flush();os.fsync(f.fileno())
    require(path.read_bytes()==encode(bundle),'NCAAF_PILOT_PRIVATE_READBACK')
    return bundle


def project_capture(bundle, *, native_batches, quote, terms):
    """Separate exact projections into #2408 custody objects, no admission/inference."""
    require(model.digest(bundle['payload'])==bundle['sha256'],'NCAAF_PILOT_BUNDLE_INTEGRITY')
    b=bundle['payload'];require(b['version']==BUNDLE_VERSION and b['status']=='CAPTURED_UNACCEPTED' and not b['accepted'] and not b['inference'],'NCAAF_PILOT_UNACCEPTED_BUNDLE')
    p=validate_plan(b['plan'],executable=True)
    require(len(b['objects'])==len(p['requests']),'NCAAF_PILOT_INCOMPLETE_BODY')
    result=[];total=0
    for request,obj in zip(p['requests'],b['objects']):
        require(type(obj['body_bytes']) is int and 0<obj['body_bytes']<=p['limits']['max_body_bytes']
            and isinstance(obj['body_b64'],str) and len(obj['body_b64'])<=((p['limits']['max_body_bytes']+2)//3)*4,'NCAAF_PILOT_BODY_LIMIT')
        total+=obj['body_bytes'];require(total<=p['limits']['max_aggregate_bytes'],'NCAAF_PILOT_AGGREGATE_LIMIT')
        raw=base64.b64decode(obj['body_b64'],validate=True)
        require(obj['request_id']==request['id'] and len(raw)==obj['body_bytes'] and hashlib.sha256(raw).hexdigest()==obj['body_sha256'],'NCAAF_PILOT_BUNDLE_INTEGRITY')
        require(obj['metadata']['request_scope']==request['params'] and obj['metadata']['provider']==request['provider'] and obj['metadata']['endpoint']==request['endpoint'],'NCAAF_PILOT_BUNDLE_INTEGRITY')
        if request['provider']=='cfbd':
            require(request['id'] in native_batches,'NCAAF_PILOT_NATIVE_PROJECTION_MISSING')
            result.append(custody.capture(raw,obj['metadata'],native_batch=native_batches[request['id']]))
        else:
            o=p['offer'];t=p['target']
            require(model.digest(terms)==o['settlement_review_sha256'],'NCAAF_PILOT_TARGET_OFFER_CONFLICT')
            require(type(quote.get('point')) in (int,float) and abs(quote['point']%1)==.5
                and type(quote.get('price')) in (int,float) and abs(quote['price'])>=100,'NCAAF_PILOT_TARGET_OFFER_CONFLICT')
            require(all(quote.get(k)==v for k,v in {'provider_event_id':t['provider_event_id'],'event_home_team':t['home_team'],
                'event_away_team':t['away_team'],'event_start_utc':t['start_utc'],'market_type':o['market'],'point':o['signed_line'],
                'price':o['price'],'book':o['bookmaker'],'product':o['product'],'period':o['period']}.items() if v is not None),'NCAAF_PILOT_TARGET_OFFER_CONFLICT')
            result.append(custody.capture(raw,obj['metadata'],quote=quote,terms=terms))
    return result


def _capture_binding(bundle,packet):
    require(isinstance(bundle,dict) and model.digest(bundle['payload'])==bundle['sha256'],'NCAAF_PILOT_BUNDLE_REQUIRED_OR_CORRUPT')
    b=bundle['payload'];require(b['status']=='CAPTURED_UNACCEPTED' and not b['accepted'] and not b['inference'],'NCAAF_PILOT_UNACCEPTED_BUNDLE')
    p=validate_plan(b['plan'],executable=True)
    require(b['version']==BUNDLE_VERSION and b['completed_objects']==b['attempted_requests']==len(p['requests'])
        and b['evidence_label']==packet['payload']['evidence_label']==p['evidence_label']
        and b['scientific_acceptance'] is False and b['wagering_authority'] is False,'NCAAF_PILOT_BUNDLE_INTEGRITY')
    require(b['authorization']['sha256']==b['authorization_sha256'] and len(b['objects'])==len(p['requests'])==len(packet['payload']['response_objects']),'NCAAF_PILOT_BUNDLE_INTEGRITY')
    event=packet['payload']['native_packet']['payload']['observation']['payload']['event']
    require(event['canonical_event_id']==p['target']['canonical_event_id'] and event['neutral_site']==p['target']['neutral_site'],'NCAAF_PILOT_TARGET_OFFER_CONFLICT')
    observation=packet['payload']['native_packet']['payload']['observation']['payload']
    native_objects=packet['payload']['native_packet']['payload']['dependency_objects']
    require(len(native_objects)+1==len(p['requests']),'NCAAF_PILOT_NATIVE_PROJECTION_MISSING')
    batches={r['id']:json.loads(base64.b64decode(obj['bytes_b64'],validate=True)) for r,obj in zip(p['requests'],native_objects)}
    projections=project_capture(bundle,native_batches=batches,quote=observation['quote'],terms=observation['source_review']['terms_review'])
    require(projections==packet['payload']['response_objects'],'NCAAF_PILOT_BUNDLE_INTEGRITY')
    receipts=[]
    for req,obj,response in zip(p['requests'],b['objects'],packet['payload']['response_objects']):
        authorize(b['plan'],b['authorization'],obj['requested_at'])
        raw=base64.b64decode(obj['body_b64'],validate=True);v=response['payload']
        require(obj['request_id']==req['id'] and obj['metadata']==v['metadata'] and obj['body_b64']==v['body_b64']
            and obj['body_sha256']==hashlib.sha256(raw).hexdigest()==v['body_sha256']
            and len(raw)==obj['body_bytes']==v['body_bytes'],'NCAAF_PILOT_BUNDLE_INTEGRITY')
        require(timestamp(obj['requested_at'])<=timestamp(obj['metadata']['received_at']),'NCAAF_PILOT_CLOCK')
        require(obj['request_clock_meaning']=='actual_local_request_start','NCAAF_PILOT_CLOCK')
        receipts.append(dict(request=req,requested_at=obj['requested_at'],request_clock_meaning=obj['request_clock_meaning'],
            metadata=obj['metadata'],body_sha256=obj['body_sha256'],body_bytes=obj['body_bytes']))
    return seal(dict(version='ncaaf-pilot-capture-binding-v1',plan=b['plan'],authorization=b['authorization'],
        bundle_sha256=bundle['sha256'],request_receipts=receipts,accepted=False,scientific_acceptance=False,wagering_authority=False))


def admitted_analysis(packet, source, *, inventory, bundle=None):
    """Existing selected market caller only; no general fetch, blend or activation."""
    import pandas as pd
    from app_core import ncaaf_pipeline_evidence as adapter
    from app_core.market_probability_model import predict_market_probabilities
    require(str(source.get('league','')).upper()=='NCAAF','NCAAF_PILOT_TARGET')
    frame=pd.DataFrame([deepcopy(source)]);frame.attrs['ncaaf_schedule']=deepcopy(inventory)
    start=timestamp(source.get('game_start_utc'))
    require(start is not None,'NCAAF_PILOT_TARGET')
    from app_core.slate_coverage import ET,build_coverage,native_ncaaf
    day=start.astimezone(ET).date().isoformat()
    # Derived presentation keys, never a replacement start/quote clock.
    if 'date' not in frame:frame['date']=pd.Timestamp(start)
    if 'game_date' not in frame:frame['game_date']=day
    at=adapter.generated_time()
    loaded=False
    try:
        adapter.load(encode(packet));loaded=True
        require(packet['payload']['version']==custody.VERSION,'NCAAF_PILOT_CUSTODY_PACKET_REQUIRED')
        binding=_capture_binding(bundle,packet)
    except (ValueError,KeyError,TypeError) as exc:
        reason=str(exc) if str(exc).startswith('NCAAF_PILOT_') or str(exc) in adapter.PUBLIC_REASONS else 'NCAAF_PILOT_BUNDLE_INTEGRITY'
        # Existing known public reason; precise pilot cause stays private.
        code=reason if reason in adapter.PUBLIC_REASONS else ('NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE' if bundle is None else 'NCAAF_PACKET_INTEGRITY')
        result=dict(ml_probability=float('nan'),ml_probability_source='',ml_target='',ml_projection=float('nan'),ml_residual_scale=float('nan'),
            ml_feature_quality='unavailable',ml_inference_status='unavailable',ml_unavailable_reason=code)
        from app_core.research_estimate_trace import origin_metadata
        metadata=origin_metadata(source,result,adapter.producer._line(source),generated_at=at)
        metadata,fields=adapter.producer.record(source,result,metadata,at)
        item=json.loads(metadata)
        rejected=dict(version=custody.RESULT_VERSION,status='unavailable',reason=code,inference_time=at,
            scientific_acceptance=False,wagering_authority=False,live_stake=0)
        if loaded:rejected['original_packet']=deepcopy(packet)
        binding=seal(dict(version='ncaaf-pilot-capture-binding-v1',status='rejected',reason=reason))
        rejected['pilot_capture']=binding
        item['ncaaf_inputs']=seal(rejected)
        result.update(fields,ml_estimate_metadata=encode(item).decode())
        predictions=pd.DataFrame([result])
    else:
        with adapter.selected([packet]): predictions=predict_market_probabilities(frame)
        item=json.loads(predictions.iloc[0].ml_estimate_metadata)
        item['ncaaf_inputs']['payload']['pilot_capture']=binding
        item['ncaaf_inputs']['sha256']=model.digest(item['ncaaf_inputs']['payload'])
        predictions.loc[predictions.index[0],'ml_estimate_metadata']=encode(item).decode()
    for field in predictions:frame[field]=predictions[field]
    frame.attrs['ncaaf_schedule']=deepcopy(inventory)
    frame=adapter.finish(frame)
    at=json.loads(frame.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['inference_time']
    report=build_coverage([native_ncaaf(inventory,day)] if inventory else [],selected_date=day,as_of=at,
        run_id='ncaaf-pilot-'+model.digest(dict(event=source.get('matchup_id'),at=at)),candidates=frame,leagues=['NCAAF'])
    frame.attrs['slate_coverage']=report
    return frame


def retain_analysis(analysis, *, path):
    """Explicit local existing evidence store; no remote default, restore or sync.

    Existing actual finalization, capture/export/replay/display consumers preserve
    research/raw probability. Empty policies supply no wagering authority.
    """
    path=Path(path)
    require(path.is_file(),'NCAAF_PILOT_EXISTING_PREDICTION_STORE_REQUIRED')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    require({'bundles','snapshots','snapshot_runtime'}<=tables,'NCAAF_PILOT_EXISTING_PREDICTION_STORE_REQUIRED')
    from app_core import prediction_evidence as evidence, research_replay
    from app_core.candidate_evidence_schema import project
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from core.streamlit_pipeline import build_best_picks_df
    from core.live_wager_contract import finalize_live_wagers
    diagnostics={};best=build_best_picks_df(analysis,diagnostics_out=diagnostics)
    audit=project(diagnostics['candidate_authority_df'])
    # Existing finalizer requires a positive denominator. A nominal unit with
    # empty policies supplies no authority; verify its actual outputs stay zero.
    final,_=finalize_live_wagers(audit,best,1,now=datetime.now(timezone.utc),policies={},config={})
    require(all(c['production_bet_amount']==0 for c in final.wager_contract),'NCAAF_PILOT_AUTHORITY_FORBIDDEN')
    # Evidence storage and installed source are different roots. Always bind
    # the actual running repository; a private storage directory has no code.
    source_root=Path(__file__).resolve().parents[1]
    context=evidence.begin_run(dict(sports=['NCAAF'],use_ml=True,use_gemini=False,bankroll=1),path=path,root=source_root)
    captured,card=evidence.capture_run(context,audit,final,analysis,path=path,authoritative_candidates=True)
    from app_core.slate_coverage import build_coverage,native_ncaaf,publication_rows
    original=analysis.attrs['slate_coverage'];inventory=analysis.attrs.get('ncaaf_schedule')
    run_id=str(captured.iloc[0].export_run_id)
    coverage=build_coverage([native_ncaaf(inventory,original['selected_date'])] if inventory else [],selected_date=original['selected_date'],
        as_of=original['as_of'],run_id=run_id,candidates=captured,final=card,leagues=['NCAAF'])
    display,_=publication_rows(card,captured,coverage)
    frames=[per_game_board(display,captured,family=f,novig_only=True,college_fallback=True) for f in ('overall','sides','totals')]
    package=build_package(*frames);validate_package(package)
    receipt=research_replay.retain_export(frames,package,card,captured,path=path)
    return dict(captured=captured,card=card,frames=frames,package=package,export=receipt,coverage=coverage,db=path)
