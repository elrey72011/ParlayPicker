"""NFL release CSV only; explicit two-attempt ceiling, never live analysis."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, urlsplit
import zlib
from app_core import nfl_scoring_profile as profile, nfl_scoring_custody as custody
from app_core import nfl_bounded_acquisition as previous
from app_core.ncaaf_pilot import _timed, _append
from app_core.nfl_source_projection import seal
from app_core.producer_provenance import clock

VERSION='nfl-scoring-original-plan-v2'
AUTH_VERSION='nfl-scoring-original-authorization-v2'
CONFIG_VERSION='nfl-scoring-controlled-bindings-v1'
PRIMARY=dict(id='scores_release',host='github.com',endpoint='/nflverse/nflverse-data/releases/download/schedules/games.csv',params={},method='GET',max_attempts=1)
QUERY_KEYS=sorted(['sp','sv','sr','spr','sig','se','sktid','skoid','skt','ske','sks','skv','rscd','rsct','jwt','response-content-disposition','response-content-type'])
REDIRECT=dict(max_hops=1,host='release-assets.githubusercontent.com',repository_path='/github-production-release-asset/452908115/',filename='games.csv',status=302,query_keys=QUERY_KEYS,sas_protocol='https_only_if_declared')
FRAMING_HEADERS=frozenset(['location','content-type','content-length','content-encoding','transfer-encoding','link','x-next-page','x-next-page-token'])
AUTHORIZED_COLLECTIONS={}
REVIEWED_PERMISSIONS={}
AUTHORIZED_CUSTODY_ROOTS={}
CONTROLLED=ContextVar('nfl_scoring_controlled_bindings',default=None)
require,digest=custody.require,custody.digest
admitted_analysis=previous.admitted_analysis
retain_analysis=previous.retain_analysis


def validate(plan, *, executable=False):
    require(isinstance(plan,dict) and set(plan)=={'payload','sha256'} and digest(plan['payload'])==plan['sha256'], 'NFL_ACQUIRE_PLAN_INTEGRITY')
    p=plan['payload']
    from app_core.ncaaf_response_custody import _credentials
    try: _credentials(p)
    except ValueError: raise ValueError('NFL_ACQUIRE_CREDENTIAL_FORBIDDEN') from None
    require(set(p)==set('version profile evidence_label source target requests redirect limits window permission authorization_ref custody_id unresolved'.split())
        and p['version']==VERSION and p['profile']==profile.PROFILE and p['source']==profile.SOURCE
        and p['evidence_label'] in {'SYNTHETIC','PROPOSAL','RETAINED'}, 'NFL_ACQUIRE_PLAN_SCHEMA')
    require(p['requests']==[PRIMARY] and p['redirect']==REDIRECT, 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
    require(p['target']==dict(season=2026,purpose='prospective_original_scoring_snapshot'), 'NFL_ACQUIRE_TARGET')
    limits=p['limits']
    require(set(limits)==set('max_attempts max_objects max_wire_bytes max_body_bytes max_aggregate_bytes max_packet_bytes max_header_bytes connect_seconds read_seconds total_seconds'.split())
        and type(limits['max_attempts']) is int and limits['max_attempts']==2 and limits['max_objects']==1
        and limits['max_wire_bytes']==profile.MAX_WIRE and limits['max_body_bytes']==profile.MAX_BODY
        and limits['max_aggregate_bytes']==profile.MAX_TOTAL and limits['max_packet_bytes']==profile.MAX_PACKET
        and limits['max_header_bytes']==profile.MAX_HEADER_BYTES
        and all(type(limits[k]) in {int,float} and 0<limits[k]<=120 for k in ('connect_seconds','read_seconds','total_seconds'))
        and limits['connect_seconds']<=limits['total_seconds'] and limits['read_seconds']<=limits['total_seconds'], 'NFL_ACQUIRE_LIMITS')
    w=p['window']
    require(set(w)=={'not_before','not_after','expires_at'} and all(clock(t) for t in w.values())
        and clock(w['not_before'])<clock(w['not_after'])<=clock(w['expires_at']), 'NFL_ACQUIRE_WINDOW')
    require(isinstance(p['unresolved'],list) and custody.known(p['custody_id']) and custody.known(p['authorization_ref']), 'NFL_ACQUIRE_PLAN_SCHEMA')
    if executable:
        require(not p['unresolved'] and p['evidence_label']!='PROPOSAL', 'NFL_ACQUIRE_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p=validate(plan)
    return dict(plan_sha256=plan['sha256'],status='BLOCKED' if p['unresolved'] else 'AWAITING_EXACT_AUTHORIZATION',
        requests_min=1,requests_max=2,odds_credits=0,cfbd_requests=0,maximum_additional_spend=0,
        network_requests=0,credentials=0,storage_operations=0,inference=False,unresolved=p['unresolved'])


def authorize(plan, authorization, at, root):
    p=validate(plan,executable=True); now=clock(at); w=p['window']
    require(now and clock(w['not_before'])<=now<=clock(w['not_after']) and now<clock(w['expires_at']), 'NFL_ACQUIRE_WINDOW_EXPIRED')
    require(isinstance(authorization,dict) and set(authorization)=={'payload','sha256'} and digest(authorization['payload'])==authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION')
    a=authorization['payload']; bindings=CONTROLLED.get()
    approved=bindings['collections'] if bindings else AUTHORIZED_COLLECTIONS
    permissions=bindings['permissions'] if bindings else REVIEWED_PERMISSIONS
    roots=bindings['roots'] if bindings else AUTHORIZED_CUSTODY_ROOTS
    require(set(a)==set('version reference plan_sha256 custody_id owner authorized_at expires_at maximum_additional_spend'.split())
        and a['version']==AUTH_VERSION and a['reference']==p['authorization_ref'] and a['plan_sha256']==plan['sha256']
        and a['custody_id']==p['custody_id'] and a['owner']=='Robert Velarde' and a['maximum_additional_spend']==0
        and approved.get(a['reference'])==authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION_UNTRUSTED')
    require(clock(a['authorized_at']) and clock(a['expires_at']) and clock(a['authorized_at'])<=now<clock(a['expires_at'])<=clock(w['expires_at']), 'NFL_ACQUIRE_AUTHORIZATION_EXPIRED')
    r=p['permission']
    require(isinstance(r,dict) and set(r)==set('reference reviewer reviewed_at effective_from effective_until source_sha256 private_retention private_research_use basis requests_sha256'.split())
        and permissions.get(r['reference'])==digest(r) and r['reviewer']=='Robert Velarde' and r['source_sha256']==digest(profile.SOURCE)
        and r['private_retention'] is True and r['private_research_use'] is True and isinstance(r['basis'],str) and len(r['basis'])>=16
        and r['requests_sha256']==digest(dict(primary=p['requests'],redirect=p['redirect'])), 'NFL_ACQUIRE_PERMISSION_UNTRUSTED')
    require(all(clock(r[k]) for k in ('reviewed_at','effective_from','effective_until'))
        and clock(r['reviewed_at'])<now and clock(r['effective_from'])<=now<clock(w['not_after'])<clock(r['effective_until']), 'NFL_ACQUIRE_PERMISSION_EXPIRED')
    require(p['custody_id'] in roots and Path(roots[p['custody_id']]).resolve()==Path(root).resolve() and Path(root).is_dir(), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    return p


@contextmanager
def controlled(plan, raw, expected_hash, *, root, wall=previous.utc, security=None):
    """Explicit owner-reviewed process configuration; no persistent catalog writes."""
    require(isinstance(raw,bytes) and 0<len(raw)<=128*1024 and hashlib.sha256(raw).hexdigest()==expected_hash, 'NFL_ACQUIRE_CONFIGURATION_INTEGRITY')
    c=json.loads(raw,object_pairs_hook=custody.unique)
    require(set(c)=={'payload','sha256'} and digest(c['payload'])==c['sha256'], 'NFL_ACQUIRE_CONFIGURATION_INTEGRITY')
    p=c['payload']
    require(set(p)=={'version','plan_sha256','permission','authorization','custody','review'} and p['version']==CONFIG_VERSION
        and p['plan_sha256']==plan['sha256'] and p['permission']==plan['payload']['permission'], 'NFL_ACQUIRE_CONFIGURATION_CONFLICT')
    r=p['review']; now=clock(wall())
    require(isinstance(r,dict) and set(r)=={'reviewer','reviewed_at','subject_sha256','status','scope'}
        and r['reviewer']=='Robert Velarde' and r['status']=='OWNER_REVIEWED' and r['scope']=='NFL scoring acquisition bindings only; no source acceptance.'
        and r['subject_sha256']==digest({k:v for k,v in p.items() if k!='review'}) and clock(r['reviewed_at']) and now and clock(r['reviewed_at'])<now, 'NFL_ACQUIRE_CONFIGURATION_REVIEW_MISSING')
    dest=p['custody']
    require(set(dest)=={'id','root','storage_receipt_sha256'} and dest['id']==plan['payload']['custody_id']
        and Path(dest['root']).resolve()==Path(root).resolve() and isinstance(dest['storage_receipt_sha256'],str)
        and re.fullmatch('[0-9a-f]{64}',dest['storage_receipt_sha256']), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    if security is None:
        from app_core.ncaaf_pilot_setup import inspect_security
        security=inspect_security
    found=security(root)
    require(found.get('status')=='VERIFIED' and all(found.get(k) is True for k in ('private_acl','encrypted','no_reparse')), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    a=p['authorization']['payload']; permission=p['permission']
    token=CONTROLLED.set(dict(collections={a['reference']:p['authorization']['sha256']},permissions={permission['reference']:digest(permission)},roots={dest['id']:dest['root']}))
    try:
        authorize(plan,p['authorization'],wall(),root)
        yield p['authorization']
    finally: CONTROLLED.reset(token)


def validate_hop_descriptor(request):
    require(isinstance(request,dict) and set(request)==set(PRIMARY)|{'query_sha256'} and request['id']=='scores_asset'
        and request['host']==REDIRECT['host'] and request['method']=='GET' and request['params']=={} and request['max_attempts']==1
        and isinstance(request['endpoint'],str) and re.fullmatch(re.escape(REDIRECT['repository_path'])+r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',request['endpoint'])
        and isinstance(request['query_sha256'],str) and re.fullmatch('[0-9a-f]{64}',request['query_sha256']), 'NFL_ACQUIRE_REDIRECT_NOT_APPROVED')


def redirect_request(location):
    require(isinstance(location,str) and len(location)<=4096, 'NFL_ACQUIRE_REDIRECT_NOT_APPROVED')
    u=urlsplit(location)
    require(u.scheme=='https' and u.hostname==REDIRECT['host'] and u.port in {None,443} and not u.username and not u.password and not u.fragment, 'NFL_ACQUIRE_REDIRECT_NOT_APPROVED')
    pairs=parse_qsl(u.query,keep_blank_values=True,max_num_fields=24)
    q=dict(pairs)
    require(len(q)==len(pairs) and set(q)<=set(QUERY_KEYS) and all(v for v in q.values()) and q.get('spr','https')=='https'
        and ('sig' in q or 'jwt' in q)
        and re.fullmatch(r'attachment;\s*filename="?games\.csv"?',q.get('response-content-disposition',q.get('rscd',''))), 'NFL_ACQUIRE_REDIRECT_NOT_APPROVED')
    safe=dict(PRIMARY,id='scores_asset',host=REDIRECT['host'],endpoint=u.path,query_sha256=hashlib.sha256(u.query.encode()).hexdigest())
    validate_hop_descriptor(safe)
    actual=dict(safe); actual['endpoint']=u.path+'?'+u.query
    return actual,safe


class HttpsTransport:
    """Manual HTTPS requests; redirects are decisions of acquire, never automatic."""
    def __call__(self,request,limits,deadline,monotonic):
        if request!=PRIMARY:
            actual,safe=redirect_request('https://'+request['host']+request['endpoint'])
            require(actual==request, 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
        conn=http.client.HTTPSConnection(request['host'],timeout=min(limits['connect_seconds'],max(.001,deadline-monotonic())))
        conn.auto_open=False; cancelled=threading.Event()
        try:
            _timed(conn.connect,min(deadline,monotonic()+limits['connect_seconds']),monotonic,cancelled,conn.close)
            conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
            _timed(lambda:conn.request('GET',request['endpoint'],headers={'Accept':'text/csv','Accept-Encoding':'identity'}),min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
            response=_timed(conn.getresponse,min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
            response_headers=checked_headers(response.getheaders())
        except Exception as exc:
            conn.close()
            if isinstance(exc,ValueError) and str(exc) in {'NFL_ACQUIRE_HEADER_CONFLICT','NFL_ACQUIRE_HEADER_LIMIT','NFL_ACQUIRE_REDIRECT_OR_PAGINATION'}:raise
            raise ValueError('NFL_ACQUIRE_TRANSPORT_FAILURE') from None
        class Stream:
            status=response.status
            headers=response_headers
            def chunks(self):
                while True:
                    require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
                    conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
                    block=_timed(lambda:response.read1(8192),min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
                    if not block: break
                    yield block
            def close(self): conn.close()
        return Stream()


def checked_headers(pairs):
    require(len(pairs)<=64 and sum(len(k)+len(v)+4 for k,v in pairs)<=profile.MAX_HEADER_BYTES, 'NFL_ACQUIRE_HEADER_LIMIT')
    names=[k.lower() for k,v in pairs]
    require(all(names.count(k)<=1 for k in FRAMING_HEADERS), 'NFL_ACQUIRE_HEADER_CONFLICT')
    h={k.lower():str(v) for k,v in pairs}
    require(not ('content-length' in h and 'transfer-encoding' in h), 'NFL_ACQUIRE_HEADER_CONFLICT')
    require(not any(h.get(k) for k in ('x-next-page','x-next-page-token')) and 'next' not in h.get('link','').lower(), 'NFL_ACQUIRE_REDIRECT_OR_PAGINATION')
    return h


def headers(stream):return checked_headers(list(stream.headers.items()))


def read(stream,limits,deadline,monotonic):
    h=headers(stream)
    require(stream.status==200 and not h.get('location'), 'NFL_ACQUIRE_HTTP_STATUS')
    require(h.get('content-type','').split(';')[0].strip() in {'text/plain','text/csv','application/csv','application/octet-stream'}, 'NFL_ACQUIRE_CONTENT_TYPE')
    encoding=h.get('content-encoding','identity')
    require(encoding in {'identity','gzip'}, 'NFL_ACQUIRE_CONTENT_ENCODING')
    declared=h.get('content-length')
    require(declared is None or declared.isdigit() and 0<int(declared)<=limits['max_wire_bytes'], 'NFL_ACQUIRE_BODY_LIMIT')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS) if encoding=='gzip' else None
    body=bytearray(); wire=0; wire_hash=hashlib.sha256()
    for chunk in stream.chunks():
        require(monotonic()<deadline, 'NFL_ACQUIRE_DEADLINE')
        require(isinstance(chunk,bytes) and 0<len(chunk)<=limits['max_wire_bytes'], 'NFL_ACQUIRE_INCOMPLETE_BODY')
        wire+=len(chunk); wire_hash.update(chunk); require(wire<=limits['max_wire_bytes'], 'NFL_ACQUIRE_BODY_LIMIT')
        body.extend(decoder.decompress(chunk,limits['max_body_bytes']-len(body)+1) if decoder else chunk)
        require(len(body)<=limits['max_body_bytes'] and (decoder is None or not decoder.unconsumed_tail), 'NFL_ACQUIRE_BODY_LIMIT')
    require(monotonic()<deadline, 'NFL_ACQUIRE_DEADLINE')
    require(declared is None or wire==int(declared), 'NFL_ACQUIRE_INCOMPLETE_BODY')
    require(decoder is None or decoder.eof and not decoder.unused_data, 'NFL_ACQUIRE_INCOMPLETE_BODY')
    return bytes(body),dict(wire_bytes=wire,wire_sha256=wire_hash.hexdigest(),encoding=encoding)


def acquire(plan,authorization,*,root,transport,wall=previous.utc,monotonic=time.monotonic):
    p=authorize(plan,authorization,wall(),root); root=Path(root)
    try: journal=(root/(plan['sha256']+'.nfl-scoring-attempts.jsonl')).open('xb')
    except FileExistsError: raise ValueError('NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME') from None
    deadline=monotonic()+p['limits']['total_seconds']; obj=None; reason=None; attempted=0; chain=[]
    try:
        with journal:
            _append(journal,dict(plan_sha256=plan['sha256'],authorization_sha256=authorization['sha256'],custody_id=p['custody_id']))
            request=safe=deepcopy(PRIMARY)
            while True:
                authorize(plan,authorization,wall(),root)
                require(monotonic()<deadline and attempted<p['limits']['max_attempts'], 'NFL_ACQUIRE_BUDGET_EXHAUSTED')
                started=wall(); attempted+=1
                _append(journal,dict(attempt=attempted,request=safe,requested_at=started))
                stream=None; location=None
                try:
                    stream=transport(deepcopy(request),deepcopy(p['limits']),deadline,monotonic)
                    h=headers(stream)
                    if stream.status==302 and attempted==1:
                        location=h.get('location'); next_request,next_safe=redirect_request(location)
                        body=None; transfer=None
                    else: body,transfer=read(stream,p['limits'],deadline,monotonic)
                    received=wall()
                finally:
                    if stream is not None: stream.close()
                require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
                authorize(plan,authorization,received,root)
                step=dict(attempt=attempted,request=safe,started_at=started,received_at=received,status=stream.status,
                    location_sha256=hashlib.sha256(location.encode()).hexdigest() if location else None)
                chain.append(step); _append(journal,step)
                if location:
                    request,safe=next_request,next_safe
                    continue
                transfer['requests']=chain
                obj=custody.retained_score(body,receipt=dict(source_id=profile.SOURCE_ID,endpoint=profile.ENDPOINT,format='csv',evidence_label=p['evidence_label'],
                    request_started_at=chain[0]['started_at'],received_at=received,observed_at=received,
                    observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='publisher_availability_unknown'),transfer=transfer)
                _append(journal,dict(received_at=received,body_sha256=obj['body_sha256']))
                break
    except Exception as exc:
        text=str(exc)
        reason=text if text.startswith(('NFL_ACQUIRE_','NFL_PRIVATE_')) and re.fullmatch('[A-Z_]+',text) else 'NFL_ACQUIRE_STOPPED'
    bundle=seal(dict(version='nfl-unaccepted-scoring-body-v2',profile=profile.PROFILE,plan=plan,authorization_sha256=authorization['sha256'],
        attempted=attempted,status='INCOMPLETE' if reason else 'CAPTURED_UNREVIEWED',reason=reason,object=obj,
        owner_reviewed=False,inference=False,source_acceptance=False,wagering_authority=False))
    raw=custody.encode(bundle).encode()
    require(len(raw)<=profile.MAX_PACKET,'NFL_ACQUIRE_SERIALIZED_LIMIT')
    path=root/(plan['sha256']+'.nfl-scoring-unreviewed.json')
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    require(path.read_bytes()==raw,'NFL_ACQUIRE_PRIVATE_READBACK')
    return bundle
