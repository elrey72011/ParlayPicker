"""One NFL/Novig spreads request. Original bytes only; never inference or admission."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlencode, unquote
import zlib
from app_core import nfl_observed_custody as custody
from app_core import nfl_acquisition_diagnostics as diagnostics
from app_core.ncaaf_pilot import _timed, _append, utc
from app_core.nfl_source_projection import seal
from app_core.nfl_scoring_acquisition import checked_headers
from app_core.producer_provenance import clock

VERSION = 'nfl-odds-acquisition-plan-v1'
AUTH_VERSION = 'nfl-odds-acquisition-authorization-v1'
CONFIG_VERSION = 'nfl-odds-controlled-bindings-v1'
LISTING_VERSION = 'nfl-odds-acquisition-plan-v2'
LISTING_AUTH_VERSION = 'nfl-odds-acquisition-authorization-v2'
LISTING_CONFIG_VERSION = 'nfl-odds-controlled-bindings-v2'
LISTING_FLAGS = frozenset({'includeSids', 'includeLinks'})
SOURCE = dict(provider='the-odds-api', host='api.the-odds-api.com',
              endpoint='/v4/sports/americanfootball_nfl/odds', method='GET')
PARAMS = dict(regions='us_ex', bookmakers='novig', markets='spreads', oddsFormat='american', dateFormat='iso')
LIMITS = dict(max_attempts=1, max_objects=1, max_wire_bytes=512*1024, max_body_bytes=512*1024,
              max_packet_bytes=2*1024*1024, max_header_bytes=16*1024,
              connect_seconds=10, read_seconds=10, total_seconds=60)
AUTHORIZED_COLLECTIONS = {}
REVIEWED_PERMISSIONS = {}
AUTHORIZED_CUSTODY_ROOTS = {}
CONTROLLED = ContextVar('nfl_odds_controlled_bindings', default=None)
require, digest = custody.require, custody.digest
ALLOWED_CODES = frozenset('''
NFL_ACQUIRE_ACCOUNT_EXPIRED NFL_ACQUIRE_ACCOUNT_UNVERIFIED
NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME NFL_ACQUIRE_AUTHORIZATION
NFL_ACQUIRE_AUTHORIZATION_EXPIRED NFL_ACQUIRE_AUTHORIZATION_UNTRUSTED
NFL_ACQUIRE_BODY_LIMIT NFL_ACQUIRE_CONFIGURATION_CONFLICT
NFL_ACQUIRE_CONFIGURATION_INTEGRITY NFL_ACQUIRE_CONFIGURATION_REVIEW_MISSING
NFL_ACQUIRE_CONFIGURATION_SIZE NFL_ACQUIRE_CONTENT_ENCODING NFL_ACQUIRE_CONTENT_TYPE
NFL_ACQUIRE_CREDENTIAL_FORBIDDEN NFL_ACQUIRE_CREDENTIAL_IN_BODY NFL_ACQUIRE_CREDENTIAL_UNAVAILABLE
NFL_ACQUIRE_CUSTODY_UNVERIFIED NFL_ACQUIRE_DEADLINE NFL_ACQUIRE_ENDPOINT_NOT_APPROVED
NFL_ACQUIRE_HEADER_CONFLICT NFL_ACQUIRE_HEADER_LIMIT NFL_ACQUIRE_HTTP_STATUS
NFL_ACQUIRE_IDENTITY_CONFLICT NFL_ACQUIRE_INCOMPLETE_BODY NFL_ACQUIRE_KICKOFF_CONFLICT
NFL_ACQUIRE_LIMITS NFL_ACQUIRE_NOVIG_SPREADS_ABSENT NFL_ACQUIRE_ORIENTATION_CONFLICT
NFL_ACQUIRE_PERMISSION_EXPIRED NFL_ACQUIRE_PERMISSION_UNTRUSTED NFL_ACQUIRE_PLAN_INTEGRITY
NFL_ACQUIRE_PLAN_SCHEMA NFL_ACQUIRE_PREREQUISITES NFL_ACQUIRE_PRIVATE_READBACK
NFL_ACQUIRE_PRIVATE_TERMINAL_REQUIRED NFL_ACQUIRE_REDIRECT_OR_PAGINATION
NFL_ACQUIRE_REQUEST_WINDOW NFL_ACQUIRE_RESPONSE_SCHEMA NFL_ACQUIRE_RESPONSE_SCOPE
NFL_ACQUIRE_RUNTIME_UNSUPPORTED NFL_ACQUIRE_SERIALIZED_LIMIT NFL_ACQUIRE_TARGET
NFL_ACQUIRE_TARGET_AMBIGUOUS NFL_ACQUIRE_TARGET_MISSING NFL_ACQUIRE_TARGET_UNVERIFIED
NFL_ACQUIRE_WINDOW NFL_ACQUIRE_WINDOW_EXPIRED NFL_ACQUIRE_WRONG_SPORT
NFL_PRIVATE_RESPONSE_SCHEMA NFL_PRIVATE_RESPONSE_SIZE NFL_PRIVATE_CREDENTIAL_FORBIDDEN
NFL_PRIVATE_CUSTODY_MISSING NFL_PRIVATE_CUSTODY_CORRUPT NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN
NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT
'''.split())


def rejection_code(exc):
    # A prefix is insufficient: it can conceal credential-bearing exception text.
    text=str(exc) if isinstance(exc,ValueError) else None
    return text if text in ALLOWED_CODES else 'NFL_ACQUIRE_STOPPED'


def hashed(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def utc_instant(value):
    return isinstance(value, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', value) and clock(value)


def check_request(r, *, version=VERSION):
    params=r.get('params',{})
    require(version in {VERSION, LISTING_VERSION}, 'NFL_ACQUIRE_PLAN_SCHEMA')
    flags=set(params)&LISTING_FLAGS if version==LISTING_VERSION else set()
    require(version!=LISTING_VERSION or flags, 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
    require(set(r)==set(SOURCE)|{'params'} and all(r[k]==v for k,v in SOURCE.items())
            and set(params)==set(PARAMS)|{'commenceTimeFrom','commenceTimeTo'}|flags
            and all(params[k]==v for k,v in PARAMS.items())
            and all(type(params[k]) is str and params[k] in {'true','false'} for k in flags), 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
    require(all(utc_instant(params[k]) for k in ('commenceTimeFrom','commenceTimeTo'))
            and clock(params['commenceTimeFrom'])<=clock(params['commenceTimeTo']), 'NFL_ACQUIRE_REQUEST_WINDOW')


def validate(plan, *, executable=False):
    require(isinstance(plan, dict) and set(plan)=={'payload','sha256'} and digest(plan['payload'])==plan['sha256'], 'NFL_ACQUIRE_PLAN_INTEGRITY')
    p=plan['payload']
    from app_core.ncaaf_response_custody import _credentials
    try: _credentials(p)
    except ValueError: raise ValueError('NFL_ACQUIRE_CREDENTIAL_FORBIDDEN') from None
    require(set(p)==set('version evidence_label target request limits window permission authorization_ref custody_id unresolved'.split())
            and p['version'] in {VERSION,LISTING_VERSION} and p['evidence_label'] in {'SYNTHETIC','PROPOSAL','RETAINED'}, 'NFL_ACQUIRE_PLAN_SCHEMA')
    r=p['request']; check_request(r,version=p['version']); params=r['params']
    t=p['target']
    from core.nfl_teams import NFL_TEAMS
    require(set(t)==set('canonical_event_id home away kickoff season neutral_site schedule_evidence_sha256'.split())
            and custody.known(t['canonical_event_id']) and t['home'] in NFL_TEAMS.values() and t['away'] in NFL_TEAMS.values()
            and t['home']!=t['away'] and type(t['season']) is int and 2000<=t['season']<=2100
            and type(t['neutral_site']) is bool and utc_instant(t['kickoff']) and hashed(t['schedule_evidence_sha256'])
            and clock(params['commenceTimeFrom'])<=clock(t['kickoff'])<=clock(params['commenceTimeTo']), 'NFL_ACQUIRE_TARGET')
    require(p['limits']==LIMITS and all(type(p['limits'][k]) is int for k in LIMITS), 'NFL_ACQUIRE_LIMITS')
    w=p['window']
    require(set(w)=={'not_before','not_after','expires_at'} and all(utc_instant(v) for v in w.values())
            and clock(w['not_before'])<clock(w['not_after'])<=clock(w['expires_at'])<clock(t['kickoff']), 'NFL_ACQUIRE_WINDOW')
    require(isinstance(p['unresolved'],list) and all(isinstance(x,str) for x in p['unresolved'])
            and custody.known(p['authorization_ref']) and custody.known(p['custody_id']), 'NFL_ACQUIRE_PLAN_SCHEMA')
    if executable: require(p['evidence_label']!='PROPOSAL' and not p['unresolved'], 'NFL_ACQUIRE_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p=validate(plan)
    return dict(plan_sha256=plan['sha256'], status='NON_EXECUTABLE_PROPOSAL' if p['evidence_label']=='PROPOSAL' else 'AWAITING_EXACT_AUTHORIZATION',
                requests_max=1, odds_credits_max=1, cfbd_requests=0, maximum_additional_spend=0,
                network_requests=0, credentials=0, storage_operations=0, inference=False, unresolved=p['unresolved'])


def authorize(plan, authorization, at, root):
    p=validate(plan,executable=True); now=clock(at); w=p['window']
    require(now and clock(w['not_before'])<=now<=clock(w['not_after']) and now<clock(w['expires_at']) and now<clock(p['target']['kickoff']), 'NFL_ACQUIRE_WINDOW_EXPIRED')
    require(isinstance(authorization,dict) and set(authorization)=={'payload','sha256'} and digest(authorization['payload'])==authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION')
    c=CONTROLLED.get()
    collections=c['collections'] if c else AUTHORIZED_COLLECTIONS
    permissions=c['permissions'] if c else REVIEWED_PERMISSIONS
    roots=c['roots'] if c else AUTHORIZED_CUSTODY_ROOTS
    a=authorization['payload']
    require(set(a)==set('version reference plan_sha256 custody_id owner authorized_at expires_at maximum_additional_spend maximum_odds_credits'.split())
            and a['version']==(LISTING_AUTH_VERSION if p['version']==LISTING_VERSION else AUTH_VERSION)
            and a['reference']==p['authorization_ref'] and a['plan_sha256']==plan['sha256']
            and a['custody_id']==p['custody_id'] and a['owner']=='Robert Velarde' and a['maximum_additional_spend']==0 and a['maximum_odds_credits']==1
            and collections.get(a['reference'])==authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION_UNTRUSTED')
    require(clock(a['authorized_at']) and clock(a['expires_at']) and clock(a['authorized_at'])<=now<clock(a['expires_at'])<=clock(w['expires_at']), 'NFL_ACQUIRE_AUTHORIZATION_EXPIRED')
    r=p['permission']
    require(isinstance(r,dict) and set(r)==set('reference reviewer reviewed_at effective_from effective_until request_sha256 private_acquisition private_retention basis'.split())
            and permissions.get(r['reference'])==digest(r) and r['reviewer']=='Robert Velarde' and r['request_sha256']==digest(p['request'])
            and r['private_acquisition'] is True and r['private_retention'] is True and isinstance(r['basis'],str) and len(r['basis'])>=16, 'NFL_ACQUIRE_PERMISSION_UNTRUSTED')
    require(all(clock(r[k]) for k in ('reviewed_at','effective_from','effective_until')) and clock(r['reviewed_at'])<now
            and clock(r['effective_from'])<=now<clock(w['expires_at'])<=clock(r['effective_until']), 'NFL_ACQUIRE_PERMISSION_EXPIRED')
    require(p['custody_id'] in roots and Path(roots[p['custody_id']]).resolve()==Path(root).resolve() and Path(root).is_dir(), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    if c:
        account=c['account']
        require(clock(account['verified_at'])<=now<clock(w['expires_at'])<=clock(account['valid_until'])
                and clock(w['expires_at'])<clock(account['resets_at']), 'NFL_ACQUIRE_ACCOUNT_EXPIRED')
    else: require(False,'NFL_ACQUIRE_ACCOUNT_UNVERIFIED')
    return p


@contextmanager
def controlled(plan, raw, expected_hash, *, root, wall=utc, security=None):
    plan_payload=validate(plan)
    require(isinstance(raw,bytes) and 0<len(raw)<=128*1024 and hashlib.sha256(raw).hexdigest()==expected_hash, 'NFL_ACQUIRE_CONFIGURATION_INTEGRITY')
    c=json.loads(raw,object_pairs_hook=custody.unique)
    from app_core.ncaaf_response_custody import _credentials
    # The structural authorization receipt is not an HTTP Authorization header.
    try:
        _credentials({k:v for k,v in c.get('payload',{}).items() if k!='authorization'})
        _credentials(c.get('payload',{}).get('authorization',{}))
    except ValueError: raise ValueError('NFL_ACQUIRE_CREDENTIAL_FORBIDDEN') from None
    require(set(c)=={'payload','sha256'} and digest(c['payload'])==c['sha256'], 'NFL_ACQUIRE_CONFIGURATION_INTEGRITY')
    p=c['payload']
    require(set(p)==set('version plan_sha256 permission authorization custody account target_review review'.split())
            and p['version']==(LISTING_CONFIG_VERSION if plan_payload['version']==LISTING_VERSION else CONFIG_VERSION)
            and p['plan_sha256']==plan['sha256'] and p['permission']==plan['payload']['permission'], 'NFL_ACQUIRE_CONFIGURATION_CONFLICT')
    r=p['review']; now=clock(wall())
    require(set(r)=={'reviewer','reviewed_at','subject_sha256','status','scope'} and r['reviewer']=='Robert Velarde'
            and r['status']=='OWNER_REVIEWED' and r['scope']=='NFL odds acquisition only; no offer admission.'
            and r['subject_sha256']==digest({k:v for k,v in p.items() if k!='review'}) and clock(r['reviewed_at']) and now and clock(r['reviewed_at'])<now, 'NFL_ACQUIRE_CONFIGURATION_REVIEW_MISSING')
    t=p['target_review']
    require(set(t)=={'reviewer','reviewed_at','target_sha256','conclusion','evidence_sha256'} and t['reviewer']=='Robert Velarde'
            and clock(t['reviewed_at']) and clock(t['reviewed_at'])<now and t['target_sha256']==digest(plan['payload']['target'])
            and t['evidence_sha256']==plan['payload']['target']['schedule_evidence_sha256']
            and t['conclusion']=='VERIFIED_NAMED_ORIENTATION_AND_KICKOFF', 'NFL_ACQUIRE_TARGET_UNVERIFIED')
    a=p['account']
    require(set(a)==set('license_holder reviewer verified_at valid_until resets_at remaining_included_credits entitlement_sha256 no_paid_overage'.split())
            and a['license_holder']==a['reviewer']=='Robert Velarde' and all(clock(a[k]) for k in ('verified_at','valid_until','resets_at'))
            and type(a['remaining_included_credits']) is int and a['remaining_included_credits']>=1
            and hashed(a['entitlement_sha256']) and a['no_paid_overage'] is True, 'NFL_ACQUIRE_ACCOUNT_UNVERIFIED')
    d=p['custody']
    require(set(d)=={'id','root','storage_receipt_sha256'} and d['id']==plan['payload']['custody_id']
            and Path(d['root']).resolve()==Path(root).resolve() and hashed(d['storage_receipt_sha256']), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    if security is None:
        from app_core.ncaaf_pilot_setup import inspect_security
        security=inspect_security
    found=security(root)
    require(found.get('status')=='VERIFIED' and all(found.get(k) is True for k in ('private_acl','encrypted','no_reparse')), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    approval=p['authorization']['payload']; permission=p['permission']
    token=CONTROLLED.set(dict(collections={approval['reference']:p['authorization']['sha256']}, permissions={permission['reference']:digest(permission)},
                             roots={d['id']:d['root']},account=a))
    try:
        authorize(plan,p['authorization'],wall(),root)
        yield p['authorization']
    finally: CONTROLLED.reset(token)


class HttpsTransport:
    """Real HTTPSConnection/_timed; no library downloads, redirects or retries."""
    def __init__(self, credentials, *, version=VERSION):
        self.credentials=credentials; self._secret=None; self.version=version

    def __call__(self, request, limits, deadline, monotonic):
        check_request(request,version=self.version)
        require(limits==LIMITS, 'NFL_ACQUIRE_LIMITS')
        started=monotonic(); phase='CONNECTION_SETUP'; status=diagnostics.UNKNOWN
        conn=None; response=None; read_socket=None; cancelled=threading.Event()
        def close():
            cancelled.set()
            shutdown=getattr(read_socket,'shutdown',None)
            if shutdown is not None:
                try:shutdown(socket.SHUT_RDWR)
                except OSError:pass  # Cleanup only; body/socket errors still reject.
            try:
                if response is not None and hasattr(response,'close'):response.close()
            finally:
                if conn is not None:conn.close()
        try:
            key=_timed(self.credentials,deadline,monotonic,cancelled,lambda:None)
            require(isinstance(key,str) and 0<len(key)<=4096 and not any(x in key for x in ('\r','\n')), 'NFL_ACQUIRE_CREDENTIAL_UNAVAILABLE')
            self._secret=key
            conn=http.client.HTTPSConnection(SOURCE['host'],timeout=min(10,max(.001,deadline-monotonic())))
            conn.auto_open=False; phase='HTTPS_CONNECT'
            _timed(conn.connect,min(deadline,monotonic()+10),monotonic,cancelled,close)
            phase='REQUEST_SEND'; conn.sock.settimeout(min(10,max(.001,deadline-monotonic())))
            params=dict(request['params'],apiKey=key)
            _timed(lambda:conn.request('GET',SOURCE['endpoint']+'?'+urlencode(params),headers={'Accept':'application/json','Accept-Encoding':'identity'}), min(deadline,monotonic()+10),monotonic,cancelled,close)
            read_socket=conn.sock
            phase='RESPONSE_RECEIVE'
            response=_timed(conn.getresponse,min(deadline,monotonic()+10),monotonic,cancelled,close)
            status=diagnostics.status(response.status); phase='RESPONSE_HEADERS'
            pairs=response.getheaders()
            require(len(pairs)<=64 and sum(len(k)+len(v)+4 for k,v in pairs)<=LIMITS['max_header_bytes'], 'NFL_ACQUIRE_HEADER_LIMIT')
            headers=checked_headers(pairs)
        except Exception as exc:
            close()
            raise diagnostics.Failure(rejection_code(exc),diagnostics.facts(exc,phase=phase,http_status=status,started=started,ended=monotonic())) from None
        class Stream:
            status=response.status
            def chunks(self):
                try:
                    while True:
                        require(not cancelled.is_set() and monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
                        if getattr(response,'fp',response) is None:break
                        read_socket.settimeout(min(10,max(.001,deadline-monotonic())))
                        block=_timed(lambda:response.read1(8192),min(deadline,monotonic()+10),monotonic,cancelled,close)
                        if not block: break
                        yield block
                except Exception as exc:
                    raise diagnostics.Failure(rejection_code(exc),diagnostics.facts(exc,phase='BODY_READ',http_status=status,started=started,ended=monotonic())) from None
            def close(self): close()
        out=Stream(); out.headers=headers
        return out

    def validate_body(self, body):
        # Escaped JSON or URL-encoded echo is forbidden, not redacted/re-serialized.
        require(self._secret is not None and self._secret.encode() not in body, 'NFL_ACQUIRE_CREDENTIAL_IN_BODY')
        values=[json.loads(body,object_pairs_hook=custody.unique)]
        while values:
            value=values.pop()
            if isinstance(value,str): require(self._secret not in unquote(value),'NFL_ACQUIRE_CREDENTIAL_IN_BODY')
            elif isinstance(value,dict): values.extend(value.keys()); values.extend(value.values())
            elif isinstance(value,list): values.extend(value)


def read_body(stream, deadline, monotonic):
    require(len(stream.headers)<=64 and sum(len(k)+len(v)+4 for k,v in stream.headers.items())<=LIMITS['max_header_bytes'], 'NFL_ACQUIRE_HEADER_LIMIT')
    h=checked_headers(list(stream.headers.items()))
    require(stream.status==200 and not h.get('location'), 'NFL_ACQUIRE_HTTP_STATUS')
    require(h.get('content-type','').split(';')[0].strip().lower()=='application/json', 'NFL_ACQUIRE_CONTENT_TYPE')
    encoding=h.get('content-encoding','identity').lower()
    require(encoding in {'identity','gzip'}, 'NFL_ACQUIRE_CONTENT_ENCODING')
    length=h.get('content-length')
    require(length is None or length.isdigit() and 0<int(length)<=512*1024, 'NFL_ACQUIRE_BODY_LIMIT')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS) if encoding=='gzip' else None
    body=bytearray(); wire=0; wire_hash=hashlib.sha256()
    for chunk in stream.chunks():
        require(monotonic()<deadline, 'NFL_ACQUIRE_DEADLINE')
        require(isinstance(chunk,bytes) and 0<len(chunk)<=512*1024, 'NFL_ACQUIRE_INCOMPLETE_BODY')
        wire+=len(chunk); wire_hash.update(chunk); require(wire<=512*1024, 'NFL_ACQUIRE_BODY_LIMIT')
        body.extend(decoder.decompress(chunk,512*1024-len(body)+1) if decoder else chunk)
        require(len(body)<=512*1024 and (decoder is None or not decoder.unconsumed_tail), 'NFL_ACQUIRE_BODY_LIMIT')
    require(monotonic()<deadline and (length is None or wire==int(length))
            and (decoder is None or decoder.eof and not decoder.unused_data), 'NFL_ACQUIRE_INCOMPLETE_BODY')
    return bytes(body),dict(wire_bytes=wire,wire_sha256=wire_hash.hexdigest(),encoding=encoding)


def target_findings(events, target):
    """Exact names only, no alias expansion. Findings are explicitly unreviewed."""
    require(isinstance(events,list), 'NFL_ACQUIRE_RESPONSE_SCHEMA')
    matches=[]; ids=set(); same_teams=[]
    for event in events:
        require(isinstance(event,dict) and custody.known(event.get('id')) and event['id'] not in ids, 'NFL_ACQUIRE_IDENTITY_CONFLICT')
        ids.add(event['id'])
        require(event.get('sport_key')=='americanfootball_nfl', 'NFL_ACQUIRE_WRONG_SPORT')
        require(custody.known(event.get('home_team')) and custody.known(event.get('away_team')) and clock(event.get('commence_time')), 'NFL_ACQUIRE_RESPONSE_SCHEMA')
        if {event['home_team'],event['away_team']}=={target['home'],target['away']}:
            same_teams.append(event)
            if event['home_team']==target['home'] and event['away_team']==target['away'] and clock(event['commence_time'])==clock(target['kickoff']): matches.append(event)
    require(len(matches)<=1 and len(same_teams)<=1, 'NFL_ACQUIRE_TARGET_AMBIGUOUS')
    if not matches:
        reason='NFL_ACQUIRE_TARGET_MISSING' if not same_teams else ('NFL_ACQUIRE_ORIENTATION_CONFLICT' if same_teams[0]['home_team']!=target['home'] else 'NFL_ACQUIRE_KICKOFF_CONFLICT')
        return dict(status='UNRESOLVED',reason=reason,provider_event_id=None,offers=[])
    e=matches[0]; books=e.get('bookmakers')
    require(isinstance(books,list) and all(isinstance(b,dict) and b.get('key')=='novig' for b in books) and len(books)<=1, 'NFL_ACQUIRE_RESPONSE_SCOPE')
    markets=books[0].get('markets') if books else []
    require(isinstance(markets,list) and len(markets)<=1 and all(isinstance(m,dict) and m.get('key')=='spreads' for m in markets), 'NFL_ACQUIRE_RESPONSE_SCOPE')
    offers=markets[0].get('outcomes') if markets else []
    require(isinstance(offers,list), 'NFL_ACQUIRE_RESPONSE_SCHEMA')
    for o in offers:
        require(isinstance(o,dict) and o.get('name') in {target['home'],target['away']} and all(type(o.get(k)) in {int,float} and math.isfinite(o[k]) for k in ('point','price')), 'NFL_ACQUIRE_RESPONSE_SCHEMA')
    require(len({o['name'] for o in offers})==len(offers) and len(offers) in {0,2}, 'NFL_ACQUIRE_RESPONSE_SCHEMA')
    return dict(status='OBSERVED_UNREVIEWED' if offers else 'UNRESOLVED', reason=None if offers else 'NFL_ACQUIRE_NOVIG_SPREADS_ABSENT',
                provider_event_id=e['id'],offers=deepcopy(offers))


def listing_references(obj, findings):
    """Pointers into original bytes only. No URL access or semantic acceptance."""
    _, events=custody.read(obj)
    result=[]
    def record(node, pointer, core):
        names=sorted(set(node)-core)
        def escape(value): return value.replace('~','~0').replace('/','~1')
        result.append(dict(body_sha256=obj['body_sha256'],json_pointer=pointer,
            optional_fields_state='PRESENT_UNREVIEWED' if names else 'UNKNOWN',
            fields=[dict(name=k,json_pointer=pointer+'/'+escape(k)) for k in names],
            trusted=False))
    for index, event in enumerate(events):
        if event['id']!=findings['provider_event_id']: continue
        pointer='/'+str(index)
        record(event,pointer,set('id sport_key sport_title commence_time home_team away_team bookmakers'.split()))
        for bi, book in enumerate(event.get('bookmakers',[])):
            bp=pointer+'/bookmakers/'+str(bi)
            record(book,bp,set('key title last_update markets'.split()))
            for mi, market in enumerate(book.get('markets',[])):
                mp=bp+'/markets/'+str(mi)
                record(market,mp,set('key last_update outcomes'.split()))
                for oi, outcome in enumerate(market.get('outcomes',[])):
                    record(outcome,mp+'/outcomes/'+str(oi),set('name description point price'.split()))
    return result


def acquire(plan, authorization, *, root, transport, wall=utc, monotonic=time.monotonic):
    p=authorize(plan,authorization,wall(),root); root=Path(root)
    journal_path=root/(plan['sha256']+'.nfl-odds-attempts.jsonl')
    try: journal=journal_path.open('xb')
    except FileExistsError: raise ValueError('NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME') from None
    start=monotonic(); deadline=start+60; obj=None; findings=None; transfer=None; failure=None; reason=None; attempted=0
    phase='JOURNAL'; status=diagnostics.UNKNOWN
    try:
        with journal:
            _append(journal,dict(plan_sha256=plan['sha256'],authorization_sha256=authorization['sha256'],custody_id=p['custody_id']))
            phase='AUTHORIZATION'; authorize(plan,authorization,wall(),root)
            require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
            requested=wall(); attempted=1; phase='JOURNAL'
            _append(journal,dict(attempt=1,request=p['request'],requested_at=requested))
            phase='AUTHORIZATION'; authorize(plan,authorization,wall(),root)
            require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
            stream=None
            try:
                phase='TRANSPORT'; stream=transport(deepcopy(p['request']),deepcopy(LIMITS),deadline,monotonic)
                status=diagnostics.status(stream.status); phase='BODY_VALIDATION'
                body,transfer=read_body(stream,deadline,monotonic)
                transport.validate_body(body)
                require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
                received=wall()
            finally:
                if stream: stream.close()
            phase='AUTHORIZATION'; authorize(plan,authorization,received,root)
            phase='CUSTODY'
            obj=custody.retained(body,receipt=dict(source_id='the-odds-api:v4:americanfootball_nfl',endpoint='https://'+SOURCE['host']+SOURCE['endpoint'],
                format='json',evidence_label=p['evidence_label'],request_started_at=requested,received_at=received,observed_at=received,
                observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='provider_market_last_update'))
            _append(journal,dict(received_at=received,http_status=status,body_sha256=obj['body_sha256'],body_bytes=len(body)))
            findings=target_findings(custody.read(obj)[1],p['target']); reason=findings['reason']
            if p['version']==LISTING_VERSION:
                findings['listing_references']=listing_references(obj,findings)
            require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
    except (Exception, KeyboardInterrupt) as exc:
        reason=rejection_code(exc)
        failure=diagnostics.checked(exc.diagnostic) if isinstance(exc,diagnostics.Failure) else diagnostics.facts(exc,phase=phase,http_status=status,started=start,ended=monotonic())
    # Original valid bytes remain in an incomplete bundle when matching fails.
    payload=dict(version='nfl-unaccepted-odds-body-v2' if p['version']==LISTING_VERSION else 'nfl-unaccepted-odds-body-v1',plan=plan,authorization_sha256=authorization['sha256'],attempted=attempted,
                 status='INCOMPLETE' if reason else 'CAPTURED_UNREVIEWED',reason=reason,object=obj,transfer=transfer,findings=findings,
                 journal_sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),owner_reviewed=False,source_acceptance=False,
                 inference=False,public_export=False,qualified=False,action='PASS',stake=0)
    if failure is not None: payload['failure_diagnostics']=failure
    bundle=seal(payload); raw=custody.encode(bundle).encode()
    require(len(raw)<=LIMITS['max_packet_bytes'],'NFL_ACQUIRE_SERIALIZED_LIMIT')
    path=root/(plan['sha256']+'.nfl-odds-unreviewed.json')
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    require(read_back(path,journal_path)==bundle, 'NFL_ACQUIRE_PRIVATE_READBACK')
    return bundle


def read_back(path, journal_path):
    raw=Path(path).read_bytes(); require(len(raw)<=LIMITS['max_packet_bytes'],'NFL_ACQUIRE_SERIALIZED_LIMIT')
    bundle=json.loads(raw,object_pairs_hook=custody.unique)
    require(set(bundle)=={'payload','sha256'} and digest(bundle['payload'])==bundle['sha256'], 'NFL_ACQUIRE_PRIVATE_READBACK')
    p=bundle['payload']; validate(p['plan'])
    with Path(journal_path).open('rb') as f: journal=f.read(128*1024+1)
    require(len(journal)<=128*1024,'NFL_ACQUIRE_PRIVATE_READBACK')
    rows=[json.loads(line,object_pairs_hook=custody.unique) for line in journal.splitlines()]
    require(rows and rows[0]==dict(plan_sha256=p['plan']['sha256'],authorization_sha256=p['authorization_sha256'],custody_id=p['plan']['payload']['custody_id']), 'NFL_ACQUIRE_PRIVATE_READBACK')
    spent=[r for r in rows if 'attempt' in r]
    require(len(spent)==p['attempted'] and (not spent or spent[0]['attempt']==1 and spent[0]['request']==p['plan']['payload']['request']), 'NFL_ACQUIRE_PRIVATE_READBACK')
    expected_version='nfl-unaccepted-odds-body-v2' if p['plan']['payload']['version']==LISTING_VERSION else 'nfl-unaccepted-odds-body-v1'
    require(p['version']==expected_version and p['attempted'] in {0,1}
            and p['journal_sha256']==hashlib.sha256(Path(journal_path).read_bytes()).hexdigest()
            and all(p[k] is False for k in ('owner_reviewed','source_acceptance','inference','public_export','qualified'))
            and p['action']=='PASS' and p['stake']==0, 'NFL_ACQUIRE_PRIVATE_READBACK')
    if p['object']: custody.read(p['object'])
    if expected_version=='nfl-unaccepted-odds-body-v2' and p['findings'] is not None:
        expected=target_findings(custody.read(p['object'])[1],p['plan']['payload']['target'])
        expected['listing_references']=listing_references(p['object'],expected)
        require(p['findings']==expected,'NFL_ACQUIRE_PRIVATE_READBACK')
    if 'failure_diagnostics' in p: diagnostics.checked(p['failure_diagnostics'])
    return bundle


def execution_report(bundle):
    p=bundle['payload']
    report={k:p[k] for k in ('status','reason','attempted','journal_sha256')}
    report['bundle_sha256']=bundle['sha256']
    report['body_sha256']=p['object']['body_sha256'] if p['object'] else None
    report['body_bytes']=len(custody.read(p['object'])[0]) if p['object'] else None
    report['receipt']=p['object']['receipt'] if p['object'] else None
    if 'failure_diagnostics' in p: report['failure_diagnostics']=diagnostics.checked(p['failure_diagnostics'])
    return report
