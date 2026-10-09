"""Transport-free, private decoded-body custody for explicitly selected NCAAF.

No request, store initializer, registration or historical inference. Native
packets and schemas remain unchanged; this is a separate successor envelope.
"""
import base64
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from app_core import ncaaf_compatible_pipeline as native
from app_core import ncaaf_prospective_chronology as chronology
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history

VERSION = 'ncaaf-original-response-inputs-v1'
RESULT_VERSION = 'ncaaf-original-response-result-v1'
OBJECT_VERSION = 'ncaaf-decoded-response-body-v1'
PROJECTION_VERSION = 'ncaaf-response-projection-v1'
SUBJECT_VERSION = 'ncaaf-response-custody-subject-v1'
ADMISSION_VERSION = 'ncaaf-response-custody-admission-v1'
REPRESENTATION = 'http-client-body-after-content-decoding-before-json-v1'
RECEIPT_MEANING = 'actual_local_decoded_body_receipt'
ODDS_ENDPOINT = 'v4/sports/americanfootball_ncaaf/odds'
MAX_BODY_BYTES = 512 * 1024
MAX_TOTAL_BODY_BYTES = 2 * 1024 * 1024
MAX_OBJECTS = 16
# Independent reviewer catalog. Parsing/upload/selection never populates it.
ACCEPTED_ADMISSIONS = {}
REASONS = frozenset('''NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE NCAAF_CUSTODY_SCHEMA
NCAAF_CUSTODY_INTEGRITY NCAAF_CUSTODY_LIMIT NCAAF_CUSTODY_REPRESENTATION
NCAAF_CUSTODY_HTTP_STATUS NCAAF_CUSTODY_BODY_SCHEMA NCAAF_CUSTODY_SCOPE_CONFLICT
NCAAF_CUSTODY_PROJECTION_CONFLICT NCAAF_CUSTODY_QUOTE_CONFLICT
NCAAF_CUSTODY_CLOCK_CONFLICT NCAAF_CUSTODY_ADMISSION_CONFLICT
NCAAF_CUSTODY_ADMISSION_NOT_TRUSTED NCAAF_CUSTODY_SUBJECT_FUTURE_FACT
NCAAF_CUSTODY_PERMISSION_CLOCK_CONFLICT NCAAF_CUSTODY_RUNTIME_CHANGED
NCAAF_CUSTODY_COMPUTATION_CONFLICT NCAAF_CUSTODY_CREDENTIAL_FIELD
NCAAF_CUSTODY_INPUT_STALE'''.split()) | native.REASONS
require = model.require

def implementation():
    """Separate installed reader/projector identity; frozen collectors unchanged."""
    root = Path(__file__).resolve().parents[1]
    paths = ('app_core/ncaaf_response_custody.py', 'app_core/ncaaf_history.py',
             'app_core/producer_provenance.py', 'app_core/source_contract.py',
             'app_core/prediction_evidence.py')
    return {p: hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}

def _unique(pairs):
    result = {}
    for key,value in pairs:
        require(key not in result, 'NCAAF_CUSTODY_BODY_SCHEMA')
        result[key] = value
    return result

def _constant(_):
    raise ValueError('NCAAF_CUSTODY_BODY_SCHEMA')

def _credentials(value):
    if isinstance(value,dict):
        require(not any(re.sub('[^a-z]','',k.lower()) in {
            'apikey','authorization','accesstoken','refreshtoken','clientsecret','password',
            'privatekey','credentials','cookie','setcookie'} for k in value), 'NCAAF_CUSTODY_CREDENTIAL_FIELD')
        for v in value.values(): _credentials(v)
    elif isinstance(value,list):
        for v in value: _credentials(v)
    elif isinstance(value,str):
        require(re.search(r'(?i)(?:bearer\s+\S+|api[_-]?key\s*[=:]|-----BEGIN [A-Z ]*PRIVATE KEY)',value) is None,
            'NCAAF_CUSTODY_CREDENTIAL_FIELD')

def _parse(raw):
    require(isinstance(raw,bytes) and 0 < len(raw) <= MAX_BODY_BYTES, 'NCAAF_CUSTODY_LIMIT')
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
        # Includes overflowing exponent floats, not just NaN literals.
        json.dumps(parsed, allow_nan=False)
        _credentials(parsed)
    except (ValueError,TypeError,UnicodeError,RecursionError) as exc:
        raise ValueError(str(exc) if str(exc) == 'NCAAF_CUSTODY_CREDENTIAL_FIELD' else 'NCAAF_CUSTODY_BODY_SCHEMA') from None
    require(isinstance(parsed,list) and all(isinstance(r,dict) for r in parsed), 'NCAAF_CUSTODY_BODY_SCHEMA')
    return parsed

def _metadata(meta):
    require(isinstance(meta,dict) and set(meta) == set('provider endpoint request_scope status representation content_type complete received_at receipt_clock_meaning'.split()), 'NCAAF_CUSTODY_SCHEMA')
    require(meta['representation']==REPRESENTATION and meta['content_type']=='application/json', 'NCAAF_CUSTODY_REPRESENTATION')
    require(type(meta['status']) is int and meta['status']==200 and meta['complete'] is True, 'NCAAF_CUSTODY_HTTP_STATUS')
    require(history.timestamp(meta['received_at']) is not None and meta['receipt_clock_meaning']==RECEIPT_MEANING, 'NCAAF_CUSTODY_CLOCK_CONFLICT')
    p = meta['request_scope']
    require(isinstance(p,dict), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    _credentials(meta)
    if meta['provider']=='cfbd':
        if meta['endpoint']=='games':
            require(set(p)=={'year','seasonType','classification'} and p['seasonType']=='both' and p['classification']=='fbs', 'NCAAF_CUSTODY_SCOPE_CONFLICT')
        else:
            require(meta['endpoint']=='games/teams' and set(p)=={'year','week','seasonType'}
                and type(p['week']) is int and 0 <= p['week'] <= 30 and p['seasonType'] in {'regular','postseason'}, 'NCAAF_CUSTODY_SCOPE_CONFLICT')
        require(type(p['year']) is int and 1900 <= p['year'] <= 2200, 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    else:
        require(meta['provider']=='odds_api' and meta['endpoint']==ODDS_ENDPOINT and set(p)==set('regions markets bookmakers oddsFormat dateFormat commenceTimeFrom commenceTimeTo'.split()), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
        for key in ('regions','markets','bookmakers'):
            require(isinstance(p[key],str) and 0 < len(p[key]) <= 256 and re.fullmatch('[a-z0-9_,]+',p[key]) is not None, 'NCAAF_CUSTODY_SCOPE_CONFLICT')
            require(len(p[key].split(','))==len(set(p[key].split(','))), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
        require(set(p['regions'].split(',')) <= {'us','us2','eu','uk','au'} and set(p['markets'].split(',')) <= {'h2h','spreads','totals'}
            and p['oddsFormat']=='american' and p['dateFormat']=='iso', 'NCAAF_CUSTODY_SCOPE_CONFLICT')
        start,end=[history.timestamp(p[k]) for k in ('commenceTimeFrom','commenceTimeTo')]
        require(start is not None and end is not None and start < end, 'NCAAF_CUSTODY_SCOPE_CONFLICT')

def _project_cfbd(rows,meta,batch):
    req=meta['request_scope']
    expected=dict(kind='games',year=req['year']) if meta['endpoint']=='games' else dict(
        kind='stats',year=req['year'],week=req['week'],season_type=req['seasonType'])
    require(isinstance(batch,dict) and set(batch)=={'request','retrieved_at','records'} and batch['request']==expected, 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    try: projected=history._clean(expected['kind'],rows)
    except (ValueError,TypeError): raise ValueError('NCAAF_CUSTODY_BODY_SCHEMA') from None
    require(projected==batch['records'], 'NCAAF_CUSTODY_PROJECTION_CONFLICT')
    if expected['kind']=='games':
        require(all(r.get('season')==req['year'] for r in rows), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    else:
        require(all(all(r[key]==value for key,value in (('season',req['year']),('week',req['week']),
            ('seasonType',req['seasonType'])) if key in r) for r in rows), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    locators=[dict(record_index=i,game_id=r.get('id'),team_ids=([r.get('homeId'),r.get('awayId')] if expected['kind']=='games' else [t.get('teamId') for t in r.get('teams',[])])) for i,r in enumerate(rows)]
    require(all(type(r['game_id']) is int for r in locators) and len({r['game_id'] for r in locators})==len(locators), 'NCAAF_CUSTODY_PROJECTION_CONFLICT')
    return dict(native_sha256=hashlib.sha256(model.encode(batch)).hexdigest(),projected_at=batch['retrieved_at'],record_locators=locators)

def _project_quote(rows,meta,q,terms):
    """Exact raw locator plus independently reviewed missing listing declarations.

    No rule, listing or product defaults. Missing feed declarations must resolve
    through the already independently accepted exact terms/offer, never a template.
    """
    from app_core.prediction_evidence import provider_quotes
    matches=[]
    for i,g in enumerate(rows):
        if (g.get('id'),g.get('sport_key'),g.get('home_team'),g.get('away_team'),g.get('commence_time')) != (
            q['provider_event_id'],'americanfootball_ncaaf',q['event_home_team'],q['event_away_team'],q['event_start_utc']): continue
        require(g.get('odds_feed_source','the_odds_api')=='the_odds_api', 'NCAAF_CUSTODY_QUOTE_CONFLICT')
        require(isinstance(g.get('bookmakers'),list), 'NCAAF_CUSTODY_BODY_SCHEMA')
        for j,b in enumerate(g['bookmakers']):
            require(isinstance(b,dict), 'NCAAF_CUSTODY_BODY_SCHEMA')
            if b.get('key')!=q['book']: continue
            require(isinstance(b.get('markets'),list), 'NCAAF_CUSTODY_BODY_SCHEMA')
            for k,m in enumerate(b.get('markets',[])):
                require(isinstance(m,dict) and isinstance(m.get('outcomes'),list), 'NCAAF_CUSTODY_BODY_SCHEMA')
                for l,o in enumerate(m.get('outcomes',[])):
                    require(isinstance(o,dict), 'NCAAF_CUSTODY_BODY_SCHEMA')
                    single=dict(g,bookmakers=[dict(b,markets=[dict(m,outcomes=[o])])])
                    derived=json.loads(provider_quotes(single))
                    if len(derived)!=1: continue
                    v=derived[0]
                    if not all(q.get(key)==v.get(key) for key in ('provider_namespace','provider_event_id','event_home_team','event_away_team','event_start_utc','book','market_type','point','price','recorded_at')): continue
                    require(not (set(q)-set(v)-{'operator','listing_id','product'})
                        and all(q[key]==v[key] for key in set(q)&set(v)), 'NCAAF_CUSTODY_QUOTE_CONFLICT')
                    declarations={}
                    for key in ('operator','listing_id','product'):
                        require(q.get(key) and terms.get(key)==q[key], 'NCAAF_CUSTODY_QUOTE_CONFLICT')
                        present=[(path,value[key]) for path,value in (('book',b),('market',m),('outcome',o)) if key in value]
                        require(all(value==q[key] for _,value in present), 'NCAAF_CUSTODY_QUOTE_CONFLICT')
                        declarations[key]=dict(value=q[key],source_paths=[path+'.'+key for path,_ in present] or ['independently_accepted_terms_review.'+key])
                    matches.append(dict(event_index=i,book_index=j,market_index=k,outcome_index=l,
                        provider_clock_path=('market.last_update' if m.get('last_update') else 'book.last_update'),
                        quote_sha256=model.digest(q),terms_sha256=model.digest(terms),listing_declarations=declarations))
    require(len(matches)==1, 'NCAAF_CUSTODY_QUOTE_CONFLICT')
    req=meta['request_scope'];start,end=[history.timestamp(req[k]) for k in ('commenceTimeFrom','commenceTimeTo')]
    qt=history.timestamp(q['event_start_utc'])
    require(qt is not None and start<=qt<=end and q['book'] in req['bookmakers'].split(',')
        and ('spreads' if q['market_type'].startswith('spread') else 'totals') in req['markets'].split(','), 'NCAAF_CUSTODY_SCOPE_CONFLICT')
    return matches[0]

def capture(raw,metadata,*,native_batch=None,quote=None,terms=None):
    """Receive actual decoded body bytes; caller must supply genuine receipts.

    This performs no transport, persistence, permission or acceptance operation.
    """
    _metadata(metadata)
    rows=_parse(raw)
    if metadata['provider']=='cfbd':
        projection=_project_cfbd(rows,metadata,native_batch)
    else:
        require(isinstance(quote,dict) and isinstance(terms,dict), 'NCAAF_CUSTODY_QUOTE_CONFLICT')
        projection=_project_quote(rows,metadata,quote,terms)
    p=dict(version=OBJECT_VERSION,metadata=deepcopy(metadata),body_b64=base64.b64encode(raw).decode('ascii'),
        body_bytes=len(raw),body_sha256=hashlib.sha256(raw).hexdigest(),projection_version=PROJECTION_VERSION,
        projection_implementation=implementation(),projection=projection)
    return dict(payload=p,sha256=model.digest(p))

def load(packet):
    require(isinstance(packet,dict) and set(packet)=={'payload','sha256'} and isinstance(packet['payload'],dict), 'NCAAF_CUSTODY_SCHEMA')
    p=packet['payload']
    require(set(p)=={'version','evidence_label','native_packet','response_objects','custody_admission'}
        and p['version']==VERSION and p['evidence_label'] in {'SYNTHETIC','RETAINED'}, 'NCAAF_CUSTODY_SCHEMA')
    require(model.digest(p)==packet['sha256'], 'NCAAF_CUSTODY_INTEGRITY')
    require(isinstance(p['native_packet'],dict) and isinstance(p['native_packet'].get('payload'),dict)
        and p['native_packet']['payload'].get('version')==native.SUCCESSOR_VERSION, 'NCAAF_CUSTODY_SCHEMA')
    n=native.load(p['native_packet'])
    require(n['payload']['version']==native.SUCCESSOR_VERSION and n['payload']['evidence_label']==p['evidence_label'], 'NCAAF_CUSTODY_SCHEMA')
    require(isinstance(p['response_objects'],list) and bool(p['response_objects']), 'NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE')
    require(len(p['response_objects']) <= MAX_OBJECTS, 'NCAAF_CUSTODY_LIMIT')
    # Never stage credential-bearing originals for later rejection retention.
    # Other invalid body/link evidence is diagnosed precisely by verify().
    for item in p['response_objects']:
        if not isinstance(item,dict) or not isinstance(item.get('payload'),dict): continue
        body=item['payload'];_credentials(body.get('metadata'))
        encoded=body.get('body_b64')
        if isinstance(encoded,str) and len(encoded) <= ((MAX_BODY_BYTES+2)//3)*4:
            try: _parse(base64.b64decode(encoded,validate=True))
            except (ValueError,TypeError) as exc:
                if str(exc)=='NCAAF_CUSTODY_CREDENTIAL_FIELD': raise
    return packet

def view(packet): return native.view(packet['payload']['native_packet'])
def result_version(packet): return RESULT_VERSION

def subject_hash(packet):
    """No future acceptance/checkpoint/inference hash in exact verification."""
    p=packet['payload']
    return model.digest(dict(version=SUBJECT_VERSION,native_subject_sha256=chronology.dependency_subject_hash(p['native_packet']),
        evidence_label=p['evidence_label'],response_objects=p['response_objects']))

def verify(packet,approval,at):
    p=load(packet)['payload'];n=p['native_packet'];o=n['payload']['observation']['payload']
    # Preserves every #2407 check, including exact native subjects and availability.
    dependency_review=native.accepted_dependencies(n,approval,at)
    native.checked_observation(n,at)
    _,index=native.objects(n,history.timestamp(at))
    bodies=[];hashes=set();consumed=set();quote_count=0;total=0;clocks=[]
    terms=o['source_review']['terms_review'];permission=dependency_review['permissions_review']
    for item in p['response_objects']:
        require(isinstance(item,dict) and set(item)=={'payload','sha256'} and isinstance(item['payload'],dict), 'NCAAF_CUSTODY_SCHEMA')
        v=item['payload']
        require(set(v)==set('version metadata body_b64 body_bytes body_sha256 projection_version projection_implementation projection'.split())
            and v['version']==OBJECT_VERSION and v['projection_version']==PROJECTION_VERSION, 'NCAAF_CUSTODY_SCHEMA')
        require(model.digest(v)==item['sha256'], 'NCAAF_CUSTODY_INTEGRITY')
        require(v['projection_implementation']==implementation(), 'NCAAF_CUSTODY_RUNTIME_CHANGED')
        require(isinstance(v['body_b64'],str) and 0 < len(v['body_b64']) <= ((MAX_BODY_BYTES+2)//3)*4, 'NCAAF_CUSTODY_LIMIT')
        try: raw=base64.b64decode(v['body_b64'],validate=True)
        except (ValueError,TypeError): raise ValueError('NCAAF_CUSTODY_INTEGRITY') from None
        require(type(v['body_bytes']) is int and len(raw)==v['body_bytes'] and hashlib.sha256(raw).hexdigest()==v['body_sha256'], 'NCAAF_CUSTODY_INTEGRITY')
        total+=len(raw)
        require(total <= MAX_TOTAL_BODY_BYTES and v['body_sha256'] not in hashes, 'NCAAF_CUSTODY_LIMIT')
        hashes.add(v['body_sha256'])
        meta=v['metadata'];_metadata(meta);rows=_parse(raw)
        receipt=history.timestamp(meta['received_at']);now=history.timestamp(at)
        require(now is not None and receipt<=now, 'NCAAF_CUSTODY_CLOCK_CONFLICT')
        if meta['provider']=='cfbd':
            # Projection cannot restart the original input's 24-hour age.
            require(0 <= (now-receipt).total_seconds() <= 86400, 'NCAAF_CUSTODY_INPUT_STALE')
            sha=v['projection'].get('native_sha256');batch=index.get(sha)
            require(batch is not None and sha not in consumed, 'NCAAF_CUSTODY_PROJECTION_CONFLICT')
            computed=_project_cfbd(rows,meta,batch);consumed.add(sha)
            projection_at=history.timestamp(batch['retrieved_at'])
            require(receipt<=projection_at, 'NCAAF_CUSTODY_CLOCK_CONFLICT')
            reviewed,start,end=[history.timestamp(permission[k]) for k in ('reviewed_at','effective_from','effective_until')]
            require(reviewed<receipt and start<=receipt<end and now<end, 'NCAAF_CUSTODY_PERMISSION_CLOCK_CONFLICT')
            clocks.append(projection_at)
        else:
            computed=_project_quote(rows,meta,o['quote'],terms);quote_count+=1
            quote,observed=[history.timestamp(x) for x in (o['quote']['recorded_at'],o['source_review']['quote_observation']['observed_at'])]
            require(quote<=receipt<=observed, 'NCAAF_CUSTODY_CLOCK_CONFLICT')
            reviewed,start,end=[history.timestamp(terms[k]) for k in ('reviewed_at','effective_from','effective_until')]
            require(reviewed<receipt and start<=receipt<end and now<end, 'NCAAF_CUSTODY_PERMISSION_CLOCK_CONFLICT')
            clocks.append(observed)
        require(v['projection']==computed, 'NCAAF_CUSTODY_PROJECTION_CONFLICT')
        clocks.append(receipt);bodies.append(raw)
    require(consumed==set(index) and quote_count==1, 'NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE')
    r=p['custody_admission']
    require(isinstance(r,dict) and set(r)=={'version','verification','acceptance'} and r['version']==ADMISSION_VERSION, 'NCAAF_CUSTODY_ADMISSION_CONFLICT')
    v,a=r['verification'],r['acceptance']
    require(isinstance(v,dict) and set(v)==set('verified_at verifier subject_version subject_sha256 permissions_sha256 terms_sha256 native_verification_sha256'.split())
        and v['subject_version']==SUBJECT_VERSION and v['subject_sha256']==subject_hash(packet)
        and v['permissions_sha256']==model.digest(permission) and v['terms_sha256']==model.digest(terms)
        and v['native_verification_sha256']==model.digest(dependency_review['dependency_verification'])
        and isinstance(v['verifier'],str) and bool(v['verifier'].strip()), 'NCAAF_CUSTODY_ADMISSION_CONFLICT')
    require(isinstance(a,dict) and set(a)==set('review_id reviewer accepted_at subject_version subject_sha256 verification_sha256'.split())
        and all(isinstance(value,str) and value.strip() for value in a.values()) and a['reviewer']!=v['verifier']
        and a['subject_version']==SUBJECT_VERSION and a['subject_sha256']==v['subject_sha256']
        and a['verification_sha256']==model.digest(v), 'NCAAF_CUSTODY_ADMISSION_CONFLICT')
    require(ACCEPTED_ADMISSIONS.get(a['review_id'])==model.digest(a), 'NCAAF_CUSTODY_ADMISSION_NOT_TRUSTED')
    vt,accepted,checkpoint,inference=[history.timestamp(x) for x in (v['verified_at'],a['accepted_at'],o['as_of'],at)]
    clocks.append(history.timestamp(dependency_review['dependency_verification']['verified_at']))
    require(vt is not None and all(clocks) and all(t<=vt for t in clocks), 'NCAAF_CUSTODY_SUBJECT_FUTURE_FACT')
    require(all((accepted,checkpoint,inference)) and vt<=accepted<=checkpoint<inference, 'NCAAF_CUSTODY_CLOCK_CONFLICT')
    return dict(native_dependency_review=deepcopy(dependency_review),custody_admission=deepcopy(r),
        original_response_available=True,body_hashes=[item['payload']['body_sha256'] for item in p['response_objects']])

def accepted_dependencies(packet,approval,at): return verify(packet,approval,at)

def infer(packet,at):
    from app_core.ncaaf_pipeline_evidence import accepted
    verified=verify(packet,accepted(packet),at)
    checked,center,win,computation=native.infer(packet['payload']['native_packet'],at)
    receipt=dict(version=RESULT_VERSION,input_sha256=packet['sha256'],custody_subject_sha256=subject_hash(packet),
        verification=verified,native_computation=computation,raw_probability=win,inference_time=at,
        scientific_acceptance=False,probability_calibration=False,wagering_authority=False,live_stake=0)
    return checked,center,win,dict(payload=receipt,sha256=model.digest(receipt))

def inspect_result(packet,saved,at):
    """Static byte/link checks only; numerical historical replay is forbidden."""
    from app_core.ncaaf_pipeline_evidence import accepted
    verified=verify(packet,accepted(packet),at)
    require(isinstance(saved,dict) and set(saved)=={'payload','sha256'} and model.digest(saved['payload'])==saved['sha256'], 'NCAAF_CUSTODY_COMPUTATION_CONFLICT')
    r=saved['payload']
    require(set(r)==set('version input_sha256 custody_subject_sha256 verification native_computation raw_probability inference_time scientific_acceptance probability_calibration wagering_authority live_stake'.split())
        and r['version']==RESULT_VERSION and r['input_sha256']==packet['sha256'] and r['custody_subject_sha256']==subject_hash(packet)
        and r['verification']==verified and r['inference_time']==at and r['scientific_acceptance'] is False
        and r['probability_calibration'] is False and r['wagering_authority'] is False and r['live_stake']==0, 'NCAAF_CUSTODY_COMPUTATION_CONFLICT')
    checked,n=native.inspect_result(packet['payload']['native_packet'],r['native_computation'],at)
    require(r['raw_probability']==n['raw_probability'], 'NCAAF_CUSTODY_COMPUTATION_CONFLICT')
    return checked,deepcopy(r)
