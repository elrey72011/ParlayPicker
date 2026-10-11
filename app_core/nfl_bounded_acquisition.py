"""One fixed original-body request. Planning has no credential or storage access.

The documented current games.csv exceeds the unchanged ceiling. Its proposed
plan is BLOCKED; implementation capability is not acquisition authorization.
"""
from copy import deepcopy
from datetime import datetime, timezone
import http.client
import os
from pathlib import Path
import threading
import time
import zlib
from app_core import nfl_observed_custody as custody
from app_core.nfl_source_projection import seal
from app_core.ncaaf_pilot import _timed, _append
from app_core.producer_provenance import clock

VERSION = 'nfl-original-body-plan-v1'
AUTH_VERSION = 'nfl-original-body-authorization-v1'
ENDPOINT = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
REQUEST = dict(id='scores', host='raw.githubusercontent.com', endpoint='/nflverse/nfldata/master/data/games.csv', params={}, method='GET', max_attempts=1)
AUTHORIZED_COLLECTIONS = {}
REVIEWED_PERMISSIONS = {}
AUTHORIZED_CUSTODY_ROOTS = {}
require = custody.require
digest = custody.digest


def utc(): return datetime.now(timezone.utc).isoformat()


def validate(plan, *, executable=False):
    require(isinstance(plan, dict) and set(plan) == {'payload','sha256'} and digest(plan['payload']) == plan['sha256'], 'NFL_ACQUIRE_PLAN_INTEGRITY')
    p = plan['payload']
    from app_core.ncaaf_response_custody import _credentials
    try: _credentials(p)
    except ValueError: raise ValueError('NFL_ACQUIRE_CREDENTIAL_FORBIDDEN') from None
    require(set(p) == set('version evidence_label target requests limits window permission authorization_ref custody_id unresolved'.split())
        and p['version'] == VERSION and p['evidence_label'] in {'SYNTHETIC','PROPOSAL','RETAINED'}, 'NFL_ACQUIRE_PLAN_SCHEMA')
    require(p['requests'] == [REQUEST], 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
    require(set(p['target']) == {'canonical_event_id','season'}, 'NFL_ACQUIRE_TARGET')
    limits = p['limits']
    require(set(limits) == set('max_attempts max_objects max_body_bytes max_aggregate_bytes connect_seconds read_seconds total_seconds'.split())
        and limits['max_attempts'] == limits['max_objects'] == 1
        and type(limits['max_body_bytes']) is int and 0 < limits['max_body_bytes'] <= custody.MAX_BODY
        and type(limits['max_aggregate_bytes']) is int and limits['max_body_bytes'] <= limits['max_aggregate_bytes'] <= custody.MAX_TOTAL
        and all(type(limits[k]) in {int,float} and 0 < limits[k] <= 120 for k in ('connect_seconds','read_seconds','total_seconds'))
        and limits['connect_seconds'] <= limits['total_seconds'] and limits['read_seconds'] <= limits['total_seconds'], 'NFL_ACQUIRE_LIMITS')
    require(set(p['window']) == {'not_before','not_after','expires_at'} and all(clock(t) for t in p['window'].values())
        and clock(p['window']['not_before']) < clock(p['window']['not_after']) <= clock(p['window']['expires_at']), 'NFL_ACQUIRE_WINDOW')
    require(isinstance(p['unresolved'], list) and custody.known(p['custody_id']) and custody.known(p['authorization_ref']), 'NFL_ACQUIRE_PLAN_SCHEMA')
    if executable:
        require(not p['unresolved'] and p['evidence_label'] != 'PROPOSAL' and custody.known(p['target']['canonical_event_id'])
            and type(p['target']['season']) is int, 'NFL_ACQUIRE_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p = validate(plan)
    return dict(plan_sha256=plan['sha256'], status='BLOCKED' if p['unresolved'] else 'AWAITING_EXACT_AUTHORIZATION',
        requests=1, credentials=0, storage_operations=0, network_requests=0, accepted=False, inference=False, unresolved=p['unresolved'])


def authorize(plan, authorization, at, root):
    p = validate(plan, executable=True); now = clock(at); w = p['window']
    require(now and clock(w['not_before']) <= now <= clock(w['not_after']) and now < clock(w['expires_at']), 'NFL_ACQUIRE_WINDOW_EXPIRED')
    require(isinstance(authorization,dict) and set(authorization) == {'payload','sha256'}
        and digest(authorization['payload']) == authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION')
    a = authorization['payload']
    require(set(a) == set('version reference plan_sha256 custody_id owner authorized_at expires_at maximum_additional_spend'.split())
        and a['version'] == AUTH_VERSION and a['reference'] == p['authorization_ref'] and a['plan_sha256'] == plan['sha256']
        and a['custody_id'] == p['custody_id'] and a['owner'] == 'Robert Velarde' and a['maximum_additional_spend'] == 0
        and AUTHORIZED_COLLECTIONS.get(a['reference']) == authorization['sha256'], 'NFL_ACQUIRE_AUTHORIZATION_UNTRUSTED')
    require(clock(a['authorized_at']) and clock(a['expires_at']) and clock(a['authorized_at']) <= now < clock(a['expires_at']) <= clock(w['expires_at']), 'NFL_ACQUIRE_AUTHORIZATION_EXPIRED')
    r = p['permission']
    require(isinstance(r,dict) and set(r) == set('reference reviewer reviewed_at effective_from effective_until endpoint private_retention private_research_use basis requests_sha256'.split())
        and REVIEWED_PERMISSIONS.get(r['reference']) == digest(r) and r['reviewer'] == 'Robert Velarde' and r['endpoint'] == ENDPOINT
        and r['private_retention'] is True and r['private_research_use'] is True and isinstance(r['basis'],str) and len(r['basis']) >= 16
        and r['requests_sha256'] == digest(p['requests']), 'NFL_ACQUIRE_PERMISSION_UNTRUSTED')
    require(all(clock(r[k]) for k in ('reviewed_at','effective_from','effective_until'))
        and clock(r['reviewed_at']) < now and clock(r['effective_from']) <= now < clock(w['not_after']) < clock(r['effective_until']), 'NFL_ACQUIRE_PERMISSION_EXPIRED')
    require(p['custody_id'] in AUTHORIZED_CUSTODY_ROOTS and Path(AUTHORIZED_CUSTODY_ROOTS[p['custody_id']]).resolve() == Path(root).resolve()
        and Path(root).is_dir(), 'NFL_ACQUIRE_CUSTODY_UNVERIFIED')
    return p


class HttpsTransport:
    """Fixed public HTTPS source only; no credentials, redirects or library calls."""
    def __call__(self, request, limits, deadline, monotonic):
        require(request == REQUEST, 'NFL_ACQUIRE_ENDPOINT_NOT_APPROVED')
        cancelled = threading.Event()
        conn = http.client.HTTPSConnection(request['host'], timeout=min(limits['connect_seconds'], max(.001,deadline-monotonic())))
        conn.auto_open = False
        try:
            _timed(conn.connect, min(deadline,monotonic()+limits['connect_seconds']), monotonic, cancelled, conn.close)
            conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
            _timed(lambda:conn.request('GET',request['endpoint'],headers={'Accept':'text/csv','Accept-Encoding':'identity'}), min(deadline,monotonic()+limits['read_seconds']), monotonic,cancelled,conn.close)
            response = _timed(conn.getresponse,min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
        except Exception:
            conn.close(); raise ValueError('NFL_ACQUIRE_TRANSPORT_FAILURE') from None
        class Stream:
            status = response.status
            headers = {k.lower():v for k,v in response.getheaders()}
            def chunks(self):
                while True:
                    require(monotonic() < deadline,'NFL_ACQUIRE_DEADLINE')
                    conn.sock.settimeout(min(limits['read_seconds'],max(.001,deadline-monotonic())))
                    block = _timed(lambda:response.read1(8192),min(deadline,monotonic()+limits['read_seconds']),monotonic,cancelled,conn.close)
                    if not block: break
                    yield block
            def close(self): conn.close()
        return Stream()


def read(stream, limit, deadline, monotonic):
    h = {k.lower():str(v) for k,v in stream.headers.items()}
    require(stream.status == 200, 'NFL_ACQUIRE_HTTP_STATUS')
    require(not any(h.get(k) for k in ('location','x-next-page','x-next-page-token')) and 'next' not in h.get('link','').lower(), 'NFL_ACQUIRE_REDIRECT_OR_PAGINATION')
    require(h.get('content-type','').split(';')[0].strip() in {'text/plain','text/csv','application/csv'}, 'NFL_ACQUIRE_CONTENT_TYPE')
    encoding = h.get('content-encoding','identity')
    require(encoding in {'identity','gzip'}, 'NFL_ACQUIRE_CONTENT_ENCODING')
    declared = h.get('content-length')
    require(declared is None or declared.isdigit() and 0 < int(declared) <= limit, 'NFL_ACQUIRE_BODY_LIMIT')
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS) if encoding == 'gzip' else None
    body = bytearray(); wire = 0
    for chunk in stream.chunks():
        require(monotonic() < deadline, 'NFL_ACQUIRE_DEADLINE')
        require(isinstance(chunk,bytes) and 0 < len(chunk) <= limit, 'NFL_ACQUIRE_INCOMPLETE_BODY')
        wire += len(chunk); require(wire <= limit,'NFL_ACQUIRE_BODY_LIMIT')
        body.extend(decoder.decompress(chunk,limit-len(body)+1) if decoder else chunk)
        require(len(body) <= limit and (decoder is None or not decoder.unconsumed_tail),'NFL_ACQUIRE_BODY_LIMIT')
    require(monotonic() < deadline, 'NFL_ACQUIRE_DEADLINE')
    require(declared is None or wire == int(declared), 'NFL_ACQUIRE_INCOMPLETE_BODY')
    require(decoder is None or decoder.eof and not decoder.unused_data, 'NFL_ACQUIRE_INCOMPLETE_BODY')
    return bytes(body)


def acquire(plan, authorization, *, root, transport, wall=utc, monotonic=time.monotonic):
    p = authorize(plan,authorization,wall(),root); root = Path(root)
    journal_path = root/(plan['sha256']+'.nfl-attempts.jsonl')
    try: journal = journal_path.open('xb')
    except FileExistsError: raise ValueError('NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME') from None
    deadline = monotonic()+p['limits']['total_seconds']; obj = None; code = None; attempted = 0
    try:
        with journal:
            _append(journal,dict(plan_sha256=plan['sha256'],authorization_sha256=authorization['sha256'],custody_id=p['custody_id']))
            authorize(plan,authorization,wall(),root)
            require(monotonic() < deadline,'NFL_ACQUIRE_DEADLINE')
            started = wall(); attempted = 1
            _append(journal,dict(attempt=1,request=REQUEST,requested_at=started))
            stream = None
            try:
                stream = transport(deepcopy(REQUEST),deepcopy(p['limits']),deadline,monotonic)
                body = read(stream,p['limits']['max_body_bytes'],deadline,monotonic)
            finally:
                if stream is not None: stream.close()
            received = wall(); authorize(plan,authorization,received,root)
            obj = custody.retained(body,receipt=dict(source_id='nflverse:nfldata:games',endpoint=ENDPOINT,format='csv',evidence_label=p['evidence_label'],
                request_started_at=started,received_at=received,observed_at=received,
                observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='publisher_availability_unknown'))
            _append(journal,dict(received_at=received,body_sha256=obj['body_sha256']))
    except Exception as exc:
        code = str(exc) if str(exc).startswith(('NFL_ACQUIRE_','NFL_PRIVATE_')) and str(exc).replace('_','').isalnum() else 'NFL_ACQUIRE_STOPPED'
    bundle = seal(dict(version='nfl-unaccepted-original-body-v1',plan=plan,authorization_sha256=authorization['sha256'],attempted=attempted,
        status='INCOMPLETE' if code else 'CAPTURED_UNREVIEWED',reason=code,object=obj,owner_reviewed=False,inference=False,source_acceptance=False,wagering_authority=False))
    path = root/(plan['sha256']+'.nfl-unreviewed.json')
    with path.open('xb') as f: f.write(custody.encode(bundle).encode()); f.flush(); os.fsync(f.fileno())
    require(path.read_bytes() == custody.encode(bundle).encode(),'NFL_ACQUIRE_PRIVATE_READBACK')
    return bundle


def admitted_analysis(packet, source, *, inventories=()):
    """Supplied inputs only; caller performs all verification before mathematics."""
    import pandas as pd
    from app_core.nfl_private_routes import selected
    from app_core.market_probability_model import predict_market_probabilities
    frame = pd.DataFrame([deepcopy(source)])
    with selected([packet]):
        predictions = predict_market_probabilities(frame)
    for field in predictions: frame[field] = predictions[field]
    # This isolated lane did not run a blend or value calculation. Supplied
    # legacy aliases cannot become a newly recorded blend, EV or authority.
    for field in ('calibrated_probability','expected_value','expected_value_percent','best_available_probability','best_available_edge','best_available_ev'):
        frame[field] = float('nan')
    from app_core.slate_coverage import ET, build_coverage
    # Independent schedule scope survives a rejected packet. Its rejected
    # descriptors cannot determine the coverage date or remove scheduled games.
    dates = {i['selected_date'] for i in inventories if i.get('selected_date')}
    require(len(dates) <= 1, 'NFL_ACQUIRE_INVENTORY_DATE_CONFLICT')
    if dates:
        day = next(iter(dates))
    else:
        start = clock(source.get('game_start_utc')) or clock(source.get('commence_time'))
        require(start is not None, 'NFL_ACQUIRE_INDEPENDENT_COVERAGE_DATE_REQUIRED')
        day = start.astimezone(ET).date().isoformat()
    at = frame.iloc[0].prediction_generated_at
    frame.attrs['nfl_private_inventories'] = deepcopy(inventories)
    frame.attrs['slate_coverage'] = build_coverage(inventories,selected_date=day,as_of=at,
        run_id='nfl-private-'+digest(dict(packet=packet['sha256'],at=at)),candidates=frame,leagues=['NFL'])
    return frame


def retain_analysis(analysis, *, path):
    """Existing local capture/export, never remote sync or store initialization."""
    import sqlite3
    path = Path(path)
    require(path.is_file(),'NFL_ACQUIRE_EXISTING_PREDICTION_STORE_REQUIRED')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    require({'bundles','snapshots','snapshot_runtime','research_replay_sources','research_replay_exports','research_source_intakes'} <= tables,
        'NFL_ACQUIRE_EXISTING_PREDICTION_STORE_REQUIRED')
    from app_core import prediction_evidence as evidence, research_replay
    from app_core.candidate_evidence_schema import project
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package, validate_package
    from core.streamlit_pipeline import build_best_picks_df
    from core.live_wager_contract import finalize_live_wagers
    from app_core.slate_coverage import build_coverage, publication_rows
    diagnostics = {}; best = build_best_picks_df(analysis,diagnostics_out=diagnostics)
    audit = project(diagnostics['candidate_authority_df'])
    final,_ = finalize_live_wagers(audit,best,1,now=datetime.now(timezone.utc),policies={},config={})
    require(all(c['production_bet_amount'] == 0 for c in final.wager_contract),'NFL_ACQUIRE_AUTHORITY_FORBIDDEN')
    context = evidence.begin_run(dict(sports=['NFL'],use_ml=True,use_gemini=False,bankroll=1),path=path,root=Path(__file__).resolve().parents[1])
    captured,card = evidence.capture_run(context,audit,final,analysis,path=path,authoritative_candidates=True)
    original = analysis.attrs['slate_coverage']
    coverage = build_coverage(analysis.attrs['nfl_private_inventories'],selected_date=original['selected_date'],as_of=original['as_of'],
        run_id=str(captured.iloc[0].export_run_id),candidates=captured,final=card,leagues=['NFL'])
    display,_ = publication_rows(card,captured,coverage)
    frames = [per_game_board(display,captured,family=f,novig_only=True) for f in ('overall','sides','totals')]
    package = build_package(*frames); validate_package(package)
    receipt = research_replay.retain_export(frames,package,card,captured,path=path)
    return dict(captured=captured,card=card,frames=frames,package=package,export=receipt,coverage=coverage,db=path)
