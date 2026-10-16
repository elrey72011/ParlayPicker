"""Separately authorized history/quote stages; original bytes stay unaccepted.

Active monotonic time and decoded byte budgets are shared, not reset by staging.
Exclusive history claims prevent a different quote plan from spending it twice.
No model invocation, offer admission, source catalog, retry or credential store.
"""
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import time
from app_core import ncaaf_pilot as transport
from app_core import ncaaf_response_custody as custody, ncaaf_model_compatibility as model
from app_core.ncaaf_history import timestamp
from app_core.ncaaf_pilot_setup import inspect_security

HISTORY_PLAN = 'ncaaf-cfbd-history-stage-plan-v1'
QUOTE_PLAN = 'ncaaf-final-quote-stage-plan-v1'
HISTORY_BUNDLE = 'ncaaf-cfbd-history-unaccepted-v1'
QUOTE_BUNDLE = 'ncaaf-final-quote-unaccepted-v1'
AUTH_VERSION = 'ncaaf-staged-collection-authorization-v1'
MAX_BUNDLE = 6*1024*1024
LIMITS = dict(max_body_bytes=512*1024, max_combined_original_bytes=2*1024*1024,
              max_bundle_bytes=MAX_BUNDLE, max_objects=16, connect_seconds=10, read_seconds=10,
              shared_active_seconds=60, max_odds_credits=1, additional_spend=0)
AUTHORIZED_COLLECTIONS = {}
ACCEPTED_ADVANCE_PERMISSIONS = {}
AUTHORIZED_CUSTODY_ROOTS = {}
AUTHORIZED_ACCOUNT_EVIDENCE = {}
AUTHORIZED_CONFIGURATIONS = {}
CONTROLLED = ContextVar('ncaaf_staged_controlled', default=None)
require = model.require
seal = transport.seal


def _lookup(name, key):
    c = CONTROLLED.get()
    return (c[name] if c is not None else globals()[name]).get(key)


def configuration(plan, raw, expected_hash, *, at):
    """Validate a typed control envelope, distinct from original-response JSON."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= 128*1024 and hashlib.sha256(raw).hexdigest() == expected_hash,
            'NCAAF_STAGED_CONFIGURATION')
    c = json.loads(raw, object_pairs_hook=custody._unique, parse_constant=custody._constant); p = _envelope(c)
    require(set(p) == set('version plan_sha256 authorization permission custody account review'.split())
            and p['version'] == 'ncaaf-staged-controlled-bindings-v1' and p['plan_sha256'] == plan['sha256']
            and [p['permission']] == plan['payload']['advance_permissions'], 'NCAAF_STAGED_CONFIGURATION')
    r = p['review']; now = timestamp(at)
    require(set(r) == {'reviewer', 'reviewed_at', 'subject_sha256', 'finding'} and r['reviewer'] == 'Robert Velarde'
            and r['subject_sha256'] == model.digest({k: v for k, v in p.items() if k != 'review'})
            and timestamp(r['reviewed_at']) is not None and timestamp(r['reviewed_at']) < now
            and isinstance(r['finding'], str) and len(r['finding'].strip()) >= 32, 'NCAAF_STAGED_CONFIGURATION_REVIEW')
    d = p['custody']
    require(set(d) == {'id', 'root', 'storage_receipt_sha256'} and d['id'] == plan['payload']['custody_id']
            and isinstance(d['root'], str) and (Path(d['root']).is_absolute() or PureWindowsPath(d['root']).is_absolute()) and isinstance(d['storage_receipt_sha256'], str)
            and len(d['storage_receipt_sha256']) == 64, 'NCAAF_STAGED_CUSTODY')
    authorize(plan, p['authorization'], at, trusted=False)
    account=p['account']; provider=plan['payload']['requests'][0]['provider']
    require(isinstance(account, dict) and set(account) == set('provider included_remaining observed_at valid_until evidence_sha256 no_paid_overage reviewer'.split())
            and account['provider'] == provider and account['reviewer'] == 'Robert Velarde'
            and type(account['included_remaining']) is int and account['included_remaining'] >= len(plan['payload']['requests'])
            and account['no_paid_overage'] is True and isinstance(account['evidence_sha256'], str) and len(account['evidence_sha256']) == 64
            and timestamp(account['observed_at']) is not None and timestamp(account['valid_until']) is not None
            and timestamp(account['observed_at']) <= now < timestamp(account['valid_until']), 'NCAAF_STAGED_ACCOUNT_UNVERIFIED')
    require(all(timestamp(t) <= timestamp(r['reviewed_at']) for t in
                (p['authorization']['payload']['authorized_at'],p['permission']['reviewed_at'],account['observed_at'])),
            'NCAAF_STAGED_CONFIGURATION_REVIEW_CLOCK')
    custody._credentials({k:v for k,v in p.items() if k != 'authorization'})
    return c


@contextmanager
def controlled(plan, raw, expected_hash, *, root, at):
    """Exact separately reviewed configuration; never writes a trust catalog."""
    c = configuration(plan, raw, expected_hash, at=at); p = c['payload']; d = p['custody']
    require(Path(d['root']).resolve() == Path(root).resolve(), 'NCAAF_STAGED_CUSTODY')
    a = p['authorization']['payload']; permission = p['permission']
    token = CONTROLLED.set(dict(AUTHORIZED_COLLECTIONS={a['reference']:p['authorization']['sha256']},
        ACCEPTED_ADVANCE_PERMISSIONS={permission['reference']:model.digest(permission)}, AUTHORIZED_CUSTODY_ROOTS={d['id']:d['root']},
        AUTHORIZED_ACCOUNT_EVIDENCE={a['reference']:p['account']}, AUTHORIZED_CONFIGURATIONS={plan['sha256']:dict(file_sha256=expected_hash, file_b64=base64.b64encode(raw).decode())}))
    try:
        _root(root, plan['payload']); authorize(plan, p['authorization'], at)
        yield p['authorization']
    finally: CONTROLLED.reset(token)


def _envelope(v):
    require(isinstance(v, dict) and set(v) == {'payload', 'sha256'} and isinstance(v['payload'], dict)
            and model.digest(v['payload']) == v['sha256'], 'NCAAF_STAGED_INTEGRITY')
    return v['payload']


def requests(target, stage):
    year = timestamp(target['start_utc']).year
    def req(i, provider, endpoint, params):
        return dict(id=i, provider=provider, host=transport.HOSTS[provider], endpoint=endpoint, params=params, max_attempts=1)
    if stage == HISTORY_PLAN:
        return [req('cfbd-season-scoring-'+str(year), 'cfbd', 'games', dict(year=year, seasonType='both', classification='fbs'))] + [
            req('cfbd-yardage-'+str(year)+'-week-'+str(w), 'cfbd', 'games/teams', dict(year=year, week=w, seasonType='regular')) for w in (1, 2, 3)]
    from datetime import timedelta
    start = timestamp(target['start_utc'])
    iso = lambda t: t.isoformat().replace('+00:00', 'Z')
    return [req('final-spreads', 'odds_api', 'v4/sports/americanfootball_ncaaf/odds', dict(regions='us', bookmakers='draftkings',
        markets='spreads', oddsFormat='american', dateFormat='iso', commenceTimeFrom=iso(start-timedelta(seconds=1)),
        commenceTimeTo=iso(start+timedelta(seconds=1))))]


def validate_plan(plan, *, executable=False):
    p = _envelope(plan)
    require(set(p) == set('version evidence_label target requests limits execution_window advance_permissions collection_authorization_ref custody_id parent_history_sha256 prequote_sha256 unresolved'.split())
            and p['version'] in {HISTORY_PLAN, QUOTE_PLAN} and p['evidence_label'] in {'SYNTHETIC', 'RETAINED', 'PROPOSAL'}, 'NCAAF_STAGED_PLAN_SCHEMA')
    custody._credentials(p)
    t = p['target']
    require(isinstance(t, dict) and set(t) == set('provider_event_id home_team away_team start_utc neutral_site'.split())
            and t['home_team'] == 'Tulane Green Wave' and t['away_team'] == 'Memphis Tigers'
            and isinstance(t['provider_event_id'], str) and bool(t['provider_event_id'])
            and timestamp(t['start_utc']) is not None and type(t['neutral_site']) is bool, 'NCAAF_STAGED_TARGET')
    require(t['start_utc'] == '2026-10-16T23:30:00Z' and t['neutral_site'] is False
            and (p['evidence_label'] != 'RETAINED' or t['provider_event_id'] == '55f376873e3cd0c2bb7cc9aa26e47235'), 'NCAAF_STAGED_TARGET')
    require(p['limits'] == LIMITS and p['requests'] == requests(t, p['version']), 'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    is_history = p['version'] == HISTORY_PLAN
    require((p['parent_history_sha256'] is None and p['prequote_sha256'] is None) if is_history else
            all(isinstance(p[k], str) and len(p[k]) == 64 for k in ('parent_history_sha256', 'prequote_sha256')), 'NCAAF_STAGED_PARENT')
    w = p['execution_window']
    require(isinstance(w, dict) and set(w) == {'not_before', 'not_after', 'authorization_expires_at'}
            and all(timestamp(v) for v in w.values()) and timestamp(w['not_before']) < timestamp(w['not_after'])
            <= timestamp(w['authorization_expires_at']) < timestamp(t['start_utc']), 'NCAAF_STAGED_WINDOW')
    require(isinstance(p['unresolved'], list) and isinstance(p['advance_permissions'], list)
            and all(isinstance(p[k], str) and p[k] for k in ('collection_authorization_ref', 'custody_id')), 'NCAAF_STAGED_PLAN_SCHEMA')
    if executable:
        require(not p['unresolved'] and p['evidence_label'] != 'PROPOSAL', 'NCAAF_STAGED_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p = validate_plan(plan)
    return dict(plan_sha256=plan['sha256'], version=p['version'], requests=p['requests'], limits=p['limits'],
                status='NON_EXECUTABLE_PROPOSAL' if p['evidence_label'] == 'PROPOSAL' or p['unresolved'] else 'AWAITING_EXACT_AUTHORIZATION',
                network_requests=0, credential_loads=0, storage_operations=0, accepted=False, inference=False)


def authorize(plan, authorization, at, *, trusted=True):
    p = validate_plan(plan, executable=True); a = _envelope(authorization); now = timestamp(at); w = p['execution_window']
    require(set(a) == set('version reference plan_sha256 custody_id authorized_at expires_at owner'.split()) and a['version'] == AUTH_VERSION
            and a['reference'] == p['collection_authorization_ref'] and a['plan_sha256'] == plan['sha256']
            and a['custody_id'] == p['custody_id'] and a['owner'] == 'Robert Velarde', 'NCAAF_STAGED_AUTHORIZATION')
    require(now is not None and timestamp(a['authorized_at']) is not None and timestamp(a['expires_at']) is not None
            and timestamp(a['authorized_at']) <= timestamp(w['not_before']) <= now <= timestamp(w['not_after'])
            and now < timestamp(a['expires_at']) <= timestamp(w['authorization_expires_at']), 'NCAAF_STAGED_AUTHORIZATION_CLOCK')
    if trusted:
        require(_lookup('AUTHORIZED_COLLECTIONS', a['reference']) == authorization['sha256'], 'NCAAF_STAGED_AUTHORIZATION_UNTRUSTED')
    provider = 'cfbd' if p['version'] == HISTORY_PLAN else 'odds_api'
    require(len(p['advance_permissions']) == 1, 'NCAAF_STAGED_PERMISSION')
    r = p['advance_permissions'][0]
    require(set(r) == set('reference provider reviewed_at effective_from effective_until permitted_use requests_sha256'.split())
            and r['provider'] == provider and r['requests_sha256'] == model.digest(p['requests'])
            and r['permitted_use'] == 'private_prospective_capture', 'NCAAF_STAGED_PERMISSION')
    require(all(timestamp(r[k]) for k in ('reviewed_at', 'effective_from', 'effective_until'))
            and timestamp(r['reviewed_at']) < timestamp(w['not_before']) and timestamp(r['effective_from']) <= now
            and timestamp(w['authorization_expires_at']) <= timestamp(r['effective_until']), 'NCAAF_STAGED_PERMISSION_CLOCK')
    if trusted:
        require(_lookup('ACCEPTED_ADVANCE_PERMISSIONS', r['reference']) == model.digest(r), 'NCAAF_STAGED_PERMISSION_UNTRUSTED')
        account = _lookup('AUTHORIZED_ACCOUNT_EVIDENCE', a['reference'])
        require(isinstance(account, dict) and set(account) == set('provider included_remaining observed_at valid_until evidence_sha256 no_paid_overage reviewer'.split())
                and account['provider'] == provider and account['reviewer'] == 'Robert Velarde'
                and type(account['included_remaining']) is int and account['included_remaining'] >= len(p['requests'])
                and account['no_paid_overage'] is True and isinstance(account['evidence_sha256'], str) and len(account['evidence_sha256']) == 64
                and timestamp(account['observed_at']) is not None and timestamp(account['valid_until']) is not None
                and timestamp(account['observed_at']) <= now < timestamp(account['valid_until']), 'NCAAF_STAGED_ACCOUNT_UNVERIFIED')
    return p


def _exclusive(path, raw):
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    require(path.read_bytes() == raw, 'NCAAF_STAGED_READBACK')


def _root(root, p):
    root = Path(root)
    require(root.is_dir() and _lookup('AUTHORIZED_CUSTODY_ROOTS', p['custody_id']) is not None
            and root.resolve() == Path(_lookup('AUTHORIZED_CUSTODY_ROOTS', p['custody_id'])).resolve(), 'NCAAF_STAGED_CUSTODY')
    require(inspect_security(root)['status'] == 'VERIFIED', 'NCAAF_STAGED_SECURITY')
    return root


def acquire(plan, authorization, *, root, transport, history_bundle=None, prequote=None, wall=transport.utc, monotonic=time.monotonic):
    """Exactly one stage; no implicit next stage, resume or admission."""
    p = authorize(plan, authorization, wall()); root = _root(root, p)
    control = deepcopy(_lookup('AUTHORIZED_CONFIGURATIONS', plan['sha256']))
    require(isinstance(control, dict) and set(control) == {'file_sha256', 'file_b64'}
            and isinstance(control['file_b64'], str) and len(control['file_b64']) <= ((128*1024+2)//3)*4, 'NCAAF_STAGED_CONFIGURATION_REQUIRED')
    c = configuration(plan, base64.b64decode(control['file_b64'], validate=True), control['file_sha256'], at=wall())['payload']
    require(c['authorization'] == authorization and Path(c['custody']['root']).resolve() == root.resolve()
            and c['account'] == _lookup('AUTHORIZED_ACCOUNT_EVIDENCE', authorization['payload']['reference']), 'NCAAF_STAGED_CONFIGURATION')
    prior_bytes = 0; prior_active = 0.
    if p['version'] == QUOTE_PLAN:
        h = read_bundle(history_bundle)
        require(h['version'] == HISTORY_BUNDLE and history_bundle['sha256'] == p['parent_history_sha256']
                and h['plan']['payload']['target'] == p['target'] and h['plan']['payload']['custody_id'] == p['custody_id'], 'NCAAF_STAGED_PARENT')
        history_control = json.loads(base64.b64decode(h['control_receipt']['file_b64'], validate=True))['payload']
        require(history_control['custody'] == c['custody'], 'NCAAF_STAGED_PARENT_CUSTODY')
        from app_core.ncaaf_staged_evidence import validate_prequote
        validate_prequote(prequote, history_bundle, at=wall(), review_required=True)
        require(prequote['sha256'] == p['prequote_sha256'], 'NCAAF_STAGED_PREQUOTE_CONFLICT')
        require(timestamp(prequote['payload']['history_review']['reviewed_at']) <= timestamp(authorization['payload']['authorized_at']),
                'NCAAF_STAGED_PREQUOTE_AUTHORIZATION_CLOCK')
        # Parent originals and journal must still exist byte-for-byte in this custody.
        require((root/(h['plan']['sha256']+'.stage.json')).read_bytes() == model.encode(history_bundle), 'NCAAF_STAGED_PARENT_CUSTODY')
        verify_journal(root, history_bundle)
        prior_bytes, prior_active = h['original_bytes'], h['active_seconds']
    started_mono = monotonic(); remaining = LIMITS['shared_active_seconds']-prior_active
    require(remaining > 0 and prior_bytes < LIMITS['max_combined_original_bytes'], 'NCAAF_STAGED_COMBINED_LIMIT')
    deadline = started_mono+remaining
    started_at=wall(); authorize(plan,authorization,started_at)
    journal_path = root/(plan['sha256']+'.attempts.jsonl')
    try: journal = journal_path.open('xb')
    except FileExistsError: raise ValueError('NCAAF_STAGED_ALREADY_SPENT') from None
    objects = []; attempts = 0; code = None; total = prior_bytes; last_elapsed = 0.
    with journal:
        transport_module = globals()['transport']
        append = transport_module._append
        append(journal, dict(kind='start', plan_sha256=plan['sha256'], authorization_sha256=authorization['sha256'],
            started_at=started_at, parent_history_sha256=p['parent_history_sha256'], control_receipt_sha256=model.digest(control), prior_original_bytes=prior_bytes, prior_active_seconds=prior_active))
        try:
            if p['version'] == QUOTE_PLAN:
                # Durable exclusive claim survives any interruption, even before GET.
                try: _exclusive(root/(p['parent_history_sha256']+'.quote-claim.json'), model.encode(dict(plan_sha256=plan['sha256'])))
                except FileExistsError: raise ValueError('NCAAF_STAGED_HISTORY_ALREADY_CLAIMED') from None
            for req in p['requests']:
                authorize(plan, authorization, wall())
                require(monotonic() < deadline, 'NCAAF_STAGED_DEADLINE')
                attempts += 1; requested = wall()
                append(journal, dict(kind='attempt', attempt=attempts, request=req, requested_at=requested))
                stream = None
                try:
                    stream = transport(deepcopy(req), dict(connect_seconds=10, read_seconds=10), deadline, monotonic)
                    raw = transport_module._read(stream, min(LIMITS['max_body_bytes'], LIMITS['max_combined_original_bytes']-total), deadline, monotonic)
                    if hasattr(transport, 'validate_body'): transport.validate_body(raw)
                    received = wall(); authorize(plan, authorization, received)
                    rows = custody._parse(raw)
                    if req['provider'] == 'odds_api': _target(rows, p['target'])
                    total += len(raw)
                    require(total <= LIMITS['max_combined_original_bytes'] and monotonic() < deadline, 'NCAAF_STAGED_COMBINED_LIMIT')
                    meta = dict(provider=req['provider'], endpoint=req['endpoint'], request_scope=req['params'], status=200,
                        representation=custody.REPRESENTATION, content_type='application/json', complete=True,
                        received_at=received, receipt_clock_meaning=custody.RECEIPT_MEANING)
                    obj = dict(request_id=req['id'], requested_at=requested, metadata=meta, body_b64=base64.b64encode(raw).decode(),
                               body_bytes=len(raw), body_sha256=hashlib.sha256(raw).hexdigest())
                    objects.append(obj); last_elapsed = monotonic()-started_mono
                    append(journal, dict(kind='complete', attempt=attempts, body_sha256=obj['body_sha256'], body_bytes=len(raw),
                                         received_at=received, stage_active_seconds=last_elapsed))
                finally:
                    if stream is not None: stream.close()
        except BaseException as exc:
            # No exception text, body, secret, headers or traceback enters evidence.
            safe = str(exc) if isinstance(exc, ValueError) else ''
            code = safe if safe in SAFE_CODES else 'NCAAF_STAGED_ACQUISITION_FAILURE'
        elapsed = monotonic()-started_mono
        if not math.isfinite(elapsed) or elapsed < last_elapsed or prior_active+elapsed >= 60:
            code = 'NCAAF_STAGED_DEADLINE'
        status = 'CAPTURED_UNREVIEWED' if code is None and len(objects) == len(p['requests']) else 'INCOMPLETE'
        append(journal, dict(kind='finish', status=status, reason=code, attempts=attempts, stage_active_seconds=elapsed))
    journal_raw = journal_path.read_bytes()
    body = dict(version=HISTORY_BUNDLE if p['version'] == HISTORY_PLAN else QUOTE_BUNDLE, evidence_label=p['evidence_label'],
        plan=deepcopy(plan), authorization=deepcopy(authorization), control_receipt=control, started_at=started_at, status=status, reason=code, attempts=attempts, objects=objects,
        original_bytes=total-prior_bytes, active_seconds=elapsed, prior_original_bytes=prior_bytes, prior_active_seconds=prior_active,
        journal_b64=base64.b64encode(journal_raw).decode(), journal_sha256=hashlib.sha256(journal_raw).hexdigest(),
        accepted=False, inference=False, scientific_acceptance=False, wagering_authority=False, wager_action='PASS', live_stake=0)
    bundle = seal(body); raw = model.encode(bundle)
    require(len(raw) <= MAX_BUNDLE, 'NCAAF_STAGED_BUNDLE_LIMIT')
    _exclusive(root/(plan['sha256']+'.stage.json'), raw)
    read_bundle(bundle, complete=False); verify_journal(root, bundle)
    return bundle


def _target(rows, t):
    candidates = [g for g in rows if g.get('id') == t['provider_event_id'] or (g.get('home_team') == t['home_team'] and g.get('away_team') == t['away_team'])]
    require(len(candidates) == 1 and candidates[0].get('id') == t['provider_event_id'], 'NCAAF_STAGED_TARGET_CONFLICT')
    g = candidates[0]
    require(g.get('sport_key') == 'americanfootball_ncaaf' and all(g.get(k) == t[k] for k in ('home_team', 'away_team'))
            and timestamp(g.get('commence_time')) == timestamp(t['start_utc']), 'NCAAF_STAGED_TARGET_CONFLICT')
    books = [b for b in g.get('bookmakers', []) if b.get('key') == 'draftkings']
    require(len(books) == 1 and len([m for m in books[0].get('markets', []) if m.get('key') == 'spreads']) == 1,
            'NCAAF_STAGED_TARGET_MARKET_MISSING')
    # Lines, prices, clocks and listing applicability remain unreviewed here.


SAFE_CODES = frozenset('NCAAF_STAGED_DEADLINE NCAAF_STAGED_COMBINED_LIMIT NCAAF_STAGED_HISTORY_ALREADY_CLAIMED NCAAF_STAGED_TARGET_CONFLICT NCAAF_STAGED_TARGET_MARKET_MISSING NCAAF_STAGED_AUTHORIZATION_CLOCK NCAAF_STAGED_PERMISSION_CLOCK NCAAF_PILOT_DEADLINE NCAAF_PILOT_BODY_LIMIT NCAAF_PILOT_INCOMPLETE_BODY NCAAF_PILOT_HTTP_STATUS NCAAF_PILOT_AUTH_OR_RATE_LIMIT NCAAF_PILOT_REDIRECT_OR_PAGINATION NCAAF_PILOT_CONTENT_TYPE NCAAF_PILOT_CONTENT_ENCODING NCAAF_PILOT_TRANSPORT_FAILURE NCAAF_PILOT_CREDENTIAL_IN_BODY NCAAF_CUSTODY_CREDENTIAL_FIELD NCAAF_CUSTODY_BODY_SCHEMA'.split())


def read_bundle(bundle, *, complete=True):
    b = _envelope(bundle)
    require(len(model.encode(bundle)) <= MAX_BUNDLE and set(b) == set('version evidence_label plan authorization control_receipt started_at status reason attempts objects original_bytes active_seconds prior_original_bytes prior_active_seconds journal_b64 journal_sha256 accepted inference scientific_acceptance wagering_authority wager_action live_stake'.split()), 'NCAAF_STAGED_BUNDLE_SCHEMA')
    p = validate_plan(b['plan'], executable=True)
    control = b['control_receipt']
    require(isinstance(control, dict) and set(control) == {'file_sha256', 'file_b64'} and isinstance(control['file_b64'], str)
            and len(control['file_b64']) <= ((128*1024+2)//3)*4, 'NCAAF_STAGED_CONFIGURATION_REQUIRED')
    c = configuration(b['plan'], base64.b64decode(control['file_b64'], validate=True), control['file_sha256'], at=b['started_at'])['payload']
    require(c['authorization'] == b['authorization'], 'NCAAF_STAGED_CONFIGURATION')
    require(b['version'] == (HISTORY_BUNDLE if p['version'] == HISTORY_PLAN else QUOTE_BUNDLE) and b['evidence_label'] == p['evidence_label']
            and all(b[k] is False for k in ('accepted', 'inference', 'scientific_acceptance', 'wagering_authority'))
            and b['wager_action'] == 'PASS' and b['live_stake'] == 0, 'NCAAF_STAGED_UNACCEPTED')
    require(type(b['attempts']) is int and 0 <= b['attempts'] <= len(p['requests']) and len(b['objects']) <= b['attempts'], 'NCAAF_STAGED_ATTEMPTS')
    if complete:
        require(b['status'] == 'CAPTURED_UNREVIEWED' and b['reason'] is None and len(b['objects']) == b['attempts'] == len(p['requests']), 'NCAAF_STAGED_INCOMPLETE')
    require(b['status'] in {'CAPTURED_UNREVIEWED', 'INCOMPLETE'} and b['reason'] in SAFE_CODES | {None, 'NCAAF_STAGED_ACQUISITION_FAILURE'}, 'NCAAF_STAGED_BUNDLE_SCHEMA')
    raw = base64.b64decode(b['journal_b64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == b['journal_sha256'], 'NCAAF_STAGED_JOURNAL')
    events = [json.loads(line) for line in raw.splitlines()]
    require(events[0] == dict(kind='start', plan_sha256=b['plan']['sha256'], authorization_sha256=b['authorization']['sha256'],
        started_at=b['started_at'], parent_history_sha256=p['parent_history_sha256'], control_receipt_sha256=model.digest(control), prior_original_bytes=b['prior_original_bytes'], prior_active_seconds=b['prior_active_seconds'])
        and events[-1] == dict(kind='finish', status=b['status'], reason=b['reason'], attempts=b['attempts'], stage_active_seconds=b['active_seconds']), 'NCAAF_STAGED_JOURNAL')
    attempts = [e for e in events if e.get('kind') == 'attempt']; done = [e for e in events if e.get('kind') == 'complete']
    require(len(events) == 2+len(attempts)+len(done) and len(attempts) == b['attempts'] and len(done) == len(b['objects']), 'NCAAF_STAGED_JOURNAL')
    ordered = [events[0]]
    for i, attempt in enumerate(attempts):
        ordered.append(attempt)
        if i < len(done): ordered.append(done[i])
    ordered.append(events[-1])
    require(events == ordered, 'NCAAF_STAGED_JOURNAL')
    elapsed = 0.; total = 0
    for i, (req, obj) in enumerate(zip(p['requests'], b['objects']), 1):
        require(set(obj) == set('request_id requested_at metadata body_b64 body_bytes body_sha256'.split()), 'NCAAF_STAGED_BUNDLE_SCHEMA')
        require(attempts[i-1] == dict(kind='attempt', attempt=i, request=req, requested_at=obj['requested_at']), 'NCAAF_STAGED_JOURNAL')
        authorize(b['plan'], b['authorization'], obj['requested_at'], trusted=False)
        custody._metadata(obj['metadata'])
        require(obj['metadata']['provider'] == req['provider'] and obj['metadata']['endpoint'] == req['endpoint']
                and obj['metadata']['request_scope'] == req['params'] and obj['request_id'] == req['id'], 'NCAAF_STAGED_REQUEST_CONFLICT')
        require(type(obj['body_bytes']) is int and 0 < obj['body_bytes'] <= LIMITS['max_body_bytes']
                and isinstance(obj['body_b64'], str) and len(obj['body_b64']) <= ((LIMITS['max_body_bytes']+2)//3)*4, 'NCAAF_STAGED_BODY')
        data = base64.b64decode(obj['body_b64'], validate=True); custody._parse(data)
        require(len(data) == obj['body_bytes'] and hashlib.sha256(data).hexdigest() == obj['body_sha256'], 'NCAAF_STAGED_BODY')
        received = obj['metadata']['received_at']; authorize(b['plan'], b['authorization'], received, trusted=False)
        require(timestamp(b['started_at']) <= timestamp(obj['requested_at']) <= timestamp(received), 'NCAAF_STAGED_CLOCK')
        e = done[i-1]
        require(set(e) == set('kind attempt body_sha256 body_bytes received_at stage_active_seconds'.split())
                and e['attempt'] == i and e['body_sha256'] == obj['body_sha256'] and e['body_bytes'] == len(data)
                and e['received_at'] == received and type(e['stage_active_seconds']) in (int, float)
                and math.isfinite(e['stage_active_seconds']) and elapsed <= e['stage_active_seconds'] <= b['active_seconds'], 'NCAAF_STAGED_JOURNAL')
        elapsed = e['stage_active_seconds']; total += len(data)
    require(total == b['original_bytes'] and all(type(b[k]) in (int, float) and math.isfinite(b[k]) and b[k] >= 0
            for k in ('active_seconds', 'prior_active_seconds', 'prior_original_bytes'))
            and b['prior_original_bytes']+total <= LIMITS['max_combined_original_bytes'], 'NCAAF_STAGED_COMBINED_LIMIT')
    if p['version'] == HISTORY_PLAN:
        require(b['prior_original_bytes'] == b['prior_active_seconds'] == 0, 'NCAAF_STAGED_PARENT')
    if complete:
        require(b['active_seconds']+b['prior_active_seconds'] < 60, 'NCAAF_STAGED_COMBINED_LIMIT')
        if p['version'] == QUOTE_PLAN: _target(custody._parse(base64.b64decode(b['objects'][0]['body_b64'])), p['target'])
    return deepcopy(b)


def verify_journal(root, bundle):
    b = _envelope(bundle); path = Path(root)/(b['plan']['sha256']+'.attempts.jsonl')
    require(path.read_bytes() == base64.b64decode(b['journal_b64'], validate=True), 'NCAAF_STAGED_JOURNAL')
    if b['plan']['payload']['version'] == QUOTE_PLAN:
        claim = Path(root)/(b['plan']['payload']['parent_history_sha256']+'.quote-claim.json')
        if b['reason'] == 'NCAAF_STAGED_HISTORY_ALREADY_CLAIMED':
            prior = json.loads(claim.read_bytes())
            require(set(prior) == {'plan_sha256'} and prior['plan_sha256'] != b['plan']['sha256'], 'NCAAF_STAGED_JOURNAL')
        else:
            require(claim.read_bytes() == model.encode(dict(plan_sha256=b['plan']['sha256'])), 'NCAAF_STAGED_JOURNAL')
    return True
