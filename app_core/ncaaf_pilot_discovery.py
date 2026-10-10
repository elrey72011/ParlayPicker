"""Versioned identifier discovery. No capture admission, inference or catalog writes.

Only the explicitly authorized one-request dispatch below can use transport.
Planning is network/credential/storage free. Production trust catalogs are empty.
"""
import base64
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import time

from app_core import ncaaf_pilot as pilot, ncaaf_response_custody as custody
from app_core.ncaaf_history import timestamp

VERSION = 'ncaaf-pilot-discovery-plan-v1'
BUNDLE_VERSION = 'ncaaf-pilot-unaccepted-identifiers-v1'
AUTH_VERSION = 'ncaaf-pilot-discovery-authorization-v1'
AUTHORIZED_DISCOVERIES = {}
ACCEPTED_DISCOVERY_PERMISSIONS = {}
AUTHORIZED_CUSTODY_ROOTS = {}
require = pilot.require
seal = pilot.seal

# Only these fixed diagnostic literals may leave the failure boundary. Exception
# messages (including apparent code prefixes) may contain provider credentials.
FAILURE_CODES = frozenset('''
NCAAF_DISCOVERY_AGGREGATE_LIMIT NCAAF_DISCOVERY_AUTHORIZATION_CONFLICT
NCAAF_DISCOVERY_AUTHORIZATION_EXPIRED NCAAF_DISCOVERY_AUTHORIZATION_MISSING
NCAAF_DISCOVERY_AUTHORIZATION_UNTRUSTED NCAAF_DISCOVERY_BODY_SCHEMA
NCAAF_DISCOVERY_CLOCK NCAAF_DISCOVERY_DEADLINE
NCAAF_DISCOVERY_PERMISSION_CLOCK NCAAF_DISCOVERY_PERMISSION_MISSING
NCAAF_DISCOVERY_PERMISSION_SCOPE NCAAF_DISCOVERY_PERMISSION_UNTRUSTED
NCAAF_DISCOVERY_RESPONSE_SCOPE NCAAF_DISCOVERY_STOPPED
NCAAF_DISCOVERY_TRANSPORT_FAILURE NCAAF_DISCOVERY_WINDOW_EXPIRED
NCAAF_PILOT_AUTH_OR_RATE_LIMIT NCAAF_PILOT_BODY_LIMIT
NCAAF_PILOT_CONTENT_ENCODING NCAAF_PILOT_CONTENT_TYPE
NCAAF_PILOT_CREDENTIAL_IN_BODY NCAAF_PILOT_CREDENTIAL_UNAVAILABLE
NCAAF_PILOT_DEADLINE NCAAF_PILOT_HTTP_STATUS NCAAF_PILOT_INCOMPLETE_BODY
NCAAF_PILOT_REDIRECT_OR_PAGINATION NCAAF_CUSTODY_BODY_SCHEMA
NCAAF_CUSTODY_CREDENTIAL_FIELD
'''.split())


def clock(value):
    require(isinstance(value, str) and re.search(r'(Z|[+-]\d\d:\d\d)$', value) is not None
            and timestamp(value) is not None, 'NCAAF_DISCOVERY_CLOCK')
    return timestamp(value)


def envelope(value, code):
    require(isinstance(value, dict) and set(value) == {'payload', 'sha256'}
            and isinstance(value['payload'], dict)
            and pilot.model.digest(value['payload']) == value['sha256'], code)
    custody._credentials(value)
    return value['payload']


def validate_plan(plan, *, executable=False):
    p = envelope(plan, 'NCAAF_DISCOVERY_PLAN_INTEGRITY')
    require(set(p) == set('version evidence_label target offer_scope requests limits execution_window advance_permission collection_authorization_ref custody_id unresolved'.split())
            and p['version'] == VERSION and p['evidence_label'] in {'SYNTHETIC', 'PROSPECTIVE', 'PROPOSAL'}, 'NCAAF_DISCOVERY_PLAN_SCHEMA')
    t = p['target']
    require(isinstance(t, dict) and set(t) == set('canonical_event_id home_team home_id away_team away_id start_utc neutral_site schedule_reference'.split())
            and all(isinstance(t[k], str) and t[k] for k in ('canonical_event_id', 'home_team', 'away_team', 'schedule_reference'))
            and t['canonical_event_id'].startswith('cfbd:') and t['home_team'] != t['away_team']
            and type(t['home_id']) is int and type(t['away_id']) is int and t['home_id'] != t['away_id']
            and type(t['neutral_site']) is bool, 'NCAAF_DISCOVERY_TARGET')
    start = clock(t['start_utc'])
    o = p['offer_scope']
    require(isinstance(o, dict) and set(o) == {'market', 'side', 'bookmaker'}
            and o['market'] == 'spreads' and o['side'] in {'home', 'away'}
            and isinstance(o['bookmaker'], str) and re.fullmatch('[a-z0-9_]+', o['bookmaker']) is not None
            and o['bookmaker'] != 'novig', 'NCAAF_DISCOVERY_SCOPE')
    require(isinstance(p['requests'], list) and len(p['requests']) == 1, 'NCAAF_DISCOVERY_BUDGET')
    r = p['requests'][0]
    require(isinstance(r, dict) and set(r) == {'id', 'provider', 'host', 'endpoint', 'params', 'max_attempts'}
            and isinstance(r['id'], str) and 0 < len(r['id']) <= 64 and r['max_attempts'] == 1
            and r['provider'] == 'odds_api' and r['host'] == pilot.HOSTS['odds_api']
            and r['endpoint'] == custody.ODDS_ENDPOINT, 'NCAAF_DISCOVERY_REQUEST')
    custody._metadata(dict(provider=r['provider'], endpoint=r['endpoint'], request_scope=r['params'], status=200,
        representation=custody.REPRESENTATION, content_type='application/json', complete=True,
        received_at='2000-01-01T00:00:00Z', receipt_clock_meaning=custody.RECEIPT_MEANING))
    q = r['params']
    require(q['regions'] == 'us' and q['markets'] == 'spreads' and q['bookmakers'] == o['bookmaker']
            and clock(q['commenceTimeFrom']) <= start < clock(q['commenceTimeTo'])
            and (clock(q['commenceTimeTo']) - clock(q['commenceTimeFrom'])).total_seconds() <= 86400,
            'NCAAF_DISCOVERY_SCOPE')
    limits = p['limits']
    require(isinstance(limits, dict) and set(limits) == set('max_attempts max_objects max_body_bytes max_aggregate_bytes connect_seconds read_seconds total_seconds'.split()), 'NCAAF_DISCOVERY_LIMITS')
    ceilings = dict(max_attempts=1, max_objects=custody.MAX_OBJECTS, max_body_bytes=custody.MAX_BODY_BYTES,
        max_aggregate_bytes=custody.MAX_TOTAL_BODY_BYTES, connect_seconds=3, read_seconds=5, total_seconds=60)
    require(all(type(limits[k]) in ((int,) if k.startswith('max_') else (int, float))
                and math.isfinite(limits[k]) and 0 < limits[k] <= v for k, v in ceilings.items())
            and limits['connect_seconds'] <= limits['total_seconds'] and limits['read_seconds'] <= limits['total_seconds'], 'NCAAF_DISCOVERY_LIMITS')
    w = p['execution_window']
    require(isinstance(w, dict) and set(w) == {'not_before', 'not_after', 'authorization_expires_at'}
            and clock(w['not_before']) < clock(w['not_after']) <= clock(w['authorization_expires_at'])
            and clock(w['not_after']) < start, 'NCAAF_DISCOVERY_WINDOW')
    require(isinstance(p['unresolved'], list) and isinstance(p['custody_id'], str) and p['custody_id'], 'NCAAF_DISCOVERY_PLAN_SCHEMA')
    if executable:
        require(not p['unresolved'] and p['evidence_label'] != 'PROPOSAL', 'NCAAF_DISCOVERY_PREREQUISITES')
    return deepcopy(p)


def planning(plan):
    p = validate_plan(plan)
    return dict(version=VERSION, plan_sha256=plan['sha256'], status='BLOCKED' if p['unresolved'] or p['evidence_label'] == 'PROPOSAL' else 'AWAITING_SEPARATE_DISCOVERY_AUTHORIZATION',
        requests=1, max_attempts=1, unresolved=p['unresolved'], network_requests=0, credential_loads=0,
        storage_operations=0, accepted=False, inference=False, wagering_authority=False)


def authorize(plan, authorization, at):
    p = validate_plan(plan, executable=True)
    now = clock(at); w = p['execution_window']
    require(clock(w['not_before']) <= now <= clock(w['not_after']) and now < clock(w['authorization_expires_at']), 'NCAAF_DISCOVERY_WINDOW_EXPIRED')
    a = envelope(authorization, 'NCAAF_DISCOVERY_AUTHORIZATION_MISSING')
    require(set(a) == set('version reference plan_sha256 custody_id authorized_at expires_at owner max_attempts max_provider_credits max_cost_usd quota_reference'.split())
            and a['version'] == AUTH_VERSION and a['reference'] == p['collection_authorization_ref']
            and a['plan_sha256'] == plan['sha256'] and a['custody_id'] == p['custody_id']
            and isinstance(a['owner'], str) and a['owner'] and isinstance(a['quota_reference'], str) and a['quota_reference']
            and type(a['max_attempts']) is int and a['max_attempts'] == 1
            and type(a['max_provider_credits']) is int and a['max_provider_credits'] == 1
            and type(a['max_cost_usd']) in (int, float) and math.isfinite(a['max_cost_usd']) and a['max_cost_usd'] >= 0,
            'NCAAF_DISCOVERY_AUTHORIZATION_CONFLICT')
    require(AUTHORIZED_DISCOVERIES.get(a['reference']) == authorization['sha256'], 'NCAAF_DISCOVERY_AUTHORIZATION_UNTRUSTED')
    require(clock(a['authorized_at']) <= now < clock(a['expires_at']) <= clock(w['authorization_expires_at']), 'NCAAF_DISCOVERY_AUTHORIZATION_EXPIRED')
    receipt = p['advance_permission']
    require(isinstance(receipt, dict) and set(receipt) == set('reference provider reviewer license_holder evidence_reference reviewed_at effective_from effective_until permitted_use requests_sha256 target_sha256'.split()), 'NCAAF_DISCOVERY_PERMISSION_MISSING')
    require(all(isinstance(receipt[k], str) and receipt[k] for k in ('reference', 'reviewer', 'license_holder', 'evidence_reference'))
            and ACCEPTED_DISCOVERY_PERMISSIONS.get(receipt['reference']) == pilot.model.digest(receipt), 'NCAAF_DISCOVERY_PERMISSION_UNTRUSTED')
    require(receipt['provider'] == 'odds_api' and receipt['permitted_use'] == 'private_identifier_discovery_retention'
            and receipt['requests_sha256'] == pilot.model.digest(p['requests'])
            and receipt['target_sha256'] == pilot.model.digest(p['target']), 'NCAAF_DISCOVERY_PERMISSION_SCOPE')
    require(clock(receipt['reviewed_at']) < now and clock(receipt['effective_from']) <= now
            and clock(w['authorization_expires_at']) <= clock(receipt['effective_until']), 'NCAAF_DISCOVERY_PERMISSION_CLOCK')
    return p


def identifiers(raw, p):
    """Exact literal descriptors only. No alias expansion or accepted crosswalk.

    Preserve indices/labels and conflicting/duplicate matches. Nothing here
    establishes neutral-site, listing, product, terms or public-output rights.
    """
    rows = custody._parse(raw); t = p['target']; candidates = []; conflicts = []
    all_ids = {}
    for i, row in enumerate(rows):
        eid = row.get('id')
        require(isinstance(eid, str) and eid and len(eid) <= 256
                and isinstance(row.get('home_team'), str) and isinstance(row.get('away_team'), str)
                and row.get('sport_key') == 'americanfootball_ncaaf', 'NCAAF_DISCOVERY_BODY_SCHEMA')
        start = clock(row.get('commence_time'))
        require(clock(p['requests'][0]['params']['commenceTimeFrom']) <= start < clock(p['requests'][0]['params']['commenceTimeTo']), 'NCAAF_DISCOVERY_RESPONSE_SCOPE')
        identity = (row['home_team'], row['away_team'], start)
        identity_conflict = eid in all_ids and all_ids[eid] != identity
        all_ids[eid] = identity
        same = row['home_team'] == t['home_team'] and row['away_team'] == t['away_team']
        reversed_ = row['home_team'] == t['away_team'] and row['away_team'] == t['home_team']
        item = dict(record_index=i, provider_event_id=eid, home_team=row['home_team'], away_team=row['away_team'],
                    start_utc=row['commence_time'], accepted=False, listing_id=None, product=None,
                    listing_status='NOT_VERIFIED', neutral_site_status='NOT_VERIFIED')
        if identity_conflict:
            conflicts.append(item)
        if same and start == clock(t['start_utc']):
            candidates.append(item)
        elif same or reversed_:
            conflicts.append(item)
    state = 'CONFLICTING' if conflicts else 'AMBIGUOUS' if len(candidates) > 1 else 'CANDIDATE_UNVERIFIED' if candidates else 'NO_EXACT_DESCRIPTOR_MATCH'
    return dict(state=state, candidates=candidates, conflicts=conflicts, records=len(rows),
                canonical_event_id=t['canonical_event_id'], accepted=False)


def acquire(plan, authorization, *, root, transport, wall=pilot.utc, monotonic=time.monotonic):
    """Explicit one spent attempt; no resume, new storage or accepted output."""
    p = authorize(plan, authorization, wall())
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'NCAAF_DISCOVERY_CUSTODY_REQUIRED')
    configured = AUTHORIZED_CUSTODY_ROOTS.get(p['custody_id'])
    require(configured is not None and Path(configured).resolve() == root.resolve(), 'NCAAF_DISCOVERY_CUSTODY_CONFLICT')
    journal_path = root / (plan['sha256'] + '.discovery-attempts.jsonl')
    try:
        journal = journal_path.open('xb')
    except FileExistsError:
        raise ValueError('NCAAF_DISCOVERY_ALREADY_ATTEMPTED_NO_RESUME') from None
    deadline = monotonic() + p['limits']['total_seconds']; objects = []; code = None; attempted = 0; result = None
    try:
        with journal:
            pilot._append(journal, dict(version=VERSION, plan_sha256=plan['sha256'], authorization_sha256=authorization['sha256'], custody_id=p['custody_id']))
            started = wall(); authorize(plan, authorization, started)
            require(monotonic() < deadline, 'NCAAF_DISCOVERY_DEADLINE')
            request = p['requests'][0]; attempted = 1
            pilot._append(journal, dict(request=request, attempt=1, requested_at=started, request_clock_meaning='actual_local_request_start'))
            stream = None
            try:
                stream = transport(deepcopy(request), deepcopy(p['limits']), deadline, monotonic)
                raw = pilot._read(stream, p['limits']['max_body_bytes'], deadline, monotonic)
                if isinstance(transport, pilot.HttpsTransport):
                    transport.validate_body(raw)
            except ValueError:
                raise
            except Exception:
                raise ValueError('NCAAF_DISCOVERY_TRANSPORT_FAILURE') from None
            finally:
                if stream is not None:
                    stream.close()
            received = wall(); authorize(plan, authorization, received)
            require(clock(received) >= clock(started), 'NCAAF_DISCOVERY_CLOCK')
            require(len(raw) <= p['limits']['max_aggregate_bytes'], 'NCAAF_DISCOVERY_AGGREGATE_LIMIT')
            # Parse/validate before persistence, but keep complete original body
            # even when its target descriptors are absent/ambiguous/conflicting.
            result = identifiers(raw, p)
            meta = dict(provider=request['provider'], endpoint=request['endpoint'], request_scope=request['params'], status=200,
                representation=custody.REPRESENTATION, content_type='application/json', complete=True,
                received_at=received, receipt_clock_meaning=custody.RECEIPT_MEANING)
            objects.append(dict(request_id=request['id'], requested_at=started, request_clock_meaning='actual_local_request_start',
                metadata=meta, body_b64=base64.b64encode(raw).decode(), body_bytes=len(raw), body_sha256=hashlib.sha256(raw).hexdigest()))
            pilot._append(journal, dict(received_at=received, body_sha256=objects[0]['body_sha256'], identifier_status=result['state']))
    except Exception as exc:
        reason = str(exc)
        code = reason if reason in FAILURE_CODES else 'NCAAF_DISCOVERY_STOPPED'
    bundle = seal(dict(version=BUNDLE_VERSION, evidence_label=p['evidence_label'], plan=deepcopy(plan), owner_authorization_receipt=deepcopy(authorization),
        status='INCOMPLETE' if code else 'DISCOVERED_UNACCEPTED', reason=code, attempted_requests=attempted,
        objects=objects, identifiers=result, accepted=False, inference=False, scientific_acceptance=False, wagering_authority=False))
    path = root / (plan['sha256'] + '.discovery-unaccepted.json')
    with path.open('xb') as f:
        f.write(pilot.encode(bundle)); f.flush(); pilot.os.fsync(f.fileno())
    require(path.read_bytes() == pilot.encode(bundle), 'NCAAF_DISCOVERY_PRIVATE_READBACK')
    return bundle


def read_bundle(path):
    """Private static read-back only. Explicit version prevents capture reuse."""
    import json
    raw = Path(path).read_bytes()
    require(len(raw) <= 3 * custody.MAX_TOTAL_BODY_BYTES, 'NCAAF_DISCOVERY_BUNDLE_LIMIT')
    b = json.loads(raw); p = envelope(b, 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
    require(p['version'] == BUNDLE_VERSION and p['accepted'] is False and p['inference'] is False
            and p['scientific_acceptance'] is False and p['wagering_authority'] is False, 'NCAAF_DISCOVERY_BUNDLE_SCOPE')
    plan = validate_plan(p['plan'], executable=True)
    require(type(p['attempted_requests']) is int and 0 <= p['attempted_requests'] <= 1
            and len(p['objects']) <= p['attempted_requests'], 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
    if p['status'] == 'DISCOVERED_UNACCEPTED':
        require(p['attempted_requests'] == len(p['objects']) == 1 and p['reason'] is None, 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
        obj = p['objects'][0]; req = plan['requests'][0]
        body = base64.b64decode(obj['body_b64'], validate=True)
        require(len(body) == obj['body_bytes'] <= plan['limits']['max_body_bytes']
                and hashlib.sha256(body).hexdigest() == obj['body_sha256']
                and obj['request_id'] == req['id'] and obj['metadata']['request_scope'] == req['params']
                and obj['metadata']['provider'] == req['provider'] and obj['metadata']['endpoint'] == req['endpoint'], 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
        custody._metadata(obj['metadata'])
        require(obj['request_clock_meaning'] == 'actual_local_request_start'
                and clock(obj['requested_at']) <= clock(obj['metadata']['received_at'])
                and identifiers(body, plan) == p['identifiers'], 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
        authorize(p['plan'], p['owner_authorization_receipt'], obj['requested_at'])
        authorize(p['plan'], p['owner_authorization_receipt'], obj['metadata']['received_at'])
    else:
        require(p['status'] == 'INCOMPLETE' and p['reason'] in FAILURE_CODES
                and not p['objects'] and p['identifiers'] is None, 'NCAAF_DISCOVERY_BUNDLE_INTEGRITY')
    return b
