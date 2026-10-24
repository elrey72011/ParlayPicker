"""Explicit filtered-history continuation; never implicit acquisition or admission.

No transport, authorization, filesystem write, model call or acceptance catalog.
The integrating reader must authenticate the retained diagnostic chain first.
Derived weekly views are not descriptions of the original HTTP requests.
"""
from copy import deepcopy
from decimal import Decimal
from datetime import timedelta
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
from app_core import ncaaf_history as history, ncaaf_response_custody as custody

VERSION = 'ncaaf-filtered-history-continuation-v1'
PROJECTION = 'ncaaf-exact-selected-game-projection-v2'
HOST = 'api.collegefootballdata.com'
MAX_BODY = 524288
MAX_ORIGINALS = 2097152
MAX_BUNDLE = 6291456
MAX_ACTIVE = Decimal('60')


def require(ok, code):
    if not ok:
        raise ValueError(code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_rows(raw):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BODY, 'FILTERED_BODY_LIMIT')
    rows = custody._parse(raw)
    require(isinstance(rows, list) and all(isinstance(r, dict) and type(r.get('id')) is int for r in rows), 'FILTERED_BODY_SCHEMA')
    require(len({r['id'] for r in rows}) == len(rows), 'FILTERED_DUPLICATE_GAME')
    return rows


def asset(raw, actual_request, requested_at, received_at):
    """Representation only; authentication and review are separate requirements."""
    checked_rows(raw)
    require(history.timestamp(requested_at) is not None and history.timestamp(received_at) is not None
            and history.timestamp(requested_at) <= history.timestamp(received_at), 'FILTERED_CLOCK')
    return dict(raw=raw, body_sha256=sha(raw), body_bytes=len(raw), actual_request=deepcopy(actual_request),
                requested_at=requested_at, received_at=received_at)


def validate_contract(c):
    require(set(c) == {'version', 'season', 'target', 'scoring_request', 'scoring_sha256',
            'expected_scoring', 'selected_yardage', 'weeks', 'venue_bridge_subject_sha256'}, 'FILTERED_CONTRACT_SCHEMA')
    require(c['version'] in {VERSION, 'ncaaf-fresh-selected-game-contract-v1'} and type(c['season']) is int, 'FILTERED_CONTRACT_SCHEMA')
    t = c['target']
    require(set(t) == {'id', 'homeId', 'homeTeam', 'awayId', 'awayTeam', 'startDate', 'neutralSite', 'venue', 'venueId'}, 'FILTERED_TARGET')
    require(all(type(t[k]) is int and t[k] > 0 for k in ('id', 'homeId', 'awayId', 'venueId'))
            and t['homeId'] != t['awayId'] and t['homeTeam'] != t['awayTeam']
            and type(t['neutralSite']) is bool and history.timestamp(t['startDate']) is not None, 'FILTERED_TARGET')
    require(c['scoring_request'] == dict(method='GET', host=HOST, endpoint='games', params=dict(
            year=c['season'], seasonType='both', classification='fbs', conference=('CUSA' if c['version']==VERSION else c['scoring_request'].get('params',{}).get('conference')))), 'FILTERED_SCOPE')
    if c['version']!=VERSION:
        import re
        require(isinstance(c['scoring_request']['params']['conference'],str) and re.fullmatch(r'[A-Z][A-Z0-9]{1,9}',c['scoring_request']['params']['conference']), 'FILTERED_SCOPE')
    require(len(c['weeks']) == 3 and len(set(c['weeks'])) == 3 and all(type(w) is int and w > 0 for w in c['weeks']), 'FILTERED_WEEKS')
    for key in ('expected_scoring', 'selected_yardage'):
        require(set(c[key]) == {t['homeTeam'], t['awayTeam']}, 'FILTERED_SELECTION')
        for name, values in c[key].items():
            require(len(values) >= 3 and len({v['id'] for v in values}) == len(values)
                    and all(set(v) == {'id', 'week', 'startDate', 'opponent_id', 'opponent'} for v in values), 'FILTERED_SELECTION')
    for name, values in c['selected_yardage'].items():
        require(len(values) == 3 and {v['week'] for v in values} == set(c['weeks'])
                and all(v in c['expected_scoring'][name] for v in values), 'FILTERED_SELECTION')
    return deepcopy(c)


def requests(c, mode='team_season'):
    validate_contract(c)
    common = dict(year=c['season'], seasonType='regular')
    params = ([dict(common, team=c['target'][side + 'Team']) for side in ('home', 'away')]
              if mode == 'team_season' else [dict(common, week=w, conference='CUSA') for w in c['weeks']])
    require(mode in {'team_season', 'conference_weekly'}, 'FILTERED_SCOPE')
    return [dict(method='GET', host=HOST, endpoint='games/teams', params=p) for p in params]


def verify_scoring(original, c, *, at):
    validate_contract(c)
    require(original['actual_request'] == c['scoring_request'] and sha(original['raw']) == original['body_sha256'] == c['scoring_sha256']
            and len(original['raw']) == original['body_bytes'], 'FILTERED_ORIGINAL_BINDING')
    now, receipt, start = (history.timestamp(v) for v in (at, original['received_at'], c['target']['startDate']))
    require(now is not None and receipt is not None and receipt <= now < start and (now - receipt).total_seconds() <= 86400, 'FILTERED_INPUT_STALE_OR_STARTED')
    games = checked_rows(original['raw'])
    candidates = [g for g in games if g.get('id') == c['target']['id'] or
                  {g.get('homeTeam'), g.get('awayTeam')} == {c['target']['homeTeam'], c['target']['awayTeam']}]
    require(len(candidates) == 1 and all(candidates[0].get(k) == v for k, v in c['target'].items())
            and candidates[0].get('completed') is False and candidates[0].get('startTimeTBD') is False, 'FILTERED_EVENT_CONFLICT')
    cutoff = start - timedelta(days=7)
    for side in ('home', 'away'):
        tid, name = c['target'][side + 'Id'], c['target'][side + 'Team']
        actual = []
        for g in games:
            participating = tid in (g.get('homeId'), g.get('awayId'))
            named = name in (g.get('homeTeam'), g.get('awayTeam'))
            require(participating == named, 'FILTERED_TEAM_CONFLICT')
            if not participating or g['id'] == c['target']['id']:
                continue
            instant = history.timestamp(g.get('startDate'))
            require(instant is not None, 'FILTERED_CLOCK')
            if instant >= cutoff:
                continue
            require(history._valid_game(g, c['season']), 'FILTERED_INCOMPLETE_HISTORY')
            own = 'home' if g['homeId'] == tid else 'away'
            other = 'away' if own == 'home' else 'home'
            actual.append(dict(id=g['id'], week=g['week'], startDate=g['startDate'], opponent_id=g[other + 'Id'], opponent=g[other + 'Team']))
        require(sorted(actual, key=lambda r:r['id']) == sorted(c['expected_scoring'][name], key=lambda r:r['id']), 'FILTERED_SCORING_COVERAGE')
    return games


def venue_subject(c, canonical_label, source_references):
    validate_contract(c)
    require(isinstance(canonical_label, str) and bool(canonical_label.strip()) and source_references, 'FILTERED_VENUE')
    return dict(version='ncaaf-exact-venue-correspondence-v1', event_id=c['target']['id'], venue_id=c['target']['venueId'],
                original_label=c['target']['venue'], reviewed_label=canonical_label,
                original_sha256=c['scoring_sha256'], sources=deepcopy(source_references))


def require_reuse_review(review, c, *, reuse_subject_sha256, original_receipt, at):
    """The integrating reader computes the subject from the verified full chain.

    New consent cannot masquerade as original collection authority. That earlier
    authority is authenticated separately, before this reuse review is checked.
    """
    require(set(review) == {'version', 'reviewer', 'reviewed_at', 'reuse_subject_sha256', 'venue_subject_sha256', 'answer'}, 'FILTERED_REUSE_REVIEW')
    reviewed = history.timestamp(review['reviewed_at'])
    require(review['version'] == 'ncaaf-diagnostic-original-reuse-review-v1' and review['reviewer'] == 'Robert Velarde'
            and review['reuse_subject_sha256'] == reuse_subject_sha256
            and review['venue_subject_sha256'] == c['venue_bridge_subject_sha256']
            and isinstance(review['answer'], str) and bool(review['answer'].strip())
            and reviewed is not None and history.timestamp(original_receipt) <= reviewed <= history.timestamp(at), 'FILTERED_REUSE_REVIEW')


def read_scoring_parent(blobs, pins, c, *, at):
    """Versioned read-only import; never grants reuse or changes original status.

    Pins belong to the exact separately reviewed reuse subject. Hashing a new
    caller-supplied set of files is not a substitute for that review/authority.
    The old method, control and verbatim approval remain part of the parent.
    """
    require(c.get('version')==VERSION,'FILTERED_CONTRACT_SCHEMA')
    keys = {'original', 'bundle', 'diagnostics', 'journal', 'control', 'method', 'approval', 'claim', 'report'}
    require(set(blobs) == set(pins) == keys and all(isinstance(raw,bytes) and sha(raw)==pins[k] for k,raw in blobs.items()), 'FILTERED_PARENT_HASH')
    require(len(blobs['original']) <= MAX_BODY and len(blobs['bundle']) + len(blobs['diagnostics']) <= MAX_BUNDLE, 'FILTERED_BODY_LIMIT')
    def parse(raw):
        return json.loads(raw, object_pairs_hook=custody._unique, parse_constant=custody._constant)
    def envelope(raw):
        value=parse(raw)
        require(set(value)=={'payload','sha256'} and digest(value['payload'])==value['sha256'], 'FILTERED_PARENT_INTEGRITY')
        return value['payload']
    b,d,control=(envelope(blobs[k]) for k in ('bundle','diagnostics','control'))
    method,approval,claim,report=(parse(blobs[k]) for k in ('method','approval','claim','report'))
    require(b['version']=='cfbd-cusa-diagnostic-only-v1' and control['version']=='cfbd-cusa-reviewed-diagnostic-control-v1'
            and d['version']=='cfbd-cusa-diagnostic-observations-v1', 'FILTERED_PARENT_VERSION')
    require(b['status']=='DIAGNOSTIC_STOPPED' and b['reason']=='PROBE_HEADER_PRODUCTION_LIMIT'
            and b['attempts']==len(b['objects'])==len(d['attempts'])==1 and b['acceptance'] is b['inference'] is False, 'FILTERED_PARENT_OUTCOME')
    require(b['method_file_sha256']==control['method_file_sha256']==d['method_file_sha256']==claim['method_file_sha256']==report['method_file_sha256']==pins['method']
            and b['control_file_sha256']==d['control_file_sha256']==claim['control_file_sha256']==report['control_file_sha256']==pins['control'], 'FILTERED_PARENT_LINK')
    require(d['bundle_sha256']==parse(blobs['bundle'])['sha256'] and d['bundle_file_sha256']==report['bundle_file_sha256']==pins['bundle']
            and d['journal_sha256']==report['journal_sha256']==pins['journal'] and report['diagnostics_file_sha256']==pins['diagnostics'], 'FILTERED_PARENT_LINK')
    require(control['execution_approval_file_sha256']==pins['approval'] and approval['method_file_sha256']==pins['method']
            and approval['decision']=='APPROVE_EXACT_DIAGNOSTIC_METHOD_AND_MANUAL_EXECUTION'
            and approval['owner']==control['authorization']['owner']=='Robert Velarde'
            and approval['received_at']==control['authorization']['authorized_at']==control['review']['reviewed_at']
            and isinstance(approval['answer_verbatim'],str) and approval['answer_verbatim'].strip(), 'FILTERED_PARENT_AUTHORITY')
    o=b['objects'][0];a=d['attempts'][0]
    require(base64.b64decode(o['body_b64'],validate=True)==blobs['original'] and o['body_sha256']==pins['original']
            and o['body_bytes']==len(blobs['original']) and o['actual_request']==c['scoring_request']==method['requests'][0]==control['requests'][0]
            and o['status']==a['http_status']==200 and o['complete'] is a['complete_original'] is True, 'FILTERED_PARENT_ORIGINAL')
    header=[p['seconds'] for p in a['phases'] if p['phase']=='response_headers']
    require(len(header)==1 and 10 < header[0] <= 30 and b['active_seconds']==d['active_seconds']==report['active_seconds']
            and type(b['active_seconds']) in (int,float) and header[0]<=b['active_seconds']<60
            and report['provider_attempts']==1 and report['production_admitted'] is report['inference'] is report['retry'] is False, 'FILTERED_PARENT_TIMING')
    requested,receipt,authorized,begin,end,expiry=(history.timestamp(v) for v in (o['requested_at'],o['received_at'],approval['received_at'],
            control['window']['not_before'],control['window']['not_after'],control['authorization']['expires_at']))
    require(all((requested,receipt,authorized,begin,end,expiry)) and authorized < begin <= requested <= receipt < end <= expiry, 'FILTERED_PARENT_CLOCK')
    journal=[parse(line) for line in blobs['journal'].splitlines()]
    require([r['kind'] for r in journal]==['start','attempt','receipt','finish']
            and journal[0]['method_file_sha256']==pins['method'] and journal[0]['control_file_sha256']==pins['control']
            and journal[1]['attempt']==journal[2]['attempt']==1 and journal[1]['actual_request']==c['scoring_request']
            and journal[1]['requested_at']==o['requested_at'] and journal[2]['received_at']==o['received_at']
            and journal[2]['body_sha256']==pins['original'] and journal[3]['attempts']==1
            and journal[3]['active_seconds']==b['active_seconds'] and journal[3]['reason']==b['reason'], 'FILTERED_PARENT_JOURNAL')
    original=asset(blobs['original'],o['actual_request'],o['requested_at'],o['received_at'])
    verify_scoring(original,c,at=at)
    return original,dict(version='ncaaf-retained-diagnostic-scoring-parent-v1',artifact_pins=deepcopy(pins),
        original_received_at=o['received_at'],original_requested_at=o['requested_at'],original_bytes=o['body_bytes'],
        charged_active_seconds=b['active_seconds'],charged_attempts=1,header_seconds=header[0],
        original_outcome=b['status'],original_reason=b['reason'],original_production_timing_suitable=False,
        provider_receipt='UNKNOWN',credit_consumption='UNKNOWN',reuse_authorization=None,admitted=False)


def project(scoring, statistics, c, *, at, mode='team_season'):
    """Read/projection validation only. Does not grant admission or run a model."""
    games = verify_scoring(scoring, c, at=at)
    wanted_requests = requests(c, mode)
    require(len(statistics) == len(wanted_requests) and [s['actual_request'] for s in statistics] == wanted_requests, 'FILTERED_SCOPE')
    originals = [scoring] + statistics
    require(sum(s['body_bytes'] for s in originals) <= MAX_ORIGINALS, 'FILTERED_AGGREGATE_LIMIT')
    selected = {v['id'] for values in c['selected_yardage'].values() for v in values}
    lookup = {g['id']:g for g in games}
    found, locators = {}, {}
    for oi, s in enumerate(statistics, 1):
        require(sha(s['raw']) == s['body_sha256'] and len(s['raw']) == s['body_bytes'], 'FILTERED_ORIGINAL_BINDING')
        now, receipt, requested = (history.timestamp(v) for v in (at, s['received_at'], s['requested_at']))
        require(all((now, receipt, requested)) and requested <= receipt <= now and (now-receipt).total_seconds() <= 86400, 'FILTERED_CLOCK')
        q = s['actual_request']['params']
        expected = ({v['id'] for v in c['selected_yardage'][q['team']]} if mode == 'team_season'
                    else {gid for gid in selected if lookup[gid]['week'] == q['week']})
        present = set()
        for ri, row in enumerate(checked_rows(s['raw'])):
            if row['id'] not in selected:
                continue  # Originals remain complete; no extra season statistics enter features.
            g = lookup[row['id']]
            require(row['id'] in expected, 'FILTERED_RESPONSE_SCOPE')
            require(all(row[k] == v for k,v in (('season',c['season']), ('week',g['week']), ('seasonType','regular')) if k in row), 'FILTERED_PERIOD_CONFLICT')
            teams = row.get('teams', [])
            require(len(teams) == 2 and {t.get('teamId') for t in teams} == {g['homeId'],g['awayId']}, 'FILTERED_STATS_TEAM')
            for team in teams:
                side = 'home' if team['teamId'] == g['homeId'] else 'away'
                require(team.get('team') == g[side+'Team'] and team.get('points') == g[side+'Points'], 'FILTERED_STATS_IDENTITY_SCORE')
                require(history._yards(team) is not None, 'FILTERED_YARDAGE_MISSING')
            require(row['id'] not in found or row == found[row['id']], 'FILTERED_STATS_CONFLICT')
            found[row['id']] = deepcopy(row)
            locators.setdefault(str(row['id']), []).append(dict(original_index=oi, record_index=ri, body_sha256=s['body_sha256']))
            present.add(row['id'])
        require(present == expected, 'FILTERED_YARDAGE_COVERAGE')
    require(set(found) == selected, 'FILTERED_YARDAGE_COVERAGE')
    batches = [dict(request=dict(kind='games', year=c['season']), retrieved_at=scoring['received_at'], records=history._clean('games', games))]
    for week in c['weeks']:
        ids = sorted(gid for gid in found if lookup[gid]['week'] == week)
        source_receipts = [statistics[p['original_index']-1]['received_at'] for gid in ids for p in locators[str(gid)]]
        batches.append(dict(request=dict(kind='stats', year=c['season'], week=week, season_type='regular'),
                            retrieved_at=max(source_receipts, key=history.timestamp), records=history._clean('stats', [found[gid] for gid in ids])))
    projection = dict(version=PROJECTION, contract_sha256=digest(c), representation='DERIVED_WEEKLY_VIEW_NOT_HTTP_REQUEST',
                      originals=[{k:deepcopy(v) for k,v in o.items() if k!='raw'} for o in originals],
                      selected_games=sorted(selected), locators=locators, batch_sha256=[digest(b) for b in batches])
    return dict(schema=1, years=[c['season']], batches=batches), projection


def remaining(prior_active, added_active=0, *, prior_bytes=62768, added_bytes=0, reserve_seconds=0):
    values = [Decimal(str(v)) for v in (prior_active, added_active, reserve_seconds)]
    require(all(v.is_finite() and v >= 0 for v in values), 'FILTERED_BUDGET')
    require(type(prior_bytes) is int and type(added_bytes) is int and min(prior_bytes,added_bytes) >= 0
            and prior_bytes + added_bytes <= MAX_ORIGINALS, 'FILTERED_AGGREGATE_LIMIT')
    available = MAX_ACTIVE - sum(values)
    require(available > 0, 'FILTERED_SHARED_DEADLINE')
    return available


def header_deadline(now, shared_deadline, allowance):
    require(type(allowance) in (int,float) and 0 < allowance <= 30, 'FILTERED_HEADER_POLICY')
    require(now < shared_deadline, 'FILTERED_SHARED_DEADLINE')
    return min(shared_deadline, now + allowance)


def continuation_budget(parent, expected_parent_sha256, *, added_active=0, added_bytes=0, reserve_seconds=20):
    """The expected subject must come from the authenticated reuse grant/plan."""
    require(digest(parent)==expected_parent_sha256 and parent['version']=='ncaaf-retained-diagnostic-scoring-parent-v1'
            and parent['original_production_timing_suitable'] is False and parent['charged_attempts']==1, 'FILTERED_PARENT_BINDING')
    return remaining(parent['charged_active_seconds'],added_active,prior_bytes=parent['original_bytes'],
                     added_bytes=added_bytes,reserve_seconds=reserve_seconds)


class RequestPolicy:
    """Exact opt-in transport scope, not collection or reuse authorization.

    Native authorization, exclusive spent claims and runtime authentication must
    still execute before this policy is used by the versioned continuation CLI.
    """
    def __init__(self, contract):
        if contract.get('version')=='ncaaf-fresh-filtered-source-v1':
            from app_core import ncaaf_fresh_history as fresh
            self._contract = fresh.validate_source(contract)
        else:
            require(contract.get('version')==VERSION,'FILTERED_CONTRACT_SCHEMA')
            self._contract = validate_contract(contract)
        self._subject = digest(self._contract)

    def check(self, request, limits):
        require(digest(self._contract) == self._subject, 'FILTERED_CONTRACT_MUTATION')
        if self._contract.get('version')=='ncaaf-fresh-filtered-source-v1':
            from app_core import ncaaf_fresh_history as fresh
            require(request in fresh.requests(self._contract) and limits['connect_seconds']==limits['read_seconds']==10,'FILTERED_SCOPE')
            return
        validate_contract(self._contract)
        expected = [dict(id='filtered-stats-'+str(i+1), provider='cfbd', host=r['host'], endpoint=r['endpoint'],
                         params=r['params'], max_attempts=1) for i,r in enumerate(requests(self._contract))]
        require(request in expected, 'FILTERED_SCOPE')
        require(limits['connect_seconds'] == limits['read_seconds'] == 10, 'FILTERED_TRANSPORT_LIMITS')

    def headers(self, now, deadline):
        require(digest(self._contract) == self._subject, 'FILTERED_CONTRACT_MUTATION')
        return header_deadline(now, deadline, 30)


HISTORY_PLAN = 'ncaaf-filtered-history-stage-plan-v4'
QUOTE_PLAN = 'ncaaf-filtered-final-quote-stage-plan-v4'
HISTORY_BUNDLE = 'ncaaf-filtered-history-unaccepted-v4'
QUOTE_BUNDLE = 'ncaaf-filtered-final-quote-unaccepted-v4'
PREQUOTE = 'ncaaf-filtered-prequote-features-v4'
BINDING = 'ncaaf-filtered-staged-evidence-v4'
OBJECT = 'ncaaf-filtered-original-response-v2'
POLICY = dict(header_ceiling_seconds=30,new_yardage_active_seconds=27.406,market_reserve_seconds=20)


def material(p):
    context=p['filtered_history'];value=context['parent']
    require(set(context)=={'contract','parent','reuse_subject','reuse_subject_sha256','reuse_review','venue_subject','budget'},'FILTERED_CONTEXT')
    require(set(value)=={'blobs_b64','pins','custody_names'} and set(value['blobs_b64'])==set(value['pins']), 'FILTERED_PARENT_SCHEMA')
    require(all(isinstance(v,str) and len(v)<=MAX_BUNDLE*4//3+4 for v in value['blobs_b64'].values()),'FILTERED_BODY_LIMIT')
    blobs={k:base64.b64decode(v,validate=True) for k,v in value['blobs_b64'].items()}
    require(sum(len(v) for v in blobs.values())<=MAX_BUNDLE,'FILTERED_BODY_LIMIT')
    return blobs,value['pins']


def parent(p,*,at):
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import parent as selected
        return selected(p,at=at)
    if is_fresh(p):return None,dict(original_bytes=0,charged_active_seconds=0)
    context=p['filtered_history'];blobs,pins=material(p);c=context['contract']
    original,facts=read_scoring_parent(blobs,pins,c,at=at)
    reuse=context['reuse_subject']
    require(digest(reuse)==context['reuse_subject_sha256'] and reuse['version']=='ncaaf-diagnostic-scoring-reuse-subject-v1'
        and reuse['original_sha256']==pins['original']==c['scoring_sha256']
        and reuse['method_file_sha256']==pins['method'] and reuse['control_file_sha256']==pins['control']
        and reuse['approval_file_sha256']==pins['approval']
        and reuse['artifact_pins']=={('original1' if k=='original' else k):pins[k] for k in ('original','bundle','diagnostics','journal','claim','report')}
        and reuse['original_received_at']==facts['original_received_at'] and reuse['original_requested_at']==facts['original_requested_at']
        and reuse['original_bytes']==facts['original_bytes'] and reuse['original_charged_active_seconds']==facts['charged_active_seconds']
        and reuse['original_spent_attempts']==1 and reuse['original_status']==facts['original_outcome']
        and reuse['original_reason']==facts['original_reason'] and reuse['original_header_seconds']==facts['header_seconds']
        and reuse['original_production_timing_suitable'] is False and reuse['admitted'] is False and reuse['reuse_review'] is None,
        'FILTERED_REUSE_SUBJECT')
    require_reuse_review(context['reuse_review'],c,reuse_subject_sha256=context['reuse_subject_sha256'],original_receipt=original['received_at'],at=at)
    require(context['budget']==POLICY,'FILTERED_BUDGET')
    venue_bridge(p,at=at)
    return original,facts


def venue_bridge(p,*,at):
    if is_fresh(p):
        # Exact venue equality was authenticated in the original scoring reader.
        return
    context=p['filtered_history'];c=context['contract'];subject=context['venue_subject'];manifest=p['target_evidence']
    require(subject==venue_subject(c,p['target']['venue'],subject['sources'])
        and digest(subject)==c['venue_bridge_subject_sha256']==context['reuse_subject']['venue_subject_sha256'], 'FILTERED_VENUE')
    allowed=[{k:r[k] for k in ('url','sha256','observed_at','representation')} for r in manifest['references']]
    require(1<=len(subject['sources'])<=len(allowed) and all(r in allowed for r in subject['sources'])
        and history.timestamp(context['reuse_review']['reviewed_at'])<=history.timestamp(at),'FILTERED_VENUE')


def validate_plan(plan,*,executable=False):
    from app_core import ncaaf_reviewed_target as target,ncaaf_staged_acquisition as stage
    p=stage._envelope(plan)
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import validate_plan as selected
        return selected(plan,executable=executable)
    if is_fresh(p):
        from app_core.ncaaf_fresh_history import validate_plan as selected
        return selected(plan,executable=executable)
    fields=set('version evidence_label target target_evidence identity_review requests limits execution_window advance_permissions collection_authorization_ref custody_id parent_history_sha256 prequote_sha256 unresolved filtered_history'.split())
    require(set(p)==fields and p['version'] in {HISTORY_PLAN,QUOTE_PLAN},'FILTERED_PLAN_SCHEMA')
    custody._credentials(p)
    t=p['target'];canonical=target.canonical(t);w=p['execution_window']
    require(set(w)=={'not_before','not_after','authorization_expires_at'} and all(history.timestamp(v) for v in w.values())
        and history.timestamp(w['not_before'])<history.timestamp(w['not_after'])<=history.timestamp(w['authorization_expires_at'])<history.timestamp(t['start_utc']), 'NCAAF_STAGED_WINDOW')
    target.validate_target(p['target_evidence'],t,w['not_before'],label=p['evidence_label'])
    require(history.timestamp(p['target_evidence']['owner_review']['reviewed_at'])<history.timestamp(w['not_before']), 'NCAAF_TARGET_REVIEW_CLOCK')
    original,_=parent(p,at=w['not_before']);c=p['filtered_history']['contract'];game=c['target']
    require(c['season']==p['target_evidence']['season'] and c['weeks']==p['target_evidence']['history_inventory']['yardage_weeks']
        and all(game[s+'Team']==canonical[s+'_school'] for s in ('home','away'))
        and game['neutralSite']==canonical['neutral_site'] and history.timestamp(game['startDate'])==history.timestamp(canonical['start_utc']), 'FILTERED_TARGET')
    for item in p['target_evidence']['history_inventory']['qualifying_games']:
        matches=[v for v in c['selected_yardage'][game[item['side']+'Team']] if v['week']==item['week']
            and v['opponent']==item['opponent'] and target.inventory_time_matches(item,v['startDate'])]
        require(len(matches)==1,'FILTERED_SELECTION')
    if p['version']==HISTORY_PLAN:
        expected=[dict(id='filtered-stats-'+str(i+1),provider='cfbd',host=r['host'],endpoint=r['endpoint'],params=r['params'],max_attempts=1) for i,r in enumerate(requests(c))]
        require(all(t[k] is None for k in target.PROVIDER) and p['identity_review'] is p['parent_history_sha256'] is p['prequote_sha256'] is None,'FILTERED_FUTURE_FACT')
    else:
        expected=target.requests(t,target.QUOTE_PLAN,p['target_evidence'])
        require(all(isinstance(p[k],str) and len(p[k])==64 for k in ('parent_history_sha256','prequote_sha256')),'NCAAF_STAGED_PARENT')
        target.validate_identity(p['identity_review'],at=w['not_before'],target=t,label=p['evidence_label'])
        require(p['identity_review']['target_manifest']==p['target_evidence'],'NCAAF_TARGET_MANIFEST_CONFLICT')
    require(p['requests']==expected and p['limits']==stage.LIMITS,'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    require(isinstance(p['unresolved'],list) and isinstance(p['advance_permissions'],list)
        and all(target.text(p[k]) for k in ('collection_authorization_ref','custody_id')),'FILTERED_PLAN_SCHEMA')
    if executable:require(not p['unresolved'] and p['evidence_label']!='PROPOSAL','NCAAF_STAGED_PREREQUISITES')
    return deepcopy(p)


def private_files(paths):
    """One bounded read-only native check; inherited file ACLs remain allowed."""
    if os.name!='nt':return False
    command=r'''
    $ErrorActionPreference='Stop'
    $current=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed=@($current,'S-1-5-18','S-1-5-32-544')
    $ok=$true
    foreach($name in (ConvertFrom-Json $env:NCAAF_FILTERED_PARENT_PATHS)) {
      $item=Get-Item -LiteralPath $name -Force
      $acl=Get-Acl -LiteralPath $name
      if($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $current){$ok=$false}
      foreach($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if($rule.AccessControlType -eq 'Allow' -and $rule.FileSystemRights -ne 0 -and $rule.IdentityReference.Value -notin $allowed){$ok=$false}
      }
      if(-not($item.Attributes -band [IO.FileAttributes]::Encrypted)){$ok=$false}
      $node=$item
      while($null -ne $node) {
        if($node.Attributes -band [IO.FileAttributes]::ReparsePoint){$ok=$false}
        if($node.PSIsContainer){$node=$node.Parent}else{$node=$node.Directory}
      }
    }
    @{verified=[bool]$ok}|ConvertTo-Json -Compress
    '''
    env=dict(os.environ,NCAAF_FILTERED_PARENT_PATHS=json.dumps([str(p) for p in paths]))
    shell=shutil.which('pwsh') or 'powershell.exe'
    if shell=='powershell.exe':env.pop('PSModulePath',None)
    try:
        result=subprocess.run([shell,'-NoProfile','-NonInteractive','-Command',command],env=env,
            capture_output=True,text=True,timeout=15,check=False)
        return result.returncode==0 and json.loads(result.stdout)=={'verified':True}
    except (OSError,ValueError,subprocess.TimeoutExpired):return False


def verify_parent_files(root,p,*,security=True):
    from pathlib import Path
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import verify_parent_files as selected
        return selected(root,p,security=security)
    if is_fresh(p):return True # No imported parent files exist in this version.
    blobs,pins=material(p);names=p['filtered_history']['parent']['custody_names']
    require(set(names)==set(pins),'FILTERED_PARENT_CUSTODY')
    paths=[]
    for key,name in names.items():
        require(isinstance(name,str) and name and Path(name).name==name and '\\' not in name and '/' not in name and name not in {'.','..'},'FILTERED_PARENT_CUSTODY')
        path=Path(root)/name
        require(not path.is_symlink() and path.is_file(),'FILTERED_PARENT_CUSTODY')
        with io.FileIO(path,'r') as stream:
            size=os.fstat(stream.fileno()).st_size
            require(0<=size<=MAX_BUNDLE,'FILTERED_BODY_LIMIT')
            raw=stream.read(size+1)
        require(raw==blobs[key],'FILTERED_PARENT_CUSTODY');paths.append(path)
    if security:require(private_files(paths),'NCAAF_STAGED_SECURITY')
    return True


def continuation_claim_name(p):
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import claim_name
        return claim_name(p)
    if is_fresh(p):
        from app_core.ncaaf_fresh_history import claim_name
        return claim_name(p)
    return p['filtered_history']['parent']['pins']['bundle']+'.continuation-claim.json'


def metadata(meta,c):
    if c.get('version')=='ncaaf-fresh-filtered-source-v1':
        from app_core.ncaaf_fresh_history import metadata as selected
        return selected(meta,c)
    require(set(meta)==set('provider endpoint request_scope status representation content_type complete received_at receipt_clock_meaning'.split())
        and meta['provider']=='cfbd' and meta['endpoint']=='games/teams' and meta['status']==200 and meta['complete'] is True
        and meta['request_scope'] in [r['params'] for r in requests(c)] and meta['representation']==custody.REPRESENTATION
        and meta['content_type']=='application/json' and meta['receipt_clock_meaning']==custody.RECEIPT_MEANING
        and history.timestamp(meta['received_at']) is not None,'FILTERED_METADATA')


def validate_response(obj,p,*,at,objects=None):
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import validate_response as selected
        return selected(obj,p,at=at,objects=objects)
    if is_fresh(p):
        from app_core.ncaaf_fresh_history import validate_response as selected
        return selected(obj,p,at=at,objects=objects)
    c=p['filtered_history']['contract'];metadata(obj['metadata'],c)
    scoring,_=parent(p,at=at)
    return validate_statistics(obj,c,scoring)


def validate_statistics(obj,c,scoring):
    games={g['id']:g for g in checked_rows(scoring['raw'])}
    q=obj['metadata']['request_scope'];wanted={v['id'] for v in c['selected_yardage'][q['team']]}
    rows=checked_rows(base64.b64decode(obj['body_b64'],validate=True));present=set()
    for row in rows:
        if row['id'] not in wanted:continue
        g=games[row['id']];teams=row.get('teams',[])
        require(all(row[k]==v for k,v in (('season',c['season']),('week',g['week']),('seasonType','regular')) if k in row),'FILTERED_PERIOD_CONFLICT')
        require(len(teams)==2 and {t.get('teamId') for t in teams}=={g['homeId'],g['awayId']},'FILTERED_STATS_TEAM')
        for team in teams:
            side='home' if team['teamId']==g['homeId'] else 'away'
            require(team.get('team')==g[side+'Team'] and team.get('points')==g[side+'Points'] and history._yards(team) is not None,'FILTERED_YARDAGE_MISSING')
        present.add(row['id'])
    require(present==wanted,'FILTERED_YARDAGE_COVERAGE')


def project_bundle(b,*,at):
    p=b['plan']['payload']
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import project_bundle as selected
        return selected(b,at=at)
    if is_fresh(p):
        from app_core.ncaaf_fresh_history import project_bundle as selected
        return selected(b,at=at)
    scoring,_=parent(p,at=at)
    values=[asset(base64.b64decode(o['body_b64'],validate=True),dict(method='GET',host=HOST,endpoint=o['metadata']['endpoint'],params=o['metadata']['request_scope']),
            o['requested_at'],o['metadata']['received_at']) for o in b['objects']]
    return project(scoring,values,p['filtered_history']['contract'],at=at)


def history_objects(b,*,at):
    if is_continuation(b['plan']['payload']):
        from app_core.ncaaf_history_continuation import history_objects as selected
        return selected(b,at=at)
    if is_fresh(b['plan']['payload']):return deepcopy(b['objects'])
    scoring,_=parent(b['plan']['payload'],at=at);blobs,_=material(b['plan']['payload'])
    original=json.loads(blobs['bundle'],object_pairs_hook=custody._unique)['payload']['objects'][0]
    meta=dict(provider='cfbd',endpoint=original['actual_request']['endpoint'],request_scope=deepcopy(original['actual_request']['params']),
        status=original['status'],representation=original['representation'],content_type=original['content_type'],complete=original['complete'],
        received_at=original['received_at'],receipt_clock_meaning=original['receipt_clock_meaning'])
    item=dict(request_id='imported-diagnostic-scoring',requested_at=scoring['requested_at'],metadata=meta,
        body_b64=original['body_b64'],body_sha256=original['body_sha256'],body_bytes=original['body_bytes'])
    return [item]+deepcopy(b['objects'])


def _response_projection(history_bundle,*,at):
    # One owned read-back and pure projection per verification operation.
    # Nothing survives the caller; mutable originals are rechecked next time.
    from app_core import ncaaf_staged_acquisition as stage
    b=stage.read_bundle(history_bundle);state,projection=project_bundle(b,at=at)
    values=[]
    for index,obj in enumerate(history_objects(b,at=at)):
        payload=dict(version=OBJECT,metadata=obj['metadata'],body_b64=obj['body_b64'],body_sha256=obj['body_sha256'],body_bytes=obj['body_bytes'],
            projection_version=PROJECTION,projection_implementation=custody.implementation(),
            projection=dict(original_index=index,history_bundle_sha256=history_bundle['sha256'],projection=projection))
        values.append(stage.seal(payload))
    return values,state


def response_objects(history_bundle,*,at):
    return _response_projection(history_bundle,at=at)[0]


def verify_response(item,policy,index,at,*,_owned_projection=None):
    packet=policy.packet['payload'];binding=packet['staged_binding']['payload']
    require(binding['version'] in {BINDING,'ncaaf-fresh-filtered-staged-evidence-v5','ncaaf-continuation-staged-evidence-v6'},'FILTERED_BINDING')
    expected,state=(_response_projection(binding['history'],at=at)
        if _owned_projection is None else _owned_projection)
    supplied=packet['custody_packet']['payload']['response_objects']
    require(len(supplied)==len(expected)+1 and supplied[:len(expected)]==expected,'FILTERED_PROJECTION_BINDING')
    n=item['payload']['projection']['original_index']
    require(type(n) is int and 0<=n<len(expected) and item==expected[n],'FILTERED_PROJECTION_BINDING')
    hashes={digest(batch) for batch in state['batches']}
    require(hashes<=set(index),'FILTERED_PROJECTION_BINDING')
    return hashes


def is_continuation(p):
    from app_core.ncaaf_history_continuation import selected
    return selected(p)


def is_fresh(p):
    return p.get('version') in {'ncaaf-fresh-filtered-history-stage-plan-v5','ncaaf-history-continuation-stage-plan-v6','ncaaf-fresh-filtered-final-quote-stage-plan-v5','ncaaf-continuation-final-quote-stage-plan-v6'}


def policy_contract(p):
    return p['filtered_history'] if is_fresh(p) else p['filtered_history']['contract']


def history_ceiling(p):
    if is_continuation(p):
        from app_core.ncaaf_history_continuation import history_ceiling as selected
        return selected(p)
    return 40 if is_fresh(p) else 27.406
