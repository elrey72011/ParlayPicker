"""Explicit evidence-bound school aliases. No global alias table or acceptance.

Only the two reviewed Memphis/Tulane spellings are new. Numeric identities are
confirmed by the separately retained CFBD schedule, never by this name review.
"""
import base64
import hashlib
from copy import deepcopy
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history
from app_core import ncaaf_response_custody as custody
from app_core.ncaaf_compatible_observation import mapping_binding

VERSION = 'ncaaf-owner-evidence-crosswalk-v1'
ALIASES = {'Memphis Tigers': 'Memphis', 'Tulane Green Wave': 'Tulane'}
require = model.require


def subject(value):
    return model.digest({k: v for k, v in value.items() if k != 'owner_review'})


def validate(value, event, schedule, crosswalk, at, *, evidence_label='RETAINED'):
    require(isinstance(value, dict) and set(value) == set('version discovery source_references prior_qualified_review event schedule crosswalk owner_review'.split())
            and value['version'] == VERSION, 'NCAAF_STAGED_ALIAS_SCHEMA')
    require(value['event'] == event and value['schedule'] == schedule and value['crosswalk'] == crosswalk,
            'NCAAF_STAGED_ALIAS_SUBJECT_CONFLICT')
    d = value['discovery']
    require(set(d) == {'body_b64', 'body_sha256', 'received_at'} and isinstance(d['body_b64'], str)
            and len(d['body_b64']) <= ((custody.MAX_BODY_BYTES+2)//3)*4, 'NCAAF_STAGED_ALIAS_EVIDENCE')
    raw = base64.b64decode(d['body_b64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == d['body_sha256'], 'NCAAF_STAGED_ALIAS_EVIDENCE')
    rows = custody._parse(raw)
    matches = [g for g in rows if g.get('id') == event['provider_event_id']]
    require(len(matches) == 1, 'NCAAF_STAGED_ALIAS_EVENT_CONFLICT')
    g = matches[0]
    require(g.get('sport_key') == 'americanfootball_ncaaf' and all(g.get(k) == event[v] for k, v in
            [('home_team', 'home_team'), ('away_team', 'away_team'), ('commence_time', 'start_utc')]),
            'NCAAF_STAGED_ALIAS_EVENT_CONFLICT')
    require(isinstance(crosswalk, list) and len(crosswalk) == 2
            and {x.get('provider_name'): x.get('schedule_name') for x in crosswalk} == ALIASES
            and all(set(x) == {'provider_name', 'schedule_name', 'team_id'} and type(x['team_id']) is int and x['team_id'] > 0 for x in crosswalk)
            and len({x['team_id'] for x in crosswalk}) == 2, 'NCAAF_STAGED_ALIAS_UNSUPPORTED')
    require(event['home_team'] == 'Tulane Green Wave' and event['away_team'] == 'Memphis Tigers'
            and event['sport'] == 'NCAAF' and event['provider_namespace'] == 'odds_api', 'NCAAF_STAGED_ALIAS_ORIENTATION')
    names = {x['provider_name']: x for x in crosswalk}
    home, away = names[event['home_team']], names[event['away_team']]
    matches = [s for s in schedule if s.get('homeId') == home['team_id'] and s.get('awayId') == away['team_id']
               and s.get('homeTeam') == home['schedule_name'] and s.get('awayTeam') == away['schedule_name']
               and history.timestamp(s.get('startDate')) == history.timestamp(event['start_utc'])]
    require(len(matches) == 1 and len(schedule) == 1, 'NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT')
    s = matches[0]
    require(type(s.get('id')) is int and s['id'] == event['schedule_game_id']
            and event['canonical_event_id'] == 'cfbd:'+str(s['id']) and s.get('completed') is False
            and s.get('startTimeTBD') is False and type(s.get('neutralSite')) is bool
            and type(event.get('neutral_site')) is bool and s['neutralSite'] == event['neutral_site'], 'NCAAF_COMPAT_EVENT_FACT_CONFLICT')
    refs = value['source_references']
    require(isinstance(refs, list) and bool(refs) and all(isinstance(r, dict)
            and set(r) == {'source', 'sha256', 'locator', 'finding'}
            and all(isinstance(r[k], str) and r[k].strip() for k in r)
            and len(r['sha256']) == 64 and len(r['finding']) >= 16 for r in refs), 'NCAAF_STAGED_ALIAS_EVIDENCE')
    r = value['owner_review']
    require(isinstance(r, dict) and set(r) == set('reviewer reviewed_at subject_sha256 finding qualifications prior_review_sha256'.split())
            and r['reviewer'] == 'Robert Velarde' and r['subject_sha256'] == subject(value)
            and isinstance(r['finding'], str) and len(r['finding'].strip()) >= 32
            and isinstance(r['qualifications'], str) and len(r['qualifications'].strip()) >= 16
            and isinstance(r['prior_review_sha256'], str) and len(r['prior_review_sha256']) == 64, 'NCAAF_STAGED_ALIAS_REVIEW')
    require(evidence_label in {'RETAINED', 'SYNTHETIC'}, 'NCAAF_STAGED_ALIAS_SCHEMA')
    prior = value['prior_qualified_review']
    require(isinstance(prior, dict) and set(prior) == {'payload', 'sha256'} and model.digest(prior['payload']) == prior['sha256']
            and prior['sha256'] == r['prior_review_sha256'], 'NCAAF_STAGED_ALIAS_REVIEW')
    if evidence_label == 'RETAINED':
        require(d['body_sha256'] == 'ea960cf84e6570345605a2ea38b9be77abcc28058173bb5374b234d8de080a65'
                and history.timestamp(d['received_at']) == history.timestamp('2026-10-16T17:00:01.125887Z')
                and r['prior_review_sha256'] == '2e15df20959cb18cb5670563fff69c787f78ffb79d100c545ec3048a08a80457', 'NCAAF_STAGED_ALIAS_EVIDENCE')
    received, reviewed, now, start = [history.timestamp(t) for t in (d['received_at'], r['reviewed_at'], at, event['start_utc'])]
    require(all((received, reviewed, now, start)) and received <= reviewed <= now < start, 'NCAAF_STAGED_ALIAS_CLOCK')
    return deepcopy(s)


def check_mapping(event, schedule, crosswalk, review, *, as_of, review_policy):
    value = review_policy.packet['payload']['alias_review']
    game = validate(value, event, schedule, crosswalk, as_of, evidence_label=review_policy.packet['payload']['evidence_label'])
    require(review.get('binding') == mapping_binding(event, schedule, crosswalk)
            and review_policy.mapping(review) and history.timestamp(value['owner_review']['reviewed_at'])
            <= history.timestamp(review.get('reviewed_at')) <= as_of, 'NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED')
    return game
