"""Ordinary, exactly reviewed filtered history acquisition. No diagnostic parent.

Original HTTP objects remain originals. Selected weekly views are derived only
after the scoring response confirms event, school identities and all histories.
This module grants neither collection permission nor review/inference authority.
"""
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import base64
import re
from app_core import ncaaf_filtered_history as filtered
from app_core import ncaaf_history as history, ncaaf_reviewed_target as target

SOURCE = 'ncaaf-fresh-filtered-source-v1'
CONTRACT = 'ncaaf-fresh-selected-game-contract-v1'
HISTORY_PLAN = 'ncaaf-fresh-filtered-history-stage-plan-v5'
QUOTE_PLAN = 'ncaaf-fresh-filtered-final-quote-stage-plan-v5'
HISTORY_BUNDLE = 'ncaaf-fresh-filtered-history-unaccepted-v5'
QUOTE_BUNDLE = 'ncaaf-fresh-filtered-final-quote-unaccepted-v5'
PREQUOTE = 'ncaaf-fresh-filtered-prequote-features-v5'
BINDING = 'ncaaf-fresh-filtered-staged-evidence-v5'
POLICY = dict(header_ceiling_seconds=30, history_active_seconds=40, market_reserve_seconds=20)
require = filtered.require


def selected(p):
    return p.get('version') in {HISTORY_PLAN, QUOTE_PLAN}


def _source(c):
    require(isinstance(c, dict) and set(c) == {'version', 'season', 'conference', 'home_team', 'away_team', 'scoring_histories', 'budget'}
        and c['version'] == SOURCE and type(c['season']) is int and 1900 <= c['season'] <= 2200
        and isinstance(c['conference'], str) and re.fullmatch(r'[A-Z][A-Z0-9]{1,9}', c['conference'])
        and all(target.text(c[k]) for k in ('home_team', 'away_team')) and c['home_team'] != c['away_team']
        and c['budget'] == POLICY, 'FRESH_SOURCE_SCOPE')
    require(isinstance(c['scoring_histories'],dict) and set(c['scoring_histories'])=={c['home_team'],c['away_team']},'FRESH_SCORING_INVENTORY')
    for rows in c['scoring_histories'].values():
        require(isinstance(rows,list) and 3<=len(rows)<=64,'FRESH_SCORING_INVENTORY')
        identities=[]
        for row in rows:
            require(isinstance(row,dict) and set(row)=={'calendar_date','calendar_zone','opponent','source_sha256'}
                and target.text(row['opponent']) and isinstance(row['source_sha256'],str)
                and re.fullmatch(r'[a-f0-9]{64}',row['source_sha256']), 'FRESH_SCORING_INVENTORY')
            try:
                day=date.fromisoformat(row['calendar_date']);ZoneInfo(row['calendar_zone'])
                require(day.isoformat()==row['calendar_date'],'FRESH_SCORING_INVENTORY')
            except (ValueError,TypeError,ZoneInfoNotFoundError):raise ValueError('FRESH_SCORING_INVENTORY') from None
            identities.append((row['calendar_date'],row['opponent']))
        require(len(set(identities))==len(identities),'FRESH_SCORING_INVENTORY')
    return c


def validate_source(c):
    """Public readers retain an independent mutable result."""
    return deepcopy(_source(c))


def requests(c):
    _source(c)
    def req(i, endpoint, params):
        return dict(id=i, provider='cfbd', host=filtered.HOST, endpoint=endpoint, params=params, max_attempts=1)
    return [req('fresh-filtered-scoring', 'games', dict(year=c['season'], seasonType='both', classification='fbs', conference=c['conference']))] + [
        req('fresh-filtered-stats-'+str(i), 'games/teams', dict(year=c['season'], seasonType='regular', team=c[side+'_team']))
        for i, side in enumerate(('home', 'away'), 1)]


def _validate_plan(plan, *, executable=False):
    from app_core import ncaaf_staged_acquisition as stage
    p=stage._envelope(plan)
    require(set(p) == set('version evidence_label target target_evidence identity_review requests limits execution_window advance_permissions collection_authorization_ref custody_id parent_history_sha256 prequote_sha256 unresolved filtered_history'.split())
        and selected(p), 'FRESH_PLAN_SCHEMA')
    c=_source(p['filtered_history']); t=p['target']; w=p['execution_window']
    canonical=target.canonical(t)
    require(isinstance(w,dict) and set(w)=={'not_before','not_after','authorization_expires_at'}
        and target.instant(w['not_before']) < target.instant(w['not_after']) <= target.instant(w['authorization_expires_at']) < target.instant(t['start_utc']), 'NCAAF_STAGED_WINDOW')
    target._validate_target(p['target_evidence'],t,w['not_before'],label=p['evidence_label'])
    require(target.instant(p['target_evidence']['owner_review']['reviewed_at']) < target.instant(w['not_before'])
        and c['season']==p['target_evidence']['season'] and c['home_team']==canonical['home_school'] and c['away_team']==canonical['away_school'], 'FRESH_TARGET_BINDING')
    evidence_hashes={r['sha256'] for r in p['target_evidence']['references']}
    cutoff=target.instant(p['target_evidence']['history_inventory']['cutoff_utc'])
    for rows in c['scoring_histories'].values():
        require(all(r['source_sha256'] in evidence_hashes and datetime.combine(date.fromisoformat(r['calendar_date']),time.min,ZoneInfo(r['calendar_zone']))<cutoff for r in rows),'FRESH_SCORING_INVENTORY')
    if p['version']==HISTORY_PLAN:
        require(all(t[k] is None for k in target.PROVIDER) and p['identity_review'] is p['parent_history_sha256'] is p['prequote_sha256'] is None, 'FILTERED_FUTURE_FACT')
        expected=requests(c)
    else:
        require(all(isinstance(p[k],str) and len(p[k])==64 for k in ('parent_history_sha256','prequote_sha256')), 'NCAAF_STAGED_PARENT')
        target.validate_identity(p['identity_review'],at=w['not_before'],target=t,label=p['evidence_label'])
        require(p['identity_review']['target_manifest']==p['target_evidence'], 'NCAAF_TARGET_MANIFEST_CONFLICT')
        expected=target.requests(t,target.QUOTE_PLAN,p['target_evidence'])
    require(p['requests']==expected and p['limits']==stage.LIMITS, 'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    require(isinstance(p['unresolved'],list) and isinstance(p['advance_permissions'],list)
        and all(target.text(p[k]) for k in ('collection_authorization_ref','custody_id')), 'FRESH_PLAN_SCHEMA')
    filtered.custody._credentials(p)
    if executable:require(not p['unresolved'] and p['evidence_label']!='PROPOSAL','NCAAF_STAGED_PREREQUISITES')
    return p


def validate_plan(plan, *, executable=False):
    return deepcopy(_validate_plan(plan,executable=executable))


def metadata(meta, c):
    _source(c)
    require(set(meta)==set('provider endpoint request_scope status representation content_type complete received_at receipt_clock_meaning'.split())
        and meta['provider']=='cfbd' and meta['status']==200 and meta['complete'] is True
        and any(meta['endpoint']==r['endpoint'] and meta['request_scope']==r['params'] for r in requests(c))
        and meta['representation']==filtered.custody.REPRESENTATION and meta['content_type']=='application/json'
        and meta['receipt_clock_meaning']==filtered.custody.RECEIPT_MEANING and history.timestamp(meta['received_at']) is not None, 'FILTERED_METADATA')


def original(obj, c):
    metadata(obj['metadata'],c)
    raw=base64.b64decode(obj['body_b64'],validate=True)
    require(filtered.sha(raw)==obj['body_sha256'] and len(raw)==obj['body_bytes'],'FILTERED_ORIGINAL_BINDING')
    return filtered.asset(raw,dict(method='GET',host=filtered.HOST,
        endpoint=obj['metadata']['endpoint'],params=obj['metadata']['request_scope']),obj['requested_at'],obj['metadata']['received_at'])


def scoring_contract(p, obj, *, at):
    """Confirm provider facts from scoring before either statistics request.

    Exact canonical labels are required; no inferred aliases or automatically
    accepted venue correspondence. A discrepancy stops with the original kept.
    """
    source=_source(p['filtered_history']); o=original(obj,source)
    require(o['actual_request']=={k:v for k,v in requests(source)[0].items() if k in {'host','endpoint','params'}} | {'method':'GET'}, 'FILTERED_SCOPE')
    rows=filtered.checked_rows(o['raw']); t=p['target']; start=target.instant(t['start_utc']); cutoff=start-timedelta(days=7)
    names={t['home_school'],t['away_school']}
    candidates=[g for g in rows if {g.get('homeTeam'),g.get('awayTeam')}==names]
    require(len(candidates)==1,'FILTERED_EVENT_CONFLICT'); g=candidates[0]
    require(all(type(g.get(k)) is int and g[k]>0 for k in ('id','homeId','awayId','venueId'))
        and g['homeId']!=g['awayId'] and g.get('homeTeam')==t['home_school'] and g.get('awayTeam')==t['away_school']
        and g.get('completed') is False and g.get('startTimeTBD') is False and g.get('season')==source['season']
        and g.get('seasonType')=='regular' and g.get('homeClassification')==g.get('awayClassification')=='fbs'
        and type(g.get('neutralSite')) is bool and g['neutralSite']==t['neutral_site']
        and target.instant(g.get('startDate'))==start and g.get('venue')==t['venue'], 'FILTERED_EVENT_CONFLICT')
    expected={}; chosen={}
    for side in ('home','away'):
        tid,name=g[side+'Id'],g[side+'Team']; values=[]
        for row in rows:
            participates=tid in (row.get('homeId'),row.get('awayId'))
            require(participates==(name in (row.get('homeTeam'),row.get('awayTeam'))),'FILTERED_TEAM_CONFLICT')
            if not participates or row['id']==g['id']:continue
            instant=target.instant(row.get('startDate'))
            if instant>=cutoff:continue
            require(history._valid_game(row,source['season']),'FILTERED_INCOMPLETE_HISTORY')
            other='away' if row['homeId']==tid else 'home'
            values.append(dict(id=row['id'],week=row['week'],startDate=row['startDate'],opponent_id=row[other+'Id'],opponent=row[other+'Team']))
        reviewed=source['scoring_histories'][name]
        require(len(values)==len(reviewed) and all(sum(v['opponent']==r['opponent'] and target.inventory_time_matches(r,v['startDate']) for v in values)==1 for r in reviewed),'FILTERED_SCORING_COVERAGE')
        expected[name]=values; chosen[name]=[]
        for item in p['target_evidence']['history_inventory']['qualifying_games']:
            if item['side']!=side:continue
            matches=[v for v in values if v['week']==item['week'] and v['opponent']==item['opponent'] and target.inventory_time_matches(item,v['startDate'])]
            require(len(matches)==1,'FILTERED_SELECTION');chosen[name].append(matches[0])
    c=dict(version=CONTRACT,season=source['season'],target={k:g[k] for k in ('id','homeId','homeTeam','awayId','awayTeam','startDate','neutralSite','venue','venueId')},
        scoring_request=o['actual_request'],scoring_sha256=o['body_sha256'],expected_scoring=expected,selected_yardage=chosen,
        weeks=p['target_evidence']['history_inventory']['yardage_weeks'],venue_bridge_subject_sha256=filtered.digest(dict(original_venue=g['venue'],reviewed_venue=t['venue'],target_manifest=p['target_evidence'])))
    filtered.verify_scoring(o,c,at=at)
    return o,c


def validate_response(obj,p,*,at,objects):
    metadata(obj['metadata'],p['filtered_history'])
    if obj['metadata']['endpoint']=='games':
        require(len(objects)==1 and obj==objects[0],'FILTERED_SCOPE')
        scoring_contract(p,obj,at=at)
    else:
        require(2<=len(objects)<=3 and objects[0]['metadata']['endpoint']=='games','FILTERED_SCOPE')
        scoring,c=scoring_contract(p,objects[0],at=at)
        filtered.validate_statistics(obj,c,scoring)


def project_bundle(b,*,at):
    require(len(b['objects'])==3,'FILTERED_YARDAGE_COVERAGE')
    p=b['plan']['payload']; scoring,c=scoring_contract(p,b['objects'][0],at=at)
    return filtered.project(scoring,[original(o,p['filtered_history']) for o in b['objects'][1:]],c,at=at)


def claim_name(p):
    return filtered.digest(dict(source=p['filtered_history'],target_manifest=p['target_evidence']))+'.fresh-history-claim.json'
