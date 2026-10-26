"""Derive new NFL subjects from originals, never from a per-game code catalog."""
from copy import deepcopy
from datetime import datetime, timezone
import math
import pandas as pd

from app_core import nfl_scenario_research as scenario, nfl_native_provenance as native
from app_core import nfl_inference_evidence as evidence, nfl_scoring_custody as custody
from app_core.producer_provenance import team
from app_core.research_replay import cell
from app_core.nfl_workflow_store import require, WorkflowError, digest
from core.nfl_teams import NFL_TEAMS

FINDINGS = ('event_mapping','scenario_definition','permissions','clocks_and_original_custody','feature_provenance','model_binding')
ATTESTATION = 'I verified the original evidence and sporting assumptions for this exact scenario.'


def instant(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        require(result.tzinfo is not None, 'NFL_WORKFLOW_CLOCK_REQUIRED')
        return result.astimezone(timezone.utc)
    except (TypeError, AttributeError, ValueError):
        raise WorkflowError('NFL_WORKFLOW_CLOCK_REQUIRED') from None


def scheduled(row):
    try:
        start = pd.Timestamp(row['gameday']+'T'+row['gametime']).tz_localize('America/New_York', ambiguous='raise', nonexistent='raise').tz_convert('UTC').isoformat()
        home = native.nfl_stats_identity(row['home_team'], schedule_code=True)
        away = native.nfl_stats_identity(row['away_team'], schedule_code=True)
        names = {value.upper():value for value in NFL_TEAMS.values()}
        require(home in names and away in names and home != away, 'NFL_WORKFLOW_IDENTITY_CONFLICT')
        return dict(canonical_event_id=row['game_id'], home=team(names[home],'NFL'), away=team(names[away],'NFL'),
                    start=start, season=int(row['season']), neutral_site=row.get('location') != 'Home')
    except (KeyError, ValueError, TypeError):
        raise WorkflowError('NFL_WORKFLOW_SCHEDULE_INVALID') from None


def eligible_games(source, at):
    _, rows = custody.read_score(source['object'])
    require(len({row['game_id'] for row in rows}) == len(rows), 'NFL_WORKFLOW_IDENTITY_CONFLICT')
    result, rejected = [], []
    for row in rows:
        if any(row.get(k) for k in ('home_score','away_score','result')):
            continue
        try:
            event = scheduled(row)
            if instant(event['start']) <= instant(at):
                continue
            require(event['neutral_site'] is False, 'NFL_WORKFLOW_NEUTRAL_SITE_UNSUPPORTED')
            result.append(dict(event=event, row=deepcopy(row), row_sha256=digest(row), venue=row.get('stadium','Unknown')))
        except ValueError as error:
            rejected.append(dict(id=row.get('game_id','Unknown'), reason=str(error)))
    return dict(games=sorted(result,key=lambda x:(x['event']['start'],x['event']['canonical_event_id'])),rejected=rejected)


def prepare_game(source, selected, at):
    obj = source['object']; _, original = custody.read_score(obj)
    event = selected['event']
    require(not event['neutral_site'] and instant(at) < instant(event['start']), 'NFL_WORKFLOW_KICKOFF_REACHED')
    matches = [row for row in original if row['game_id'] == event['canonical_event_id']]
    require(len(matches) == 1 and scheduled(matches[0]) == event and digest(matches[0]) == selected['row_sha256'], 'NFL_WORKFLOW_IDENTITY_CONFLICT')
    permission = source['permission']
    require(instant(permission['reviewed_at']) <= instant(obj['receipt']['request_started_at'])
            and instant(permission['effective_from']) <= instant(obj['receipt']['request_started_at'])
            and instant(obj['receipt']['observed_at']) <= instant(at) < instant(permission['effective_until']), 'NFL_WORKFLOW_SOURCE_PERMISSION_EXPIRED')
    rows = [row for row in original if int(row['season']) == event['season']]
    typed = []
    for row in rows:
        values = [float(row[k]) if row.get(k) else None for k in ('home_score','away_score','result')]
        require(all(v is None or math.isfinite(v) for v in values), 'NFL_WORKFLOW_HISTORY_CONFLICT')
        if any(v is not None for v in values):
            require(all(v is not None for v in values) and values[0]>=0 and values[1]>=0
                    and values[2]==values[0]-values[1]
                    and row['gameday'] <= instant(obj['receipt']['observed_at']).date().isoformat()
                    and row['game_id'] != event['canonical_event_id'], 'NFL_WORKFLOW_HISTORY_CONFLICT')
        converted = dict(row,season=int(row['season']),**dict(zip(('home_score','away_score','result'),values)))
        typed.append({key:cell(value) for key,value in converted.items()})
    facts = dict(columns=list(rows[0]),rows=typed)
    stats, histories = {}, {}
    for side in ('home','away'):
        members = native.selected_rows(facts,event['season'],at,native.nfl_stats_identity(event[side]))
        require(bool(members), 'NFL_WORKFLOW_HISTORY_MISSING')
        stats[side] = native.aggregates(members)
        histories[side] = [dict(game_id=native.value(x[1]['game_id']), gameday=native.value(x[1]['gameday'])) for x in members]
    features = {}
    for name in evidence.FEATURES:
        if name == 'feature_diff_last5':
            features[name] = stats['home']['last5_win_pct']-stats['away']['last5_win_pct']
        else:
            _,side,key = name.split('_',2); features[name]=stats[side][native.MAPPING[key]]
    return dict(selected=deepcopy(selected),source=deepcopy(source),features=features,
                feature_order=list(evidence.FEATURES),histories=histories,prepared_at=at,
                model=scenario.model_binding(),reader=scenario.reader_binding())


def observed_pair(original, prepared, at, *, max_age_seconds=900):
    from app_core import nfl_observed_custody as odds, nfl_odds_acquisition as acquire
    _, events = odds.read(original)
    e = prepared['selected']['event']
    target = dict(home=next(v for v in NFL_TEAMS.values() if team(v,'NFL')==e['home']),
                  away=next(v for v in NFL_TEAMS.values() if team(v,'NFL')==e['away']),
                  kickoff=e['start'],canonical_event_id=e['canonical_event_id'],season=e['season'],neutral_site=False,
                  schedule_evidence_sha256=prepared['selected']['row_sha256'])
    findings = acquire.target_findings(events,target)
    require(findings['reason'] is None, 'NFL_WORKFLOW_MARKET_IDENTITY_CONFLICT')
    event=next(row for row in events if row['id']==findings['provider_event_id'])
    require(len(event['bookmakers'])==1,'NFL_WORKFLOW_MARKET_UNSUPPORTED')
    book=event['bookmakers'][0]
    require(book['key']=='novig' and len(book['markets'])==1,'NFL_WORKFLOW_MARKET_UNSUPPORTED')
    market=book['markets'][0]
    require(market['key']=='spreads' and len(market['outcomes'])==2,'NFL_WORKFLOW_MARKET_UNSUPPORTED')
    pair={team(row['name'],'NFL'):row for row in market['outcomes']}
    require(set(pair)=={e['home'],e['away']},'NFL_WORKFLOW_MARKET_IDENTITY_CONFLICT')
    for row in pair.values():
        require(type(row.get('point')) in (int,float) and math.isfinite(row['point']) and abs(row['point']%1)==.5,'NFL_WORKFLOW_INTEGER_LINE_UNSUPPORTED')
        require(type(row.get('price')) in (int,float) and math.isfinite(row['price']) and abs(row['price'])>=100,'NFL_WORKFLOW_MARKET_UNSUPPORTED')
    require(pair[e['home']]['point']+pair[e['away']]['point']==0,'NFL_WORKFLOW_MARKET_IDENTITY_CONFLICT')
    updated=instant(market['last_update']);observed=instant(original['receipt']['observed_at']);current=instant(at)
    require(current<instant(e['start']),'NFL_WORKFLOW_KICKOFF_REACHED')
    require(updated<=observed<=current and (current-updated).total_seconds()<=max_age_seconds,'NFL_WORKFLOW_QUOTE_STALE')
    return dict(pair=pair,updated_at=market['last_update'],observed_at=original['receipt']['observed_at'],provider_event_id=event['id'])


def packets(prepared, original, at):
    pair=observed_pair(original,prepared,at);e=prepared['selected']['event'];source=prepared['source']
    result=[]
    for side in ('home','away'):
        selected=e[side]
        p=dict(version=scenario.VERSION,evidence_label=source['object']['receipt']['evidence_label'],
            scoring=source['object'],event=e,event_verification=dict(schedule_row_sha256=prepared['selected']['row_sha256'],
                schedule_timezone='America/New_York',observed_at=prepared['prepared_at'],
                source_references=[dict(reference='Authenticated private NFL schedule/scoring capture '+source['object']['body_sha256'],sha256=source['object']['body_sha256'])]),
            scenario=dict(selected_team=selected,signed_handicap=pair['pair'][selected]['point'],period='full_game',overtime=True,condition=scenario.CONDITION),
            permission=source['permission'],features=prepared['features'],feature_order=prepared['feature_order'],
            feature_available_at=at,model=prepared['model'],reader=prepared['reader'],owner_review=None,original_inference_time=None)
        packet=scenario.seal(p)
        try:
            scenario.checked(packet,at)
        except ValueError as error:
            require(str(error)=='NFL_SCENARIO_OWNER_REVIEW_MISSING',str(error))
        else:
            raise WorkflowError('NFL_WORKFLOW_UNEXPECTED_REVIEW')
        result.append(packet)
    return result


def reviewed(packet, checked_findings, at, *, basis):
    require(set(checked_findings)==set(FINDINGS) and all(value is True for value in checked_findings.values()),'NFL_WORKFLOW_OWNER_FINDINGS_REQUIRED')
    require(isinstance(basis,str) and len(basis.strip())>=16,'NFL_WORKFLOW_REVIEW_BASIS_REQUIRED')
    p=deepcopy(packet['payload']);subject=scenario.subject(p)
    p['owner_review']=dict(version=scenario.REVIEW,reviewer='Robert Velarde',reviewed_at=at,
        subject_sha256=subject,status='OWNER_REVIEWED',purpose='PRIVATE_RESEARCH',attestation=ATTESTATION,
        findings={name:dict(conclusion='VERIFIED',basis=basis+' Owner explicitly checked '+name+' for this exact subject.') for name in FINDINGS})
    require(scenario.subject(p)==subject,'NFL_WORKFLOW_SUBJECT_CHANGED')
    return scenario.seal(p)
