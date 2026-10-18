"""Exact, reviewed event data; never wildcard review or source acceptance.

Historical named contracts are deliberately untouched. Provider identities are
resolved only from original CFBD/discovery bodies and a later substantive review.
"""
import base64
from copy import deepcopy
from datetime import date,datetime,time,timedelta
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
from urllib.parse import urlsplit
import hashlib
from pathlib import Path
import sys
from app_core import ncaaf_future_target as prior, ncaaf_history as history
from app_core import ncaaf_model_compatibility as model, ncaaf_response_custody as custody
from app_core.ncaaf_compatible_observation import mapping_binding

VERSION='ncaaf-reviewed-target-manifest-v1'
HISTORY_PLAN='ncaaf-reviewed-history-stage-plan-v3'
QUOTE_PLAN='ncaaf-reviewed-final-quote-stage-plan-v3'
HISTORY_BUNDLE='ncaaf-reviewed-history-unaccepted-v3'
QUOTE_BUNDLE='ncaaf-reviewed-final-quote-unaccepted-v3'
IDENTITY_VERSION='ncaaf-original-discovery-crosswalk-v3'
CANONICAL=frozenset('home_school away_school venue start_utc neutral_site'.split())
PROVIDER=frozenset('provider_event_id provider_home_team provider_away_team'.split())
OWNER=prior.OWNER
require=model.require
subject=prior.subject
provider_target=prior.provider_target
discovery_original=prior.discovery_original

# Exact immutable predecessor/current pairs, generated with successor bindings.
HISTORICAL_COMPONENTS = {'app_core/ncaaf_staged_acquisition.py': {'before_sha256': 'a5ae1d18641cd549cf9853180cee522a49e6d8c3be5ca0c55b9bf79fd4fe3c6b', 'after_sha256': '49409e1e67de7142956a05a314b1913e65e3e4429a90af2123c500a71d1d3f2b'}, 'app_core/ncaaf_staged_evidence.py': {'before_sha256': '42bd522232a54457b3794c3748e326fe2e38d25180d59ea3618c3217cbcfab6c', 'after_sha256': 'e832cc334af972581b5f1472cbb598cfa1e6e0ff3886a21d244128f43f81722b'}, 'app_core/ncaaf_owner_spread.py': {'before_sha256': '647853f7025308f74b2c1abe8990e43bc8a5a0f62b3e7719bc177621aaaa6f98', 'after_sha256': 'caef9dd13839c0dea41800aa80d8e48b19621df43ef767696a347be80b4eecfe'}, 'app_core/ncaaf_pipeline_evidence.py': {'before_sha256': '8c14aa0e76ca6fe61daa5511bce55dbecb2ef1c165866a261a3a19521833b1da', 'after_sha256': '3171c6a9fd74244c86ed10a21b175623ced7ab10f899fdbca40c4d5aaa7d19ba'}}  # REVIEWED_COMPONENTS


def source_hash(path,*,historical=False):
    root=Path(__file__).resolve().parents[1]
    path=Path(path);relative=path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
    current=hashlib.sha256((root/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    record=HISTORICAL_COMPONENTS.get(relative)
    if record is not None:
        require(current==record['after_sha256'],'NCAAF_TARGET_RUNTIME_CONFLICT')
        if historical:return record['before_sha256']
    return current


def instant(value):
    t=history.timestamp(value)
    require(isinstance(value,str) and t is not None,'NCAAF_TARGET_CLOCK')
    return t


def text(value):
    return isinstance(value,str) and 0<len(value.strip())<=2048 and value==value.strip()


def check_runtime_receipt(r):
    require(isinstance(r,dict) and set(r)=={'code_head','interpreter_sha256','operation_manifest_sha256','sources'}
        and isinstance(r['code_head'],str) and len(r['code_head'])==40 and all(c in '0123456789abcdef' for c in r['code_head'])
        and all(isinstance(r[k],str) and len(r[k])==64 and all(c in '0123456789abcdef' for c in r[k])
                for k in ('interpreter_sha256','operation_manifest_sha256'))
        and isinstance(r['sources'],dict) and set(r['sources'])==set(RUNTIME_SOURCES)
        and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in r['sources'].values()),
        'NCAAF_TARGET_RUNTIME_RECEIPT')


RUNTIME_SOURCES=('app_core/ncaaf_reviewed_target.py','app_core/ncaaf_reviewed_staged.py',
    'app_core/ncaaf_staged_acquisition.py','app_core/ncaaf_staged_evidence.py','app_core/ncaaf_owner_spread.py',
    'app_core/ncaaf_pipeline_evidence.py','app_core/ncaaf_history.py','app_core/ncaaf_research.py',
    'app_core/ncaaf_spread_mass.py','app_core/ncaaf_spread_pair.py','app_core/ncaaf_spread_execution.py',
    'app_core/ncaaf_pilot.py','app_core/ncaaf_transport_diagnostics.py','scripts/ncaaf_staged_acquisition.py',
    'scripts/ncaaf_reviewed_target_operator.py')


def runtime_receipt(code_head,operation_manifest_sha256):
    root=Path(__file__).resolve().parents[1]
    return dict(code_head=code_head,interpreter_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        operation_manifest_sha256=operation_manifest_sha256,sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in RUNTIME_SOURCES})


def canonical(t):
    require(isinstance(t,dict) and set(t)==CANONICAL|PROVIDER,'NCAAF_TARGET_SCHEMA')
    c={k:t[k] for k in CANONICAL}
    require(all(text(c[k]) for k in ('home_school','away_school','venue'))
            and c['home_school']!=c['away_school'] and type(c['neutral_site']) is bool,'NCAAF_TARGET_CONFLICT')
    instant(c['start_utc'])
    return c


def inventory(value):
    require(isinstance(value,dict) and set(value)=={'season','yardage_weeks','cutoff_utc','qualifying_games'},'NCAAF_TARGET_INVENTORY')
    weeks=value['yardage_weeks']
    require(type(value['season']) is int and isinstance(weeks,list) and len(weeks)==3
            and all(type(w) is int and 0<=w<=30 for w in weeks) and weeks==sorted(set(weeks)), 'NCAAF_TARGET_INVENTORY')
    games=value['qualifying_games']
    require(isinstance(games,list) and 6<=len(games)<=64,'NCAAF_TARGET_INVENTORY')
    seen=set();counts={s:set() for s in ('home','away')}
    for g in games:
        require(isinstance(g,dict) and set(g) in ({'side','week','start_utc','opponent','source_sha256'},
                {'side','week','calendar_date','calendar_zone','opponent','source_sha256'})
                and g['side'] in counts and g['week'] in weeks and text(g['opponent'])
                and isinstance(g['source_sha256'],str) and len(g['source_sha256'])==64,'NCAAF_TARGET_INVENTORY')
        key=(g['side'],g['week']);require(key not in seen,'NCAAF_TARGET_INVENTORY');seen.add(key)
        if 'start_utc' in g:earliest=instant(g['start_utc'])
        else:
            try:
                require(text(g['calendar_zone']) and isinstance(g['calendar_date'],str),'NCAAF_TARGET_INVENTORY')
                day=date.fromisoformat(g['calendar_date']);zone=ZoneInfo(g['calendar_zone'])
                require(day.isoformat()==g['calendar_date'],'NCAAF_TARGET_INVENTORY')
                earliest=datetime.combine(day,time.min,zone)
            except (ValueError,ZoneInfoNotFoundError):raise ValueError('NCAAF_TARGET_INVENTORY') from None
        require(earliest<instant(value['cutoff_utc']),'NCAAF_TARGET_CUTOFF')
        counts[g['side']].add(g['week'])
    require(all(len(v)==3 for v in counts.values()),'NCAAF_TARGET_INVENTORY')
    return weeks


def inventory_time_matches(item,actual):
    actual=instant(actual)
    if 'start_utc' in item:return actual==instant(item['start_utc'])
    return actual.astimezone(ZoneInfo(item['calendar_zone'])).date().isoformat()==item['calendar_date']


def validate_target(value,t,at,*,label):
    c=canonical(t);now=instant(at)
    require(isinstance(value,dict) and set(value)=={'version','evidence_label','sport','season','canonical_target','references','history_inventory','owner_review'}
            and value['version']==VERSION and value['evidence_label']==label
            and label in {'RETAINED','SYNTHETIC','PROPOSAL'} and value['sport']=='NCAAF'
            and type(value['season']) is int and value['season']==instant(c['start_utc']).year
            and value['canonical_target']==c and len(model.encode(value))<=65536,'NCAAF_TARGET_MANIFEST')
    refs=value['references'];require(isinstance(refs,list) and 1<=len(refs)<=8,'NCAAF_TARGET_EVIDENCE')
    clocks=[];seen=set();hashes=set()
    for r in refs:
        require(isinstance(r,dict) and set(r)=={'url','representation','observed_at','locator','excerpt','facts','sha256'}
                and r['representation']=='attributed_public_reference_not_original_response_custody'
                and all(text(r[k]) for k in ('url','locator','excerpt')) and r['facts']==c
                and r['sha256']==model.digest({k:v for k,v in r.items() if k!='sha256'}),'NCAAF_TARGET_EVIDENCE')
        u=urlsplit(r['url'])
        require(u.scheme=='https' and u.hostname and not u.username and not u.password
                and r['url'] not in seen,'NCAAF_TARGET_EVIDENCE')
        seen.add(r['url']);hashes.add(r['sha256']);clocks.append(instant(r['observed_at']))
    inv=value['history_inventory'];inventory(inv)
    require(inv['season']==value['season'] and instant(inv['cutoff_utc'])==instant(c['start_utc'])-timedelta(days=7)
            and all(g['source_sha256'] in hashes for g in inv['qualifying_games']),'NCAAF_TARGET_CUTOFF')
    reviewed=prior._review(value['owner_review'],value,at,dict(named_event=c,source_basis=refs,history_inventory=inv))
    require(all(v<=reviewed for v in clocks) and reviewed<=now<instant(c['start_utc']),'NCAAF_TARGET_REVIEW_CLOCK')
    return deepcopy(value)


def requests(t,kind,manifest):
    from app_core import ncaaf_staged_acquisition as stage
    if kind==QUOTE_PLAN:return stage.requests(t,stage.QUOTE_PLAN)
    require(kind==HISTORY_PLAN,'NCAAF_TARGET_PLAN_VERSION')
    year=manifest['season'];weeks=inventory(manifest['history_inventory'])
    def req(i,endpoint,params):
        return dict(id=i,provider='cfbd',host=stage.transport.HOSTS['cfbd'],endpoint=endpoint,params=params,max_attempts=1)
    return [req('cfbd-season-scoring-'+str(year),'games',dict(year=year,seasonType='both',classification='fbs'))]+[
        req('cfbd-yardage-'+str(year)+'-week-'+str(w),'games/teams',dict(year=year,week=w,seasonType='regular')) for w in weeks]


def validate_plan(plan,*,executable=False):
    from app_core import ncaaf_staged_acquisition as stage
    p=stage._envelope(plan)
    require(set(p)==set('version evidence_label target target_evidence identity_review requests limits execution_window advance_permissions collection_authorization_ref custody_id parent_history_sha256 prequote_sha256 unresolved'.split())
            and p['version'] in {HISTORY_PLAN,QUOTE_PLAN},'NCAAF_TARGET_PLAN_SCHEMA')
    custody._credentials(p);w=p['execution_window'];t=p['target'];c=canonical(t)
    require(isinstance(w,dict) and set(w)=={'not_before','not_after','authorization_expires_at'}
            and instant(w['not_before'])<instant(w['not_after'])<=instant(w['authorization_expires_at'])<instant(c['start_utc']), 'NCAAF_STAGED_WINDOW')
    validate_target(p['target_evidence'],t,w['not_before'],label=p['evidence_label'])
    require(instant(p['target_evidence']['owner_review']['reviewed_at'])<instant(w['not_before']),'NCAAF_TARGET_REVIEW_CLOCK')
    if p['version']==HISTORY_PLAN:
        require(all(t[k] is None for k in PROVIDER) and p['identity_review'] is None
                and p['parent_history_sha256'] is p['prequote_sha256'] is None,'NCAAF_TARGET_FUTURE_FACT')
    else:
        require(all(text(t[k]) for k in PROVIDER) and all(isinstance(p[k],str) and len(p[k])==64
                for k in ('parent_history_sha256','prequote_sha256')),'NCAAF_TARGET_QUOTE_IDENTITY')
        validate_identity(p['identity_review'],at=w['not_before'],target=t,label=p['evidence_label'])
        require(p['identity_review']['target_manifest']==p['target_evidence'],'NCAAF_TARGET_MANIFEST_CONFLICT')
    require(p['requests']==requests(t,p['version'],p['target_evidence']) and p['limits']==stage.LIMITS,'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    require(isinstance(p['unresolved'],list) and isinstance(p['advance_permissions'],list)
            and all(text(p[k]) for k in ('collection_authorization_ref','custody_id')),'NCAAF_TARGET_PLAN_SCHEMA')
    if executable:require(not p['unresolved'] and p['evidence_label']!='PROPOSAL','NCAAF_STAGED_PREREQUISITES')
    return deepcopy(p)


def validate_identity(value,*,at,target,label):
    require(isinstance(value,dict) and set(value)=={'version','evidence_label','target_manifest','discovery','schedule','crosswalk','source_references','owner_review'}
            and value['version']==IDENTITY_VERSION and value['evidence_label']==label,'NCAAF_TARGET_IDENTITY_SCHEMA')
    manifest=value['target_manifest'];validate_target(manifest,target,at,label=label);c=canonical(target)
    obj=prior.discovery_original(value['discovery'],label)
    q=obj['metadata']['request_scope'];expected=requests(target,QUOTE_PLAN,manifest)[0]['params']
    require(q==expected,'NCAAF_TARGET_IDENTITY_REQUEST')
    rows=custody._parse(base64.b64decode(obj['body_b64'],validate=True))
    matches=[g for g in rows if g.get('id')==target['provider_event_id'] or
        (g.get('home_team')==target['provider_home_team'] and g.get('away_team')==target['provider_away_team'])]
    require(len(matches)==1,'NCAAF_TARGET_IDENTITY_CONFLICT');g=matches[0]
    require(g.get('id')==target['provider_event_id'] and g.get('sport_key')=='americanfootball_ncaaf'
        and all(g.get(side+'_team')==target['provider_'+side+'_team'] for side in ('home','away'))
        and instant(g.get('commence_time'))==instant(c['start_utc']),'NCAAF_TARGET_IDENTITY_CONFLICT')
    schedule=value['schedule'];require(isinstance(schedule,list) and len(schedule)==1,'NCAAF_TARGET_IDENTITY_CONFLICT')
    game=schedule[0]
    require(all(game.get(side+'Team')==c[side+'_school'] for side in ('home','away'))
        and all(type(game.get(k)) is int and game[k]>0 for k in ('id','homeId','awayId')) and game['homeId']!=game['awayId']
        and game.get('completed') is False and game.get('startTimeTBD') is False
        and type(game.get('neutralSite')) is bool and game['neutralSite']==c['neutral_site']
        and instant(game.get('startDate'))==instant(c['start_utc']) and game.get('seasonType')=='regular'
        and game.get('season')==manifest['season'],'NCAAF_TARGET_IDENTITY_CONFLICT')
    walk=[dict(provider_name=target['provider_'+s+'_team'],schedule_name=c[s+'_school'],team_id=game[s+'Id']) for s in ('home','away')]
    require(value['crosswalk']==walk and len({v['provider_name'] for v in walk})==2,'NCAAF_TARGET_IDENTITY_CONFLICT')
    refs=value['source_references'];require(isinstance(refs,list) and len(refs)==2,'NCAAF_TARGET_ALIAS_EVIDENCE')
    sources={r['url'] for r in manifest['references']}
    for r,x in zip(refs,walk):
        require(set(r)=={'provider_name','schedule_name','source','observed_at','locator','excerpt','sha256'}
            and r['provider_name']==x['provider_name'] and r['schedule_name']==x['schedule_name'] and r['source'] in sources
            and text(r['locator']) and text(r['excerpt'])
            and r['sha256']==model.digest({k:v for k,v in r.items() if k!='sha256'}),'NCAAF_TARGET_ALIAS_EVIDENCE')
    reviewed=prior._review(value['owner_review'],value,at,dict(target_manifest=manifest,original_discovery=value['discovery'],
        confirmed_schedule=schedule,school_orientation=walk,source_basis=refs))
    require(instant(obj['requested_at'])<=instant(obj['metadata']['received_at'])<=reviewed
        and all(instant(r['observed_at'])<=reviewed for r in refs),'NCAAF_TARGET_IDENTITY_CLOCK')
    d=value['discovery']['bundle']['payload']['plan']['payload']['target']
    require(d['canonical_event_id']=='cfbd:'+str(game['id']) and d['home_id']==game['homeId'] and d['away_id']==game['awayId']
        and d['neutral_site']==c['neutral_site'] and instant(d['start_utc'])==instant(c['start_utc']),'NCAAF_TARGET_IDENTITY_CONFLICT')
    return deepcopy(game)


def same_target(left,right):
    return canonical(left)==canonical(right)


def check_mapping(event,schedule,crosswalk,review,*,as_of,review_policy):
    packet=review_policy.packet['payload'];value=packet['alias_review']
    from app_core import ncaaf_reviewed_staged as staged
    require(packet['staged_binding']['payload']['version']==staged.BINDING_VERSION,'NCAAF_TARGET_VERSION_CONFLICT')
    t=dict(value['target_manifest']['canonical_target'],provider_event_id=event.get('provider_event_id'),
        provider_home_team=event.get('home_team'),provider_away_team=event.get('away_team'))
    game=validate_identity(value,at=as_of.isoformat(),target=t,label=packet['evidence_label'])
    require(value['schedule']==schedule and value['crosswalk']==crosswalk
        and event==dict(sport='NCAAF',provider_namespace='odds_api',provider_event_id=t['provider_event_id'],canonical_event_id='cfbd:'+str(game['id']),
            schedule_game_id=game['id'],home_team=t['provider_home_team'],away_team=t['provider_away_team'],start_utc=t['start_utc'],neutral_site=t['neutral_site'])
        and review.get('binding')==mapping_binding(event,schedule,crosswalk) and review_policy.mapping(review)
        and instant(value['owner_review']['reviewed_at'])<=instant(review.get('reviewed_at'))<=as_of,'NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED')
    return game
