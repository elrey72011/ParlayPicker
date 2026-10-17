"""Explicit Nebraska/Oregon successor. References and review are never acceptance.

The history plan deliberately has no future Odds identity. Its original CFBD
body must confirm the canonical event before a later reviewed discovery joins it.
No global aliases, inferred numeric IDs, links fetched or trust catalog writes.
"""
import base64
from copy import deepcopy
import hashlib
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history
from app_core import ncaaf_response_custody as custody
from app_core.ncaaf_compatible_observation import mapping_binding

HISTORY_PLAN = 'ncaaf-named-history-stage-plan-v2'
QUOTE_PLAN = 'ncaaf-reviewed-final-quote-stage-plan-v2'
HISTORY_BUNDLE = 'ncaaf-named-history-unaccepted-v2'
QUOTE_BUNDLE = 'ncaaf-reviewed-final-quote-unaccepted-v2'
VERSION = 'ncaaf-nebraska-oregon-target-review-v1'
IDENTITY_VERSION = 'ncaaf-original-discovery-crosswalk-v2'
OWNER = 'Robert Velarde'
CANONICAL = dict(home_school='Oregon', away_school='Nebraska', venue='Autzen Stadium, Eugene, Oregon',
                 start_utc='2026-10-18T03:00:00Z', neutral_site=False)
SOURCES = frozenset({
    'https://huskers.com/news/2026/10/11/nebraska-oregon-kickoff-time-and-tv-set',
    'https://goducks.com/news/2026/10/13/football-ducks-back-home-to-host-nebraska-saturday-night'})
require = model.require


def subject(value):
    return model.digest({k:v for k,v in value.items() if k != 'owner_review'})


def _review(review, value, at, facets):
    require(isinstance(review,dict) and set(review)=={'reviewer','reviewed_at','subject_sha256','qualifications','conclusions'}
        and review['reviewer']==OWNER and review['subject_sha256']==subject(value)
        and isinstance(review['qualifications'],str) and len(review['qualifications'].strip())>=16,
        'NCAAF_FRESH_TARGET_REVIEW')
    reviewed,now=history.timestamp(review['reviewed_at']),history.timestamp(at)
    require(reviewed is not None and now is not None and reviewed<=now,
        'NCAAF_FRESH_TARGET_REVIEW_CLOCK')
    require(isinstance(review['conclusions'],dict) and set(review['conclusions'])==set(facets), 'NCAAF_FRESH_TARGET_REVIEW')
    for key,value in facets.items():
        c=review['conclusions'][key]
        require(isinstance(c,dict) and set(c)=={'conclusion','evidence_sha256','finding'}
            and c['conclusion']=='VERIFIED' and c['evidence_sha256']==model.digest(value)
            and isinstance(c['finding'],str) and len(c['finding'].strip())>=32, 'NCAAF_FRESH_TARGET_REVIEW')
    return reviewed


def validate_target(value, target, at, *, label):
    require(isinstance(target,dict) and set(target)==set(CANONICAL)|{'provider_event_id','provider_home_team','provider_away_team'}
        and {k:target[k] for k in CANONICAL}==CANONICAL, 'NCAAF_FRESH_TARGET_CONFLICT')
    require(isinstance(value,dict) and set(value)=={'version','evidence_label','canonical_target','references','owner_review'}
        and value['version']==VERSION and value['evidence_label']==label and label in {'RETAINED','SYNTHETIC','PROPOSAL'}
        and value['canonical_target']==CANONICAL, 'NCAAF_FRESH_TARGET_EVIDENCE')
    refs=value['references']
    require(isinstance(refs,list) and len(refs)==2 and {r.get('url') for r in refs}==SOURCES, 'NCAAF_FRESH_TARGET_EVIDENCE')
    clocks=[]
    for r in refs:
        require(set(r)=={'url','representation','observed_at','locator','excerpt','facts','sha256'}
            and r['representation']=='attributed_public_reference_not_original_response_custody'
            and r['facts']==CANONICAL and all(isinstance(r[k],str) and r[k].strip() for k in ('locator','excerpt'))
            and r['sha256']==model.digest({k:v for k,v in r.items() if k!='sha256'}), 'NCAAF_FRESH_TARGET_EVIDENCE')
        clocks.append(history.timestamp(r['observed_at']))
    reviewed=_review(value['owner_review'],value,at,dict(named_event=CANONICAL,source_basis=refs))
    require(all(t is not None and t<=reviewed for t in clocks) and reviewed<history.timestamp(CANONICAL['start_utc']),
        'NCAAF_FRESH_TARGET_REVIEW_CLOCK')
    return deepcopy(value)


def validate_plan(plan, *, executable=False):
    from app_core import ncaaf_staged_acquisition as stage
    p=stage._envelope(plan)
    require(set(p)==set('version evidence_label target target_evidence identity_review requests limits execution_window advance_permissions collection_authorization_ref custody_id parent_history_sha256 prequote_sha256 unresolved'.split())
        and p['version'] in {HISTORY_PLAN,QUOTE_PLAN}, 'NCAAF_FRESH_PLAN_SCHEMA')
    custody._credentials(p)
    w=p['execution_window']; t=p['target']; is_history=p['version']==HISTORY_PLAN
    require(isinstance(w,dict) and set(w)=={'not_before','not_after','authorization_expires_at'}
        and all(history.timestamp(v) for v in w.values())
        and history.timestamp(w['not_before'])<history.timestamp(w['not_after'])<=history.timestamp(w['authorization_expires_at'])
        <history.timestamp(CANONICAL['start_utc']), 'NCAAF_STAGED_WINDOW')
    validate_target(p['target_evidence'],t,w['not_before'],label=p['evidence_label'])
    require(history.timestamp(p['target_evidence']['owner_review']['reviewed_at'])<history.timestamp(w['not_before']),
        'NCAAF_FRESH_TARGET_REVIEW_CLOCK')
    if is_history:
        require(all(t[k] is None for k in ('provider_event_id','provider_home_team','provider_away_team'))
            and p['identity_review'] is None and p['parent_history_sha256'] is None and p['prequote_sha256'] is None,
            'NCAAF_FRESH_HISTORY_FUTURE_FACT')
    else:
        require(all(isinstance(t[k],str) and t[k].strip() for k in ('provider_event_id','provider_home_team','provider_away_team'))
            and isinstance(p['identity_review'],dict)
            and all(isinstance(p[k],str) and len(p[k])==64 for k in ('parent_history_sha256','prequote_sha256')),
            'NCAAF_FRESH_QUOTE_IDENTITY')
        validate_identity(p['identity_review'],at=w['not_before'],target=t,label=p['evidence_label'])
    expected=stage.requests(t,stage.HISTORY_PLAN if is_history else stage.QUOTE_PLAN)
    require(p['requests']==expected and p['limits']==stage.LIMITS,'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    require(isinstance(p['unresolved'],list) and isinstance(p['advance_permissions'],list)
        and all(isinstance(p[k],str) and p[k] for k in ('collection_authorization_ref','custody_id')), 'NCAAF_STAGED_PLAN_SCHEMA')
    if executable: require(not p['unresolved'] and p['evidence_label']!='PROPOSAL','NCAAF_STAGED_PREREQUISITES')
    return deepcopy(p)


def validate_identity(value, *, at, target, label):
    require(isinstance(value,dict) and set(value)=={'version','evidence_label','discovery','schedule','crosswalk','source_references','owner_review'}
        and value['version']==IDENTITY_VERSION and value['evidence_label']==label, 'NCAAF_FRESH_IDENTITY_SCHEMA')
    d=value['discovery']; obj=discovery_original(d,label)
    raw=base64.b64decode(obj['body_b64'],validate=True)
    q=obj['metadata']['request_scope']
    expected=dict(regions='us',bookmakers='draftkings',markets='spreads',oddsFormat='american',dateFormat='iso',
        commenceTimeFrom='2026-10-18T02:59:59Z',commenceTimeTo='2026-10-18T03:00:01Z')
    require(q==expected,'NCAAF_FRESH_IDENTITY_REQUEST')
    rows=custody._parse(raw)
    matches=[g for g in rows if g.get('id')==target['provider_event_id'] or
        (g.get('home_team')==target['provider_home_team'] and g.get('away_team')==target['provider_away_team'])]
    require(len(matches)==1,'NCAAF_FRESH_IDENTITY_CONFLICT'); g=matches[0]
    require(g.get('id')==target['provider_event_id'] and g.get('sport_key')=='americanfootball_ncaaf'
        and g.get('home_team')==target['provider_home_team'] and g.get('away_team')==target['provider_away_team']
        and history.timestamp(g.get('commence_time'))==history.timestamp(CANONICAL['start_utc']), 'NCAAF_FRESH_IDENTITY_CONFLICT')
    schedule=value['schedule']; crosswalk=value['crosswalk']
    require(isinstance(schedule,list) and len(schedule)==1 and isinstance(crosswalk,list) and len(crosswalk)==2,
        'NCAAF_FRESH_IDENTITY_CONFLICT')
    game=schedule[0]
    require(game.get('homeTeam')==CANONICAL['home_school'] and game.get('awayTeam')==CANONICAL['away_school']
        and all(type(game.get(k)) is int and game[k]>0 for k in ('id','homeId','awayId')) and game['homeId']!=game['awayId']
        and game.get('completed') is False and game.get('startTimeTBD') is False and game.get('neutralSite') is False
        and history.timestamp(game.get('startDate'))==history.timestamp(CANONICAL['start_utc']), 'NCAAF_FRESH_IDENTITY_CONFLICT')
    expected_walk=[dict(provider_name=target['provider_'+side+'_team'],schedule_name=CANONICAL[side+'_school'],team_id=game[side+'Id']) for side in ('home','away')]
    require(crosswalk==expected_walk and len({x['provider_name'] for x in crosswalk})==2, 'NCAAF_FRESH_IDENTITY_CONFLICT')
    refs=value['source_references']
    require(isinstance(refs,list) and len(refs)==2 and all(set(r)=={'provider_name','schedule_name','source','locator','excerpt','sha256'}
        and r['provider_name']==x['provider_name'] and r['schedule_name']==x['schedule_name']
        and all(isinstance(r[k],str) and r[k].strip() for k in ('source','locator','excerpt'))
        and (r['source'] in SOURCES or label=='SYNTHETIC' and r['source'].startswith('SYNTHETIC'))
        and r['sha256']==model.digest({k:v for k,v in r.items() if k!='sha256'}) for r,x in zip(refs,crosswalk)),
        'NCAAF_FRESH_IDENTITY_SUPPORT')
    reviewed=_review(value['owner_review'],value,at,dict(original_discovery=d,confirmed_schedule=schedule,school_orientation=crosswalk,source_basis=refs))
    requested,received=history.timestamp(obj['requested_at']),history.timestamp(obj['metadata']['received_at'])
    require(requested is not None and received is not None and requested<=received<=reviewed<history.timestamp(CANONICAL['start_utc']),
        'NCAAF_FRESH_IDENTITY_CLOCK')
    return deepcopy(game)


def discovery_original(value,label):
    """Static original bundle/control/journal linkage, no trust activation or IO."""
    import json
    from app_core import ncaaf_pilot_discovery as discovery
    require(isinstance(value,dict) and set(value)=={'bundle','journal_b64','configuration_receipt'},'NCAAF_FRESH_DISCOVERY_RECEIPT')
    bundle=value['bundle']; b=discovery.envelope(bundle,'NCAAF_FRESH_DISCOVERY_RECEIPT')
    p=discovery.validate_plan(b['plan'],executable=True)
    require(b['version']==discovery.BUNDLE_VERSION and b['evidence_label']==label==p['evidence_label']
        and b['status']=='DISCOVERED_UNACCEPTED' and b['reason'] is None and b['attempted_requests']==len(b['objects'])==1
        and all(b[k] is False for k in ('accepted','inference','scientific_acceptance','wagering_authority')),
        'NCAAF_FRESH_DISCOVERY_RECEIPT')
    obj=b['objects'][0]; req=p['requests'][0]; meta=obj['metadata']; custody._metadata(meta)
    require(obj['request_id']==req['id'] and meta['request_scope']==req['params'] and meta['provider']==req['provider']
        and meta['endpoint']==req['endpoint'] and isinstance(obj['body_b64'],str)
        and len(obj['body_b64'])<=((custody.MAX_BODY_BYTES+2)//3)*4,'NCAAF_FRESH_DISCOVERY_RECEIPT')
    raw=base64.b64decode(obj['body_b64'],validate=True)
    require(type(obj['body_bytes']) is int and 0<len(raw)==obj['body_bytes']<=custody.MAX_BODY_BYTES
        and hashlib.sha256(raw).hexdigest()==obj['body_sha256']
        and discovery.identifiers(raw,p)==b['identifiers'],'NCAAF_FRESH_DISCOVERY_RECEIPT')
    a=b['owner_authorization_receipt']; ap=discovery.envelope(a,'NCAAF_FRESH_DISCOVERY_RECEIPT')
    permission=p['advance_permission']
    require(ap['version']==discovery.AUTH_VERSION and ap['plan_sha256']==b['plan']['sha256']
        and ap['reference']==p['collection_authorization_ref'] and ap['custody_id']==p['custody_id']
        and ap['owner']==OWNER and ap['max_attempts']==ap['max_provider_credits']==1 and ap['max_cost_usd']==0
        and ap['expires_at']==p['execution_window']['authorization_expires_at']
        and permission['requests_sha256']==model.digest(p['requests']) and permission['target_sha256']==model.digest(p['target'])
        and permission['provider']=='odds_api' and permission['permitted_use']=='private_identifier_discovery_retention',
        'NCAAF_FRESH_DISCOVERY_RECEIPT')
    requested,received=[history.timestamp(x) for x in (obj['requested_at'],meta['received_at'])]
    require(requested is not None and received is not None and requested<=received
        and history.timestamp(ap['authorized_at'])<=requested<=received<history.timestamp(ap['expires_at'])
        and history.timestamp(p['execution_window']['not_before'])<=requested<=received<=history.timestamp(p['execution_window']['not_after'])
        and history.timestamp(permission['reviewed_at'])<requested
        and history.timestamp(permission['effective_from'])<=requested<=received<history.timestamp(permission['effective_until']),
        'NCAAF_FRESH_DISCOVERY_RECEIPT')
    jr=base64.b64decode(value['journal_b64'],validate=True)
    require(len(jr)<=128*1024,'NCAAF_FRESH_DISCOVERY_RECEIPT')
    events=[json.loads(line,object_pairs_hook=custody._unique,parse_constant=custody._constant) for line in jr.splitlines()]
    require(events==[
        dict(version=discovery.VERSION,plan_sha256=b['plan']['sha256'],authorization_sha256=a['sha256'],custody_id=p['custody_id']),
        dict(request=req,attempt=1,requested_at=obj['requested_at'],request_clock_meaning='actual_local_request_start'),
        dict(received_at=meta['received_at'],body_sha256=obj['body_sha256'],identifier_status=b['identifiers']['state'])],
        'NCAAF_FRESH_DISCOVERY_JOURNAL')
    control=value['configuration_receipt']
    require(isinstance(control,dict) and set(control)=={'file_b64','file_sha256'} and isinstance(control['file_b64'],str)
        and len(control['file_b64'])<=((128*1024+2)//3)*4,'NCAAF_FRESH_DISCOVERY_CONFIGURATION')
    cr=base64.b64decode(control['file_b64'],validate=True)
    require(hashlib.sha256(cr).hexdigest()==control['file_sha256'],'NCAAF_FRESH_DISCOVERY_CONFIGURATION')
    cfg=json.loads(cr,object_pairs_hook=custody._unique,parse_constant=custody._constant)
    require(set(cfg)=={'payload','sha256'} and model.digest(cfg['payload'])==cfg['sha256']
        and set(cfg['payload'])=={'subject','review'},'NCAAF_FRESH_DISCOVERY_CONFIGURATION')
    subject=cfg['payload']['subject']; review=cfg['payload']['review']; mappings=subject['bindings']
    from pathlib import Path
    require(set(subject)=={'version','plan_sha256','permission','authorization','bindings','code_head','reader_sha256'}
        and subject['version']=='ncaaf-discovery-reviewed-control-subject-v1'
        and isinstance(subject['code_head'],str) and len(subject['code_head'])==40
        and all(c in '0123456789abcdef' for c in subject['code_head'])
        and subject['reader_sha256']==hashlib.sha256(Path(discovery.__file__).read_bytes()).hexdigest()
        and subject['plan_sha256']==b['plan']['sha256'] and subject['permission']==permission and subject['authorization']==a
        and set(mappings)=={'AUTHORIZED_DISCOVERIES','ACCEPTED_DISCOVERY_PERMISSIONS','AUTHORIZED_CUSTODY_ROOTS'}
        and mappings['AUTHORIZED_DISCOVERIES']=={ap['reference']:a['sha256']}
        and mappings['ACCEPTED_DISCOVERY_PERMISSIONS']=={permission['reference']:model.digest(permission)}
        and set(mappings['AUTHORIZED_CUSTODY_ROOTS'])=={p['custody_id']}
        and set(review)=={'reviewer','reviewed_at','subject_sha256','status','scope'}
        and review['reviewer']==OWNER and review['subject_sha256']==model.digest(subject)
        and review['status']=='OWNER_REVIEWED' and review['scope']=='NCAAF identifier discovery acquisition only; no offer admission.'
        and history.timestamp(review['reviewed_at']) is not None
        and history.timestamp(ap['authorized_at'])<=history.timestamp(review['reviewed_at'])<requested,
        'NCAAF_FRESH_DISCOVERY_CONFIGURATION')
    return deepcopy(obj)


def provider_target(target):
    return dict(provider_event_id=target['provider_event_id'],home_team=target['provider_home_team'],away_team=target['provider_away_team'],
        start_utc=target['start_utc'],neutral_site=target['neutral_site'])


def same_target(left,right):
    return all(left.get(k)==right.get(k)==v for k,v in CANONICAL.items())


def check_mapping(event,schedule,crosswalk,review,*,as_of,review_policy):
    value=review_policy.packet['payload']['alias_review']
    t=dict(CANONICAL,provider_event_id=event.get('provider_event_id'),provider_home_team=event.get('home_team'),provider_away_team=event.get('away_team'))
    game=validate_identity(value,at=as_of.isoformat(),target=t,label=review_policy.packet['payload']['evidence_label'])
    require(value['schedule']==schedule and value['crosswalk']==crosswalk
        and event==dict(sport='NCAAF',provider_namespace='odds_api',provider_event_id=t['provider_event_id'],canonical_event_id='cfbd:'+str(game['id']),
            schedule_game_id=game['id'],home_team=t['provider_home_team'],away_team=t['provider_away_team'],start_utc=t['start_utc'],neutral_site=False)
        and review.get('binding')==mapping_binding(event,schedule,crosswalk) and review_policy.mapping(review)
        and history.timestamp(value['owner_review']['reviewed_at'])<=history.timestamp(review.get('reviewed_at'))<=as_of,
        'NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED')
    return game
