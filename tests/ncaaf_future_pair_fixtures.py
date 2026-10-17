"""Labelled synthetic evidence only. No provider, account, Git history or live key."""
import base64
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from app_core import ncaaf_future_target as target, ncaaf_future_staged as evidence
from app_core import ncaaf_staged_acquisition as stage,ncaaf_staged_evidence as prior
from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_spread_execution as execution
from app_core import ncaaf_model_compatibility as model,ncaaf_response_custody as custody,ncaaf_pipeline_evidence as adapter
from app_core import ncaaf_prospective_chronology as chronology
from app_core.ncaaf_compatible_observation import mapping_binding
from test_ncaaf_pilot import Stream

NOW=datetime(2026,10,17,18,30,5,tzinfo=timezone.utc)
HOME='SYNTHETIC Oregon Ducks';AWAY='SYNTHETIC Nebraska Cornhuskers';EID='SYNTHETIC-NO-REAL-EVENT-ID'
CANONICAL=deepcopy(target.CANONICAL)


class Clock:
    def __init__(self,value='2026-10-17T18:00:01Z'):self.value=value;self.elapsed=0.
    def wall(self):return self.value
    def mono(self):return self.elapsed


def _finding(sha,facet):
    return dict(conclusion='VERIFIED',evidence_sha256=sha,finding='SYNTHETIC substantive fixture finding for '+facet+'; no actual human review or acceptance.')


def target_review():
    refs=[]
    for url in sorted(target.SOURCES):
        r=dict(url=url,representation='attributed_public_reference_not_original_response_custody',observed_at='2026-10-17T15:00:00Z',
            locator='SYNTHETIC paragraph1',excerpt='SYNTHETIC schedule excerpt, not an authentic captured document.',facts=deepcopy(CANONICAL))
        r['sha256']=model.digest(r);refs.append(r)
    v=dict(version=target.VERSION,evidence_label='SYNTHETIC',canonical_target=deepcopy(CANONICAL),references=refs,owner_review={})
    v['owner_review']=dict(reviewer=owner.OWNER,reviewed_at='2026-10-17T16:00:00Z',subject_sha256=target.subject(v),
        qualifications='SYNTHETIC qualified review only; no actual owner answer.',
        conclusions={k:_finding(model.digest(x),k) for k,x in dict(named_event=CANONICAL,source_basis=refs).items()})
    return v


def bodies():
    t=dict(id=987654,season=2026,week=7,seasonType='regular',startDate=CANONICAL['start_utc'],startTimeTBD=False,
        completed=False,neutralSite=False,homeId=111,homeTeam='Oregon',homePoints=None,awayId=222,awayTeam='Nebraska',awayPoints=None)
    games=[t];stats={w:[] for w in (1,2,3)}
    for w in (1,2,3):
        for tid,name in ((111,'Oregon'),(222,'Nebraska')):
            g=dict(t,id=tid*10+w,week=w,startDate=f'2026-09-{w*7:02}T17:00:00Z',completed=True,
                homeId=tid,homeTeam=name,homePoints=17,awayId=9000+tid,awayTeam='SYNTHETIC opponent '+name,awayPoints=14)
            games.append(g);stats[w].append(dict(id=g['id'],season=2026,week=w,seasonType='regular',teams=[
                dict(teamId=g['homeId'],points=17,stats=[dict(category='totalYards',stat='400')]),
                dict(teamId=g['awayId'],points=14,stats=[dict(category='totalYards',stat='300')])]))
    return [model.encode(games)]+[model.encode(stats[w]) for w in (1,2,3)]


def odds(line=-7,clock='2026-10-17T18:30:00Z'):
    return model.encode([dict(id=EID,sport_key='americanfootball_ncaaf',home_team=HOME,away_team=AWAY,commence_time=CANONICAL['start_utc'],
        bookmakers=[dict(key='draftkings',last_update=clock,markets=[dict(key='spreads',last_update=clock,outcomes=[
            dict(name=HOME,point=line,price=-110),dict(name=AWAY,point=-line,price=-110)])])])])


def plan(kind=target.HISTORY_PLAN,h=None,pref=None,identity=None):
    quote=kind==target.QUOTE_PLAN;t=dict(CANONICAL,provider_event_id=EID if quote else None,
        provider_home_team=HOME if quote else None,provider_away_team=AWAY if quote else None)
    req=stage.requests(t,stage.QUOTE_PLAN if quote else stage.HISTORY_PLAN)
    return stage.seal(dict(version=kind,evidence_label='SYNTHETIC',target=t,target_evidence=target_review(),identity_review=identity,
        requests=req,limits=deepcopy(stage.LIMITS),execution_window=dict(not_before='2026-10-17T18:30:00Z' if quote else '2026-10-17T18:00:00Z',
            not_after='2026-10-17T18:45:00Z' if quote else '2026-10-17T18:15:00Z',authorization_expires_at='2026-10-17T18:45:30Z' if quote else '2026-10-17T18:15:30Z'),
        advance_permissions=[dict(reference='SYNTHETIC-permission-'+kind,provider='odds_api' if quote else 'cfbd',reviewed_at='2026-10-17T17:00:00Z',
            effective_from='2026-10-17T17:00:00Z',effective_until='2026-10-18T02:00:00Z',permitted_use='private_prospective_capture',requests_sha256=model.digest(req))],
        collection_authorization_ref='SYNTHETIC-authority-'+kind,custody_id='SYNTHETIC-private',
        parent_history_sha256=h['sha256'] if h else None,prequote_sha256=pref['sha256'] if pref else None,unresolved=[]))


def stage_configuration(p,monkeypatch,root):
    q=stage.is_quote(p['payload']);a=stage.seal(dict(version=stage.AUTH_VERSION,reference=p['payload']['collection_authorization_ref'],plan_sha256=p['sha256'],
        custody_id=p['payload']['custody_id'],authorized_at='2026-10-17T18:20:00Z' if q else '2026-10-17T17:30:00Z',
        expires_at=p['payload']['execution_window']['authorization_expires_at'],owner=owner.OWNER))
    account=dict(provider=p['payload']['requests'][0]['provider'],included_remaining=10,observed_at='2026-10-17T17:00:00Z',
        valid_until='2026-10-18T02:00:00Z',evidence_sha256='a'*64,no_paid_overage=True,reviewer=owner.OWNER)
    c=dict(version='ncaaf-staged-controlled-bindings-v1',plan_sha256=p['sha256'],authorization=a,permission=p['payload']['advance_permissions'][0],
        custody=dict(id=p['payload']['custody_id'],root=str(root),storage_receipt_sha256='c'*64),account=account,review={})
    c['review']=dict(reviewer=owner.OWNER,reviewed_at='2026-10-17T18:25:00Z' if q else '2026-10-17T17:45:00Z',
        subject_sha256=model.digest({k:v for k,v in c.items() if k!='review'}),finding='SYNTHETIC exact controlled configuration finding; no actual application or human approval.')
    raw=model.encode(stage.seal(c))
    monkeypatch.setattr(stage,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage,'ACCEPTED_ADVANCE_PERMISSIONS',{v['reference']:model.digest(v) for v in p['payload']['advance_permissions']})
    monkeypatch.setattr(stage,'AUTHORIZED_CUSTODY_ROOTS',{p['payload']['custody_id']:str(root)})
    monkeypatch.setattr(stage,'AUTHORIZED_ACCOUNT_EVIDENCE',{a['payload']['reference']:account})
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{p['sha256']:dict(file_sha256=hashlib.sha256(raw).hexdigest(),file_b64=base64.b64encode(raw).decode())})
    monkeypatch.setattr(stage,'inspect_security',lambda r:dict(status='VERIFIED'))
    return a,raw


def capture(monkeypatch,root,p=None,raw=None,clock=None,h=None,pref=None):
    p=p or plan();raw=raw or bodies();clock=clock or Clock();a,_=stage_configuration(p,monkeypatch,root);sent=[]
    def transport(req,limits,deadline,mono):
        sent.append(deepcopy(req));clock.elapsed+=.01;return Stream(raw[len(sent)-1])
    b=stage.acquire(p,a,root=root,transport=transport,history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    return b,sent,clock


def review_prequote(p,h):
    p=deepcopy(p);v=p['payload'];v['history_review']=dict(version='ncaaf-owner-prequote-review-v2',reviewer=owner.OWNER,
        reviewed_at='2026-10-17T18:03:00Z',subject_sha256=prior.prequote_subject(p),qualifications='SYNTHETIC documented comparison basis only, not an actual owner answer.',
        conclusions={k:_finding(sha,k) for k,sha in prior.prequote_review_subjects(v,h).items()})
    return stage.seal(v)


def identity(monkeypatch,root,pref):
    from app_core import ncaaf_pilot_discovery as d
    raw=odds(-7,'2026-10-17T18:01:04Z');clock=Clock('2026-10-17T18:01:05Z')
    t=dict(canonical_event_id='cfbd:987654',home_team='Oregon',home_id=111,away_team='Nebraska',away_id=222,start_utc=CANONICAL['start_utc'],neutral_site=False,
        schedule_reference='SYNTHETIC retained CFBD original '+pref['payload']['history_bundle_sha256'])
    request=dict(id='SYNTHETIC-discovery',provider='odds_api',host='api.the-odds-api.com',endpoint='v4/sports/americanfootball_ncaaf/odds',
        params=dict(regions='us',bookmakers='draftkings',markets='spreads',oddsFormat='american',dateFormat='iso',
            commenceTimeFrom='2026-10-18T02:59:59Z',commenceTimeTo='2026-10-18T03:00:01Z'),max_attempts=1)
    permission=dict(reference='SYNTHETIC-discovery-permission',provider='odds_api',reviewer=owner.OWNER,license_holder='SYNTHETIC owner',
        evidence_reference='SYNTHETIC qualified source-use review',reviewed_at='2026-10-17T18:00:30Z',effective_from='2026-10-17T18:00:30Z',
        effective_until='2026-10-17T18:16:00Z',permitted_use='private_identifier_discovery_retention',requests_sha256=model.digest([request]),target_sha256=model.digest(t))
    p=stage.seal(dict(version=d.VERSION,evidence_label='SYNTHETIC',target=t,offer_scope=dict(market='spreads',side='home',bookmaker='draftkings'),requests=[request],
        limits=dict(max_attempts=1,max_objects=16,max_body_bytes=524288,max_aggregate_bytes=2097152,connect_seconds=3,read_seconds=5,total_seconds=60),
        execution_window=dict(not_before='2026-10-17T18:01:00Z',not_after='2026-10-17T18:15:00Z',authorization_expires_at='2026-10-17T18:15:30Z'),
        advance_permission=permission,collection_authorization_ref='SYNTHETIC-discovery-authority',custody_id='SYNTHETIC-private',unresolved=[]))
    a=stage.seal(dict(version=d.AUTH_VERSION,reference=p['payload']['collection_authorization_ref'],plan_sha256=p['sha256'],custody_id='SYNTHETIC-private',
        authorized_at='2026-10-17T18:00:40Z',expires_at='2026-10-17T18:15:30Z',owner=owner.OWNER,max_attempts=1,max_provider_credits=1,max_cost_usd=0,quota_reference='SYNTHETIC included'))
    bindings=dict(AUTHORIZED_DISCOVERIES={a['payload']['reference']:a['sha256']},ACCEPTED_DISCOVERY_PERMISSIONS={permission['reference']:model.digest(permission)},
        AUTHORIZED_CUSTODY_ROOTS={'SYNTHETIC-private':str(root)})
    for k,v in bindings.items():monkeypatch.setattr(d,k,v)
    b=d.acquire(p,a,root=root,transport=lambda *args:Stream(raw),wall=clock.wall,monotonic=clock.mono)
    from pathlib import Path
    sub=dict(version='ncaaf-discovery-reviewed-control-subject-v1',plan_sha256=p['sha256'],permission=permission,authorization=a,bindings=bindings,
        code_head='a'*40,reader_sha256=hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest())
    config=stage.seal(dict(subject=sub,review=dict(reviewer=owner.OWNER,reviewed_at='2026-10-17T18:00:50Z',subject_sha256=model.digest(sub),status='OWNER_REVIEWED',
        scope='NCAAF identifier discovery acquisition only; no offer admission.'))); cr=model.encode(config)
    v=dict(version=target.IDENTITY_VERSION,evidence_label='SYNTHETIC',discovery=dict(bundle=b,
        journal_b64=base64.b64encode((root/(p['sha256']+'.discovery-attempts.jsonl')).read_bytes()).decode(),
        configuration_receipt=dict(file_b64=base64.b64encode(cr).decode(),file_sha256=hashlib.sha256(cr).hexdigest())),schedule=pref['payload']['schedule'],
        crosswalk=[dict(provider_name=HOME,schedule_name='Oregon',team_id=111),dict(provider_name=AWAY,schedule_name='Nebraska',team_id=222)],source_references=[],owner_review={})
    for x in v['crosswalk']:
        r=dict(provider_name=x['provider_name'],schedule_name=x['schedule_name'],source='SYNTHETIC official identity comparison',locator='SYNTHETIC record1',
            excerpt='SYNTHETIC named school and orientation comparison, not an authentic document.')
        r['sha256']=model.digest(r);v['source_references'].append(r)
    facets=dict(original_discovery=v['discovery'],confirmed_schedule=v['schedule'],school_orientation=v['crosswalk'],source_basis=v['source_references'])
    v['owner_review']=dict(reviewer=owner.OWNER,reviewed_at='2026-10-17T18:02:00Z',subject_sha256=target.subject(v),
        qualifications='SYNTHETIC qualified finding only; no actual human review.',conclusions={k:_finding(model.digest(x),k) for k,x in facets.items()})
    return v


def resign(packet,reviewed='2026-10-17T18:30:04Z'):
    p=packet['payload'];c=p['custody_packet'];n=c['payload']['native_packet'];o=n['payload']['observation']['payload'];r=o['source_review'];q=o['quote']
    r['quote_observation']['quote_sha256']=model.digest(q)
    r['offer_verification'].update(quote_sha256=model.digest(q),observation_sha256=model.digest(r['quote_observation']),terms_sha256=model.digest(r['terms_review']),mapping_sha256=model.digest(o['mapping_review']))
    r['owner_verification'].update(subject_version=owner.ReviewPolicy.OBSERVATION_VERSION,subject_sha256=chronology.subject_hash(o))
    n['payload']['observation']['sha256']=model.digest(o);n['sha256']=model.digest(n['payload'])
    d=p['dependency_review'];v=d['dependency_verification'];v.update(subject_sha256=chronology.dependency_subject_hash(n),dependency_hashes=[x['sha256'] for x in n['payload']['dependency_objects']],permissions_sha256=model.digest(d['permissions_review']))
    d['owner_verification'].update(subject_sha256=v['subject_sha256'],verification_sha256=model.digest(v))
    admission=c['payload']['custody_admission'];v=admission['verification'];v.update(subject_sha256=custody.subject_hash(c),permissions_sha256=model.digest(d['permissions_review']),terms_sha256=model.digest(r['terms_review']),native_verification_sha256=model.digest(d['dependency_verification']))
    admission['owner_verification'].update(subject_sha256=v['subject_sha256'],verification_sha256=model.digest(v));c['sha256']=model.digest(c['payload'])
    p['owner_review']=dict(version=owner.REVIEW_VERSION,review_id='SYNTHETIC combined exact-input review',reviewer=owner.OWNER,reviewed_at=reviewed,
        review_status=owner.STATUS,use_scope=owner.SCOPE,subject_version=owner.SUBJECT_VERSION,subject_sha256=owner.subject_hash(packet),attestation=owner.ATTESTATION,
        conclusions={k:_finding(v,k) for k,v in owner.evidence_subjects(packet).items()})
    packet['sha256']=model.digest(p);return packet


def fixture(synthetic,monkeypatch,root,line=-7):
    from test_ncaaf_owner_terms import fixture as old_fixture
    from app_core import ncaaf_owner_terms_research as old
    packet,row=old_fixture(synthetic,monkeypatch,'spread_home',-3.5)
    versions={getattr(old.ReviewPolicy,k):getattr(owner.ReviewPolicy,k) for k in ('VERSION','RESULT_VERSION','NATIVE_VERSION','NATIVE_RESULT_VERSION','OBSERVATION_VERSION','OFFER_VERSION','DEPENDENCY_VERSION','CUSTODY_VERSION')}
    versions[old.projection.VERSION]=owner.projection.VERSION
    def migrate(v):
        if isinstance(v,dict):return {k:migrate(x) for k,x in v.items()}
        if isinstance(v,list):return [migrate(x) for x in v]
        if isinstance(v,str) and v in versions:return versions[v]
        if isinstance(v,str) and v.startswith('2026-10-09T12:00:'):return v.replace('2026-10-09T12:00:','2026-10-17T18:30:')
        return v
    packet=migrate(packet);p=packet['payload'];c=p['custody_packet'];n=c['payload']['native_packet'];o=n['payload']['observation']['payload']
    h,_,_=capture(monkeypatch,root);pref=evidence.prepare(h,model_packet=o['model'],prepared_at='2026-10-17T18:01:00Z');ident=identity(monkeypatch,root,pref);pref=review_prequote(pref,h)
    q,_,_=capture(monkeypatch,root,plan(target.QUOTE_PLAN,h,pref,ident),[odds(line)],Clock('2026-10-17T18:30:01Z'),h,pref)
    p.update(version=owner.VERSION if float(line).is_integer() else owner.HALF_VERSION,staged_binding=evidence.join(h,pref,q),alias_review=ident)
    event=dict(sport='NCAAF',provider_namespace='odds_api',provider_event_id=EID,canonical_event_id='cfbd:987654',schedule_game_id=987654,home_team=HOME,away_team=AWAY,start_utc=CANONICAL['start_utc'],neutral_site=False)
    o.update(as_of='2026-10-17T18:30:04Z',event=event,schedule=pref['payload']['schedule'],crosswalk=ident['crosswalk'],model=pref['payload']['model'],
        features=dict(order=list(evidence.research.FEATURES),values=pref['payload']['ordered_features'],available_at=pref['payload']['prepared_at']))
    o['mapping_review'].update(binding=mapping_binding(event,o['schedule'],o['crosswalk']),reviewed_at='2026-10-17T18:30:02Z')
    o['quote'].update(provider_event_id=EID,event_home_team=HOME,event_away_team=AWAY,event_start_utc=CANONICAL['start_utc'],book='draftkings',operator='draftkings',point=line,
        recorded_at='2026-10-17T18:30:00Z',price=-110,period_source=owner.projection.VERSION,rules_source=owner.projection.VERSION)
    o['source_review']['terms_review']['operator']='draftkings'
    bs=evidence.batches(h);n['payload']['dependency_objects']=[dict(sha256=hashlib.sha256(model.encode(b)).hexdigest(),bytes_b64=base64.b64encode(model.encode(b)).decode()) for b in bs]
    o['feature_dependencies']=[dict(game_id=g['id'],team_id=tid,season=2026,start_utc=g['startDate'],available_at=b['retrieved_at'],kind='scoring' if i==0 else 'yardage',source_sha256=n['payload']['dependency_objects'][i]['sha256'])
        for i,b in enumerate(bs) for g in bs[0]['records'] if g['completed'] and (not i or g['week']==b['request']['week']) for tid in (111,222) if tid in (g['homeId'],g['awayId'])]
    with owner.projection.selected(packet):
        c['payload']['response_objects']=[custody.capture(base64.b64decode(obj['body_b64']),obj['metadata'],native_batch=bs[i]) if i<4 else
            custody.capture(base64.b64decode(obj['body_b64']),obj['metadata'],quote=o['quote'],terms=o['source_review']['terms_review'],owner_reviewed=True)
            for i,obj in enumerate(h['payload']['objects']+q['payload']['objects'])]
    resign(packet)
    from app_core.prediction_evidence import provider_quotes
    row.update(home_team=HOME,away_team=AWAY,provider_event_id=EID,game_start_utc=CANONICAL['start_utc'],provider_quotes=provider_quotes(json.loads(odds(line))[0]),odds_american=-110,
        matchup_id=HOME+'|'+AWAY+'|2026-10-17',best_pick=HOME+f' {line:+.1f}',spread_line=line)
    row['provider_quotes']=json.dumps([x for x in json.loads(row['provider_quotes']) if x['market_type']=='spread_home'])
    monkeypatch.setattr(adapter,'generated_time',lambda:NOW.isoformat())
    return packet,row


def other_side(packet,row):
    packet=deepcopy(packet);row=deepcopy(row);p=packet['payload'];c=p['custody_packet'];o=c['payload']['native_packet']['payload']['observation']['payload'];q=o['quote']
    q.update(market_type='spread_away',point=-q['point'])
    last=p['staged_binding']['payload']['quote']['payload']['objects'][0]
    with owner.projection.selected(packet):
        c['payload']['response_objects'][-1]=custody.capture(base64.b64decode(last['body_b64']),last['metadata'],quote=q,terms=o['source_review']['terms_review'],owner_reviewed=True)
    resign(packet)
    from app_core.prediction_evidence import provider_quotes
    row.update(market_type='spread_away',spread_line=q['point'],best_pick=AWAY+f" {q['point']:+.1f}",
        provider_quotes=json.dumps([x for x in json.loads(provider_quotes(json.loads(base64.b64decode(last['body_b64']))[0])) if x['market_type']=='spread_away']))
    return packet,row


def pair_fixture(synthetic,monkeypatch,root,line=-7):
    h,hr=fixture(synthetic,monkeypatch,root,line);a,ar=other_side(h,hr)
    p=pair.prepare(h,a,prepared_at='2026-10-17T18:30:02Z')
    p['payload']['owner_review']=dict(version=pair.REVIEW_VERSION,reviewer=owner.OWNER,reviewed_at='2026-10-17T18:30:04Z',
        subject_sha256=pair.subject(p),qualifications='SYNTHETIC qualified comparison basis only; no actual human review.',
        finding='SYNTHETIC owner reviewed both exact child subjects and same observed market; no actual answer.',
        child_reviews=dict(home=h['payload']['owner_review'],away=a['payload']['owner_review']))
    return stage.seal(p['payload']),dict(home=hr,away=ar)


def inference_authorization(packet,monkeypatch,*,paired=False):
    cs=pair.load(packet) if paired else dict(home=packet)
    a=stage.seal(dict(version=execution.VERSION,reference='SYNTHETIC-explicit-inference',subject_sha256=packet['sha256'],
        input_hashes=[p['sha256'] for p in cs.values()],authorized_at='2026-10-17T18:30:04Z',expires_at='2026-10-17T18:44:59Z',
        owner=owner.OWNER,max_computations=len(cs)))
    monkeypatch.setattr(execution,'AUTHORIZED_INFERENCES',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(execution,'inspect_security',lambda r:dict(status='VERIFIED'))
    return a
