"""Synthetic-only versioned continuation, actual CLI/readers and budget controls."""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from app_core import ncaaf_filtered_history as filtered,ncaaf_staged_acquisition as stage
from app_core import ncaaf_reviewed_target as target,ncaaf_reviewed_staged as evidence
from app_core import ncaaf_staged_evidence as subjects,ncaaf_transport_diagnostics as observations
from app_core import ncaaf_response_custody as custody,ncaaf_history as history
import ncaaf_reviewed_target_fixtures as legacy
from test_ncaaf_model_compatibility import synthetic

SECRET='SYNTHETIC_ONLY_NOT_A_PROVIDER_KEY'


def encoded(v):return stage.model.encode(v)
def sha(raw):return hashlib.sha256(raw).hexdigest()


def context(monkeypatch,root):
    monkeypatch.setattr(legacy,'CANONICAL',dict(legacy.CANONICAL,start_utc='2026-10-22T23:00:00Z'))
    monkeypatch.setattr(legacy,'WEEKS',(3,4,6))
    manifest=legacy.target_review();originals=[json.loads(raw) for raw in legacy.bodies()]
    games=originals[0];game=games[0];game['venue']='SYNTHETIC short stadium';game['venueId']=4899
    for tid,name in ((111,legacy.CANONICAL['home_school']),(222,legacy.CANONICAL['away_school'])):
        for date,week in (('09-01',1),('09-02',2),('10-14',7)):
            games.append(dict(game,id=tid*100+week,week=week,startDate='2026-'+date+'T17:00:00Z',completed=True,
                homeId=tid,homeTeam=name,homePoints=21,awayId=9500+tid,awayTeam='SYNTHETIC extra '+name,awayPoints=20))
    request=dict(method='GET',host=filtered.HOST,endpoint='games',params=dict(year=2026,seasonType='both',classification='fbs',conference='CUSA'))
    body=encoded(games);requested='2026-10-17T17:40:00Z';received='2026-10-17T17:40:12.500000Z'
    original=dict(request_index=0,actual_request=request,requested_at=requested,received_at=received,
        receipt_clock_meaning='actual_local_complete_response_receipt',status=200,representation='complete_original_response_bytes',
        content_type='application/json',complete=True,body_bytes=len(body),body_sha256=sha(body),body_b64=base64.b64encode(body).decode())
    method=encoded(dict(version='SYNTHETIC_ONLY',requests=[request]));mh=sha(method)
    approval=encoded(dict(owner=legacy.owner.OWNER,received_at='2026-10-17T17:20:00Z',method_file_sha256=mh,
        decision='APPROVE_EXACT_DIAGNOSTIC_METHOD_AND_MANUAL_EXECUTION',answer_verbatim='SYNTHETIC original diagnostic authorization, not an actual user answer.'))
    control=encoded(stage.seal(dict(version='cfbd-cusa-reviewed-diagnostic-control-v1',method_file_sha256=mh,requests=[request],
        execution_approval_file_sha256=sha(approval),authorization=dict(owner=legacy.owner.OWNER,authorized_at='2026-10-17T17:20:00Z',expires_at='2026-10-17T17:55:30Z'),
        review=dict(reviewed_at='2026-10-17T17:20:00Z'),window=dict(not_before='2026-10-17T17:40:00Z',not_after='2026-10-17T17:55:00Z'))))
    ch=sha(control);active=12.594
    journal=b''.join(encoded(r)+b'\n' for r in [dict(kind='start',method_file_sha256=mh,control_file_sha256=ch),
        dict(kind='attempt',attempt=1,actual_request=request,requested_at=requested),dict(kind='receipt',attempt=1,received_at=received,body_sha256=sha(body)),
        dict(kind='finish',attempts=1,active_seconds=active,reason='PROBE_HEADER_PRODUCTION_LIMIT')])
    bundle=stage.seal(dict(version='cfbd-cusa-diagnostic-only-v1',method_file_sha256=mh,control_file_sha256=ch,status='DIAGNOSTIC_STOPPED',
        reason='PROBE_HEADER_PRODUCTION_LIMIT',attempts=1,objects=[original],active_seconds=active,acceptance=False,inference=False))
    bundle_raw=encoded(bundle)
    diagnostics=encoded(stage.seal(dict(version='cfbd-cusa-diagnostic-observations-v1',method_file_sha256=mh,control_file_sha256=ch,
        bundle_sha256=bundle['sha256'],bundle_file_sha256=sha(bundle_raw),journal_sha256=sha(journal),active_seconds=active,
        attempts=[dict(http_status=200,complete_original=True,phases=[dict(phase='response_headers',seconds=12.375,visits=1)])])))
    claim=encoded(dict(method_file_sha256=mh,control_file_sha256=ch))
    report=encoded(dict(method_file_sha256=mh,control_file_sha256=ch,bundle_file_sha256=sha(bundle_raw),journal_sha256=sha(journal),
        diagnostics_file_sha256=sha(diagnostics),active_seconds=active,provider_attempts=1,production_admitted=False,inference=False,retry=False))
    blobs=dict(original=body,bundle=bundle_raw,diagnostics=diagnostics,journal=journal,control=control,method=method,approval=approval,claim=claim,report=report)
    names={k:'SYNTHETIC-parent-'+k+'.json' for k in blobs}
    for k,raw in blobs.items():(root/names[k]).write_bytes(raw)
    def rows_for(name,tid):
        return [dict(id=g['id'],week=g['week'],startDate=g['startDate'],opponent_id=g['awayId'] if g['homeId']==tid else g['homeId'],
            opponent=g['awayTeam'] if g['homeId']==tid else g['homeTeam']) for g in games if g['completed'] and tid in (g['homeId'],g['awayId'])]
    expected={game[s+'Team']:rows_for(game[s+'Team'],game[s+'Id']) for s in ('home','away')}
    c=dict(version=filtered.VERSION,season=2026,target={k:game[k] for k in ('id','homeId','homeTeam','awayId','awayTeam','startDate','neutralSite','venue','venueId')},
        scoring_request=request,scoring_sha256=sha(body),expected_scoring=expected,
        selected_yardage={n:[r for r in rows if r['week'] in (3,4,6)] for n,rows in expected.items()},weeks=[3,4,6],venue_bridge_subject_sha256='0'*64)
    refs=[{k:r[k] for k in ('url','sha256','observed_at','representation')} for r in manifest['references']]
    venue=filtered.venue_subject(c,legacy.CANONICAL['venue'],refs);c['venue_bridge_subject_sha256']=filtered.digest(venue)
    reuse=dict(version='ncaaf-diagnostic-scoring-reuse-subject-v1',original_sha256=sha(body),method_file_sha256=mh,control_file_sha256=ch,approval_file_sha256=sha(approval),
        artifact_pins={('original1' if k=='original' else k):sha(blobs[k]) for k in ('original','bundle','diagnostics','journal','claim','report')},
        original_received_at=received,original_requested_at=requested,original_bytes=len(body),original_charged_active_seconds=active,
        original_spent_attempts=1,original_status='DIAGNOSTIC_STOPPED',original_reason='PROBE_HEADER_PRODUCTION_LIMIT',original_header_seconds=12.375,
        original_production_timing_suitable=False,admitted=False,reuse_review=None,venue_subject_sha256=filtered.digest(venue))
    context=dict(contract=c,parent=dict(blobs_b64={k:base64.b64encode(v).decode() for k,v in blobs.items()},pins={k:sha(v) for k,v in blobs.items()},custody_names=names),
        reuse_subject=reuse,reuse_subject_sha256=filtered.digest(reuse),venue_subject=venue,budget=deepcopy(filtered.POLICY),
        reuse_review=dict(version='ncaaf-diagnostic-original-reuse-review-v1',reviewer=legacy.owner.OWNER,reviewed_at='2026-10-17T17:45:00Z',
            reuse_subject_sha256=filtered.digest(reuse),venue_subject_sha256=filtered.digest(venue),answer='SYNTHETIC qualified exact reuse and venue review only.'))
    p=legacy.plan()['payload'];p.update(version=filtered.HISTORY_PLAN,filtered_history=context)
    p['requests']=[dict(id='filtered-stats-'+str(i+1),provider='cfbd',host=r['host'],endpoint=r['endpoint'],params=r['params'],max_attempts=1) for i,r in enumerate(filtered.requests(c))]
    p['advance_permissions'][0]['requests_sha256']=stage.model.digest(p['requests'])
    p['collection_authorization_ref']='SYNTHETIC-filtered-authority'
    stats=[];lookup={g['id']:g for g in games}
    for req in p['requests']:
        name=req['params']['team'];selected={r['id'] for r in c['selected_yardage'][name]}
        rows=[deepcopy(row) for batch in originals[1:] for row in batch if row['id'] in selected]
        for row in rows:
            g=lookup[row['id']]
            for team in row['teams']:team['team']=g['homeTeam'] if team['teamId']==g['homeId'] else g['awayTeam']
        extra=next(g for g in games if g['completed'] and name in (g['homeTeam'],g['awayTeam']) and g['week']==7)
        rows.append(dict(id=extra['id'],teams=[dict(teamId=extra[s+'Id'],team=extra[s+'Team'],points=extra[s+'Points'],stats=[dict(category='totalYards',stat='9999')]) for s in ('home','away')]))
        stats.append(encoded(rows))
    return stage.seal(p),stats


def configure(p,monkeypatch,root):
    a,raw=legacy.stage_configuration(p,monkeypatch,root);c=json.loads(raw)['payload']
    quote=stage.is_quote(p['payload'])
    a['payload']['authorized_at']='2026-10-17T18:20:00Z' if quote else '2026-10-17T17:50:00Z';a=stage.seal(a['payload'])
    c.update(version='ncaaf-fresh-filtered-controlled-bindings-v5' if filtered.is_fresh(p['payload']) else 'ncaaf-filtered-controlled-bindings-v4',authorization=a,
        runtime=target.runtime_receipt(c['runtime']['code_head'],'a'*64,filtered=True,fresh=filtered.is_fresh(p['payload'])))
    if stage.is_continuation(p['payload']):
        c['version']='ncaaf-continuation-controlled-bindings-v6'
        c['runtime']=target.runtime_receipt(c['runtime']['code_head'],'a'*64,filtered=True,fresh=True,continuation=True)
    c['review']['reviewed_at']='2026-10-17T18:25:00Z' if quote else '2026-10-17T17:55:00Z'
    c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
    raw=encoded(stage.seal(c))
    monkeypatch.setattr(stage,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{p['sha256']:dict(file_sha256=sha(raw),file_b64=base64.b64encode(raw).decode())})
    monkeypatch.setattr(filtered,'private_files',lambda paths:True) # Synthetic files only, never an OS custody assertion.
    return a,raw


class Clock:
    def __init__(self):self.elapsed=0.
    def mono(self):return self.elapsed
    def wall(self):return '2026-10-17T18:00:01Z'


class Response:
    def __init__(self,raw,clock,scenario):self.raw=raw;self.clock=clock;self.scenario=scenario;self.fp=self;self.status=scenario.get('status',200)
    def getheaders(self):return [('Content-Type','application/json'),('Content-Length',str(self.scenario.get('declared',len(self.raw))))]+self.scenario.get('headers',[])
    def read1(self,size):
        self.clock.elapsed+=self.scenario.get('body',.1);self.fp=None;return self.raw
    def close(self):self.fp=None


class Connection:
    def __init__(self,raw,clock,scenario):self.raw=raw;self.clock=clock;self.scenario=scenario;self.sock=self;self.closed=False;self.writes=0
    def connect(self):self.clock.elapsed+=self.scenario.get('connect',.1)
    def settimeout(self,value):pass
    def request(self,method,path,headers):
        assert method=='GET' and 'team=' in path and 'week=' not in path and headers['Authorization']=='Bearer '+SECRET
        self.clock.elapsed+=self.scenario.get('write',.1);self.writes+=1
    def getresponse(self):
        if self.scenario.get('cancel')=='interrupt':raise KeyboardInterrupt()
        if self.scenario.get('cancel'):raise OSError('SYNTHETIC arbitrary exception must never be retained')
        self.clock.elapsed+=self.scenario.get('header',12.375)
        return Response(self.raw,self.clock,self.scenario)
    def shutdown(self,*a):self.closed=True
    def close(self):self.closed=True


def capture(monkeypatch,root,*,scenarios=None,p=None,stats=None,cli=False):
    if p is None:p,stats=context(monkeypatch,root)
    a,control=configure(p,monkeypatch,root);clock=Clock();connections=[];scenarios=scenarios or [{} for _ in p['payload']['requests']]
    if cli:
        from scripts import ncaaf_staged_acquisition as command
    def connect(host,timeout):
        i=len(connections);assert host==filtered.HOST and i<len(p['payload']['requests']),'Unexpected request/retry'
        c=Connection(stats[i],clock,scenarios[i]);connections.append(c);return c
    monkeypatch.setattr(stage.transport.http.client,'HTTPSConnection',connect)
    if cli:
        paths={k:root/('SYNTHETIC-'+k+'.json') for k in ('plan','authorization','control')}
        paths['plan'].write_bytes(encoded(p));paths['authorization'].write_bytes(encoded(a));paths['control'].write_bytes(control)
        monkeypatch.setattr(command,'verify_runtime',lambda:None) # Synthetic transport seam only; source checks still execute.
        monkeypatch.setattr(command.sys.stdin,'isatty',lambda:True)
        monkeypatch.setattr(command.getpass,'getpass',lambda prompt:SECRET)
        monkeypatch.setattr(stage.transport,'utc',clock.wall)
        original_acquire=stage.acquire
        def acquire(*args,**kwargs):kwargs['monotonic']=clock.mono;return original_acquire(*args,**kwargs)
        monkeypatch.setattr(stage,'acquire',acquire)
        code=command.main(['execute',str(paths['plan']),'--authorization',str(paths['authorization']),
            '--controlled-bindings',str(paths['control']),'--bindings-sha256',sha(control),'--custody-root',str(root),'--transport-diagnostics'])
        b=json.loads((root/(p['sha256']+'.stage.json')).read_bytes())
        assert code==(0 if b['payload']['status']=='CAPTURED_UNREVIEWED' else 1)
        assert command.main(['readback',str(root/(p['sha256']+'.stage.json')),'--custody-root',str(root)])==0
    else:
        collector=observations.Collector();transport=stage.transport.HttpsTransport(lambda _:SECRET,diagnostics=collector,
            filtered_history_policy=filtered.RequestPolicy(filtered.policy_contract(p['payload'])))
        b=stage.acquire(p,a,root=root,transport=transport,wall=clock.wall,monotonic=clock.mono,diagnostics=collector)
    assert all(c.closed for c in connections)
    for path in root.iterdir():
        if path.is_file():assert SECRET.encode() not in path.read_bytes()
    return b,connections,control


def test_exact_plan_parent_and_metadata_preserve_scoring(monkeypatch,tmp_path):
    p,stats=context(monkeypatch,tmp_path);q=stage.validate_plan(p,executable=True)
    scoring,parent=filtered.parent(q,at=q['execution_window']['not_before'])
    assert parent['charged_active_seconds']==12.594 and parent['original_production_timing_suitable'] is False
    assert sum(len(v) for v in q['filtered_history']['contract']['expected_scoring'].values())==12
    assert any(g['startDate'].startswith('2026-10-14') for g in filtered.checked_rows(scoring['raw']))
    assert stage.planning(p)['network_requests']==0 and len(q['requests'])==2
    with pytest.raises(ValueError):target.validate_plan(p)


@pytest.mark.parametrize('cli',[False,True])
def test_actual_configuration_transport_custody_and_readback(monkeypatch,tmp_path,cli,capsys):
    b,connections,_=capture(monkeypatch,tmp_path,cli=cli);v=stage.read_bundle(b)
    assert v['status']=='CAPTURED_UNREVIEWED' and v['attempts']==2 and len(connections)==2
    assert v['active_seconds']==pytest.approx(25.35) and v['prior_active_seconds']==12.594
    p=v['plan']['payload'];scoring,parent=filtered.parent(p,at=v['started_at'])
    assert v['prior_original_bytes']==len(scoring['raw']) and v['active_seconds']+v['prior_active_seconds']<40
    assert stage.verify_journal(tmp_path,b)
    raw=(tmp_path/(v['plan']['sha256']+'.transport-diagnostics.json')).read_bytes()
    observations.read(tmp_path,b,raw,sha(raw))
    state,projection=filtered.project_bundle(v,at='2026-10-17T18:01:00Z')
    game=next(g for g in state['batches'][0]['records'] if not g['completed'])
    audit,rows,_=history.build_dataset(state,feature_targets=[game])
    assert rows[0]['home_prior_games']==rows[0]['away_prior_games']==6
    assert rows[0]['home_yards_games']==rows[0]['away_yards_games']==3
    assert rows[0]['home_yards_per_game']==rows[0]['away_yards_per_game']==400
    assert len(projection['selected_games'])==6
    objects=filtered.response_objects(b,at='2026-10-17T18:01:00Z')
    assert len(objects)==3 and objects[0]['payload']['metadata']['representation']=='complete_original_response_bytes'
    assert objects[1]['payload']['metadata']['request_scope']==p['requests'][0]['params']
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize('scenario,second,reason',[
    ({'header':28},{},'NCAAF_STAGED_DEADLINE'),
    ({},{'header':15},'NCAAF_STAGED_DEADLINE'),
    ({'connect':10.1},{},'NCAAF_PILOT_DEADLINE'),
    ({'write':10.1},{},'NCAAF_PILOT_DEADLINE'),
    ({'body':10.1},{},'NCAAF_PILOT_DEADLINE'),
    ({'declared':524289},{},'NCAAF_PILOT_BODY_LIMIT'),
    ({'declared':1},{},'NCAAF_PILOT_INCOMPLETE_BODY'),
    ({'cancel':True},{},'NCAAF_PILOT_TRANSPORT_FAILURE'),
    ({'cancel':'interrupt'},{},'NCAAF_STAGED_ACQUISITION_FAILURE'),
    ({'status':302},{},'NCAAF_PILOT_HTTP_STATUS'),
    ({'headers':[('X-Next-Page','2')]},{},'NCAAF_PILOT_REDIRECT_OR_PAGINATION')])
def test_first_failure_and_cumulative_budget_preserve_accounting(monkeypatch,tmp_path,scenario,second,reason):
    b,connections,_=capture(monkeypatch,tmp_path,scenarios=[scenario,second]);v=stage.read_bundle(b,complete=False)
    assert v['status']=='INCOMPLETE' and v['reason']==reason
    assert len(connections)==(2 if scenario=={} else 1) and sum(c.writes for c in connections)<=2
    assert v['prior_active_seconds']==12.594 and stage.verify_journal(tmp_path,b)
    with pytest.raises(ValueError):stage.read_bundle(b)


@pytest.mark.parametrize('attack',['missing','conflicting-score','wrong-team','week','duplicate','missing-yards'])
def test_selected_coverage_failure_keeps_complete_original_and_stops_second(monkeypatch,tmp_path,attack):
    p,stats=context(monkeypatch,tmp_path);rows=json.loads(stats[0])
    if attack=='missing':rows.pop(0)
    elif attack=='conflicting-score':rows[0]['teams'][0]['points']+=1
    elif attack=='wrong-team':rows[0]['teams'][0]['teamId']=99
    elif attack=='week':rows[0]['week']=9
    elif attack=='duplicate':rows.append(deepcopy(rows[0]))
    else:rows[0]['teams'][0]['stats']=[]
    stats[0]=encoded(rows);b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=stats)
    v=stage.read_bundle(b,complete=False)
    assert v['reason']=='NCAAF_STAGED_TARGET_CONFLICT' and len(connections)==1 and len(v['objects'])==1
    assert base64.b64decode(v['objects'][0]['body_b64'])==stats[0] and stage.verify_journal(tmp_path,b)
    sidecar=json.loads((tmp_path/(v['plan']['sha256']+'.transport-diagnostics.json')).read_bytes())['payload']
    assert sidecar['attempts'][0]['complete_original'] is True and sidecar['attempts'][0]['http_status']==200
    assert sidecar['attempts'][0]['reason'] is None and sidecar['reason']=='NCAAF_STAGED_TARGET_CONFLICT'


@pytest.mark.parametrize('field',['raw-original','journal','claim','approval','diagnostics'])
def test_parent_file_tamper_rejects_before_request(monkeypatch,tmp_path,field):
    p,stats=context(monkeypatch,tmp_path)
    key='original' if field=='raw-original' else field
    path=tmp_path/p['payload']['filtered_history']['parent']['custody_names'][key]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):capture(monkeypatch,tmp_path,p=p,stats=stats)
    assert not list(tmp_path.glob('*.attempts.jsonl'))


@pytest.mark.parametrize('attack',['reuse-subject','venue','selection','scope','header-budget','account-reset','review-clock','target-review'])
def test_resealed_plan_changes_reject(monkeypatch,tmp_path,attack):
    p,_=context(monkeypatch,tmp_path);v=p['payload'];c=v['filtered_history']
    if attack=='reuse-subject':c['reuse_review']['reuse_subject_sha256']='0'*64
    elif attack=='venue':c['venue_subject']['original_label']='Another stadium'
    elif attack=='selection':c['contract']['selected_yardage'][legacy.CANONICAL['home_school']].pop()
    elif attack=='scope':v['requests'][0]['params']['week']=3
    elif attack=='header-budget':c['budget']['new_yardage_active_seconds']=60
    elif attack=='account-reset':c['reuse_subject']['original_charged_active_seconds']=0
    elif attack=='review-clock':c['reuse_review']['reviewed_at']='2026-10-17T17:00:00Z'
    else:v['target_evidence']['owner_review']=None
    with pytest.raises((ValueError,TypeError)):stage.validate_plan(stage.seal(v),executable=True)


def test_spent_journal_and_parent_claim_prevent_duplicate_dispatch(monkeypatch,tmp_path):
    p,stats=context(monkeypatch,tmp_path);bundle,_,_=capture(monkeypatch,tmp_path,p=p,stats=stats)
    original=(tmp_path/(p['sha256']+'.attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='ALREADY_SPENT'):capture(monkeypatch,tmp_path,p=p,stats=stats)
    assert (tmp_path/(p['sha256']+'.attempts.jsonl')).read_bytes()==original
    other=deepcopy(p['payload']);other['collection_authorization_ref']+='-second';other=stage.seal(other)
    rejected,connections,_=capture(monkeypatch,tmp_path,p=other,stats=stats)
    assert rejected['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED'
    assert rejected['payload']['attempts']==0 and connections==[]
    assert (tmp_path/(p['sha256']+'.attempts.jsonl')).read_bytes()==original


def filtered_selection(synthetic,monkeypatch,root,line):
    """Full synthetic private packet with actual continuation/readers and caller."""
    from datetime import timedelta
    from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_private_refresh as refresh
    from app_core.ncaaf_compatible_observation import mapping_binding
    baseline=root/'baseline';baseline.mkdir();completed=root/'completed';completed.mkdir();discovered=root/'discovered';discovered.mkdir()
    p,stats=context(monkeypatch,completed)
    packet,row=legacy.fixture(synthetic,monkeypatch,baseline,line)
    b,_,_=capture(monkeypatch,completed,p=p,stats=stats)
    pref=evidence.prepare(b,model_packet=pair._observation(packet)['model'],prepared_at='2026-10-17T18:01:00Z')
    ident=legacy.identity(monkeypatch,discovered,pref);pref=legacy.review_prequote(pref,b)
    q=legacy.plan(target.QUOTE_PLAN,b,pref,ident)['payload']
    q.update(version='ncaaf-fresh-filtered-final-quote-stage-plan-v5' if filtered.is_fresh(p['payload']) else filtered.QUOTE_PLAN,filtered_history=deepcopy(p['payload']['filtered_history']))
    if stage.is_continuation(p['payload']):
        q.update(version='ncaaf-continuation-final-quote-stage-plan-v6',limits=deepcopy(p['payload']['limits']),continuation=deepcopy(p['payload']['continuation']))
    qp=stage.seal(q);authorization,control=configure(qp,monkeypatch,completed)
    clock=Clock();clock.wall=lambda:'2026-10-17T18:30:01Z';collector=observations.Collector()
    from test_ncaaf_pilot import Stream
    def quote_transport(req,limits,deadline,mono):
        assert req['provider']=='odds_api' and len(q['requests'])==1
        collector.enter('request_write');collector.wrote_request();collector.enter('response_headers');collector.observed_status(200)
        clock.elapsed+=.01;stream=Stream(legacy.odds(line));stream.diagnostics=collector;return stream
    quote_transport.diagnostics=collector
    quote=stage.acquire(qp,authorization,root=completed,transport=quote_transport,history_bundle=b,prequote=pref,
        wall=clock.wall,monotonic=clock.mono,diagnostics=collector)
    value=packet['payload'];c=value['custody_packet'];n=c['payload']['native_packet'];o=n['payload']['observation']['payload']
    value.update(version=owner.CONDITIONAL_INTEGER if float(line).is_integer() else owner.CONDITIONAL_HALF,
        staged_binding=evidence.join(b,pref,quote),alias_review=ident)
    event=dict(sport='NCAAF',provider_namespace='odds_api',provider_event_id=legacy.EID,canonical_event_id='cfbd:987654',schedule_game_id=987654,
        home_team=legacy.HOME,away_team=legacy.AWAY,start_utc=legacy.CANONICAL['start_utc'],neutral_site=False)
    o.update(event=event,schedule=pref['payload']['schedule'],crosswalk=ident['crosswalk'],model=pref['payload']['model'],
        features=dict(order=list(evidence.research.FEATURES),values=pref['payload']['ordered_features'],available_at=pref['payload']['prepared_at']))
    o['mapping_review']['binding']=mapping_binding(event,o['schedule'],o['crosswalk'])
    bs=evidence.batches(b);n['payload']['dependency_objects']=[dict(sha256=sha(encoded(batch)),bytes_b64=base64.b64encode(encoded(batch)).decode()) for batch in bs]
    o['feature_dependencies']=[dict(game_id=g['id'],team_id=tid,season=2026,start_utc=g['startDate'],available_at=batch['retrieved_at'],
        kind='scoring' if i==0 else 'yardage',source_sha256=n['payload']['dependency_objects'][i]['sha256'])
        for i,batch in enumerate(bs) for g in bs[0]['records'] if g['completed'] and (not i or g['week']==batch['request']['week'])
        for tid in (111,222) if tid in (g['homeId'],g['awayId'])]
    terms=o['source_review']['terms_review'];q=o['quote']
    terms.update(product=None,listing_id=None,rule_edition=None,rules_document_sha256=None)
    q.update(product=None,listing_id=None,period_source=owner.projection.CONDITIONAL_VERSION,rules_source=owner.projection.CONDITIONAL_VERSION)
    o['source_review']['offer_verification']['rule_edition']=None
    with owner.projection.selected(packet):
        obj=quote['payload']['objects'][0]
        c['payload']['response_objects']=filtered.response_objects(b,at=o['as_of'])+[
            custody.capture(base64.b64decode(obj['body_b64']),obj['metadata'],quote=q,terms=terms,owner_reviewed=True)]
    legacy.resign(packet)
    assert owner.load(packet)==packet
    away,_=legacy.other_side(packet,row)
    combined=pair.prepare(packet,away,prepared_at='2026-10-17T18:30:02Z')
    combined['payload']['owner_review']=dict(version=pair.REVIEW_VERSION,reviewer=owner.OWNER,reviewed_at='2026-10-17T18:30:04Z',
        subject_sha256=pair.subject(combined),qualifications='SYNTHETIC exact continuation and both-side mathematical review.',
        finding='SYNTHETIC original lineage, same market and both conditional research child reviews.',
        child_reviews=dict(home=packet['payload']['owner_review'],away=away['payload']['owner_review']))
    combined=stage.seal(combined['payload']);grant=legacy.inference_authorization(combined,monkeypatch,paired=True)
    import test_source_contract_pipeline as chronology_fixture
    monkeypatch.setattr(chronology_fixture,'NOW',legacy.NOW)
    monkeypatch.setattr(chronology_fixture,'CAPTURE',(legacy.NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:legacy.NOW)
    monkeypatch.setattr('app_core.prediction_evidence.now_utc',lambda:(legacy.NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.public_board.datetime',chronology_fixture.FrozenDateTime)
    return dict(version=refresh.VERSION,packet=combined,authorization=grant),completed,pref


@pytest.mark.parametrize('line',[-7,-3.5])
def test_filtered_native_private_refresh_storage_export_replay_and_display(synthetic,monkeypatch,tmp_path,line):
    from app_core import ncaaf_private_refresh as refresh
    selection,root,pref=filtered_selection(synthetic,monkeypatch,tmp_path,line)
    failure_codes=[];native=refresh.run
    def observed(value):
        try:return native(value)
        except Exception as exc:
            import re
            code=str(exc)
            failure_codes.append(code if re.fullmatch(r'(?:NCAAF|FILTERED)_[A-Z_0-9]{1,96}',code) else type(exc).__name__)
            raise
    monkeypatch.setattr(refresh,'run',observed)
    updates,warnings,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=selection))
    assert not warnings and not errors,(errors,failure_codes)
    assert updates['pipeline_status']=='PRIVATE_CONDITIONAL_RESEARCH'
    ranked=updates['ncaaf_private_refresh_ranked'];assert len(ranked)==2 and ranked[0]['cover']>=ranked[1]['cover']
    assert pref['payload']['history_row']['home_prior_games']==pref['payload']['history_row']['away_prior_games']==6
    assert pref['payload']['history_row']['home_yards_games']==pref['payload']['history_row']['away_yards_games']==3
    for row in ranked:
        assert row['cover']+row['push']+row['loss']==pytest.approx(1)
        assert row['push']>0 if float(line).is_integer() else row['push']==0
        assert row['estimated_ev_per_unit']==pytest.approx(row['cover']*(1+100/110)+row['push']-1)
        assert row['applicability']=='UNVERIFIED_CONDITIONAL_RESEARCH' and row['calibration']=='UNVALIDATED' and row['stake']==0
    result=updates['ncaaf_private_refresh_result'];raw=Path(result['path']).read_bytes()
    assert refresh.read_private_export(raw,root=root,database=root/'ncaaf-private-refresh.sqlite')==ranked
    assert list(root.glob('*.inference-attempts.jsonl'))



def test_filtered_receipt_cutoff_is_strict(monkeypatch,tmp_path):
    p,_=context(monkeypatch,tmp_path);a,_=configure(p,monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='AUTHORIZATION_CLOCK'):
        stage.authorize(p,a,p['payload']['execution_window']['not_after'])


@pytest.mark.parametrize('late',[False,True])
def test_private_prompt_time_excluded_but_last_minute_stops_dispatch(monkeypatch,tmp_path,late):
    from scripts import ncaaf_staged_acquisition as command
    p,stats=context(monkeypatch,tmp_path);a,control=configure(p,monkeypatch,tmp_path)
    files={k:tmp_path/('SYNTHETIC-private-'+k+'.json') for k in ('plan','authorization','control')}
    for k,value in [('plan',encoded(p)),('authorization',encoded(a)),('control',control)]:files[k].write_bytes(value)
    clock=Clock();wall=['2026-10-17T18:00:01Z'];connections=[]
    def prompt(_):
        clock.elapsed+=600 # Explicit synthetic private input delay; never acquisition time.
        wall[0]='2026-10-17T18:14:00Z' if late else '2026-10-17T18:10:01Z'
        return SECRET
    def connect(host,timeout):
        conn=Connection(stats[len(connections)],clock,{})
        connections.append(conn);return conn
    monkeypatch.setattr(command,'verify_runtime',lambda:None)
    monkeypatch.setattr(command.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(command.getpass,'getpass',prompt)
    monkeypatch.setattr(stage.transport,'utc',lambda:wall[0])
    monkeypatch.setattr(stage.transport.http.client,'HTTPSConnection',connect)
    native=stage.acquire
    def acquire(*args,**kwargs):kwargs['monotonic']=clock.mono;return native(*args,**kwargs)
    monkeypatch.setattr(stage,'acquire',acquire)
    code=command.main(['execute',str(files['plan']),'--authorization',str(files['authorization']),
        '--controlled-bindings',str(files['control']),'--bindings-sha256',sha(control),'--custody-root',str(tmp_path),'--transport-diagnostics'])
    assert code==(1 if late else 0)
    if late:
        assert not connections and not list(tmp_path.glob('*.attempts.jsonl'))
    else:
        b=stage.read_bundle(json.loads((tmp_path/(p['sha256']+'.stage.json')).read_bytes()))
        assert b['active_seconds']==pytest.approx(25.35) and len(connections)==2
    assert all(SECRET.encode() not in path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
