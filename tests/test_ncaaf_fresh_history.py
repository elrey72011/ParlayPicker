"""Network-blocked fresh source integration; never authentic inputs or authority."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from app_core import ncaaf_fresh_history as fresh, ncaaf_filtered_history as filtered
from app_core import ncaaf_staged_acquisition as stage, ncaaf_history as history
from app_core import ncaaf_transport_diagnostics as diagnostics
import test_ncaaf_filtered_history as continuation
import ncaaf_reviewed_target_fixtures as legacy
from test_ncaaf_model_compatibility import synthetic


def context(monkeypatch, root):
    p=legacy.plan()['payload']; p['version']=fresh.HISTORY_PLAN
    p['filtered_history']=dict(version=fresh.SOURCE,season=2026,conference='ACC',home_team=legacy.CANONICAL['home_school'],
        away_team=legacy.CANONICAL['away_school'],budget=deepcopy(fresh.POLICY))
    bodies=[json.loads(raw) for raw in legacy.bodies()]; games=bodies[0]
    for g in games:g['venueId']=700
    for tid,name in ((111,legacy.CANONICAL['home_school']),(222,legacy.CANONICAL['away_school'])):
        for week in (8,9,10):
            games.append(dict(games[0],id=tid*100+week,week=week,startDate=f'2026-09-{week-7:02}T17:00:00Z',completed=True,
                homeId=tid,homeTeam=name,homePoints=23,awayId=9500+tid,awayTeam='SYNTHETIC FCS '+name,awayPoints=10,
                awayClassification='fcs'))
    p['filtered_history']['scoring_histories']={games[0][side+'Team']:[dict(calendar_date=g['startDate'][:10],calendar_zone='UTC',
        opponent=g['awayTeam'] if g['homeId']==games[0][side+'Id'] else g['homeTeam'],source_sha256=p['target_evidence']['references'][0]['sha256'])
        for g in games if g['completed'] and games[0][side+'Id'] in (g['homeId'],g['awayId'])] for side in ('home','away')}
    p['requests']=fresh.requests(p['filtered_history'])
    p['advance_permissions'][0]['requests_sha256']=stage.model.digest(p['requests'])
    lookup={g['id']:g for g in games}; stats=[]
    for side in ('home','away'):
        tid=games[0][side+'Id']; rows=[]
        for batch in bodies[1:]:
            for row in batch:
                if not any(t['teamId']==tid for t in row['teams']):continue
                row=deepcopy(row);g=lookup[row['id']]
                for team in row['teams']:team['team']=g['homeTeam'] if team['teamId']==g['homeId'] else g['awayTeam']
                rows.append(row)
        extra=next(g for g in games if g.get('homeId')==tid and g['week']==8)
        rows.append(dict(id=extra['id'],teams=[dict(teamId=extra[s+'Id'],team=extra[s+'Team'],points=extra[s+'Points'],stats=[dict(category='totalYards',stat='9999')]) for s in ('home','away')]))
        stats.append(stage.model.encode(rows))
    return stage.seal(p),[stage.model.encode(games)]+stats


class Connection(continuation.Connection):
    def request(self,method,path,headers):
        assert method=='GET' and 'week=' not in path and ('conference=ACC' in path or 'conference=CUSA' in path or 'team=' in path)
        assert headers['Authorization']=='Bearer '+continuation.SECRET
        self.clock.elapsed+=self.scenario.get('write',.1);self.writes+=1


def capture(monkeypatch,root,**kwargs):
    monkeypatch.setattr(continuation,'Connection',Connection)
    if kwargs.get('p') is None:kwargs['p'],kwargs['stats']=context(monkeypatch,root)
    return continuation.capture(monkeypatch,root,**kwargs)


@pytest.mark.parametrize('conference,second_target',[('ACC',False),('CUSA',True)])
@pytest.mark.parametrize('cli',[False,True])
def test_fresh_originals_actual_cli_projection_and_diagnostic_readback(monkeypatch,tmp_path,conference,second_target,cli):
    legacy.select(monkeypatch,int(second_target));p,bodies=context(monkeypatch,tmp_path)
    p['payload']['filtered_history']['conference']=conference
    p['payload']['requests']=fresh.requests(p['payload']['filtered_history'])
    p['payload']['advance_permissions'][0]['requests_sha256']=stage.model.digest(p['payload']['requests']);p=stage.seal(p['payload'])
    b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=bodies,cli=cli)
    v=stage.read_bundle(b);assert v['version']==fresh.HISTORY_BUNDLE
    assert v['attempts']==len(connections)==len(v['objects'])==3
    assert v['prior_active_seconds']==v['prior_original_bytes']==0 and v['active_seconds']==pytest.approx(38.025)
    assert not any('parent' in k for k in p['payload']['filtered_history'])
    assert not list(tmp_path.glob('SYNTHETIC-parent-*'))
    for o,raw,request in zip(v['objects'],bodies,p['payload']['requests']):
        assert base64.b64decode(o['body_b64'])==raw and o['metadata']['request_scope']==request['params']
    assert stage.verify_journal(tmp_path,b)
    sidecar=(tmp_path/(p['sha256']+'.transport-diagnostics.json')).read_bytes()
    diagnostics.read(tmp_path,b,sidecar,filtered.sha(sidecar))
    state,projection=filtered.project_bundle(v,at='2026-10-17T18:01:00Z')
    _,rows,_=history.build_dataset(state,feature_targets=[state['batches'][0]['records'][0]])
    assert rows[0]['home_prior_games']==rows[0]['away_prior_games']==6
    assert rows[0]['home_yards_games']==rows[0]['away_yards_games']==3
    assert rows[0]['home_yards_per_game']==rows[0]['away_yards_per_game']==400
    assert len(projection['selected_games'])==6 and len(filtered.response_objects(b,at='2026-10-17T18:01:00Z'))==3
    assert all(continuation.SECRET.encode() not in f.read_bytes() for f in tmp_path.iterdir() if f.is_file())


@pytest.mark.parametrize('attack',['orientation','start','neutral','venue','identity','duplicate','missing-scoring','missing-selected','missing-unselected-scoring','not-completed'])
def test_scoring_conflict_stops_before_statistics_preserves_original(monkeypatch,tmp_path,attack):
    p,bodies=context(monkeypatch,tmp_path);rows=json.loads(bodies[0]);g=rows[0]
    if attack=='orientation':g['homeTeam'],g['awayTeam']=g['awayTeam'],g['homeTeam']
    elif attack=='start':g['startDate']='2026-10-21T23:30:00Z'
    elif attack=='neutral':g['neutralSite']=not g['neutralSite']
    elif attack=='venue':g['venue']='SYNTHETIC unreviewed alias'
    elif attack=='identity':g['homeId']=g['awayId']
    elif attack=='duplicate':rows.append(deepcopy(g))
    elif attack=='missing-scoring':rows[:]=[r for r in rows if r.get('homeId')!=111 or not r['completed']]
    elif attack=='missing-selected':rows.pop(1)
    elif attack=='missing-unselected-scoring':rows.pop(-1)
    else:rows[1]['completed']=False
    bodies[0]=stage.model.encode(rows)
    b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=bodies)
    v=stage.read_bundle(b,complete=False)
    assert v['status']=='INCOMPLETE' and v['reason']=='NCAAF_STAGED_TARGET_CONFLICT'
    assert len(connections)==v['attempts']==len(v['objects'])==1
    assert base64.b64decode(v['objects'][0]['body_b64'])==bodies[0] and stage.verify_journal(tmp_path,b)


@pytest.mark.parametrize('scenarios,attempts,reason',[
    ([{'header':30.1},{},{}],1,'NCAAF_PILOT_DEADLINE'),
    ([{'header':20},{'header':20},{}],2,'NCAAF_STAGED_DEADLINE'),
    ([{},{},{'header':15}],3,'NCAAF_STAGED_DEADLINE'),
    ([{'connect':10.1},{},{}],1,'NCAAF_PILOT_DEADLINE'),
    ([{'body':10.1},{},{}],1,'NCAAF_PILOT_DEADLINE'),
    ([{'declared':1},{},{}],1,'NCAAF_PILOT_INCOMPLETE_BODY'),
    ([{'cancel':'interrupt'},{},{}],1,'NCAAF_STAGED_ACQUISITION_FAILURE'),
])
def test_fresh_budget_cancellation_and_first_failure(monkeypatch,tmp_path,scenarios,attempts,reason):
    b,connections,_=capture(monkeypatch,tmp_path,scenarios=scenarios)
    v=stage.read_bundle(b,complete=False)
    assert v['status']=='INCOMPLETE' and v['attempts']==len(connections)==attempts and v['reason']==reason
    assert v['prior_active_seconds']==v['prior_original_bytes']==0 and stage.verify_journal(tmp_path,b)
    assert all(c.closed for c in connections)


@pytest.mark.parametrize('line',[-7,-3.5])
def test_fresh_histories_review_quote_refresh_storage_export_replay(synthetic,monkeypatch,tmp_path,line):
    from app_core import ncaaf_private_refresh as refresh
    # Reuse the actual existing private caller fixture; only its source route changes.
    monkeypatch.setattr(continuation,'context',context)
    monkeypatch.setattr(continuation,'Connection',Connection)
    selection,root,pref=continuation.filtered_selection(synthetic,monkeypatch,tmp_path,line)
    updates,warnings,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=selection))
    assert not warnings and not errors
    assert updates['pipeline_status']=='PRIVATE_CONDITIONAL_RESEARCH'
    rows=updates['ncaaf_private_refresh_ranked']
    assert len(rows)==2 and rows[0]['cover']>=rows[1]['cover']
    assert pref['payload']['version']==fresh.PREQUOTE
    for row in rows:
        assert row['cover']+row['push']+row['loss']==pytest.approx(1)
        assert row['push']>0 if float(line).is_integer() else row['push']==0
        assert row['estimated_ev_per_unit']==pytest.approx(row['cover']*(1+100/110)+row['push']-1)
        assert row['calibration']=='UNVALIDATED' and row['stake']==0
    # The public display performs one fresh pair validation and returns that
    # exact comparison. Repeated calls and warm mutations still revalidate.
    from app_core import ncaaf_spread_pair as pair
    calls=[];native=pair.inspect_row
    def observed(*args,**kwargs):
        calls.append(1);return native(*args,**kwargs)
    monkeypatch.setattr(pair,'inspect_row',observed)
    source=updates['analysis_df'].iloc[0].to_dict()
    loads=[];native_load=pair.load
    def load_once(*args,**kwargs):
        loads.append(1);return native_load(*args,**kwargs)
    monkeypatch.setattr(pair,'load',load_once)
    displayed=pair.private_display(source);assert len(calls)==1 and len(loads)==1
    assert displayed['comparison']==native(json.loads(source['ml_estimate_metadata'])['ncaaf_inputs']['payload']['original_packet'],json.loads(source['ml_estimate_metadata'])['ncaaf_inputs']['payload'])
    loads.clear()
    pair.private_display(source);assert len(calls)==2 and len(loads)==1
    changed=deepcopy(source);metadata=json.loads(changed['ml_estimate_metadata'])
    metadata['ncaaf_inputs']['payload']['outcome_probabilities']['cover']=0.
    metadata['ncaaf_inputs']=stage.seal(metadata['ncaaf_inputs']['payload'])
    changed['ml_estimate_metadata']=stage.model.encode(metadata).decode()
    with pytest.raises(ValueError):pair.private_display(changed)
    raw=Path(updates['ncaaf_private_refresh_result']['path']).read_bytes()
    assert refresh.read_private_export(raw,root=root,database=root/'ncaaf-private-refresh.sqlite')==rows
    assert list(root.glob('*.inference-attempts.jsonl'))


def test_fresh_route_through_actual_normal_refresh_button(synthetic,monkeypatch,tmp_path):
    import test_ncaaf_private_normal_refresh as normal
    monkeypatch.setattr(continuation,'context',context)
    monkeypatch.setattr(continuation,'Connection',Connection)
    selection,root,_=continuation.filtered_selection(synthetic,monkeypatch,tmp_path,-7)
    monkeypatch.setattr(normal,'selection',lambda *args,**kwargs:selection)
    # Execute the existing actual button, display, no-implicit-run and rerun
    # assertions unchanged, with its approved 180-second harness budget.
    normal.test_actual_normal_refresh_button_uses_private_caller(synthetic,monkeypatch,root)


def test_fresh_readback_keeps_actual_age_and_rejects_substituted_queries(monkeypatch,tmp_path):
    b,_,_=capture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='STALE|CLOCK'):
        filtered.project_bundle(b['payload'],at='2026-10-18T18:01:00Z')
    altered=deepcopy(b);altered['payload']['objects'][0]['metadata']['request_scope'].pop('conference')
    altered=stage.seal(altered['payload'])
    with pytest.raises(ValueError):stage.read_bundle(altered)
    assert b['payload']['objects'][0]['metadata']['request_scope']['conference']=='ACC'


def test_fresh_source_mutation_and_duplicate_dispatch_reject(monkeypatch,tmp_path):
    p,bodies=context(monkeypatch,tmp_path)
    policy=filtered.RequestPolicy(p['payload']['filtered_history'])
    policy._contract['conference']='CUSA'
    with pytest.raises(ValueError,match='MUTATION'):
        policy.check(p['payload']['requests'][0],dict(connect_seconds=10,read_seconds=10))
    bundle,_,_=capture(monkeypatch,tmp_path,p=p,stats=bodies)
    before=(tmp_path/(p['sha256']+'.attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='ALREADY_SPENT'):
        capture(monkeypatch,tmp_path,p=p,stats=bodies)
    assert (tmp_path/(p['sha256']+'.attempts.jsonl')).read_bytes()==before
    other=deepcopy(p['payload']);other['collection_authorization_ref']+='-second';other=stage.seal(other)
    stopped,connections,_=capture(monkeypatch,tmp_path,p=other,stats=bodies)
    assert stopped['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED'
    assert stopped['payload']['attempts']==0 and not connections and stage.verify_journal(tmp_path,bundle)


def test_fresh_runtime_does_not_redefine_older_filtered_receipts():
    from app_core import ncaaf_reviewed_target as target
    old=target.runtime_receipt('0'*40,'a'*64,filtered=True)
    new=target.runtime_receipt('0'*40,'a'*64,filtered=True,fresh=True)
    assert 'app_core/ncaaf_fresh_history.py' not in old['sources']
    assert set(new['sources'])==set(old['sources'])|{'app_core/ncaaf_fresh_history.py'}
    target.check_runtime_receipt(old,filtered=True)
    target.check_runtime_receipt(new,filtered=True,fresh=True)
    with pytest.raises(ValueError):target.check_runtime_receipt(old,filtered=True,fresh=True)
    with pytest.raises(ValueError):target.check_runtime_receipt(new,filtered=True)


@pytest.mark.parametrize('old_runtime',[False,True])
def test_fresh_native_operator_preflight_requires_exact_versioned_runtime(monkeypatch,tmp_path,old_runtime):
    monkeypatch.setattr(sys,'path',list(sys.path)) # Native bootstrap owns only this test's copy.
    from scripts import ncaaf_reviewed_target_operator as operator
    from app_core import ncaaf_reviewed_target as target
    p,_=context(monkeypatch,tmp_path)
    authorization,raw=continuation.configure(p,monkeypatch,tmp_path)
    paths={k:str(tmp_path/('SYNTHETIC-'+k+'.json')) for k in ('plan','authorization','configuration')}
    Path(paths['plan']).write_bytes(stage.model.encode(p))
    Path(paths['authorization']).write_bytes(stage.model.encode(authorization))
    m=dict(version=operator.VERSION,**paths,custody_root=str(tmp_path),setup_cutoff='2026-10-17T17:56:00Z',
        latest_start='2026-10-17T18:13:00Z',history=None,prequote=None)
    c=json.loads(raw)['payload']
    m['runtime']=target.runtime_receipt(c['runtime']['code_head'],stage.model.digest(m),filtered=True,fresh=not old_runtime)
    c['runtime']=deepcopy(m['runtime']);c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
    raw=stage.model.encode(stage.seal(c));Path(paths['configuration']).write_bytes(raw)
    manifest=stage.model.encode(stage.seal(m));path=tmp_path/'SYNTHETIC-operation.json';path.write_bytes(manifest)
    if old_runtime:
        with pytest.raises(ValueError):operator.preflight(path,filtered.sha(manifest),filtered.sha(raw),at='2026-10-17T18:00:01Z')
    else:
        actual,plan,control,root=operator.preflight(path,filtered.sha(manifest),filtered.sha(raw),at='2026-10-17T18:00:01Z')
        assert actual==m and plan==p and control==raw and root==tmp_path
    assert not list(tmp_path.glob('*.attempts.jsonl')) and not list(tmp_path.glob('*.operator-claim.json'))


def test_derived_projection_cannot_authorize_transport_or_relabel_diagnostic_parent(monkeypatch,tmp_path):
    root=tmp_path/'SYNTHETIC-old';root.mkdir()
    p,_=continuation.context(monkeypatch,root)
    p['payload']['filtered_history']['contract']['version']=fresh.CONTRACT
    with pytest.raises(ValueError,match='CONTRACT_SCHEMA'):
        filtered.RequestPolicy(p['payload']['filtered_history']['contract'])
    with pytest.raises(ValueError,match='CONTRACT_SCHEMA'):
        filtered.parent(p['payload'],at=p['payload']['execution_window']['not_before'])


def test_week_zero_scoring_is_retained_without_entering_selected_yardage(monkeypatch,tmp_path):
    p,bodies=context(monkeypatch,tmp_path);games=json.loads(bodies[0])
    row=next(g for g in games if g['completed'] and g['week']==8 and g['homeId']==111)
    row['week']=0;bodies[0]=stage.model.encode(games)
    bundle,_,_=capture(monkeypatch,tmp_path,p=p,stats=bodies)
    state,projection=filtered.project_bundle(bundle['payload'],at='2026-10-17T18:01:00Z')
    assert any(g['id']==row['id'] and g['week']==0 for g in state['batches'][0]['records'])
    assert row['id'] not in projection['selected_games']
    _,features,_=history.build_dataset(state,feature_targets=[state['batches'][0]['records'][0]])
    assert features[0]['home_prior_games']==6 and features[0]['home_yards_games']==3


def test_public_validation_results_are_owned_and_warm_authorization_still_rejects_mutation(monkeypatch,tmp_path):
    from app_core import ncaaf_reviewed_target as target
    p,_=context(monkeypatch,tmp_path);before=deepcopy(p)
    view=fresh.validate_plan(p,executable=True)
    view['filtered_history']['scoring_histories'][view['target']['home_school']][0]['opponent']='SYNTHETIC changed copy'
    manifest=target.validate_target(p['payload']['target_evidence'],p['payload']['target'],p['payload']['execution_window']['not_before'],label='SYNTHETIC')
    manifest['references'][0]['excerpt']='SYNTHETIC changed manifest copy'
    source=fresh.validate_source(p['payload']['filtered_history']);source['budget']['history_active_seconds']=1
    assert p==before
    a,_=continuation.configure(p,monkeypatch,tmp_path)
    authorized=stage.authorize(p,a,'2026-10-17T18:00:01Z')
    authorized['target']['venue']='SYNTHETIC changed authorized copy'
    assert p==before
    p['payload']['filtered_history']['budget']['history_active_seconds']=41
    with pytest.raises(ValueError):stage.authorize(p,a,'2026-10-17T18:00:01Z')
    p=stage.seal(p['payload'])
    with pytest.raises(ValueError):stage.authorize(p,a,'2026-10-17T18:00:01Z')


def test_credential_normalization_preserves_unknown_keys_and_rechecks_values_after_warm_use():
    import re
    from app_core import ncaaf_response_custody as custody
    keys=list(custody._SCHEMA_KEYS)+['API-KEY','api_key','SET-cookie','İ_key','AπI-key','SYNTHETIC private key','',*map(chr,range(128))]
    before=dict(custody._SCHEMA_KEYS)
    for key in keys:assert custody._credential_key(key)==re.sub('[^a-z]','',key.lower())
    value={'version':'SYNTHETIC safe', 'payload':[{'calendar_date':'SYNTHETIC safe'}]}
    custody._credentials(value)
    value['version']='Bearer SYNTHETIC sensitive value'
    with pytest.raises(ValueError,match='CREDENTIAL_FIELD'):custody._credentials(value)
    value['version']='SYNTHETIC safe';value['payload'][0]['API-KEY']='SYNTHETIC excluded'
    with pytest.raises(ValueError,match='CREDENTIAL_FIELD'):custody._credentials(value)
    assert dict(custody._SCHEMA_KEYS)==before
    with pytest.raises(TypeError):custody._SCHEMA_KEYS['SYNTHETIC mutation']='mutation'


def test_owned_bundle_snapshot_checks_every_receipt_clock_and_rejects_late_resealed_evidence(monkeypatch,tmp_path):
    bundle,_,_=capture(monkeypatch,tmp_path);before=deepcopy(bundle)
    checked=[];native=stage._authorization
    def observed(p,plan_sha256,authorization,at,*,trusted):
        checked.append(at)
        return native(p,plan_sha256,authorization,at,trusted=trusted)
    monkeypatch.setattr(stage,'_authorization',observed)
    read=stage.read_bundle(bundle)
    assert len(checked)==1+2*len(bundle['payload']['objects'])
    read['plan']['payload']['target']['venue']='SYNTHETIC changed returned snapshot'
    read['objects'][0]['metadata']['request_scope']['conference']='SYNTHETIC changed returned snapshot'
    assert bundle==before
    altered=deepcopy(bundle['payload'])
    altered['objects'][-1]['metadata']['received_at']=altered['plan']['payload']['execution_window']['not_after']
    with pytest.raises(ValueError,match='NCAAF_STAGED_AUTHORIZATION_CLOCK'):
        stage.read_bundle(stage.seal(altered))


def test_projection_reuse_is_owned_by_one_verification_and_warm_mutation_rejects(synthetic,monkeypatch,tmp_path):
    from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_response_custody as custody
    monkeypatch.setattr(continuation,'context',context)
    monkeypatch.setattr(continuation,'Connection',Connection)
    value,root,_=continuation.filtered_selection(synthetic,monkeypatch,tmp_path,-7)
    child=pair.children(value['packet'])['home'];policy=owner.ReviewPolicy(child)
    packet=child['payload']['custody_packet']
    native=filtered._response_projection;calls=[]
    def observed(*args,**kwargs):
        calls.append(1);return native(*args,**kwargs)
    monkeypatch.setattr(filtered,'_response_projection',observed)
    owner_load=owner.load;loads=[]
    def counted_load(*args,**kwargs):
        loads.append(1);return owner_load(*args,**kwargs)
    monkeypatch.setattr(owner,'load',counted_load)
    incoming=owner.projection.SELECTED.get()
    owner.accepted_dependencies(child,None,'2026-10-17T18:30:10Z')
    assert len(loads)==1 and owner.projection.SELECTED.get()==incoming
    owner.accepted_dependencies(child,None,'2026-10-17T18:30:10Z')
    assert len(loads)==2 and owner.projection.SELECTED.get()==incoming
    calls.clear()
    assert owner.accepted_dependencies.__name__=='accepted_dependencies'
    for failure in (RuntimeError,KeyboardInterrupt):
        def interrupt(*args,**kwargs):
            assert owner.projection.SELECTED.get() is not None
            raise failure('SYNTHETIC scoped interruption')
        wrapped=owner._with_scope(lambda *args,**kwargs:None,interrupt)
        with pytest.raises(failure):wrapped(child)
        assert owner.projection.SELECTED.get()==incoming
    # The legacy unwrapped API retains its private-mode rejection stage.
    with pytest.raises(ValueError,match='PRIVATE_RESEARCH_NOT_SELECTED'):
        owner.infer.__wrapped__({},'2026-10-17T18:30:10Z')
    calls.clear() # Interruption preflight owns its separate projection work.
    with owner.projection.selected(child):
        custody.verify(packet,policy.approval,'2026-10-17T18:30:10Z',review_policy=policy)
        assert len(calls)==1
        custody.verify(packet,policy.approval,'2026-10-17T18:30:10Z',review_policy=policy)
        assert len(calls)==2 # No cross-operation projection or acceptance reuse.
        item=packet['payload']['response_objects'][0]
        item['payload']['projection']['original_index']=2
        item.update(stage.seal(item['payload']))
        packet.update(stage.seal(packet['payload']))
        with pytest.raises(ValueError,match='PROJECTION_BINDING'):
            custody.verify(packet,policy.approval,'2026-10-17T18:30:10Z',review_policy=policy)
    with pytest.raises(ValueError):owner.accepted_dependencies(child,None,'2026-10-17T18:30:10Z')
    assert owner.projection.SELECTED.get()==incoming
