"""Synthetic-only continuation: real transport/CLI/custody and private caller."""
import base64
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import pytest
from app_core import ncaaf_history_continuation as continuation
from app_core import ncaaf_staged_acquisition as stage, ncaaf_filtered_history as filtered
from app_core import ncaaf_fresh_history as fresh, ncaaf_transport_diagnostics as diagnostics
from app_core import ncaaf_reviewed_target as target, ncaaf_history as history
import test_ncaaf_fresh_history as ordinary
import test_ncaaf_filtered_history as fixtures
import ncaaf_reviewed_target_fixtures as legacy
from test_ncaaf_model_compatibility import synthetic

encode=stage.model.encode
sha=filtered.sha


def context(monkeypatch, root):
    """Produce a spent three-attempt V5 capture; never edit its originals."""
    p, bodies = ordinary.context(monkeypatch, root)
    native_configure = fixtures.configure
    operation = {}
    def old_configure(plan, patch, custody):
        a, raw = native_configure(plan, patch, custody); c=json.loads(raw)['payload']
        m=dict(version='ncaaf-reviewed-target-operation-v1',plan=str(root/'SYNTHETIC-retained-plan.json'),
            authorization=str(root/'SYNTHETIC-retained-authorization.json'),configuration=str(root/'SYNTHETIC-retained-configuration.json'),
            custody_root=str(root),setup_cutoff='2026-10-17T17:59:59Z',latest_start='2026-10-17T18:10:00Z',history=None,prequote=None)
        c['runtime']['operation_manifest_sha256']=stage.model.digest(m)
        m['runtime']=deepcopy(c['runtime']);operation.update(stage.seal(m))
        c['review']['reviewed_at']=a['payload']['authorized_at']
        c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
        raw=encode(stage.seal(c))
        patch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{plan['sha256']:dict(file_sha256=sha(raw),file_b64=base64.b64encode(raw).decode())})
        return a,raw
    with monkeypatch.context() as original:
        original.setattr(fixtures,'configure',old_configure)
        original.setattr(fixtures,'Connection',ordinary.Connection)
        parent, _, control = BASE_CAPTURE(original,root,p=p,stats=bodies,scenarios=[{},{},{'header':15}])
    b=stage.read_bundle(parent,complete=False)
    assert b['status']=='INCOMPLETE' and b['attempts']==3 and len(b['objects'])==2
    blobs=dict(bundle=encode(parent),journal=base64.b64decode(b['journal_b64']),
        plan=encode(p),stage_claim=(root/fresh.claim_name(p['payload'])).read_bytes(),
        diagnostics=(root/(p['sha256']+'.transport-diagnostics.json')).read_bytes(),configuration=control,
        authorization=encode(b['authorization']),operation=encode(operation),derivation=encode(dict(label='SYNTHETIC derivation only')),
        template=encode(dict(label='SYNTHETIC separate derivation template')),
        release=encode(dict(status='RELEASE_CLEARANCE_PASS',main=json.loads(control)['payload']['runtime']['code_head'],
            checks=dict(expected=23,passed=23),artifacts=dict(expected=20,verified=20),
            application_outcomes=dict(native_reconciliation=dict(status='PASS')))))
    method=dict(native_operation_file_sha256=sha(blobs['operation']),derivation_sha256=sha(blobs['derivation']),release_clearance_receipt_sha256=sha(blobs['release']))
    blobs['method']=encode(method)
    blobs['approval']=encode(dict(package_sha256=sha(blobs['method']),derivation_template_sha256=sha(blobs['template']),
        owner=target.OWNER,decision='APPROVED_CONSOLIDATED_FRESH_HISTORY_CAPTURE',received_at='2026-10-17T17:50:00Z'))
    claim=dict(version='ncaaf-reviewed-target-operation-v1',configuration_file_sha256=sha(control),
        manifest_file_sha256=sha(blobs['operation']),plan_sha256=p['sha256'])
    blobs['claim']=encode(claim)
    blobs['application_readback']=encode(dict(status='INDEPENDENT_READBACK_PASS',configuration_file_sha256=sha(control),
        plan_sha256=p['sha256'],package_sha256=sha(blobs['method']),actual_approval_at='2026-10-17T17:50:00Z',at='2026-10-17T17:59:00Z'))
    blobs['native_readback']=encode(dict(version='actual-exclusive-history-application-v1',configuration_applied=True,claims_created=0,
        provider_requests=0,inference=False,approval_file_sha256=sha(blobs['approval']),package_sha256=sha(blobs['method']),
        plan_sha256=p['sha256'],authorization_sha256=b['authorization']['sha256'],configuration_file_sha256=sha(control),
        configuration_subject_sha256=json.loads(control)['payload']['review']['subject_sha256'],
        actual_approval_at='2026-10-17T17:50:00Z',at='2026-10-17T17:59:00Z',
        files={k:dict(path=str(root/('SYNTHETIC-retained-'+k+'.json')),sha256=sha(blobs[k])) for k in ('plan','authorization','configuration','operation')}))
    report=dict(claim,claim_sha256=sha(blobs['claim']),bundle_file_sha256=sha(blobs['bundle']),bundle_sha256=parent['sha256'],
        journal_sha256=sha(blobs['journal']),diagnostics_file_sha256=sha(blobs['diagnostics']),provider_attempts=3,status=b['status'],
        reason=b['reason'],retry=False,readback='PASS',finished_at='2026-10-17T18:00:01.500000Z')
    blobs['report']=encode(report)
    value=deepcopy(p['payload'])
    value.update(version=continuation.HISTORY_PLAN,requests=[deepcopy(value['requests'][2])],
        limits=dict(stage.LIMITS,shared_active_seconds=120),collection_authorization_ref='SYNTHETIC-new-continuation-authority')
    value['execution_window']['not_before']='2026-10-17T18:00:03Z'
    value['advance_permissions'][0]['requests_sha256']=stage.model.digest(value['requests'])
    material=dict(blobs_b64={k:base64.b64encode(v).decode() for k,v in blobs.items()},pins={k:sha(v) for k,v in blobs.items()},
        custody_names={k:'SYNTHETIC-retained-'+k+'.json' for k in blobs})
    material['custody_names']['stage_claim']=fresh.claim_name(p['payload'])
    for key,name in material['custody_names'].items():
        path=root/name
        if path.exists():assert path.read_bytes()==blobs[key]
        else:path.write_bytes(blobs[key])
    value['continuation']=dict(version='ncaaf-spent-history-parent-v6',parent=material,budget=deepcopy(continuation.POLICY),review={})
    value['continuation']['review']=dict(reviewer=target.OWNER,reviewed_at='2026-10-17T18:00:02Z',subject_sha256=continuation.subject(value),
        finding='SYNTHETIC exact parent and amended cumulative budget review; never a real owner answer.')
    monkeypatch.setattr(filtered,'private_files',lambda paths:True) # Synthetic files only, not a real custody assertion.
    return stage.seal(value),[bodies[2]]


def configure(plan, monkeypatch, root):
    a,raw=BASE_CONFIGURE(plan,monkeypatch,root)
    if not stage.is_continuation(plan['payload']) or stage.is_quote(plan['payload']):return a,raw
    c=json.loads(raw)['payload'];a['payload']['authorized_at']='2026-10-17T18:00:02.500000Z';a=stage.seal(a['payload'])
    c.update(authorization=a);c['review']['reviewed_at']='2026-10-17T18:00:02.900000Z'
    c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
    raw=encode(stage.seal(c))
    monkeypatch.setattr(stage,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{plan['sha256']:dict(file_sha256=sha(raw),file_b64=base64.b64encode(raw).decode())})
    return a,raw


BASE_CONFIGURE=fixtures.configure
BASE_CAPTURE=fixtures.capture
BASE_CLOCK=fixtures.Clock


class Clock(BASE_CLOCK):
    def wall(self):return '2026-10-17T18:00:03Z'


def capture(monkeypatch,root,**kwargs):
    if kwargs.get('p') is None:kwargs['p'],kwargs['stats']=context(monkeypatch,root)
    with monkeypatch.context() as patch:
        patch.setattr(fixtures,'configure',configure)
        patch.setattr(fixtures,'Clock',Clock)
        patch.setattr(fixtures,'Connection',ordinary.Connection)
        return BASE_CAPTURE(patch,root,**kwargs)


@pytest.mark.parametrize('cli',[False,True])
def test_actual_single_request_join_and_original_clocks(monkeypatch,tmp_path,cli):
    p,bodies=context(monkeypatch,tmp_path);old,_=continuation.parent(p['payload'],at='2026-10-17T18:00:03Z')
    before={path.name:path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=bodies,cli=cli)
    v=stage.read_bundle(b); assert v['version']==continuation.HISTORY_BUNDLE
    assert v['attempts']==len(connections)==len(v['objects'])==1
    assert v['prior_active_seconds']==old['active_seconds'] and v['prior_original_bytes']==old['original_bytes']
    objects=filtered.history_objects(v,at='2026-10-17T18:01:00Z')
    assert objects[:2]==old['objects'] and len(objects)==3
    assert objects[2]['metadata']['request_scope']==p['payload']['requests'][0]['params']
    assert objects[0]['metadata']['received_at']=='2026-10-17T18:00:01Z'
    state,projection=filtered.project_bundle(v,at='2026-10-17T18:01:00Z')
    _,rows,_=history.build_dataset(state,feature_targets=[state['batches'][0]['records'][0]])
    assert rows[0]['home_prior_games']==rows[0]['away_prior_games']==6
    assert rows[0]['home_yards_games']==rows[0]['away_yards_games']==3
    assert rows[0]['home_yards_per_game']==rows[0]['away_yards_per_game']==400
    assert len(projection['selected_games'])==6 and len(filtered.response_objects(b,at='2026-10-17T18:01:00Z'))==3
    assert stage.verify_journal(tmp_path,b)
    raw=(tmp_path/(p['sha256']+'.transport-diagnostics.json')).read_bytes();diagnostics.read(tmp_path,b,raw,sha(raw))
    assert all((tmp_path/name).read_bytes()==raw for name,raw in before.items())


@pytest.mark.parametrize('attack',['bundle','journal','diagnostics','claim','stage_claim','plan','native_readback','report','configuration','authorization','operation','method','approval','derivation','template','application_readback','release'])
def test_altered_parent_material_rejects_before_dispatch(monkeypatch,tmp_path,attack):
    p,_=context(monkeypatch,tmp_path);p['payload']['continuation']['parent']['blobs_b64'][attack]=base64.b64encode(b'{}').decode()
    with pytest.raises(ValueError,match='INTEGRITY|LINKAGE'):stage.validate_plan(stage.seal(p['payload']),executable=True)


@pytest.mark.parametrize('attack',['target','request','limit','review','clock','custody','source'])
def test_wrong_successor_contract_rejects(monkeypatch,tmp_path,attack):
    p,_=context(monkeypatch,tmp_path);v=p['payload']
    if attack=='target':v['target']['home_school']='SYNTHETIC wrong school'
    elif attack=='request':v['requests'][0]['params']['team']=v['target']['home_school']
    elif attack=='limit':v['limits']['shared_active_seconds']=121
    elif attack=='review':v['continuation']['review']['subject_sha256']='0'*64
    elif attack=='clock':v['continuation']['review']['reviewed_at']='2026-10-17T18:00:04Z'
    elif attack=='custody':v['custody_id']+='-wrong'
    else:v['filtered_history']['conference']='CUSA'
    with pytest.raises(ValueError):stage.validate_plan(stage.seal(v),executable=True)


@pytest.mark.parametrize('scenario,reason',[({'header':30.1},'NCAAF_PILOT_DEADLINE'),({'body':10.1},'NCAAF_PILOT_DEADLINE'),
    ({'connect':10.1},'NCAAF_PILOT_DEADLINE'),({'declared':1},'NCAAF_PILOT_INCOMPLETE_BODY'),
    ({'connect':9.9,'write':9.9,'header':29.9,'body':9.9},'NCAAF_STAGED_DEADLINE'),
    ({'cancel':'interrupt'},'NCAAF_STAGED_ACQUISITION_FAILURE')])
def test_failure_is_spent_and_cannot_make_second_successor(monkeypatch,tmp_path,scenario,reason):
    p,bodies=context(monkeypatch,tmp_path)
    b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=bodies,scenarios=[scenario])
    assert b['payload']['status']=='INCOMPLETE' and b['payload']['reason']==reason and b['payload']['attempts']==1
    assert continuation.claim_name(p['payload'])==continuation.parent(p['payload'],at='2026-10-17T18:01:00Z')[0]['plan']['sha256']+'.history-continuation-claim-v6.json'
    assert all(c.closed for c in connections)
    with pytest.raises(ValueError,match='ALREADY_SPENT'):capture(monkeypatch,tmp_path,p=p,stats=bodies)
    other=deepcopy(p['payload']);other['collection_authorization_ref']+='-duplicate';other=stage.seal(other)
    failed,sent,_=capture(monkeypatch,tmp_path,p=other,stats=bodies)
    assert failed['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED' and failed['payload']['attempts']==0 and not sent
    with pytest.raises(ValueError,match='INCOMPLETE'):stage.read_bundle(b)


@pytest.mark.parametrize('attack',['missing','conflict','other-week'])
def test_missing_conflicting_or_substituted_yardage_rejects(monkeypatch,tmp_path,attack):
    p,bodies=context(monkeypatch,tmp_path);rows=json.loads(bodies[0])
    if attack=='missing':rows.pop(0)
    elif attack=='conflict':rows[0]['teams'][0]['points']+=1
    else:rows[0]['week']=14
    b,connections,_=capture(monkeypatch,tmp_path,p=p,stats=[encode(rows)])
    assert b['payload']['reason']=='NCAAF_STAGED_TARGET_CONFLICT' and b['payload']['attempts']==len(connections)==1
    with pytest.raises(ValueError):stage.read_bundle(b)


def test_budget_full_precision_and_no_reset():
    assert continuation.remaining(40.01600000000326)==Decimal('59.98399999999674')
    assert continuation.remaining(40.01600000000326,59,prior_bytes=104932,added_bytes=100)==Decimal('.98399999999674')
    for args in [(100,0),(40.01600000000326,59.984),(float('nan'),0)]:
        with pytest.raises(ValueError):continuation.remaining(*args)
    assert stage.active_limit(dict(version=fresh.HISTORY_PLAN))==60
    assert stage.active_limit(dict(version=continuation.HISTORY_PLAN))==120


def test_parent_expiry_and_unmodified_incomplete_rejection(monkeypatch,tmp_path):
    p,_=context(monkeypatch,tmp_path);old,_=continuation.parent(p['payload'],at='2026-10-17T18:01:00Z')
    with pytest.raises(ValueError,match='INCOMPLETE'):stage.read_bundle(stage.seal(old))
    with pytest.raises(ValueError,match='CLOCK'):continuation.parent(p['payload'],at='2026-10-18T18:01:00Z')
    name=next(iter(p['payload']['continuation']['parent']['custody_names'].values()))
    before=(tmp_path/name).read_bytes();(tmp_path/name).write_bytes(b'{}')
    try:
        with pytest.raises(ValueError,match='CUSTODY'):continuation.verify_parent_files(tmp_path,p['payload'],security=False)
    finally:(tmp_path/name).write_bytes(before)
    assert continuation.verify_parent_files(tmp_path,p['payload'],security=False)


def test_readonly_inspection_is_not_authority_and_custody_cannot_move(monkeypatch,tmp_path):
    p,_=context(monkeypatch,tmp_path)
    _,raw=configure(p,monkeypatch,tmp_path)
    control=json.loads(raw)['payload'];control['custody']['root']=str(tmp_path/'different-root')
    control['review']['subject_sha256']=stage.model.digest({k:v for k,v in control.items() if k!='review'})
    changed=encode(stage.seal(control))
    with pytest.raises(ValueError,match='PARENT_CUSTODY'):
        stage.configuration(p,changed,sha(changed),at='2026-10-17T18:00:03Z')
    p['payload']['continuation']['review']={}
    facts=continuation.inspect_parent(p['payload'],at='2026-10-17T18:01:00Z')
    assert facts['attempts']==3 and facts['provider_requests']==0 and facts['status']=='INCOMPLETE'
    with pytest.raises(ValueError,match='REVIEW'):stage.validate_plan(stage.seal(p['payload']),executable=True)


def selection(synthetic,monkeypatch,root,line):
    monkeypatch.setattr(fixtures,'context',context)
    monkeypatch.setattr(fixtures,'capture',capture)
    monkeypatch.setattr(fixtures,'configure',configure)
    # Keep the synthetic inference grants/clock and cleanup owned by the test,
    # through Refresh and replay. Stage capture uses its own bounded inner view.
    return fixtures.filtered_selection(synthetic,monkeypatch,root,line)


@pytest.mark.parametrize('line',[-7,-3.5])
def test_continuation_preparation_refresh_storage_export_replay_display(synthetic,monkeypatch,tmp_path,line):
    from app_core import ncaaf_private_refresh as refresh, ncaaf_spread_pair as pair
    selected,root,pref=selection(synthetic,monkeypatch,tmp_path,line)
    assert pref['payload']['version']==continuation.PREQUOTE
    updates,warnings,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=selected))
    assert not warnings and not errors
    rows=updates['ncaaf_private_refresh_ranked'];assert len(rows)==2 and rows[0]['cover']>=rows[1]['cover']
    for row in rows:
        assert row['cover']+row['push']+row['loss']==pytest.approx(1)
        assert row['push']>0 if float(line).is_integer() else row['push']==0
        assert row['estimated_ev_per_unit']==pytest.approx(row['cover']*(1+100/110)+row['push']-1)
    raw=Path(updates['ncaaf_private_refresh_result']['path']).read_bytes()
    assert refresh.read_private_export(raw,root=root,database=root/'ncaaf-private-refresh.sqlite')==rows
    assert pair.private_display(updates['analysis_df'].iloc[0].to_dict())


def test_continuation_actual_normal_refresh_button(synthetic,monkeypatch,tmp_path):
    import test_ncaaf_private_normal_refresh as normal
    selected,root,_=selection(synthetic,monkeypatch,tmp_path,-7)
    monkeypatch.setattr(normal,'selection',lambda *args,**kwargs:selected)
    normal.test_actual_normal_refresh_button_uses_private_caller(synthetic,monkeypatch,root)
