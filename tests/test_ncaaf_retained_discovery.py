"""Network-blocked synthetic bodies with real mixed-provenance stage contracts."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import pytest
from app_core import ncaaf_pilot_discovery as discovery,ncaaf_reviewed_target as target
from app_core import ncaaf_staged_acquisition as stage,ncaaf_reviewed_staged as evidence
from app_core import ncaaf_private_refresh as refresh,ncaaf_spread_pair as pair
from test_ncaaf_model_compatibility import synthetic
import ncaaf_retained_discovery_fixtures as retained


def selection(synthetic,monkeypatch,root,line=-7):
    import test_ncaaf_fresh_history as fresh
    import test_ncaaf_filtered_history as filtered
    monkeypatch.setattr(fresh,'legacy',retained)
    monkeypatch.setattr(filtered,'legacy',retained)
    monkeypatch.setattr(filtered,'context',fresh.context)
    monkeypatch.setattr(filtered,'Connection',fresh.Connection)
    return filtered.filtered_selection(synthetic,monkeypatch,root,line)


@pytest.mark.parametrize('line',[-7,-3.5])
def test_retained_history_prospective_discovery_full_private_pipeline(synthetic,monkeypatch,tmp_path,line):
    value,root,pref=selection(synthetic,monkeypatch,tmp_path,line)
    children=pair.children(value['packet']);home=children['home'];p=home['payload']
    binding=p['staged_binding']['payload'];original=p['alias_review']['discovery']
    assert binding['history']['payload']['evidence_label']==binding['quote']['payload']['evidence_label']=='RETAINED'
    assert pref['payload']['evidence_label']==p['alias_review']['evidence_label']=='RETAINED'
    assert original['bundle']['payload']['evidence_label']==original['bundle']['payload']['plan']['payload']['evidence_label']=='PROSPECTIVE'
    before=stage.model.encode(original);originals=stage.model.encode(binding['history'])
    assert target.discovery_original(original,'RETAINED')['body_sha256']==original['bundle']['payload']['objects'][0]['body_sha256']
    assert evidence.join(binding['history'],binding['prequote'],binding['quote'])==p['staged_binding']
    from app_core import ncaaf_owner_spread as owner
    for late in ['2026-10-17T18:45:01Z',retained.CANONICAL['start_utc']]:
        with pytest.raises(ValueError):owner.accepted_dependencies(home,None,late)
    updates,warnings,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=value))
    assert not warnings and not errors and updates['pipeline_status']=='PRIVATE_CONDITIONAL_RESEARCH'
    ranked=updates['ncaaf_private_refresh_ranked'];assert len(ranked)==2 and ranked[0]['cover']>=ranked[1]['cover']
    for row in ranked:
        assert row['cover']+row['push']+row['loss']==pytest.approx(1)
        assert (row['push']>0)if float(line).is_integer()else row['push']==0
        assert row['estimated_ev_per_unit']==pytest.approx(row['cover']*(1+100/110)+row['push']-1)
        assert row['evidence_label']=='RETAINED' and row['applicability']=='UNVERIFIED_CONDITIONAL_RESEARCH'
        assert pair.private_display(updates['analysis_df'].iloc[0].to_dict())
    raw=Path(updates['ncaaf_private_refresh_result']['path']).read_bytes()
    assert refresh.read_private_export(raw,root=root,database=root/'ncaaf-private-refresh.sqlite')==ranked
    assert stage.model.encode(original)==before and stage.model.encode(binding['history'])==originals
    assert original['bundle']['payload']['accepted'] is original['bundle']['payload']['inference'] is False
    assert list(root.glob('*.inference-attempts.jsonl'))


def test_retained_history_prospective_discovery_actual_refresh_button(synthetic,monkeypatch,tmp_path):
    import test_ncaaf_private_normal_refresh as normal
    value,root,_=selection(synthetic,monkeypatch,tmp_path)
    monkeypatch.setattr(normal,'selection',lambda *args,**kwargs:value)
    normal.test_actual_normal_refresh_button_uses_private_caller(synthetic,monkeypatch,root)


@pytest.fixture
def original(monkeypatch,tmp_path):
    """Minimal complete native discovery/control/journal, no history reconstruction."""
    pref=dict(payload=dict(history_bundle_sha256='a'*64,schedule=[json.loads(retained.bodies()[0])[0]]))
    v=retained.identity(monkeypatch,tmp_path,pref)
    assert v['evidence_label']=='RETAINED'
    return v['discovery']


def test_bridge_preserves_original_bytes_and_reauthenticates(original):
    before=stage.model.encode(original)
    one=target.discovery_original(original,'RETAINED');one['metadata']['status']=999
    two=target.discovery_original(original,'RETAINED')
    assert two['metadata']['status']==200 and stage.model.encode(original)==before
    assert target.prior.discovery_original(original,'PROSPECTIVE')==two
    assert original['bundle']['payload']['accepted'] is False


@pytest.mark.parametrize('attack',['body','bundle-label','plan-label','both-labels','synthetic','proposal','configuration','journal','authority','expired'])
def test_altered_or_rehashed_discovery_is_not_authenticated(original,attack):
    value=deepcopy(original);b=value['bundle']['payload']
    if attack=='body':b['objects'][0]['body_b64']=base64.b64encode(b'[]').decode()
    elif attack=='bundle-label':b['evidence_label']='RETAINED'
    elif attack=='plan-label':b['plan']=stage.seal(dict(b['plan']['payload'],evidence_label='RETAINED'))
    elif attack in {'both-labels','synthetic','proposal'}:
        label={'both-labels':'RETAINED','synthetic':'SYNTHETIC','proposal':'PROPOSAL'}[attack]
        b['evidence_label']=label;b['plan']=stage.seal(dict(b['plan']['payload'],evidence_label=label))
    elif attack=='configuration':
        c=json.loads(base64.b64decode(value['configuration_receipt']['file_b64']));c['payload']['subject']['code_head']='f'*40
        raw=stage.model.encode(stage.seal(c['payload']));value['configuration_receipt']=dict(file_b64=base64.b64encode(raw).decode(),file_sha256=hashlib.sha256(raw).hexdigest())
        # A substituted self-rehashed control still fails its unchanged owner-review subject.
    elif attack=='journal':value['journal_b64']=base64.b64encode(b'{}\n').decode()
    else:
        a=b['owner_authorization_receipt']['payload']
        if attack=='authority':a['plan_sha256']='0'*64
        else:a['expires_at']='2026-10-17T18:00:00Z'
        b['owner_authorization_receipt']=stage.seal(a)
    value['bundle']=stage.seal(b)
    with pytest.raises(ValueError):target.discovery_original(value,'RETAINED')
    assert target.discovery_original(original,'RETAINED')['metadata']['status']==200


@pytest.mark.parametrize('label',['SYNTHETIC','PROPOSAL'])
def test_prospective_discovery_not_admitted_as_other_generation(original,label):
    with pytest.raises(ValueError):target.discovery_original(original,label)


@pytest.mark.parametrize('label',['RETAINED','PROPOSAL'])
def test_bridge_does_not_change_native_plan_labels(original,label):
    p=deepcopy(original['bundle']['payload']['plan']['payload']);p['evidence_label']=label
    with pytest.raises(ValueError):discovery.validate_plan(stage.seal(p),executable=True)


def staged_inputs(synthetic,monkeypatch,root):
    """Stage-level negative controls need no unrelated private-store preparation."""
    import test_ncaaf_fresh_history as fresh
    import test_ncaaf_filtered_history as filtered
    from app_core import ncaaf_model_compatibility as model,ncaaf_transport_diagnostics as diagnostic
    monkeypatch.setattr(fresh,'legacy',retained);monkeypatch.setattr(filtered,'legacy',retained)
    h,_,_=fresh.capture(monkeypatch,root)
    checked=model.read_model(synthetic[0],synthetic[1],model.envelope(family='spread'))
    pref=evidence.prepare(h,model_packet=model.export_model(checked),prepared_at='2026-10-17T18:01:00Z')
    identity=retained.identity(monkeypatch,root,pref);pref=retained.review_prequote(pref,h)
    payload=retained.plan(target.QUOTE_PLAN,h,pref,identity)['payload']
    payload.update(version='ncaaf-fresh-filtered-final-quote-stage-plan-v5',filtered_history=deepcopy(h['payload']['plan']['payload']['filtered_history']))
    plan=stage.seal(payload);authority,_=filtered.configure(plan,monkeypatch,root)
    clock=retained.Clock('2026-10-17T18:30:01Z');collector=diagnostic.Collector()
    def transport(*args):
        collector.enter('request_write');collector.wrote_request();collector.enter('response_headers');collector.observed_status(200)
        clock.elapsed+=.01;stream=retained.Stream(retained.odds());stream.diagnostics=collector;return stream
    transport.diagnostics=collector
    q=stage.acquire(plan,authority,root=root,transport=transport,history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono,diagnostics=collector)
    assert q['payload']['status']=='CAPTURED_UNREVIEWED'
    assert evidence.join(h,pref,q)['payload']['history']==h
    return h,pref,q,authority


@pytest.mark.parametrize('attack',['target','review','stale','kickoff','mixed-quote'])
def test_join_and_caller_keep_target_review_and_clock_gates(synthetic,monkeypatch,tmp_path,attack):
    h,p,q,authority=staged_inputs(synthetic,monkeypatch,tmp_path)
    if attack=='target':q['payload']['plan']['payload']['target']['home_school']='SYNTHETIC wrong target'
    elif attack=='review':p['payload']['history_review']=None;p=stage.seal(p['payload'])
    elif attack=='mixed-quote':q['payload']['evidence_label']='SYNTHETIC'
    else:
        at='2026-10-17T18:45:01Z'if attack=='stale'else retained.CANONICAL['start_utc']
        with pytest.raises(ValueError):
            if attack=='stale':stage.authorize(q['payload']['plan'],authority,at)
            else:evidence.validate_prequote(p,h,at=at,review_required=True)
        assert not list(tmp_path.glob('*.inference-attempts.jsonl'));return
    q=stage.seal(q['payload'])
    with pytest.raises(ValueError):evidence.join(h,p,q)
    assert not list(tmp_path.glob('*.inference-attempts.jsonl'))
