"""Network-free normal Refresh proof. Synthetic histories confer no live authority."""
import base64
from copy import deepcopy
from datetime import timedelta
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from app_core import ncaaf_owner_spread as owner, ncaaf_spread_pair as pair
from app_core import ncaaf_private_refresh as refresh, ncaaf_response_custody as custody
from app_core import ncaaf_staged_acquisition as stage, ncaaf_pipeline_evidence as adapter
from ncaaf_future_pair_fixtures import fixture, other_side, resign, inference_authorization, NOW
from test_ncaaf_model_compatibility import synthetic


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def selection(synthetic,monkeypatch,root,line=-7):
    home,row=fixture(synthetic,monkeypatch,root,line)
    p=home['payload'];o=pair._observation(home);t=o['source_review']['terms_review'];q=o['quote']
    p['version']=owner.CONDITIONAL_INTEGER if owner.integer(home) else owner.CONDITIONAL_HALF
    t.update(product=None,listing_id=None,rule_edition=None,rules_document_sha256=None)
    q.update(product=None,listing_id=None,period_source=owner.projection.CONDITIONAL_VERSION,
             rules_source=owner.projection.CONDITIONAL_VERSION)
    o['source_review']['offer_verification']['rule_edition']=None
    last=p['staged_binding']['payload']['quote']['payload']['objects'][0]
    with owner.projection.selected(home):
        p['custody_packet']['payload']['response_objects'][-1]=custody.capture(
            base64.b64decode(last['body_b64']),last['metadata'],quote=q,terms=t,owner_reviewed=True)
    resign(home)
    away,_=other_side(home,row)
    packet=pair.prepare(home,away,prepared_at='2026-10-17T18:30:02Z')
    packet['payload']['owner_review']=dict(version=pair.REVIEW_VERSION,reviewer=owner.OWNER,
        reviewed_at='2026-10-17T18:30:04Z',subject_sha256=pair.subject(packet),
        qualifications='SYNTHETIC conditional comparison only, not an owner answer.',
        finding='SYNTHETIC review of exact child inputs and explicit mathematical assumptions.',
        child_reviews=dict(home=home['payload']['owner_review'],away=away['payload']['owner_review']))
    packet=stage.seal(packet['payload'])
    authorization=inference_authorization(packet,monkeypatch,paired=True)
    import test_source_contract_pipeline as ef
    monkeypatch.setattr(ef,'NOW',NOW)
    monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:NOW)
    monkeypatch.setattr('app_core.prediction_evidence.now_utc',lambda:(NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.public_board.datetime',ef.FrozenDateTime)
    return dict(version=refresh.VERSION,packet=packet,authorization=authorization)


@pytest.mark.parametrize('line',[-7,-3.5])
def test_private_refresh_native_storage_export_replay_and_display(synthetic,monkeypatch,tmp_path,line):
    value=selection(synthetic,monkeypatch,tmp_path,line)
    before=deepcopy(value)
    updates,warnings,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=value))
    assert not warnings and not errors
    assert updates['pipeline_status']=='PRIVATE_CONDITIONAL_RESEARCH'
    ranked=updates['ncaaf_private_refresh_ranked']
    assert len(ranked)==2 and ranked[0]['cover']>=ranked[1]['cover']
    assert all(r['cover']+r['push']+r['loss']==pytest.approx(1) for r in ranked)
    assert ranked[0]['push']==ranked[1]['push']
    assert (ranked[0]['push']>0) if float(line).is_integer() else ranked[0]['push']==0
    for r in ranked:
        assert r['estimated_ev_per_unit']==pytest.approx(r['cover']*(1+100/110)+r['push']-1)
        assert r['applicability']=='UNVERIFIED_CONDITIONAL_RESEARCH' and r['stake']==0
    assert len(updates['ncaaf_private_refresh_receipts'])==2
    assert updates['best_picks_df'].cover.tolist()==[row['cover'] for row in ranked]
    assert 'calibrated_probability' not in updates['best_picks_df']
    result=updates['ncaaf_private_refresh_result'];path=__import__('pathlib').Path(result['path'])
    raw=path.read_bytes();csv=__import__('pathlib').Path(result['csv_path'])
    assert refresh.read_private_export(raw,root=tmp_path,database=tmp_path/'ncaaf-private-refresh.sqlite')==ranked
    altered=json.loads(raw);altered['payload']['rows'][0]['cover']=0.
    with pytest.raises(ValueError):
        refresh.read_private_export(owner.model.encode(stage.seal(altered['payload'])),root=tmp_path,database=tmp_path/'ncaaf-private-refresh.sqlite')
    original_csv=csv.read_bytes()
    try:
        csv.write_bytes(original_csv+b'ALTERED')
        with pytest.raises(ValueError):
            refresh.read_private_export(raw,root=tmp_path,database=tmp_path/'ncaaf-private-refresh.sqlite')
    finally:csv.write_bytes(original_csv)
    assert value==before
    assert list(tmp_path.glob('*.inference-attempts.jsonl'))
    again,_,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=value))
    assert errors and again['pipeline_status']=='UNAVAILABLE'


def test_actual_normal_refresh_button_uses_private_caller(synthetic,monkeypatch,tmp_path):
    from tests.apptest_owned_script import app_from_owned_script
    value=selection(synthetic,monkeypatch,tmp_path)
    # Actual main, sidebar button callback and private caller. No publishing token.
    app=app_from_owned_script(tmp_path,'import streamlit_app\nstreamlit_app.main()\n',default_timeout=180)
    app.session_state['ncaaf_private_refresh']=True
    app.session_state['ncaaf_private_refresh_selection']=value
    app.run()
    assert not app.exception
    assert not list(tmp_path.glob('*.inference-attempts.jsonl'))
    next(b for b in app.button if b.label=='Refresh picks').click().run()
    assert not app.exception
    assert app.session_state['pipeline_status']=='PRIVATE_CONDITIONAL_RESEARCH'
    assert len(app.session_state['ncaaf_private_refresh_ranked'])==2
    assert len(app.dataframe)>=1
    journals=list(tmp_path.glob('*.inference-attempts.jsonl'));assert len(journals)==1
    raw=journals[0].read_bytes()
    app.run()
    assert not app.exception and journals[0].read_bytes()==raw


@pytest.mark.parametrize('attack',['body','event','review','assumptions','kickoff','authority','configuration','journal'])
def test_private_refresh_conflicts_reject_without_fallback(synthetic,monkeypatch,tmp_path,attack):
    value=selection(synthetic,monkeypatch,tmp_path)
    if attack=='authority':value['authorization']['payload']['subject_sha256']='0'*64
    elif attack=='journal':
        (tmp_path/(value['authorization']['sha256']+'.inference-attempts.jsonl')).write_bytes(b'SYNTHETIC spent')
    else:
        children=pair.children(value['packet'])
        home=children['home'];p=home['payload'];o=pair._observation(home)
        if attack=='body':p['custody_packet']['payload']['response_objects'][0]['payload']['body_b64']='QQ=='
        elif attack=='event':o['event']['provider_event_id']='DIFFERENT'
        elif attack=='review':p['owner_review']=None
        elif attack=='assumptions':o['quote']['period']='first_half'
        elif attack=='kickoff':o['event']['start_utc']='2026-10-16T00:00:00Z'
        else:p['staged_binding']['payload']['history']['payload']['control_receipt']['file_sha256']='0'*64
        home=stage.seal(p)
        # Rehashing the shared envelope supplies no new substantive review.
        value['packet']=pair.prepare(home,children['away'],prepared_at=value['packet']['payload']['prepared_at'])
    updates,_,errors=refresh.pipeline(dict(ncaaf_private_refresh_selection=value))
    assert errors and updates['pipeline_status']=='UNAVAILABLE'
    assert updates['analysis_df'].empty and not list(tmp_path.glob('*.computation.json'))


def test_missing_private_inputs_do_not_call_live_pipeline(monkeypatch):
    import streamlit_app
    monkeypatch.setattr(streamlit_app,'run_analysis_pipeline',lambda **kw:pytest.fail('LIVE FALLBACK'))
    updates,_,errors=streamlit_app._run_pipeline(dict(ncaaf_private_refresh=True))
    assert errors and updates['pipeline_status']=='UNAVAILABLE'


def test_ranking_is_not_ev_and_no_push_denominator_is_labelled():
    from app_core.ncaaf_spread_mass import conditional_no_push
    assert conditional_no_push(dict(cover=0.,push=1.,loss=0.))==dict(semantics='cover_conditional_on_no_push',state='UNKNOWN',value=None)
    assert conditional_no_push(dict(cover=.45,push=.1,loss=.45))['value']==pytest.approx(.5)


def test_strict_terms_reader_does_not_accept_conditional_contract(synthetic,monkeypatch,tmp_path):
    value=selection(synthetic,monkeypatch,tmp_path)
    child=pair.children(value['packet'])['home']
    child['payload']['version']=owner.VERSION
    child=stage.seal(child['payload'])
    with pytest.raises(ValueError):owner.projection.subjects(child)


@pytest.mark.parametrize('attack',['source','binding'])
def test_physical_private_source_graph_authentication_is_fresh(tmp_path,attack):
    from scripts import ncaaf_private_refresh_scope as scope
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    binding=scope.authenticate(root)
    for relative in [scope.BINDING,*binding['components']]:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(scope.physical(root/relative))
    assert scope.authenticate(tmp_path)==binding
    path=tmp_path/('app_core/ncaaf_owner_spread.py' if attack=='source' else scope.BINDING)
    path.write_bytes(path.read_bytes()+b'\nALTERED\n')
    with pytest.raises(ValueError):scope.authenticate(tmp_path)
