"""SYNTHETIC standard Odds shape and substantive reviewed terms; no real inputs."""
import base64
from copy import deepcopy
import hashlib
import json
import pytest
from app_core import ncaaf_owner_research as old,ncaaf_response_custody as custody,ncaaf_model_compatibility as model,ncaaf_pipeline_evidence as adapter
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
from test_ncaaf_owner_research import fixture as old_fixture,verify_owner

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def source_shaped(synthetic,monkeypatch,kind='spread_home',line=-3.5):
    packet,row=old_fixture(synthetic,monkeypatch,kind,line);c=packet['payload']['custody_packet']
    for obj in c['payload']['response_objects']:
        b=obj['payload']
        if b['metadata']['provider']!='odds_api':continue
        body=json.loads(base64.b64decode(b['body_b64']))
        m=body[0]['bookmakers'][0]['markets'][0];m.pop('period');m.pop('settlement_rules')
        # Current provider docs specify market-level observation, not operator update.
        m['last_update']=body[0]['bookmakers'][0]['last_update']
        raw=json.dumps(body,indent=3).encode();b.update(body_b64=base64.b64encode(raw).decode(),body_bytes=len(raw),body_sha256=hashlib.sha256(raw).hexdigest())
        b['projection']['provider_clock_path']='market.last_update';obj['sha256']=model.digest(b)
    resign(packet)
    return packet,row


def resign(packet):
    c=packet['payload']['custody_packet'];r=c['payload']['custody_admission'];v=r['verification'];a=r['owner_verification']
    v['subject_sha256']=custody.subject_hash(c);a['subject_sha256']=v['subject_sha256'];a['verification_sha256']=model.digest(v)
    c['sha256']=model.digest(c['payload']);verify_owner(packet)


def test_original_reader_rejects_missing_feed_scope_through_actual_caller(synthetic,monkeypatch):
    packet,row=source_shaped(synthetic,monkeypatch)
    with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='unavailable' and result['ml_unavailable_reason']=='NCAAF_CUSTODY_QUOTE_CONFLICT'


def fixture(synthetic,monkeypatch,kind='spread_home',line=-3.5):
    from app_core import ncaaf_owner_terms_research as new,ncaaf_prospective_chronology as chronology
    packet,row=source_shaped(synthetic,monkeypatch,kind,line)
    def migrate(v):
        if isinstance(v,dict):return {k:migrate(x) for k,x in v.items()}
        if isinstance(v,list):return [migrate(x) for x in v]
        return v[:-1]+'2' if isinstance(v,str) and v.startswith('ncaaf-owner-') and v.endswith('-v1') else v
    packet=migrate(packet);p=packet['payload'];c=p['custody_packet'];n=c['payload']['native_packet'];o=n['payload']['observation']['payload'];q=o['quote'];r=o['source_review']
    q['period_source']=q['rules_source']=new.projection.VERSION
    r['quote_observation']['quote_sha256']=model.digest(q)
    r['offer_verification'].update(quote_sha256=model.digest(q),observation_sha256=model.digest(r['quote_observation']))
    r['owner_verification']['subject_sha256']=chronology.subject_hash(o)
    n['payload']['observation']['sha256']=model.digest(o);n['sha256']=model.digest(n['payload'])
    d=p['dependency_review'];v=d['dependency_verification'];v['subject_sha256']=chronology.dependency_subject_hash(n)
    d['owner_verification'].update(subject_sha256=v['subject_sha256'],verification_sha256=model.digest(v))
    for obj in c['payload']['response_objects']:
        b=obj['payload']
        if b['metadata']['provider']=='odds_api':
            raw=json.loads(base64.b64decode(b['body_b64']))
            from app_core.prediction_evidence import provider_quotes
            row['provider_quotes']=provider_quotes(raw[0])
            with new.projection.selected(packet):b['projection']=custody._project_quote(raw,b['metadata'],q,r['terms_review'],owner_reviewed=True)
            obj['sha256']=model.digest(b)
    c['payload']['custody_admission']['verification']['native_verification_sha256']=model.digest(v)
    resign(packet)
    review=p['owner_review'];review.update(version=new.REVIEW_VERSION,subject_version=new.SUBJECT_VERSION,subject_sha256=new.subject_hash(packet))
    review['conclusions']={k:dict(conclusion='VERIFIED',evidence_sha256=v,finding='SYNTHETIC owner inspected original response and exact applicable terms for '+k) for k,v in new.evidence_subjects(packet).items()}
    packet['sha256']=model.digest(p)
    return packet,row


def test_successor_actual_caller_private_capture_export_replay(synthetic,monkeypatch,tmp_path):
    from app_core import ncaaf_owner_terms_research as new,research_display,research_replay
    packet,row=fixture(synthetic,monkeypatch);before=deepcopy(packet)
    with pytest.raises(ValueError):old.load(packet)
    with pytest.raises(ValueError,match='PRIVATE_RESEARCH_NOT_SELECTED'):
        with adapter.selected([packet]):pass
    with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='success',result['ml_unavailable_reason']
    assert result['ml_probability']==pytest.approx(.636830651175619)
    assert adapter.diagnose(dict(row,**result))==dict(status='COMPLETE',reason='AVAILABLE')
    from test_ncaaf_compatible_pipeline import actual as pipeline
    from app_core import ncaaf_compatible_pipeline as native
    original=native.view
    monkeypatch.setattr(native,'view',lambda p:new.view(p) if p['payload']['version']==new.VERSION else original(p))
    selected=adapter.selected;monkeypatch.setattr(adapter,'selected',lambda packets=():selected(packets,private_research=True))
    analysis,_=pipeline(monkeypatch,packet)
    assert analysis.loc[analysis.market_type.eq('spread_home')].iloc[0].ml_inference_status=='success',analysis[['market_type','ml_unavailable_reason']].to_dict('records')
    from test_source_contract_pipeline import exported
    import test_source_contract_pipeline as evidence_fixture
    from test_ncaaf_owner_research import NOW
    from datetime import timedelta
    monkeypatch.setattr(evidence_fixture,'NOW',NOW);monkeypatch.setattr(evidence_fixture,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    saved=exported(monkeypatch,tmp_path,analysis)
    from app_core.per_game_boards import per_game_board
    private_frames=[per_game_board(saved['card'],saved['captured'],family=f,novig_only=True,college_fallback=True,private_research=True) for f in ('overall','sides','totals')]
    shown=json.loads(private_frames[1].iloc[0].research_display)
    assert shown['probability']==result['ml_probability'] and 'OWNER_REVIEWED' in shown['basis'] and shown['ev'] is None,shown
    receipt=research_replay.retain_export(private_frames,saved['package'],saved['card'],saved['captured'],path=saved['db'])
    _,sources=research_replay.read_export(receipt['export_id'],path=saved['db'])
    frame=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    retained=frame.loc[frame.market_type.eq('spread_home')].iloc[0]
    assert research_display.from_export(retained)['probability'] is None
    assert 'body_b64' not in json.dumps(saved['package']) and all(x['production_bet_amount']==0 for x in saved['card'].wager_contract)
    assert json.loads(retained.ml_estimate_metadata)['ncaaf_inputs']['payload']['original_packet']==before and packet==before
    assert new.projection.SELECTED.get() is None and not adapter.ACCEPTED_PACKETS
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'SYNTHETIC-private-display.py'
    script.write_text('import streamlit as st\nfrom app.ui.ncaaf_pipeline_research import render\nrender(st.session_state["games"],st.session_state["candidates"])',encoding='utf-8')
    app=AppTest.from_file(str(script));app.session_state['games']=saved['card'];app.session_state['candidates']=saved['captured'];app.session_state['ncaaf_private_research']=True;app.run()
    assert not app.exception and any(m.value==f"{result['ml_probability']:.2%}" for m in app.metric)


@pytest.mark.parametrize('attack',['terms','raw_override','body','review','late_review','stale','started','borrowed_old','price'])
def test_missing_or_conflicting_scope_cannot_infer(synthetic,monkeypatch,attack):
    from app_core import ncaaf_owner_terms_research as new,ncaaf_research as math
    packet,row=fixture(synthetic,monkeypatch);p=packet['payload'];o=p['custody_packet']['payload']['native_packet']['payload']['observation']['payload']
    if attack=='terms':o['source_review']['terms_review']['product']='UNKNOWN'
    elif attack=='raw_override':
        b=p['custody_packet']['payload']['response_objects'][-1]['payload'];raw=json.loads(base64.b64decode(b['body_b64']));raw[0]['bookmakers'][0]['markets'][0]['period']='first_half'
        data=json.dumps(raw).encode();b.update(body_b64=base64.b64encode(data).decode(),body_bytes=len(data),body_sha256=hashlib.sha256(data).hexdigest())
    elif attack=='body':p['custody_packet']['payload']['response_objects'][-1]['payload']['body_b64']='QQ=='
    elif attack=='review':p['owner_review']['conclusions']['offer_and_settlement']['conclusion']='UNKNOWN'
    elif attack=='late_review':p['owner_review']['reviewed_at']='2026-10-09T12:00:05Z'
    elif attack=='stale':o['quote']['recorded_at']='2026-10-09T11:00:00Z'
    elif attack=='started':o['event']['start_utc']='2026-10-09T11:00:00Z'
    elif attack=='borrowed_old':p['owner_review']['version']=old.REVIEW_VERSION
    else:row['odds_american']+=10
    packet['sha256']=model.digest(p)
    monkeypatch.setattr(math,'centers',lambda *a,**k:pytest.fail('REJECTED SCOPE REACHED NUMERICAL INFERENCE'))
    try:
        with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    except ValueError:pass
    else:assert result['ml_inference_status']=='unavailable'


def test_selected_away_same_math_and_no_scope_default(synthetic,monkeypatch):
    from app_core import ncaaf_owner_terms_research as new
    packet,row=fixture(synthetic,monkeypatch,'spread_away',3.5)
    with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='success' and result['ml_probability']==pytest.approx(.363169348824381)
    assert new.projection.transport_facts({}, {}, {}, {}) is None


def test_consistently_rehashed_conflicting_original_scope_rejects(synthetic,monkeypatch):
    from app_core import ncaaf_owner_terms_research as new,ncaaf_research as math
    packet,row=fixture(synthetic,monkeypatch);p=packet['payload'];c=p['custody_packet']
    obj=c['payload']['response_objects'][-1];b=obj['payload'];raw=json.loads(base64.b64decode(b['body_b64']))
    raw[0]['bookmakers'][0]['markets'][0]['period']='first_half'
    data=json.dumps(raw).encode();b.update(body_b64=base64.b64encode(data).decode(),body_bytes=len(data),body_sha256=hashlib.sha256(data).hexdigest());obj['sha256']=model.digest(b)
    resign(packet)
    p['owner_review'].update(version=new.REVIEW_VERSION,subject_version=new.SUBJECT_VERSION,subject_sha256=new.subject_hash(packet),conclusions={k:dict(conclusion='VERIFIED',evidence_sha256=v,finding='SYNTHETIC fully rehashed but conflicting declaration for '+k) for k,v in new.evidence_subjects(packet).items()})
    packet['sha256']=model.digest(p)
    monkeypatch.setattr(math,'centers',lambda *a,**k:pytest.fail('REHASHED CONFLICT REACHED INFERENCE'))
    with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    assert result['ml_unavailable_reason']=='NCAAF_CUSTODY_QUOTE_CONFLICT'
