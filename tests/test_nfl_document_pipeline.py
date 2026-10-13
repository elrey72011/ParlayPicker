"""SYNTHETIC retrieval -> existing PDF/math -> private browser. No real bodies."""
from copy import deepcopy
import json
import pytest
from app_core import nfl_document_acquisition as acquisition,nfl_document_research as reader,nfl_private_routes as routes
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_document_acquisition import run,Stream
from test_nfl_pdf_evidence import fixture as original_fixture,resign
from test_source_contract_pipeline import INFERENCE

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def fixture(monkeypatch,root,away=False,reference_away=False):
    packet,baseline=original_fixture(monkeypatch,away,reference_away)
    from app_core.nfl_pdf_evidence import read_pdf
    raw=read_pdf(packet['payload']['objects']['rules'])[0]
    bundle,_,_=run(root,Stream(raw));obj=acquisition.project(bundle,'rules',prepared_at='2026-10-06T19:25:02.200Z')
    p=packet['payload'];p['objects']['rules']=obj
    p['permissions']['rules'].update(source_id=obj['receipt']['source_id'],endpoint=obj['receipt']['endpoint'])
    resign(p);packet=acquisition.seal(p)
    outer=dict(version=reader.VERSION,evidence_label='SYNTHETIC',event=deepcopy(p['event']),quote=deepcopy(p['quote']),packet=packet,
        document_captures=dict(rules=dict(bundle=bundle,document_id='rules')),owner_review={})
    review(outer)
    return acquisition.seal(outer),baseline


def review(p):
    p['owner_review']=dict(version='nfl-owner-document-linkage-review-v1',reviewer='Robert Velarde',reviewed_at='2026-10-06T19:25:05Z',
        subject_sha256=reader.subject(p),conclusion='VERIFIED_ORIGINAL_RETRIEVAL_LINKAGE',basis='SYNTHETIC substantive review of exact original wire, decoded bytes, receipts and extracted object linkage; fixture only.')


def test_v2_public_guide_actual_caller_capture_replay_and_display(monkeypatch,tmp_path):
    from app_core import nfl_public_document_research as successor
    from test_nfl_document_acquisition import capture_guide
    from test_nfl_owner_research import actual_pipeline,private_app
    from test_source_contract_pipeline import exported
    from app_core import research_replay
    from streamlit.testing.v1 import AppTest
    packet,baseline=original_fixture(monkeypatch);p=packet['payload']
    _,text=acquisition.custody.read(p['objects']['clock_document'])
    text=text.replace('https://synthetic.invalid/provider-guide',acquisition.GUIDE_URL)
    text='<html><body>'+text+'\nSYNTHETIC published example: apiKey=YOUR_API_KEY</body></html>'
    capture,_,calls=capture_guide(tmp_path,text.encode())
    obj=acquisition.project(capture,'clock');p['objects']['clock_document']=obj
    p['permissions']['clock_document'].update(source_id=obj['receipt']['source_id'],endpoint=obj['receipt']['endpoint'])
    projection=p['projections']['clock_document']['payload'];projection['values']['source_reference']=acquisition.GUIDE_URL
    for fact in projection['facts'].values():
        for support in fact['supports']:
            exact=support['text']['exact'].replace('https://synthetic.invalid/provider-guide',acquisition.GUIDE_URL)
            start=text.index(exact);support['text']=dict(start=start,end=start+len(exact),exact=exact)
    resign(p)
    outer=dict(version=successor.VERSION,evidence_label='SYNTHETIC',event=deepcopy(p['event']),quote=deepcopy(p['quote']),packet=acquisition.seal(p),document_captures=dict(clock_document=dict(bundle=capture,document_id='clock')),owner_review={})
    review(outer);outer['owner_review']['subject_sha256']=successor.subject(outer);selected=acquisition.seal(outer)
    with routes.selected([selected]):frame=actual_pipeline(monkeypatch)
    assert frame.iloc[0].ml_inference_status=='success'
    assert frame.iloc[0].ml_probability==pytest.approx(baseline.iloc[0].ml_probability,abs=1e-12)
    retained=exported(monkeypatch,tmp_path,frame)
    receipt=research_replay.retain_export(retained['frames'],retained['package'],retained['card'],retained['captured'],path=retained['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=retained['db'])
    replay=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    shown=routes.private_display(replay.iloc[0])
    assert shown['raw_probability']==frame.iloc[0].ml_probability and shown['stake']==0 and shown['wager_action']=='PASS'
    app=AppTest.from_function(private_app);app.session_state['private_candidates']=replay;app.run()
    assert not app.exception and any(m.value==f'{frame.iloc[0].ml_probability:.2%}' for m in app.metric)
    assert calls==[acquisition.GUIDE_URL] and 'YOUR_API_KEY' not in json.dumps(shown)
    old=deepcopy(outer);old['version']=reader.VERSION
    old['owner_review']['subject_sha256']=reader.subject(old)
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_MISSING'):reader.checked(acquisition.seal(old),INFERENCE)


@pytest.mark.parametrize('away,reference_away',[(False,False),(False,True),(True,False),(True,True)])
def test_actual_retrieval_caller_export_replay_browser(monkeypatch,tmp_path,away,reference_away):
    from app_core import research_replay
    from test_nfl_owner_research import actual_pipeline,private_app
    from test_source_contract_pipeline import exported
    from streamlit.testing.v1 import AppTest
    packet,baseline=fixture(monkeypatch,tmp_path,away,reference_away);original=deepcopy(packet)
    assert reader.checked(packet,INFERENCE)[0]==packet['payload']['packet']['payload']['features']
    with routes.selected([packet]):analysis=actual_pipeline(monkeypatch)
    i=int(away);row=analysis.iloc[i];assert row.ml_inference_status=='success'
    assert row.ml_probability==pytest.approx(baseline.iloc[i].ml_probability,abs=1e-12)
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    frame=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer']);display=routes.private_display(frame.iloc[i])
    assert display['raw_probability']==row.ml_probability and display['wager_action']=='PASS' and display['stake']==0 and display['calibrated'] is False
    assert json.loads(frame.iloc[i].ml_estimate_metadata)['nfl_private_inputs']['payload']['document_packet']==original
    assert 'body_bytes_base64' not in json.dumps(saved['package']) and all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    for family in ('overall','sides','totals'):assert len(saved['package']['games'][family])==1
    app=AppTest.from_function(private_app);app.session_state['private_candidates']=frame;app.run()
    assert not app.exception and any(m.value==f'{row.ml_probability:.2%}' for m in app.metric)
    assert any('wire bytes' in c.value for c in app.caption) and packet==original


@pytest.mark.parametrize('attack',['body','receipt','excerpt','subject','late_review','stale','started','applicability','edition','override','orientation','price','line','old_review'])
def test_tampering_and_unsupported_facts_reject_before_inference(monkeypatch,tmp_path,attack):
    packet,baseline=fixture(monkeypatch,tmp_path);p=packet['payload'];inner=p['packet']['payload']
    if attack=='body':p['document_captures']['rules']['bundle']['payload']['objects'][0]['body_bytes_base64']='QQ=='
    elif attack=='receipt':p['document_captures']['rules']['bundle']['payload']['objects'][0]['receipt']['observed_at']='2026-10-06T19:26:00Z'
    elif attack=='excerpt':inner['projections']['rules']['payload']['facts']['period']['supports'][0]['pdf']['exact']='first half'
    elif attack=='subject':p['owner_review']['subject_sha256']='0'*64
    elif attack=='late_review':p['owner_review']['reviewed_at']='2026-10-06T19:26:00Z'
    elif attack=='stale':inner['quote']['recorded_at']='2026-10-06T18:00:00Z'
    elif attack=='started':inner['event']['start']='2026-10-06T18:00:00Z'
    elif attack=='applicability':inner['projections']['listing']['payload']['values']['applicability']['listing_id']='OTHER'
    elif attack=='edition':inner['projections']['listing']['payload']['values']['rule_edition']='NFL-001 edition 2026-10-02'
    elif attack=='override':inner['projections']['listing']['payload']['values']['overtime']=False
    elif attack=='orientation':inner['event']['home'],inner['event']['away']=inner['event']['away'],inner['event']['home']
    elif attack=='price':inner['quote']['price']+=10
    elif attack=='line':inner['quote']['point']+=1
    else:p['owner_review']['version']='nfl-owner-pdf-verification-v5'
    # Rehashing is not owner evidence or a correction of contradictory facts.
    p['packet']=acquisition.seal(inner)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:pytest.fail('REJECTED DOC REACHED INFERENCE'))
    try:
        with routes.selected([acquisition.seal(p)]):result=reader.predict(baseline.iloc[0])
    except ValueError:pass
    else:assert result['ml_inference_status']=='unavailable' and result['ml_unavailable_reason'] in reader.REASONS


def test_old_packet_cannot_borrow_new_review(monkeypatch,tmp_path):
    from app_core import nfl_pdf_research as old
    packet,_=fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='PACKET_SCHEMA'):old.checked(packet,INFERENCE)
    inner=packet['payload']['packet'];assert old.checked(inner,INFERENCE)
    with pytest.raises(ValueError,match='PACKET_SCHEMA'):reader.checked(inner,INFERENCE)


def test_rehashed_retrieval_linkage_still_must_match_originals(monkeypatch,tmp_path):
    packet,_=fixture(monkeypatch,tmp_path);p=packet['payload'];b=p['document_captures']['rules']['bundle']['payload']
    b['objects'][0]['receipt']['observed_at']='2026-10-06T19:25:02.100Z'
    p['document_captures']['rules']['bundle']=acquisition.seal(b);review(p)
    with pytest.raises(ValueError,match='DOCUMENT_BINDING_CONFLICT'):reader.checked(acquisition.seal(p),INFERENCE)
