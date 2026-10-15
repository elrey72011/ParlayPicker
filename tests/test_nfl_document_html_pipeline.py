"""SYNTHETIC V3 HTML custody -> V8 private path. No authentic numerical work."""
from copy import deepcopy
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from app_core import nfl_document_acquisition as acquisition,nfl_html_document_research as reader,nfl_private_routes as routes
from test_nfl_pdf_evidence import fixture as original_fixture,resign
from test_nfl_document_acquisition import capture_guide
from test_nfl_document_pipeline import review
from test_source_contract_pipeline import INFERENCE

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield

def fixture(monkeypatch,root):
    packet,baseline=original_fixture(monkeypatch);p=packet['payload']
    _,text=acquisition.custody.read(p['objects']['clock_document'])
    text=text.replace('https://synthetic.invalid/provider-guide',acquisition.GUIDE_URL)
    text='<html><section id="api-key"><a href="#api-key">SYNTHETIC heading</a></section><body>'+text+'\napiKey=YOUR_API_KEY</body></html>'
    capture,_,calls=capture_guide(root,text.encode(),version=acquisition.VERSION_V3)
    assert capture['payload']['status']=='CAPTURED_UNREVIEWED'
    obj=acquisition.project(capture,'clock');p['objects']['clock_document']=obj
    p['permissions']['clock_document'].update(source_id=obj['receipt']['source_id'],endpoint=obj['receipt']['endpoint'])
    projection=p['projections']['clock_document']['payload'];projection['values']['source_reference']=acquisition.GUIDE_URL
    for fact in projection['facts'].values():
        for support in fact['supports']:
            exact=support['text']['exact'].replace('https://synthetic.invalid/provider-guide',acquisition.GUIDE_URL)
            start=text.index(exact);support['text']=dict(start=start,end=start+len(exact),exact=exact)
    resign(p)
    outer=dict(version=reader.VERSION,evidence_label='SYNTHETIC',event=deepcopy(p['event']),quote=deepcopy(p['quote']),packet=acquisition.seal(p),
        document_captures=dict(clock_document=dict(bundle=capture,document_id='clock')),owner_review={})
    review(outer);outer['owner_review']['subject_sha256']=reader.subject(outer)
    return acquisition.seal(outer),baseline,calls

def test_html_v8_actual_caller_export_static_replay_private_display(monkeypatch,tmp_path):
    from test_nfl_owner_research import actual_pipeline,private_app
    from test_source_contract_pipeline import exported
    from app_core import research_replay
    from streamlit.testing.v1 import AppTest
    packet,baseline,calls=fixture(monkeypatch,tmp_path)
    assert reader.checked(packet,INFERENCE)[0]==packet['payload']['packet']['payload']['features']
    with routes.selected([packet]):frame=actual_pipeline(monkeypatch)
    assert frame.iloc[0].ml_inference_status=='success'
    assert frame.iloc[0].ml_probability==pytest.approx(baseline.iloc[0].ml_probability,abs=1e-12)
    retained=exported(monkeypatch,tmp_path,frame)
    receipt=research_replay.retain_export(retained['frames'],retained['package'],retained['card'],retained['captured'],path=retained['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=retained['db'])
    replay=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    shown=routes.private_display(replay.iloc[0])
    assert shown['raw_probability']==frame.iloc[0].ml_probability and shown['stake']==0 and shown['wager_action']=='PASS'
    assert shown['calibrated'] is False and shown['review_status']=='OWNER_REVIEWED' and shown['purpose']=='PRIVATE_RESEARCH'
    assert 'bytes_base64' not in json.dumps(saved['package'])
    assert json.loads(replay.iloc[0].ml_estimate_metadata)['nfl_private_inputs']['payload']['document_packet']==packet
    app=AppTest.from_function(private_app);app.session_state['private_candidates']=replay;app.run()
    assert not app.exception and any(m.value==f'{frame.iloc[0].ml_probability:.2%}' for m in app.metric)
    assert calls==[acquisition.GUIDE_URL] and 'YOUR_API_KEY' not in json.dumps(shown)

def test_older_reader_cannot_borrow_html_custody_or_review(monkeypatch,tmp_path):
    from app_core import nfl_public_document_research as old,nfl_public_guide_custody
    packet,_,_=fixture(monkeypatch,tmp_path)
    obj=packet['payload']['packet']['payload']['objects']['clock_document']
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_MISSING'):nfl_public_guide_custody.read(obj)
    p=deepcopy(packet['payload']);p['version']=old.VERSION;p['owner_review']['subject_sha256']=old.subject(p)
    with pytest.raises(ValueError):old.checked(acquisition.seal(p),INFERENCE)
    assert acquisition.CONTROLLED.get() is None

@pytest.mark.parametrize('attack',['body','receipt','old_subject','late_review','stale','started'])
def test_changed_evidence_or_review_rejects_before_inference(monkeypatch,tmp_path,attack):
    packet,baseline,_=fixture(monkeypatch,tmp_path);p=deepcopy(packet['payload']);inner=p['packet']['payload']
    if attack=='body':inner['objects']['clock_document']['bytes_base64']='QQ=='
    elif attack=='receipt':inner['objects']['clock_document']['receipt']['observed_at']='2026-10-06T19:26:00Z'
    elif attack=='old_subject':p['owner_review']['subject_sha256']='0'*64
    elif attack=='late_review':p['owner_review']['reviewed_at']='2026-10-06T19:26:00Z'
    elif attack=='stale':inner['quote']['recorded_at']='2026-10-06T18:00:00Z'
    else:inner['event']['start']='2026-10-06T18:00:00Z'
    p['packet']=acquisition.seal(inner)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:pytest.fail('REJECTED HTML REACHED INFERENCE'))
    try:
        with routes.selected([acquisition.seal(p)]):result=reader.predict(baseline.iloc[0])
    except ValueError:
        pass  # Load-time validation is also before numerical inference.
    else:
        assert result['ml_inference_status']=='unavailable' and result['ml_unavailable_reason'] in reader.REASONS
