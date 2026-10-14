"""Existing NCAAF staging, manufactured clocks/evidence; no reader changes."""
from copy import deepcopy
from datetime import timedelta
import json
import pytest
from app_core import ncaaf_pilot as capture,ncaaf_pilot_discovery as discovery,ncaaf_model_compatibility as model
from app_core.ncaaf_history import timestamp
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
from test_ncaaf_pilot_discovery import fixture as discovery_fixture,trust,event,body
from test_ncaaf_pilot import fixture as capture_fixture,Stream
from test_ncaaf_owner_research import fixture as owner_fixture,actual

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def test_unknown_id_discovery_then_advance_listing_then_capture_then_owner_caller(synthetic,monkeypatch,tmp_path):
    owner,row=owner_fixture(synthetic,monkeypatch)
    obs=owner['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']
    terms=obs['source_review']['terms_review'];q=obs['quote']
    before=timestamp(terms['reviewed_at'])-timedelta(minutes=10);at=before.isoformat()
    plan,auth=discovery_fixture(monkeypatch,tmp_path);p=plan['payload']
    p['target'].update(home_team=q['event_home_team'],away_team=q['event_away_team'],start_utc=q['event_start_utc'])
    p['execution_window']=dict(not_before=(before-timedelta(seconds=1)).isoformat(),not_after=(before+timedelta(seconds=1)).isoformat(),authorization_expires_at=(before+timedelta(seconds=2)).isoformat())
    p['advance_permission'].update(reviewed_at=(before-timedelta(days=1)).isoformat(),effective_from=(before-timedelta(days=2)).isoformat(),target_sha256=model.digest(p['target']))
    p['requests'][0]['params'].update(commenceTimeFrom=q['event_start_utc'],commenceTimeTo=(timestamp(q['event_start_utc'])+timedelta(seconds=1)).isoformat())
    p['advance_permission']['requests_sha256']=model.digest(p['requests']);plan=discovery.seal(p);auth=trust(plan,monkeypatch)
    auth['payload'].update(authorized_at=(before-timedelta(seconds=30)).isoformat(),expires_at=p['execution_window']['authorization_expires_at']);auth=discovery.seal(auth['payload'])
    monkeypatch.setattr(discovery,'AUTHORIZED_DISCOVERIES',{auth['payload']['reference']:auth['sha256']})
    raw=body([dict(event(q['provider_event_id']),home_team=q['event_home_team'],away_team=q['event_away_team'],commence_time=q['event_start_utc'])])
    found=discovery.acquire(plan,auth,root=tmp_path,transport=lambda *a:Stream(raw),wall=lambda:at)
    assert found['payload']['status']=='DISCOVERED_UNACCEPTED' and not found['payload']['inference']
    assert 'provider_event_id' not in p['target'] and 'quote_sha256' not in model.encode(plan).decode()
    planned,*_=capture_fixture(synthetic,monkeypatch,root=tmp_path)
    assert capture.validate_plan(planned,executable=True)['target']['provider_event_id']==q['provider_event_id']
    assert timestamp(at)<timestamp(terms['reviewed_at'])<timestamp(obs['source_review']['quote_observation']['observed_at'])
    # Fully reviewed manufactured inputs use the unchanged actual owner caller.
    from app_core import ncaaf_pipeline_evidence as adapter
    with adapter.selected([owner],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='success' and result['ml_probability']==pytest.approx(.636830651175619)
    # Identifier discovery cannot substitute for original response custody/review.
    from app_core import ncaaf_owner_research as reader
    with pytest.raises((ValueError,KeyError,TypeError)):reader.view(found)
