"""Exact evidence-bound aliases, not fuzzy matching or global registration."""
import base64
from copy import deepcopy
import hashlib
import json
import pytest
from app_core import ncaaf_owner_event_crosswalk as aliases, ncaaf_model_compatibility as model
from app_core import ncaaf_owner_mapping as old
from app_core.ncaaf_history import timestamp
from test_ncaaf_staged_acquisition import prepared, synthetic, offline
from test_ncaaf_staged_evidence import alias_review


def test_old_reader_preserved_and_explicit_alias_review(synthetic,monkeypatch,tmp_path):
    _,p=prepared(synthetic,monkeypatch,tmp_path);v=alias_review(p);before=deepcopy(v)
    args=(v['event'],v['schedule'],v['crosswalk'])
    assert aliases.validate(v,*args,'2026-10-16T18:30:00Z',evidence_label='SYNTHETIC')==v['schedule'][0]
    with pytest.raises(ValueError,match='ALIAS_UNREVIEWED'):
        old.check_mapping(*args,{},as_of=timestamp('2026-10-16T18:30:00Z'),review_policy=None)
    assert v==before


@pytest.mark.parametrize('attack',['provider','school','orientation','kickoff','duplicate','neutral','numeric','review','subject','evidence','source','late','missing','provisional'])
def test_alias_conflicts_and_rehashed_changes_reject(synthetic,monkeypatch,tmp_path,attack):
    _,p=prepared(synthetic,monkeypatch,tmp_path);v=alias_review(p);event=v['event'];schedule=v['schedule'];walk=v['crosswalk']
    if attack=='provider':walk[0]['provider_name']='Tulane Other'
    elif attack=='school':walk[0]['schedule_name']='Different school'
    elif attack=='orientation':event['home_team'],event['away_team']=event['away_team'],event['home_team']
    elif attack=='kickoff':event['start_utc']='2026-10-16T23:31:00Z'
    elif attack=='duplicate':schedule.append(deepcopy(schedule[0]))
    elif attack=='neutral':event['neutral_site']=True
    elif attack=='numeric':walk[0]['team_id']=235
    elif attack=='review':v['owner_review']['qualifications']=''
    elif attack=='subject':v['owner_review']['subject_sha256']='f'*64
    elif attack=='evidence':v['discovery']['body_b64']='W10='
    elif attack=='source':v['source_references']=[]
    elif attack=='late':v['owner_review']['reviewed_at']='2026-10-16T23:30:00Z'
    elif attack=='missing':v['owner_review']={}
    else:schedule[0]['homeId']=999
    with pytest.raises((ValueError,KeyError)):
        aliases.validate(v,event,schedule,walk,'2026-10-16T18:30:00Z',evidence_label='SYNTHETIC')


def test_rehashed_original_orientation_conflict_still_rejects(synthetic,monkeypatch,tmp_path):
    _,p=prepared(synthetic,monkeypatch,tmp_path);v=alias_review(p)
    rows=json.loads(base64.b64decode(v['discovery']['body_b64']));rows[0]['home_team']='Memphis Tigers'
    raw=model.encode(rows);v['discovery'].update(body_b64=base64.b64encode(raw).decode(),body_sha256=hashlib.sha256(raw).hexdigest())
    v['owner_review']['subject_sha256']=aliases.subject(v)
    with pytest.raises(ValueError,match='EVENT_CONFLICT'):
        aliases.validate(v,v['event'],v['schedule'],v['crosswalk'],'2026-10-16T18:30:00Z',evidence_label='SYNTHETIC')
