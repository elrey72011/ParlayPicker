import pandas as pd
from app_core.imported_recaps import import_exports, imported_selections
from app_core.public_history import report


def frames():
    r={'matchup_id':'2026-09-08|boston|seattle','export_run_id':'20260908T170000Z','league':'MLB','matchup':'Seattle at Boston','pick':'Boston +1.5','market_type':'spread_home','odds':-110,'Bettable':True,'Play_Stake':5,'status':'APPROVED','win_probability':.6,'ev':.1,'start':'2026-09-08T20:00:00Z'}
    return [pd.DataFrame([r]) for _ in range(3)]


def test_imports_deduplicate_and_never_claim_public_approval():
    batch=import_exports(*frames())
    rows=report([],[],[batch,batch])
    assert len(rows)==3
    assert all(r['group']=='Imported research' and r['published_at']=='' for r in rows)


def test_imported_original_line_grades_separately():
    from app_core.result_team_names import normalize_result_team as norm
    batch=import_exports(*frames())
    scores=[{'recorded_at':'2026-09-09T10:00:00Z','scores':[{'sport':'MLB','away':norm('Seattle Mariners'),'home':norm('Boston'),'event_id':'1','start':'2026-09-08T20:00:00+00:00','away_score':4,'home_score':3}]}]
    rows=report([],scores,[batch])
    assert all(r['outcome']=='WIN' for r in rows)
    assert 'Boston +1.5' in rows[0]['picks']


def test_mixed_run_rejected_and_late_export_excluded():
    import pytest
    f=frames();f[1]['export_run_id']='20260909T170000Z'
    with pytest.raises(ValueError):import_exports(*f)
    f=frames()
    for frame in f:frame['export_run_id']='20260908T210000Z'
    assert imported_selections([import_exports(*f)])==[]

def test_import_time_change_accepts_only_unique_same_day_event():
    from app_core.public_history import grade_leg
    from app_core.result_team_names import normalize_result_team as norm
    batch=import_exports(*frames());leg=batch['games']['overall'][0]
    score={'sport':'MLB','away':norm('Seattle Mariners'),'home':norm('Boston'),'event_id':'1','start':'2026-09-08T22:00:00+00:00','away_score':4,'home_score':3}
    assert grade_leg(leg,[score])[0]=='PENDING'
    assert grade_leg(leg,[score],imported=True)[0]=='WIN'
    assert grade_leg(leg,[score,{**score,'event_id':'2'}],imported=True)[0]=='PENDING'
