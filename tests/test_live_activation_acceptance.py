import pytest
from activation_fixture import engineering_proof,setup,NOW
from core.exposure_ledger import append,events,snapshot
pytestmark=pytest.mark.activation_acceptance

def test_A02_funded_and_negative_controls():assert engineering_proof()['ENGINEERING_NONZERO_PATH_VERIFIED']
def test_A23_A29_parlay_confirmation():assert engineering_proof(parlay=True)['NEGATIVE_CONTROL_FAILURES']==0

def test_A14_A19_commit_and_settle(tmp_path):
 p=tmp_path/'ledger.db';setup(p)
 assert snapshot(p,now=NOW)['committed']['total']==0
 event={'status':'RECOMMENDED','bet_id':'b1','sportsbook':'DraftKings','source_snapshot_id':'TEST-ONLY','stake_dollars':2.,'legs':[{'sport':'NFL','game_id':'one','team_ids':['a','b'],'market':'spread_home','selection':'a -2.5','line':-2.5,'odds':-110}]}
 append(p,event,confirmed=True,now=NOW)
 assert snapshot(p,now=NOW)['committed']['total']==0
 append(p,dict(event,status='COMMITTED'),confirmed=True,now=NOW)
 assert snapshot(p,now=NOW)['committed']['total']==.002
 append(p,{'status':'SETTLED','bet_id':'b1'},confirmed=True,now=NOW)
 assert snapshot(p,now=NOW)['committed']['total']==0
 assert len(events(p))==4

@pytest.mark.parametrize('cap',['game_cap','team_cap','daily_cap','weekly_cap','total_cap'])
def test_A16_saturated_cap(tmp_path,cap):
 from activation_fixture import run
 r,p,c=setup(tmp_path/'cap.db');c['exposure'][cap]=0
 assert run(r,p,c)['production_bet_amount']==0

def test_A15_stale_exposure(tmp_path):
 from activation_fixture import run
 from datetime import timedelta
 r,p,c=setup(tmp_path/'stale.db');c['exposure']['as_of']=(NOW-timedelta(hours=1)).isoformat()
 assert run(r,p,c)['production_bet_amount']==0

def test_A31_ledger_mutation_rejected(tmp_path):
 import sqlite3
 from core.exposure_ledger import connect
 from contextlib import closing
 p=tmp_path/'immutable.db';setup(p)
 with closing(connect(p)) as db:
  with pytest.raises(sqlite3.IntegrityError):db.execute('DELETE FROM events')

def test_A03_test_artifact_rejected_and_hashes(tmp_path):
 from core.activation_policy import build,verify
 from dataclasses import asdict
 from datetime import timedelta
 from core.exposure_ledger import digest
 import json
 _,policy,_=setup(tmp_path/'artifact.db')
 r={'sport':'NFL','deployment_state':policy.deployment_state,'test_only':True,'blockers':[],'tier_results':{'PROVISIONAL_VALIDATED':[]},'expires_at':(NOW+timedelta(days=1)).isoformat(),'supported_policy':{'policy':asdict(policy),'exposure_limits':{'total_cap':.05,'daily_cap':.05,'weekly_cap':.1,'game_cap':.01,'team_cap':.01}}}
 r['validation_hash']=digest(r)
 d=tmp_path/'validation';d.mkdir();(d/'NFL.json').write_text(json.dumps(r))
 value=build(d)
 with pytest.raises(ValueError,match='Test validation'):verify(value,now=NOW)
 verify(value,now=NOW,test_mode=True)
 value['sports']['NFL']['kelly_fraction']=1
 with pytest.raises(ValueError,match='hash'):verify(value,now=NOW,test_mode=True)

def test_A04_expiry(tmp_path):
 from core.activation_policy import build,verify
 from dataclasses import asdict
 from datetime import timedelta
 from core.exposure_ledger import digest
 import json
 _,policy,_=setup(tmp_path/'expired.db')
 r={'sport':'NFL','deployment_state':policy.deployment_state,'test_only':True,'blockers':[],'tier_results':{'PROVISIONAL_VALIDATED':[]},'expires_at':(NOW-timedelta(seconds=1)).isoformat(),'supported_policy':{'policy':asdict(policy),'exposure_limits':{'total_cap':.05,'daily_cap':.05,'weekly_cap':.1,'game_cap':.01,'team_cap':.01}}}
 r['validation_hash']=digest(r);d=tmp_path/'v';d.mkdir();(d/'NFL.json').write_text(json.dumps(r))
 with pytest.raises(ValueError,match='Expired'):verify(build(d),now=NOW,test_mode=True)

def test_A32_no_random_or_same_slate_authority():
 from core.activation_validation import reasons
 assert 'missing_slate_id' in reasons({})
 assert 'missing_prediction_timestamp' in reasons({})
 assert 'missing_model_version' in reasons({})

@pytest.mark.parametrize('change',[{'maturity':'PROVISIONAL'},{'gemini_review_status':'TIMEOUT'}])
def test_A24_A25_excluded_legs(change):
 from test_live_wager_contract import leg
 from app_core.production_parlays import build_production_parlays
 rows=[leg(0),leg(1)];rows[0]['wager_contract'].update(change)
 assert not build_production_parlays(rows,NOW)

def test_A26_cross_book():
 from test_live_wager_contract import leg
 from app_core.production_parlays import build_production_parlays
 rows=[leg(0),leg(1)];rows[0]['quote_source']='FanDuel';rows[0]['wager_contract']['sportsbook']='FanDuel'
 assert not build_production_parlays(rows,NOW)

def test_A32_same_slate_rejected(tmp_path,monkeypatch):
 from core import activation_validation as v
 from datetime import timedelta
 db=tmp_path/'e.db'
 choices={'sport':'NFL','market_family':'spread','development_through':(NOW-timedelta(days=2)).isoformat(),'versions':{k:'v1' for k in v.VERSIONS}}
 plan=v.freeze_plan(db,choices,now=NOW-timedelta(days=1))
 rows=[dict(choices['versions'],sport='NFL',market_type='spread_home',game_id=str(i),slate_id='NFL:2026:WEEK_01',game_start_utc=(NOW-timedelta(days=3) if i==0 else NOW).isoformat(),prediction_generated_at=(NOW-timedelta(hours=1)).isoformat()) for i in range(2)]
 monkeypatch.setattr(v,'read_dataset',lambda _: (rows,[]))
 with pytest.raises(ValueError,match='Same-slate'):v.validate(db,'NFL',plan)

def test_A34_missing_close_is_not_invented(tmp_path):
 from app_core.activation_closing import record_close,observations
 from datetime import timedelta
 db=tmp_path/'close.db'
 c={'game_start_utc':(NOW+timedelta(minutes=10)).isoformat(),'game_id':'1','sport':'NFL','market_type':'spread_home','quote_bookmaker':'Novig','provider_namespace':'odds_api','provider_event_id':'provider:1','snapshot_id':'s','candidate_id':'c','market_line_used':-2.5,'odds_american':-110}
 q={'game_id':'1','sport':'NFL','market_type':'spread_home','sportsbook':'Novig','provider_namespace':'odds_api','provider_event_id':'provider:1','quote_recorded_at':NOW.isoformat(),'line':-3.,'price':-110}
 with pytest.raises(ValueError):record_close(db,c,dict(q,sportsbook='DraftKings'),captured_at=NOW.isoformat())
 record_close(db,c,q,captured_at=NOW.isoformat())
 assert len(observations(db))==1
 assert observations(db)[0]['price_clv'] is None
 with pytest.raises(ValueError):record_close(db,c,q,captured_at=(NOW+timedelta(hours=1)).isoformat())
