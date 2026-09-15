"""TEST-ONLY engineering fixtures. Never production evidence."""
from dataclasses import replace
from datetime import datetime,timezone,timedelta
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd
from core.exposure_ledger import append,snapshot,digest
from core.live_wager_contract import finalize_live_wagers
from core.sport_policy import SportPolicy

NOW=datetime(2026,9,14,15,tzinfo=timezone.utc)

def setup(path):
    append(path,dict(status='CONFIGURED',bankroll=1000.,unit_value=1.,currency='USD',total_cap=.05,daily_cap=.05,weekly_cap=.1,game_cap=.01,team_cap=.01),confirmed=True,now=NOW)
    p=SportPolicy('NFL','TEST-ONLY-v1',validation_id='TEST-ONLY-NFL-PROVISIONAL-001',deployment_state='PROVISIONAL_VALIDATED',provisional_allowed=True,provisional_stake_cap=.0025,kelly_fraction=.1,sport_exposure_cap=.03)
    r=dict(sport='NFL',league='NFL',game_id='one',matchup_id='one',home_team='Home',away_team='Away',market_type='spread_home',line=-2.5,spread_line=-2.5,selection='Home -2.5',best_pick='Home -2.5',team_ids=['home','away'],odds_american=-110,book='DraftKings',conservative_probability=.58,mean_probability=.62,identity_verified=True,exact_quote_verified=True,start=(NOW+timedelta(hours=3)).isoformat(),quote_time=(NOW-timedelta(minutes=5)).isoformat(),prediction_generated_at=NOW.isoformat(),model_trained_through=(NOW-timedelta(days=2)).isoformat(),model_available_at=(NOW-timedelta(days=1)).isoformat(),calibration_available_at=(NOW-timedelta(days=1)).isoformat(),selection_policy_version='s1',sport_policy_version='TEST-ONLY-v1',evidence_version='e1',model_validated=True,model_version='m1',calibration_validated=True,calibration_version='c1',evidence_frozen_at=(NOW-timedelta(days=1)).isoformat(),evidence_snapshot_id='TEST-ONLY',evidence_effective_sample_size=30,critical_feature_error=False,gemini_review_status='APPROVE',calibration_uncertainty=.02,prior_clv_lower=.01,current_regime_conflict=False,validated_evidence_family='spread')
    cfg={'automatic_maturity':True,'exposure':snapshot(path,now=NOW),'validation_results':{'NFL':{'validation_through':(NOW-timedelta(days=1)).isoformat(),'validation_slates':['NFL:TEST:PAST'],'metrics':{'price_clv_lower_95':.01},'market_family':'spread','versions':{'model_version':'m1','calibration_version':'c1','selection_policy_version':'s1','sport_policy_version':'TEST-ONLY-v1','evidence_version':'e1'},'supported_policy':{'exposure_limits':{'total_cap':.05,'daily_cap':.05,'weekly_cap':.1,'game_cap':.01,'team_cap':.01},'maturity_rules':{'PROVISIONAL':{'max_uncertainty':.05,'min_prior_clv':0}}}}}}
    r['slate_id']='NFL:TEST:FUTURE'
    validation=cfg['validation_results']['NFL']
    validation.update(sport='NFL', deployment_state=p.deployment_state,
                      expires_at=(NOW+timedelta(days=1)).isoformat())
    validation['validation_hash']=digest(validation)
    p=replace(p,validation_id=validation['validation_hash'])
    return r,p,cfg

def run(row,policy,config):
    out,_=finalize_live_wagers(pd.DataFrame([row]),pd.DataFrame([row]),1000,now=NOW,policies={'NFL':policy},config=config)
    return out.iloc[0]

def engineering_proof(parlay=False):
    with TemporaryDirectory() as folder:
        r,p,c=setup(Path(folder)/'ledger.sqlite3')
        normal=run(r,p,c);amount=float(normal['production_bet_amount'])
        assert 0<amount<=2.5
        controls=[{'identity_verified':False},{'conservative_probability':.2},{'quote_time':(NOW-timedelta(hours=1)).isoformat()},{'start':NOW.isoformat()},{'model_version':None},{'calibration_version':None},{'evidence_snapshot_id':None},{'gemini_review_status':'HARD_VETO'},{'market_type':'moneyline_home'}]
        for change in controls:assert run(dict(r,**change),p,c)['production_bet_amount']==0
        assert run(r,replace(p,deployment_state='UNVALIDATED'),c)['production_bet_amount']==0
        import copy
        outage=copy.deepcopy(c);outage['gemini_outage']={'mode':'capped','cap':.001,'multiplier':.5}
        reduced=run(dict(r,gemini_review_status='TIMEOUT'),p,outage)
        assert 0<reduced['production_bet_amount']<amount and reduced['production_bet_amount']<=1
        assert reduced['maturity']==normal['maturity']
        assert run(dict(r,gemini_review_status='TIMEOUT'),p,c)['production_bet_amount']==0
        if parlay:
            from test_live_wager_contract import leg
            from app_core.production_parlays import build_production_parlays
            from core.parlay_confirmation import recommend
            legs=[leg(0),leg(1)]
            ticket=build_production_parlays(legs,NOW)[0]
            for i,l in enumerate(ticket['legs']):l['team_ids']=[f'home{i}',f'away{i}']
            confirmation={'ticket_hash':digest(ticket),'sportsbook':ticket['sportsbook'],'confirmed_at':NOW.isoformat(),'decimal_odds':10.}
            pp={'joint_method':'frechet_lower','validation_id':'TEST-ONLY','expires_at':(NOW+timedelta(days=1)).isoformat(),'stake_cap':.001,'kelly_fraction':.1,'sport_caps':{'NFL':.03}}
            assert recommend(ticket,None,pp,c['exposure'],now=NOW)['recommended_stake']==0
            assert recommend(ticket,confirmation,pp,c['exposure'],now=NOW)['recommended_stake']>0
            assert recommend(ticket,dict(confirmation,decimal_odds=1.01),pp,c['exposure'],now=NOW)['recommended_stake']==0
        return {'ENGINEERING_NONZERO_PATH_VERIFIED':True,'POSITIVE_PATHS':1,'NEGATIVE_CONTROL_FAILURES':0,'AUTOMATED_BET_PLACEMENT':False,'normal_stake':amount,'outage_stake':float(reduced['production_bet_amount'])}
