"""Synthetic policy fixtures validate code, never authorize a real sport policy."""
from copy import deepcopy
from datetime import timedelta
import pandas as pd
import pytest
from test_wager_integrity_audit import candidate, policy, NOW
from core.wager_decisions import candidate_decision
from core.live_wager_contract import snapshot, enforce_frame, finalize_live_wagers
from app_core.production_parlays import production_parlay_leg_eligible, canonical_funnel, build_production_parlays
from app_core.public_parlays import build_research_parlays
from app_core.export_scope import label_wager_export


def leg(i=0, book='DraftKings', maturity='STANDARD', **changes):
    source=candidate(game_id=str(i),matchup_id=str(i),selection=f'Home{i} -2.5',book=book,maturity=maturity,**changes)
    decision=candidate_decision(source,policy(),NOW)
    decision['recommended_stake']=decision['recommended_fraction']*1000
    c=snapshot(decision,NOW)
    return dict(sport='NFL',game=f'Away{i} at Home{i}',pick=source['selection'],market='spread_home',odds=-110,
        win_estimate=.62,ev=.1,status='PASS',as_of=NOW.isoformat(),start=source['start'],quote_time=source['quote_time'],quote_source=book,wager_contract=c)


@pytest.mark.parametrize('tier',['PROVISIONAL','STANDARD','PREMIUM'])
def test_validated_maturity_stakes(tier):
    r=candidate_decision(candidate(maturity=tier),policy(),NOW)
    assert r['production_eligible'] and r['recommended_fraction']>0 and r['maturity']==tier


@pytest.mark.parametrize('change',[{'conservative_probability':.3},{'identity_verified':False},{'exact_quote_verified':False},
    {'start':NOW.isoformat()},{'market_type':'moneyline_home'},{'book':'Unsupported'},{'critical_feature_error':True},
    {'calibration_validated':False},{'model_validated':False},{'maturity':'RESEARCH'}])
def test_outage_never_promotes(change):
    r=candidate_decision(candidate(gemini_status='TIMEOUT',**change),policy(),NOW,outage_policy={'mode':'capped','cap':.001,'multiplier':.5})
    assert not r['production_eligible'] and r['recommended_fraction']==0


@pytest.mark.parametrize('status',['TIMEOUT','UNAVAILABLE','SERVICE_ERROR'])
def test_outage_cap_and_maturity(status):
    r=candidate_decision(candidate(gemini_status=status),policy(),NOW,outage_policy={'mode':'capped','cap':.001,'multiplier':.5})
    assert 0<r['recommended_fraction']<=.001 and r['maturity']=='STANDARD'
    assert r['gemini_review_status']==status and r['gemini_outage_capped']
    assert candidate_decision(candidate(gemini_status=status),policy(),NOW)['recommended_fraction']==0


def test_veto_still_blocks():
    assert candidate_decision(candidate(gemini_status='HARD_VETO'),policy(),NOW,outage_policy={'mode':'capped','cap':.001,'multiplier':.5})['recommended_fraction']==0


@pytest.mark.parametrize('tiers',[('STANDARD','STANDARD'),('STANDARD','PREMIUM'),('PREMIUM','PREMIUM')])
@pytest.mark.parametrize('book',['DraftKings','FanDuel','BetMGM','Novig'])
def test_same_book_qualified_tickets_without_legacy_approved(tiers,book):
    rows=[leg(i,book,tier) for i,tier in enumerate(tiers)]
    tickets=build_production_parlays(rows,NOW)
    assert len(tickets)==1
    t=tickets[0]
    assert t['status']=='QUALIFIED — VERIFY TICKET PRICE'
    assert t['recommended_stake']==0 and t['actual_ticket_price_verified'] is False
    assert t['win_estimate']==pytest.approx(.58**2)


@pytest.mark.parametrize('tier',['PROVISIONAL','QUALIFIED','RESEARCH'])
def test_immature_legs_do_not_qualify(tier):
    assert not production_parlay_leg_eligible(leg(maturity=tier),NOW)


@pytest.mark.parametrize('change',[{'odds':None},{'quote_time':(NOW-timedelta(hours=1)).isoformat()},
    {'start':NOW.isoformat()},{'market':'moneyline_home'},{'quote_source':'Unknown'}])
def test_parlay_integrity(change):
    r=leg();r.update(change)
    assert not production_parlay_leg_eligible(r,NOW)


def test_cross_book_duplicate_and_correlated_exclusion():
    assert build_production_parlays([leg(0),leg(1,'FanDuel')],NOW)==[]
    assert build_production_parlays([leg(0),leg(0)],NOW)==[]
    f=canonical_funnel([leg(0),leg(1,'FanDuel')],NOW)
    assert f['exclusions']['no_same_book_partner']==2


def test_funnel_and_triples():
    rows=[leg(i) for i in range(4)]
    f=canonical_funnel(rows,NOW)
    assert f['counts']['total_best_picks']==4 and f['counts']['parlay_eligible']==4
    assert f['counts']['valid_2leg_pairs']==6 and f['counts']['valid_3leg_combinations']==4
    assert len(build_production_parlays(rows,NOW))>1
    assert any(len(t['legs'])==3 for t in build_production_parlays(rows,NOW))


def test_research_multiple_reuse_dedup_limits():
    rows=[leg(i) for i in range(4)]
    out=build_research_parlays(rows+[deepcopy(rows[0])],NOW,diversified=True,top_n=5)
    assert len(out)==5
    assert all(t['recommended_stake']==0 and t['status']=='RESEARCH ONLY' for t in out)
    assert len({tuple(r['game'] for r in t['legs']) for t in out})==5
    assert len(build_research_parlays(rows,NOW,diversified=True,top_n=2))==2


def test_export_cannot_refund_failed_contract():
    c=leg()['wager_contract'];c['production_eligible']=False;c['production_bet_amount']=0
    frame=pd.DataFrame([{'wager_contract':c,'best_pick':'Home0 -2.5','market_type':'spread_home','Kelly_Bet_Size':500,'production_eligible':True,'Bettable':True}])
    out=label_wager_export(frame)
    assert out.iloc[0]['Kelly_Bet_Size']==0 and not out.iloc[0]['Bettable']


def test_live_candidate_first_selects_qualified_runner_up():
    high=candidate(league='NFL',home_team='Home',away_team='Away',matchup_id='one',selection='Home -2.5',best_pick='Home -2.5',spread_line=-2.5,conservative_probability=.7,mean_probability=.75,odds_american=-400)
    low=candidate(league='NFL',home_team='Home',away_team='Away',matchup_id='one',selection='Over 42.5',best_pick='Over 42.5',market_type='total_over',line=42.5,total_line=42.5)
    best=pd.DataFrame([dict(high,gemini_review_status='APPROVE')])
    config={'gemini_outage':{'mode':'capped','cap':.001,'multiplier':.5},'exposure':{'as_of':NOW.isoformat(),'committed':{},'total_cap':.03,'daily_cap':.03,'weekly_cap':.05,'game_cap':.01,'team_cap':.01}}
    out,audit=finalize_live_wagers(pd.DataFrame([high,low]),best,1000,now=NOW,policies={'NFL':policy()},config=config)
    assert out.iloc[0]['best_pick']=='Over 42.5'
    assert 0<out.iloc[0]['production_bet_amount']<=1
    assert len(audit)==2
    assert out.iloc[0]['wager_contract']['gemini_review_status']=='UNAVAILABLE'


def test_portfolio_cannot_override_canonical_moneyline_or_pass():
    from core.streamlit_pipeline import optimize_portfolio_allocation
    c=leg()['wager_contract'];c['market_type']='moneyline_home'
    row={'wager_contract':c,'Kelly_Bet_Size':100,'production_eligible':True,'market_type':'moneyline_home'}
    out=optimize_portfolio_allocation(pd.DataFrame([row]),1000)
    assert out.iloc[0]['production_bet_amount']==0


def test_public_package_roundtrip_preserves_contract_and_history():
    from app_core.public_board import build_package,validate_package
    from datetime import datetime
    # Build using the established complete board fixture then attach canonical evidence.
    import inspect
    from app_core.public_parlays import build_parlays
    rows=[leg(0),leg(1)]
    package={'schema_version':2,'built_at':NOW.isoformat(),'stale_after_minutes':30,
        'parlay_policy':'canonical-v3','games':{k:deepcopy(rows) for k in ['overall','sides','totals']},'props':[],'dfs':[],
        'parlays':build_production_parlays(rows,NOW)}
    for board in package['games'].values():
        for r in board:r['player']=''
    package['parlays']=build_production_parlays(package['games']['overall'],NOW)
    before=deepcopy(package)
    validate_package(package)
    assert package==before
    tampered=deepcopy(package)
    tampered['parlays'][0]['recommended_stake']=10
    with pytest.raises(ValueError):validate_package(tampered)


def test_independent_funnel_exclusions():
    r=leg();r['odds']=0;r['start']=NOW.isoformat();r['quote_time']=(NOW-timedelta(hours=1)).isoformat()
    f=canonical_funnel([r],NOW)
    assert f['exclusions']['invalid_price']==1 and f['exclusions']['game_started']==1 and f['exclusions']['stale_quote']==1
    assert f['counts']['parlay_eligible']==0


def test_total_premium_and_sport_isolation():
    from dataclasses import replace
    from core.sport_policy import research_policies
    r=candidate_decision(candidate(market_type='total_under',line=42.5,maturity='PREMIUM'),policy(),NOW)
    assert r['production_eligible']
    policies=research_policies()
    baseline={sport:candidate_decision(candidate(sport=sport,book='Novig'),p,NOW) for sport,p in policies.items()}
    policies['NFL']=policy()
    for sport in ['MLB','NBA','NHL','NCAAB']:
        assert candidate_decision(candidate(sport=sport,book='Novig'),policies[sport],NOW)==baseline[sport]


def test_missing_expired_policy_is_zero(tmp_path,monkeypatch):
    import json
    from core.live_wager_contract import load_configuration
    p=tmp_path/'policy.json';p.write_text(json.dumps({'expires_at':(NOW-timedelta(seconds=1)).isoformat(),'validated_at':(NOW-timedelta(days=1)).isoformat(),'sports':{}}))
    monkeypatch.setenv('PARLAYPICKER_WAGER_POLICY_PATH',str(p))
    policies,config,reason=load_configuration(NOW)
    assert reason and all(x.standard_stake_cap==0 for x in policies.values())
