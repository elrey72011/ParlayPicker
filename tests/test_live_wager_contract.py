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


def test_live_candidate_first_selects_qualified_runner_up(tmp_path):
    from activation_fixture import setup
    base,validated_policy,config=setup(tmp_path/'ledger.db')
    validated_policy=_bound_study(validated_policy,config,market_family='total')
    high=dict(base,selection='Home -2.5',best_pick='Home -2.5',
              conservative_probability=.7,mean_probability=.75,odds_american=-400)
    low=dict(base,selection='Over 42.5',best_pick='Over 42.5',market_type='total_over',
             line=42.5,total_line=42.5)
    best=pd.DataFrame([dict(high,gemini_review_status='APPROVE')])
    config['gemini_outage']={'mode':'capped','cap':.001,'multiplier':.5}
    out,audit=finalize_live_wagers(pd.DataFrame([high,low]),best,1000,now=NOW,
                                   policies={'NFL':validated_policy},config=config)
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


def _bound_study(policy, config, **updates):
    from dataclasses import replace
    from core.exposure_ledger import digest
    study=config['validation_results'][policy.sport]
    study.update(updates)
    study.pop('validation_hash',None)
    study['validation_hash']=digest(study)
    bound=replace(policy,validation_id=study['validation_hash'])
    from activation_fixture import bind_test_market
    bind_test_market(bound,config)
    return bound


@pytest.mark.parametrize('family,market',[('spread','spread_home'),('total','total_under')])
def test_verified_family_populates_without_candidate_self_assertion(tmp_path,family,market):
    from activation_fixture import setup,run
    r,p,c=setup(tmp_path/'ledger.db')
    r.pop('validated_evidence_family')
    r.update(market_type=market, total_line=8.5 if family=='total' else None)
    if family=='total':r.update(line=8.5,selection='Under 8.5',best_pick='Under 8.5')
    p=_bound_study(p,c,market_family=family)
    result=run(r,p,c)
    assert result['validated_evidence_family']==family
    assert result['production_bet_amount']>0


@pytest.mark.parametrize('change',[
    {'market_type':'total_over'}, {'model_version':'wrong'}, {'calibration_version':'wrong'},
    {'evidence_version':'wrong'}, {'sport_policy_version':'wrong'}, {'selection_policy_version':'wrong'},
    {'model_validated':None}, {'calibration_validated':None}, {'model_trained_through':None},
    {'slate_id':'NFL:TEST:PAST'}, {'prediction_generated_at':(NOW-timedelta(days=2)).isoformat()},
])
def test_invalid_family_context_cannot_promote(tmp_path,change):
    from activation_fixture import setup,run
    r,p,c=setup(tmp_path/'ledger.db');r.update(change)
    result=run(r,p,c)
    assert result['production_bet_amount']==0
    assert result['maturity'] not in {'PROVISIONAL','STANDARD','PREMIUM'}


@pytest.mark.parametrize('failure',['missing','sport','hash','expired'])
def test_unverified_study_cannot_supply_family(tmp_path,failure):
    from activation_fixture import setup,run
    r,p,c=setup(tmp_path/'ledger.db')
    if failure=='missing':c['validation_results']={}
    if failure=='sport':p=_bound_study(p,c,sport='NCAAF')
    if failure=='hash':c['validation_results']['NFL']['metrics']={'price_clv_lower_95':1}
    if failure=='expired':p=_bound_study(p,c,expires_at=NOW.isoformat())
    result=run(r,p,c)
    assert result['production_bet_amount']==0
    assert result.get('validated_evidence_family') is None


def test_eligible_alternative_and_canonical_order(tmp_path):
    from activation_fixture import setup
    r,p,c=setup(tmp_path/'ledger.db')
    first=dict(r,candidate_id='first',identity_verified=False,mean_probability=.8)
    other=dict(r,candidate_id='other',market_type='spread_away',line=2.5,spread_line=2.5,
               selection='Away +2.5',best_pick='Away +2.5')
    rows=pd.DataFrame([first,other])
    out,audit=finalize_live_wagers(rows,rows.iloc[:1],1000,now=NOW,policies={'NFL':p},config=c,reviews=rows)
    assert out.iloc[0]['candidate_id']=='other'
    assert out.iloc[0]['best_pick']=='Away +2.5'
    assert out.iloc[0]['production_bet_amount']>0
    assert len(audit)==2
    # Both eligible, deterministic ties and conservative EV take precedence.
    first['identity_verified']=True
    for probabilities,winner in [((.58,.60),'other'),((.60,.58),'first'),((.58,.58),'other')]:
        first['conservative_probability'],other['conservative_probability']=probabilities
        rows=pd.DataFrame([first,other])
        out,_=finalize_live_wagers(rows.iloc[::-1],rows.iloc[:1],1000,now=NOW,policies={'NFL':p},config=c,reviews=rows)
        assert out.iloc[0]['candidate_id']==winner
    rows['identity_verified']=False
    out,_=finalize_live_wagers(rows,rows.iloc[:1],1000,now=NOW,policies={'NFL':p},config=c,reviews=rows)
    assert out.iloc[0]['production_bet_amount']==0
    money=dict(other,market_type='moneyline_home',conservative_probability=.99)
    out,audit=finalize_live_wagers(pd.concat([rows,pd.DataFrame([money])]),rows.iloc[:1],1000,now=NOW,policies={'NFL':p},config=c,reviews=rows)
    assert out.iloc[0]['production_bet_amount']==0
    assert len(audit)==2


@pytest.mark.parametrize('period',['daily','weekly'])
def test_terminal_authority_passes_absolute_period_caps(tmp_path,monkeypatch,period):
    from dataclasses import replace
    from activation_fixture import setup,bind_test_market
    from core.exposure_ledger import digest
    import core.live_wager_contract as live
    r,p,c=setup(tmp_path/'ledger.db')
    c['automatic_maturity']=False
    c['exposure'].update(total_cap=.1,daily_cap=.2,weekly_cap=.2,game_cap=.1,team_cap=.1,
                         committed={'total':.04,period:.04})
    c['exposure'][period+'_cap']=.05
    c['exposure']['snapshot_hash']=digest({k:v for k,v in c['exposure'].items() if k!='snapshot_hash'})
    p=replace(p,sport_exposure_cap=.1)
    bind_test_market(p,c)
    def decision(row,*args,**kwargs):
        return dict(row,production_eligible=True,conservative_ev=.1,recommended_fraction=.02,
                    strategic_action='BET NOW',reason_for_pass=[])
    monkeypatch.setattr(live,'candidate_decision',decision)
    rows=pd.DataFrame([r])
    out,_=live.finalize_live_wagers(rows,rows,1000,now=NOW,policies={'NFL':p},config=c)
    assert out.iloc[0]['production_bet_amount']==pytest.approx(10.)
    assert c['exposure']['committed']=={'total':.04,period:.04}
