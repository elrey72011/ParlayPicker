"""Audit contracts use synthetic fixtures, never evidence for live promotion."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import math
import pytest
import pandas as pd
from core.sport_policy import SportPolicy, research_policies
from core.wager_decisions import candidate_decision, select_matchups, allocate_exposure, moneyline_context
from core.football_evidence import freeze_evidence, reliability_distribution
from core.kelly_optimizer import kelly_fraction
from core.production_gate import evaluate_absolute_production_gate
from core.clv import price_clv, line_clv, closing_line_value
from app_core.public_history import grading_team_name
from app_core.public_board import pick_record

NOW = datetime(2026, 9, 14, 15, tzinfo=timezone.utc)

def policy(sport='NFL'):
    return SportPolicy(sport, 'fixture-v1', validation_id='SYNTHETIC-TEST-ONLY',
        provisional_allowed=sport in {'NFL','NCAAF'}, provisional_stake_cap=.002,
        standard_stake_cap=.01, premium_stake_cap=.02, sport_exposure_cap=.03,
        kelly_fraction=.1, historical_prior_strength=1, historical_prior_decay=.5,
        historical_effective_sample_cap=2)

def candidate(**changes):
    return dict(dict(sport='NFL', game_id='one', market_type='spread_home', line=-2.5,
        selection='Home -2.5', team_ids=['home','away'], odds_american=-110, book='testbook',
        conservative_probability=.58, mean_probability=.62,
        identity_verified=True, exact_quote_verified=True,
        start='2026-09-14T18:00:00Z', quote_time='2026-09-14T14:55:00Z',
        model_validated=True, model_version='m1', calibration_validated=True,
        calibration_version='c1', evidence_frozen_at='2026-09-14T14:00:00Z',
        evidence_snapshot_id='synthetic', evidence_effective_sample_size=30, critical_feature_error=False,
        maturity='STANDARD', gemini_status='CONFIRM'), **changes)

@pytest.mark.parametrize('market', ['moneyline_home','moneyline_away','h2h','h2h_home','ml','unknown'])
def test_no_moneyline_or_unknown_market_can_receive_stake(market):
    result=candidate_decision(candidate(market_type=market, conservative_probability=.9, mean_probability=.95),policy(),NOW)
    assert result['recommended_fraction']==0
    assert 'unsupported_production_market' in result['reason_for_pass']

@pytest.mark.parametrize('change,reason', [
    ({'quote_time':'2026-09-14T14:29:59Z'},'stale_or_missing_quote'),
    ({'quote_time':'2026-09-14T15:00:01Z'},'stale_or_missing_quote'),
    ({'start':NOW.isoformat()},'started_or_missing_start'),
    ({'line':None},'missing_exact_line'),
    ({'alternate':True},'unverified_alternate_quote'),
    ({'identity_verified':False},'unverified_mapping'),
    ({'model_validated':False},'unvalidated_model'),
    ({'calibration_validated':False},'unvalidated_calibration'),
    ({'critical_feature_error':True},'critical_features_unverified'),
    ({'evidence_frozen_at':'2026-09-15T14:00:00Z'},'missing_or_future_evidence'),
    ({'gemini_status':'UNAVAILABLE'},'gemini_hold'),
    ({'gemini_stake_multiplier':1.1},'gemini_hold'),
    ({'conservative_probability':float('nan')},'missing_or_invalid_conservative_probability'),
    ({'odds_american':float('inf')},'invalid_exact_price'),
])
def test_hard_failures_preserve_reason_and_zero_stake(change,reason):
    out=candidate_decision(candidate(**change),policy(),NOW)
    assert out['recommended_fraction']==0 and reason in out['reason_for_pass']

def test_conservative_value_selects_qualified_runner_up_not_highest_probability():
    rows=[candidate(selection='High probability expensive',conservative_probability=.7,mean_probability=.75,odds_american=-400),
          candidate(selection='Better priced',market_type='total_over',line=42.5)]
    out=select_matchups(rows,{'NFL':policy()},NOW)
    assert len(out)==1 and out[0]['selection']=='Better priced'
    assert out[0]['best_research_lean']=='High probability expensive'

def test_ml_only_matchup_remains_pass_and_missing_identity_is_not_merged():
    out=select_matchups([candidate(market_type='moneyline_home')],{'NFL':policy()},NOW)
    assert len(out)==1 and out[0]['recommended_fraction']==0 and out[0]['best_research_lean'] is None
    with pytest.raises(ValueError): select_matchups([candidate(game_id=None)],{'NFL':policy()},NOW)

def test_research_policies_are_isolated_and_cannot_fund_any_sport():
    policies=research_policies()
    assert len({id(p) for p in policies.values()})==6
    for sport,p in policies.items():
        out=candidate_decision(candidate(sport=sport),p,NOW)
        assert out['recommended_fraction']==0 and 'unvalidated_sport_policy' in out['reason_for_pass']
    with pytest.raises(ValueError): replace(policies['MLB'],provisional_allowed=True)

def test_provisional_stake_and_gemini_reduction_never_promote_or_increase():
    base=candidate_decision(candidate(maturity='PROVISIONAL'),policy(),NOW)
    reduced=candidate_decision(candidate(maturity='PROVISIONAL',gemini_status='REDUCE',gemini_stake_multiplier=.5),policy(),NOW)
    assert 0 < reduced['recommended_fraction']==base['recommended_fraction']*.5 <= .001
    bad=candidate_decision(candidate(conservative_probability=.4,gemini_status='CONFIRM'),policy(),NOW)
    assert bad['recommended_fraction']==0

def test_whole_line_requires_push_semantics_and_correct_ev():
    assert 'missing_or_invalid_push_probability' in candidate_decision(candidate(line=-3),policy(),NOW)['reason_for_pass']
    out=candidate_decision(candidate(line=-3,push_probability=.05),policy(),NOW)
    assert out['conservative_ev']==pytest.approx(.58*(1+100/110)+.05-1)
    assert out['minimum_decimal_price']==pytest.approx(.95/.58)

def test_shared_existing_exposure_caps_and_invalid_inputs():
    rows=[candidate_decision(candidate(game_id=str(i)),policy(),NOW) for i in range(5)]
    out=allocate_exposure(rows,2000,total_cap=.02,game_cap=.005,sport_caps={'NFL':.02},committed={'total':.019})
    assert sum(r['recommended_stake'] for r in out)==pytest.approx(2)
    assert all(r['recommended_fraction']<=.005 for r in out)
    with pytest.raises(ValueError): allocate_exposure(rows,float('nan'),total_cap=.1,game_cap=.1,sport_caps={})

def evidence(**changes):
    return dict(dict(id='old',sport='NFL',season=2025,week=1,probability=.6,outcome='WIN',market='spread_home',
        model_version='m1',calibration_version='c1',source_hash='original',pregame_verified=True,
        training_cutoff='2025-09-01T12:00:00Z',prediction_at='2025-09-07T12:00:00Z',
        start='2025-09-07T17:00:00Z',outcome_at='2025-09-07T21:00:00Z',recorded_at='2025-09-07T22:00:00Z'),**changes)

def freeze(rows,p=None):
    return freeze_evidence(rows,p or policy(),season=2026,week=2,frozen_at='2026-09-14T14:00:00Z',slate_start='2026-09-14T18:00:00Z')

def test_weekly_freeze_excludes_future_other_sport_duplicate_and_postgame_data():
    rows=[evidence(),evidence(),evidence(id='other',sport='MLB'),evidence(id='future',outcome_at='2026-09-15T12:00:00Z'),
          evidence(id='sameweek',season=2026,week=2),evidence(id='post',prediction_at='2025-09-07T18:00:00Z')]
    snapshot=freeze(rows)
    assert len(snapshot['records'])==1 and sum(snapshot['excluded'].values())==5
    assert snapshot['maximum_outcome_at']=='2025-09-07T21:00:00+00:00'
    assert snapshot['source_hashes']==['original']

def test_historical_mass_cap_and_snapshot_mutation_rejected():
    snapshot=freeze([evidence(id=str(i)) for i in range(10)])
    assert snapshot['historical_mass']==pytest.approx(2)
    result=reliability_distribution(.62,'spread_home',snapshot,policy(),model_version='m1',calibration_version='c1')
    assert result['status']=='RESEARCH_ONLY' and result['conservative_probability']<=.62
    assert result['p10']<result['p50']<result['p90'] and result['sd']>0
    assert result['effective_sample_size']<=2+1e-9
    snapshot['records'][0]['outcome']='LOSS'
    with pytest.raises(ValueError,match='modified'):
        reliability_distribution(.62,'spread_home',snapshot,policy(),model_version='m1',calibration_version='c1')

def test_model_versions_and_market_families_do_not_pool():
    snapshot=freeze([evidence()])
    for market,version in [('total_over','m1'),('spread_home','m2')]:
        out=reliability_distribution(.6,market,snapshot,policy(),model_version=version,calibration_version='c1')
        assert out['conservative_probability'] is None
    with pytest.raises(ValueError): freeze([],policy('MLB'))

def test_moneyline_context_requires_both_prices_and_is_never_a_wager():
    out=moneyline_context(-150,130)
    assert out['home_probability']+out['away_probability']==pytest.approx(1)
    assert not out['wager_eligible']
    assert not moneyline_context(-150,None)['available']

@pytest.mark.parametrize('p,d', [(float('nan'),2),(float('inf'),2),(.6,float('inf')),(True,2),(.6,1),(None,2)])
def test_invalid_kelly_is_zero(p,d): assert kelly_fraction(p,d)==0

def test_infinite_model_ev_cannot_pass_absolute_gate():
    assert not evaluate_absolute_production_gate(.7,.5,float('inf'))['production_gate_pass'].any()

def test_nfl_aliases_are_sport_specific_and_ambiguous_cities_stay_distinct():
    assert grading_team_name('Green Bay','NFL')==grading_team_name('Green Bay Packers','NFL')
    assert grading_team_name('Minnesota','NFL')=='MINNESOTA VIKINGS'
    assert grading_team_name('New York','NFL') != grading_team_name('New York Giants','NFL')
    assert grading_team_name('Los Angeles Rams','NFL') != grading_team_name('Los Angeles Chargers','NFL')

def test_missing_market_publication_stays_pass_but_moneyline_cannot_publish():
    row={'league':'NFL','matchup':'A at B','pick':'Unavailable','market_type':'','Bettable':True,'Play_Stake':1}
    assert pick_record(row,as_of=NOW.isoformat())['status']=='PASS'
    with pytest.raises(ValueError,match='context only'): pick_record(dict(row,market_type='moneyline_home'),as_of=NOW.isoformat())

def test_clv_never_compares_mixed_probability_bases_or_converts_points():
    assert price_clv(-110,-120,None,100)==price_clv(-110,-120)
    assert line_clv('spread_home',-2.5,-3.5)==1
    assert line_clv('spread_away',3.5,2.5)==1
    assert closing_line_value('Over',8,9,-150,150)['beat_close'] is None


def test_legacy_rolling_features_do_not_cross_team_or_sport_boundaries():
    from app_core.feature_engine import prepare_features_for_inference
    history=pd.DataFrame([dict(game_id=1,sport='NFL',commence_time='2026-09-01',home_team='A',away_team='B',home_score=50,away_score=0)])
    today=pd.DataFrame([dict(game_id=2,sport='NFL',commence_time='2026-09-14',home_team='X',away_team='Y',home_score=None,away_score=None),
                        dict(game_id=3,sport='MLB',commence_time='2026-09-14',home_team='A',away_team='B',home_score=None,away_score=None)])
    history=history.assign(home_streak=0,away_streak=0,public_betting_pct=50,sharp_money_indicator=0)
    today=today.assign(home_streak=0,away_streak=0,public_betting_pct=50,sharp_money_indicator=0)
    out=prepare_features_for_inference(history,today)
    assert out.home_win_pct.eq(.5).all() and out.away_win_pct.eq(.5).all()
    assert out.home_ppg.eq(.5).all() and out.away_ppg.eq(.5).all()


def test_calibration_flag_without_distinct_dates_is_not_production_evidence(tmp_path, monkeypatch):
    from core.probability_calibration import save_calibration,load_calibration
    path=tmp_path/'cal.json'
    monkeypatch.setattr('core.probability_calibration.DEFAULT_CALIBRATION_PATH',path)
    knots=[[.5,.45],[.7,.6]]
    for dates in ({}, {'train_end':'2026-08-16','test_start':'2026-08-16'}, {'train_end':'2026-08-17','test_start':'2026-08-16'}):
        save_calibration(knots,path,meta={'validation':dict(promotable=True,**dates)})
        assert load_calibration() is None


def test_verified_alternate_is_explicit_action_and_team_exposure_is_shared():
    out=candidate_decision(candidate(alternate=True,alternate_quote_verified=True),policy(),NOW)
    assert out['strategic_action']=='BET ALT LINE'
    rows=[out,dict(out,game_id='two')]
    funded=allocate_exposure(rows,1000,total_cap=.1,game_cap=.01,sport_caps={'NFL':.1},team_cap=.004)
    assert sum(x['recommended_fraction'] for x in funded)<=.004+1e-9
    assert allocate_exposure([dict(out,team_ids=None)],1000,total_cap=.1,game_cap=.01,sport_caps={'NFL':.1})[0]['recommended_stake']==0


def test_frozen_evidence_survives_json_roundtrip_without_policy_drift():
    import json
    snapshot=json.loads(json.dumps(freeze([evidence()])))
    result=reliability_distribution(.6,'spread_home',snapshot,policy(),model_version='m1',calibration_version='c1')
    assert result['status']=='RESEARCH_ONLY'
    with pytest.raises(ValueError):
        reliability_distribution(.6,'spread_home',snapshot,replace(policy(),historical_prior_decay=.7),model_version='m1',calibration_version='c1')


def test_cached_moneyline_cannot_be_pick_of_day_or_new_lock():
    from app_core.pick_of_day import _game_candidates
    from app_core.locked_picks import lock_candidates, lock_audit, locked_selections
    from test_public_history import pub
    at = '2026-09-09T19:56:00+00:00'
    frame = pd.DataFrame([dict(market_type='moneyline_home', Pick_Status='Actionable',
        production_eligible=True, Kelly_Bet_Size=10, best_pick='Boston ML')])
    assert _game_candidates(frame).empty
    package = pub()['package']
    old = lock_candidates(package, at)
    package['games']['overall'][0].update(market='moneyline_home', pick='Boston ML')
    assert lock_candidates(package, at) == []
    assert lock_audit(package, at)[0]['Lock status'] == 'Context only market'
    # Saved records retain their immutable identity and remain readable.
    assert locked_selections(old) == old
