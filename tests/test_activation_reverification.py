import pandas as pd
import pytest
from app_core.candidate_evidence_schema import project, pool_status, FIELDS
from core.activation_studies import studies

pytestmark = pytest.mark.activation_acceptance


def test_per_game_pool_preserves_expected_missing_candidates():
    rows=pd.DataFrame([dict(snapshot_id='s',matchup_id=g,league='NFL',market_type=m,best_pick=m,odds_american=-110,best_available_candidate_count=2) for g,m in [('a','spread_home'),('a','spread_away'),('b','total_over')]])
    saved=project(rows)
    assert set(FIELDS)<=set(saved)
    assert pool_status(saved[saved.matchup_id.eq('a')])
    assert not pool_status(saved[saved.matchup_id.eq('b')])
    assert saved.slate_id.isna().all()
    assert saved.model_trained_through.isna().all()


def test_different_line_candidates_have_distinct_identity():
    rows=pd.DataFrame([dict(snapshot_id='s',matchup_id='g',market_type='total_over',best_pick='Over',total_line=v,odds_american=-110) for v in [8,8.5]])
    assert project(rows).candidate_id.nunique()==2


def test_missing_study_inputs_do_not_pass():
    row=dict(sport='NFL',market_type='spread_home',candidate_outcome='WIN',conservative_probability=.6)
    result=studies([row],{})
    assert 'missing_probability_intervals' in result['uncertainty']['blockers']
    assert result['prior_validation']['blockers']
    assert result['ml_ablation']['with_ml']['blocker']
    assert result['premium']['n']==0


def test_study_reports_band_coverage_without_binary_interval_claim():
    rows=[dict(sport='NFL',market_type='spread_home',candidate_outcome=o,conservative_probability=.6,probability_interval_lower=.5,probability_interval_upper=.9) for o in ['WIN','WIN','LOSS']]
    r=studies(rows,{'uncertainty_quantile':.1})
    assert r['uncertainty']['lower_coverage']==1
    assert 'not_individual_binary' in r['uncertainty']['method']


def test_recommendation_after_commit_cannot_erase_exposure(tmp_path):
    from activation_fixture import setup, NOW
    from core.exposure_ledger import append, snapshot
    ledger=tmp_path/'l.db';setup(ledger)
    e={'status':'COMMITTED','bet_id':'x','source_snapshot_id':'s','sportsbook':'DraftKings','stake_dollars':2.,'legs':[{'sport':'NFL','game_id':'g','team_ids':['a','b'],'market':'spread_home','selection':'a -2.5','line':-2.5,'odds':-110}]}
    append(ledger,e,confirmed=True,now=NOW)
    append(ledger,dict(e,status='RECOMMENDED'),confirmed=True,now=NOW)
    assert snapshot(ledger,now=NOW)['committed']['total']==.002
    append(ledger,{'status':'SETTLED','bet_id':'x'},confirmed=True,now=NOW)
    assert snapshot(ledger,now=NOW)['committed']['total']==0


def test_changed_actual_line_does_not_reuse_probability():
    from core.owner_wager_records import placement_record
    c={'sport':'NFL','game_id':'g','market_type':'spread_home','selection':'a -2.5','line':-2.5,'odds':-110,'conservative_probability':.6}
    r=placement_record(c,sportsbook='DraftKings',line=-4.5,odds=-110,stake=1,bet_id='b',snapshot_id='s',team_ids=['a','b'])
    assert r['actual_conservative_ev'] is None
    assert r['value_warning']


def test_automatic_close_uses_provider_namespace_and_all_directions(tmp_path,monkeypatch):
    from datetime import timedelta
    from activation_fixture import NOW
    from app_core import prediction_evidence as pe
    from app_core.activation_closing import capture_live,observations
    import json
    row={'snapshot_id':'s','candidate_id':'c','sport':'MLB','matchup_id':'g','game_id':'g','market_type':'spread_home','game_start_utc':(NOW+timedelta(minutes=10)).isoformat(),'provider_namespace':'odds_api','provider_event_id':'p','quote_bookmaker':'novig','market_line_used':-1.5,'odds_american':-110}
    monkeypatch.setattr(pe,'load_snapshots',lambda _: [('s',pd.DataFrame([row]),pd.DataFrame())])
    q={'provider_namespace':'odds_api','provider_event_id':'p','book':'novig','market_type':'spread_home','point':-1.5,'price':-120,'recorded_at':NOW.isoformat()}
    fetch=lambda _:pd.DataFrame([{'matchup_id':'g','provider_quotes':json.dumps([q])}])
    db=tmp_path/'c.db'
    assert capture_live(db,fetch=fetch,now=NOW)['verified']==1
    q['provider_namespace']='espn'
    assert capture_live(db,fetch=fetch,now=NOW)['unavailable']==1
    assert len(observations(db))==1


def test_outcome_refresh_uses_deterministic_match_and_preserves_provider(tmp_path,monkeypatch):
    from app_core import prediction_evidence as pe
    row={'snapshot_id':'s','matchup_id':'g','sport':'MLB','league':'MLB','home_team':'Chicago Cubs','away_team':'Pittsburgh Pirates','game_start_utc':'2026-09-01T17:00:00+00:00','market_type':'spread_away','best_pick':'Pittsburgh Pirates +1.5','candidate_outcome':'PENDING'}
    monkeypatch.setattr(pe,'materialize',lambda _: (pd.DataFrame([row]),pd.DataFrame()))
    captured=[]
    monkeypatch.setattr(pe,'record_scores',lambda frame,**kw:captured.extend(frame.to_dict('records')) or len(frame))
    score={'sport':'MLB','home':'Chicago Cubs','away':'Pittsburgh Pirates','start':'2026-09-01T19:00:00+00:00','home_score':2,'away_score':1,'result_source':'ESPN','provider_event_id':'x','event_id':'x','completed':True}
    result=pe.refresh_outcomes(tmp_path/'x.db',fetch=lambda day,sports:{'recorded_at':'2026-09-02T00:00:00+00:00','scores':[score],'events':[score]})
    assert result['revisions']==1
    assert captured[0]['result_source']=='ESPN'
    assert captured[0]['result_provider_event_id']=='x'


def test_prospective_uncertainty_cannot_use_same_slate_or_other_sport():
    from core.prospective_uncertainty import forecast
    c={'sport':'NFL','season':2026,'slate_id':'NFL:2026:WEEK_03','market_type':'spread_home','calibrated_probability':.6,'prediction_generated_at':'2026-09-14T12:00:00Z','model_version':'m','calibration_version':'c'}
    r=dict(c,game_id='g',candidate_id='id',candidate_outcome='WIN',outcome_recorded_at='2026-09-13T22:00:00Z')
    settings={'historical_prior_decay':.8,'historical_effective_sample_cap':20,'current_season_weight':1,'parent_weight':.5,'uncertainty_quantile':.1}
    assert forecast(c,[r],settings)['uncertainty_status']=='NO_PRIOR_ADMISSIBLE_SLATES'
    assert forecast(c,[dict(r,sport='MLB',slate_id='past')],settings)['uncertainty_status']=='NO_PRIOR_ADMISSIBLE_SLATES'
    value=forecast(c,[dict(r,slate_id='NFL:2026:WEEK_02')],settings)
    assert value['conservative_probability']<=.6
    assert value['current_season_weight']==1
    assert value['uncertainty_status']=='PROSPECTIVE_ESTIMATE_NOT_DEPLOYMENT_AUTHORITY'
