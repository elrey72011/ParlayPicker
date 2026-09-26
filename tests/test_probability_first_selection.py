"""The owner-selected objective is candidate win chance, not composite value."""
import pandas as pd
from pregame_selection_fixture import build_pregame_best_picks_df
import pytest
from core.streamlit_pipeline import build_best_picks_df, classify_best_available_picks
from app_core.per_game_boards import per_game_board
from test_best_available_candidate_audit import _candidate
from test_per_game_boards import final


@pytest.fixture(autouse=True)
def no_fitted_history(monkeypatch):
    monkeypatch.setattr('core.empirical_tiers.load_bucket_stats',lambda:{})
    monkeypatch.setattr('core.probability_calibration.load_calibration',lambda:None)


@pytest.mark.parametrize('league',['MLB','NFL','NCAAF','WNBA'])
def test_highest_forecast_wins_across_families_despite_value_and_direction(league):
    rows=[_candidate('spread_home',.58,.4,league=league),_candidate('spread_away',.42,.5,league=league),
          _candidate('total_over',.68,-.1,league=league),_candidate('total_under',.32,.7,league=league)]
    for row in rows:
        if row['market_type'].startswith('total'):
            if league in {'NFL','NCAAF'}:row.update(total_line=44.5,live_total_line=44.5)
            if row['market_type']=='total_over': row['odds_american']=-400
            row['kalshi_probability']=.25 if row['market_type']=='total_over' else .75
    d={};best=build_pregame_best_picks_df(pd.DataFrame(rows),diagnostics_out=d)
    assert best.iloc[0].market_type=='total_over'
    assert best.iloc[0].best_available_probability==pytest.approx(.68)
    audit=d['candidate_audit_df'];winner=audit[audit.best_available_selected].iloc[0]
    assert winner.best_available_score==pytest.approx(.68)
    assert winner.best_available_rank==1
    assert winner.best_available_score_gap==pytest.approx(.10)
    assert winner.best_available_selection_policy=='probability-first-v1'
    assert not audit.wager_approved.any()
    classified=classify_best_available_picks(best)
    assert not classified.wager_approved.any()
    assert classified.filter(items=['Play_Stake','Kelly_Bet_Size','production_bet_amount']).fillna(0).sum().sum()==0


def test_exact_ties_repeat_independently_of_input_order_and_ev():
    rows=[_candidate('total_over',.60,10),_candidate('spread_home',.60,-.1)]
    a=build_pregame_best_picks_df(pd.DataFrame(rows));b=build_pregame_best_picks_df(pd.DataFrame(rows[::-1]))
    assert a.iloc[0].market_type==b.iloc[0].market_type=='spread_home'


@pytest.mark.parametrize('invalid',[None,float('nan'),float('inf'),1.5,-.2,True])
def test_invalid_estimate_never_becomes_a_fifty_percent_candidate(invalid):
    d={};best=build_pregame_best_picks_df(pd.DataFrame([_candidate('total_under',invalid,.5),_candidate('spread_home',.45,-.1)]),diagnostics_out=d)
    assert best.iloc[0].market_type=='spread_home'
    missing=d['candidate_audit_df'].query("market_type == 'total_under'").iloc[0]
    assert pd.isna(missing.best_available_probability)
    assert missing.best_available_probability_source=='unavailable'


def test_all_missing_forecasts_remain_coverage_only():
    best=build_pregame_best_picks_df(pd.DataFrame([_candidate('total_under',None,.5),_candidate('spread_home',None,.5)]))
    assert len(best)==1
    assert pd.isna(best.iloc[0].best_available_probability)
    classified=classify_best_available_picks(best)
    assert not classified.wager_approved.any()
    assert classified.filter(items=['Play_Stake','Kelly_Bet_Size','production_bet_amount']).fillna(0).sum().sum()==0


def test_public_estimate_matches_selection_probability_and_price():
    board=pd.DataFrame([final(production_win_probability=.51,best_available_selection_policy='probability-first-v1',
          best_available_probability=.72,best_available_probability_source='calibrated_probability_pair_normalized')])
    row=per_game_board(board).iloc[0]
    assert row.win_probability==.72
    assert row.probability_basis=='Candidate win estimate (pair-normalized)'
    assert row.ev==pytest.approx(.72*(1+100/110)-1)
    assert row.edge==pytest.approx(.72-110/210)
    board.loc[0,'best_available_probability']=None
    row=per_game_board(board).iloc[0]
    assert row.win_probability is None
    assert not row.Bettable and row.Play_Stake==0


def test_nfl_board_labels_context_model_and_retains_evidence():
    board = pd.DataFrame([final(
        league="NFL",
        production_win_probability=.51,
        best_available_selection_policy="probability-first-v1",
        best_available_probability=.56,
        best_available_probability_source="calibrated_probability",
        ml_probability_source="score-distribution-v1:nfl",
        selection_probability_source="nfl_score_distribution_recent_form_injury",
        nfl_context_status="complete",
        feature_home_last_game_summary="L 7-27 at PHI (2026-09-13)",
        feature_away_last_game_summary="W 28-20 vs DAL (2026-09-13)",
        injury_home_summary="Puka Nacua (WR) Questionable",
        injury_away_summary="",
        injury_context_source="espn_injuries",
        injury_context_status="available",
    )])

    row = per_game_board(board).iloc[0]

    assert row.probability_basis.startswith("NFL score model + market")
    assert row.home_recent_result.startswith("L 7-27")
    assert "Puka Nacua" in row.home_injury_context
    assert row.nfl_context_status == "complete"
