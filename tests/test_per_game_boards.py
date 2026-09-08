import pandas as pd
from app_core.per_game_boards import per_game_board


def final(gid='g1', **updates):
    return {'matchup_id':gid,'export_run_id':'run1','league':'MLB','Home':'Home','Away':'Away','Local Date':'2026-09-08',
            'best_pick':'Home -1.5','market_type':'spread_home','odds_american':-110,
            'Bettable':True,'Play_Stake':5,'production_win_probability':.62,'production_edge':.05,'production_expected_value':.1,**updates}


def candidate(gid='g1', **updates):
    return {'matchup_id':gid,'export_run_id':'run1','league':'MLB','home_team':'Home','away_team':'Away','game_date':'2026-09-08',
            'best_pick':'Under 8.5','market_type':'total_under','odds_american':-105,'ml_probability':.61,'calibrated_probability':.55,
            'best_available_family_rank':1,'best_available_rank':2,'best_available_score':.58,**updates}


def test_all_three_views_have_each_game_and_independent_family_selection():
    board=pd.DataFrame([final(),final('g2',best_pick='Over 9',market_type='total_over',Bettable=False,Play_Stake=0)])
    audit=pd.DataFrame([candidate(),candidate('g1',best_pick='Over 8.5',market_type='total_over',best_available_family_rank=2,ml_probability=.99),
                        candidate('g2',best_pick='Away +1.5',market_type='spread_away')])
    overall=per_game_board(board,audit)
    sides=per_game_board(board,audit,'sides')
    totals=per_game_board(board,audit,'totals')
    assert len(overall)==len(sides)==len(totals)==2
    assert overall['pick'].tolist()==['Home -1.5','Over 9']
    assert sides['pick'].tolist()==['Home -1.5','Away +1.5']
    assert totals['pick'].tolist()==['Under 8.5','Over 9']
    assert not totals.Bettable.any()
    assert totals.Play_Stake.sum()==0
    assert sides.Bettable.tolist()==[True,False]


def test_stale_run_and_other_event_cannot_supply_missing_market():
    for row in [candidate(export_run_id='old'),candidate('other')]:
        result=per_game_board(pd.DataFrame([final()]),pd.DataFrame([row]),'totals')
        assert result.iloc[0]['pick']=='No Bet — market unavailable'
        assert not result.iloc[0].Bettable


def test_only_exact_price_final_ticket_inherits_approval():
    board=pd.DataFrame([final()])
    row=candidate(best_pick='Home -1.5',market_type='spread_home',odds_american=-110)
    matched=per_game_board(board,pd.DataFrame([row]),'sides').iloc[0]
    assert matched.Bettable and matched.Play_Stake==5
    row['odds_american']=-115
    changed=per_game_board(board,pd.DataFrame([row]),'sides').iloc[0]
    assert not changed.Bettable and changed.Play_Stake==0


def test_ambiguous_doubleheader_fails_closed_without_event_id():
    board=pd.DataFrame([final(matchup_id='')])
    rows=pd.DataFrame([candidate('game1'),candidate('game2')])
    assert per_game_board(board,rows,'totals').iloc[0]['pick']=='No Bet — market unavailable'


def test_selection_score_never_becomes_win_probability():
    result=per_game_board(pd.DataFrame([final()]),pd.DataFrame([candidate(calibrated_probability=None,ml_probability=.99,best_available_score=.99)]),'totals').iloc[0]
    assert result.win_probability is None
    assert result.probability_basis=='Unavailable'
    assert result.selection_score==.99


def test_missing_audit_keeps_every_game_in_each_view():
    board=pd.DataFrame([final(),final('g2')])
    assert len(per_game_board(board,None,'totals'))==2
    assert per_game_board(board,None,'totals')['pick'].eq('No Bet — market unavailable').all()
