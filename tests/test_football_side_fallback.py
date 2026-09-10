import pandas as pd
import pytest
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows, build_best_picks_df
from app_core.per_game_boards import per_game_board


def quote_row(index=0):
    # Synthetic teams, lines and prices; no user exports are needed by the tests.
    row={'league':'NCAAF' if index==0 else 'MLB',
         'home_team':'Home Team', 'away_team':'Away Team',
         'game_date':'2099-09-10', 'game_time_est':'2099-09-10 8:00 PM ET',
         'commence_time_raw':'2099-09-11T00:00:00Z'}
    for book in ('fanduel','draftkings','betmgm'):
        row.update({f'{book}_over_point':45.5 if index==0 else 8.5,
                    f'{book}_under_point':45.5 if index==0 else 8.5,
                    f'{book}_over_price':-110, f'{book}_under_price':-110})
        if index==0:
            point=7.5 if book!='betmgm' else 6.5
            row.update({f'{book}_home_point':-point, f'{book}_away_point':point,
                        f'{book}_home_price':-110, f'{book}_away_price':-110})
    if index==1:
        # Two providers assign opposite signed run lines to the same teams.
        for book in ('novig','fanduel','draftkings','betmgm'):
            row.update({f'{book}_h2h_home_price':-120,
                        f'{book}_h2h_away_price':110})
        for book in ('novig','draftkings'):
            row.update({f'{book}_home_point':1.5, f'{book}_away_point':-1.5,
                        f'{book}_home_price':-200, f'{book}_away_price':175})
        row.update(fanduel_home_point=-1.5, fanduel_away_point=1.5,
                   fanduel_home_price=190, fanduel_away_price=-250)
    return row


def rank(row):
    expanded,_=_expand_live_odds_to_bet_rows(pd.DataFrame([row]),None)
    # Synthetic ranking inputs, not predictions for the source games.
    expanded['calibrated_probability']=expanded.market_type.map(
        {'spread_home':.48,'spread_away':.52,'total_over':.6,'total_under':.4})
    expanded['model_probability']=expanded.calibrated_probability
    expanded['expected_value']=-.05
    expanded['edge']=-.01
    expanded['market_probability']=.5
    diagnostics={}
    final=build_best_picks_df(expanded,diagnostics_out=diagnostics)
    return expanded,final,diagnostics['candidate_audit_df']


@pytest.mark.parametrize('league',['NCAAF','NFL'])
def test_football_without_novig_or_moneylines_uses_corroborated_spreads(league):
    row=quote_row(); row['league']=league
    expanded,final,audit=rank(row)
    sides=audit[audit.market_type.str.startswith('spread')].set_index('market_type')
    assert len(sides)==2
    assert sides.loc['spread_home','spread_line']==-7.5
    assert sides.loc['spread_home','odds_american']==-110
    assert sides.loc['spread_away','spread_line']==7.5
    assert sides.loc['spread_away','odds_american']==-110
    assert set(sides.opposing_odds_source)=={'fanduel'}
    assert set(sides.line_source)=={'fanduel_standard_spread_consensus'}
    final['Bettable']=False; final['Play_Stake']=0
    board=per_game_board(final,audit,'sides').iloc[0]
    assert board['pick'] in {'Home Team -7.5','Away Team +7.5'}
    assert board.Play_Stake==0 and board.status=='PASS'


@pytest.mark.parametrize('fault',['single_book','wrong_sign','missing_price','extreme_price'])
def test_missing_novig_does_not_bypass_pair_validation(fault):
    row=quote_row()
    if fault=='single_book':
        row={k:v for k,v in row.items() if not k.startswith(('draftkings_','betmgm_'))}
    elif fault=='wrong_sign': row['draftkings_away_point']=-7.5
    elif fault=='missing_price': row['draftkings_away_price']=None
    else: row['draftkings_away_price']=-100000
    _,_,audit=rank(row)
    assert not audit.market_type.str.startswith('spread').any()


def test_conflicting_mlb_signed_quotes_still_cannot_be_ranked():
    _,_,audit=rank(quote_row(1))
    assert not audit.market_type.str.startswith('spread').any()
