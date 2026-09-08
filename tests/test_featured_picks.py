import pandas as pd
from app_core.featured_picks import featured_picks


def card(**kw):
    return {'market_type':'spread_home','best_pick':'Home -1.5','production_win_probability':.62,
            'production_edge':.06,'production_expected_value':.1,'odds_american':-110,'Bettable':False,**kw}


def test_approved_precedes_higher_probability_research_and_families_stay_separate():
    board=pd.DataFrame([card(best_pick='Research',production_win_probability=.95),
                        card(best_pick='Approved',Bettable=True),
                        card(best_pick='Over 8.5',market_type='total_over',Bettable=True,production_win_probability=.7)])
    assert featured_picks(board).iloc[0].best_pick=='Over 8.5'
    assert featured_picks(board,'sides').iloc[0].best_pick=='Approved'
    assert featured_picks(board,'totals').iloc[0].best_pick=='Over 8.5'
    assert board.Bettable.tolist()==[False,True,True]


def test_missing_final_probability_never_uses_composite_score_and_started_excluded():
    board=pd.DataFrame([card(production_win_probability=None,WinProbability=.99,best_available_score=.99),
                        card(Bet_Decision='STARTED'),card(odds_american=0),card(market_type='batter_hits_over')])
    assert featured_picks(board).empty


def test_fixed_probability_then_edge_then_ev_ranking():
    board=pd.DataFrame([card(best_pick='A',production_expected_value=.9),
                        card(best_pick='B',production_edge=.08),
                        card(best_pick='C',production_win_probability=.63,production_edge=.01)])
    assert featured_picks(board).best_pick.tolist()==['C','B','A']


def _app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    t,d=st.tabs(['Today','Details'])
    render_daily_dashboard(t.empty(),d.empty(),pd.DataFrame([
        {'league':'MLB','Home':'Home','Away':'Away','best_pick':'Over 8.5','market_type':'total_over',
         'production_win_probability':.65,'production_edge':.1,'production_expected_value':.2,
         'odds_american':-110,'Play_Stake':0,'production_eligible':False},
    ]))


def test_featured_research_card_has_pass_label_and_no_approved_export():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_function(_app).run()
    assert not app.exception
    assert {'Overall Best Pick','Sides','Totals'} <= {t.label for t in app.tabs}
    assert any('PASS' in m.value for m in app.markdown)
    assert any(m.value=='65.0%' for m in app.metric)
    assert not app.get('download_button')
