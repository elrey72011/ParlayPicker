import pandas as pd
import pytest
from app_core.results_overview import summarize_results, approved_wager_mask


def test_scope_denominators_and_missing_prices():
    rows = pd.DataFrame([
        {'league':'MLB','Bettable':True,'Play_Stake':5,'Outcome':'WIN','odds_american':150},
        {'league':'MLB','Bettable':True,'Play_Stake':5,'Outcome':'LOSS','odds_american':-110},
        {'league':'MLB','Bettable':True,'Play_Stake':5,'Outcome':'PUSH','odds_american':-110},
        {'league':'MLB','Bettable':True,'Play_Stake':5,'Outcome':'PENDING','odds_american':-110},
        {'league':'MLB','Bettable':True,'Play_Stake':5,'Outcome':'WIN','odds_american':None},
        {'league':'MLB','Bettable':False,'Play_Stake':0,'Outcome':'WIN','odds_american':100},
    ])
    result=summarize_results(rows).set_index('Record')
    approved=result.loc['App-approved (paper)']
    assert approved['Selections']==5
    assert approved['Pending / unresolved']==1
    assert approved['Decisions (W+L)']==3
    assert approved['Win rate']==pytest.approx(2/3)
    assert approved['Paper profit (units)']==pytest.approx(.5)
    assert approved['Paper ROI']==pytest.approx(.5/3)
    assert approved['Unpriced settled rows']==1
    assert result.loc['Research / unapproved','Wins']==1


def test_stake_alone_and_false_approval_are_not_approved():
    rows=pd.DataFrame([
        {'Play_Stake':10,'Bettable':'false','production_eligible':True},
        {'Play_Stake':10,'Bettable':'true','production_eligible':False},
        {'Play_Stake':0,'Bettable':'true','production_eligible':True},
        {'Play_Stake':10,'Bettable':'true','production_eligible':True},
    ])
    assert approved_wager_mask(rows).tolist()==[False,False,False,True]
    assert not approved_wager_mask(pd.DataFrame([{'Play_Stake':10}])).any()


def test_pending_only_has_no_rate_or_return_and_invalid_odds_never_imputed():
    pending=summarize_results(pd.DataFrame([{'league':'NFL','Outcome':'N/A'}])).iloc[0]
    assert pending['Pending / unresolved']==1
    assert pd.isna(pending['Win rate'])
    assert pd.isna(pending['Paper ROI'])
    bad=summarize_results(pd.DataFrame([{'Outcome':o,'odds_american':p} for o,p in [('WIN',0),('LOSS',float('inf')),('WIN',1.9)]])).iloc[0]
    assert bad['Unpriced settled rows']==3
    assert pd.isna(bad['Paper profit (units)'])


def test_sports_are_not_pooled_and_voids_are_excluded():
    result=summarize_results(pd.DataFrame([{'league':'MLB','Outcome':'VOID'}, {'league':'NFL','Outcome':'WIN','odds_american':100}])).set_index('Sport')
    assert result.loc['MLB','Void / DNP']==1
    assert pd.isna(result.loc['MLB','Win rate'])
    assert result.loc['NFL','Win rate']==1


def _overview_app():
    import pandas as pd
    import streamlit as st
    from app.ui.results_overview import render_recap_overview, render_evidence_overview
    slot=st.empty()
    render_recap_overview(slot,None)
    render_recap_overview(slot,pd.DataFrame([{'league':'MLB','Outcome':'N/A','Play_Stake':0,'Bettable':False}]))
    render_evidence_overview()


def test_overview_reruns_without_fetching_or_creating_evidence(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import requests
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Results overview must not call providers')
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path/'absent'))
    app=AppTest.from_function(_overview_app).run()
    assert not app.exception
    assert any('Actual betting returns: unavailable' in i.value for i in app.info)
    assert not (tmp_path/'absent').exists()
    app.run()
    assert not app.exception
    assert not (tmp_path/'absent').exists()
    assert not calls


def test_unfunded_status_and_infinite_stake_veto_approval():
    rows=pd.DataFrame([{'Bettable':True,'Play_Stake':5,'Stake_Status':'unfunded'},
                       {'Bettable':True,'Play_Stake':float('inf'),'Stake_Status':'funded'}])
    assert not approved_wager_mask(rows).any()


def _recap_app():
    import pandas as pd
    import streamlit as st
    from unittest.mock import patch
    from app.ui.results_dashboard import render_results_dashboard
    st.session_state.setdefault('fixture_outcome', 'WIN')
    if st.button('Change fixture source'):
        st.session_state['fixture_outcome']='LOSS'
    frame=pd.DataFrame([{'league':'MLB','Home':'Home','Away':'Away','best_pick':'Over 8.5',
                         'Outcome':st.session_state['fixture_outcome'],'odds_american':-110,
                         'Bettable':False,'Play_Stake':0}])
    def grade(df):
        st.session_state['test_fetches']=st.session_state.get('test_fetches',0)+1
        return df
    with patch('app.ui.results_dashboard._render_prop_results_recap'), patch('app.ui.results_dashboard._render_candidate_results_recap'), patch('app.ui.threshold_dashboard.render_threshold_dashboard'), patch('app.ui.results_dashboard.grade_picks_with_live_results', side_effect=grade):
        render_results_dashboard(frame)


def test_full_recap_fetch_is_one_shot_and_new_source_replaces_cache(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR', str(tmp_path/'evidence'))
    app=AppTest.from_function(_recap_app).run()
    assert not app.exception
    app.button(key='perf_fetch_scores_button').click().run()
    assert not app.exception
    assert app.session_state['test_fetches']==1
    app.run()
    assert app.session_state['test_fetches']==1
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['perf_edited_picks'].iloc[0].Outcome=='LOSS'
    assert app.session_state['test_fetches']==1
