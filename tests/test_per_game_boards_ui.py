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
    assert any('65.0%' in df.value.to_string() for df in app.dataframe)
    assert all(b.proto.label != 'Download approved game wagers' for b in app.get('download_button'))
