import pandas as pd
from streamlit.testing.v1 import AppTest


def app():
    import pandas as pd
    import streamlit as st
    from app.ui.publish_panel import render_publish_panel
    frame = pd.DataFrame([{'matchup_id':'g','export_run_id':'20260908T210000Z','league':'MLB','Home':'B','Away':'A',
        'game_date':'2026-09-08','best_pick':st.session_state.get('test_pick','Over 8.5'),'market_type':'total_over',
        'odds_american':-110,'Bettable':False,'Play_Stake':0,'production_win_probability':.55,
        'game_time_est':'2026-09-08 7:00 PM ET'}])
    import json
    kind='total_under' if st.session_state.get('test_pick','').startswith('Under') else 'total_over'
    frame['market_type']=kind
    frame['total_line']=8.5
    frame['provider_quotes']=json.dumps([{'book':'novig','market_type':kind,'point':8.5,'price':-110,'recorded_at':'2026-09-08T20:59:00Z'}])
    render_publish_panel(frame, None)


def test_locked_without_token(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','')
    at = AppTest.from_function(app).run()
    assert not at.exception
    assert not at.button
    assert 'Publishing is locked' in at.info[0].value


def test_preview_explicit_publish_and_invalidation(tmp_path, monkeypatch):
    token='test-only-publish-token'
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN',token)
    monkeypatch.setenv('PARLAYPICKER_PUBLICATION_DIR',str(tmp_path/'site'))
    at=AppTest.from_function(app).run()
    assert not at.button
    at.text_input[0].set_value(token).run()
    at.button(key='publication_build').click().run()
    assert not at.exception
    assert not (tmp_path/'site/index.html').exists()
    assert at.session_state['publication_preview']['package']['props']==[]
    at.button(key='publication_publish').click().run()
    assert not at.exception
    assert (tmp_path/'site/index.html').exists()
    original=(tmp_path/'site/index.html').read_text(encoding='utf-8')
    at.session_state['test_pick']='Under 8.5'
    at.run()
    assert not at.exception
    assert any(b.key=='publication_publish' for b in at.button)
    assert at.session_state['publication_preview']['package']['games']['overall'][0]['pick']=='Under 8.5'
    assert (tmp_path/'site/index.html').read_text(encoding='utf-8')==original
    assert any('Preview updated' in i.value for i in at.info)


def test_wrong_token_cannot_publish(monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','test-only-publish-token')
    at=AppTest.from_function(app).run()
    at.text_input[0].set_value('wrong').run()
    assert not at.button


def test_locks_render_before_embedded_preview(monkeypatch):
    from app.ui import public_results, lock_picks
    import streamlit.components.v1 as components
    calls=[]
    monkeypatch.setattr(public_results,'render_history',lambda setting: [])
    monkeypatch.setattr(lock_picks,'render_lock_picks',lambda *args: calls.append('locks'))
    monkeypatch.setattr(components,'html',lambda *args,**kwargs: calls.append('preview'))
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','test-only-publish-token')
    at=AppTest.from_function(app).run()
    at.text_input(key='publication_token').set_value('test-only-publish-token').run()
    assert not at.exception
    assert calls==['locks','preview']
