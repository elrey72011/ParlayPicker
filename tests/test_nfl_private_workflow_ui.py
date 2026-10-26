"""Real Streamlit widget tests with explicitly synthetic dependencies only."""
from datetime import datetime,timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app_core.nfl_workflow_store import Store
from nfl_workflow_fixtures import KEY,TOKEN


def application(path, *, line=2.5):
    root=Path(__file__).resolve().parents[1]
    anchor=datetime.now(timezone.utc).isoformat()
    return f'''
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0,{str(root)!r})
sys.path.insert(0,{str(root/'tests')!r})
import streamlit as st
from app.ui.nfl_private_workflow import render
from app.ui.nfl_workflow_state import revoke
from nfl_workflow_fixtures import SyntheticProvider,make_workflow,TOKEN
supplied=st.text_input('Owner token',type='password',key='publication_token')
def setting(name,default=''):
    return TOKEN if name=='PARLAYPICKER_PUBLISH_TOKEN' else default
if supplied!=TOKEN:
    revoke(st.session_state)
    st.info('Owner access required')
else:
    def factory(access):
        if 'SYNTHETIC_provider' not in st.session_state:
            st.session_state['SYNTHETIC_provider']=SyntheticProvider(anchor=datetime.fromisoformat({anchor!r}))
            st.session_state['SYNTHETIC_provider'].line={line!r}
        service,_=make_workflow(Path({str(path)!r}),provider=st.session_state['SYNTHETIC_provider'])
        return service
    render(setting,service_factory=factory)
'''


def press(at,label):
    next(button for button in at.button if button.label==label).click().run(timeout=20)
    assert not at.exception


def check(at,label):
    next(item for item in at.checkbox if item.label==label).check()


def exercise_widgets(tmp_path,monkeypatch,*,line):
    path=tmp_path/'SYNTHETIC-ui.sqlite3';code=application(path,line=line)
    app=tmp_path/'SYNTHETIC-control-app.py';app.write_text(code,encoding='utf-8')
    at=AppTest.from_file(app,default_timeout=20).run()
    assert not path.exists()
    at.text_input(key='publication_token').input(TOKEN).run()
    assert not path.exists()  # Correct token alone must not access any database.
    at.checkbox(key='nfl_workflow_enabled').check().run()
    press(at,'Prepare bounded scoring capture')
    next(x for x in at.checkbox if x.label.startswith('I reviewed the source permission')).check()
    next(x for x in at.checkbox if x.label.startswith('I reviewed private research use')).check()
    press(at,'Authorize and capture NFL schedule/history')
    select=at.multiselect(key='nfl_workflow_selected_games')
    select.set_value(['SYNTHETIC-UNCATALOGUED-DAL-BUF','SYNTHETIC-UNCATALOGUED-PHI-KC']).run()
    press(at,'Prepare selected NFL games')
    next(x for x in at.checkbox if x.label.startswith('I verified current included entitlement')).check()
    next(x for x in at.checkbox if x.label.startswith('I authorize these bounded requests')).check()
    press(at,'Authorize and capture reviewed NFL lines')
    # Changing the reviewed event set destroys every earlier finding/subject.
    for item in at.checkbox:
        if str(item.key).startswith('nfl_workflow_review_'):item.check()
    at.multiselect(key='nfl_workflow_selected_games').set_value(['SYNTHETIC-UNCATALOGUED-PHI-KC']).run()
    assert not any(str(x.key).startswith('nfl_workflow_review_') for x in at.checkbox)
    assert 'nfl_workflow_pair_ids' not in at.session_state
    at.multiselect(key='nfl_workflow_selected_games').set_value(['SYNTHETIC-UNCATALOGUED-DAL-BUF','SYNTHETIC-UNCATALOGUED-PHI-KC']).run()
    press(at,'Prepare selected NFL games')
    press(at,'Review captured lines without another request')
    assert all(not x.value for x in at.checkbox if str(x.key).startswith('nfl_workflow_review_'))
    for item in at.checkbox:
        if str(item.key).startswith('nfl_workflow_review_'):item.check()
    press(at,'Authorize one calculation per side and save privately')
    assert len(at.metric)==4
    if line==-1.5:
        assert sum('No conditional model preference: opposing native cover estimates are tied.' in item.value for item in at.markdown)==2
        assert not any('Conditional model preference:' in item.value for item in at.markdown)
    else:
        assert any('Conditional model preference:' in item.value for item in at.markdown)
    store=Store.synthetic_fixture(path,KEY)
    originals={r['id']:r['payload'] for r in store.list('saved-index')}
    assert len(originals)==4
    # A new Streamlit session has no prior widget/review state, but results persist.
    fresh=AppTest.from_file(app,default_timeout=20).run()
    fresh.text_input(key='publication_token').input(TOKEN).run()
    fresh.checkbox(key='nfl_workflow_enabled').check().run()
    fresh.radio(key='nfl_workflow_task').set_value('View saved forecasts').run()
    selection=fresh.multiselect(key='nfl_workflow_saved_selected')
    selection.set_value(list(originals)).run()
    def denied(*a,**k):raise AssertionError('Saved display must not acquire, predict or mutate')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',denied)
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities',denied)
    monkeypatch.setattr(Store,'put',denied);monkeypatch.setattr(Store,'claim',denied)
    press(fresh,'Load saved NFL forecasts')
    assert len(fresh.metric)==4
    fresh.number_input(key='nfl_workflow_saved_page').set_value(2).run()
    assert not fresh.exception and not fresh.metric
    fresh.number_input(key='nfl_workflow_saved_page').set_value(1).run()
    fresh.multiselect(key='nfl_workflow_saved_selected').set_value(list(originals)).run()
    press(fresh,'Load saved NFL forecasts')
    assert len(fresh.metric)==4
    assert {r['id']:r['payload'] for r in store.list('saved-index')}==originals
    fresh.text_input(key='publication_token').input('wrong').run()
    assert len(fresh.metric)==0 and 'nfl_workflow_saved_receipts' not in fresh.session_state


def test_two_games_through_widgets_calculate_reopen_and_view_without_actions(tmp_path,monkeypatch):
    exercise_widgets(tmp_path,monkeypatch,line=2.5)


def test_tied_native_estimates_have_no_preferred_side_in_real_controls(tmp_path,monkeypatch):
    exercise_widgets(tmp_path,monkeypatch,line=-1.5)


def test_inactive_wrong_token_and_selection_changes_invalidate_review(tmp_path,monkeypatch):
    path=tmp_path/'SYNTHETIC-locked.sqlite3';app=tmp_path/'SYNTHETIC-locked-app.py'
    app.write_text(application(path),encoding='utf-8');at=AppTest.from_file(app,default_timeout=20).run()
    def denied(*a,**k):raise AssertionError('No storage operation before activation')
    with monkeypatch.context() as m:
        m.setattr(Store,'verify',denied)
        m.setattr('app_core.nfl_scenario_zip.read_archives',denied)
        m.setattr('app_core.nfl_scenario_research.compute',denied)
        m.setattr('app_core.nfl_scoring_acquisition.HttpsTransport',denied)
        m.setattr('app_core.nfl_odds_acquisition.HttpsTransport',denied)
        at.text_input(key='publication_token').input('wrong').run()
        assert not at.exception and not path.exists()
        at.text_input(key='publication_token').input(TOKEN).run()
        assert not at.exception and not path.exists()
