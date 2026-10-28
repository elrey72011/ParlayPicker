"""Real Streamlit widget tests with explicitly synthetic dependencies only."""
from datetime import datetime,timedelta,timezone
import json
import re

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app_core.nfl_workflow_store import Store
from nfl_workflow_fixtures import KEY,TOKEN


SYNTHETIC_ANCHOR=datetime(2026,10,27,23,57,27,tzinfo=timezone.utc)


def application(path, *, line=2.5, anchor=SYNTHETIC_ANCHOR, advancing=False):
    root=Path(__file__).resolve().parents[1]
    anchor=anchor.isoformat()
    return f'''
import sys
from pathlib import Path
from datetime import datetime,timedelta
sys.path.insert(0,{str(root)!r})
sys.path.insert(0,{str(root/'tests')!r})
import streamlit as st
from app.ui.nfl_private_workflow import render
from app.ui.nfl_workflow_state import revoke
from nfl_workflow_fixtures import SyntheticClock,SyntheticProvider,make_workflow,TOKEN
supplied=st.text_input('Owner token',type='password',key='publication_token')
def setting(name,default=''):
    return TOKEN if name=='PARLAYPICKER_PUBLISH_TOKEN' else default
if supplied!=TOKEN:
    revoke(st.session_state)
    st.info('Owner access required')
else:
    def factory(access):
        if 'SYNTHETIC_provider' not in st.session_state:
            start=datetime.fromisoformat({anchor!r})
            if Path({str(path)!r}).exists():start+=timedelta(minutes=5)
            clock=SyntheticClock(start,advancing={advancing!r})
            st.session_state['SYNTHETIC_provider']=SyntheticProvider(anchor=datetime.fromisoformat({anchor!r}),clock=clock)
            st.session_state['SYNTHETIC_provider'].line={line!r}
        service,_=make_workflow(Path({str(path)!r}),provider=st.session_state['SYNTHETIC_provider'],clock=lambda: st.session_state['SYNTHETIC_provider'].clock().isoformat())
        return service
    render(setting,service_factory=factory)
'''


def ui_diagnostics(at,stage):
    # Never retain widget values, raw errors, exception bodies or credentials.
    reasons=sorted({code for item in at.caption for code in re.findall(
        r"Reason: ((?:NFL_|PRIVATE_)[A-Z_]+)",str(item.value))})
    return dict(stage=stage,reason_codes=reasons,error_count=len(at.error),
        exception_count=len(at.exception),button_labels=[item.label for item in at.button])


def healthy(at,stage,*,expected_reason=None):
    evidence=ui_diagnostics(at,stage)
    unexpected=[code for code in evidence['reason_codes'] if code!=expected_reason]
    if at.exception or unexpected or (at.error and expected_reason is None):
        record=json.dumps(evidence,sort_keys=True)
        print('SYNTHETIC_UI_FAILURE '+record,flush=True)
        raise AssertionError(record)
    return evidence


def press(at,label,*,expected_reason=None):
    healthy(at,'before '+label)
    buttons=[button for button in at.button if button.label==label]
    if len(buttons)!=1:
        record=json.dumps(ui_diagnostics(at,'missing control '+label),sort_keys=True)
        print('SYNTHETIC_UI_FAILURE '+record,flush=True)
        raise AssertionError(record)
    buttons[0].click().run(timeout=20)
    healthy(at,'after '+label,expected_reason=expected_reason)
    assert not at.exception


def reviewed_reset(at,reset):
    next(item for item in at.date_input if item.label=='Verified allowance reset date (UTC)').set_value(reset.date())
    next(item for item in at.time_input if item.label=='Verified allowance reset time (UTC)').set_value(reset.timetz().replace(tzinfo=None))


def check(at,label):
    next(item for item in at.checkbox if item.label==label).check()


def exercise_widgets(tmp_path,monkeypatch,*,line):
    path=tmp_path/'SYNTHETIC-ui.sqlite3';code=application(path,line=line,advancing=True)
    app=tmp_path/'SYNTHETIC-control-app.py';app.write_text(code,encoding='utf-8')
    at=AppTest.from_file(app,default_timeout=20).run()
    assert not path.exists()
    at.text_input(key='publication_token').input(TOKEN).run()
    assert not path.exists()  # Correct token alone must not access any database.
    at.checkbox(key='nfl_workflow_enabled').check().run()
    press(at,'Prepare bounded scoring capture')
    next(x for x in at.checkbox if x.label.startswith('I reviewed the source permission')).check()
    next(x for x in at.checkbox if x.label.startswith('I reviewed private research use')).check()
    next(item for item in at.date_input if item.label=='Private source-use permission ends after this UTC date').set_value((SYNTHETIC_ANCHOR+timedelta(days=3)).date())
    press(at,'Authorize and capture NFL schedule/history')
    select=at.multiselect(key='nfl_workflow_selected_games')
    select.set_value(['SYNTHETIC-UNCATALOGUED-DAL-BUF','SYNTHETIC-UNCATALOGUED-PHI-KC']).run()
    press(at,'Prepare selected NFL games')
    next(x for x in at.checkbox if x.label.startswith('I verified current included entitlement')).check()
    next(x for x in at.checkbox if x.label.startswith('I authorize these bounded requests')).check()
    reviewed_reset(at,SYNTHETIC_ANCHOR+timedelta(days=2))
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
    viewed=datetime.fromisoformat(fresh.session_state['nfl_workflow_saved_viewed_at'])
    assert all(datetime.fromisoformat(row['inference_time'])<viewed for row in originals.values())
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


@pytest.mark.parametrize('anchor,reset,accepted',[
    ('2026-10-27T23:57:27+00:00','2026-10-28T00:00:00+00:00',False),
    ('2026-10-28T00:03:00+00:00','2026-10-29T00:00:00+00:00',True),
    ('2026-10-27T23:40:00+00:00','2026-10-28T00:00:00+00:00',False),
    ('2026-10-27T23:39:59+00:00','2026-10-28T00:00:00+00:00',True),
],ids=['before-midnight-expired','after-midnight-valid','exact-window-end-rejected','one-second-beyond-valid'])
def test_reviewed_reset_boundary_in_real_controls(tmp_path,anchor,reset,accepted):
    anchor=datetime.fromisoformat(anchor);reset=datetime.fromisoformat(reset)
    path=tmp_path/'SYNTHETIC-boundary.sqlite3';app=tmp_path/'SYNTHETIC-boundary-app.py'
    app.write_text(application(path,anchor=anchor),encoding='utf-8')
    at=AppTest.from_file(app,default_timeout=20).run()
    at.text_input(key='publication_token').input(TOKEN).run()
    at.checkbox(key='nfl_workflow_enabled').check().run()
    press(at,'Prepare bounded scoring capture')
    next(x for x in at.checkbox if x.label.startswith('I reviewed the source permission')).check()
    next(x for x in at.checkbox if x.label.startswith('I reviewed private research use')).check()
    next(x for x in at.date_input if x.label=='Private source-use permission ends after this UTC date').set_value((anchor+timedelta(days=3)).date())
    press(at,'Authorize and capture NFL schedule/history')
    at.multiselect(key='nfl_workflow_selected_games').set_value(['SYNTHETIC-UNCATALOGUED-DAL-BUF']).run()
    press(at,'Prepare selected NFL games')
    reviewed_reset(at,reset)
    next(x for x in at.checkbox if x.label.startswith('I verified current included entitlement')).check()
    next(x for x in at.checkbox if x.label.startswith('I authorize these bounded requests')).check()
    press(at,'Authorize and capture reviewed NFL lines',expected_reason=None if accepted else 'NFL_WORKFLOW_ACCOUNT_EXPIRED')
    reasons=ui_diagnostics(at,'boundary outcome')['reason_codes']
    assert ('NFL_WORKFLOW_ACCOUNT_EXPIRED' in reasons)==(not accepted)
    provider=at.session_state['SYNTHETIC_provider']
    assert sum(kind=='market' for kind,_ in provider.calls)==int(accepted)
    assert any(str(x.key).startswith('nfl_workflow_review_') for x in at.checkbox)==accepted
    assert not at.metric  # No numerical calculation is authorized by this check.


@pytest.mark.parametrize('failure',['exception','reason'])
def test_widget_failure_diagnostics_retain_safe_stage_not_private_body(capsys,failure):
    from types import SimpleNamespace
    private='SYNTHETIC-PRIVATE-password-host-url-exception'
    at=SimpleNamespace(caption=[SimpleNamespace(value='Reason: NFL_WORKFLOW_ACCOUNT_EXPIRED '+private)] if failure=='reason' else [],
        error=[SimpleNamespace(value=private)],exception=[SimpleNamespace(value=private)] if failure=='exception' else [],button=[])
    with pytest.raises(AssertionError) as raised:
        healthy(at,'SYNTHETIC failed quote stage')
    retained=capsys.readouterr().out+str(raised.value)
    assert 'SYNTHETIC failed quote stage' in retained
    assert private not in retained
    if failure=='reason':assert 'NFL_WORKFLOW_ACCOUNT_EXPIRED' in retained
