"""Date-only account review through real widgets; all evidence is SYNTHETIC."""
from datetime import timedelta

import pytest
from streamlit.testing.v1 import AppTest

from test_nfl_private_workflow_ui import application, press, check, healthy, SYNTHETIC_ANCHOR
from nfl_workflow_fixtures import TOKEN, KEY
from app_core.nfl_workflow_store import Store


def prepared(tmp_path):
    path=tmp_path/'SYNTHETIC-date-ui.sqlite3'
    app=tmp_path/'SYNTHETIC-date-ui.py'
    app.write_text(application(path),encoding='utf-8')
    at=AppTest.from_file(app,default_timeout=20).run()
    at.text_input(key='publication_token').input(TOKEN).run()
    at.checkbox(key='nfl_workflow_enabled').check().run()
    press(at,'Prepare bounded scoring capture')
    next(x for x in at.checkbox if x.label.startswith('I reviewed the source permission')).check()
    next(x for x in at.checkbox if x.label.startswith('I reviewed private research use')).check()
    next(x for x in at.date_input if x.label=='Private source-use permission ends after this UTC date').set_value((SYNTHETIC_ANCHOR+timedelta(days=3)).date())
    press(at,'Authorize and capture NFL schedule/history')
    at.multiselect(key='nfl_workflow_selected_games').set_value(['SYNTHETIC-UNCATALOGUED-DAL-BUF']).run()
    press(at,'Prepare selected NFL games')
    at.radio[-1].set_value('Date only').run()
    return at,path


def findings(at):
    check(at,'I verified current included entitlement and private use for these exact requests.')
    check(at,'I authorize these bounded requests only, with $0 additional spending, and am ready to review the lines promptly.')


@pytest.mark.parametrize('days,reason',[(5,None),(0,'NFL_WORKFLOW_ACCOUNT_EXPIRED'),(None,'NFL_WORKFLOW_ACCOUNT_RESET_INVALID')])
def test_date_only_real_controls_admit_or_stop_before_provider_claim(tmp_path,monkeypatch,days,reason):
    at,path=prepared(tmp_path)
    field=next(x for x in at.date_input if x.label=='Allowance reset date (as displayed)')
    assert field.value is None
    if days is not None:field.set_value((SYNTHETIC_ANCHOR+timedelta(days=days)).date()).run()
    assert next(x for x in at.radio if x.label=='Timezone shown for that reset date').value=='UNKNOWN'
    assert not at.time_input
    assert any('Actual reset time remains UNKNOWN' in c.value for c in at.caption)
    if days is not None:assert any('Conservative reset bound (not the actual reset)' in c.value for c in at.caption)
    findings(at)
    def denied(*a,**k):pytest.fail('Account capture must not predict')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',denied)
    press(at,'Authorize and capture reviewed NFL lines',expected_reason=reason)
    provider=at.session_state['SYNTHETIC_provider']
    assert sum(kind=='market' for kind,_ in provider.calls)==int(reason is None)
    assert not at.metric
    store=Store.synthetic_fixture(path,KEY)
    operations=[r for r in store.list('operation') if r['payload']['kind']=='market']
    assert len(operations)==1
    if reason is None:
        review=store.claim_record('operation:'+operations[0]['id'])['payload']['approval']
        assert review['reset_observation']['timezone']=='UNKNOWN' and review['reset_time']=='UNKNOWN'
        assert 'reset_at' not in review
    else:assert not store.spent('operation:'+operations[0]['id'])


def test_changed_date_or_timezone_requires_fresh_account_findings(tmp_path):
    at,path=prepared(tmp_path)
    next(x for x in at.date_input if x.label=='Allowance reset date (as displayed)').set_value((SYNTHETIC_ANCHOR+timedelta(days=5)).date()).run()
    findings(at)
    next(x for x in at.date_input if x.label=='Allowance reset date (as displayed)').set_value((SYNTHETIC_ANCHOR+timedelta(days=6)).date()).run()
    assert all(not x.value for x in at.checkbox if str(x.key).startswith(('nfl_workflow_private_','nfl_workflow_no_overage_')))
    findings(at)
    next(x for x in at.radio if x.label=='Timezone shown for that reset date').set_value('UTC').run()
    assert all(not x.value for x in at.checkbox if str(x.key).startswith(('nfl_workflow_private_','nfl_workflow_no_overage_')))
    assert not at.time_input and not at.metric
    healthy(at,'changed account facts')
    assert not any(kind=='market' for kind,_ in at.session_state['SYNTHETIC_provider'].calls)
