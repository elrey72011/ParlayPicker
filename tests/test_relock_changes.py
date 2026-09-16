from copy import deepcopy
import pytest
from app_core.relock_changes import compare, acknowledged, review_token, latest_removed
from app_core.locked_picks import lock_candidates
from app_core import public_history as history
from test_public_history import pub, leg, Memory
from test_locked_picks import AT


def row(pick='Boston +1.5', market='spread_home', odds=-110):
    return dict(id='game', legs=[dict(leg(pick,market), odds=odds)])


@pytest.mark.parametrize('old,new,severity', [
    (row(),row(odds=-120),'NORMAL'),
    (row(),row('Boston +2.5'),'NORMAL'),
    (row('Over 8.5','total_over'),row('Under 8.5','total_under'),'WARNING'),
    (row(),row('Under 8.5','total_under'),'HIGH'),
    (row('Under 8.5','total_under'),row(),'HIGH'),
    (row(),row('Seattle +1.5','spread_away',-174),'CRITICAL')])
def test_severity_and_acknowledgment(old,new,severity):
    change=compare(old,new)
    assert change['severity']==severity
    assert acknowledged(change)==(severity=='NORMAL')
    assert acknowledged(change,True)==(severity!='CRITICAL')
    assert acknowledged(change,True,'RELOCK')
    if severity=='CRITICAL':
        assert change['flags']['price_changed']
        assert not acknowledged(change,True,'relock')
    assert change['flags']['market_favorite_changed'] is None


def test_storage_exact_review_and_immutable_history(monkeypatch):
    monkeypatch.setattr(history,'now',lambda:AT)
    store=history.History('site-1234','folder',Memory());package=pub()['package']
    original=store.lock_picks(package,[lock_candidates(package,AT)[0]['id']])[0]
    store.remove_locks([history.digest(original)],'Owner correction')
    removed=deepcopy(store.all('lock_removals'))
    package['games']['overall'][0].update(pick='Seattle +1.5',market='spread_away',odds=-174)
    candidate=lock_candidates(package,AT)[0];identity=candidate['id'];prior=latest_removed(removed,identity)
    token=review_token(prior,candidate)
    before=deepcopy(store.client.data)
    with pytest.raises(ValueError,match='review changed'):
        store.lock_picks(package,[identity],relock_review={})
    assert store.client.data==before
    changed=deepcopy(package);changed['games']['overall'][0]['odds']=-175
    with pytest.raises(ValueError,match='review changed'):
        store.lock_picks(changed,[identity],relock_review={identity:token})
    saved=store.lock_picks(package,[identity],relock_review={identity:token})
    assert saved[0]['legs']==candidate['legs']
    assert store.all('lock_removals')==removed
    assert removed[0]['lock']==original


def test_ui_critical_confirmation_and_cancel(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results,lock_picks,sftp_publish
    monkeypatch.setattr(history,'now',lambda:AT)
    monkeypatch.setattr(lock_picks,'now',lambda:AT)
    store=history.History('site-1234','folder',Memory());package=pub()['package']
    candidate=lock_candidates(package,AT)[0]
    original=store.lock_picks(package,[candidate['id']])[0]
    store.remove_locks([history.digest(original)],'Owner correction')
    package['games']['overall'][0].update(pick='Seattle +1.5',market='spread_away',odds=-174)
    monkeypatch.setattr(public_results,'history',lambda _:store)
    monkeypatch.setattr(sftp_publish,'publish_action',lambda *args:'Published')
    code="""import streamlit as st
from app.ui.lock_picks import render_lock_picks
if 'public_results_site-1234' not in st.session_state:
 st.session_state['public_results_site-1234']={'publications':[],'revisions':[],'imports':[],'locks':[],'rows':[]}
render_lock_picks(PACKAGE,lambda _: 'site-1234')
""".replace('PACKAGE',repr(package))
    at=AppTest.from_string(code).run()
    assert not at.exception
    assert at.button(key='lock_picks_action').disabled
    assert 'Boston +1.5' in str(at.dataframe[-1].value)
    next(c for c in at.checkbox if c.key.startswith('relock_ack_')).check().run()
    assert at.button(key='lock_picks_action').disabled
    next(c for c in at.text_input if c.key.startswith('relock_type_')).set_value('RELOCK').run()
    at.button(key='lock_picks_action').click().run()
    assert not store.all('locks')
    at.button(key='relock_cancel').click().run()
    assert not store.all('locks')
    at.button(key='lock_picks_action').click().run()
    at.button(key='relock_confirm').click().run()
    assert not at.exception
    assert store.all('locks')[0]['legs'][0]['pick']=='Seattle +1.5'
    assert store.all('lock_removals')[0]['lock']==original


def test_favorite_requires_exact_paired_saved_quotes():
    a=row(); b=row()
    for r,prices in [(a,(-150,130)),(b,(130,-150))]:
        l=r['legs'][0]
        l.update(provider_namespace='odds_api',provider_event_id='event',quote_source='Novig',quote_time=AT)
        l['provider_quotes']=[dict(provider_namespace='odds_api',provider_event_id='event',book='novig',recorded_at=AT,
                                  market_type='moneyline_'+side,price=price) for side,price in zip(('home','away'),prices)]
    assert compare(a,b)['flags']['market_favorite_changed'] is True
    b['legs'][0]['provider_quotes'][0]['provider_event_id']='other'
    assert compare(a,b)['flags']['market_favorite_changed'] is None


@pytest.mark.parametrize('new,expected', [(row(odds=-120),'NORMAL'), (row('Under 8.5','total_under'),'HIGH'),
                                         (row('Seattle +1.5','spread_away'),'CRITICAL')])
def test_review_always_shows_old_and_new_and_binds_ack(new,expected):
    from streamlit.testing.v1 import AppTest
    change=compare(row(),new)
    code="""import streamlit as st
from app.ui.lock_picks import render_relock_review
ready=render_relock_review(CHANGE,st.session_state.get('token','first'))
st.button('Continue',disabled=not ready)
""".replace('CHANGE',repr(change))
    app=AppTest.from_string(code).run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert 'PREVIOUS LOCK' in frame and 'CURRENT SELECTION' in frame
    assert app.button[0].disabled==(expected!='NORMAL')
    if expected!='NORMAL':
        app.checkbox[0].check().run()
        if expected=='CRITICAL':app.text_input[0].set_value('RELOCK').run()
        assert not app.button[0].disabled
        app.session_state['token']='changed-board'
        app.run()
        assert app.button[0].disabled
