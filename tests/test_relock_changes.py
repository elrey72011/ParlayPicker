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
    assert at.button(key='lock_picks_action').disabled
    next(c for c in at.checkbox if c.key.startswith('relock_ack_')).check().run()
    next(c for c in at.text_input if c.key.startswith('relock_type_')).set_value('RELOCK').run()
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
""".replace('CHANGE',repr(change.to_dict()))
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


def test_comparison_contract_is_immutable_and_prefers_structured_teams():
    from dataclasses import FrozenInstanceError
    a=row();b=row('Seattle +1.5','spread_away')
    # Deliberately contradictory display labels must not override saved orientation.
    a['legs'][0].update(home_team='Team H',away_team='Team A')
    b['legs'][0].update(home_team='Team H',away_team='Team A')
    c=compare(a,b)
    assert c.previous['Selected team']=='Team H'
    assert c.current['Selected team']=='Team A'
    assert c.change_code=='SELECTED_TEAM_REVERSED'
    assert c.state=='REMOVED_LOCK_TEAM_REVERSAL'
    with pytest.raises(FrozenInstanceError): c.severity='NORMAL'
    with pytest.raises(TypeError): c.previous['Pick']='changed'
    legacy=row();legacy['legs'][0].pop('game')
    assert compare(legacy,b).previous['Selected team']=='Boston'


def test_metadata_flags_unknown_family_and_outcomes_do_not_classify():
    a=row();b=row();a['legs'][0].update(quote_source='Novig',quote_time=AT,provider_namespace='a',provider_event_id='1')
    b['legs'][0].update(quote_source='DraftKings',quote_time='2026-09-09T19:57:00Z',provider_namespace='b',provider_event_id='2')
    c=compare(a,b)
    assert c.flags['sportsbook_changed'] and c.flags['quote_time_changed'] and c.flags['provider_identity_changed']
    assert c.change_code=='PRICE_CHANGED'
    b['legs'][0]['outcome']='LOSS';a['legs'][0]['outcome']='WIN'
    assert compare(a,b).severity==c.severity
    assert compare(a,b).change_code==c.change_code
    a['legs'][0]['market']=''
    assert not compare(a,b).flags['market_family_changed']
    a['legs'][0]['market']='legacy_side'
    assert compare(a,b).flags['market_family_changed']


def test_latest_removal_uses_chronology_and_exact_hash():
    a=row();a['published_at']=AT;b=deepcopy(a);b['legs'][0]['odds']=-120
    removals=[dict(lock=a,lock_id='game',lock_hash=history.digest(a),removed_at='2026-09-09T16:00:00-04:00',reason='one'),
              dict(lock=b,lock_id='game',lock_hash=history.digest(b),removed_at='2026-09-09T19:59:00Z',reason='two')]
    assert latest_removed(removals,'game')['reason']=='one'
    removals[0]['lock']['legs'][0]['odds']=-140
    with pytest.raises(ValueError): latest_removed(removals,'game')


def test_favorite_standard_spread_requires_opposing_pair():
    from app_core.relock_changes import saved_favorite
    r=row();l=r['legs'][0]
    l.update(provider_namespace='odds_api',provider_event_id='event',quote_source='Novig',quote_time=AT)
    l['provider_quotes']=[dict(provider_namespace='odds_api',provider_event_id='event',book='novig',recorded_at=AT,
                              market_type='spread_'+side,point=line) for side,line in [('home',-1.5),('away',1.5)]]
    assert saved_favorite(l)=='Boston'
    l['provider_quotes'].pop()
    assert saved_favorite(l) is None


@pytest.mark.parametrize('mutation',['odds','pick','market','quote_source','provider_event_id','candidate_id','start'])
def test_reviewed_candidate_mutations_cause_zero_writes(monkeypatch,mutation):
    from app_core.relock_changes import validate_candidates,RelockReviewExpired
    monkeypatch.setattr(history,'now',lambda:AT)
    package=pub()['package'];reviewed={r['id']:r for r in lock_candidates(package,AT)}
    values={'odds':-130,'pick':'Boston +2.5','market':'spread_away','quote_source':'Unavailable',
            'provider_event_id':'different','candidate_id':'different','start':'2026-09-09T19:00:00Z'}
    package['games']['overall'][0][mutation]=values[mutation]
    with pytest.raises((RelockReviewExpired,ValueError)):
        validate_candidates(package,list(reviewed),reviewed,AT)


def test_concurrent_active_lock_and_changed_removal_block_before_writes(monkeypatch):
    from app_core.relock_changes import RelockAlreadyLocked,RelockReviewExpired
    monkeypatch.setattr(history,'now',lambda:AT)
    store=history.History('site-1234','folder',Memory());package=pub()['package']
    candidate=lock_candidates(package,AT)[0];identity=candidate['id']
    original=store.lock_picks(package,[identity])[0]
    store.remove_locks([history.digest(original)],'first removal')
    prior=latest_removed(store.all('lock_removals'),identity);token=review_token(prior,candidate)
    monkeypatch.setattr(history,'now',lambda:'2026-09-09T19:56:30+00:00')
    second=store.lock_picks(package,[identity])[0]
    before=deepcopy(store.client.data)
    with pytest.raises(RelockAlreadyLocked): store.lock_picks(package,[identity],relock_review={identity:token})
    assert store.client.data==before
    # New generation differs, including acceptance timestamp, so it has its own tombstone.
    monkeypatch.setattr(history,'now',lambda:'2026-09-09T19:57:00+00:00')
    store.remove_locks([history.digest(second)],'second removal')
    before=deepcopy(store.client.data)
    with pytest.raises(RelockReviewExpired):store.lock_picks(package,[identity],relock_review={identity:token})
    assert store.client.data==before


def test_conditional_write_loser_is_not_reported_as_success(monkeypatch):
    from app_core.relock_changes import RelockAlreadyLocked
    monkeypatch.setattr(history,'now',lambda:AT)
    store=history.History('site-1234','folder',Memory());package=pub()['package'];candidate=lock_candidates(package,AT)[0]
    put=store.put
    def raced(key,value,**kwargs):
        if key.startswith('locks/'):
            winner=deepcopy(value);winner['legs'][0]['odds']=-120
            put(key,winner,first=True)
        return put(key,value,**kwargs)
    monkeypatch.setattr(store,'put',raced)
    with pytest.raises(RelockAlreadyLocked) as error:store.lock_picks(package,[candidate['id']],relock_review={})
    assert error.value.after_write
    assert len(store.all('locks'))==1
    assert store.all('locks')[0]['legs'][0]['odds']==-120


@pytest.mark.parametrize('later',['2026-09-09T20:00:00Z','2026-09-10T19:56:00Z'])
def test_review_expiring_during_history_read_blocks_before_write(monkeypatch,later):
    from app_core.relock_changes import RelockReviewExpired
    times=iter([AT,later])
    monkeypatch.setattr(history,'now',lambda:next(times))
    store=history.History('site-1234','folder',Memory());package=pub()['package'];candidate=lock_candidates(package,AT)[0]
    with pytest.raises(RelockReviewExpired):store.lock_picks(package,[candidate['id']],relock_review={})
    assert store.client.data=={}


@pytest.mark.parametrize('interruption',['none','stale','active_elsewhere'])
def test_price_only_one_click_and_final_authority(monkeypatch,interruption):
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results,lock_picks,sftp_publish
    clock=[AT]
    monkeypatch.setattr(history,'now',lambda:clock[0])
    monkeypatch.setattr(lock_picks,'now',lambda:clock[0])
    store=history.History('site-1234','folder',Memory());package=pub()['package']
    package['games']['overall'][0]['start']='2026-09-09T22:00:00Z'
    candidate=lock_candidates(package,AT)[0]
    original=store.lock_picks(package,[candidate['id']])[0]
    store.remove_locks([history.digest(original)],'Owner correction')
    package['games']['overall'][0]['odds']=-120
    monkeypatch.setattr(public_results,'history',lambda _:store)
    publications=[]
    monkeypatch.setattr(sftp_publish,'publish_action',lambda *args:publications.append(args) or 'Published')
    code="""import streamlit as st
from app.ui.lock_picks import render_lock_picks
if 'public_results_site-1234' not in st.session_state:
 st.session_state['public_results_site-1234']={'publications':[],'revisions':[],'imports':[],'locks':[],'rows':[]}
render_lock_picks(PACKAGE,lambda _: 'site-1234')
""".replace('PACKAGE',repr(package))
    app=AppTest.from_string(code).run()
    assert not app.exception
    assert app.button(key='lock_picks_action').label=='Re-lock at current price'
    assert not list(app.checkbox)
    if interruption=='stale':
        clock[0]='2026-09-09T20:40:00Z'
    elif interruption=='active_elsewhere':
        clock[0]='2026-09-09T19:57:00Z'
        store.lock_picks(package,[candidate['id']])
    app.button(key='lock_picks_action').click().run()
    assert not app.exception
    assert not any(b.key=='relock_confirm' for b in app.button)
    if interruption=='none':
        assert len(publications)==1
        assert store.all('locks')[0]['legs'][0]['odds']==-120
        assert any('Re-lock saved at the current price' in x.value for x in app.success)
    else:
        assert not publications
        if interruption=='active_elsewhere':
            assert any('already locked elsewhere' in x.value for x in app.error)
        else:
            assert not store.all('locks')
    assert store.all('lock_removals')[0]['lock']==original


def test_quote_expiry_during_storage_read_zero_writes(monkeypatch):
    from app_core.relock_changes import RelockReviewExpired
    package=pub()['package'];package['games']['overall'][0]['start']='2026-09-09T22:00:00Z'
    candidate=lock_candidates(package,AT)[0]
    times=iter([AT,'2026-09-09T20:40:00Z'])
    monkeypatch.setattr(history,'now',lambda:next(times))
    store=history.History('site-1234','folder',Memory())
    with pytest.raises(RelockReviewExpired):store.lock_picks(package,[candidate['id']],relock_review={})
    assert store.client.data=={}
