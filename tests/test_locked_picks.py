from copy import deepcopy
from datetime import datetime, timezone
import pytest
from test_public_history import Memory, pub, scores
from app_core import public_history as history
from app_core.locked_picks import lock_candidates, locked_selections

AT='2026-09-09T19:56:00+00:00'

@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(history,'now',lambda:AT)
    return history.History('site-1234','folder',Memory())

def test_original_pick_survives_rerun_and_separate_publication(store):
    package=pub()['package'];ids=[r['id'] for r in lock_candidates(package,AT)]
    original=store.lock_picks(package,ids)
    newer=deepcopy(package);newer['games']['overall'][0].update(pick='Boston -1.5',odds=120)
    assert store.lock_picks(newer,ids)==original
    rows=history.report([dict(pub(),package=newer)], [{'recorded_at':AT,'scores':scores()}],locks=store.all('locks'))
    locked=[r for r in rows if r['group']=='Locked']
    assert len(locked)==1 and locked[0]['outcome']=='WIN'
    assert 'Boston +1.5' in locked[0]['picks'] and locked[0]['odds']=='-110'
    assert locked[0]['published_at']==AT
    assert any(r['group']=='Approved' and r['category']=='overall' and r['outcome']=='LOSS' for r in rows)

@pytest.mark.parametrize('at',['2026-09-09T20:00:00+00:00','2026-09-10T19:56:00+00:00','2026-09-09T19:54:00+00:00'])
def test_rejects_started_other_day_future_quote(store,monkeypatch,at):
    package=pub()['package'];ids=[r['id'] for r in lock_candidates(package,AT)]
    monkeypatch.setattr(history,'now',lambda:at)
    with pytest.raises(ValueError):store.lock_picks(package,ids)
    assert store.all('locks')==[]

def test_stale_empty_unknown_and_duplicate_games_fail_closed(store):
    package=pub()['package']
    for ids in ([],['invalid']):
        with pytest.raises(ValueError):store.lock_picks(package,ids)
    package['games']['overall'][0]['as_of']='2026-09-09T19:30:00+00:00'
    assert lock_candidates(package,AT)==[]
    package=pub()['package'];package['games']['overall'].append(deepcopy(package['games']['overall'][0]))
    with pytest.raises(ValueError):lock_candidates(package,AT)
    assert store.all('locks')==[]

def test_restore_rejects_date_tampering(store):
    rows=lock_candidates(pub()['package'],AT);rows[0]['date']='2026-09-08'
    with pytest.raises(ValueError):locked_selections(rows)

def test_scheduler_grades_unpublished_locks(store,monkeypatch):
    from app_core import public_grading_scheduler as scheduler
    monkeypatch.setattr(scheduler,'is_open',lambda:True)
    package=pub()['package'];store.lock_picks(package,[r['id'] for r in lock_candidates(package,AT)])
    result=scheduler.run('site-1234','folder',store.client,{'MLB'},clock=lambda:datetime(2026,9,10,tzinfo=timezone.utc),fetch=lambda *a:{'recorded_at':AT,'scores':scores()})
    assert result['newly_settled']==1 and result['pending']==0

def test_explicit_button_only_and_rerun_idempotence(store,monkeypatch):
    monkeypatch.setattr("app_core.public_record.START_DATE","2026-09-09")
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results, lock_picks, sftp_publish
    published=[]
    monkeypatch.setattr(sftp_publish,"publish_action",lambda package,setting:published.append(package) or "Published")
    monkeypatch.setattr(public_results,'history',lambda setting:store)
    monkeypatch.setattr(lock_picks,'now',lambda:AT)
    code="""import streamlit as st
from app.ui.lock_picks import render_lock_picks
if 'public_results_site-1234' not in st.session_state:
    st.session_state['public_results_site-1234']={'publications':[],'revisions':[],'imports':[],'locks':[],'rows':[]}
render_lock_picks(PACKAGE,lambda key:'site-1234')
""".replace('PACKAGE',repr(pub()['package']))
    app=AppTest.from_string(code).run()
    assert not app.exception and store.all('locks')==[]
    app.button(key='lock_picks_action').click().run()
    assert not app.exception and len(store.all('locks'))==1
    app.run()
    assert not app.exception and len(store.all('locks'))==1
    assert len(published)==1
    assert published[0]['results'][0]['group']=='Locked'

def test_public_schema_scope(store):
    from app_core.public_board import validate_package
    from app_core.public_parlays import build_parlays
    package=pub()['package'];locks=lock_candidates(package,AT)
    package.update(schema_version=5,results=history.report([],[],locks=locks),parlays=build_parlays(package['games']['overall'],datetime.fromisoformat(package['built_at'])))
    validate_package(package)
    package['results'][0]['category']='sides'
    with pytest.raises(ValueError):validate_package(package)


def test_removal_is_archived_idempotent_and_excluded_after_restore(store):
    package=pub()['package'];ids=[r['id'] for r in lock_candidates(package,AT)]
    original=store.lock_picks(package,ids)[0]
    key=history.digest(original)
    assert store.remove_locks([key],'Owner correction')==[]
    assert store.remove_locks([key],'Retry')==[]
    restored=history.History('site-1234','folder',store.client)
    assert restored.all('locks')==[]
    assert restored._all('locks')==[original]
    assert restored.all('lock_removals')[0]['lock']==original
    assert not history.report([],[],locks=restored.all('locks'))


def test_relock_uses_new_quote_and_old_removal_does_not_remove_it(store,monkeypatch):
    package=pub()['package'];ids=[r['id'] for r in lock_candidates(package,AT)]
    original=store.lock_picks(package,ids)[0]
    store.remove_locks([history.digest(original)],'Wrong selection')
    newer=deepcopy(package);newer['games']['overall'][0].update(pick='Boston -1.5',odds=120)
    monkeypatch.setattr(history,'now',lambda:'2026-09-09T19:57:00+00:00')
    replacement=store.lock_picks(newer,ids)[0]
    assert replacement['legs'][0]['pick']=='Boston -1.5'
    store.remove_locks([history.digest(original)],'Repeated request')
    assert store.all('locks')==[replacement]
    assert len(store._all('locks'))==2
    assert store.lock_picks(package,ids)==[replacement]


def test_correction_preserves_other_lock_exactly(store):
    package=pub()['package'];first=store.lock_picks(package,[r['id'] for r in lock_candidates(package,AT)])[0]
    other=deepcopy(package)
    other['games']['overall'][0]['game']='Pittsburgh at Chicago Cubs'
    second=store.lock_picks(other,[r['id'] for r in lock_candidates(other,AT)])[0]
    assert store.remove_locks([history.digest(first)],'Keep PIT/CHC')==[second]
    with pytest.raises(ValueError):store.remove_locks(['unknown'],'Bad request')


def test_public_lock_league_comes_from_original_lock_not_current_board(store):
    from app_core.public_board import validate_package
    package=pub()['package']
    locks=lock_candidates(package,AT)
    rows=history.report([],[],locks=locks)
    assert rows[0]['sport']==locks[0]['legs'][0]['sport']=='MLB'
    package.update(schema_version=5,parlays=[],results=rows)
    validate_package(package)
    legacy=deepcopy(package)
    del legacy['results'][0]['sport']
    validate_package(legacy)
    package['results'][0]['sport']=123
    with pytest.raises(ValueError):validate_package(package)
