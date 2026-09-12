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
    assert lock_candidates(package,AT)==[]
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


def test_lock_audit_reconciles_rows_and_preserves_original_locks():
    from app_core.locked_picks import lock_audit
    from collections import Counter
    package=pub()['package']
    original=package['games']['overall'][0]
    rows=[]
    def add(name, **changes):
        rows.append(dict(original, game=name+' at Boston', **changes))
    add('Ready')
    add('Started', start=AT)
    add('Tomorrow', start='2026-09-10T20:00:00Z')
    add('Missing', quote_source='Unavailable', quote_time=None, quote_reason='Provider snapshot has no exact market')
    add('Stale quote', quote_source='Novig', quote_time='2026-09-09T19:00:00Z')
    add('Stale analysis', as_of='2026-09-09T19:00:00Z')
    add('Invalid market', market='unknown')
    add('No start', start=None)
    add('Future quote', quote_source='Novig', quote_time='2026-09-09T19:59:00Z')
    add('Future analysis', as_of='2026-09-09T19:59:00Z')
    package['games']['overall']=rows
    before=deepcopy(package)
    audit=lock_audit(package,AT)
    counts=Counter(r['Lock status'] for r in audit)
    assert len(counts)==10 and sum(counts.values())==10
    candidates=lock_candidates(package,AT)
    assert len(candidates)==counts['Eligible now']==1
    assert candidates[0]['legs'][0]['game']==audit[0]['Game']
    assert audit[3]['Reason']=='Provider snapshot has no exact market'
    assert package==before
    rows[0]['as_of']='2026-09-09T19:00:00Z'
    assert lock_audit(package,AT,candidates)[0]['Lock status']=='Already locked'
    assert candidates[0]['legs'][0]['as_of']==original['as_of']


def test_lock_audit_duplicate_and_thirty_minute_boundary():
    from app_core.locked_picks import lock_audit
    package=pub()['package'];package['stale_after_minutes']=30
    row=package['games']['overall'][0]
    row.update(as_of='2026-09-09T19:26:00Z',quote_source='Novig',quote_time='2026-09-09T19:26:00Z')
    assert lock_audit(package,AT)[0]['Lock status']=='Eligible now'
    assert lock_audit(package,'2026-09-09T19:56:01Z')[0]['Lock status']=='Stale quote'
    package['games']['overall'].append(deepcopy(row))
    assert [r['Lock status'] for r in lock_audit(package,AT)]==['Duplicate game entry','Duplicate game entry']
    assert lock_candidates(package,AT)==[]  # Ambiguity excludes only this game.


def test_lock_ui_explains_missing_quotes_without_requesting_pointless_refresh(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import lock_picks
    monkeypatch.setattr(lock_picks,'now',lambda:AT)
    package=pub()['package']
    package['games']['overall'][0].update(quote_source='Unavailable',quote_time=None,quote_reason='No matching exact quote')
    code="""import streamlit as st
from app.ui.lock_picks import render_lock_picks
st.session_state['public_results_site']={'publications':[],'revisions':[],'locks':[]}
render_lock_picks(PACKAGE,lambda key:'site')
""".replace('PACKAGE',repr(package))
    app=AppTest.from_string(code).run()
    assert not app.exception
    assert any('Quote unavailable: 1' in r.value for r in app.caption)
    assert 'Click Refresh picks' not in ' '.join(r.value for r in app.info)
    assert app.button(key='lock_picks_action').disabled
    assert any(r.label=='Why games cannot be locked' for r in app.expander)


@pytest.mark.parametrize('second_start', [None, '2026-09-09T22:00:00Z'])
def test_duplicate_group_does_not_block_unrelated_lock(store, second_start):
    from app_core.locked_picks import lock_audit
    package = pub()['package']
    duplicate = deepcopy(package['games']['overall'][0])
    if second_start:
        duplicate['start'] = second_start
    other = dict(duplicate, game='Ready at Boston')
    package['games']['overall'].extend([duplicate, other])
    candidates = lock_candidates(package, AT)
    assert [r['legs'][0]['game'] for r in candidates] == ['Ready at Boston']
    audit = lock_audit(package, AT)
    assert [r['Lock status'] for r in audit] == ['Duplicate game entry'] * 2 + ['Eligible now']
    assert store.lock_picks(package, [candidates[0]['id']]) == candidates


def test_existing_alias_duplicates_do_not_hide_new_selector(store, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import lock_picks
    from app_core.locked_picks import lock_audit
    package = pub()['package']
    original = package['games']['overall'][0]
    original.update(sport='NCAAF', game='Grambling State at Tcu', pick='Tcu -20.5')
    saved = store.lock_picks(package, [r['id'] for r in lock_candidates(package, AT)])
    before = deepcopy(saved)
    package['games']['overall'].extend([dict(original, game='Grambling at Tcu'),
                                       dict(original, game='Ready at Tcu')])
    assert [r['Lock status'] for r in lock_audit(package, AT, saved)] == ['Already locked', 'Already locked', 'Eligible now']
    monkeypatch.setattr(lock_picks, 'now', lambda: AT)
    code = "import streamlit as st\nfrom app.ui.lock_picks import render_lock_picks\n"
    code += "st.session_state['public_results_site']=" + repr(dict(publications=[], revisions=[], locks=saved)) + "\n"
    code += "render_lock_picks(" + repr(package) + ", lambda key: 'site')"
    app = AppTest.from_string(code).run()
    assert not app.exception
    assert len(app.multiselect(key='lock_pick_selection').value) == 1
    assert not app.button(key='lock_picks_action').disabled
    assert store.all('locks') == before


@pytest.mark.parametrize('pick,market', [('Cal Poly line unresolved','spread_away'),
    ('Cal Poly','spread_away'), ('Over','total_over'), ('Under 48.5','total_over')])
def test_unresolved_new_pick_is_not_lockable(store, pick, market):
    from app_core.locked_picks import lock_audit
    package = pub()['package']
    package['games']['overall'][0].update(pick=pick, market=market)
    assert lock_candidates(package, AT) == []
    assert lock_audit(package, AT)[0]['Lock status'] == 'Unresolved pick'
    with pytest.raises(ValueError):
        store.lock_picks(package, ['invalid'])
    assert store.all('locks') == []


def test_new_pick_validation_does_not_rewrite_archived_locks():
    saved = lock_candidates(pub()['package'], AT)
    saved[0]['legs'][0]['pick'] = 'Boston line unresolved'
    before = deepcopy(saved)
    assert locked_selections(saved) == before
