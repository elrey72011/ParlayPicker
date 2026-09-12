from datetime import datetime, timedelta, timezone
import pytest
from app_core.locked_picks import lock_candidates, locked_selections
from app_core.public_board import build_package, validate_package
from app_core.public_history import selections
from app_core.public_parlays import build_parlays
from app_core import public_prop_history
from app_core.per_game_boards import exact_book_quote
from test_public_prop_history import publication
from test_public_board import boards
from test_per_game_boards import quoted_candidate

NOW = datetime(2026, 9, 9, 19, 56, tzinfo=timezone.utc)


def aged_publication(seconds, minutes=30):
    p = publication()
    p['confirmed_at'] = NOW.isoformat()
    p['package']['stale_after_minutes'] = minutes
    p['package']['built_at'] = NOW.isoformat()
    for rows in [*p['package']['games'].values(), p['package']['props']]:
        for row in rows:
            row['as_of'] = (NOW-timedelta(seconds=seconds)).isoformat()
    return p


@pytest.mark.parametrize('seconds,accepted', [(16*60,True),(29*60,True),(1800,True),(1801,False),(-1,False)])
def test_lock_and_game_prop_history_share_30_minute_boundary(seconds, accepted):
    p = aged_publication(seconds)
    leg = p['package']['games']['overall'][0]
    leg.update(quote_source='Novig', quote_time=leg['as_of'])
    locks = lock_candidates(p['package'], NOW.isoformat())
    assert bool(locks) is accepted
    if locks:
        assert locked_selections(locks) == locks
    assert bool(selections([p])) is accepted
    assert bool(public_prop_history.selections([p])) is accepted


def test_old_publications_keep_15_minute_policy_and_started_still_blocked():
    old = aged_publication(20*60, minutes=15)
    assert selections([old]) == []
    assert public_prop_history.selections([old]) == []
    assert lock_candidates(old['package'], NOW.isoformat()) == []
    current = aged_publication(20*60)
    for rows in [*current['package']['games'].values(),current['package']['props']]:
        for row in rows:
            row['start'] = NOW.isoformat()
    assert selections([current]) == []
    assert public_prop_history.selections([current]) == []
    assert lock_candidates(current['package'], NOW.isoformat()) == []


@pytest.mark.parametrize('book',['novig','draftkings','fanduel'])
@pytest.mark.parametrize('seconds,accepted', [(1200,True),(1800,True),(1801,False),(-1,False)])
def test_api_exact_quote_age(book,seconds,accepted):
    at = datetime(2026,9,11,19,59,tzinfo=timezone.utc)+timedelta(seconds=seconds)
    row = quoted_candidate(book,export_run_id=at.isoformat())
    assert bool(exact_book_quote(row,book)) is accepted


def test_parlays_validate_under_original_package_window():
    p = aged_publication(20*60)['package']
    a = p['games']['overall'][0]
    a.update(quote_source='Novig',quote_time=a['as_of'])
    b = dict(a,game='Chicago Cubs at New York Yankees')
    p['games']['overall'].append(b)
    p.update(schema_version=2,parlays=build_parlays([a,b],NOW))
    assert len(p['parlays']) == 1
    validate_package(p)
    p['stale_after_minutes'] = 15
    p['parlays'] = []
    validate_package(p)  # Legacy 15-minute package must not gain new pairs.
    a['quote_time'] = (NOW-timedelta(seconds=1801)).isoformat()
    assert build_parlays([a,b],NOW) == []


def test_new_packages_use_30_minutes_and_reject_unbounded_policy():
    p = build_package(*boards())
    assert p['stale_after_minutes'] == 30
    validate_package(p)
    for minutes in (0,31,True,60):
        with pytest.raises(ValueError):
            validate_package(dict(p,stale_after_minutes=minutes))
