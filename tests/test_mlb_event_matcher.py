from copy import deepcopy
import pytest
from app_core.mlb_event_matcher import match_mlb_event, provider_ids, scheduled_eastern_date
from app_core.public_history import grade_leg


def saved(**kw):
    return dict(sport='MLB', game='Kansas City at Boston', start='2026-09-13T17:35:00Z', pick='Kansas City +1.5', market='spread_away', **kw)


def event(**kw):
    return {**dict(sport='MLB', away='Kansas City Royals', home='Boston Red Sox', start='2026-09-13T21:20:00Z', away_score=1, home_score=4, final=True, provider_ids={'mlb':'1'}), **kw}


@pytest.mark.parametrize('start', ['2026-09-13T18:20:00Z','2026-09-13T19:35:00Z','2026-09-13T21:35:00Z','2026-09-13T21:20:00Z'])
def test_unique_delays(start):
    leg=saved(); before=deepcopy(leg)
    assert match_mlb_event(leg,[event(start=start)]).status=='MATCHED_UNIQUE_EVENT'
    assert grade_leg(leg,[event(start=start)])[0]=='LOSS'
    assert leg==before


def test_doubleheader_never_nearest():
    events=[event(start='2026-09-13T17:35:00Z'),event(provider_ids={'mlb':'2'})]
    for order in (events, events[::-1]):
        assert match_mlb_event(saved(),order).status=='DOUBLEHEADER_AMBIGUOUS'


@pytest.mark.parametrize('field',['game_number','gameNumber','doubleheader_game','provider_game_number'])
def test_game_number(field):
    result=match_mlb_event(saved(**{field:'2'}),[event(game_number=1),event(provider_ids={'mlb':'2'},game_number=2)])
    assert result.status=='MATCHED_GAME_NUMBER'
    assert result.event['game_number']==2


def test_provider_identity_and_conflict():
    assert match_mlb_event(saved(mlb_game_pk='2'),[event(),event(provider_ids={'mlb':'2'})]).status=='MATCHED_PROVIDER_ID'
    assert match_mlb_event(saved(mlb_game_pk='1'),[event(away='Seattle Mariners')]).status=='PROVIDER_ID_CONFLICT'
    assert match_mlb_event(saved(odds_event_id='1'),[event()]).status=='MATCHED_UNIQUE_EVENT'
    assert match_mlb_event(saved(odds_event_id='1'),[event(away='Seattle Mariners')]).status=='PROVIDER_ID_NOT_FOUND'
    assert provider_ids({'game_id':'1'})=={}
    assert provider_ids({'game_id':'1','game_id_provider':'ESPN'})=={'espn':'1'}


def test_reschedule_not_delay():
    score=event(official_date='2026-09-14')
    assert match_mlb_event(saved(),[score]).status=='DATE_MISMATCH'
    result=match_mlb_event(saved(mlb_game_pk='1'),[score])
    assert result.status=='RESCHEDULED_NEEDS_REVIEW'
    assert result.settlement_review_required
    assert grade_leg(saved(mlb_game_pk='1'),[score])[0]=='PENDING'


def test_official_date_and_timezone():
    assert str(scheduled_eastern_date({'start':'2026-09-14T02:00:00Z'}))=='2026-09-13'
    assert scheduled_eastern_date({'start':'2026-09-13T12:00:00'}) is None
    assert scheduled_eastern_date({'provider_recorded_at':'2026-09-13T12:00:00Z'}) is None
    assert match_mlb_event(saved(),[event(official_date='2026-09-13', start='2026-09-14T10:00:00Z')]).status=='MATCHED_UNIQUE_EVENT'


def test_no_match_and_invalid():
    assert match_mlb_event(saved(),[]).status=='NO_MATCH'
    assert match_mlb_event({'sport':'MLB'},[]).status=='INVALID_SAVED_EVENT'
    assert match_mlb_event(saved(),[event(home_score=None)]).status=='INVALID_RESULT_EVENT'


def test_conflicting_scoped_ids_and_unscoped_duplicates():
    assert match_mlb_event(saved(provider_ids={'mlb':'1','espn':'2'}),[event(provider_ids={'mlb':'1','espn':'3'})]).status=='PROVIDER_ID_CONFLICT'
    unknown=event(provider_ids={})
    assert match_mlb_event(saved(),[unknown,dict(unknown)]).status=='DOUBLEHEADER_AMBIGUOUS'
