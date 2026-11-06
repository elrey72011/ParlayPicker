"""SYNTHETIC orchestrator behavior (roadmap M1.4). No network, no real games."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app_core.pipeline import orchestrator as orch
from app_core.pipeline.operations import COMPLETED, FAILED, Operations
from app_core.pipeline.orchestrator import Request, run_batch
from app_core.pipeline.store import Store

KEY = 'SYNTHETIC-orchestrator-test-key-not-production'
NOW = datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)
KICKOFF = '2026-11-08T18:00:00+00:00'
HISTORY = [dict(game_date=d, home=h, away=a, home_score=hs, away_score=as_) for d, h, a, hs, as_ in [
    ('2026-09-07', 'KANSAS CITY CHIEFS', 'BUFFALO BILLS', 27, 20),
    ('2026-09-14', 'BUFFALO BILLS', 'DENVER BRONCOS', 24, 17),
    ('2026-09-21', 'DENVER BRONCOS', 'KANSAS CITY CHIEFS', 10, 31),
    ('2026-10-05', 'BUFFALO BILLS', 'KANSAS CITY CHIEFS', 30, 27),
]]
KC_BUF = dict(event_id='SYNTHETIC-1', home='KANSAS CITY CHIEFS', away='BUFFALO BILLS', kickoff_utc=KICKOFF)
DEN_BUF = dict(event_id='SYNTHETIC-2', home='DENVER BRONCOS', away='BUFFALO BILLS', kickoff_utc=KICKOFF)


@pytest.fixture
def make_ops(tmp_path):
    def make(clock=lambda: NOW):
        return Operations(Store.sqlite(tmp_path / 'SYNTHETIC-orch.sqlite', KEY.encode(), owner='SYNTHETIC owner',
                                       clock=lambda: clock().isoformat()), clock=clock)
    return make


class History:
    def __init__(self):
        self.calls = 0

    def __call__(self, sport, request):
        self.calls += 1
        return HISTORY


def test_mixed_batch_gives_every_request_a_status_and_reason(make_ops):
    history = History()
    requests = [
        Request('NFL', KC_BUF, 'spread_home', -2.5, quote_observed_at=(NOW - timedelta(minutes=5)).isoformat()),
        Request('NFL', DEN_BUF, 'total_under', 44.5),                       # quote time unknown: still runs
        Request('NFL', KC_BUF, 'spread_home', -3.0),                        # integer line
        Request('NFL', KC_BUF, 'total_over', 45.5, quote_observed_at=(NOW - timedelta(hours=2)).isoformat()),
        Request('NFL', dict(KC_BUF, kickoff_utc='2026-11-08T12:00:00+00:00'), 'spread_away', 2.5),
        Request('NBA', dict(KC_BUF, event_id='SYNTHETIC-NBA'), 'spread_home', -4.5),
        Request('NFL', dict(KC_BUF, event_id='SYNTHETIC-3', away='NEW YORK JETS'), 'spread_home', -6.5),
    ]
    rows = run_batch(requests, ops=make_ops(), history_for=history, as_of_utc=NOW.isoformat())
    assert [(r['status'], r['reason']) for r in rows] == [
        ('AVAILABLE', 'AVAILABLE'), ('AVAILABLE', 'AVAILABLE'), ('UNAVAILABLE', 'NFL_INTEGER_LINE_UNSUPPORTED'),
        ('UNAVAILABLE', 'QUOTE_STALE'), ('UNAVAILABLE', 'GAME_STARTED'), ('UNAVAILABLE', 'ADAPTER_SPORT_NOT_SUPPORTED'),
        ('UNAVAILABLE', 'NFL_NO_COMPLETED_POINT_IN_TIME_GAMES')]
    assert rows[0]['quote_age_seconds'] == 300 and rows[1]['quote_age_seconds'] == 'UNKNOWN'
    assert 0.2 <= rows[0]['probability'] <= 0.8 and rows[0]['model_id'] == 'score-distribution-v1:nfl'
    assert rows[6]['state'] is None  # failed inside its operation: reason recorded, no probability invented


def test_rerender_and_restart_reuse_stored_forecasts_without_recomputing(make_ops):
    history = History()
    requests = [Request('NFL', KC_BUF, 'spread_home', -2.5), Request('NFL', DEN_BUF, 'total_under', 44.5)]
    first = run_batch(requests, ops=make_ops(), history_for=history, as_of_utc=NOW.isoformat())
    assert history.calls == 2
    later = NOW + timedelta(minutes=40)
    again = run_batch(requests, ops=make_ops(lambda: later), history_for=history, as_of_utc=later.isoformat())
    assert history.calls == 2  # nothing recomputed
    assert [r['probability'] for r in again] == [r['probability'] for r in first]
    assert [r['operation_id'] for r in again] == [r['operation_id'] for r in first]


def test_a_failed_calculation_is_reported_and_not_retried_silently(make_ops):
    def broken(sport, request):
        raise ValueError('NFL_HISTORY_SCHEMA')
    rows = run_batch([Request('NFL', KC_BUF, 'spread_home', -2.5)], ops=make_ops(), history_for=broken,
                     as_of_utc=NOW.isoformat())
    assert (rows[0]['status'], rows[0]['reason']) == ('UNAVAILABLE', 'NFL_HISTORY_SCHEMA')
    history = History()
    again = run_batch([Request('NFL', KC_BUF, 'spread_home', -2.5)], ops=make_ops(), history_for=history,
                      as_of_utc=NOW.isoformat())
    assert (again[0]['reason'], again[0]['state'], history.calls) == ('NFL_HISTORY_SCHEMA', FAILED, 0)


def test_nfl_history_downloads_once_and_failed_attempts_get_recorded_successors(make_ops):
    calls = []

    def flaky(season):
        calls.append(season)
        if len(calls) == 1:
            raise ConnectionError('SYNTHETIC outage')
        return HISTORY
    ops = make_ops()
    assert orch.nfl_history(ops, NOW.isoformat(), fetch=flaky) == HISTORY
    assert calls == [2026, 2026]
    assert orch.nfl_history(make_ops(), NOW.isoformat(), fetch=flaky) == HISTORY  # restart: no new download
    assert len(calls) == 2

    def down(season):
        calls.append(season)
        raise ConnectionError('SYNTHETIC outage')
    tomorrow = NOW + timedelta(days=1)
    with pytest.raises(orch.AdapterError, match='NFL_HISTORY_UNAVAILABLE'):
        orch.nfl_history(make_ops(lambda: tomorrow), tomorrow.isoformat(), fetch=down)
    assert len(calls) == 2 + orch.HISTORY_ATTEMPTS_PER_DAY  # bounded, never unlimited retries


def test_requests_from_ordinary_refresh_rows():
    frame = pd.DataFrame([
        dict(league='NFL', market_type='spread_home', spread_line=-2.5, home_team='Kansas City Chiefs',
             away_team='Buffalo Bills', provider_event_id='SYNTHETIC-E', game_start_utc=KICKOFF, odds_american=-110),
        dict(league='NFL', market_type='total_over', total_line=44.5, home_team='Kansas City Chiefs',
             away_team='Buffalo Bills', provider_event_id='SYNTHETIC-E', game_start_utc=KICKOFF),
        dict(league='NFL', market_type='h2h_home', home_team='Kansas City Chiefs', away_team='Buffalo Bills'),
        dict(league='NBA', market_type='spread_home', spread_line=-4.5, home_team='A', away_team='B'),
    ])
    requests = orch.requests_from_analysis(frame)
    assert [(r.market_type, r.line) for r in requests] == [('spread_home', -2.5), ('total_over', 44.5)]
    assert requests[0].event == dict(event_id='SYNTHETIC-E', home='KANSAS CITY CHIEFS', away='BUFFALO BILLS',
                                     kickoff_utc=KICKOFF)
    assert requests[0].extra['odds_american'] == -110


def test_shadow_is_off_by_default_and_reports_missing_configuration():
    frame = pd.DataFrame([dict(league='NFL', market_type='spread_home', spread_line=-2.5)])
    assert orch.shadow(frame, setting=lambda k, d='': d) == ([], 'PIPELINE_V2_DISABLED')
    on = {'PARLAYPICKER_PIPELINE_V2': 'true'}
    assert orch.shadow(frame, setting=lambda k, d='': on.get(k, d)) == ([], 'PIPELINE_STORE_NOT_CONFIGURED')
    assert orch.shadow(pd.DataFrame(), setting=lambda k, d='': on.get(k, d)) == ([], 'NO_NFL_SPREAD_OR_TOTAL_ROWS')


def test_shadow_end_to_end_with_a_local_store(tmp_path):
    settings = {'PARLAYPICKER_PIPELINE_V2': 'true', 'PARLAYPICKER_PIPELINE_STORE_KEY': KEY,
                'PARLAYPICKER_PIPELINE_SQLITE_PATH': str(tmp_path / 'SYNTHETIC-shadow.sqlite')}
    frame = pd.DataFrame([dict(league='NFL', market_type=m, spread_line=-2.5 if m == 'spread_home' else None,
                               total_line=44.5 if m == 'total_over' else None, home_team='Kansas City Chiefs',
                               away_team='Buffalo Bills', provider_event_id='SYNTHETIC-E', game_start_utc=KICKOFF)
                          for m in ('spread_home', 'total_over')])
    downloads = []
    rows, note = orch.shadow(frame, setting=lambda k, d='': settings.get(k, d), clock=lambda: NOW,
                             fetch=lambda season: downloads.append(season) or HISTORY)
    assert note is None and [r['status'] for r in rows] == ['AVAILABLE', 'AVAILABLE'] and downloads == [2026]
    again, _ = orch.shadow(frame, setting=lambda k, d='': settings.get(k, d), clock=lambda: NOW,
                           fetch=lambda season: downloads.append(season) or HISTORY)
    assert downloads == [2026] and [r['probability'] for r in again] == [r['probability'] for r in rows]
    assert all(r['state'] == COMPLETED for r in again)
