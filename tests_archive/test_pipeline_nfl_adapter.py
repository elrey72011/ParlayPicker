"""SYNTHETIC NFL adapter behavior (roadmap M1.3). No network, no real games."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app_core.pipeline.adapter import AdapterError, Forecast, adapter_for
from app_core.pipeline.nfl import NFLAdapter, team_aggregates
from app_core.pipeline.operations import COMPLETED, Operations
from app_core.pipeline.store import Store

AS_OF = '2026-11-08T12:00:00+00:00'
KICKOFF = '2026-11-08T18:00:00+00:00'
GAMES = [  # (date, home, away, home_score, away_score)
    ('2026-09-07', 'KC', 'BUF', 27, 20), ('2026-09-14', 'BUF', 'KC', 17, 24),
    ('2026-09-21', 'KC', 'DEN', 31, 10), ('2026-09-28', 'DEN', 'BUF', 13, 23),
    ('2026-10-05', 'BUF', 'DEN', 30, 27), ('2026-10-12', 'DEN', 'KC', 20, 17),
    ('2026-10-19', 'KC', 'BUF', 21, 24), ('2026-11-01', 'BUF', 'KC', 28, 10),
    ('2026-11-08', 'KC', 'DEN', 99, 0),   # on the as-of date: must be excluded
    ('2026-11-15', 'DEN', 'BUF', 0, 99),  # in the future: must be excluded
]
HISTORY = [dict(game_date=d, home=h, away=a, home_score=hs, away_score=as_) for d, h, a, hs, as_ in GAMES]
EVENT = dict(event_id='SYNTHETIC-NFL-KC-BUF', home='KC', away='BUF', kickoff_utc=KICKOFF)


@pytest.fixture
def nfl():
    return NFLAdapter()


def test_registry_returns_the_nfl_adapter():
    assert isinstance(adapter_for('NFL'), NFLAdapter)
    with pytest.raises(AdapterError, match='ADAPTER_SPORT_NOT_SUPPORTED'):
        adapter_for('NBA')


def test_feature_aggregates_match_ordinary_refresh_fetch_nfl_stats(monkeypatch):
    """Parity: the adapter computes exactly the inputs ordinary Refresh computes today."""
    from app_core import feature_processing, nfl_native_provenance
    from core.nfl_teams import nfl_stats_identity
    schedule = pd.DataFrame([dict(gameday=d, home_team=h, away_team=a, home_score=hs, away_score=as_, result=hs - as_)
                             for d, h, a, hs, as_ in GAMES])

    class FakeNfl:
        @staticmethod
        def import_schedules(years):
            return schedule.copy()
    monkeypatch.setattr(feature_processing, 'nfl', FakeNfl)
    monkeypatch.setattr(nfl_native_provenance, 'observe', lambda df, module: None)
    monkeypatch.setattr(nfl_native_provenance, 'retain_stats', lambda *a, **k: None)
    legacy = {row['team_norm']: row for row in feature_processing.fetch_nfl_stats(2026, as_of_date='2026-11-08')}
    eligible = [r for r in HISTORY if r['game_date'] < '2026-11-08']
    for code in ('KC', 'BUF', 'DEN'):
        ours, theirs = team_aggregates(eligible, code), legacy[nfl_stats_identity(code, schedule_code=True)]
        for key in ('win_pct', 'points_per_game', 'points_allowed_per_game', 'last5_win_pct',
                    'recent_point_margin', 'games_played'):
            assert ours[key] == pytest.approx(theirs[key]), (code, key)


def test_features_use_only_games_before_the_as_of_date(nfl):
    features = nfl.features(EVENT, HISTORY, as_of_utc=AS_OF)
    assert features.provenance['history_games'] == 8 and features.provenance['last_game_date'] == '2026-11-01'
    assert features.values['feature_home_games_played'] == 6.0  # KC: 6 completed games before Nov 8
    leaked = nfl.features(EVENT, [r for r in HISTORY if r['home_score'] != 99 and r['away_score'] != 99], as_of_utc=AS_OF)
    assert leaked.values == features.values


def test_features_after_kickoff_or_without_history_are_rejected(nfl):
    with pytest.raises(AdapterError, match='ADAPTER_FEATURES_AFTER_KICKOFF'):
        nfl.features(EVENT, HISTORY, as_of_utc='2026-11-08T18:00:01+00:00')
    with pytest.raises(AdapterError, match='NFL_NO_COMPLETED_POINT_IN_TIME_GAMES'):
        nfl.features(dict(EVENT, away='NYJ'), HISTORY, as_of_utc=AS_OF)
    with pytest.raises(AdapterError, match='NFL_HISTORY_SCORE'):
        nfl.features(EVENT, [dict(HISTORY[0], home_score=27.5)], as_of_utc=AS_OF)


def test_spread_sides_and_total_sides_are_complementary(nfl):
    features = nfl.features(EVENT, HISTORY, as_of_utc=AS_OF)
    home = nfl.forecast(features, 'spread_home', -2.5)
    away = nfl.forecast(features, 'spread_away', 2.5)
    over, under = nfl.forecast(features, 'total_over', 45.5), nfl.forecast(features, 'total_under', 45.5)
    for forecast in (home, away, over, under):
        assert forecast.status == 'AVAILABLE' and forecast.model_id == 'score-distribution-v1:nfl'
        assert 0.20 <= forecast.probability <= 0.80 and forecast.features_sha256 == features.sha256
    assert home.probability + away.probability == pytest.approx(1.0)
    assert over.probability + under.probability == pytest.approx(1.0)
    assert home.target == 'spread_cover' and over.target == 'total_over'


def test_forecast_matches_a_direct_call_of_the_unchanged_model(nfl):
    from app_core.market_probability_model import predict_market_probabilities
    features = nfl.features(EVENT, HISTORY, as_of_utc=AS_OF)
    direct = predict_market_probabilities(pd.DataFrame([dict(features.values, League='NFL', market_type='spread_home',
                                                             spread_line=-2.5, ml_feature_eligible=True,
                                                             stats_resolution_status='resolved')])).iloc[0]
    assert nfl.forecast(features, 'spread_home', -2.5).probability == pytest.approx(float(direct['ml_probability']))


@pytest.mark.parametrize('market_type,line,reason', [
    ('spread_home', -3.0, 'NFL_INTEGER_LINE_UNSUPPORTED'),
    ('total_over', 44, 'NFL_INTEGER_LINE_UNSUPPORTED'),
    ('spread_away', None, 'NFL_EXACT_LINE_MISSING'),
    ('moneyline_home', None, 'NFL_MONEYLINE_NOT_QUALIFIED'),
])
def test_unsupported_requests_return_a_specific_reason_not_a_probability(nfl, market_type, line, reason):
    forecast = nfl.forecast(nfl.features(EVENT, HISTORY, as_of_utc=AS_OF), market_type, line)
    assert (forecast.status, forecast.reason, forecast.probability) == ('UNAVAILABLE', reason, None)


def test_unknown_market_is_rejected(nfl):
    with pytest.raises(AdapterError, match='ADAPTER_MARKET_UNKNOWN'):
        nfl.forecast(nfl.features(EVENT, HISTORY, as_of_utc=AS_OF), 'first_half_spread_home', -1.5)


def test_grading_follows_declared_settlement_including_pushes(nfl):
    def forecast(market_type, line):
        return Forecast('NFL', 'SYNTHETIC', market_type, line, 0.5, 'AVAILABLE', 'AVAILABLE', 'm', 't', 's', 'x')
    assert nfl.grade(forecast('spread_home', -2.5), home_score=24, away_score=20) == 'WIN'
    assert nfl.grade(forecast('spread_away', 2.5), home_score=24, away_score=20) == 'LOSS'
    assert nfl.grade(forecast('spread_home', -4.0), home_score=24, away_score=20) == 'PUSH'
    assert nfl.grade(forecast('total_over', 44.5), home_score=24, away_score=20) == 'LOSS'
    assert nfl.grade(forecast('total_under', 44.5), home_score=24, away_score=20) == 'WIN'
    assert nfl.grade(forecast('total_over', 44.0), home_score=24, away_score=20) == 'PUSH'
    with pytest.raises(AdapterError, match='ADAPTER_GRADE_UNAVAILABLE_FORECAST'):
        nfl.grade(nfl.forecast(nfl.features(EVENT, HISTORY, as_of_utc=AS_OF), 'spread_home', -3.0),
                  home_score=24, away_score=20)


def test_two_different_games_run_through_operations_without_game_specific_code(nfl, tmp_path):
    clock = lambda: datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)
    ops = Operations(Store.sqlite(tmp_path / 'SYNTHETIC-nfl.sqlite', b'SYNTHETIC-nfl-adapter-test-key-not-prod',
                                  owner='SYNTHETIC owner'), clock=clock)
    games = [(dict(EVENT), 'spread_home', -2.5),
             (dict(event_id='SYNTHETIC-NFL-DEN-BUF', home='DEN', away='BUF', kickoff_utc=KICKOFF), 'total_under', 41.5)]
    results = {}
    for event, market_type, line in games:
        subject = dict(event=event['event_id'], market_type=market_type, line=line)
        op = ops.create('NFL', 'calculation', subject, not_after=KICKOFF, review_required=False)

        def work(operation, event=event, market_type=market_type, line=line):
            return nfl.forecast(nfl.features(event, HISTORY, as_of_utc=AS_OF), market_type, line).record()
        results[event['event_id']] = ops.run('NFL', op['id'], work)['output']
        assert ops.status('NFL', op['id'])['state'] == COMPLETED
    assert {r['event_id'] for r in results.values()} == {'SYNTHETIC-NFL-KC-BUF', 'SYNTHETIC-NFL-DEN-BUF'}
    assert all(r['status'] == 'AVAILABLE' for r in results.values())
