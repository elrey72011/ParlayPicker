"""NBA in the shared-pipeline shadow (roadmap M4.1). No network.

History tests use a SYNTHETIC baseline. The end-to-end test uses the shipped
baseline and research artifact, with the ESPN day fetch stubbed out.
"""
import csv
from datetime import datetime, timezone

import pandas as pd
import pytest

from app_core.pipeline import nba_history, orchestrator as orch
from app_core.pipeline.adapter import AdapterError
from app_core.pipeline.nba import ARTIFACT, Model
from app_core.pipeline.operations import COMPLETED, Operations
from app_core.pipeline.store import Store

KEY = 'SYNTHETIC-nba-shadow-test-key-not-production'
AS_OF = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
BASE = [dict(game_date='2026-11-04', home='Boston Celtics', away='Miami Heat', home_score=110, away_score=100)]


@pytest.fixture
def baseline(tmp_path):
    path = tmp_path / 'SYNTHETIC-baseline.csv'
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(BASE[0]))
        writer.writeheader()
        writer.writerows(BASE)
    return path


@pytest.fixture
def make_ops(tmp_path):
    def make(now=AS_OF):
        return Operations(Store.sqlite(tmp_path / 'SYNTHETIC-nba.sqlite', KEY.encode(), owner='SYNTHETIC owner',
                                       clock=lambda: now.isoformat()), clock=lambda: now)
    return make


def day_games(day):
    return [dict(game_date=day.isoformat(), home='LA Clippers', away='Utah Jazz', home_score=101, away_score=99),
            dict(game_date=day.isoformat(), home='Team Stars', away='World', home_score=1, away_score=0)]


def test_completed_days_are_fetched_once_and_reused_after_restart(make_ops, baseline):
    calls = []
    games = nba_history.history(make_ops(), AS_OF.isoformat(), baseline=baseline,
                                fetch=lambda day: calls.append(day.isoformat()) or day_games(day))
    assert calls == ['2026-11-05', '2026-11-06']  # never the as-of day, whose games may not be final
    assert [g['game_date'] for g in games] == ['2026-11-04', '2026-11-05', '2026-11-06']  # non-franchise dropped
    again = nba_history.history(make_ops(), AS_OF.isoformat(), baseline=baseline,
                                fetch=lambda day: calls.append(day.isoformat()) or day_games(day))
    assert again == games and len(calls) == 2


def test_failed_days_get_bounded_successors_and_a_backlog_is_refused(make_ops, baseline):
    calls = []

    def down(day):
        calls.append(day)
        raise ConnectionError('SYNTHETIC outage')
    with pytest.raises(AdapterError, match='NBA_HISTORY_DAY_UNAVAILABLE:2026-11-05'):
        nba_history.history(make_ops(), AS_OF.isoformat(), baseline=baseline, fetch=down)
    assert len(calls) == nba_history.ATTEMPTS_PER_DAY
    with pytest.raises(AdapterError, match='NBA_HISTORY_BACKLOG'):
        nba_history.history(make_ops(), '2027-01-15T00:00:00+00:00', baseline=baseline, fetch=down)
    assert len(calls) == nba_history.ATTEMPTS_PER_DAY  # a long gap fetches nothing


def test_odds_feed_names_map_to_espn_franchises():
    assert nba_history.team('Los Angeles Clippers') == 'LA Clippers'
    assert nba_history.team('Boston Celtics') == 'Boston Celtics'
    assert nba_history.team('Team Stars') is None


def frame():
    kickoff = '2026-11-08T00:30:00+00:00'
    return pd.DataFrame([
        dict(league='NBA', market_type='spread_home', spread_line=-3.5, home_team='Boston Celtics',
             away_team='Los Angeles Clippers', provider_event_id='SYNTHETIC-NBA-E', game_start_utc=kickoff),
        dict(league='NBA', market_type='total_over', total_line=226.5, home_team='Boston Celtics',
             away_team='Los Angeles Clippers', provider_event_id='SYNTHETIC-NBA-E', game_start_utc=kickoff)])


def test_nba_stays_out_of_the_shadow_unless_enabled(tmp_path):
    settings = {'PARLAYPICKER_PIPELINE_V2': 'true', 'PARLAYPICKER_PIPELINE_STORE_KEY': KEY,
                'PARLAYPICKER_PIPELINE_SQLITE_PATH': str(tmp_path / 'SYNTHETIC-off.sqlite')}
    assert orch.shadow(frame(), setting=lambda k, d='': settings.get(k, d)) == ([], 'NO_NFL_SPREAD_OR_TOTAL_ROWS')
    assert orch.requests_from_analysis(frame()) == []  # default stays NFL-only


def test_shipped_artifact_is_research_only():
    model = Model.load(ARTIFACT)
    assert model.status == 'RESEARCH_ONLY' and model.trained_through < nba_history.load_baseline()[-1]['game_date']


def test_enabled_nba_shadow_forecasts_real_teams_and_reuses_stored_results(tmp_path):
    settings = {'PARLAYPICKER_PIPELINE_V2': 'true', 'PARLAYPICKER_PIPELINE_NBA': 'true',
                'PARLAYPICKER_PIPELINE_STORE_KEY': KEY,
                'PARLAYPICKER_PIPELINE_SQLITE_PATH': str(tmp_path / 'SYNTHETIC-on.sqlite')}
    fetched = []

    def run():
        return orch.shadow(frame(), setting=lambda k, d='': settings.get(k, d), clock=lambda: AS_OF,
                           fetch_nba_day=lambda day: fetched.append(day) or [])
    rows, note = run()
    assert note is None and [r['status'] for r in rows] == ['AVAILABLE', 'AVAILABLE'], rows
    assert [r['sport'] for r in rows] == ['NBA', 'NBA'] and 0 < rows[0]['probability'] < 1
    assert len(fetched) == 2  # 2026-11-05 and 2026-11-06, after the shipped baseline
    again, _ = run()
    assert len(fetched) == 2 and [r['probability'] for r in again] == [r['probability'] for r in rows]
    assert all(r['state'] == COMPLETED for r in again)
