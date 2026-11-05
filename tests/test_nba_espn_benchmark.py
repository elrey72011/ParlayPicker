"""SYNTHETIC offline NBA ESPN benchmark (owner decision D9). No network, no real games."""
from datetime import date, timedelta
import random

import pytest

from app_core.nba_espn_benchmark import Model, examples, fit

TEAMS = [f'SYNTHETIC-T{i}' for i in range(10)]
STRENGTH = {team: (i - 4.5) * 1.6 for i, team in enumerate(TEAMS)}


def season(days=130, seed=7):
    """Ten teams with stable strengths: five games a day, home edge ~2.5 points, noisy scores."""
    rng = random.Random(seed)
    games, start = [], date(2026, 10, 21)
    for d in range(days):
        teams = TEAMS[:]
        rng.shuffle(teams)
        for i in range(0, 10, 2):
            home, away = teams[i], teams[i + 1]
            base = 112 + rng.gauss(0, 6)
            margin = 2.5 + STRENGTH[home] - STRENGTH[away] + rng.gauss(0, 11)
            games.append(dict(game_date=(start + timedelta(days=d)).isoformat(), home=home, away=away,
                              home_score=max(60, int(round(base + margin / 2))),
                              away_score=max(60, int(round(base - margin / 2)))))
    return games


HISTORY = season()


@pytest.fixture(scope='module')
def model():
    return fit(HISTORY)


def test_fit_uses_an_honest_time_ordered_holdout_and_beats_the_naive_baseline(model):
    assert model.train_games >= 200 and model.holdout_games > 0
    later = [g['game_date'] for g in HISTORY if g['game_date'] > model.trained_through]
    assert later and min(later) > model.trained_through
    assert model.holdout['margin']['rmse'] < model.holdout['margin']['baseline_rmse']
    assert model.limitations == ('NO_AUTHENTIC_INJURY_LINEUP_FEATURE', 'NO_QUOTE_LINEAGE')
    assert model.status == 'OFFLINE_BENCHMARK'


def test_too_little_history_refuses_to_fit():
    with pytest.raises(ValueError, match='NBA_TOO_FEW_TRAINING_GAMES'):
        fit(HISTORY[:150])


def test_training_script_pairs_game_finder_rows_and_fits_from_csv(tmp_path):
    from scripts import train_nba_model as cli
    rows = [dict(GAME_ID='1', MATCHUP='BOS vs. NYK', TEAM_NAME='Boston Celtics', PTS=110, GAME_DATE='2026-11-01'),
            dict(GAME_ID='1', MATCHUP='NYK @ BOS', TEAM_NAME='New York Knicks', PTS=104, GAME_DATE='2026-11-01'),
            dict(GAME_ID='2', MATCHUP='LAL vs. GSW', TEAM_NAME='Los Angeles Lakers', PTS=None, GAME_DATE='2026-11-02'),
            dict(GAME_ID='2', MATCHUP='GSW @ LAL', TEAM_NAME='Golden State Warriors', PTS=99, GAME_DATE='2026-11-02'),
            dict(GAME_ID='3', MATCHUP='MIA vs. ORL', TEAM_NAME='Miami Heat', PTS=101, GAME_DATE='2026-11-03')]
    assert cli.games_from_game_finder(rows) == [dict(game_date='2026-11-01', home='Boston Celtics',
                                                     away='New York Knicks', home_score=110, away_score=104)]
    import csv as _csv
    names = dict(zip(TEAMS, sorted(cli.NBA_FRANCHISES)))  # the training filter keeps NBA franchises only
    franchise_history = [dict(g, home=names[g['home']], away=names[g['away']]) for g in HISTORY]
    path = tmp_path / 'games.csv'
    with path.open('w', newline='') as fh:
        writer = _csv.DictWriter(fh, fieldnames=['game_date', 'home', 'away', 'home_score', 'away_score'])
        writer.writeheader()
        writer.writerows(franchise_history)
    out = tmp_path / 'model.json'
    assert cli.main(['--csv', str(path), '--output', str(out)]) == 0
    assert Model.load(out).train_games >= 200


def test_espn_parser_keeps_completed_non_neutral_regular_season_games_on_the_us_date():
    from scripts.train_nba_model import games_from_espn

    def event(eid, season_type=2, neutral=False, completed=True, home=110, away=101):
        return dict(id=eid, date='2026-11-06T00:30Z', season=dict(type=season_type), competitions=[dict(
            neutralSite=neutral, status=dict(type=dict(completed=completed)), competitors=[
                dict(homeAway='home', score=str(home), team=dict(displayName='SYNTHETIC Home')),
                dict(homeAway='away', score=str(away), team=dict(displayName='SYNTHETIC Away'))])])
    payload = dict(events=[event('1'), event('2', season_type=3), event('3', neutral=True),
                           event('4', completed=False), event('5', home='x')])
    assert games_from_espn(payload, '2026-11-05') == [dict(game_date='2026-11-05', home='SYNTHETIC Home',
                                                           away='SYNTHETIC Away', home_score=110, away_score=101,
                                                           espn_event_id='1')]


def test_all_star_and_duplicate_games_are_excluded_from_training():
    from scripts.train_nba_model import franchise_games
    games = [dict(game_date='2026-02-15', home='Team Stripes', away='Team Stars', home_score=42, away_score=40),
             dict(game_date='2026-02-20', home='Boston Celtics', away='Miami Heat', home_score=110, away_score=100),
             dict(game_date='2026-02-20', home='Boston Celtics', away='Miami Heat', home_score=110, away_score=100),
             dict(game_date='2026-02-21', home='LA Clippers', away='World', home_score=90, away_score=80)]
    assert franchise_games(games) == [games[1]]


def test_features_use_only_earlier_games_and_the_artifact_round_trips(model, tmp_path):
    later = [dict(game_date='2027-03-05', home=TEAMS[0], away=TEAMS[1], home_score=200, away_score=0)]
    before = {r[0]: r[1:3] for r in examples(HISTORY)}
    after = {r[0]: r[1:3] for r in examples(HISTORY + later) if r[0] in before}
    assert after == before  # a later blowout never changes an earlier game's features
    path = tmp_path / 'benchmark.json'
    model.save(path)
    assert Model.load(path) == model
