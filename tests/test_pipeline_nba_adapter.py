"""SYNTHETIC NBA adapter and model fitting (roadmap M4.1). No network, no real games."""
from datetime import date, timedelta
import random

import pytest

from app_core.pipeline.adapter import AdapterError
from app_core.pipeline.nba import NBAAdapter, Model, fit

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
EVENT = dict(event_id='SYNTHETIC-NBA-1', home=TEAMS[9], away=TEAMS[0], kickoff_utc='2027-03-01T00:30:00+00:00')
AS_OF = '2027-02-28T18:00:00+00:00'


@pytest.fixture(scope='module')
def model():
    return fit(HISTORY)


def test_fit_uses_an_honest_time_ordered_holdout_and_beats_the_naive_baseline(model):
    assert model.train_games >= 200 and model.holdout_games > 0
    later = [g['game_date'] for g in HISTORY if g['game_date'] > model.trained_through]
    assert later and min(later) > model.trained_through
    assert model.holdout['margin']['rmse'] < model.holdout['margin']['baseline_rmse']
    assert model.limitations == ('NO_AUTHENTIC_INJURY_LINEUP_FEATURE',)


def test_too_little_history_refuses_to_fit():
    with pytest.raises(AdapterError, match='NBA_TOO_FEW_TRAINING_GAMES'):
        fit(HISTORY[:150])


def test_features_use_only_earlier_games_and_reject_neutral_sites():
    nba = NBAAdapter(model=None)
    cutoff = [g for g in HISTORY if g['game_date'] < '2027-02-28']
    on_the_day = cutoff + [dict(game_date='2027-02-28', home=EVENT['home'], away=EVENT['away'], home_score=200, away_score=0)]
    assert nba.features(EVENT, on_the_day, as_of_utc=AS_OF).values == nba.features(EVENT, cutoff, as_of_utc=AS_OF).values
    with pytest.raises(AdapterError, match='NBA_NEUTRAL_SITE_UNSUPPORTED'):
        nba.features(dict(EVENT, neutral_site=True), HISTORY, as_of_utc=AS_OF)
    with pytest.raises(AdapterError, match='NBA_INSUFFICIENT_TEAM_HISTORY'):
        nba.features(dict(EVENT, away='SYNTHETIC-NEW'), HISTORY, as_of_utc=AS_OF)


def test_without_a_trained_model_no_probability_is_invented(tmp_path):
    nba = NBAAdapter(model=None, artifact=tmp_path / 'absent.json')
    forecast = nba.forecast(nba.features(EVENT, HISTORY, as_of_utc=AS_OF), 'spread_home', -4.5)
    assert (forecast.status, forecast.reason, forecast.probability) == ('UNAVAILABLE', 'NBA_MODEL_NOT_TRAINED', None)


def test_sides_are_complementary_and_integer_lines_report_pushes(model):
    nba = NBAAdapter(model=model)
    features = nba.features(EVENT, HISTORY, as_of_utc=AS_OF)
    home, away = nba.forecast(features, 'spread_home', -4.5), nba.forecast(features, 'spread_away', 4.5)
    assert home.status == away.status == 'AVAILABLE' and home.push_probability == 0
    assert home.probability + away.probability == pytest.approx(1.0)
    over, under = nba.forecast(features, 'total_over', 224.5), nba.forecast(features, 'total_under', 224.5)
    assert over.probability + under.probability == pytest.approx(1.0)
    whole_home, whole_away = nba.forecast(features, 'spread_home', -4.0), nba.forecast(features, 'spread_away', 4.0)
    assert whole_home.push_probability > 0 and whole_home.push_probability == whole_away.push_probability
    assert whole_home.probability + whole_away.probability == pytest.approx(1.0)  # conditional on no push
    assert home.projection > 0  # the stronger home team is projected to win
    assert nba.forecast(features, 'moneyline_home', None).reason == 'NBA_MONEYLINE_NOT_QUALIFIED'
    assert nba.forecast(features, 'spread_home', -4.25).reason == 'NBA_LINE_INCREMENT_UNSUPPORTED'


def test_model_artifact_round_trips(model, tmp_path):
    path = tmp_path / 'nba.json'
    model.save(path)
    loaded = Model.load(path)
    assert loaded == model
    nba = NBAAdapter(artifact=path)
    assert nba.forecast(nba.features(EVENT, HISTORY, as_of_utc=AS_OF), 'total_over', 224.5).status == 'AVAILABLE'


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
