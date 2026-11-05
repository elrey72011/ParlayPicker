"""Offline NBA benchmark trained on ESPN final scores (owner decision D9, 2026-11-05).

The validated NBA path is the existing research model,
``app_core.prospective_research_models`` (``pregame-score-ridge-normal-v1``). The
research scheduler retrains it from captured Odds API events and final scores,
and every training game is bound to a captured quote. This module is not that path
and grants no authority: its ESPN games have no quote lineage, so
``prospective_evidence.insert_model`` cannot accept it.

What it is for: the same features and ridge method, fitted on a longer history
(two full seasons), with an honest time-ordered holdout. It answers whether more
history helps before anyone changes the research model. Reproduce it with
``scripts/train_nba_model.py``.

Features, per team: its last 2–8 completed games within 240 days, dated strictly
before the game. They are points for, points against, margin, and rest days
capped at 7. Only games between the 30 NBA franchises are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path

from app_core.prospective_research_models import (HISTORY_DAYS, MAX_TEAM_HISTORY, MIN_TEAM_HISTORY, RIDGE_STRENGTHS,
                                                  _expected, _fit)

VERSION = 'nba-espn-benchmark-v1'
LIMITATIONS = ('NO_AUTHENTIC_INJURY_LINEUP_FEATURE', 'NO_QUOTE_LINEAGE')
MIN_TRAIN_GAMES = 200
HOLDOUT_SHARE = 0.2
HISTORY_FIELDS = {'game_date', 'home', 'away', 'home_score', 'away_score'}
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# The 30 NBA franchises as ESPN names them. All-Star teams ("Team Shaq", "World")
# and exhibition opponents are excluded.
NBA_FRANCHISES = frozenset({
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
    'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets',
    'Indiana Pacers', 'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks',
    'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic',
    'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
    'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'})


def franchise_games(games):
    """Only games between two NBA franchises, with exact duplicates removed."""
    seen, out = set(), []
    for game in games:
        key = (game['game_date'], game['home'], game['away'])
        if game['home'] in NBA_FRANCHISES and game['away'] in NBA_FRANCHISES and key not in seen:
            seen.add(key)
            out.append(game)
    return out


def games_from_espn(payload, day=None):
    """Completed, non-neutral regular-season games from one ESPN scoreboard payload.

    ``day`` is the requested scoreboard date (US local). It is the game date, because
    ESPN's own ``date`` field is UTC and would move evening games to the next day.
    """
    games = []
    for event in payload.get('events', []):
        if (event.get('season') or {}).get('type') != 2:
            continue
        competition = (event.get('competitions') or [{}])[0]
        if competition.get('neutralSite') or not competition.get('status', {}).get('type', {}).get('completed'):
            continue
        sides = {c.get('homeAway'): c for c in competition.get('competitors', [])}
        if set(sides) != {'home', 'away'}:
            continue
        try:
            hs, as_ = int(sides['home']['score']), int(sides['away']['score'])
        except (KeyError, TypeError, ValueError):
            continue
        games.append(dict(game_date=day or str(event['date'])[:10], home=sides['home']['team']['displayName'],
                          away=sides['away']['team']['displayName'], home_score=hs, away_score=as_,
                          espn_event_id=str(event.get('id'))))
    return games


def _form(history, team, before: date):
    """Last 2–8 games for ``team`` dated within 240 days before ``before``, newest first."""
    window = before - timedelta(days=HISTORY_DAYS)
    rows = []
    for row in history:
        played = date.fromisoformat(row['game_date'])
        if not window <= played < before or team not in (row['home'], row['away']):
            continue
        home = row['home'] == team
        rows.append((played, row['home_score'] if home else row['away_score'],
                     row['away_score'] if home else row['home_score']))
    rows.sort(reverse=True)
    rows = rows[:MAX_TEAM_HISTORY]
    if len(rows) < MIN_TEAM_HISTORY:
        return None
    n = len(rows)
    return dict(points_for=sum(r[1] for r in rows) / n, points_against=sum(r[2] for r in rows) / n,
                margin=sum(r[1] - r[2] for r in rows) / n,
                rest_days=min(7.0, max(0.0, float((before - rows[0][0]).days))), games=n)


def vectors(home, away):
    """Feature vectors, in exactly the research model's order."""
    total = [1.0, home['points_for'], home['points_against'], away['points_for'], away['points_against'],
             home['rest_days'] / 7.0, away['rest_days'] / 7.0]
    margin = [1.0, home['margin'], away['margin'], home['points_for'] - away['points_against'],
              away['points_for'] - home['points_against'], home['rest_days'] / 7.0, away['rest_days'] / 7.0]
    return margin, total


def _validate(history):
    for row in history:
        if not (isinstance(row, dict) and HISTORY_FIELDS <= set(row)) or row['home'] == row['away']:
            raise ValueError('NBA_HISTORY_SCHEMA')
        if not all(type(row[k]) is int and row[k] >= 0 for k in ('home_score', 'away_score')):
            raise ValueError('NBA_HISTORY_SCORE')


@dataclass(frozen=True)
class Model:
    margin: dict
    total: dict
    trained_through: str
    train_games: int
    holdout_games: int
    holdout: dict
    version: str = VERSION
    limitations: tuple = LIMITATIONS
    status: str = 'OFFLINE_BENCHMARK'   # no wager, pick or validation authority

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != VERSION:
            raise ValueError('NBA_BENCHMARK_VERSION_UNSUPPORTED')
        data['limitations'] = tuple(data.get('limitations', ()))
        return cls(**data)


def examples(history):
    """(game_date, margin vector, total vector, margin, total) for every game with enough prior form."""
    _validate(history)
    rows = []
    for row in sorted(history, key=lambda r: r['game_date']):
        played = date.fromisoformat(row['game_date'])
        home, away = _form(history, row['home'], played), _form(history, row['away'], played)
        if home is None or away is None:
            continue
        margin_x, total_x = vectors(home, away)
        rows.append((row['game_date'], margin_x, total_x, row['home_score'] - row['away_score'],
                     row['home_score'] + row['away_score']))
    return rows


def fit(history, *, holdout_share=HOLDOUT_SHARE):
    """Fit margin and total ridge models on the earlier games and test them on the later ones."""
    rows = examples(history)
    split = int(len(rows) * (1 - holdout_share))
    train, test = rows[:split], rows[split:]
    if len(train) < MIN_TRAIN_GAMES or not test:
        raise ValueError(f'NBA_TOO_FEW_TRAINING_GAMES:{len(train)}')
    if train[-1][0] > test[0][0]:
        raise ValueError('NBA_HOLDOUT_NOT_LATER')
    models, report = {}, {}
    for name, xi, yi in (('margin', 1, 3), ('total', 2, 4)):
        # Choose the ridge strength on the last 20% of the training window, then refit on all of it.
        inner = int(len(train) * 0.8)
        best = min(RIDGE_STRENGTHS, key=lambda a: _rmse(_fit([dict(features=r[xi], target=r[yi]) for r in train[:inner]], a),
                                                         train[inner:], xi, yi))
        model = _fit([dict(features=r[xi], target=r[yi]) for r in train], best)
        baseline = sum(r[yi] for r in train) / len(train)
        report[name] = dict(rmse=round(_rmse(model, test, xi, yi), 4),
                            baseline_rmse=round(math.sqrt(sum((r[yi] - baseline) ** 2 for r in test) / len(test)), 4),
                            ridge_strength=best)
        models[name] = model
    return Model(models['margin'], models['total'], trained_through=train[-1][0], train_games=len(train),
                 holdout_games=len(test), holdout=report)


def _rmse(model, rows, xi, yi):
    return math.sqrt(sum((r[yi] - _expected(model, r[xi])) ** 2 for r in rows) / len(rows)) if rows else float('inf')
