"""NBA adapter for the shared pipeline (roadmap M4.1; owner decision D2).

Features follow the existing NBA research model
(``app_core.prospective_research_models``). For each team they use its last 2–8
completed games within 240 days, dated strictly before the as-of date: points
for, points against, margin, and rest days capped at 7.

The model is a ridge regression for the full-game margin and total. It is fitted
here, on completed games only, and saved as a versioned artifact. Probabilities
use the research model's discrete-score ``score_probabilities``, so an integer
line gets a real push probability. ``Forecast.probability`` is then
P(win | no push), and ``push_probability`` is reported separately.

Without a fitted artifact every forecast is UNAVAILABLE (``NBA_MODEL_NOT_TRAINED``):
no hand-set parameters are invented. Moneyline is not qualified (roadmap P3).
Neutral-site games are rejected until neutral handling is qualified. Injury
and lineup inputs are absent and disclosed (``NO_AUTHENTIC_INJURY_LINEUP_FEATURE``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path

from app_core.pipeline.adapter import AdapterError, FeatureSet, Forecast, Market, SportAdapter, instant, register, require
from app_core.prospective_research_models import (HISTORY_DAYS, MAX_TEAM_HISTORY, MIN_TEAM_HISTORY, RIDGE_STRENGTHS,
                                                  _expected, _fit, score_probabilities)

MODEL_VERSION = 'nba-score-ridge-v1'
SEMANTICS = 'full_game_including_overtime'
LIMITATIONS = ('NO_AUTHENTIC_INJURY_LINEUP_FEATURE',)
ARTIFACT = Path(__file__).resolve().parents[2] / 'models' / 'nba-score-ridge-v1.json'
MIN_TRAIN_GAMES = 200
HOLDOUT_SHARE = 0.2
MARKETS = {
    'spread_home': Market('spread_home', 'SPREAD', 'home'),
    'spread_away': Market('spread_away', 'SPREAD', 'away'),
    'total_over': Market('total_over', 'TOTAL', 'over'),
    'total_under': Market('total_under', 'TOTAL', 'under'),
    'moneyline_home': Market('moneyline_home', 'MONEYLINE', 'home', supported=False,
                             unsupported_reason='NBA_MONEYLINE_NOT_QUALIFIED'),
    'moneyline_away': Market('moneyline_away', 'MONEYLINE', 'away', supported=False,
                             unsupported_reason='NBA_MONEYLINE_NOT_QUALIFIED'),
}
HISTORY_FIELDS = {'game_date', 'home', 'away', 'home_score', 'away_score'}


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
        require(isinstance(row, dict) and HISTORY_FIELDS <= set(row), 'NBA_HISTORY_SCHEMA')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('home_score', 'away_score')), 'NBA_HISTORY_SCORE')
        require(row['home'] != row['away'], 'NBA_HISTORY_SCHEMA')


@dataclass(frozen=True)
class Model:
    margin: dict
    total: dict
    trained_through: str
    train_games: int
    holdout_games: int
    holdout: dict
    version: str = MODEL_VERSION
    limitations: tuple = LIMITATIONS

    def save(self, path=ARTIFACT):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path=ARTIFACT):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        require(data.get('version') == MODEL_VERSION, 'NBA_MODEL_VERSION_UNSUPPORTED')
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
        raise AdapterError(f'NBA_TOO_FEW_TRAINING_GAMES:{len(train)}')
    require(train[-1][0] <= test[0][0], 'NBA_HOLDOUT_NOT_LATER')
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


class NBAAdapter(SportAdapter):
    sport = 'NBA'

    def __init__(self, model: Model | None = None, artifact=ARTIFACT):
        self._model, self._artifact = model, Path(artifact)

    def model(self):
        if self._model is None and self._artifact.is_file():
            self._model = Model.load(self._artifact)
        return self._model

    def markets(self):
        return dict(MARKETS)

    def features(self, event, history, *, as_of_utc):
        require(isinstance(event, dict) and {'event_id', 'home', 'away', 'kickoff_utc'} <= set(event), 'NBA_EVENT_SCHEMA')
        if event.get('neutral_site'):
            raise AdapterError('NBA_NEUTRAL_SITE_UNSUPPORTED')
        _validate(history)
        before = instant(as_of_utc).date()
        home, away = _form(history, event['home'], before), _form(history, event['away'], before)
        if home is None or away is None:
            raise AdapterError('NBA_INSUFFICIENT_TEAM_HISTORY')
        margin_x, total_x = vectors(home, away)
        values = {f'margin_x{i}': v for i, v in enumerate(margin_x)}
        values.update({f'total_x{i}': v for i, v in enumerate(total_x)})
        return FeatureSet('NBA', event['event_id'], event['home'], event['away'], event['kickoff_utc'], as_of_utc,
                          values, dict(home_games=home['games'], away_games=away['games'],
                                       cutoff_rule='game_date < as_of date', limitations=list(LIMITATIONS)))

    def forecast(self, features, market_type, line):
        market = self.market(market_type)
        if not market.supported:
            return self.unavailable(features, market_type, line, market.unsupported_reason, model_id=MODEL_VERSION)
        if not (isinstance(line, (int, float)) and not isinstance(line, bool) and math.isfinite(line)):
            return self.unavailable(features, market_type, line, 'NBA_EXACT_LINE_MISSING', model_id=MODEL_VERSION)
        model = self.model()
        if model is None:
            return self.unavailable(features, market_type, line, 'NBA_MODEL_NOT_TRAINED', model_id=MODEL_VERSION)
        if market.family == 'TOTAL':
            x = [features.values[f'total_x{i}'] for i in range(7)]
            fit_, canonical_line = model.total, float(line)
            selection = 'canonical' if market.side == 'over' else 'opposite'
        else:
            x = [features.values[f'margin_x{i}'] for i in range(7)]
            fit_ = model.margin
            # The research orientation is the home side: an away handicap of +L is a home handicap of -L.
            canonical_line = float(line) if market.side == 'home' else -float(line)
            selection = 'canonical' if market.side == 'home' else 'opposite'
        mean = _expected(fit_, x)
        try:
            probs = score_probabilities(mean, fit_['residual_sd'], market.family, canonical_line, selection=selection)
        except ValueError:
            return self.unavailable(features, market_type, line, 'NBA_LINE_INCREMENT_UNSUPPORTED', model_id=MODEL_VERSION)
        decided = probs['win'] + probs['loss']
        return Forecast('NBA', features.event_id, market_type, float(line), probs['win'] / decided, 'AVAILABLE',
                        'AVAILABLE', MODEL_VERSION, 'total' if market.family == 'TOTAL' else 'spread_cover', SEMANTICS,
                        features.sha256, projection=round(mean, 3), push_probability=round(probs['push'], 6))


ADAPTER = NBAAdapter()
if ARTIFACT.is_file():  # NBA is a supported sport only once a trained, reviewed artifact exists
    register(ADAPTER)
