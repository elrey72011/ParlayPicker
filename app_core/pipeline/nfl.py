"""NFL adapter for the shared pipeline (roadmap M1.3).

The model is unchanged: forecasts call ``predict_market_probabilities`` with
NFL ``score-distribution-v1`` parameters, the same function ordinary Refresh
and the Guided workflow use. Features match ``fetch_nfl_stats`` exactly. A
parity test enforces that: completed games scheduled on a date strictly before
the ``as_of`` date, with the same per-team aggregates.

Supported: full-game spread (home/away) and total (over/under) at half-point
lines, overtime included. Integer lines are rejected because the model has no
push probability. Moneyline is not qualified (roadmap P3).

Legacy private NFL packets (eight versions in ``nfl_private_routes``) keep
their existing reader unchanged. New NFL sources supply completed-game
``history`` rows to ``features`` instead of adding another packet module.
"""
from __future__ import annotations

from datetime import date
import math

import pandas as pd

from app_core.pipeline.adapter import (AdapterError, FeatureSet, Forecast, Market, SportAdapter, instant,
                                       register, require)

MODEL_ID = 'score-distribution-v1:nfl'
SEMANTICS = 'full_game_including_overtime_half_point_line'
MARKETS = {
    'spread_home': Market('spread_home', 'SPREAD', 'home'),
    'spread_away': Market('spread_away', 'SPREAD', 'away'),
    'total_over': Market('total_over', 'TOTAL', 'over'),
    'total_under': Market('total_under', 'TOTAL', 'under'),
    'moneyline_home': Market('moneyline_home', 'MONEYLINE', 'home', supported=False,
                             unsupported_reason='NFL_MONEYLINE_NOT_QUALIFIED'),
    'moneyline_away': Market('moneyline_away', 'MONEYLINE', 'away', supported=False,
                             unsupported_reason='NFL_MONEYLINE_NOT_QUALIFIED'),
}
HISTORY_FIELDS = {'game_date', 'home', 'away', 'home_score', 'away_score'}


def team_aggregates(history, team):
    """Per-team aggregates, identical to ``fetch_nfl_stats`` for one team."""
    games = []
    for row in sorted(history, key=lambda r: r['game_date']):
        if team == row['home']:
            scored, allowed = row['home_score'], row['away_score']
        elif team == row['away']:
            scored, allowed = row['away_score'], row['home_score']
        else:
            continue
        games.append(dict(scored=float(scored), allowed=float(allowed), won=scored > allowed,
                          margin=float(scored) - float(allowed)))
    if not games:
        return None
    count, recent = len(games), games[-5:]
    return dict(
        win_pct=sum(g['won'] for g in games) / count,
        points_per_game=sum(g['scored'] for g in games) / count,
        points_allowed_per_game=sum(g['allowed'] for g in games) / count,
        last5_win_pct=sum(g['won'] for g in recent) / len(recent),
        recent_point_margin=sum(g['margin'] for g in recent) / len(recent),
        games_played=count,
    )


class NFLAdapter(SportAdapter):
    sport = 'NFL'

    def markets(self):
        return dict(MARKETS)

    def features(self, event, history, *, as_of_utc):
        require(isinstance(event, dict) and {'event_id', 'home', 'away', 'kickoff_utc'} <= set(event),
                'NFL_EVENT_SCHEMA')
        cutoff_day = instant(as_of_utc).date()
        eligible = []
        for row in history:
            require(isinstance(row, dict) and HISTORY_FIELDS <= set(row), 'NFL_HISTORY_SCHEMA')
            require(all(type(row[k]) is int and row[k] >= 0 for k in ('home_score', 'away_score')), 'NFL_HISTORY_SCORE')
            played = date.fromisoformat(row['game_date'])
            if played < cutoff_day:  # same date-level cutoff as fetch_nfl_stats
                eligible.append(row)
        home, away = (team_aggregates(eligible, event[side]) for side in ('home', 'away'))
        if home is None or away is None:
            raise AdapterError('NFL_NO_COMPLETED_POINT_IN_TIME_GAMES')
        values = {
            'feature_home_win_pct': home['win_pct'], 'feature_away_win_pct': away['win_pct'],
            'feature_home_ppg': home['points_per_game'], 'feature_away_ppg': away['points_per_game'],
            'feature_home_oppg': home['points_allowed_per_game'], 'feature_away_oppg': away['points_allowed_per_game'],
            'feature_home_games_played': float(home['games_played']), 'feature_away_games_played': float(away['games_played']),
            'feature_home_recent_point_margin': home['recent_point_margin'],
            'feature_away_recent_point_margin': away['recent_point_margin'],
            'feature_diff_last5': home['last5_win_pct'] - away['last5_win_pct'],
        }
        provenance = dict(history_games=len(eligible), cutoff_rule='game_date < as_of date',
                          last_game_date=max(r['game_date'] for r in eligible))
        return FeatureSet('NFL', event['event_id'], event['home'], event['away'], event['kickoff_utc'],
                          as_of_utc, values, provenance)

    def forecast(self, features, market_type, line):
        market = self.market(market_type)
        if not market.supported:
            return self.unavailable(features, market_type, line, market.unsupported_reason, model_id=MODEL_ID)
        if not (isinstance(line, (int, float)) and not isinstance(line, bool) and math.isfinite(line)):
            return self.unavailable(features, market_type, line, 'NFL_EXACT_LINE_MISSING', model_id=MODEL_ID)
        if (line * 2) % 2 != 1:
            return self.unavailable(features, market_type, line, 'NFL_INTEGER_LINE_UNSUPPORTED', model_id=MODEL_ID)
        from app_core.market_probability_model import predict_market_probabilities
        row = dict(features.values, League='NFL', league='NFL', market_type=market_type,
                   home_team=features.home, away_team=features.away, game_start_utc=features.kickoff_utc,
                   ml_feature_eligible=True, stats_resolution_status='resolved')
        row['total_line' if market.family == 'TOTAL' else 'spread_line'] = float(line)
        out = predict_market_probabilities(pd.DataFrame([row])).iloc[0]
        if out['ml_inference_status'] != 'success':
            reason = str(out['ml_unavailable_reason'] or 'NFL_MODEL_UNAVAILABLE')
            return self.unavailable(features, market_type, line, reason, model_id=MODEL_ID)
        require(out['ml_probability_source'] == MODEL_ID, 'NFL_MODEL_IDENTITY_CONFLICT')
        return Forecast('NFL', features.event_id, market_type, float(line), float(out['ml_probability']),
                        'AVAILABLE', 'AVAILABLE', MODEL_ID, str(out['ml_target']), SEMANTICS, features.sha256,
                        projection=float(out['ml_projection']))


ADAPTER = register(NFLAdapter())
