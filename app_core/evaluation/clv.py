"""Closing-line value (CLV) from verified closing observations (roadmap E1.4).

Reads ``closing_observations`` from a prediction-evidence SQLite database,
opened read-only. Each record's hash is verified exactly as
``app_core.activation_closing.observations`` does. A pick can have several
closing observations; its close is the latest one captured before kickoff.

Reported per group (overall, league, league × market):
- ``n``: picks with a close;
- beat-close rate: share of picks whose line or price beat the close, with a
  Wilson 95% interval;
- mean line CLV (points in the pick's favor) and mean price CLV, each with a
  bootstrap 95% interval.

Positive CLV over a large sample is the fastest reliable sign of an edge. It is
evidence, not proof of profit.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sqlite3

from app_core.evaluation.scoreboard import BOOTSTRAP, SEED, _bootstrap_mean, _round, wilson
from core.exposure_ledger import digest

FAMILY = {'spread_home': 'SPREAD', 'spread_away': 'SPREAD', 'total_over': 'TOTAL', 'total_under': 'TOTAL',
          'h2h_home': 'MONEYLINE', 'h2h_away': 'MONEYLINE', 'moneyline_home': 'MONEYLINE', 'moneyline_away': 'MONEYLINE'}


def read_observations(path):
    """Verified closing observations from an evidence database, without writing to it."""
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    con = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='closing_observations'").fetchone()
        if not exists:
            return []
        rows = con.execute('SELECT observation_id,payload FROM closing_observations').fetchall()
    finally:
        con.close()
    out = []
    for key, raw in rows:
        value = json.loads(raw)
        if digest(value) != key:
            raise ValueError('CLOSING_OBSERVATION_CHANGED')
        out.append(value)
    return out


def latest_closes(observations):
    """One close per pick: the latest verified observation."""
    best = {}
    for obs in observations:
        if not obs.get('quote_verified'):
            continue
        key = (obs.get('snapshot_id'), obs.get('candidate_id'))
        if key not in best or obs.get('closing_capture_at', '') > best[key].get('closing_capture_at', ''):
            best[key] = obs
    return list(best.values())


def _group(obs):
    quote = obs.get('quote', {})
    league = str(quote.get('sport', 'UNKNOWN')).upper()
    return league, f"{league} {FAMILY.get(str(quote.get('market_type', '')).lower(), 'UNKNOWN')}"


def metrics(closes):
    rng = random.Random(SEED)
    out = dict(n=len(closes))
    decided = [c['beat_close'] for c in closes if c.get('beat_close') is not None]
    if decided:
        wins = sum(bool(v) for v in decided)
        low, high = wilson(wins, len(decided))
        out.update(beat_close_rate=_round(wins / len(decided)), beat_close_ci=(_round(low), _round(high)),
                   decided=len(decided))
    for name in ('line_clv', 'price_clv'):
        values = [float(c[name]) for c in closes if isinstance(c.get(name), (int, float))]
        if values:
            out[f'mean_{name}'] = _round(sum(values) / len(values))
            out[f'mean_{name}_ci'] = tuple(_round(v) for v in _bootstrap_mean(values, rng))
            out[f'{name}_n'] = len(values)
    return out


def report(observations):
    closes = latest_closes(observations)
    by_league, by_market = {}, {}
    for close in closes:
        league, market = _group(close)
        by_league.setdefault(league, []).append(close)
        by_market.setdefault(market, []).append(close)
    return dict(overall=metrics(closes),
                by_league={k: metrics(v) for k, v in sorted(by_league.items())},
                by_league_market={k: metrics(v) for k, v in sorted(by_market.items())})


def markdown(clv_report):
    def fmt(v, ci=None):
        if v is None:
            return '—'
        text = f'{v:+.3f}'
        return text + (f' [{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '')
    head = '| Group | Picks with close | Beat close [95% CI] | Mean line CLV [95% CI] | Mean price CLV [95% CI] |\n|---|---|---|---|---|'

    def line(name, m):
        return (f"| {name} | {m['n']} | {fmt(m.get('beat_close_rate'), m.get('beat_close_ci'))} "
                f"| {fmt(m.get('mean_line_clv'), m.get('mean_line_clv_ci'))} "
                f"| {fmt(m.get('mean_price_clv'), m.get('mean_price_clv_ci'))} |")
    parts = ['### Closing-line value', head, line('All', clv_report['overall'])]
    parts += [line(k, v) for k, v in clv_report['by_league'].items()]
    parts += [line(k, v) for k, v in clv_report['by_league_market'].items()]
    return '\n'.join(parts) + '\n'
