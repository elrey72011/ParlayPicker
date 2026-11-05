"""Pick-quality scoreboard (roadmap E1.3).

The scoreboard is read-only: it computes metrics from graded picks and never
alters a stored pick or prediction. For every group (overall, league,
league × market, month) it reports:

- ``n``: graded WIN/LOSS picks. Pushes are counted separately and excluded
  from probability scoring.
- Win rate, with a Wilson 95% interval.
- Brier score and log loss for the model probability, the recorded de-vigged
  market probability (``p − edge``) and a constant 50%.
- ``skill_vs_market``: market log loss minus model log loss, with a paired
  bootstrap 95% interval. A positive value means the model beat the market.
- ROI per unit at the offered price: decimal odds = (EV + 1) / p, using the
  pipeline's own recorded EV. Bootstrap 95% interval.
- Calibration bins (predicted vs. realized).
- CLV: reported as UNAVAILABLE until closing prices are captured (E1.4).

Without a confidence interval a number is noise. Read the interval first.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

BOOTSTRAP = 2000
SEED = 20261105
EPS = 1e-6
BINS = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0001)
MAX_ABS_EV = 1.0        # |EV| per unit staked above this indicates a percent/fraction unit mix-up
MIN_DECIMAL, MAX_DECIMAL = 1.01, 5.0
MAX_MARGIN = 0.06       # implied (with vig) minus de-vigged market probability for one side


@dataclass(frozen=True)
class Graded:
    date: str
    league: str
    market: str            # SPREAD, TOTAL, MONEYLINE or UNKNOWN
    pick: str
    probability: float     # model probability for the picked side
    market_probability: float | None   # de-vigged market probability for the picked side
    decimal_odds: float | None
    result: str            # WIN, LOSS or PUSH

    @property
    def outcome(self):
        return 1.0 if self.result == 'WIN' else 0.0

    @property
    def units(self):
        if self.decimal_odds is None or self.result == 'PUSH':
            return 0.0 if self.result == 'PUSH' else None
        return self.decimal_odds - 1.0 if self.result == 'WIN' else -1.0


def _family(market_type):
    return {'spread': 'SPREAD', 'total_over': 'TOTAL', 'total_under': 'TOTAL', 'moneyline': 'MONEYLINE'}.get(
        market_type, 'UNKNOWN')


def from_picks(picks):
    """Graded rows from ``scripts.backtest_low_line_over.Pick`` records. Ungraded rows are skipped."""
    rows = []
    for pick in picks:
        if pick.result not in {'WIN', 'LOSS', 'PUSH'}:
            continue
        p = pick.eff_win_prob if pick.eff_win_prob is not None else pick.win_prob
        if p is None or not 0.0 < p < 1.0:
            continue
        # Recorded EV and edge are used only when they are internally consistent.
        # Some early Drive renderings mix percent and fraction units; those rows
        # keep their probability and outcome but get no price or market benchmark.
        decimal = (pick.ev + 1.0) / p if pick.ev is not None and abs(pick.ev) <= MAX_ABS_EV else None
        if decimal is not None and not MIN_DECIMAL <= decimal <= MAX_DECIMAL:
            decimal = None
        market = p - pick.edge if pick.edge is not None and decimal is not None else None
        if market is not None and not (0.0 < market < 1.0 and 0.0 <= 1.0 / decimal - market <= MAX_MARGIN):
            market = None
        rows.append(Graded(pick.date, (pick.league or 'UNKNOWN').upper(), _family(pick.market_type),
                           pick.best_pick, p, market, decimal, pick.result))
    return rows


def load(directory):
    """Graded rows from a folder of exports (.csv / .xlsx / Drive .txt renderings)."""
    from scripts.backtest_low_line_over import load_dir
    return from_picks(load_dir(Path(directory)))


# -- metrics -------------------------------------------------------------------

def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def log_loss(ps, ys):
    return -sum(y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS)) for p, y in zip(ps, ys)) / len(ps)


def wilson(wins, n, z=1.96):
    if n == 0:
        return (None, None)
    phat = wins / n
    centre = phat + z * z / (2 * n)
    spread = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return ((centre - spread) / (1 + z * z / n), (centre + spread) / (1 + z * z / n))


def _bootstrap_mean(values, rng):
    """95% percentile interval of the mean of ``values``."""
    n = len(values)
    draws = sorted(sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP))
    return (draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1])


def _row_log_loss(p, y):
    return -(y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS)))


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def metrics(rows):
    """All scoreboard metrics for one group of graded rows."""
    rng = random.Random(SEED)
    scored = [r for r in rows if r.result in {'WIN', 'LOSS'}]
    out = dict(n=len(scored), pushes=sum(r.result == 'PUSH' for r in rows),
               clv='UNAVAILABLE: closing prices not captured (roadmap E1.4)')
    if not scored:
        return out
    ps, ys = [r.probability for r in scored], [r.outcome for r in scored]
    wins = int(sum(ys))
    low, high = wilson(wins, len(scored))
    out.update(win_rate=_round(wins / len(scored)), win_rate_ci=(_round(low), _round(high)),
               mean_probability=_round(sum(ps) / len(ps)),
               model_brier=_round(brier(ps, ys)), model_log_loss=_round(log_loss(ps, ys)),
               coin_flip_log_loss=_round(math.log(2)))
    paired = [r for r in scored if r.market_probability is not None]
    out['market_coverage'] = len(paired)
    if paired:
        mp = [r.market_probability for r in paired]
        my = [r.outcome for r in paired]
        out.update(market_brier=_round(brier(mp, my)), market_log_loss=_round(log_loss(mp, my)))
        # Log loss is a mean, so the skill difference is the mean of per-row differences.
        gains = [_row_log_loss(r.market_probability, r.outcome) - _row_log_loss(r.probability, r.outcome) for r in paired]
        out['skill_vs_market'] = _round(sum(gains) / len(gains))
        out['skill_vs_market_ci'] = tuple(_round(v) for v in _bootstrap_mean(gains, rng))
    priced = [r.units for r in rows if r.units is not None]
    out['priced'] = len(priced)
    if priced:
        out['roi_per_unit'] = _round(sum(priced) / len(priced))
        out['roi_ci'] = tuple(_round(v) for v in _bootstrap_mean(priced, rng))
    bins = []
    for lo, hi in zip(BINS, BINS[1:]):
        members = [r for r in scored if lo <= r.probability < hi]
        if members:
            bins.append(dict(range=f'{lo:.2f}-{min(hi, 1.0):.2f}', n=len(members),
                             predicted=_round(sum(r.probability for r in members) / len(members)),
                             realized=_round(sum(r.outcome for r in members) / len(members))))
    out['calibration'] = bins
    return out


def scoreboard(rows):
    """Metrics overall, by league, by league × market and by month."""
    def group(key):
        groups = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row)
        return {k: metrics(v) for k, v in sorted(groups.items())}
    dates = sorted(r.date for r in rows)
    return dict(
        coverage=dict(rows=len(rows), first_date=dates[0] if dates else None, last_date=dates[-1] if dates else None),
        overall=metrics(rows),
        by_league=group(lambda r: r.league),
        by_league_market=group(lambda r: f'{r.league} {r.market}'),
        by_month=group(lambda r: r.date[:7]),
    )


def verdict(m, minimum=150):
    """A plain-language reading of one group's metrics."""
    if m.get('n', 0) < minimum:
        return f'TOO FEW GRADED PICKS ({m.get("n", 0)} < {minimum})'
    ci = m.get('skill_vs_market_ci')
    if not ci:
        return 'NO MARKET BENCHMARK'
    if ci[0] > 0:
        return 'BEATS MARKET (log loss; CI above 0)'
    if ci[1] < 0:
        return 'WORSE THAN MARKET (log loss; CI below 0)'
    return 'NO DEMONSTRATED EDGE (CI includes 0)'


def markdown(board, minimum=150):
    def fmt(v):
        if isinstance(v, tuple):
            return '[' + ', '.join('—' if x is None else f'{x:+.3f}' if abs(x) < 1 else f'{x:.3f}' for x in v) + ']'
        return '—' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)
    head = ('| Group | n | Win rate [95% CI] | Model log loss | Market log loss | Skill vs market [95% CI] '
            '| ROI/unit [95% CI] | Verdict |\n|---|---|---|---|---|---|---|---|')

    def line(name, m):
        return (f"| {name} | {m.get('n', 0)} | {fmt(m.get('win_rate'))} {fmt(m.get('win_rate_ci'))} "
                f"| {fmt(m.get('model_log_loss'))} | {fmt(m.get('market_log_loss'))} "
                f"| {fmt(m.get('skill_vs_market'))} {fmt(m.get('skill_vs_market_ci'))} "
                f"| {fmt(m.get('roi_per_unit'))} {fmt(m.get('roi_ci'))} | {verdict(m, minimum)} |")
    cov = board['coverage']
    parts = [f"Graded slates {cov['first_date']} to {cov['last_date']}, {cov['rows']} graded rows. "
             f"A coin flip has log loss {math.log(2):.3f}. CLV: {board['overall']['clv']}.", '',
             '### Overall', head, line('All', board['overall']), '', '### By league', head]
    parts += [line(k, v) for k, v in board['by_league'].items()]
    parts += ['', '### By league and market', head] + [line(k, v) for k, v in board['by_league_market'].items()]
    parts += ['', '### By month', head] + [line(k, v) for k, v in board['by_month'].items()]
    parts += ['', '### Calibration (overall)', '| Predicted range | n | Mean predicted | Realized |', '|---|---|---|---|']
    parts += [f"| {b['range']} | {b['n']} | {b['predicted']:.3f} | {b['realized']:.3f} |"
              for b in board['overall'].get('calibration', [])]
    return '\n'.join(parts) + '\n'
