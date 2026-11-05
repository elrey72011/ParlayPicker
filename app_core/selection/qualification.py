"""Which league × market may produce a BET (owner decision D1, 2026-11-05).

A league × market is *qualified* only when **all** of these hold on graded,
prospective evidence:

1. at least ``MIN_GRADED`` (150) graded picks;
2. skill vs market (market log loss minus model log loss) whose 95% interval
   lies entirely above 0;
3. at least ``MIN_CLOSES`` (150) picks with a verified close, and a mean line
   CLV whose 95% interval lies entirely above 0.

Each failure is reported with a specific reason. Without closing-line evidence,
a market cannot qualify. Nothing here changes which picks are made: the result
only feeds the shadow EV selector until decision D7.
"""
from __future__ import annotations

MIN_GRADED = 150
MIN_CLOSES = 150


def assess(board, clv_report=None):
    """{'NFL SPREAD': {'qualified': bool, 'reasons': [...]}, ...} from scoreboard (+ CLV) reports."""
    clv_groups = (clv_report or {}).get('by_league_market', {})
    out = {}
    for group, m in (board or {}).get('by_league_market', {}).items():
        reasons = []
        if m.get('n', 0) < MIN_GRADED:
            reasons.append(f"GRADED_PICKS_BELOW_{MIN_GRADED}:{m.get('n', 0)}")
        ci = m.get('skill_vs_market_ci')
        if not ci:
            reasons.append('NO_MARKET_BENCHMARK')
        elif ci[0] is None or ci[0] <= 0:
            reasons.append('SKILL_VS_MARKET_NOT_ABOVE_ZERO')
        c = clv_groups.get(group)
        if not c:
            reasons.append('NO_CLOSING_LINE_EVIDENCE')
        else:
            if c.get('n', 0) < MIN_CLOSES:
                reasons.append(f"CLOSES_BELOW_{MIN_CLOSES}:{c.get('n', 0)}")
            clv_ci = c.get('mean_line_clv_ci')
            if not clv_ci or clv_ci[0] is None or clv_ci[0] <= 0:
                reasons.append('CLV_NOT_ABOVE_ZERO')
        out[group] = dict(qualified=not reasons, reasons=reasons)
    return out


def qualified_set(assessment):
    """{(league, family)} for the EV selector."""
    return frozenset(tuple(group.split(' ', 1)) for group, a in assessment.items() if a['qualified'])
