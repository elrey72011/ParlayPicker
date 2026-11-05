"""Which sport × market the EV shadow may mark BET: the repository's own authority chain.

A market is qualified only when the canonical prospective-evidence store reports a
validated deployment state for it (``prospective_evidence.deployment_state``):

1. its frozen validation plan (``app_core/prospective_validation_plans.py``) is
   bound to an exact model and calibration;
2. both validation and holdout pass the plan's frozen thresholds (≥ 200 independent
   events, Brier ≤ 0.24, log loss ≤ 0.68, calibration error ≤ 0.08, ≥ 90% coverage,
   ≥ 80% certified closes, paper ROI ≥ 0);
3. an independent deployment review recorded the result, and it still re-evaluates
   to the same report.

Even then it grants no stake: funding still needs the owner's market activation and
a current exposure ledger (``core/sport_market_gate.py``), and the shadow never
stakes anything. ``market_readiness`` supplies the evidence-derived next blocker
for every market that is not qualified.
"""
from __future__ import annotations

import sqlite3

from app_core import prospective_evidence
from core.market_policy import SPORT_MARKET_FAMILIES

VALIDATED = frozenset({'PROVISIONAL_VALIDATED', 'STANDARD_VALIDATED', 'PREMIUM_VALIDATED'})


def assess(path=None, *, readiness=prospective_evidence.market_readiness):
    """{'NFL SPREAD': {'qualified', 'deployment_state', 'next_blocker'}, ...} for every sport × market."""
    out = {}
    for sport, families in SPORT_MARKET_FAMILIES.items():
        for family in families:
            try:
                row = readiness(path, sport, family)
                state, blocker = row.get('deployment_state', 'UNVALIDATED'), row.get('next_blocker')
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                state, blocker = 'UNVALIDATED', 'EVIDENCE_UNREADABLE:' + type(exc).__name__
            qualified = state in VALIDATED
            out[f'{sport} {family}'] = dict(qualified=qualified, deployment_state=state,
                                            next_blocker=None if qualified else blocker or 'NOT_VALIDATED')
    return out


def qualified_set(assessment):
    """{(sport, family)} for the EV selector."""
    return frozenset(tuple(group.split(' ', 1)) for group, a in assessment.items() if a['qualified'])
