"""Staking terms frozen with a plan: the policy the owner activates and the stake-tier rules.

``prospective_evidence.deployment_state`` reads a validated market's ``deployment_criteria`` from
its **frozen plan**, and the live chain needs two things from it:

* ``validated_policy``: the exact ``core.sport_policy.SportPolicy`` the owner may activate
  (``core.sport_market_activation._policy_matches_frozen_plan`` compares digests);
* ``maturity_rules.PROVISIONAL``: ``max_uncertainty`` and ``min_prior_clv`` for a pick to reach
  the provisional stake tier (``core.candidate_maturity.assign``).

Without them a validated plan can never stake. Terms are named and versioned here, the Bound
plan draft writes the named set into the plan, and the frozen plan keeps its own copy (so a
later edit here never changes a frozen plan). Decided 2026-11-08 by owner delegation;
``docs/roadmap/GO-LIVE-STAKING-TERMS-2026-11-08.md``.
"""
from __future__ import annotations

from dataclasses import asdict

PINNACLE_PROVISIONAL_V1 = "pinnacle-provisional-v1"


def _policy(sport: str, name: str) -> dict:
    from core.sport_policy import SportPolicy
    policy = asdict(SportPolicy(
        sport=sport, version=f"{sport.lower()}-{name}", deployment_state="PROVISIONAL_VALIDATED",
        provisional_allowed=True,
        # Evidence behind a pick: the validation + holdout graded games (about 2 x 200 games;
        # 300 leaves room for pushes and voids that are not scored).
        provisional_minimum_evidence=300.0, minimum_evidence=600,
        kelly_fraction=0.10,                 # a tenth of Kelly while the edge is new
        provisional_stake_cap=0.005,         # at most 0.5% of bankroll on one pick
        standard_stake_cap=0.0, premium_stake_cap=0.0,
        min_conservative_edge=0.0,           # the edge must stay positive after the uncertainty haircut
        uncertainty_quantile=0.25,           # cautious probability: 25% quantile of the validated evidence
        sport_exposure_cap=0.03))            # at most 3% of bankroll open on this sport
    policy.pop("validation_id")              # supplied by the reviewed validation artifact
    return policy


def terms(name: str, sport: str) -> dict:
    """The deployment_criteria additions for ``name`` and ``sport``."""
    if name != PINNACLE_PROVISIONAL_V1:
        raise ValueError(f"unknown staking terms {name!r}")
    return {
        "staking_terms": name,
        "validated_policy": _policy(sport, "pinnacle-provisional-v1"),
        "maturity_rules": {
            # max_uncertainty: mean minus cautious (25% quantile) probability, on the validated
            # games; about 0.017 at 400 games. min_prior_clv: holdout bets' CLV lower 95% bound.
            "PROVISIONAL": {"max_uncertainty": 0.03, "min_prior_clv": 0.0},
        },
        "prior_clv_metric": "HOLDOUT_BET_CLV_LOWER_95",
        "effective_evidence": "VALIDATION_PLUS_HOLDOUT_SCORED_GAMES",
        "cautious_probability": "UNCERTAINTY_QUANTILE_OF_VALIDATED_EVIDENCE",
        # Drift brake: no stake while the market's last 50 settled live bets have mean CLV < 0.
        "regime_brake": {"window_bets": 50, "min_mean_clv": 0.0},
    }
