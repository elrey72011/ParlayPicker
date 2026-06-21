"""Parametric player-prop models. v1: pitcher strikeouts.

A starting pitcher's strikeouts in a game are count data, modeled here as a Poisson (or
mildly overdispersed negative-binomial) variable with mean ``expected_ks`` -- the pitcher's
projected strikeouts (recent K/9 x expected innings, adjusted for opponent strikeout rate
and park). That yields P(strikeouts over the book's line) DIRECTLY, with no trained-model
cold-start: the edge comes from the projection disagreeing with the line, and the result is
calibratable against historical pitcher game logs.

Strikeouts are slightly overdispersed vs Poisson, so ``dispersion`` (variance/mean) > 1
switches to a negative binomial with the same mean. dispersion = 1.0 is exact Poisson.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _count_distribution(expected_ks: float, dispersion: float):
    """A scipy frozen distribution for strikeout counts with mean ``expected_ks``.

    dispersion <= 1.0 -> Poisson(mu). dispersion > 1.0 -> negative binomial with
    variance = dispersion * mean (overdispersed), matching the same mean.
    """
    mu = float(expected_ks)
    if mu <= 0:
        raise ValueError("expected_ks must be positive")
    if dispersion <= 1.0:
        return stats.poisson(mu)
    # NB with mean mu and variance mu*dispersion:
    #   var/mean = 1/p = dispersion -> p = 1/dispersion ; n = mu*p/(1-p) = mu/(dispersion-1)
    p = 1.0 / dispersion
    n = mu / (dispersion - 1.0)
    return stats.nbinom(n, p)


def strikeout_over_under_probs(
    expected_ks: float, line: float, dispersion: float = 1.0
) -> tuple[float, float, float]:
    """Return ``(p_over, p_under, p_push)`` for a strikeout total at ``line``.

    A half-line (e.g. 6.5) cannot push: over = P(X >= 7), under = P(X <= 6). An integer
    line L pushes on exactly L: over = P(X >= L+1), under = P(X <= L-1), push = P(X = L).
    """
    dist = _count_distribution(expected_ks, dispersion)
    line = float(line)
    floor_line = int(np.floor(line))
    is_integer_line = abs(line - round(line)) < 1e-9
    if is_integer_line:
        L = int(round(line))
        p_push = float(dist.pmf(L))
        p_under = float(dist.cdf(L - 1))
        p_over = float(dist.sf(L))  # P(X >= L+1)
    else:
        p_push = 0.0
        p_over = float(dist.sf(floor_line))  # P(X >= floor_line + 1)
        p_under = float(dist.cdf(floor_line))  # P(X <= floor_line)
    return p_over, p_under, p_push


def strikeout_pick_probability(
    expected_ks: float, line: float, side: str, dispersion: float = 1.0
) -> float:
    """P(the chosen ``side`` wins), de-pushed so over/under sum to 1.

    ``side`` is 'over' or 'under'. Push mass is removed (a push refunds), so the returned
    number is the win probability conditional on a decision -- directly comparable to a
    de-vigged market price.
    """
    p_over, p_under, _ = strikeout_over_under_probs(expected_ks, line, dispersion)
    decisive = p_over + p_under
    if decisive <= 0:
        return 0.5
    s = str(side).strip().lower()
    if s.startswith("o"):
        return p_over / decisive
    if s.startswith("u"):
        return p_under / decisive
    raise ValueError(f"side must be 'over' or 'under', got {side!r}")


def project_expected_strikeouts(
    k_per_9: float,
    expected_innings: float,
    opponent_k_rate: float | None = None,
    league_k_rate: float = 0.22,
    park_factor: float = 1.0,
) -> float:
    """Project a starter's mean strikeouts for the game.

    Base = (k_per_9 / 9) * expected_innings. Scaled by the opponent's strikeout-proneness
    (opponent_k_rate / league_k_rate, clamped) and a park factor. Opponent and park default
    to neutral when unknown, so the projection degrades to the pitcher's own rate.
    """
    base = (float(k_per_9) / 9.0) * float(expected_innings)
    opp_adj = 1.0
    if opponent_k_rate is not None and league_k_rate > 0:
        opp_adj = float(np.clip(opponent_k_rate / league_k_rate, 0.80, 1.25))
    return max(0.1, base * opp_adj * float(park_factor))
