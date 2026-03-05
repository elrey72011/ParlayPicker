"""Portfolio optimization utilities for bet allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from parlaypicker.core.bankroll_manager import kelly_fraction
from parlaypicker.core.ev_engine import expected_value as ev_expected_value
from parlaypicker.core.portfolio_engine.correlation_matrix import build_correlation_matrix
from parlaypicker.core.portfolio_engine.risk_model import bet_variance


@dataclass(frozen=True)
class Bet:
    game_id: str
    probability: float
    odds: float
    ev: float


def payout_from_odds(odds: float) -> float:
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def expected_return(prob: float, odds: float) -> float:
    payout = payout_from_odds(odds)
    return prob * payout - (1 - prob)


def _portfolio_variance(weights: np.ndarray, bets: Sequence[Bet]) -> float:
    variances = np.array([bet_variance(bet.probability, bet.odds) for bet in bets], dtype=float)
    std = np.sqrt(np.maximum(variances, 1e-12))
    corr = build_correlation_matrix(bets)
    covariance = np.outer(std, std) * corr
    return float(weights.T @ covariance @ weights)


def optimize_portfolio(
    bets: Sequence[Bet],
    bankroll: float,
    max_fraction: float = 0.05,
    risk_aversion: float = 0.30,
) -> np.ndarray:
    n = len(bets)
    if n == 0:
        return np.array([], dtype=float)

    returns = np.array([bet.ev for bet in bets], dtype=float)
    raw_weights = np.maximum(returns, 0)
    if raw_weights.sum() == 0:
        return np.zeros(n, dtype=float)

    weights = raw_weights / raw_weights.sum()
    variances = np.array([bet_variance(bet.probability, bet.odds) for bet in bets], dtype=float)
    risk_penalty = 1.0 / (1.0 + risk_aversion * variances)
    adjusted_weights = weights * risk_penalty
    if adjusted_weights.sum() == 0:
        return np.zeros(n, dtype=float)

    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    allocations = adjusted_weights * bankroll

    kelly_caps = np.array(
        [
            bankroll * min(max_fraction, kelly_fraction(bet.probability, bet.odds))
            for bet in bets
        ],
        dtype=float,
    )
    allocations = np.minimum(allocations, kelly_caps)

    residual = bankroll - allocations.sum()
    if residual > 1e-9:
        positive = np.where(raw_weights > 0)[0]
        if len(positive):
            residual_split = residual / len(positive)
            for idx in positive:
                free_capacity = max(kelly_caps[idx] - allocations[idx], 0.0)
                add = min(residual_split, free_capacity)
                allocations[idx] += add

    return allocations


def portfolio_metrics(bets: Sequence[Bet], allocations: np.ndarray, bankroll: float) -> dict[str, float]:
    if len(bets) == 0 or bankroll <= 0:
        return {
            "expected_portfolio_return": 0.0,
            "portfolio_variance": 0.0,
            "portfolio_risk": 0.0,
            "sharpe_ratio": 0.0,
            "total_bankroll_exposure": 0.0,
            "max_drawdown_estimate": 0.0,
        }

    fractions = allocations / bankroll
    expected_returns = np.array(
        [ev_expected_value(bet.probability, bet.odds) for bet in bets],
        dtype=float,
    )
    expected_portfolio_return = float(fractions @ expected_returns)

    variance = _portfolio_variance(fractions, bets)
    risk = float(np.sqrt(max(variance, 0.0)))
    sharpe = expected_portfolio_return / risk if risk > 0 else 0.0

    losses = np.array([(1.0 / bankroll) * allocation for allocation in allocations], dtype=float)
    max_drawdown_estimate = float(np.clip(losses.sum(), 0.0, 1.0))

    return {
        "expected_portfolio_return": expected_portfolio_return,
        "portfolio_variance": variance,
        "portfolio_risk": risk,
        "sharpe_ratio": sharpe,
        "total_bankroll_exposure": float(allocations.sum() / bankroll),
        "max_drawdown_estimate": max_drawdown_estimate,
    }


def simulate_portfolio(
    bets: Sequence[Bet],
    allocations: np.ndarray,
    bankroll: float,
    seasons: int = 10000,
    seed: int = 7,
) -> dict[str, float]:
    if len(bets) == 0 or bankroll <= 0:
        return {
            "mean_return": 0.0,
            "median_return": 0.0,
            "p05_return": 0.0,
            "p95_return": 0.0,
            "probability_of_profit": 0.0,
        }

    rng = np.random.default_rng(seed)
    profits = np.zeros(seasons, dtype=float)

    for i in range(seasons):
        season_profit = 0.0
        for bet, allocation in zip(bets, allocations):
            if allocation <= 0:
                continue
            payout = payout_from_odds(bet.odds)
            win = rng.random() < bet.probability
            season_profit += allocation * payout if win else -allocation
        profits[i] = season_profit / bankroll

    return {
        "mean_return": float(np.mean(profits)),
        "median_return": float(np.median(profits)),
        "p05_return": float(np.quantile(profits, 0.05)),
        "p95_return": float(np.quantile(profits, 0.95)),
        "probability_of_profit": float(np.mean(profits > 0)),
    }
