"""Canonical expected value calculations."""
from __future__ import annotations


def expected_value(prob: float, odds: float) -> float:
    payout = odds / 100 if odds > 0 else 100 / abs(odds)
    return prob * payout - (1 - prob)


def edge_percentage(true_probability: float, implied_probability: float) -> float:
    return true_probability - implied_probability


def bet_signal(expected_value_value: float, min_edge: float = 0.03) -> bool:
    return expected_value_value > min_edge
