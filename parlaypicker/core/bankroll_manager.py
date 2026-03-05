"""Bankroll management utilities."""
from __future__ import annotations

from parlaypicker.utils.config import KELLY_MULTIPLIER


def kelly_fraction(prob: float, odds: float, multiplier: float = KELLY_MULTIPLIER) -> float:
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    q = 1 - prob
    raw = max((b * prob - q) / b, 0)
    return raw * multiplier
