"""Risk model for single-bet variance."""
from __future__ import annotations


def _payout_from_odds(odds: float) -> float:
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def bet_variance(prob: float, odds: float) -> float:
    payout = _payout_from_odds(odds)

    win_return = payout
    loss_return = -1.0

    mean = prob * win_return + (1 - prob) * loss_return
    variance = prob * (win_return - mean) ** 2 + (1 - prob) * (loss_return - mean) ** 2
    return variance
