from __future__ import annotations

import numpy as np

from core.models.prediction_models import Prediction


def calculate_ev(prob, odds):
    prob_arr = np.asarray(prob, dtype=float)
    odds_arr = np.asarray(odds, dtype=float)

    payout = np.where(odds_arr > 0, odds_arr / 100.0, 100.0 / np.abs(odds_arr))
    ev = prob_arr * payout - (1 - prob_arr)

    if np.isscalar(prob) and np.isscalar(odds):
        return float(ev)
    return ev


def compute_ev(prediction: Prediction) -> float:
    """Typed EV helper that avoids fragile DataFrame column access."""
    # Use max() so highly confident ML Arbitrage edges aren't dragged below breakeven by Kalshi's baseline
    prob = max(prediction.ai_probability, prediction.ml_probability)
    return float(calculate_ev(prob, -110))
