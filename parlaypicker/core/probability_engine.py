"""Canonical probability calculations."""
from __future__ import annotations

from typing import Mapping, Any
import numpy as np

from parlaypicker.utils.config import MODEL_WEIGHTS


def american_to_prob(odds: float | np.ndarray) -> float | np.ndarray:
    odds_arr = np.asarray(odds, dtype=float)
    probs = np.empty_like(odds_arr, dtype=float)

    zero_mask = odds_arr == 0
    pos_mask = odds_arr > 0
    neg_mask = ~(zero_mask | pos_mask)

    probs[zero_mask] = 0.5
    probs[pos_mask] = 100.0 / (odds_arr[pos_mask] + 100.0)
    probs[neg_mask] = np.abs(odds_arr[neg_mask]) / (np.abs(odds_arr[neg_mask]) + 100.0)

    return float(probs) if np.isscalar(odds) else probs


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    total = home_prob + away_prob
    if total <= 0:
        return home_prob, away_prob
    return home_prob / total, away_prob / total


def ensemble_probability(row: Mapping[str, Any]) -> float:
    prob = sum(float(row.get(f"{k}_prob", 0) or 0) * w for k, w in MODEL_WEIGHTS.items())
    return max(min(prob, 0.99), 0.01)


def calibrated_probability(raw_prob: float, slope: float = 1.0, intercept: float = 0.0) -> float:
    calibrated = slope * raw_prob + intercept
    return max(min(calibrated, 0.99), 0.01)
