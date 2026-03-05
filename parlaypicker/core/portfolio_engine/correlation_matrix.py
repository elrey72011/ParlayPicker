"""Correlation model between bets."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def build_correlation_matrix(bets: Sequence[object]) -> np.ndarray:
    n = len(bets)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if getattr(bets[i], "game_id", None) == getattr(bets[j], "game_id", None):
                matrix[i, j] = 0.8

    return matrix
