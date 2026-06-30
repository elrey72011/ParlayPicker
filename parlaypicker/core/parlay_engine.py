"""Parlay optimization helpers."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable
import pandas as pd


def parlay_probability(probabilities: Iterable[float]) -> float:
    result = 1.0
    for p in probabilities:
        result *= p
    return result


def _american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def _decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    return -100 / (decimal_odds - 1)


def build_parlays(df: pd.DataFrame, leg_count: int = 2) -> pd.DataFrame:
    rows = []
    for combo in combinations(df.to_dict("records"), leg_count):
        probs = [r["true_probability"] for r in combo]
        decimal_odds = [_american_to_decimal(r["odds"]) for r in combo]
        combined_decimal = 1.0
        for d in decimal_odds:
            combined_decimal *= d
        rows.append({
            "game_id": "+".join(str(r["game_id"]) for r in combo),
            "bet_type": "parlay",
            "odds": _decimal_to_american(combined_decimal),
            "true_probability": parlay_probability(probs),
        })
    return pd.DataFrame(rows)
