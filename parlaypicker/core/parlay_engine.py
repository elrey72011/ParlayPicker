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


def build_parlays(df: pd.DataFrame, leg_count: int = 2) -> pd.DataFrame:
    rows = []
    for combo in combinations(df.to_dict("records"), leg_count):
        probs = [r["true_probability"] for r in combo]
        odds = [r["odds"] for r in combo]
        rows.append({
            "game_id": "+".join(str(r["game_id"]) for r in combo),
            "bet_type": "parlay",
            "odds": sum(odds),
            "true_probability": parlay_probability(probs),
        })
    return pd.DataFrame(rows)
