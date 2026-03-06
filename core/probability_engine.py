from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(home_prob, away_prob=None):
    if away_prob is None and isinstance(home_prob, Iterable):
        probs = list(home_prob)
        if len(probs) != 2:
            raise ValueError("remove_vig expects exactly two probabilities")
        home_prob, away_prob = probs

    total = float(home_prob) + float(away_prob)
    if total == 0:
        return 0.5, 0.5

    return float(home_prob) / total, float(away_prob) / total


def normalize_probability_components(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize probability inputs and calculate weighted consensus with optional sources."""
    weight_map = {
        "market_prob": 0.4,
        "ml_prob": 0.3,
        "ai_prob": 0.2,
        "kalshi_prob": 0.1,
    }

    for col in weight_map:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def _row_consensus(row):
        available = {k: v for k, v in weight_map.items() if pd.notna(row[k])}
        if not available:
            return 0.5

        total_weight = sum(available.values())
        return sum((available[k] / total_weight) * float(row[k]) for k in available)

    df["consensus_prob"] = df.apply(_row_consensus, axis=1).clip(lower=0.0, upper=1.0)
    return df


# Backward compatible aliases
american_odds_to_probability = american_to_prob
american_odds_to_prob = american_to_prob
