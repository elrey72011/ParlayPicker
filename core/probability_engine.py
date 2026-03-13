from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def american_to_prob(odds):
    if pd.isna(odds):
        return pd.NA
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
    """Normalize probability inputs and calculate weighted consensus probability."""
    for col in ["market_prob", "ml_prob", "ai_prob"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ai_fallback = df["ai_prob"].fillna(df["market_prob"])
    df["consensus_prob"] = (
        df["market_prob"] * 0.4
        + df["ml_prob"] * 0.4
        + ai_fallback * 0.2
    ).clip(lower=0.0, upper=1.0)
    return df


# Backward compatible aliases
american_odds_to_probability = american_to_prob
american_odds_to_prob = american_to_prob
