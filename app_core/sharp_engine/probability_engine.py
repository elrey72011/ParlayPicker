"""Probability engines for market baselines and multi-signal ensembles."""

from typing import Any, Mapping


def american_to_prob(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    total = home_prob + away_prob
    if total == 0:
        return home_prob, away_prob
    return home_prob / total, away_prob / total


def ensemble_probability(row: Mapping[str, Any]) -> float:
    weights = {
        "market": 0.45,
        "ml": 0.30,
        "kalshi": 0.15,
        "theover": 0.07,
        "sentiment": 0.03,
    }

    prob = (
        weights["market"] * float(row.get("market_prob", 0) or 0)
        + weights["ml"] * float(row.get("ml_prob", 0) or 0)
        + weights["kalshi"] * float(row.get("kalshi_prob", 0) or 0)
        + weights["theover"] * float(row.get("theover_prob", 0) or 0)
        + weights["sentiment"] * float(row.get("sentiment_prob", 0) or 0)
    )

    return max(min(prob, 0.99), 0.01)
