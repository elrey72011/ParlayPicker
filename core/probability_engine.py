from __future__ import annotations


def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(probabilities):
    total = sum(probabilities)
    if total == 0:
        return [0 for _ in probabilities]
    return [p / total for p in probabilities]


# Backward compatible aliases
american_odds_to_probability = american_to_prob
american_odds_to_prob = american_to_prob
