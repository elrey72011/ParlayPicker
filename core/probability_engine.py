def american_odds_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


# Backward compatible alias
american_odds_to_prob = american_odds_to_probability
