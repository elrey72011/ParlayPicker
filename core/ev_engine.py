def calculate_ev(probability, american_odds):
    if probability is None:
        probability = 0.5

    if american_odds > 0:
        payout = american_odds / 100
    else:
        payout = 100 / abs(american_odds)

    ev = probability * payout - (1 - probability)

    return ev
