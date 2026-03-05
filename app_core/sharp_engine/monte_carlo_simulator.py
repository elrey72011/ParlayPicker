"""Monte Carlo bankroll risk simulation."""

import numpy as np


def simulate_bankroll(probabilities, odds, bets=10000):
    bankroll = 1000

    for p, o in zip(probabilities, odds):
        if np.random.rand() < p:
            if o > 0:
                bankroll += o / 100
            else:
                bankroll += 100 / abs(o)
        else:
            bankroll -= 1

    return bankroll
