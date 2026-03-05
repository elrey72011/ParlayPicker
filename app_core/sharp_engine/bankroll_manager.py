"""Bankroll sizing methods."""


def kelly_fraction(prob, odds):
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)

    q = 1 - prob

    return max((b * prob - q) / b, 0)
