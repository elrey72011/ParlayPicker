"""Expected value and signal utilities."""


def expected_value(prob, odds):
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)

    return prob * payout - (1 - prob)


def bet_signal(row, min_edge=0.03):
    return row["expected_value"] > min_edge
