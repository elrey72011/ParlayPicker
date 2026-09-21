"""Display existing estimates at the saved price, with no authorization side effects."""
from core.wager_decisions import decimal_price, finite

FIELDS = ("model_win_probability", "break_even_probability", "estimated_price_edge",
          "estimated_expected_value", "value_status")


def display(probability, odds, expected_value):
    p, ev, decimal = finite(probability), finite(expected_value), decimal_price(odds)
    if p is not None and not 0 <= p <= 1:
        p = None
    break_even = 1 / decimal if decimal is not None else None
    # EV is the existing producer estimate, never reconstructed from display p.
    status = "VALUE UNAVAILABLE"
    if decimal is not None and ev is not None:
        status = "NEGLIGIBLE ESTIMATED VALUE" if 0 < abs(ev) < .0005 else "POSITIVE ESTIMATED VALUE" if ev > 0 else "NEGATIVE ESTIMATED VALUE" if ev < 0 else "ZERO ESTIMATED VALUE"
    return dict(model_win_probability=p, break_even_probability=break_even,
                estimated_price_edge=p-break_even if p is not None and break_even is not None else None,
                estimated_expected_value=ev, value_status=status)
