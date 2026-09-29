"""Display existing estimates at the saved price, with no authorization side effects."""
import math

from core.wager_decisions import decimal_price, finite

FIELDS = ("model_win_probability", "break_even_probability", "estimated_price_edge",
          "estimated_expected_value", "value_status")


def display(probability, odds, expected_value, *, push_probability=None):
    p, ev, decimal = finite(probability), finite(expected_value), decimal_price(odds)
    push = finite(push_probability)
    if p is not None and not 0 <= p <= 1:
        p = None
    break_even = 1 / decimal if decimal is not None else None
    edge = p-break_even if p is not None and break_even is not None else None
    if push is not None:
        from core.price_value import price_value
        priced = price_value(p, push, decimal)
        if (priced is None or ev is None or
                not math.isclose(ev, priced["expected_value"], rel_tol=0.0, abs_tol=1e-9)):
            break_even = edge = None
        else:
            break_even, edge = priced["break_even"], priced["edge"]
    # EV is the existing producer estimate, never reconstructed from display p.
    status = "VALUE UNAVAILABLE"
    if decimal is not None and ev is not None:
        status = "NEGLIGIBLE ESTIMATED VALUE" if 0 < abs(ev) < .0005 else "POSITIVE ESTIMATED VALUE" if ev > 0 else "NEGATIVE ESTIMATED VALUE" if ev < 0 else "ZERO ESTIMATED VALUE"
    return dict(model_win_probability=p, break_even_probability=break_even,
                estimated_price_edge=edge,
                estimated_expected_value=ev, value_status=status)
