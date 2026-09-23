"""Price and probability arithmetic shared by strict and controlled decisions.

Probabilities passed here are unconditional: a push is its own outcome, and
win + push + loss must equal one. The caller establishes the source semantics.
"""

from __future__ import annotations

import math


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def price_value(p_win, p_push, decimal, *, minimum_edge=0.0):
    """Return push-aware price metrics, or None for invalid probability mass."""
    win, push, price, edge_floor = (
        _finite(p_win), _finite(p_push), _finite(decimal), _finite(minimum_edge)
    )
    if (
        win is None or push is None or price is None or edge_floor is None
        or not 0 <= win <= 1 or not 0 <= push < 1 or win + push > 1 + 1e-10
        or price <= 1 or edge_floor < 0
    ):
        return None
    loss = max(0.0, 1.0 - win - push)
    break_even = (1.0 - push) / price
    ev = win * price + push - 1.0
    edge = win - break_even
    # The EV gate is strict. A zero-EV equality price is therefore excluded
    # even when the minimum probability edge itself is zero.
    min_price = (max((1.0 - push) / (win - edge_floor),
                     math.nextafter((1.0 - push) / win, math.inf))
                 if win > edge_floor and win > 0 else None)
    if min_price is not None and not math.isfinite(min_price):
        min_price = None
    if min_price is not None:
        while (win * min_price + push - 1.0 <= 0
               or win - (1.0 - push) / min_price < edge_floor):
            min_price = math.nextafter(min_price, math.inf)
    kelly = max(0.0, ev / ((price - 1.0) * (1.0 - push)))
    return dict(p_win=win, p_push=push, p_loss=loss, break_even=break_even,
                expected_value=ev, edge=edge, minimum_decimal_price=min_price,
                full_kelly=kelly)
