"""Synthetic price mechanics; these are not model-validation evidence."""

import math

import pytest

from core.price_value import price_value


def test_unconditional_push_mass_changes_ev_and_edge():
    priced = price_value(.525, .05, 1 + 100 / 110)
    assert priced["p_loss"] == pytest.approx(.425)
    assert priced["expected_value"] == pytest.approx(.525 * (1 + 100 / 110) + .05 - 1)
    assert priced["break_even"] == pytest.approx(.95 / (1 + 100 / 110))
    assert priced["edge"] == pytest.approx(.525 - priced["break_even"])


@pytest.mark.parametrize("win,push,decimal", [
    (.7, .4, 2), (float("nan"), 0, 2), (.5, float("inf"), 2),
    (.5, 0, 1), (-.1, 0, 2),
])
def test_invalid_mass_or_price_fails_closed(win, push, decimal):
    assert price_value(win, push, decimal) is None


def test_minimum_price_satisfies_positive_ev_and_required_edge():
    floor = price_value(.58, .05, 2, minimum_edge=.02)["minimum_decimal_price"]
    assert math.isfinite(floor)
    at_floor = price_value(.58, .05, floor, minimum_edge=.02)
    assert at_floor["expected_value"] > 0
    assert at_floor["edge"] >= .02 - 1e-12
    worse = price_value(.58, .05, floor - 1e-5, minimum_edge=.02)
    assert worse["edge"] < .02 or worse["expected_value"] <= 0
