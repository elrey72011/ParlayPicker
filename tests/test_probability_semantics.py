import pytest
from core.probability_semantics import conditional_probabilities


@pytest.mark.parametrize("push,market_push", [(None,.1),(.1,None),(1,.1),(-.1,.1),(.5,.1),(float("nan"),.1)])
def test_invalid_push_mass_cannot_be_scored(push, market_push):
    assert conditional_probabilities({"probability_semantics":"win_unconditional_with_push",
        "calibrated_probability":.6, "market_probability":.5,
        "push_probability":push, "market_push_probability":market_push}) is None
