"""Pitcher-strikeout prop model (v1 parametric).

P(Ks over line) from a Poisson/NB on the projected mean, no trained-model cold-start.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.prop_model import (
    strikeout_over_under_probs,
    strikeout_pick_probability,
    project_expected_strikeouts,
)


def test_halfline_over_under_sum_to_one_no_push():
    p_over, p_under, p_push = strikeout_over_under_probs(6.0, 6.5)
    assert p_push == 0.0
    assert abs((p_over + p_under) - 1.0) < 1e-9


def test_line_above_mean_favors_under():
    # Mean 6.0, line 6.5 -> under should be favored.
    p_over, p_under, _ = strikeout_over_under_probs(6.0, 6.5)
    assert p_under > p_over
    # Poisson(6): P(X>=7) = 1 - cdf(6) ~= 0.394
    assert abs(p_over - 0.394) < 0.01


def test_line_below_mean_favors_over():
    p_over, p_under, _ = strikeout_over_under_probs(7.5, 6.5)
    assert p_over > p_under


def test_integer_line_has_push_mass():
    p_over, p_under, p_push = strikeout_over_under_probs(7.0, 7.0)
    assert p_push > 0.0
    assert abs((p_over + p_under + p_push) - 1.0) < 1e-9


def test_overdispersion_pushes_mass_to_tails():
    # A higher dispersion (NB) fattens the tails: for a line above the mean, the over
    # probability rises vs Poisson (more chance of a big strikeout game).
    line, mu = 9.5, 7.0
    pois_over = strikeout_over_under_probs(mu, line, dispersion=1.0)[0]
    nb_over = strikeout_over_under_probs(mu, line, dispersion=1.5)[0]
    assert nb_over > pois_over


def test_pick_probability_is_depushed():
    # On an integer line, the de-pushed pick prob removes the push mass.
    p = strikeout_pick_probability(7.0, 7.0, "over")
    p_over, p_under, _ = strikeout_over_under_probs(7.0, 7.0)
    assert abs(p - p_over / (p_over + p_under)) < 1e-9


def test_pick_probability_side_validation():
    with pytest.raises(ValueError):
        strikeout_pick_probability(6.0, 6.5, "yes")


def test_projection_opponent_adjustment():
    # A high-strikeout opponent raises the projection vs a neutral one.
    base = project_expected_strikeouts(9.0, 6.0)  # 6.0 K/game neutral
    assert abs(base - 6.0) < 1e-9
    high_k_opp = project_expected_strikeouts(9.0, 6.0, opponent_k_rate=0.27, league_k_rate=0.22)
    assert high_k_opp > base
    low_k_opp = project_expected_strikeouts(9.0, 6.0, opponent_k_rate=0.17, league_k_rate=0.22)
    assert low_k_opp < base


def test_projection_opponent_adjustment_is_clamped():
    # Extreme opponent rates can't move the projection more than the clamp allows.
    huge = project_expected_strikeouts(9.0, 6.0, opponent_k_rate=0.50, league_k_rate=0.22)
    assert huge <= 6.0 * 1.25 + 1e-9
