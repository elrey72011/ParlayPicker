"""Moneyline parlay-leg eligibility (pure -- no network)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.moneyline_parlay import (
    american_to_implied,
    moneyline_leg_eligible,
)


def test_implied_conversion():
    assert abs(american_to_implied(-200) - (200 / 300)) < 1e-9
    assert abs(american_to_implied(150) - (100 / 250)) < 1e-9


def test_value_dog_is_eligible():
    # +150 implies 40%; model says 48% -> +8% edge, positive EV, inside range.
    r = moneyline_leg_eligible(0.48, 150)
    assert r["eligible"] is True
    assert r["edge"] > 0


def test_no_edge_leg_rejected():
    # -150 implies 60%; model says 60% -> no edge.
    r = moneyline_leg_eligible(0.60, -150)
    assert r["eligible"] is False
    assert "edge" in r["reason"]


def test_heavy_favorite_capped_out():
    # -300 is outside the default [-250, 250] range even with a model edge.
    r = moneyline_leg_eligible(0.82, -300)
    assert r["eligible"] is False
    assert "outside" in r["reason"]


def test_longshot_capped_out():
    r = moneyline_leg_eligible(0.40, 320)
    assert r["eligible"] is False
    assert "outside" in r["reason"]


def test_positive_edge_but_negative_ev_rejected():
    # Construct an edge that still can't overcome the price's EV math is hard for ML
    # (edge>0 implies EV>0 at the same price), so verify the missing-data guard instead.
    r = moneyline_leg_eligible(None, -120)
    assert r["eligible"] is False
    assert r["edge"] is None


def test_custom_thresholds_override():
    # Tighten the edge bar so a small-edge dog no longer qualifies.
    r = moneyline_leg_eligible(0.42, 150, min_edge=0.05)
    assert r["eligible"] is False
