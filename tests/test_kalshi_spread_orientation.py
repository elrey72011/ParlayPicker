"""Kalshi run-line orientation (19 Jun).

A "S wins by over L" contract prices ONLY S -L (YES) and Opp(S) +L (NO). Deriving any
other side via 1 - raw is wrong because the 1-run margin band sits between the two -L
sides. This inflated Pittsburgh -1.5 to 1 - P(Colorado by 2+) = 0.705 (really P(PIT +1.5)).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import orient_spread_kalshi_prob


# raw_prob = P(subject wins by > L)
RAW = 0.295  # P(Colorado by 2+) from the 18:35 export


def test_home_favorite_subject_uses_raw():
    # "Houston wins by >1.5", pick Houston -1.5 (home favorite, subject) -> raw.
    p = orient_spread_kalshi_prob(0.355, subject_is_home=True, pick_is_home=True, pick_line=-1.5)
    assert p == 0.355


def test_away_underdog_opponent_uses_complement():
    # "Colorado wins by >1.5", pick Pittsburgh +1.5 (away underdog, opponent) -> 1 - raw.
    p = orient_spread_kalshi_prob(RAW, subject_is_home=True, pick_is_home=False, pick_line=1.5)
    assert abs(p - (1 - RAW)) < 1e-9


def test_away_favorite_off_home_market_is_unpriceable():
    # The Pittsburgh -1.5 bug: away favorite priced off the HOME team's contract. The old
    # code returned 1 - 0.295 = 0.705 (= P(PIT +1.5)); now it's a miss.
    p = orient_spread_kalshi_prob(RAW, subject_is_home=True, pick_is_home=False, pick_line=-1.5)
    assert p is None


def test_home_underdog_off_home_market_is_unpriceable():
    p = orient_spread_kalshi_prob(0.40, subject_is_home=True, pick_is_home=True, pick_line=1.5)
    assert p is None


def test_away_favorite_on_its_own_market_is_priced():
    # When Kalshi DOES carry the away team's own market ("Pittsburgh wins by >1.5"),
    # subject == pick team, so the away favorite is priced directly (no miss).
    p = orient_spread_kalshi_prob(0.40, subject_is_home=False, pick_is_home=False, pick_line=-1.5)
    assert p == 0.40


def test_missing_line_is_unpriceable():
    assert orient_spread_kalshi_prob(0.40, subject_is_home=True, pick_is_home=True, pick_line=pd.NA) is None
