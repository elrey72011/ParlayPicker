"""MLB run-line cover conversion (19 Jun).

A run line pays on the margin, not the win. ml_probability is the moneyline P(win), which
overstates a -1.5 favorite's cover (must win by 2+) and understates a +1.5 dog's (covers
unless it loses by 2+). 19 Jun: Pittsburgh -1.5 carried P(win)=0.54 as its cover prob and
lost outright. The conversion shifts P(win) by the favorite's share of the 1-run band.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _apply_mlb_runline_cover

BAND = 0.135


def _run(model, league, mtype, pick):
    return _apply_mlb_runline_cover(
        pd.Series(model), pd.Series(league), pd.Series(mtype), pd.Series(pick), BAND
    )


def test_favorite_cover_is_shifted_down():
    # Pittsburgh -1.5 at P(win)=0.54 -> cover 0.54 - 0.135 = 0.405 (was staked at 0.54).
    out = _run([0.54], ["MLB"], ["spread_away"], ["Pittsburgh -1.5"])
    assert abs(out.iloc[0] - 0.405) < 1e-9


def test_underdog_cover_is_shifted_up():
    # Colorado +1.5 at P(win)=0.46 -> cover 0.46 + 0.135 = 0.595.
    out = _run([0.46], ["MLB"], ["spread_home"], ["Colorado +1.5"])
    assert abs(out.iloc[0] - 0.595) < 1e-9


def test_favorite_and_dog_sides_sum_to_one():
    # No push on a +-1.5 line: the two sides of the same game must still sum to 1.
    fav = _run([0.62], ["MLB"], ["spread_away"], ["Houston -1.5"]).iloc[0]
    dog = _run([0.38], ["MLB"], ["spread_home"], ["Cleveland +1.5"]).iloc[0]
    assert abs((fav + dog) - 1.0) < 1e-9


def test_totals_pass_through_unchanged():
    out = _run([0.66], ["MLB"], ["total_over"], ["Over 8.5"])
    assert out.iloc[0] == 0.66


def test_non_mlb_spread_passes_through_unchanged():
    # NBA point spreads / NHL puck lines have different margin distributions; untouched.
    out = _run([0.58], ["NBA"], ["spread_home"], ["Lakers -4.5"])
    assert out.iloc[0] == 0.58


def test_result_is_clipped_to_bounds():
    out = _run([0.10], ["MLB"], ["spread_away"], ["Padres -1.5"])
    assert out.iloc[0] == 0.02
