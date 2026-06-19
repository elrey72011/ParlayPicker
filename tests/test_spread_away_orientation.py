"""Away run-line sign derivation (19 Jun).

Regression for the "Cleveland -1.5" bug: for CLE @ HOU the live board read
CLE +1.5 / HOU -1.5, but the export emitted the away team (Cleveland) at -1.5.
A spread is an exact mirror (away = -home), and Novig (a P2P exchange) lists
both outcomes under the home team's signed point, so the away point must be
derived from the home point, never trusted directly.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _derive_spread_away_line


def test_away_line_is_mirror_of_home():
    # HOU -1.5 home favorite -> CLE (away) is +1.5.
    row = {"novig_home_point": -1.5, "novig_away_point": -1.5}  # P2P: away carries home sign
    assert _derive_spread_away_line(row) == 1.5


def test_underdog_home_gives_negative_away():
    row = {"novig_home_point": 1.5, "novig_away_point": 1.5}
    assert _derive_spread_away_line(row) == -1.5


def test_falls_back_to_other_book_home_point_when_novig_home_missing():
    # The exact failure mode: novig home point absent, novig away point wrong-signed.
    row = {"novig_away_point": -1.5, "fanduel_home_point": -1.5}
    assert _derive_spread_away_line(row) == 1.5  # not -1.5


def test_standard_book_away_point_used_only_when_no_home_point_anywhere():
    # Standard books sign per-team correctly, so their away point is trustworthy.
    row = {"draftkings_away_point": 1.5}
    assert _derive_spread_away_line(row) == 1.5


def test_no_usable_points_returns_na():
    row = {"novig_away_point": -1.5}  # only the untrustworthy novig away sign
    assert pd.isna(_derive_spread_away_line(row))
