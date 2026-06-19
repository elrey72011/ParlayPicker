"""Consistent-book sourcing for totals (19 Jun, preventive).

Mirrors the spread fix: a thin exchange (novig) posting an off-market total line
should not set the line/odds. Use the consensus line across books and price from a
book at that line. No-op when books agree (the normal case).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import (
    _consensus_total_line,
    _consistent_total_book,
    _raw_book_odds_diag,
    _expand_live_odds_to_bet_rows,
)


def test_consensus_ignores_single_outlier():
    row = {
        "novig_over_point": 7.0, "novig_under_point": 7.0,   # outlier
        "fanduel_over_point": 8.5, "fanduel_under_point": 8.5,
        "draftkings_over_point": 8.5, "draftkings_under_point": 8.5,
        "betmgm_over_point": 8.5, "betmgm_under_point": 8.5,
    }
    assert _consensus_total_line(row) == 8.5
    assert _consistent_total_book(row, "over") == "fanduel"  # first book at consensus, skipping novig


def test_novig_used_when_it_agrees():
    row = {
        "novig_over_point": 8.5, "novig_under_point": 8.5,
        "fanduel_over_point": 8.5, "fanduel_under_point": 8.5,
    }
    assert _consistent_total_book(row, "over") == "novig"


def test_expand_uses_consensus_total_line_and_price():
    row = {
        "league": "MLB", "home_team": "Miami", "away_team": "San Francisco",
        "game_date": "2026-06-19", "matchup_id": "m", "commence_time_raw": "2026-06-19T00:11:00Z",
        # novig posts an off-market 7.0; the field agrees on 8.5
        "novig_over_point": 7.0, "novig_over_price": -130, "novig_under_point": 7.0, "novig_under_price": 110,
        "fanduel_over_point": 8.5, "fanduel_over_price": -105, "fanduel_under_point": 8.5, "fanduel_under_price": -115,
        "draftkings_over_point": 8.5, "draftkings_over_price": -108, "draftkings_under_point": 8.5, "draftkings_under_price": -112,
        "betmgm_over_point": 8.5, "betmgm_over_price": -110, "betmgm_under_point": 8.5, "betmgm_under_price": -110,
    }
    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    over = out[out.market_type == "total_over"].iloc[0]
    assert float(over["total_line"]) == 8.5        # consensus, not novig's 7.0
    assert float(over["odds_american"]) == -105.0  # fanduel's price at the consensus line


def test_diag_includes_total_points():
    row = {
        "novig_home_point": -1.5, "novig_away_point": 1.5,
        "novig_h2h_home_price": -120, "novig_h2h_away_price": 110,
        "novig_over_point": 8.5, "novig_under_point": 8.5,
    }
    s = _raw_book_odds_diag(row)
    assert "tot O=+8.5/U=+8.5" in s
