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
    # novig's 7.0 is a WILD outlier (1.5 off the 8.5 median, > NOVIG_TOTAL_OUTLIER_TOL) -> a
    # likely stale exchange posting, so fall back to the consensus book.
    assert float(over["total_line"]) == 8.5        # consensus, not novig's outlier 7.0
    assert float(over["odds_american"]) == -105.0  # fanduel's price at the consensus line


def test_expand_uses_novig_line_when_within_tolerance():
    # The real 26-Jun bug: novig posts 8.5 while the field is at 9.0 (a normal 0.5 book
    # disagreement). The user bets on novig, so its 8.5 — and its price — must be used.
    row = {
        "league": "MLB", "home_team": "Detroit", "away_team": "Houston",
        "game_date": "2026-06-26", "matchup_id": "m2", "commence_time_raw": "2026-06-26T22:41:00Z",
        "novig_over_point": 8.5, "novig_over_price": -106, "novig_under_point": 8.5, "novig_under_price": 104,
        "fanduel_over_point": 9.0, "fanduel_over_price": -112, "fanduel_under_point": 9.0, "fanduel_under_price": -104,
        "draftkings_over_point": 9.0, "draftkings_over_price": -115, "draftkings_under_point": 9.0, "draftkings_under_price": -105,
        "betmgm_over_point": 9.0, "betmgm_over_price": -115, "betmgm_under_point": 9.0, "betmgm_under_price": -105,
    }
    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    over = out[out.market_type == "total_over"].iloc[0]
    assert float(over["total_line"]) == 8.5        # novig's real line, NOT the 9.0 consensus
    assert float(over["odds_american"]) == -106.0  # and novig's own price


def test_diag_includes_total_points():
    row = {
        "novig_home_point": -1.5, "novig_away_point": 1.5,
        "novig_h2h_home_price": -120, "novig_h2h_away_price": 110,
        "novig_over_point": 8.5, "novig_under_point": 8.5,
    }
    s = _raw_book_odds_diag(row)
    assert "tot O=+8.5/U=+8.5" in s


def test_standard_book_mode_prevents_unquoted_synthetic_median():
    row = {
        # The 30-Jul Atlanta/Washington failure: including Novig produced a synthetic
        # 9.75 median that no book quoted, so both total candidates disappeared.
        "novig_over_point": 4.5, "novig_under_point": 4.5,
        "fanduel_over_point": 9.5, "fanduel_under_point": 9.5,
        "draftkings_over_point": 10.0, "draftkings_under_point": 10.0,
        "betmgm_over_point": 10.0, "betmgm_under_point": 10.0,
    }
    assert _consensus_total_line(row) == 10.0
    assert _consistent_total_book(row, "over") == "draftkings"
    assert _consistent_total_book(row, "under") == "draftkings"


def test_expand_recovers_totals_from_standard_book_agreement():
    row = {
        "league": "MLB", "home_team": "Atlanta", "away_team": "Washington",
        "game_date": "2026-07-30", "matchup_id": "atl-wsh",
        "commence_time_raw": "2026-07-30T23:15:00Z",
        "novig_over_point": 4.5, "novig_over_price": 390,
        "novig_under_point": 4.5, "novig_under_price": -600,
        "fanduel_over_point": 9.5, "fanduel_over_price": -102,
        "fanduel_under_point": 9.5, "fanduel_under_price": -118,
        "draftkings_over_point": 10.0, "draftkings_over_price": -108,
        "draftkings_under_point": 10.0, "draftkings_under_price": -112,
        "betmgm_over_point": 10.0, "betmgm_over_price": -105,
        "betmgm_under_point": 10.0, "betmgm_under_price": -115,
    }
    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    totals = out[out["market_type"].isin(["total_over", "total_under"])].copy()

    assert set(totals["market_type"]) == {"total_over", "total_under"}
    assert set(pd.to_numeric(totals["total_line"])) == {10.0}
    over = totals[totals["market_type"] == "total_over"].iloc[0]
    under = totals[totals["market_type"] == "total_under"].iloc[0]
    assert float(over["odds_american"]) == -108.0
    assert float(under["odds_american"]) == -112.0

