"""Source the run line from a book whose spread agrees with its own moneyline (19 Jun).

The 10:49 raw_book_odds_diag proved novig + fanduel published HOU/CLE with the spread
flipped vs their own moneyline, while draftkings + betmgm had it right. We now take the
line AND price from the first internally-consistent book.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _consistent_spread_book, _expand_live_odds_to_bet_rows


def _cleveland_row():
    # HOU home / CLE away. novig+fanduel flipped, draftkings+betmgm correct.
    return {
        "league": "MLB", "home_team": "Houston", "away_team": "Cleveland",
        "game_date": "2026-06-19", "matchup_id": "m", "commence_time_raw": "2026-06-19T00:11:00Z",
        "novig_home_point": 1.5, "novig_away_point": -1.5, "novig_home_price": -150, "novig_away_price": 174,
        "novig_h2h_home_price": -115, "novig_h2h_away_price": 113,
        "fanduel_home_point": 1.5, "fanduel_away_point": -1.5, "fanduel_home_price": -150, "fanduel_away_price": 160,
        "fanduel_h2h_home_price": -120, "fanduel_h2h_away_price": 102,
        "draftkings_home_point": -1.5, "draftkings_away_point": 1.5, "draftkings_home_price": 150, "draftkings_away_price": -190,
        "draftkings_h2h_home_price": -125, "draftkings_h2h_away_price": 104,
        "betmgm_home_point": -1.5, "betmgm_away_point": 1.5, "betmgm_home_price": 148, "betmgm_away_price": -185,
        "betmgm_h2h_home_price": -130, "betmgm_h2h_away_price": 105,
    }


def test_picks_first_consistent_book_skipping_flipped_ones():
    assert _consistent_spread_book(_cleveland_row()) == "draftkings"


def test_picks_novig_when_novig_is_consistent():
    # Colorado home (+1.5 underdog, ml +127) / Pittsburgh away (-1.5 fav, ml -130):
    # novig already agrees with its moneyline, so it's used unchanged.
    row = {
        "novig_home_point": 1.5, "novig_away_point": -1.5,
        "novig_h2h_home_price": 127, "novig_h2h_away_price": -130,
    }
    assert _consistent_spread_book(row) == "novig"


def test_none_when_no_book_consistent():
    row = {
        "novig_home_point": 1.5, "novig_away_point": -1.5,
        "novig_h2h_home_price": -115, "novig_h2h_away_price": 113,  # spread says away fav, ml says home fav
    }
    assert _consistent_spread_book(row) is None


def test_expand_orients_cleveland_to_plus_1_5_from_consistent_book():
    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([_cleveland_row()]), None)
    away = out[out.market_type == "spread_away"].iloc[0]
    home = out[out.market_type == "spread_home"].iloc[0]
    assert float(away["spread_line"]) == 1.5      # Cleveland +1.5 (was -1.5)
    assert float(away["odds_american"]) == -190.0  # draftkings' away price, not novig's
    assert float(home["spread_line"]) == -1.5     # Houston -1.5
