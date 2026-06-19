"""Raw per-book odds diagnostic string (19 Jun).

Surfaces what the live feed actually delivered per book, so a flipped spread
orientation (e.g. the Cleveland -1.5 case) can be diagnosed from the export:
does the feed report the home spread wrong-signed at the source, or is it a
parse bug?
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _fmt_odds_token, _raw_book_odds_diag


def test_fmt_signs_and_missing():
    assert _fmt_odds_token(-1.5) == "-1.5"
    assert _fmt_odds_token(1.5) == "+1.5"
    assert _fmt_odds_token(-120) == "-120"
    assert _fmt_odds_token(115) == "+115"
    assert _fmt_odds_token(pd.NA) == "·"
    assert _fmt_odds_token(None) == "·"


def test_diag_string_for_flipped_houston_cleveland():
    # Reality: HOU (home) -1.5 favorite, CLE (away) +1.5. If the feed instead reports
    # the home spread as +1.5 while the moneyline favors the home team, the string makes
    # that source-level contradiction visible.
    row = {
        "novig_home_point": 1.5, "novig_away_point": -1.5,
        "novig_h2h_home_price": -120, "novig_h2h_away_price": 115,
    }
    s = _raw_book_odds_diag(row)
    assert s == "novig: sp H=+1.5/A=-1.5 ml H=-120/A=+115"


def test_multiple_books_joined_and_empty_omitted():
    row = {
        "novig_home_point": -1.5, "novig_away_point": 1.5,
        "fanduel_home_point": -1.5, "fanduel_away_point": 1.5,
        # draftkings/betmgm absent -> omitted entirely
    }
    s = _raw_book_odds_diag(row)
    assert "novig: sp H=-1.5/A=+1.5" in s
    assert "fanduel: sp H=-1.5/A=+1.5" in s
    assert "draftkings" not in s and "betmgm" not in s
    assert s.count("|") == 1


def test_no_data_returns_na():
    assert pd.isna(_raw_book_odds_diag({"league": "MLB"}))
