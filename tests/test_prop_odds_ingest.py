"""Odds API pitcher-strikeout prop parsing (no network -- fixture only)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.prop_odds_ingest import parse_strikeout_props


def _event(bookmakers):
    return {"home_team": "Astros", "away_team": "Guardians", "bookmakers": bookmakers}


_DK = {
    "key": "draftkings",
    "markets": [
        {
            "key": "pitcher_strikeouts",
            "outcomes": [
                {"name": "Over", "description": "Framber Valdez", "point": 6.5, "price": -120},
                {"name": "Under", "description": "Framber Valdez", "point": 6.5, "price": -110},
                {"name": "Over", "description": "Tanner Bibee", "point": 5.5, "price": 100},
                {"name": "Under", "description": "Tanner Bibee", "point": 5.5, "price": -130},
            ],
        }
    ],
}


def test_parses_per_pitcher_over_under():
    rows = parse_strikeout_props(_event([_DK]))
    by = {r["pitcher"]: r for r in rows}
    assert by["Framber Valdez"]["line"] == 6.5
    assert by["Framber Valdez"]["over_odds"] == -120
    assert by["Framber Valdez"]["under_odds"] == -110
    assert by["Tanner Bibee"]["over_odds"] == 100
    assert by["Framber Valdez"]["book"] == "draftkings"
    assert by["Framber Valdez"]["home_team"] == "Astros"


def test_skips_pitcher_missing_a_side():
    book = {
        "key": "draftkings",
        "markets": [{"key": "pitcher_strikeouts", "outcomes": [
            {"name": "Over", "description": "Lonely Arm", "point": 6.5, "price": -115},
        ]}],
    }
    assert parse_strikeout_props(_event([book])) == []


def test_book_priority_prefers_first_available():
    fd = {
        "key": "fanduel",
        "markets": [{"key": "pitcher_strikeouts", "outcomes": [
            {"name": "Over", "description": "Framber Valdez", "point": 6.5, "price": -125},
            {"name": "Under", "description": "Framber Valdez", "point": 6.5, "price": -105},
        ]}],
    }
    # novig absent; priority is (novig, draftkings, fanduel, ...) -> draftkings wins.
    rows = parse_strikeout_props(_event([fd, _DK]))
    assert all(r["book"] == "draftkings" for r in rows)


def test_no_strikeout_market_returns_empty():
    book = {"key": "draftkings", "markets": [{"key": "totals", "outcomes": []}]}
    assert parse_strikeout_props(_event([book])) == []


def test_non_dict_input_is_safe():
    assert parse_strikeout_props(None) == []
    assert parse_strikeout_props([]) == []
