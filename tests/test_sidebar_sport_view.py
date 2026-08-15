from app.ui.sidebar_controls import (
    ALL_SPORTS_LABEL,
    FALLBACK_SPORTS,
    _initial_sport_view,
    _sports_for_view,
)


def test_nfl_view_resolves_to_nfl_only():
    assert _sports_for_view("NFL", FALLBACK_SPORTS) == ["NFL"]


def test_all_sports_view_preserves_the_full_default_slate():
    assert _sports_for_view(ALL_SPORTS_LABEL, FALLBACK_SPORTS) == FALLBACK_SPORTS


def test_existing_single_nfl_session_migrates_to_nfl_view():
    assert _initial_sport_view(["NFL"], FALLBACK_SPORTS) == "NFL"


def test_existing_multi_sport_session_migrates_to_all_sports_view():
    assert (
        _initial_sport_view(["MLB", "NFL", "WNBA"], FALLBACK_SPORTS)
        == ALL_SPORTS_LABEL
    )


def test_unknown_view_fails_open_to_all_supported_sports():
    assert _sports_for_view("Unknown", FALLBACK_SPORTS) == FALLBACK_SPORTS
