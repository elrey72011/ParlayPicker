import pandas as pd
import pytest

from app_core.kalshi_integrator import (
    _build_row_kalshi_date_code,
    _event_match_score,
    league_series_ticker,
    orient_spread_kalshi_prob,
    spread_market_subject_is_home,
    team_code_for_league,
)


WNBA_MISS_ROWS = [
    ("spread_home", "Dallas", "Atlanta", "KXWNBASPREAD-26JUL29DALATL"),
    ("spread_away", "Dallas", "Atlanta", "KXWNBASPREAD-26JUL29DALATL"),
    ("total_over", "Dallas", "Atlanta", "KXWNBATOTAL-26JUL29DALATL"),
    ("total_under", "Dallas", "Atlanta", "KXWNBATOTAL-26JUL29DALATL"),
    ("spread_home", "Phoenix", "Golden State", "KXWNBASPREAD-26JUL29PHXGS"),
    ("spread_away", "Phoenix", "Golden State", "KXWNBASPREAD-26JUL29PHXGS"),
    ("total_over", "Phoenix", "Golden State", "KXWNBATOTAL-26JUL29PHXGS"),
    ("total_under", "Phoenix", "Golden State", "KXWNBATOTAL-26JUL29PHXGS"),
]


@pytest.mark.parametrize("market_type,home_team,away_team,event_ticker", WNBA_MISS_ROWS)
def test_all_reported_wnba_misses_route_to_a_series_and_match_their_event(
    market_type, home_team, away_team, event_ticker
):
    expected_series = "KXWNBASPREAD" if market_type.startswith("spread") else "KXWNBATOTAL"
    assert league_series_ticker("WNBA", market_type) == expected_series

    event = {
        "title": f"{away_team} at {home_team}",
        "sub_title": "Women's professional basketball",
        "event_ticker": event_ticker,
    }
    assert _event_match_score(
        event, home_team, away_team, "WNBA", date_code="26JUL29"
    ) >= 25


def test_wnba_moneyline_uses_wnba_game_series():
    assert league_series_ticker("WNBA", "moneyline") == "KXWNBAGAME"


def test_wnba_team_codes_do_not_inherit_other_league_city_codes():
    assert team_code_for_league("WNBA", "Golden State") == "GS"
    assert team_code_for_league("NBA", "Golden State") == "GSW"
    assert team_code_for_league("WNBA", "Washington") == "WSH"
    assert team_code_for_league("NBA", "Washington") == "WAS"
    assert team_code_for_league("WNBA", "Los Angeles") == "LA"
    assert team_code_for_league("NHL", "Los Angeles") == "LAK"


def test_wnba_kalshi_date_prefers_local_game_time_over_utc_midnight_date():
    row = pd.Series(
        {
            "game_date": "2026-07-30T00:00:00Z",
            "game_time_est": "2026-07-29 8:00 PM ET",
            "matchup_id": "ATLANTA|DALLAS|2026-07-29",
        }
    )
    assert _build_row_kalshi_date_code(row, "WNBA") == "26JUL29"


def test_wnba_short_code_ticker_identifies_golden_state_as_spread_subject():
    market = {
        "ticker": "KXWNBASPREAD-26JUL29GSPHX-GS4",
        "title": "Women's professional basketball spread",
    }
    assert (
        spread_market_subject_is_home(
            market, "WNBA", home_team="Phoenix", away_team="Golden State"
        )
        is False
    )


@pytest.mark.parametrize(
    "pick_is_home,pick_line,expected_probability",
    [
        (False, -3.5, 0.58),  # Golden State -3.5 is the ticker subject's YES side.
        (True, 3.5, 0.42),    # Phoenix +3.5 is the complementary NO side.
    ],
)
def test_reported_phoenix_golden_state_spread_sides_are_priceable(
    pick_is_home, pick_line, expected_probability
):
    subject_is_home = spread_market_subject_is_home(
        {
            "ticker": "KXWNBASPREAD-26JUL29GSPHX-GS4",
            "title": "WNBA spread",
        },
        "WNBA",
        home_team="Phoenix",
        away_team="Golden State",
    )
    probability = orient_spread_kalshi_prob(
        0.58,
        subject_is_home=subject_is_home,
        pick_is_home=pick_is_home,
        pick_line=pick_line,
    )
    assert probability == pytest.approx(expected_probability)
