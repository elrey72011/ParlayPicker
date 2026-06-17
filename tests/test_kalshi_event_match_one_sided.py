"""Kalshi event match must reject one-sided (wrong-game) matches.

17 Jun: a Dodgers/Tampa Bay pick matched a "Los Angeles A vs Arizona" event on the
shared city token while the real opponent was absent. A correct event carries BOTH
teams; a one-sided match is rejected so we take a Kalshi miss, not another game's price.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import _event_match_score

_ACCEPT_FLOOR = 10  # enrich_with_kalshi_markets sets best_event_match=None below this


def test_one_sided_city_collision_is_rejected():
    wrong = {
        "title": "Los Angeles A vs Arizona Total Runs?",
        "sub_title": "LAA at ARI",
        "event_ticker": "KXMLBGAME-26JUN17LAAARI",
    }
    score = _event_match_score(wrong, "Los Angeles Dodgers", "Tampa Bay", "MLB", date_code="26JUN17")
    assert score < _ACCEPT_FLOOR


def test_both_teams_present_still_matches():
    right = {
        "title": "Los Angeles Dodgers vs Tampa Bay",
        "sub_title": "LAD at TB",
        "event_ticker": "KXMLBGAME-26JUN17LADTB",
    }
    score = _event_match_score(right, "Los Angeles Dodgers", "Tampa Bay", "MLB", date_code="26JUN17")
    assert score >= 25


def test_strong_single_team_match_is_penalized_not_hard_rejected():
    # When the present team matches STRONGLY (code/full name) and only the opponent is
    # absent, it is penalized but kept — the opponent may just be coded differently on
    # the same-day event (a team plays one game/day, so this is rarely a wrong game).
    # Contrast with the weak city-token-only one-sided match, which IS rejected above.
    ev = {
        "title": "Los Angeles Dodgers vs Colorado",
        "sub_title": "LAD at COL",
        "event_ticker": "KXMLBGAME-26JUN17LADCOL",
    }
    score = _event_match_score(ev, "Los Angeles Dodgers", "Tampa Bay", "MLB", date_code="26JUN17")
    assert score >= _ACCEPT_FLOOR  # kept (penalized), not hard-rejected
