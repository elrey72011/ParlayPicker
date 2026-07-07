"""Kalshi MLB matching regressions (4 Jul): two games missed Kalshi every run.

1. Cubs/StL event-matched to "Boston vs Los Angeles A" because the guessed
   2-letter code "SL" (for upload name "Saint Louis") substring-matched inside
   ticker "...BOSLAA" (+60 strong signal), outscoring the true STLCHC event —
   whose title says "St. Louis". The wrong-game title guard then killed the
   price, leaving a permanent miss. Fixes: real MLB codes in KALSHI_TEAM_CODES
   and a >= 3-char floor on code substring checks.
2. Yankees run-lines were spread_side_unpriceable: Kalshi truncates the title
   ("New York Y wins by over 1.5 runs?") so exact containment never found the
   subject. Fix: find_team_reference, prefix-tolerant like the title guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.kalshi_integrator import (
    _event_match_score,
    find_team_reference,
    team_code_for_league,
)

_STLCHC = {
    "title": "St. Louis vs Chicago C: Total Runs",
    "sub_title": "",
    "event_ticker": "KXMLBTOTAL-26JUL042008STLCHC",
}
_BOSLAA = {
    "title": "Boston vs Los Angeles A: Total Runs",
    "sub_title": "",
    "event_ticker": "KXMLBTOTAL-26JUL042138BOSLAA",
}
_MINNYY = {
    "title": "Minnesota vs New York Y: Total Runs",
    "sub_title": "",
    "event_ticker": "KXMLBTOTAL-26JUL041335MINNYY",
}


def test_mlb_team_codes_resolve_to_kalshi_ticker_codes():
    assert team_code_for_league("MLB", "Saint Louis") == "STL"
    assert team_code_for_league("MLB", "Chicago Cubs") == "CHC"
    assert team_code_for_league("MLB", "New York Yankees") == "NYY"


def test_cubs_game_prefers_its_own_event_over_boslaa():
    own = _event_match_score(_STLCHC, "Chicago Cubs", "Saint Louis", "MLB", date_code="26JUL04")
    wrong = _event_match_score(_BOSLAA, "Chicago Cubs", "Saint Louis", "MLB", date_code="26JUL04")
    assert own > wrong, f"own event {own} must outscore wrong event {wrong}"
    assert own >= 100  # both codes hit the ticker: a strong, unambiguous match


def test_short_guessed_code_is_not_a_strong_signal():
    # "Saint Louis" used to guess code "SL", which substring-matched "...BOSLAA"
    # as a +60 strong away signal. With the mapped STL code and the 3-char floor,
    # the wrong event must score below the acceptance threshold (25).
    wrong = _event_match_score(_BOSLAA, "Chicago Cubs", "Saint Louis", "MLB", date_code="26JUL04")
    assert wrong < 25, f"wrong-game event must not clear acceptance, got {wrong}"


def test_yankees_event_still_strong_with_truncated_title():
    score = _event_match_score(_MINNYY, "New York Yankees", "Minnesota", "MLB", date_code="26JUL04")
    assert score >= 100


def test_find_team_reference_handles_kalshi_truncation():
    title = " New York Y Wins By Over 15 Runs "
    assert find_team_reference(title, "new york yankees") >= 0
    assert find_team_reference(title, "minnesota") == -1
    # Full-name containment still works and wins
    assert find_team_reference(" Cincinnati Wins By Over 15 Runs ", "cincinnati") >= 0


def test_find_team_reference_needs_four_char_fragment():
    # A too-short fragment can't anchor a subject ("a" must not match everything).
    assert find_team_reference(" A Wins By Over 15 Runs ", "arizona diamondbacks") == -1
    assert find_team_reference("", "cincinnati") == -1
    assert find_team_reference(" Cincinnati Wins ", "") == -1


def test_same_line_spread_pair_prefers_priceable_subject(monkeypatch):
    """A run-line event has two markets at the same strike (one per subject).
    A pick on HOME +1.5 is only priceable off the AWAY-subject contract; the
    selector must prefer it over list order (7 Jul: SF +1.5 went
    spread_side_unpriceable because the SF-subject market was picked)."""
    import pandas as pd
    import app_core.kalshi_integrator as ki

    class FakeResponse:
        def __init__(self, data, status_code=200):
            self._d, self.status_code = data, status_code
        def json(self):
            return self._d

    def fake_request(url, **kwargs):
        if "/events" in url and "series_ticker" in str(kwargs.get("params", {})):
            return FakeResponse({"events": [{
                "event_ticker": "KXMLBSPREAD-26JUL072145TORSF",
                "title": "Toronto vs San Francisco: Spread",
                "close_time": "2026-07-08T05:00:00Z",
            }]})
        if "KXMLBSPREAD-26JUL072145TORSF" in url:
            return FakeResponse({"event": {"markets": [
                # SF-subject market FIRST so naive list order picks it
                {"ticker": "KXMLBSPREAD-26JUL072145TORSF-SF2",
                 "title": "San Francisco wins by over 1.5 runs?",
                 "floor_strike": 1.5, "yes_bid_dollars": 0.30, "yes_ask_dollars": 0.32},
                {"ticker": "KXMLBSPREAD-26JUL072145TORSF-TOR2",
                 "title": "Toronto wins by over 1.5 runs?",
                 "floor_strike": 1.5, "yes_bid_dollars": 0.38, "yes_ask_dollars": 0.39},
            ]}})
        return FakeResponse({}, 404)

    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()
    monkeypatch.setattr(ki, "_make_kalshi_request", fake_request)

    df = pd.DataFrame([{
        "league": "MLB", "home_team": "San Francisco", "away_team": "Toronto",
        "game_date": "2026-07-08T01:46:00Z", "market_type": "spread_home",
        "spread_line": 1.5, "best_pick": "San Francisco +1.5",
        "pick_team": "San Francisco",
    }])
    out = ki.enrich_with_kalshi_markets(df)
    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert "Toronto" in str(out.loc[0, "kalshi_market_title"])
    assert abs(float(out.loc[0, "kalshi_probability"]) - 0.615) < 0.01
