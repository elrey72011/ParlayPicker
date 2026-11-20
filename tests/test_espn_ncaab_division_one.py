"""Final scores for every Division I college basketball game, not ESPN's featured few (2026-11-20).

ESPN's default men's college basketball scoreboard lists only its featured games: 5 of the 36 played on
2026-11-19. The evidence store settles saved picks from that scoreboard, so nearly every college
basketball pick stayed without a result (the grading run of 2026-11-20 left 2,662 candidates at
NO_FINAL_PROVIDER_RESULT, and TheOver's tally had a result for 12 of 197 bets). The scoreboard is now
also asked for group 50, all of Division I. SYNTHETIC responses.
"""
from datetime import date
from types import SimpleNamespace

from app_core import espn_results, result_providers


def test_college_basketball_asks_for_all_of_division_one_as_well():
    urls = espn_results._scoreboard_urls("NCAAB", "20261119")
    assert urls[0].endswith("/mens-college-basketball/scoreboard?dates=20261119")
    assert urls[1].endswith("/mens-college-basketball/scoreboard?dates=20261119&groups=50&limit=500") and len(urls) == 2


def test_other_leagues_ask_what_they_asked_before():
    assert [len(espn_results._scoreboard_urls(league, "20261119")) for league in ("NBA", "NHL", "NFL", "MLB", "WNBA")] == [1] * 5
    football = espn_results._scoreboard_urls("NCAAF", "20261119")
    assert len(football) == 2 and football[1].endswith("&groups=81&limit=300")


def _event(event_id, away, home, away_score, home_score):
    team = lambda name, side, score: {"homeAway": side, "score": str(score), "team": {"displayName": name}}
    return {"id": event_id, "date": "2026-11-20T00:00Z", "competitions": [{
        "date": "2026-11-20T00:00Z", "status": {"type": {"completed": True, "state": "post", "name": "STATUS_FINAL"}},
        "competitors": [team(home, "home", home_score), team(away, "away", away_score)]}]}


def test_a_game_on_both_scoreboards_is_one_result_and_the_division_one_only_game_is_found(monkeypatch):
    featured, small = _event("1", "Houston Cougars", "Kansas Jayhawks", 92, 83), _event("2", "Maine Black Bears", "Quinnipiac Bobcats", 69, 67)

    def get(url, timeout=10):
        events = [featured, small] if "groups=50" in url else [featured]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"events": events})
    monkeypatch.setattr(result_providers.requests, "get", get)
    result = result_providers.fetch_results(date(2026, 11, 19), ["NCAAB"])
    found = result["scores"]
    assert sorted(s["event_id"] for s in found) == ["1", "2"] and not result["errors"] and len(result["events"]) == 2
    assert {s["home"]: (s["away_score"], s["home_score"]) for s in found}["Quinnipiac Bobcats"] == (69, 67)
