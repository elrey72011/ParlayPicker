"""Graded results of the replay's first-edge bets, any-time vs late entry (owner 2026-11-13). SYNTHETIC quotes."""
import pytest

from app_core import pinnacle_model as pm
from tests.test_pace_comparison import NOW, _scope


def _score(event_id, home, away, version=1, available="2030-01-01T23:00:00+00:00"):
    return {"event_id": event_id, "result_id": f"r-{event_id}-{version}", "home_score": home, "away_score": away,
            "grading_version": version, "available_at": available}


def test_replay_results_grades_the_same_bets_at_their_own_line(monkeypatch):
    # Every game: novig H -2.5 at +106 (decimal 2.06) is the first capture 3% better than Pinnacle.
    games = [("won", 60, 20), ("lost", 60, 20), ("ungraded", 60, 20), ("none", 0, 0)]
    events, quotes = _scope(games)
    scores = [_score("won", 80, 70),
              _score("lost", 80, 70), _score("lost", 71, 70, version=2),  # a later grading version wins
              _score("none", 80, 70)]                                        # no bet on this game
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    out = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)
    assert out["status"] == "MEASURED" and out["games_started"] == 4
    found = out["by_edge"]["2%"]
    assert (found["bets"], found["graded"], found["ungraded"]) == (3, 2, 1)
    assert (found["wins"], found["losses"], found["pushes"]) == (1, 1, 0)
    assert found["units"] == pytest.approx(1.06 - 1)
    assert found["roi_per_bet"] == pytest.approx((1.06 - 1) / 2) and found["roi_se"] == pytest.approx(2.06 / 2)
    assert out["by_edge"].keys() == {"1%", "2%"}


def test_a_push_returns_the_stake(monkeypatch):
    events, quotes = _scope([("push", 60, 20)])
    for quote in quotes:
        quote["line"] = -3.0 if quote["selection"] == "H" else 3.0
    monkeypatch.setattr("app_core.prospective_research_models._read_scope",
                        lambda *a: (events, quotes, [_score("push", 73, 70)]))
    found = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)["by_edge"]["2%"]
    assert (found["graded"], found["pushes"], found["units"], found["roi_per_bet"]) == (1, 1, 0.0, 0.0)


def test_late_entry_grades_only_bets_made_close_to_the_start(monkeypatch):
    games = [("early", 150, 20), ("late", 45, 20), ("spanning", 70, 30)]
    events, quotes = _scope(games)
    scores = [_score("early", 80, 70), _score("late", 71, 70), _score("spanning", 80, 70)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    any_time = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)["by_edge"]["2%"]
    late = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW,
                             entry_within_minutes=pm.LATE_ENTRY_MINUTES)["by_edge"]["2%"]
    assert (any_time["bets"], any_time["wins"], any_time["losses"]) == (3, 2, 1)
    assert (late["bets"], late["wins"], late["losses"]) == (2, 1, 1)          # the early win is not a late bet


def test_no_started_games_and_outside_the_hashed_prediction_functions(monkeypatch):
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: ({}, [], []))
    empty = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)
    assert empty["status"] == "NO_GAMES" and empty["by_edge"] == {}
    assert "replay_results" not in pm.PREDICTION_FUNCTIONS


def test_the_report_prints_any_time_and_late_rows_and_reports_errors(tmp_path, monkeypatch):
    import scripts.market_readiness_report as report
    store = tmp_path / "store.sqlite3"
    store.write_text("")
    calls = []

    def graded(path, sport, family, *, now, lookback_days, books, entry_within_minutes):
        calls.append((sport, family, lookback_days, entry_within_minutes))
        if sport == "NBA":
            raise ValueError("boom")
        return {"sport": sport, "market_family": family, "status": "MEASURED", "by_edge": {
            "1%": {"bets": 4, "graded": 3, "ungraded": 1, "wins": 2, "losses": 1, "pushes": 0, "units": 0.9,
                   "roi_per_bet": 0.3, "roi_se": 0.5}}}
    monkeypatch.setattr(pm, "replay_results", graded)
    rows = report.replay_results(store, now=NOW)
    assert {(c[2], c[3]) for c in calls} == {(7, None), (7, 60), (14, None), (14, 60)}
    text = "\n".join(report._results_markdown(rows))
    assert "READS results" in text
    assert "| NCAAB SPREAD | last 7 days | any time | 1% | 4 | 3 | 2 / 1 / 0 | +0.90 | +30.0% (50.0%) |" in text
    assert "| NCAAB SPREAD | last 7 days | last 60 min | 1% |" in text
    assert "ValueError: boom" in text
    assert report._results_markdown([]) == []
