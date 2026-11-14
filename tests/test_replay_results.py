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


def test_results_are_split_by_what_pinnacle_did_afterwards_and_by_side(monkeypatch):
    """2026-11-14: the week's gap bets beat Pinnacle's later price and still lost. The split says where
    the results fell: priced positive or not at Pinnacle's last capture, line moved for or against, not priced."""
    games = [("won", 60, 20), ("lost", 60, 20)]
    events, quotes = _scope(games)
    scores = [_score("won", 80, 70), _score("lost", 71, 70)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    found = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)["by_edge"]["2%"]
    parts = found["splits"]
    assert {part["by"] for part in parts} == {"after_pinnacle", "side"}
    for by in ("after_pinnacle", "side"):
        mine = [part for part in parts if part["by"] == by]
        assert sum(part["graded"] for part in mine) == found["graded"]
        assert sum(part["wins"] for part in mine) == found["wins"]
        assert sum(part["units"] for part in mine) == pytest.approx(found["units"])
    kinds = {part["group"] for part in parts if part["by"] == "after_pinnacle"}
    assert kinds <= {"PRICED_POSITIVE", "PRICED_NOT_POSITIVE", "MOVED_IN_FAVOUR", "MOVED_AGAINST", "MOVED_LEVEL",
                     "NO_LATE_CAPTURE", "ENTERED_AT_LAST_CAPTURE"}
    assert {part["group"] for part in parts if part["by"] == "side"} <= {"home", "away"}
    assert "_after_pinnacle" not in pm.PREDICTION_FUNCTIONS


def test_the_report_prints_the_split_for_seven_days_at_one_percent():
    from scripts import market_readiness_report as report
    part = dict(by="after_pinnacle", group="MOVED_AGAINST", graded=4, wins=1, losses=3, pushes=0, units=-2.0,
                roi_per_bet=-0.5, roi_se=0.4)
    row = dict(sport="NCAAB", market_family="SPREAD", window="last 7 days", entry="any time",
               by_edge={"1%": dict(bets=4, splits=[part]), "2%": dict(bets=1, splits=[dict(part, group="X")])})
    text = "\n".join(report._results_splits_markdown([row, dict(row, window="last 14 days")]))
    assert "| NCAAB SPREAD | any time | after_pinnacle | MOVED_AGAINST | 4 | 1 / 3 / 0 | -2.00 | -50.0% (40.0%) |" in text
    assert text.count("MOVED_AGAINST") == 1 and "| X |" not in text
    assert report._results_splits_markdown([dict(row, by_edge={})]) == []


def test_every_graded_bet_is_listed_for_a_hand_check(monkeypatch):
    events, quotes = _scope([("won", 60, 20), ("lost", 60, 20), ("ungraded", 60, 20)])
    scores = [_score("won", 80, 70), _score("lost", 71, 70)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    found = pm.replay_results("db", "NCAAB", "SPREAD", now=NOW)["by_edge"]["2%"]
    listed = found["graded_bets"]
    assert len(listed) == found["graded"] == 2
    assert sorted(bet["outcome"] for bet in listed) == ["LOSS", "WIN"]
    for bet in listed:
        assert bet["selection"] in (bet["home_team"], bet["away_team"]) and bet["decimal_odds"] == pytest.approx(2.06)
        assert {"scheduled_start", "line", "sportsbook", "entered_at", "home_score", "away_score", "after_pinnacle",
                "closing_value"} <= set(bet)


def test_the_report_lists_ncaab_late_bets_only():
    from scripts import market_readiness_report as report
    bet = dict(scheduled_start="2030-01-02T00:00:00+00:00", home_team="H", away_team="A", selection="H", line=-2.5,
               decimal_odds=2.06, sportsbook="Novig", entered_at="2030-01-01T23:10:00+00:00", home_score=71, away_score=70,
               outcome="LOSS", after_pinnacle="PRICED_POSITIVE", closing_value=0.021)
    row = dict(sport="NCAAB", market_family="SPREAD", window="last 7 days", entry="last 60 min",
               by_edge={"1%": dict(bets=1, graded_bets=[bet, dict(bet, decimal_odds=1.8, closing_value=None)])})
    text = "\n".join(report._results_listed_markdown([row, dict(row, entry="any time"), dict(row, sport="NBA")]))
    assert "| SPREAD | 2030-01-02T00:00 | A at H | H -2.5 | +106 | Novig | 23:10 | 70-71 | LOSS | PRICED_POSITIVE | +2.1% |" in text
    assert "| -125 | Novig |" in text and text.count("A at H") == 2
    assert report._results_listed_markdown([dict(row, by_edge={"1%": dict(bets=0, graded_bets=[])})]) == []
