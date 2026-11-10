"""Line movement before a game and whether the stake books followed (owner 2026-11-10). SYNTHETIC quotes."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import pinnacle_model as pm
from tests.test_pinnacle_plan_selection import both

START = datetime(2030, 1, 1, 20, 0, tzinfo=timezone.utc)
NOW = START + timedelta(hours=1)


def _event(event_id, start=START):
    return {"event_id": event_id, "home_team": "H", "away_team": "A", "scheduled_start": start.isoformat(),
            "observed_at": (start - timedelta(hours=8)).isoformat()}


def _capture(event_id, minutes_before, lines, prices=None):
    """One capture: {book: home line}; Pinnacle's home price can differ (``prices``)."""
    received = START - timedelta(minutes=minutes_before)
    stamp = (received - timedelta(seconds=30)).isoformat()
    out = []
    for book, line in lines.items():
        out += both(book, line, (prices or {}).get(book, -110), -110, received.isoformat(), stamp, event_id)
    return out


def test_pinnacles_move_and_how_fast_each_stake_book_followed(monkeypatch):
    events = {name: _event(name) for name in ("moved", "held", "single")}
    events["later"] = _event("later", START + timedelta(days=1))                 # not started: never counted
    quotes = (
        # Pinnacle goes -2.5 -> -3.5 at 60 minutes and stays; Novig follows at 55, FanDuel never does
        _capture("moved", 120, {"pinnacle": -2.5, "Novig": -2.5, "FanDuel": -2.5})
        + _capture("moved", 60, {"pinnacle": -3.5, "Novig": -2.5, "FanDuel": -2.5})
        + _capture("moved", 55, {"pinnacle": -3.5, "Novig": -3.5, "FanDuel": -2.5})
        + _capture("moved", 5, {"pinnacle": -3.5, "Novig": -3.5, "FanDuel": -2.5, "OtherBook": -3.5})
        # the line holds; only Pinnacle's price moves
        + _capture("held", 120, {"pinnacle": -6.5, "Novig": -6.5})
        + _capture("held", 10, {"pinnacle": -6.5, "Novig": -6.5}, prices={"pinnacle": -130})
        + _capture("single", 30, {"pinnacle": -1.5, "Novig": -1.5})
        + _capture("later", 30, {"pinnacle": -1.5}))
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    out = pm.line_movement("db", "NBA", "SPREAD", now=NOW)
    assert out["status"] == "MEASURED" and out["games_started"] == 3 and out["games_tracked"] == 2
    assert out["games_moved"] == 1 and out["share_moved"] == 0.5 and out["mean_abs_move"] == 1.0
    assert (out["stake_books_at_final_line"], out["stake_book_games"]) == (1, 2) and out["share_followed"] == 0.5
    assert out["median_follow_minutes"] == 5.0                                   # Novig, five minutes after Pinnacle settled
    held = (130 / 230) / (130 / 230 + 110 / 210) - 0.5
    assert out["mean_prob_shift_when_held"] == pytest.approx(held) and out["captures_per_game"] == 3.0

    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: ({}, [], []))
    assert pm.line_movement("db", "NBA", "SPREAD", now=NOW) == {
        "sport": "NBA", "market_family": "SPREAD", "games_started": 0, "games_tracked": 0, "status": "NO_GAMES"}
    assert "line_movement" not in pm.PREDICTION_FUNCTIONS


def test_a_line_that_moves_and_comes_back_is_not_a_move_and_a_late_settle_times_the_follow(monkeypatch):
    events = {"back": _event("back"), "twice": _event("twice")}
    quotes = (_capture("back", 120, {"pinnacle": -2.5}) + _capture("back", 60, {"pinnacle": -3.5})
              + _capture("back", 5, {"pinnacle": -2.5})
              + _capture("twice", 120, {"pinnacle": -2.5, "Novig": -4.5})          # Novig was at -4.5 before Pinnacle got there
              + _capture("twice", 90, {"pinnacle": -4.5, "Novig": -4.5})
              + _capture("twice", 60, {"pinnacle": -3.5, "Novig": -4.5})
              + _capture("twice", 30, {"pinnacle": -4.5, "Novig": -4.5}))           # settles at -4.5 only here
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    out = pm.line_movement("db", "NBA", "SPREAD", now=NOW)
    assert out["games_tracked"] == 2 and out["games_moved"] == 1 and out["mean_abs_move"] == 2.0
    assert out["median_follow_minutes"] == 0.0 and out["share_followed"] == 1.0


def test_the_report_prints_the_movement_table_and_reports_errors(tmp_path, monkeypatch):
    import scripts.market_readiness_report as report
    store = tmp_path / "SYNTHETIC.sqlite3"
    store.write_bytes(b"")

    def fake(path, sport, family, *, now):
        if (sport, family) == ("NBA", "TOTAL"):
            raise ValueError("bad scope")
        return dict(sport=sport, market_family=family, games_started=8, games_tracked=6, games_moved=3, share_moved=0.5,
                    mean_abs_move=0.83, mean_prob_shift_when_held=0.012, stake_books_at_final_line=9, stake_book_games=12,
                    share_followed=0.75, median_follow_minutes=5.0, captures_per_game=30, status="MEASURED")
    monkeypatch.setattr(pm, "line_movement", fake)
    rows = report.line_movement(store, extra={"NHL": ("TOTAL",)})
    assert len(rows) == sum(len(f) for f in pm.FAMILIES.values()) + 1
    text = "\n".join(report._movement_markdown(rows))
    assert "| NBA SPREAD | 6 / 8 | 3 (50%) | 0.83 | 1.2% | 9 of 12 (75%) | 5 | 30 |" in text
    assert "ValueError: bad scope" in text and "no betting splits are captured" in text
    assert report._movement_markdown([dict(sport="NHL", market_family="TOTAL", games_tracked=0)]) == []
    assert report.line_movement(tmp_path / "missing.sqlite3") == []


def test_stake_books_are_matched_by_label_because_stored_names_are_provider_keys(monkeypatch):
    # First real run, 2026-11-10: "0 of 0" stake books, because the store holds "novig", not "Novig".
    events = {"moved": _event("moved")}
    quotes = (_capture("moved", 120, {"pinnacle": -2.5, "novig": -2.5, "draftkings": -2.5, "bovada": -2.5})
              + _capture("moved", 60, {"pinnacle": -3.5, "novig": -3.5, "draftkings": -2.5, "bovada": -3.5}))
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    out = pm.line_movement("db", "NBA", "SPREAD", now=NOW)
    assert (out["stake_books_at_final_line"], out["stake_book_games"]) == (1, 2) and out["median_follow_minutes"] == 0.0
