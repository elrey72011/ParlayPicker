"""The running record of the side Pinnacle moved toward (owner 2026-11-14). SYNTHETIC tracks and scores."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import pinnacle_model as pm

START = datetime(2030, 1, 2, 0, 0, tzinfo=timezone.utc)          # 7 PM ET on the slate night of 2030-01-01
NOW = START + timedelta(hours=6)


def _event(event_id, start=START):
    return {"event_id": event_id, "home_team": "H", "away_team": "A", "scheduled_start": start.isoformat()}


def _track(*rows):
    return [(START - timedelta(minutes=minutes), line, p) for minutes, line, p in rows]


def test_the_side_is_read_from_the_first_capture_to_the_last_one_an_hour_out():
    event = _event("g")
    # The home line went from -3 to -4.5 by 90 minutes out; the later capture at 30 minutes is not used.
    assert pm._moved_side(event, _track((300, -3.0, 0.50), (90, -4.5, 0.50), (30, -2.0, 0.50)), "SPREAD") == ("H", -4.5, "LINE")
    assert pm._moved_side(event, _track((300, -3.0, 0.50), (90, -2.0, 0.50)), "SPREAD") == ("A", 2.0, "LINE")
    assert pm._moved_side(event, _track((300, 150.5, 0.50), (90, 152.0, 0.48)), "TOTAL") == ("Over", 152.0, "LINE")
    assert pm._moved_side(event, _track((300, 150.5, 0.50), (90, 149.0, 0.52)), "TOTAL") == ("Under", 149.0, "LINE")
    # The line held: the price decides, when it shifted enough.
    assert pm._moved_side(event, _track((300, -3.0, 0.50), (90, -3.0, 0.52)), "SPREAD") == ("H", -3.0, "PRICE")
    assert pm._moved_side(event, _track((300, 150.5, 0.52), (90, 150.5, 0.50)), "TOTAL") == ("Under", 150.5, "PRICE")
    assert pm._moved_side(event, _track((300, -3.0, 0.50), (90, -3.0, 0.503)), "SPREAD") is None
    # One capture before the cutoff, or none: nothing was known in time.
    assert pm._moved_side(event, _track((300, -3.0, 0.50), (30, -5.0, 0.50)), "SPREAD") is None
    assert pm._moved_side(event, [], "SPREAD") is None


def _score(event_id, home, away, version=1):
    return {"event_id": event_id, "result_id": f"r-{event_id}-{version}", "home_score": home, "away_score": away,
            "grading_version": version, "available_at": "2030-01-02T04:00:00+00:00"}


def test_each_night_records_how_the_moved_toward_side_finished(monkeypatch):
    later = START + timedelta(days=1)
    events = {e["event_id"]: e for e in (_event("line-win"), _event("price-loss"), _event("flat"), _event("ungraded"),
                                         _event("push"), _event("next-night", later))}
    tracks = {
        "line-win": _track((300, -3.0, 0.50), (90, -4.5, 0.50)),       # toward H at -4.5; H wins by 10
        "price-loss": _track((300, -3.0, 0.50), (90, -3.0, 0.53)),     # toward H at -3; H wins by 1
        "flat": _track((300, -3.0, 0.50), (90, -3.0, 0.50)),
        "ungraded": _track((300, -3.0, 0.50), (90, -5.0, 0.50)),
        "push": _track((300, -3.0, 0.50), (90, -4.0, 0.50)),           # toward H at -4; H wins by 4
        "next-night": [(later - timedelta(minutes=m), line, 0.5) for m, line in ((300, 2.0), (90, 4.0))],  # toward A +... 
    }
    scores = [_score("line-win", 80, 70), _score("price-loss", 71, 70), _score("flat", 80, 70), _score("push", 74, 70),
              _score("next-night", 70, 80)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, [], scores))
    monkeypatch.setattr(pm, "_main_lines", lambda event, quotes, family: tracks[event["event_id"]])
    out = pm.movement_results("db", "NCAAB", "SPREAD", now=later + timedelta(hours=6))
    first, second = out["nights"]
    assert (first["night"], second["night"]) == ("2030-01-01", "2030-01-02")
    assert (first["games_started"], first["no_move"], first["ungraded"]) == (5, 1, 1)
    assert (first["wins"], first["losses"], first["pushes"]) == (1, 1, 1)
    assert first["line_moves"] == [1, 0] and first["price_moves"] == [0, 1]
    assert first["roi_per_bet"] == pytest.approx((100 / 110 - 1 + 0) / 3)
    # The home line rose from 2 to 4: toward the away side, which reads the line as -4 and won by 10.
    assert (second["wins"], second["losses"]) == (1, 0)
    total = out["total"]
    assert (total["games_started"], total["wins"], total["losses"], total["pushes"]) == (6, 2, 1, 1)
    assert total["line_moves"] == [2, 0] and out["cutoff_minutes"] == pm.MOVE_CUTOFF_MINUTES
    assert not {"movement_results", "_moved_side"} & set(pm.PREDICTION_FUNCTIONS)


def test_the_report_prints_one_row_per_market_with_the_nights_and_the_total(tmp_path, monkeypatch):
    from scripts import market_readiness_report as report
    night = dict(games_started=5, no_move=1, ungraded=1, wins=2, losses=1, pushes=0, line_moves=[2, 0],
                 price_moves=[0, 1], roi_per_bet=0.27, roi_se=0.5)
    row = dict(sport="NCAAB", market_family="SPREAD", cutoff_minutes=60,
               nights=[dict(night, night="2030-01-01"), dict(night, night="2030-01-02", pushes=1)],
               total=dict(night, games_started=10, no_move=2, wins=4, losses=2, pushes=1, line_moves=[4, 0],
                          price_moves=[0, 2]))
    text = "\n".join(report._movement_results_markdown([row, dict(sport="NBA", market_family="TOTAL",
                                                                   evaluation_error="KeyError: x")]))
    assert "| NCAAB SPREAD | 2-1 | 2-1 /1 | 4-2 /1 | 67% | +27.0% (50.0%) | 2 / 10 | 4-0, 0-2 |" in text
    assert "2030-01-01 | 2030-01-02" in text and "| NBA TOTAL | KeyError: x |" in text
    assert report._movement_results_markdown([dict(sport="NHL", market_family="TOTAL", nights=[],
                                                    total=dict(night, games_started=0))]) == []
    store = tmp_path / "s.sqlite3"
    assert report.movement_results(store) == []                      # no store, no section
    store.write_bytes(b"")
    monkeypatch.setattr(pm, "movement_results", lambda path, sport, family, **k: dict(sport=sport, market_family=family))
    assert len(report.movement_results(store, extra={"NHL": ("TOTAL",)})) == 5
