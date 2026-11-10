"""Capture pace: how many first-edge bets survive a slower poll (owner question 2026-11-09). SYNTHETIC quotes."""
from datetime import datetime, timedelta, timezone

from app_core import pinnacle_model as pm
from tests.test_pinnacle_plan_selection import both

START = datetime(2030, 1, 1, 20, 0, tzinfo=timezone.utc)
NOW = START + timedelta(hours=1)
FIRST = START - timedelta(hours=8)


def _game(event_id, gap_at, gap_minutes):
    """A game captured every 5 minutes for its last 3 hours; one stake book is 3% better than
    Pinnacle's fair price from ``gap_at`` minutes before the start, for ``gap_minutes``."""
    event = {"event_id": event_id, "home_team": "H", "away_team": "A", "observed_at": FIRST.isoformat(),
             "scheduled_start": START.isoformat()}
    quotes = both("pinnacle", -2.5, -110, -110, stamp=(FIRST - timedelta(minutes=2)).isoformat(), event=event_id)
    quotes += both("novig", -2.5, -110, -110, stamp=(FIRST - timedelta(minutes=2)).isoformat(), event=event_id)
    for minutes in range(180, 0, -5):
        received = START - timedelta(minutes=minutes)
        stamp = (received - timedelta(seconds=30)).isoformat()
        open_gap = gap_at >= minutes > gap_at - gap_minutes
        quotes += both("pinnacle", -2.5, -110, -110, received.isoformat(), stamp, event_id)
        quotes += both("novig", -2.5, +106 if open_gap else -110, -110, received.isoformat(), stamp, event_id)
    return event, quotes


def _scope(games):
    events, quotes = {}, []
    for event_id, gap_at, gap_minutes in games:
        event, found = _game(event_id, gap_at, gap_minutes)
        events[event_id] = event
        quotes += found
    return events, quotes


def test_thinning_keeps_the_first_capture_and_whole_later_captures_at_the_pace():
    events, quotes = _scope([("g", 60, 10)])
    times = lambda sample: sorted({pm.capture_time(events["g"], q) for q in sample})  # noqa: E731
    assert len(times(quotes)) == 37                                            # the first capture and 36 five-minute polls
    thin = pm.thin_captures(events, quotes, 15)
    kept = times(thin)
    assert kept[0] == FIRST and len(kept) == 13                                # every third poll
    assert all(b - a >= timedelta(minutes=15) for a, b in zip(kept[1:], kept[2:]))
    per_capture = {}
    for q in thin:
        per_capture.setdefault(pm.capture_time(events["g"], q), set()).add((q["sportsbook"], q["selection"]))
    assert all(len(found) == 4 for found in per_capture.values())              # both books, both sides, together
    shifted = times(pm.thin_captures(events, quotes, 15, offset=5))
    assert shifted[0] == FIRST and shifted[1] != kept[1]                       # another alignment sees other polls
    assert len(times(pm.thin_captures(events, quotes, 60))) == 4 and pm.thin_captures({}, quotes, 15) == []


def test_slower_paces_find_fewer_of_the_short_gaps(monkeypatch):
    games = [("long", 120, 60), ("mid", 90, 20), ("short-a", 60, 10), ("short-b", 45, 10), ("none", 0, 0)]
    events, quotes = _scope(games)
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    out = pm.pace_comparison("db", "NCAAB", "SPREAD", now=NOW)
    assert out["status"] == "COMPARED" and out["games_started"] == 5
    by_pace = {p["pace"]: p for p in out["paces"]}
    assert list(by_pace) == ["as captured", "15 min", "30 min", "60 min"]
    assert by_pace["as captured"]["bets"]["1%"] == {"mean": 4, "low": 4, "high": 4}   # every gap is seen at 5 minutes
    assert by_pace["as captured"]["captures_per_game"] == 37
    means = [by_pace[label]["bets"]["1%"]["mean"] for label in ("as captured", "15 min", "30 min", "60 min")]
    assert means == sorted(means, reverse=True) and means[-1] < means[0]       # never more bets at a slower pace
    assert by_pace["60 min"]["bets"]["1%"]["high"] <= 2 and by_pace["60 min"]["bets"]["1%"]["low"] >= 1   # the hour-long gap is always seen
    assert by_pace["15 min"]["captures_per_game"] == 13 and by_pace["60 min"]["captures_per_game"] == 4
    assert by_pace["15 min"]["bets"]["2%"]["mean"] <= by_pace["as captured"]["bets"]["2%"]["mean"]
    later = pm.pace_comparison("db", "NCAAB", "SPREAD", now=NOW + timedelta(days=3))
    assert later["status"] == "NO_GAMES" and later["paces"] == []


def test_readiness_report_prints_the_pace_table_and_reports_errors(tmp_path, monkeypatch):
    from scripts import market_readiness_report as mr
    assert mr.pace_comparison(tmp_path / "SYNTHETIC-absent.sqlite3") == []
    store = tmp_path / "SYNTHETIC-store.sqlite3"
    store.write_bytes(b"")

    def compare(path, sport, family, *, now, books):
        if sport == "NBA":
            raise ValueError("SYNTHETIC_FAILURE")
        return {"sport": sport, "market_family": family, "games_started": 20, "status": "COMPARED", "paces": [
            {"pace": "as captured", "captures_per_game": 30, "bets": {"1%": {"mean": 9, "low": 9, "high": 9},
                                                                       "2%": {"mean": 4, "low": 4, "high": 4}}},
            {"pace": "15 min", "captures_per_game": 11, "bets": {"1%": {"mean": 6.3, "low": 5, "high": 8},
                                                                  "2%": {"mean": 3, "low": 3, "high": 3}}}]}
    monkeypatch.setattr(pm, "pace_comparison", compare)
    rows = mr.pace_comparison(store, extra=mr.EXTRA_REPLAY_SCOPES)
    assert len(rows) == 6 and sum("evaluation_error" in r for r in rows) == 2
    text = "\n".join(mr._pace_markdown(rows))
    assert "| NCAAB SPREAD | 20 | 9 / 4 [30] | 6.3 (5-8) / 3 [11] |  |  |" in text
    assert "| NHL PUCK_LINE | 20 |" in text and "SYNTHETIC_FAILURE" in text
    assert mr._pace_markdown([]) == []


def test_one_game_at_a_time_and_slim_sources_change_no_entry_or_capture_time():
    """The speed-ups are exact: same entry quote per game under both rules, same capture times."""
    import json
    events, quotes = _scope([("long", 120, 60), ("mid", 90, 20), ("short-a", 60, 10), ("none", 0, 0)])
    stored = [dict(q, raw_source=json.dumps(dict(q["raw_source"], event_response={"big": "x" * 500}))
                   if "response_received_at" in q["raw_source"] else json.dumps(q["raw_source"])) for q in quotes]
    slim = pm.slim_sources(stored)
    assert [pm.capture_time(events[q["event_id"]], q) for q in slim] == \
        [pm.capture_time(events[q["event_id"]], q) for q in stored]
    assert all(q["raw_source"] in ({}, {"event_response": None, "market": None,
                                        "response_received_at": q["raw_source"].get("response_received_at")}) for q in slim)
    assert stored[0]["raw_source"] != slim[0]["raw_source"] and isinstance(stored[-1]["raw_source"], str)   # copies
    key = lambda chosen: {e: (q["sportsbook"], q["selection"], q["line"], q["quote_timestamp"]) for e, q in chosen.items()}  # noqa: E731
    for rules in ({}, {"entry_rule": pm.FIRST_EDGE_CAPTURE, "min_edge": 0.01},
                  {"entry_rule": pm.FIRST_EDGE_CAPTURE, "min_edge": 0.02, "books": frozenset({"Novig"})}):
        whole = pm.plan_quotes(events, stored, NOW, **rules)
        assert key(pm.plan_quotes_by_event(events, stored, NOW, **rules)) == key(whole) and len(whole) == 4
        assert key(pm.plan_quotes_by_event(events, pm.by_event(slim), NOW, **rules)) == key(whole)
    assert pm.plan_quotes_by_event({"absent": events["long"]}, stored, NOW) == {}
