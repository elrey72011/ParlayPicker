"""Capture pace: how many first-edge bets survive a slower poll (owner question 2026-11-09). SYNTHETIC quotes."""
import pytest
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


def test_the_nightly_record_counts_each_et_slate_night_by_itself(monkeypatch):
    events, quotes = _scope([("a", 60, 20), ("b", 60, 20), ("none", 0, 0)])
    late, found = _game("late", 60, 20)                         # same games a night earlier: shift every time back a day
    day = timedelta(days=1)
    shift = lambda text: (datetime.fromisoformat(text) - day).isoformat()  # noqa: E731
    late = dict(late, scheduled_start=shift(late["scheduled_start"]), observed_at=shift(late["observed_at"]))
    found = [dict(q, **{k: shift(q[k]) for k in q if isinstance(q[k], str) and q[k][:4] in ("2029", "2030") and "T" in q[k]})
             for q in found]
    events["late"] = late
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes + found, []))
    out = pm.nightly_record("db", "NCAAB", "SPREAD", now=NOW)
    assert [n["night"] for n in out["nights"]] == ["2029-12-31", "2030-01-01"]   # 8 PM UTC is 3 PM ET the same day
    assert [n["games_started"] for n in out["nights"]] == [1, 3]
    assert [n["bets"]["2%"] for n in out["nights"]] == [1, 2] and out["nights"][1]["captures_per_game"] == 37
    assert pm.slate_night({"scheduled_start": "2030-01-02T04:30:00+00:00"}).isoformat() == "2030-01-01"   # 11:30 PM ET


def _comparison(captured, by_pace, captures=36):
    paces = [{"pace": "as captured", "captures_per_game": captures, "bets": {"1%": {"mean": captured}}}]
    paces += [{"pace": f"{m} min", "captures_per_game": 1, "bets": {"1%": {"mean": n}}} for m, n in by_pace.items()]
    return {"paces": paces}


def test_the_recommended_pace_is_the_slowest_that_keeps_85_percent_and_needs_dense_captures():
    rows = [_comparison(10, {15: 10, 30: 9, 60: 6}), _comparison(10, {15: 10, 30: 8.5, 60: 7})]
    out = pm.recommended_pace(rows)
    assert out["status"] == "RECOMMENDED" and out["minutes"] == 30 and out["kept"]["60 min"] == 0.65
    assert pm.recommended_pace([_comparison(20, {15: 16, 30: 12, 60: 8})])["status"] == "KEEP_CURRENT"
    few = pm.recommended_pace([_comparison(4, {15: 4, 30: 4, 60: 4})])
    assert few["status"] == "NOT_ENOUGH_BETS" and few["minutes"] is None
    # captures already thinned to 30 minutes say nothing about 15: no ratchet to ever slower polls
    sparse = pm.recommended_pace([_comparison(20, {15: 20, 30: 20, 60: 19}, captures=6)])
    assert sparse["status"] == "NEEDS_DENSE_CAPTURES" and sparse["minutes"] is None
    assert pm.recommended_pace([])["minutes"] is None


def _close(event_id, selection, line, american, minutes_before, book="novig", verified=1):
    decimal = 1 + (american / 100 if american > 0 else 100 / -american)
    return {"close_id": f"c-{event_id}-{selection}-{line}-{minutes_before}", "event_id": event_id, "selection": selection,
            "sportsbook": book, "line": line, "decimal_odds": decimal, "close_verified": verified,
            "close_timestamp": (START - timedelta(minutes=minutes_before)).isoformat()}


def test_replay_clv_measures_first_edge_bets_against_the_comparable_certified_close(monkeypatch):
    # Owner 2026-11-10: bets are volume, not edge. Do they beat the close at the same book, selection and line?
    games = [("beat", 60, 20), ("lost", 60, 20), ("moved", 60, 20), ("unfetched", 60, 20), ("none", 0, 0)]
    events, quotes = _scope(games)
    closes = [
        _close("beat", "H", -2.5, -110, 1),            # bet at +106, closed at -110: beat the close
        _close("beat", "H", -2.5, -200, 300),          # a close from before the bet's quote never counts
        _close("lost", "H", -2.5, +120, 1),            # closed at a better price than the bet
        _close("moved", "H", -3.5, -110, 1),           # the market moved off the bet's line: fetched, not comparable
        _close("moved", "H", -2.5, -110, 1, book="fanduel"),            # another book's close is not comparable
        _close("unfetched", "H", -2.5, -110, 1, verified=0),            # an unverified close is no close
        _close("none", "H", -2.5, -110, 1),
    ]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    monkeypatch.setattr(pm.evidence, "read_records", lambda path, table, **k: closes if table == "prospective_close" else [])
    out = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW)
    assert out["status"] == "MEASURED" and out["games_started"] == 5
    found = out["by_edge"]["2%"]
    assert (found["bets"], found["close_fetched"], found["comparable"]) == (4, 3, 2) and found["coverage"] == 0.5
    assert found["not_comparable"] == {"CLOSE_NOT_FETCHED": 1, "BOOK_NOT_IN_CLOSE": 0, "LINE_MOVED": 1,
                                       "CLOSE_BEFORE_BET": 0}
    beat, lost = 2.06 / (1 + 100 / 110) - 1, 2.06 / 2.2 - 1
    assert found["mean_price_clv"] == pytest.approx((beat + lost) / 2) and found["beat_close_share"] == 0.5
    assert found["price_clv_se"] == pytest.approx(abs(beat - lost) / 2) and found["t"] == pytest.approx(
        found["mean_price_clv"] / found["price_clv_se"])
    # the same night with other closes: only a close before the bet, and only another book's close
    closes[:] = [_close("beat", "H", -2.5, -200, 300), _close("lost", "H", -2.5, -110, 1, book="fanduel")]
    other = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW)["by_edge"]["2%"]
    assert other["comparable"] == 0 and other["mean_price_clv"] is None and other["not_comparable"] == {
        "CLOSE_NOT_FETCHED": 2, "BOOK_NOT_IN_CLOSE": 1, "LINE_MOVED": 0, "CLOSE_BEFORE_BET": 1}
    bets = pm.first_edge_bets(events, quotes, NOW, 0.02)
    assert {(e, b["sportsbook"], b["selection"], b["line"]) for e, b, _ in bets} == {
        (g, "novig", "H", -2.5) for g in ("beat", "lost", "moved", "unfetched")}
    assert all(ev >= 0.02 for _, _, ev in bets)

    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: ({}, [], []))
    empty = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW)
    assert empty["status"] == "NO_GAMES" and empty["by_edge"] == {}


def test_replay_clv_is_outside_the_functions_a_running_plan_hashes():
    assert "replay_clv" not in pm.PREDICTION_FUNCTIONS and "first_edge_bets" not in pm.PREDICTION_FUNCTIONS
    # No literal hash here: code_hash() goes through ast.dump, whose text differs between Python versions.
    assert len(pm.code_hash()) == 64 and pm.code_hash() == pm.code_hash()


def test_a_later_entry_only_bets_on_captures_close_to_the_start(monkeypatch):
    # Owner 2026-11-10: a bet taken early often sits at a line the book later leaves.
    games = [("early", 150, 20), ("late", 45, 20), ("spanning", 70, 30), ("none", 0, 0)]
    events, quotes = _scope(games)
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    monkeypatch.setattr(pm.evidence, "read_records", lambda path, table, **k: [])
    any_time = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW)
    late = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW, entry_within_minutes=pm.LATE_ENTRY_MINUTES)
    assert any_time["entry_within_minutes"] is None and any_time["by_edge"]["2%"]["bets"] == 3
    assert late["entry_within_minutes"] == 60 and late["by_edge"]["2%"]["bets"] == 2      # the early gap is gone
    slim = pm.slim_sources(quotes)
    kept = pm.late_captures(events, slim, 60)
    assert kept and all(timedelta(0) <= START - pm.capture_time(events[q["event_id"]], q) <= timedelta(minutes=60)
                        for q in kept)
    assert {e for e, _, _ in pm.first_edge_bets(events, kept, NOW, 0.02)} == {"late", "spanning"}
    assert pm.late_captures({}, slim, 60) == []



def test_replay_clv_splits_the_bets_by_the_stake_book_they_were_at(monkeypatch):
    # Owner 2026-11-11: the stake is placed at Novig, so the split by book matters before a freeze.
    def game(event_id, book):
        event = {"event_id": event_id, "home_team": "H", "away_team": "A", "observed_at": FIRST.isoformat(),
                 "scheduled_start": START.isoformat()}
        received = START - timedelta(minutes=45)
        stamp = (received - timedelta(seconds=30)).isoformat()
        quotes = (both("pinnacle", -2.5, -110, -110, received.isoformat(), stamp, event_id)
                  + both(book, -2.5, +106, -110, received.isoformat(), stamp, event_id))
        return event, quotes
    events, quotes = {}, []
    for event_id, book in (("n1", "novig"), ("n2", "novig"), ("d1", "draftkings")):
        events[event_id], found = game(event_id, book)
        quotes += found
    closes = [_close("n1", "H", -2.5, -110, 1, book="novig"), _close("d1", "H", -2.5, +120, 1, book="draftkings")]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    monkeypatch.setattr(pm.evidence, "read_records", lambda path, table, **k: closes if table == "prospective_close" else [])
    found = pm.replay_clv("db", "NCAAB", "SPREAD", now=NOW, books=pm.PLAN_BOOKS)["by_edge"]["2%"]
    assert found["bets"] == 3 and found["comparable"] == 2
    assert set(found["by_book"]) == {"Novig", "DraftKings"}                      # labels, not provider keys
    novig, draftkings = found["by_book"]["Novig"], found["by_book"]["DraftKings"]
    assert (novig["bets"], novig["close_fetched"], novig["comparable"]) == (2, 1, 1) and novig["coverage"] == 0.5
    assert novig["mean_price_clv"] == pytest.approx(2.06 / (1 + 100 / 110) - 1) and novig["beat_close_share"] == 1.0
    assert (draftkings["bets"], draftkings["comparable"]) == (1, 1) and draftkings["mean_price_clv"] < 0
    assert sum(b["bets"] for b in found["by_book"].values()) == found["bets"]
