"""Market sentiment from real-money prices: the lean and its running record. SYNTHETIC quotes and scores."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import market_sentiment as ms

START = datetime(2030, 1, 2, 0, 0, tzinfo=timezone.utc)          # 7 PM ET on the slate night of 2030-01-01
NOW = START + timedelta(hours=6)


def _event(event_id, start=START):
    return {"event_id": event_id, "home_team": "H", "away_team": "A", "scheduled_start": start.isoformat(),
            "observed_at": (start - timedelta(hours=12)).isoformat()}


def _quote(event_id, book, selection, line, decimal, minutes, *, family="SPREAD", start=START, verified=True):
    captured = (start - timedelta(minutes=minutes)).isoformat()
    return {"event_id": event_id, "sportsbook": book, "market_family": family, "selection": selection, "line": line,
            "decimal_odds": decimal, "quote_timestamp": captured, "quote_verified": verified,
            "raw_source": {"event_response": None, "market": None, "response_received_at": captured}}


def _pair(event_id, book, home_line, home_decimal, away_decimal, minutes, **kw):
    return [_quote(event_id, book, "H", home_line, home_decimal, minutes, **kw),
            _quote(event_id, book, "A", -home_line, away_decimal, minutes, **kw)]


def _total(event_id, book, line, over_decimal, under_decimal, minutes, **kw):
    return [_quote(event_id, book, "Over", line, over_decimal, minutes, family="TOTAL", **kw),
            _quote(event_id, book, "Under", line, under_decimal, minutes, family="TOTAL", **kw)]


def test_the_lean_is_the_side_the_exchange_rates_higher_than_the_retail_books():
    event = _event("g")
    quotes = (_pair("g", "novig", -3.0, 1.80, 2.05, 90) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
              + _pair("g", "fanduel", -3.0, 1.95, 1.87, 90))
    found = ms.read(event, quotes, "SPREAD")
    exchange = (1 / 1.80) / (1 / 1.80 + 1 / 2.05)
    retail = (0.5 + (1 / 1.95) / (1 / 1.95 + 1 / 1.87)) / 2
    assert (found["selection"], found["line"], found["against_books"]) == ("H", -3.0, 2)
    assert found["lean_probability"] == pytest.approx(exchange)
    assert found["against_probability"] == pytest.approx(retail)
    assert found["gap"] == pytest.approx(exchange - retail)
    assert found["best_decimal"] == 1.95                       # the better of the two retail prices for H


def test_an_away_lean_is_read_at_the_away_sides_own_line():
    event = _event("g")
    quotes = _pair("g", "novig", -3.0, 2.05, 1.80, 90) + _pair("g", "betmgm", -3.0, 1.91, 1.91, 90)
    found = ms.read(event, quotes, "SPREAD")
    assert (found["selection"], found["line"], found["best_decimal"]) == ("A", 3.0, 1.91)
    assert found["gap"] < 0                                    # the gap is always the home side's


def test_totals_lean_to_the_over_or_the_under_at_the_same_line():
    event = _event("g")
    over = _total("g", "novig", 150.5, 1.80, 2.05, 90) + _total("g", "fanduel", 150.5, 1.91, 1.91, 90)
    under = _total("g", "novig", 150.5, 2.05, 1.80, 90) + _total("g", "fanduel", 150.5, 1.91, 1.91, 90)
    assert (ms.read(event, over, "TOTAL")["selection"], ms.read(event, over, "TOTAL")["line"]) == ("Over", 150.5)
    assert (ms.read(event, under, "TOTAL")["selection"], ms.read(event, under, "TOTAL")["line"]) == ("Under", 150.5)


def test_venues_that_agree_give_no_lean():
    event = _event("g")
    quotes = _pair("g", "novig", -3.0, 1.92, 1.94, 90) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    found = ms.read(event, quotes, "SPREAD")
    assert abs(found["gap"]) < ms.MIN_GAP
    assert (found["selection"], found["line"], found["best_decimal"]) == (None, None, None)


def test_only_what_was_known_an_hour_before_the_start_is_read():
    event = _event("g")
    early = _pair("g", "novig", -3.0, 2.05, 1.80, 300) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 300)
    hour_out = _pair("g", "novig", -3.0, 1.80, 2.05, 90) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    late = _pair("g", "novig", -3.0, 2.05, 1.80, 30) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 30)
    # The newest capture at least 60 minutes out decides; the one at 30 minutes is never read.
    assert ms.read(event, early + hour_out + late, "SPREAD")["selection"] == "H"
    assert ms.read(event, early + late, "SPREAD")["selection"] == "A"
    assert ms.read(event, late, "SPREAD") is None


def test_a_capture_without_a_shared_line_falls_back_to_the_one_before():
    event = _event("g")
    shared = _pair("g", "novig", -3.0, 1.80, 2.05, 300) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 300)
    apart = _pair("g", "novig", -3.5, 1.95, 1.90, 90) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    found = ms.read(event, shared + apart, "SPREAD")
    assert (found["selection"], found["home_line"]) == ("H", -3.0)
    assert found["captured"] == START - timedelta(minutes=300)
    assert ms.read(event, apart, "SPREAD") is None
    # No retail book at all, or no exchange quote: nothing to compare.
    assert ms.read(event, _pair("g", "novig", -3.0, 1.80, 2.05, 90), "SPREAD") is None
    assert ms.read(event, _pair("g", "draftkings", -3.0, 1.91, 1.91, 90), "SPREAD") is None


def test_the_main_line_is_the_one_priced_nearest_even_and_junk_pairs_are_left_out():
    event = _event("g")
    quotes = (_pair("g", "novig", -3.0, 1.90, 1.94, 90) + _pair("g", "novig", -6.5, 2.60, 1.52, 90)
              + _pair("g", "draftkings", -3.0, 1.80, 2.02, 90) + _pair("g", "draftkings", -6.5, 2.50, 1.55, 90))
    assert ms.read(event, quotes, "SPREAD")["home_line"] == -3.0
    # Both sides far worse than any real price: an alternate line nobody would take.
    junk = _pair("g", "novig", -3.0, 1.30, 1.30, 90) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    assert ms.read(event, junk, "SPREAD") is None
    one_sided = [_quote("g", "novig", "H", -3.0, 1.80, 90)] + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    assert ms.read(event, one_sided, "SPREAD") is None
    unverified = _pair("g", "novig", -3.0, 1.80, 2.05, 90, verified=False) + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90)
    assert ms.read(event, unverified, "SPREAD") is None


def test_the_reference_book_can_be_the_one_whose_lean_is_read():
    event = _event("g")
    quotes = (_pair("g", "pinnacle", -3.0, 1.85, 2.03, 90) + _pair("g", "novig", -3.0, 2.05, 1.80, 90)
              + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90))
    assert ms.read(event, quotes, "SPREAD", lean_book=ms.SHARP)["selection"] == "H"
    assert ms.read(event, quotes, "SPREAD", lean_book=ms.EXCHANGE)["selection"] == "A"
    # The retail books are compared by name however the store spells them.
    labelled = _pair("g", "Novig", -3.0, 1.80, 2.05, 90) + _pair("g", "DraftKings", -3.0, 1.91, 1.91, 90)
    assert ms.read(event, labelled, "SPREAD")["selection"] == "H"


def _score(event_id, home, away, version=1):
    return {"event_id": event_id, "result_id": f"r-{event_id}-{version}", "home_score": home, "away_score": away,
            "grading_version": version, "available_at": "2030-01-02T04:00:00+00:00"}


def test_each_night_records_how_the_leaned_side_finished(monkeypatch):
    later = START + timedelta(days=1)
    events = {e["event_id"]: e for e in (_event("win"), _event("loss"), _event("push"), _event("agree"),
                                         _event("no-quote"), _event("ungraded"), _event("next", later))}
    lean_home = lambda g, line, **kw: (_pair(g, "novig", line, 1.80, 2.05, 90, **kw)
                                       + _pair(g, "draftkings", line, 1.95, 1.87, 90, **kw))
    quotes = (lean_home("win", -3.0) + lean_home("loss", -3.0) + lean_home("push", -4.0) + lean_home("ungraded", -3.0)
              + _pair("agree", "novig", -3.0, 1.92, 1.94, 90) + _pair("agree", "draftkings", -3.0, 1.91, 1.91, 90)
              + _pair("no-quote", "draftkings", -3.0, 1.91, 1.91, 90)
              + _pair("next", "novig", -3.0, 2.05, 1.80, 90, start=later) + _pair("next", "betmgm", -3.0, 1.87, 1.95, 90, start=later))
    scores = [_score("win", 80, 70), _score("loss", 71, 70), _score("push", 74, 70), _score("agree", 80, 70),
              _score("no-quote", 80, 70), _score("next", 70, 80), _score("win", 60, 70, version=0)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    out = ms.results("db", "NCAAB", "SPREAD", now=later + timedelta(hours=6))
    first, second = out["nights"]
    assert (first["night"], second["night"]) == ("2030-01-01", "2030-01-02")
    assert (first["games_started"], first["no_quote"], first["no_lean"], first["ungraded"]) == (6, 1, 1, 1)
    assert (first["wins"], first["losses"], first["pushes"]) == (1, 1, 1)
    # Paid at a flat -110, and at the retail price for the leaned side (H at 1.95).
    assert first["roi_flat"] == pytest.approx((100 / 110 - 1 + 0) / 3)
    assert first["roi_priced"] == pytest.approx((0.95 - 1 + 0) / 3)
    assert first["priced_bets"] == 3
    assert first["strong"] == [1, 1]                            # both decided sides had a gap of a point or more
    # The next night leaned to the away side at +3, which won by ten at the retail price of 1.95.
    assert (second["wins"], second["losses"], second["roi_priced"]) == (1, 0, pytest.approx(0.95))
    total = out["total"]
    assert (total["games_started"], total["wins"], total["losses"], total["pushes"]) == (7, 2, 1, 1)
    assert (out["lean_book"], out["against"], out["cutoff_minutes"]) == ("novig", list(ms.RETAIL), 60)


def test_the_newest_final_is_the_one_graded_and_older_nights_are_left_out(monkeypatch):
    old = START - timedelta(days=30)
    events = {"g": _event("g"), "old": _event("old", old)}
    quotes = (_pair("g", "novig", -3.0, 1.80, 2.05, 90) + _pair("g", "fanduel", -3.0, 1.91, 1.91, 90)
              + _pair("old", "novig", -3.0, 1.80, 2.05, 90, start=old) + _pair("old", "fanduel", -3.0, 1.91, 1.91, 90, start=old))
    scores = [_score("g", 60, 70, version=1), _score("g", 80, 70, version=2), _score("old", 80, 70)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    out = ms.results("db", "NCAAB", "SPREAD", now=NOW)
    assert (out["total"]["games_started"], out["total"]["wins"], out["total"]["losses"]) == (1, 1, 0)
    sharp = ms.results("db", "NCAAB", "SPREAD", now=NOW, lean_book=ms.SHARP)
    assert (sharp["lean_book"], sharp["total"]["no_quote"], sharp["total"]["wins"]) == ("pinnacle", 1, 0)


def test_the_report_prints_one_row_per_market_and_venue(tmp_path, monkeypatch):
    from scripts import market_readiness_report as report
    night = dict(games_started=5, no_quote=1, no_lean=1, ungraded=0, wins=2, losses=1, pushes=0, strong=[1, 1],
                 roi_flat=0.27, roi_flat_se=0.5, roi_priced=0.30, roi_priced_se=0.52, priced_bets=3)
    row = dict(sport="NCAAB", market_family="SPREAD", lean_book="novig", against=list(ms.RETAIL), cutoff_minutes=60,
               nights=[dict(night, night="2030-01-01"), dict(night, night="2030-01-02", pushes=1)],
               total=dict(night, games_started=10, no_quote=2, no_lean=2, wins=4, losses=2, pushes=1, strong=[2, 2]))
    sharp = dict(row, lean_book="pinnacle", total=dict(row["total"], roi_priced=None, roi_priced_se=None))
    text = "\n".join(report._sentiment_results_markdown([row, sharp, dict(sport="NBA", market_family="TOTAL",
                                                                           lean_book="novig", evaluation_error="KeyError: x")]))
    assert ("| NCAAB SPREAD | Novig exchange | 2-1 | 2-1 /1 | 4-2 /1 | 67% | +27.0% (50.0%) | +30.0% (52.0%) | 2-2 | "
            "2 / 2 / 10 |") in text
    assert "| NCAAB SPREAD | Pinnacle | 2-1 | 2-1 /1 | 4-2 /1 | 67% | +27.0% (50.0%) |  | 2-2 | 2 / 2 / 10 |" in text
    assert "2030-01-01 | 2030-01-02" in text and "| NBA TOTAL | Novig exchange | KeyError: x |" in text
    assert "This READS results" in text and "not a proven edge" in text
    assert report._sentiment_results_markdown([dict(sport="NHL", market_family="TOTAL", lean_book="novig", nights=[],
                                                     total=dict(night, games_started=0))]) == []
    store = tmp_path / "s.sqlite3"
    assert report.sentiment_results(store) == []                     # no store, no section
    store.write_bytes(b"")
    seen = []
    monkeypatch.setattr(ms, "results_by_venue", lambda path, sport, family, **k: seen.append(k) or [
        dict(sport=sport, market_family=family, lean_book=book) for book in k["lean_books"]])
    assert len(report.sentiment_results(store, extra={"NHL": ("TOTAL",)})) == 10   # five markets, two venues each
    assert len(seen) == 5                                             # the store is read once per market
    assert {k["lean_books"] for k in seen} == {("novig", "pinnacle")} and {k["nights"] for k in seen} == {14}

    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(ms, "results_by_venue", boom)
    failed = report.sentiment_results(store)
    assert len(failed) == 8 and all(r["evaluation_error"].startswith("KeyError") for r in failed)


def test_nothing_that_predicts_or_stakes_reads_the_sentiment_record():
    from app_core import pinnacle_model as pm
    assert not {"results", "results_by_venue", "read", "lean", "book_fairs"} & set(pm.PREDICTION_FUNCTIONS)
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    users = [p.relative_to(root).as_posix() for folder in ("app", "app_core", "core", "scripts")
             for p in (root / folder).rglob("*.py")
             if p.name != "market_sentiment.py" and "market_sentiment" in p.read_text(encoding="utf-8")]
    assert users == ["scripts/market_readiness_report.py"]         # the read-only report, and nothing else


def test_both_venues_come_from_one_read_of_the_store(monkeypatch):
    events = {"g": _event("g")}
    quotes = (_pair("g", "pinnacle", -3.0, 1.85, 2.03, 90) + _pair("g", "novig", -3.0, 2.05, 1.80, 90)
              + _pair("g", "draftkings", -3.0, 1.91, 1.91, 90))
    reads = []
    monkeypatch.setattr("app_core.prospective_research_models._read_scope",
                        lambda *a: reads.append(a) or (events, quotes, [_score("g", 80, 70)]))
    exchange, sharp = ms.results_by_venue("db", "NCAAB", "SPREAD", now=NOW)
    assert len(reads) == 1
    assert (exchange["lean_book"], exchange["total"]["wins"], exchange["total"]["losses"]) == ("novig", 0, 1)
    assert (sharp["lean_book"], sharp["total"]["wins"], sharp["total"]["losses"]) == ("pinnacle", 1, 0)
    assert ms.results("db", "NCAAB", "SPREAD", now=NOW, lean_book=ms.SHARP) == sharp


# ---- the direction record: a base rate beside the sentiment record (2026-11-16) ----

def test_the_retail_main_line_is_the_one_most_books_quote():
    event, limit = _event("g"), START - timedelta(minutes=60)
    quotes = (_pair("g", "draftkings", -3.0, 1.91, 1.91, 90) + _pair("g", "fanduel", -3.0, 1.95, 1.87, 90)
              + _pair("g", "betmgm", -3.5, 2.00, 1.83, 90) + _pair("g", "novig", -4.0, 1.95, 1.95, 90))
    captured, line, best = ms.main_line(ms.book_fairs(event, quotes, "SPREAD", limit))
    assert (captured, line) == (START - timedelta(minutes=90), -3.0)
    assert best == {"H": 1.95, "A": 1.91}                      # the best retail price for each side at that line
    # One book at each of two lines: the one priced nearest even.
    apart = _pair("g", "draftkings", -3.0, 1.80, 2.02, 90) + _pair("g", "fanduel", -3.5, 1.91, 1.91, 90)
    assert ms.main_line(ms.book_fairs(event, apart, "SPREAD", limit))[1] == -3.5
    # A newer capture with no retail book in it falls back to the one before.
    older = _pair("g", "betmgm", -2.5, 1.91, 1.91, 300) + _pair("g", "novig", -3.0, 1.91, 1.91, 90)
    assert ms.main_line(ms.book_fairs(event, older, "SPREAD", limit))[:2] == (START - timedelta(minutes=300), -2.5)
    assert ms.main_line(ms.book_fairs(event, _pair("g", "novig", -3.0, 1.91, 1.91, 90), "SPREAD", limit)) is None
    assert ms.main_line({}) is None


def test_the_direction_record_counts_each_side_at_the_retail_main_line(monkeypatch):
    later = START + timedelta(days=1)
    events = {e["event_id"]: e for e in (_event("fav"), _event("dog"), _event("push"), _event("pickem"),
                                         _event("no-quote"), _event("ungraded"), _event("next", later))}
    retail = lambda g, line, **kw: (_pair(g, "draftkings", line, 1.95, 1.87, 90, **kw)
                                    + _pair(g, "fanduel", line, 1.91, 1.93, 90, **kw))
    quotes = (retail("fav", -3.0) + retail("dog", -3.0) + retail("push", -4.0) + retail("pickem", 0.0)
              + retail("ungraded", -3.0) + _pair("no-quote", "novig", -3.0, 1.91, 1.91, 90)
              + retail("next", 2.5, start=later))
    scores = [_score("fav", 80, 70), _score("dog", 71, 70), _score("push", 74, 70), _score("pickem", 70, 75),
              _score("no-quote", 80, 70), _score("next", 70, 71)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    away, dog = ms.direction_results("db", "NCAAB", "SPREAD", now=later + timedelta(hours=6))
    assert (away["first"], away["second"], dog["first"], dog["second"]) == ("Away", "Home", "Underdog", "Favourite")
    assert (away["books"], away["cutoff_minutes"]) == (list(ms.RETAIL), 60)

    first, second = away["nights"]
    assert (first["night"], second["night"]) == ("2030-01-01", "2030-01-02")
    assert (first["games_started"], first["no_quote"], first["ungraded"], first["no_side"]) == (6, 1, 1, 0)
    # Away covered at +3 after a one-point loss and won the pick'em; it lost when the home side won by ten.
    assert (first["wins"], first["losses"], first["pushes"]) == (2, 1, 1)
    assert first["roi_first"] == pytest.approx((-1 + 0.93 + 0 + 0.93) / 4)      # the away side's best price is 1.93
    assert first["roi_second"] == pytest.approx((0.95 - 1 + 0 - 1) / 4)         # the home side's is 1.95
    assert (second["wins"], second["losses"], second["roi_first"], second["roi_second"]) == (
        0, 1, pytest.approx(-1.0), pytest.approx(0.95))

    first, second = dog["nights"]
    # The pick'em has no favourite, so it is left out of this split.
    assert (first["games_started"], first["no_side"], first["wins"], first["losses"], first["pushes"]) == (6, 1, 1, 1, 1)
    assert first["roi_first"] == pytest.approx((-1 + 0.93 + 0) / 3)
    assert first["roi_second"] == pytest.approx((0.95 - 1 + 0) / 3)
    # The next night the home side was the underdog at +2.5 and lost by one: the underdog covered.
    assert (second["wins"], second["losses"], second["roi_first"]) == (1, 0, pytest.approx(0.95))
    assert (dog["total"]["wins"], dog["total"]["losses"], dog["total"]["pushes"], dog["total"]["games_started"]) == (2, 1, 1, 7)


def test_the_direction_record_for_a_total_is_under_against_over(monkeypatch):
    events = {"low": _event("low"), "high": _event("high"), "exact": _event("exact")}
    quotes = (_total("low", "betmgm", 150.5, 1.91, 1.95, 90) + _total("high", "betmgm", 150.5, 1.91, 1.95, 90)
              + _total("exact", "betmgm", 150.0, 1.91, 1.95, 90) + _total("exact", "betmgm", 149.5, 1.91, 1.95, 30))
    scores = [_score("low", 70, 60), _score("high", 90, 80), _score("exact", 80, 70)]
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, scores))
    out = ms.direction_results("db", "NCAAB", "TOTAL", now=NOW)
    assert [(r["first"], r["second"]) for r in out] == [("Under", "Over")]
    total = out[0]["total"]
    # The line 30 minutes out is never read: 150 an hour before the start is a push on 150 points.
    assert (total["wins"], total["losses"], total["pushes"], total["games_started"]) == (1, 1, 1, 3)
    assert total["roi_first"] == pytest.approx((0.95 - 1 + 0) / 3)
    assert total["roi_second"] == pytest.approx((-1 + 0.91 + 0) / 3)


def test_the_report_prints_the_direction_record(tmp_path, monkeypatch):
    from scripts import market_readiness_report as report
    night = dict(games_started=5, no_quote=1, ungraded=0, no_side=0, wins=3, losses=1, pushes=0,
                 roi_first=0.4, roi_first_se=0.5, roi_second=-0.5, roi_second_se=0.45)
    row = dict(sport="NCAAB", market_family="TOTAL", first="Under", second="Over", books=list(ms.RETAIL),
               cutoff_minutes=60, nights=[dict(night, night="2030-01-01"), dict(night, night="2030-01-02", pushes=1)],
               total=dict(night, games_started=10, no_quote=2, wins=6, losses=2, pushes=1))
    text = "\n".join(report._direction_results_markdown(
        [row, dict(sport="NBA", market_family="SPREAD", evaluation_error="KeyError: x")]))
    assert "| NCAAB TOTAL | Under-Over | 3-1 | 3-1 /1 | 6-2 /1 | 75% | +40.0% (50.0%) | -50.0% (45.0%) | 2 / 10 |" in text
    assert "| NBA SPREAD |  | KeyError: x |" in text
    assert "A base rate, not a signal" in text and "READS results" in text
    assert report._direction_results_markdown([dict(row, nights=[], total=dict(night, games_started=0))]) == []
    store = tmp_path / "s.sqlite3"
    assert report.direction_results(store) == []                     # no store, no section
    store.write_bytes(b"")
    seen = []
    monkeypatch.setattr(ms, "direction_results", lambda path, sport, family, **k: seen.append(k) or [
        dict(sport=sport, market_family=family)])
    assert len(report.direction_results(store, extra={"NHL": ("TOTAL",)})) == 5
    assert {k["nights"] for k in seen} == {14}

    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(ms, "direction_results", boom)
    failed = report.direction_results(store)
    assert len(failed) == 4 and all(r["evaluation_error"].startswith("KeyError") for r in failed)


def test_the_direction_record_is_outside_the_prediction_functions():
    from app_core import pinnacle_model as pm
    assert not {"direction_results", "main_line"} & set(pm.PREDICTION_FUNCTIONS)
