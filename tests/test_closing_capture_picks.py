"""Closing prices for the board's saved best picks. SYNTHETIC games, quotes and clocks; no network.

Until 2026-11-19 no close was ever saved: the capture looked for a pick's game by the saved game key
(date and normalised names), which the live odds frame never uses (it keys a game by the provider's two
team names). These tests hold the capture to the provider's own event id, to the saved best pick only,
and to the loop's window, grid and credit cap.
"""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from activation_fixture import NOW
from app_core import activation_closing as ac

ROOT = Path(__file__).resolve().parents[1]
GAME = "2026-09-14|texas|southern utah"                 # how a saved pick names its game
LIVE_KEY = "Southern Utah Thunderbirds_Texas Longhorns"  # how the live odds frame names the same game
EVENT = "3f1c0e5a9b7d4c2e8a6f1b0d9c8e7a6b"              # the provider's event id, on both
START = NOW + timedelta(minutes=10)

spec = importlib.util.spec_from_file_location("closing_capture_loop_picks", ROOT / "scripts" / "closing_capture_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def _candidates(start=START, game=GAME, event=EVENT, pick="total-under", prefix="", **changes):
    """The four candidates of one game as the app saves them, one marked as the best pick."""
    base = dict(game_id=game, matchup_id=game, sport="NCAAB", league="NCAAB", home_team="Texas", away_team="Southern Utah",
                game_date=start.date().isoformat(), game_start_utc=start.isoformat(), quote_binding_verified=True,
                odds_recorded_at=(start - timedelta(hours=5)).isoformat(), provider_namespace="odds_api",
                provider_event_id=event, best_available_candidate_count=4, wager_approved=False)
    rows = [dict(base, candidate_id="spread-home", market_type="spread_home", best_pick="Texas -33.5", spread_line=-33.5,
                 market_line_used=-33.5, odds_american=-110, quote_bookmaker="DraftKings"),
            dict(base, candidate_id="spread-away", market_type="spread_away", best_pick="Southern Utah +33.5", spread_line=33.5,
                 market_line_used=33.5, odds_american=-110, quote_bookmaker="DraftKings"),
            dict(base, candidate_id="total-over", market_type="total_over", best_pick="Over 156.5", total_line=156.5,
                 market_line_used=156.5, odds_american=-108, quote_bookmaker="Novig"),
            dict(base, candidate_id="total-under", market_type="total_under", best_pick="Under 156.5", total_line=156.5,
                 market_line_used=156.5, odds_american=-106, quote_bookmaker="Novig")]
    for row in rows:
        row["best_available_selected"] = row["selected_as_best_pick"] = row["candidate_id"] == pick
        row["candidate_id"] = prefix + row["candidate_id"]
        row.update(changes)
    return rows


def _quote(market, point, price, book, *, event=EVENT, recorded=START - timedelta(minutes=12), namespace="odds_api"):
    return dict(provider_namespace=namespace, provider_event_id=event, book=book, market_type=market, point=point,
                price=price, recorded_at=recorded.isoformat() if recorded else None)


def _live(*, key=LIVE_KEY, game_id=EVENT, feed="the_odds_api", quotes=None):
    """The live odds frame: one row a game, keyed by the provider's names, carrying the provider's event id."""
    quotes = quotes if quotes is not None else [
        _quote("total_under", 155.5, -112, "novig"), _quote("total_over", 155.5, -108, "novig"),
        _quote("total_under", 155.5, -110, "draftkings"), _quote("spread_home", -34.5, -110, "draftkings"),
        _quote("spread_away", 34.5, -110, "draftkings"), _quote("moneyline_home", None, -5000, "novig")]
    other = [_quote("total_under", 140.5, -110, "novig", event="another-event")]
    return pd.DataFrame([
        {"matchup_id": "Duke Blue Devils_Kansas Jayhawks", "game_id": "another-event", "league": "NCAAB",
         "odds_feed_source": "the_odds_api", "provider_quotes": json.dumps(other)},
        {"matchup_id": key, "game_id": game_id, "league": "NCAAB", "odds_feed_source": feed,
         "provider_quotes": json.dumps(quotes)}])


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """One run saved through capture_run and read back by the real reader."""
    from app_core import prediction_evidence as pe
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(pe, "now_utc", lambda: (NOW - timedelta(hours=4)).isoformat())
    database = tmp_path / "evidence.db"
    context = pe.begin_run({}, path=database, root=root)
    audit = pd.DataFrame(_candidates())
    pe.capture_run(context, audit, audit[audit.best_available_selected], audit, path=database, authoritative_candidates=True)
    return database


def _snapshots(monkeypatch, *runs):
    """Saved runs, oldest first, as load_snapshots yields them."""
    from app_core import prediction_evidence as pe
    frames = [(name, pd.DataFrame([dict(row, snapshot_id=name) for row in rows]), pd.DataFrame()) for name, rows in runs]
    monkeypatch.setattr(pe, "load_snapshots", lambda _database: frames)


# The game is found by the provider's event --------------------------------------------------------

def test_a_saved_best_pick_gets_its_close_although_the_two_game_keys_differ(saved_run):
    from core.clv import line_clv, price_clv
    asked = []
    out = ac.capture_live(saved_run, fetch=lambda sports: asked.append(sports) or _live(), now=NOW)
    assert out == {"verified": 1, "unavailable": 0, "reasons": {}}
    assert asked == [["NCAAB"]]
    stored, = ac.observations(saved_run)           # the best pick only, not the game's other three candidates
    assert stored["candidate_id"] == "total-under"
    assert stored["quote"]["sportsbook"] == "Novig" and stored["quote"]["provider_event_id"] == EVENT
    assert (stored["quote"]["line"], stored["quote"]["price"]) == (155.5, -112)
    assert stored["line_clv"] == line_clv("total_under", 156.5, 155.5) == 1.0   # Under 156.5 held; it closed 155.5
    assert stored["price_clv"] is None and stored["beat_close"] is True          # prices compare only at one line
    assert price_clv(-106, -112) is not None


@pytest.mark.parametrize("live", [
    dict(game_id="some-other-event"),                 # a row of another game whose quotes name the pick's event
    dict(game_id=None),                               # a row with no event id of its own
    dict(feed="espn_ncaaf_fcs_scoreboard"),           # the same id under another feed
])
def test_a_row_that_is_not_the_picks_own_event_is_never_read(saved_run, live):
    detail = {}
    out = ac.capture_live(saved_run, fetch=lambda sports: _live(**live), now=NOW, unmatched=detail)
    assert out == {"verified": 0, "unavailable": 1, "reasons": {"missing_or_ambiguous_exact_quote": 1}}
    assert detail == {"game_not_in_live_odds": 1}
    assert ac.observations(saved_run) == []


def test_the_detail_names_the_book_when_the_event_is_there_but_the_book_is_not(saved_run):
    detail = {}
    quotes = [_quote("total_under", 155.5, -110, "draftkings")]
    ac.capture_live(saved_run, fetch=lambda sports: _live(quotes=quotes), now=NOW, unmatched=detail)
    assert detail == {"book_not_quoting_this_game": 1}


# Which saved rows get a close ----------------------------------------------------------------------

def test_each_game_gets_one_close_for_the_newest_runs_best_pick(tmp_path, monkeypatch):
    other = dict(game="2026-09-14|duke|kansas", event="event-b", prefix="b-")
    _snapshots(monkeypatch,
               ("run-1", _candidates(pick="spread-home") + _candidates(pick="total-over", **other)),
               ("run-2", _candidates(pick="total-under") + _candidates(pick="none", **other)))   # no pick for game b now
    quotes_b = [_quote(market, point, -110, book, event="event-b") for market, point, book in
                (("total_over", 150.5, "novig"), ("total_under", 150.5, "novig"),
                 ("spread_home", -30.5, "draftkings"), ("spread_away", 30.5, "draftkings"))]
    live = pd.concat([_live(), _live(key="Duke_Kansas", game_id="event-b", quotes=quotes_b).tail(1)], ignore_index=True)
    database = tmp_path / "evidence.db"
    assert ac.capture_live(database, fetch=lambda sports: live, now=NOW) == {"verified": 2, "unavailable": 0, "reasons": {}}
    assert {(o["snapshot_id"], o["candidate_id"]) for o in ac.observations(database)} == {
        ("run-2", "total-under"),      # game a: the later run changed its pick, and only that pick is read
        ("run-1", "b-total-over")}     # game b: the newest run that holds a pick for it


def test_a_game_with_no_saved_best_pick_costs_no_request(tmp_path, monkeypatch):
    _snapshots(monkeypatch, ("run-1", _candidates(pick="none")))
    out = ac.capture_live(tmp_path / "evidence.db", fetch=lambda sports: pytest.fail("no request is due"), now=NOW)
    assert out == {"verified": 0, "unavailable": 0, "reason": "no_candidates_in_closing_window"}


@pytest.mark.parametrize("minutes_to_start,within,read", [
    (10, 720, True), (20, 720, False),        # the loop's 12-minute window
    (20, 1800, True), (40, 3600, False),      # never more than the 30 minutes record_close accepts
])
def test_the_window_decides_which_games_are_read(tmp_path, monkeypatch, minutes_to_start, within, read):
    start = NOW + timedelta(minutes=minutes_to_start)
    _snapshots(monkeypatch, ("run-1", _candidates(start=start)))
    quotes = [_quote("total_under", 155.5, -112, "novig", recorded=NOW - timedelta(minutes=1))]
    asked = []
    out = ac.capture_live(tmp_path / "evidence.db", fetch=lambda sports: asked.append(sports) or _live(quotes=quotes),
                          now=NOW, within_seconds=within)
    assert (out.get("verified"), len(asked)) == ((1, 1) if read else (0, 0))


# When and why a found quote is refused -------------------------------------------------------------

def test_a_quote_updated_while_the_job_was_working_is_still_a_close(tmp_path, monkeypatch):
    """The capture time is read after the answer; read before, such a quote looked recorded after its capture."""
    start = datetime.now(timezone.utc) + timedelta(minutes=10)
    _snapshots(monkeypatch, ("run-1", _candidates(start=start)))

    def fetch(sports):
        return _live(quotes=[_quote("total_under", 155.5, -112, "novig", recorded=datetime.now(timezone.utc))])
    database = tmp_path / "evidence.db"
    assert ac.capture_live(database, fetch=fetch) == {"verified": 1, "unavailable": 0, "reasons": {}}
    stored, = ac.observations(database)
    assert stored["quote"]["quote_recorded_at"] <= stored["closing_capture_at"] < start.isoformat()


@pytest.mark.parametrize("quote,candidate,why", [
    (dict(recorded=START - timedelta(minutes=31)), {}, "refused_quote_older_than_30_minutes_before_start"),
    (dict(recorded=NOW + timedelta(minutes=1)), {}, "refused_quote_recorded_after_capture"),
    (dict(recorded=None), {}, "refused_quote_has_no_time"),
    (dict(namespace="another_feed"), dict(provider_namespace="another_feed"),
     "refused_closing_provider_namespace_missing_or_different"),
    (dict(point=None), {}, "refused_invalid_close"),
])
def test_the_detail_says_which_rule_refused_a_found_quote(tmp_path, monkeypatch, quote, candidate, why):
    _snapshots(monkeypatch, ("run-1", _candidates(**candidate)))
    found = _quote("total_under", **{"point": 155.5, "price": -112, "book": "novig", **quote})
    feed = quote.get("namespace", "the_odds_api")
    detail, database = {}, tmp_path / "evidence.db"
    out = ac.capture_live(database, fetch=lambda sports: _live(quotes=[found], feed=feed), now=NOW, unmatched=detail)
    assert out == {"verified": 0, "unavailable": 1, "reasons": {"invalid_or_stale_close": 1}}
    assert detail == {why: 1}


def test_a_pick_saved_without_market_line_used_is_compared_at_its_bound_line(tmp_path):
    candidate = dict(_candidates(pick="total-over")[2], snapshot_id="run-1")
    del candidate["market_line_used"]
    quote = dict(game_id=GAME, sport="NCAAB", market_type="total_over", sportsbook="Novig", provider_namespace="odds_api",
                 provider_event_id=EVENT, quote_recorded_at=NOW.isoformat(), line=157.5, price=-110)
    database = tmp_path / "evidence.db"
    ac.record_close(database, candidate, quote, captured_at=NOW.isoformat())
    ac.record_close(database, {k: v for k, v in candidate.items() if k != "total_line"}, dict(quote, line=158.5),
                    captured_at=NOW.isoformat())
    first, second = sorted(ac.observations(database), key=lambda o: o["quote"]["line"])
    assert first["line_clv"] == 1.0 and first["beat_close"] is True     # Over 156.5 held; it closed 157.5
    assert second["line_clv"] is None and second["beat_close"] is None  # no saved line at all: nothing invented


# The slate loop ------------------------------------------------------------------------------------

def _clock():
    now = [0.0]
    return now, (lambda: now[0]), (lambda seconds: now.__setitem__(0, now[0] + seconds))


def test_the_loop_passes_its_window_counts_credits_and_stops_at_the_cap(capsys):
    now, clock, sleep = _clock()
    seen = []

    def capture(database, *, fetch=None, unmatched=None, within_seconds=1800):
        seen.append(within_seconds)
        fetch(["NCAAB", "NCAAF", "NHL"])
        return {"verified": 1, "unavailable": 0, "reasons": {}}
    asked = []
    totals = loop.run("db", minutes=340, interval=10, capture=capture, restore=lambda path: None, backup=lambda path: None,
                      clock=clock, sleep=sleep, window=12, max_credits=40, fetch=lambda sports: asked.append(sports))
    assert seen == [720, 720, 720] and len(asked) == 3            # 19 a cycle: 19, 38, then 57 reaches the cap of 40
    assert totals == {"cycles": 3, "verified": 3, "unavailable": 0, "backups": 3, "errors": 0, "credits_estimate": 57}
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["sports"] == ["NCAAB", "NCAAF", "NHL"] and lines[0]["credits_estimate"] == 19
    assert lines[-1] == {"stopped": "credit_cap", "credits_estimate": 57, "max_credits": 40}


def test_a_cycle_that_asks_nothing_prints_and_counts_no_credits(capsys):
    now, clock, sleep = _clock()
    totals = loop.run("db", minutes=11, interval=10, restore=lambda path: None, backup=lambda path: None, clock=clock,
                      sleep=sleep, max_credits=40, fetch=lambda sports: pytest.fail("no request is due"),
                      capture=lambda database, *, fetch=None: {"verified": 0, "reason": "no_candidates_in_closing_window"})
    assert totals == {"cycles": 2, "verified": 0, "unavailable": 0, "backups": 0, "errors": 0}
    assert "credits_estimate" not in capsys.readouterr().out


def test_cycles_keep_to_the_grid_however_long_one_takes():
    now, clock, _ = _clock()
    work, naps = iter([90, 700, 30, 0]), []

    def capture(database):
        now[0] += next(work)                 # this cycle's own work: 90 s, then a stall of 700 s, then 30 s
        return {"verified": 0}

    def sleep(seconds):
        naps.append(seconds)
        now[0] += seconds
    totals = loop.run("db", minutes=45, interval=10, capture=capture, restore=lambda path: None, backup=lambda path: None,
                      clock=clock, sleep=sleep)
    assert naps == [510, 500, 570]           # cycles start at 0, 600, 1800 and 2400 s: the stalled grid point is skipped
    assert totals["cycles"] == 4


def test_an_error_inside_a_capture_is_counted_once_and_not_run_again():
    now, clock, sleep = _clock()
    calls = []

    def capture(database, unmatched=None):
        calls.append(1)
        raise TypeError("a fault inside the capture, not a missing parameter")
    totals = loop.run("db", minutes=1, interval=5, capture=capture, restore=lambda path: None, backup=lambda path: None,
                      clock=clock, sleep=sleep)
    assert len(calls) == 1 and totals["errors"] == 1


def test_a_window_no_longer_than_the_interval_is_refused(capsys):
    with pytest.raises(SystemExit):
        loop.main(["--interval", "10", "--window", "10"])
    assert "--window must be longer than --interval" in capsys.readouterr().err


def test_the_workflow_runs_the_loop_capped_and_with_a_window_longer_than_its_interval():
    import re
    import yaml
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "closing-capture-slate.yml").read_text())
    command = next(step["run"] for step in workflow["jobs"]["slate-closes"]["steps"] if "closing_capture_loop.py" in step.get("run", ""))
    options = {name: float(value) for name, value in re.findall(r"--(interval|window|max-credits) (\d+)", command)}
    assert set(options) == {"interval", "window", "max-credits"}
    assert options["interval"] < options["window"] <= 30 and 0 < options["max-credits"] <= 300


# The provider's own count ---------------------------------------------------------------------------

def test_the_odds_client_keeps_the_providers_usage_counters(monkeypatch, tmp_path):
    import requests
    from unittest.mock import Mock
    from app_core import odds_api
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(odds_api, "last_usage", {})
    answer = requests.Response()
    answer.status_code = 200
    answer.headers.update({"x-requests-last": "3", "x-requests-used": "120", "x-requests-remaining": "880"})
    answer.json = Mock(return_value=[])
    monkeypatch.setattr(requests, "get", Mock(return_value=answer))
    odds_api.TheOddsAPIClient("PRIVATE").get_odds("basketball_ncaab")
    assert odds_api.last_usage == {"sport": "basketball_ncaab", "last": "3", "used": "120", "remaining": "880"}
