"""Fixed-interval price watch (roadmap M2, D28) and the junk-price filter (D29). SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import importlib

import pytest

from app_core import pinnacle_model as pm
from app_core import price_watch as pw
from app_core import prospective_evidence as pe

NOW = datetime(2026, 11, 10, 22, 0, tzinfo=timezone.utc)         # 5:00 PM ET
ON = {pw.WATCH_ENV: "nba, ncaab"}.get


@pytest.fixture
def clock(monkeypatch):
    state = [NOW]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    return state


def _event(path, sport, name, start):
    key = {"NBA": "basketball_nba", "NCAAB": "basketball_ncaab"}[sport]
    raw = {"id": name, "sport_key": key, "home_team": f"{name} Home", "away_team": f"{name} Away",
           "commence_time": start.isoformat()}
    pe.insert_event(path, {"event_id": name, "sport": sport, "game_id": name, "provider_namespace": "THE_ODDS_API",
                           "provider_event_id": name, "home_team": raw["home_team"], "away_team": raw["away_team"],
                           "home_team_id": f"{name}-h", "away_team_id": f"{name}-a",
                           "scheduled_start": start.isoformat(), "observed_at": (NOW - timedelta(hours=8)).isoformat(),
                           "source_id": f"capture-{name}", "raw_source": raw})
    return raw


class _Reply:
    status_code = 200

    def __init__(self, payload):
        self.payload, self.headers = payload, {"x-requests-last": "4"}

    def json(self):
        return self.payload


class _Provider:
    def __init__(self, clock, games):
        self.clock, self.games, self.calls = clock, games, []

    def __call__(self, url, params=None, **kwargs):
        self.calls.append((url, params))
        stamp = self.clock[0].isoformat()
        wanted = set(params["eventIds"].split(","))
        return _Reply([dict(raw, bookmakers=[{"key": book, "markets": [{"key": "spreads", "last_update": stamp, "outcomes": [
            {"name": raw["home_team"], "point": -3.5, "price": -110},
            {"name": raw["away_team"], "point": 3.5, "price": -110}]}]} for book in ("pinnacle", "novig")])
            for raw in self.games if raw["id"] in wanted and raw["sport_key"] in url])


def _poll(watcher, path, provider, budget, clock):
    return watcher.poll(path, key="k", request_get=provider, budget=budget, clock=lambda: clock[0])


def test_off_by_default_and_after_the_approved_week():
    assert pw.enabled({}.get, now=NOW) is False
    assert pw.sports(ON, now=NOW) == ("NBA", "NCAAB")
    assert pw.sports({pw.WATCH_ENV: "OFF"}.get, now=NOW) == ()
    assert pw.sports({pw.WATCH_ENV: "NHL,NBA"}.get, now=NOW) == ("NBA",)
    assert pw.enabled(ON, now=pw.UNTIL) is False


def test_watches_games_in_their_last_three_hours_at_each_sports_interval(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    games = [_event(path, "NBA", "nba-soon", NOW + timedelta(hours=2)),
             _event(path, "NBA", "nba-late", NOW + timedelta(hours=5)),
             _event(path, "NCAAB", "cbb-soon", NOW + timedelta(hours=1))]
    provider = _Provider(clock, games)
    budget = pw.budget_for({}, lambda: None, clock=lambda: clock[0])
    watcher = pw.Watcher(ON)
    first = _poll(watcher, path, provider, budget, clock)
    assert set(first["sports"]) == {"NBA", "NCAAB"} and first["capture"] == {"requested": True, "stored_quotes": 8}
    assert [p["eventIds"] for _, p in provider.calls] == ["nba-soon", "cbb-soon"]     # not the game 5 hours out
    assert all("pinnacle" in p["bookmakers"] and p["markets"] == "spreads,totals" for _, p in provider.calls)
    clock[0] = NOW + timedelta(minutes=10)
    assert _poll(watcher, path, provider, budget, clock)["capture"] is None           # neither sport is due
    clock[0] = NOW + timedelta(minutes=15)
    assert set(_poll(watcher, path, provider, budget, clock)["sports"]) == {"NBA"}    # NBA every 15 minutes
    clock[0] = NOW + timedelta(minutes=30)
    assert set(_poll(watcher, path, provider, budget, clock)["sports"]) == {"NBA", "NCAAB"}   # NCAAB every 30
    assert budget.usage("ODDS")["daily"] == 5 * 4
    clock[0] = NOW + timedelta(hours=2, minutes=1)                # both started; the later game is now inside 3 hours
    assert set(_poll(watcher, path, provider, budget, clock)["sports"]) == {"NBA"}
    assert provider.calls[-1][1]["eventIds"] == "nba-late" and len(provider.calls) == 6
    quotes = pe.read_records(path, "prospective_quote", sport="NCAAB")
    assert {q["sportsbook"] for q in quotes} == {"pinnacle", "novig"}


def test_no_game_in_the_window_spends_nothing_and_the_allowance_stops_the_watch(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    games = [_event(path, "NBA", "nba-late", NOW + timedelta(hours=5))]
    provider = _Provider(clock, games)
    budget = pw.budget_for({}, lambda: None, clock=lambda: clock[0])
    watcher = pw.Watcher(ON)
    assert _poll(watcher, path, provider, budget, clock)["capture"] is None and provider.calls == []
    clock[0] = NOW + timedelta(hours=3)
    for _ in range(pw.WATCH_LIMITS["daily"] // 4):
        budget.reserve("ODDS", 4)
    capped = _poll(watcher, path, provider, budget, clock)
    assert capped["sports"]["NBA"]["blocker"] == "api_budget:ODDS:daily" and provider.calls == []
    assert pw.WATCH_LIMITS == {"daily": 200, "rolling_31_days": 1400}                 # owner D28: 7 days, both sports


def test_the_shared_loop_runs_the_watch(tmp_path):
    loop = importlib.import_module("scripts.news_capture_loop")
    assert loop.run(1, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None, setting={}.get,
                    source=pw)["status"] == "OFF"

    class Watcher:
        def poll(self, path, **kwargs):
            assert "fetch" in kwargs
            return {"news": [], "capture": {"requested": True, "stored_quotes": 8}}

    syncs = []
    original = pw.load_state, pw.save_state
    pw.load_state, pw.save_state = (lambda client, folder: {}), (lambda *a, **k: None)
    try:
        ticks = iter([0, 400])
        result = loop.run(5, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None, setting=ON,
                          sync=lambda *a: syncs.append(1), clock=lambda: next(ticks), sleep=lambda s: None,
                          watcher=Watcher(), source=pw)
    finally:
        pw.load_state, pw.save_state = original
    assert result["polls"] == 1 and result["captures"] == 1 and result["stored_quotes"] == 8 and len(syncs) == 2


def _quote(quote_id, book, selection, line, odds, family="SPREAD", stamp="2026-11-10T22:00:00+00:00"):
    return {"quote_id": quote_id, "event_id": "g", "sportsbook": book, "selection": selection, "line": line,
            "decimal_odds": odds, "market_family": family, "quote_timestamp": stamp, "quote_verified": 1}


def test_junk_filter_drops_only_two_sided_prices_above_the_overround_limit():
    quotes = [_quote("ok-h", "novig", "HOME", -3.5, 1.95), _quote("ok-a", "novig", "AWAY", 3.5, 1.93),
              _quote("junk-h", "novig", "HOME", -9.5, 1.40), _quote("junk-a", "novig", "AWAY", 9.5, 1.40),
              _quote("alone", "novig", "HOME", -6.5, 1.20),
              _quote("edge-o", "draftkings", "OVER", 220.5, 1.80, "TOTAL"),
              _quote("edge-u", "draftkings", "UNDER", 220.5, 1.70, "TOTAL"),
              _quote("pin-h", "pinnacle", "HOME", -9.5, 1.30), _quote("pin-a", "pinnacle", "AWAY", 9.5, 1.30)]
    kept, dropped = pm.without_junk(quotes)
    assert dropped == 2 and [q["quote_id"] for q in kept] == ["ok-h", "ok-a", "alone", "edge-o", "edge-u", "pin-h", "pin-a"]
    assert 1 / 1.80 + 1 / 1.70 < pm.MAX_OVERROUND < 1 / 1.40 + 1 / 1.40
    assert pm.without_junk(quotes, max_overround=1.10)[1] == 4


def test_the_filter_does_not_touch_the_functions_a_running_plan_hashes():
    assert "without_junk" not in pm.PREDICTION_FUNCTIONS and "replay_estimate" not in pm.PREDICTION_FUNCTIONS


def test_fast_mode_is_off_unless_switched_on_and_changes_only_interval_and_allowance(tmp_path, clock):
    assert pw.fast({}.get) is False and pw.fast({pw.FAST_ENV: "yes"}.get) is False
    assert pw.interval_minutes("NBA", {}.get) == 15 and pw.interval_minutes("NCAAB", {}.get) == 30
    assert pw.limits({}.get) == pw.WATCH_LIMITS == {"daily": 200, "rolling_31_days": 1400}
    on = {pw.WATCH_ENV: "NBA,NCAAB", pw.FAST_ENV: "True"}.get
    assert pw.fast(on) and pw.interval_minutes("NBA", on) == pw.interval_minutes("NCAAB", on) == 5
    assert pw.limits(on) == {"daily": 600, "rolling_31_days": 4200}
    assert pw.enabled(on, now=pw.UNTIL) is False                                      # the end date is the same

    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    games = [_event(path, "NBA", "nba-soon", NOW + timedelta(hours=2)),
             _event(path, "NCAAB", "cbb-soon", NOW + timedelta(hours=1))]
    provider = _Provider(clock, games)
    budget = pw.budget_for({}, lambda: None, clock=lambda: clock[0], setting=on)
    watcher = pw.Watcher(on)
    for minutes in (0, 3, 5, 10):
        clock[0] = NOW + timedelta(minutes=minutes)
        _poll(watcher, path, provider, budget, clock)
    assert len(provider.calls) == 6                                                   # both sports at 0, 5 and 10; none at 3
    assert budget.usage("ODDS")["daily"] == 24
    for _ in range((600 - 24) // 4):
        budget.reserve("ODDS", 4)
    clock[0] = NOW + timedelta(minutes=15)
    capped = _poll(watcher, path, provider, budget, clock)
    assert capped["sports"]["NBA"]["blocker"] == "api_budget:ODDS:daily" and len(provider.calls) == 6


def test_the_workflow_passes_fast_mode_only_from_the_input_or_the_variable():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/price-watch.yml").read_text(encoding="utf-8")
    assert "PARLAYPICKER_PRICE_WATCH_FAST: ${{ inputs.fast && 'true' || vars.PARLAYPICKER_PRICE_WATCH_FAST }}" in text


def test_nhl_is_an_extra_sport_with_its_own_switch_and_allowance(tmp_path, clock):
    base = {pw.WATCH_ENV: "NBA,NCAAB"}
    assert pw.extra_sports(base.get) == () and pw.sports(base.get, now=NOW) == ("NBA", "NCAAB")
    assert pw.limits(base.get) == {"daily": 200, "rolling_31_days": 1400}             # D28 as approved, NHL off
    nhl = {**base, pw.EXTRA_ENV: "nhl, mlb"}
    assert pw.sports(nhl.get, now=NOW) == ("NBA", "NCAAB", "NHL") and pw.interval_minutes("NHL", nhl.get) == 15
    assert pw.limits(nhl.get) == {"daily": 300, "rolling_31_days": 2100}
    fast = {**nhl, pw.FAST_ENV: "true"}
    assert pw.interval_minutes("NHL", fast.get) == 5 and pw.limits(fast.get) == {"daily": 900, "rolling_31_days": 6300}
    assert pw.sports({pw.EXTRA_ENV: "OFF", **base}.get, now=NOW) == ("NBA", "NCAAB")
    assert pw.sports(nhl.get, now=pw.UNTIL) == ()                                      # the end date covers NHL too

    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    start = NOW + timedelta(hours=2)
    raw = {"id": "nhl-soon", "sport_key": "icehockey_nhl", "home_team": "nhl-soon Home", "away_team": "nhl-soon Away",
           "commence_time": start.isoformat()}
    pe.insert_event(path, {"event_id": "nhl-soon", "sport": "NHL", "game_id": "nhl-soon", "provider_namespace": "THE_ODDS_API",
                           "provider_event_id": "nhl-soon", "home_team": raw["home_team"], "away_team": raw["away_team"],
                           "home_team_id": "nhl-h", "away_team_id": "nhl-a", "scheduled_start": start.isoformat(),
                           "observed_at": (NOW - timedelta(hours=8)).isoformat(), "source_id": "capture-nhl-soon",
                           "raw_source": raw})
    calls = []

    def provider(url, params=None, **kwargs):
        stamp = clock[0].isoformat()
        calls.append((url, params))
        return _Reply([dict(raw, bookmakers=[{"key": book, "markets": [
            {"key": "spreads", "last_update": stamp, "outcomes": [
                {"name": raw["home_team"], "point": -1.5, "price": 180}, {"name": raw["away_team"], "point": 1.5, "price": -220}]},
            {"key": "totals", "last_update": stamp, "outcomes": [
                {"name": "Over", "point": 6.5, "price": -110}, {"name": "Under", "point": 6.5, "price": -110}]}]}
            for book in ("pinnacle", "draftkings")])])
    only = {pw.EXTRA_ENV: "NHL"}.get
    report = pw.Watcher(only).poll(path, key="k", request_get=provider, clock=lambda: clock[0],
                                   budget=pw.budget_for({}, lambda: None, clock=lambda: clock[0], setting=only))
    assert set(report["sports"]) == {"NHL"} and report["capture"]["stored_quotes"] == 8
    assert "icehockey_nhl" in calls[0][0] and calls[0][1]["markets"] == "spreads,totals"
    quotes = pe.read_records(path, "prospective_quote", sport="NHL")
    assert {q["market_family"] for q in quotes} == {"PUCK_LINE", "TOTAL"}
    assert {q["sportsbook"] for q in quotes} == {"pinnacle", "draftkings"}
    from app_core import news_capture as nc
    assert nc.families("NHL") == ("PUCK_LINE", "TOTAL") and nc.families("NBA") == ("SPREAD", "TOTAL")


def test_automatic_pace_is_off_by_default_and_can_only_slow_a_sport_down(tmp_path, clock, monkeypatch):
    assert pw.auto_pace({}.get) is False and pw.auto_pace({pw.AUTO_ENV: "yes"}.get) is False
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    games = [_event(path, "NBA", "nba-soon", NOW + timedelta(hours=2)),
             _event(path, "NCAAB", "cbb-soon", NOW + timedelta(hours=1))]
    asked = []

    def recommended(_path, sport, now):
        asked.append(sport)
        if sport == "NCAAB":
            raise RuntimeError("replay failed")                                       # never stops the watch
        return {"status": "RECOMMENDED", "minutes": 30}
    monkeypatch.setattr(pw, "recommended", recommended)

    def calls(setting, minutes):
        provider = _Provider(clock, games)
        budget = pw.budget_for({}, lambda: None, clock=lambda: clock[0], setting=setting)
        watcher, reports = pw.Watcher(setting), []
        for minute in minutes:
            clock[0] = NOW + timedelta(minutes=minute)
            reports.append(_poll(watcher, path, provider, budget, clock))
        return [url.split("/sports/")[1].split("/")[0] for url, _ in provider.calls], reports

    off, reports = calls({pw.WATCH_ENV: "NBA,NCAAB"}.get, (0, 15, 30))
    assert off.count("basketball_nba") == 3 and asked == [] and "pace" not in reports[0]["sports"]["NBA"]
    on, reports = calls({pw.WATCH_ENV: "NBA,NCAAB", pw.AUTO_ENV: "true"}.get, (0, 15, 30))
    assert on.count("basketball_nba") == 2 and on.count("basketball_ncaab") == 2     # NBA slowed to 30; NCAAB as before
    assert asked == ["NBA", "NCAAB"]                                                  # once a run for each sport
    assert reports[0]["sports"]["NBA"]["pace"] == {"interval_minutes": 30, "status": "RECOMMENDED"}
    assert reports[0]["sports"]["NCAAB"]["pace"] == {"interval_minutes": 30, "status": "ERROR:RuntimeError"}
    # a recommendation faster than the mode's interval is ignored: NCAAB stays at 30 on a 15-minute recommendation
    monkeypatch.setattr(pw, "recommended", lambda *a: {"status": "RECOMMENDED", "minutes": 15})
    quick, _ = calls({pw.WATCH_ENV: "NCAAB", pw.AUTO_ENV: "true"}.get, (0, 15, 30))
    assert quick.count("basketball_ncaab") == 2
    fast, _ = calls({pw.WATCH_ENV: "NBA", pw.AUTO_ENV: "true", pw.FAST_ENV: "true"}.get, (0, 5, 10, 15))
    assert fast.count("basketball_nba") == 2                                          # fast mode slowed to 15, never faster


def test_the_recommendation_never_reads_games_from_before_the_watch(monkeypatch):
    seen = []
    monkeypatch.setattr(pm, "pace_comparison", lambda *a, **k: seen.append(k["lookback_days"]) or {"paces": []})
    pw.recommended("db", "NBA", pw.WATCH_START + timedelta(days=2))
    pw.recommended("db", "NBA", pw.WATCH_START + timedelta(days=30))
    assert seen == [2.0, 2.0, 7.0, 7.0]


def test_the_workflow_passes_automatic_pace_only_from_the_variable():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/price-watch.yml").read_text(encoding="utf-8")
    assert "PARLAYPICKER_PRICE_WATCH_AUTO_PACE: ${{ vars.PARLAYPICKER_PRICE_WATCH_AUTO_PACE }}" in text


def test_polls_start_an_interval_apart_however_long_the_backup_takes(tmp_path):
    # Run 2026-11-10: a "5-minute" watch stored a capture every 9 minutes (5 of sleep after a 4-minute backup).
    loop = importlib.import_module("scripts.news_capture_loop")
    assert loop.poll_wait(5, 1000.0, None) == 300                              # first poll: the whole interval
    assert loop.poll_wait(5, 1240.0, 1000.0) == 60 + loop.WAKE_PAD_SECONDS     # 4 minutes of work: sleep the fifth
    assert loop.poll_wait(5, 1400.0, 1000.0) == loop.WAKE_PAD_SECONDS          # work longer than the interval

    class Watcher:
        def poll(self, path, **kwargs):
            return {"news": [], "capture": {"requested": True, "stored_quotes": 8}}

    slept = []
    original = pw.load_state, pw.save_state, getattr(pw, "after_poll")
    pw.load_state, pw.save_state, pw.after_poll = (lambda client, folder: {}), (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        # clock readings: start, then after each poll's backup. Backups take 240 s; sleeps are instant here.
        ticks = iter([0, 240, 240 + 300 + 240, 2000])
        loop.run(20, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None, setting=ON,
                 sync=lambda *a: None, clock=lambda: next(ticks), sleep=slept.append, watcher=Watcher(), source=pw)
    finally:
        pw.load_state, pw.save_state, pw.after_poll = original
    assert slept == [300, 63]          # the first wait is the whole interval; the next makes the cycle 5 minutes


def test_steady_mode_polls_every_sport_every_fifteen_minutes_for_a_modestly_larger_allowance(tmp_path, clock):
    # Owner 2026-11-10: extend the faster polling only where it pays. The replay's gain was 30 -> 15 minutes (NCAAB).
    assert pw.steady({}.get) is False and pw.steady({pw.STEADY_ENV: "yes"}.get) is False
    base = {pw.WATCH_ENV: "NBA,NCAAB", pw.EXTRA_ENV: "NHL"}
    on = dict(base, **{pw.STEADY_ENV: "True"}).get
    assert [pw.interval_minutes(s, on) for s in ("NBA", "NCAAB", "NHL")] == [15, 15, 15]
    assert [pw.interval_minutes(s, base.get) for s in ("NBA", "NCAAB", "NHL")] == [15, 30, 15]     # normal is unchanged
    assert pw.limits(on) == {"daily": 400, "rolling_31_days": 2800}                               # 300 + NHL's 100
    assert pw.limits(base.get) == {"daily": 300, "rolling_31_days": 2100}
    both = dict(base, **{pw.STEADY_ENV: "true", pw.FAST_ENV: "true"}).get
    assert pw.interval_minutes("NCAAB", both) == 5 and pw.limits(both) == {"daily": 900, "rolling_31_days": 6300}

    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    games = [_event(path, "NCAAB", "cbb-soon", NOW + timedelta(hours=2))]
    provider = _Provider(clock, games)
    steady_only = {pw.WATCH_ENV: "NCAAB", pw.STEADY_ENV: "true"}.get
    budget = pw.budget_for({}, lambda: None, clock=lambda: clock[0], setting=steady_only)
    watcher = pw.Watcher(steady_only)
    for minutes in (0, 5, 15, 20, 30):
        clock[0] = NOW + timedelta(minutes=minutes)
        _poll(watcher, path, provider, budget, clock)
    assert len(provider.calls) == 3                                    # 0, 15 and 30: twice the normal watch's captures


def test_the_workflow_passes_steady_mode_only_from_the_input_or_the_variable():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/price-watch.yml").read_text(encoding="utf-8")
    assert "PARLAYPICKER_PRICE_WATCH_STEADY: ${{ inputs.steady && 'true' || vars.PARLAYPICKER_PRICE_WATCH_STEADY }}" in text
