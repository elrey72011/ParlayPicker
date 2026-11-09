"""News-triggered NBA capture: news from consecutive reports, one capture, a recheck, a capped budget. SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import importlib

import pytest

from app_core import nba_injury_reports as reports
from app_core import news_capture as nc
from app_core import prospective_evidence as pe

START = datetime(2026, 11, 10, 0, 30, tzinfo=timezone.utc)        # 7:30 PM ET
BASE = datetime(2026, 11, 9, 22, 0, tzinfo=timezone.utc)          # 5:00 PM ET report
EVENT = "provider-lal-bos"


def _row(player, status):
    return {"game_date": "11/09/2026", "game_time": "07:30 (ET)", "matchup": "BOS@LAL",
            "team": "Los Angeles Lakers", "player": player, "status": status, "reason": "Injury/Illness"}


@pytest.fixture
def clock(monkeypatch):
    state = [BASE]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    return state


def _store(tmp_path):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    raw = {"id": EVENT, "sport_key": "basketball_nba", "home_team": "Los Angeles Lakers",
           "away_team": "Boston Celtics", "commence_time": START.isoformat()}
    pe.insert_event(path, {"event_id": "lal-bos", "sport": "NBA", "game_id": EVENT, "provider_namespace": "THE_ODDS_API",
                           "provider_event_id": EVENT, "home_team": "Los Angeles Lakers",
                           "away_team": "Boston Celtics", "home_team_id": "lal", "away_team_id": "bos",
                           "scheduled_start": START.isoformat(), "observed_at": (BASE - timedelta(hours=6)).isoformat(),
                           "source_id": "capture-lal-bos", "raw_source": raw})
    return path


def _odds(stamp, line):
    def book(key, home, away):
        return {"key": key, "markets": [
            {"key": "spreads", "last_update": stamp.isoformat(), "outcomes": [
                {"name": "Los Angeles Lakers", "point": line, "price": home},
                {"name": "Boston Celtics", "point": -line, "price": away}]},
            {"key": "totals", "last_update": stamp.isoformat(), "outcomes": [
                {"name": "Over", "point": 224.5, "price": -110}, {"name": "Under", "point": 224.5, "price": -110}]}]}
    return [{"id": EVENT, "sport_key": "basketball_nba", "home_team": "Los Angeles Lakers",
             "away_team": "Boston Celtics", "commence_time": START.isoformat(),
             "bookmakers": [book("pinnacle", -105, -105), book("novig", 102, -112), book("draftkings", -110, -110)]}]


class _Reply:
    status_code = 200

    def __init__(self, payload, spent="4"):
        self.payload, self.headers = payload, {"x-requests-last": spent}

    def json(self):
        return self.payload


class _Provider:
    def __init__(self, clock, spent="4"):
        self.clock, self.spent, self.calls = clock, spent, []

    def __call__(self, url, params=None, **kwargs):
        self.calls.append(params)
        return _Reply(_odds(self.clock[0], -4.5 - len(self.calls)), self.spent)


def _reports(monkeypatch, published):
    """``published``: {ET slot: rows}; fetch serves a slot's URL once it is published."""
    by_url = {reports.slot_url(slot): slot for slot in published}
    monkeypatch.setattr(reports, "parse", lambda body: published[by_url[body.decode()]])
    return lambda url: (200, url.encode()) if url in by_url else (403, b"")


def _budget(clock):
    state = {}
    return nc.budget_for(state, lambda: None, clock=lambda: clock[0]), state


def test_off_by_default():
    assert nc.enabled({}.get) is False
    assert nc.enabled({nc.NEWS_ENV: "nhl, nba"}.get) is True


def test_news_captures_the_game_at_once_and_ten_minutes_later(tmp_path, clock, monkeypatch):
    path = _store(tmp_path)
    et = lambda t: t.astimezone(reports.ET)  # noqa: E731
    fetch = _reports(monkeypatch, {et(BASE): [_row("Davis, Anthony", "Questionable")],
                                   et(BASE + timedelta(minutes=15)): [_row("Davis, Anthony", "Out")]})
    provider, (budget, state) = _Provider(clock), _budget(clock)
    watcher = nc.Watcher()
    poll = lambda: watcher.poll(path, fetch=fetch, key="k", request_get=provider, budget=budget,  # noqa: E731
                                clock=lambda: clock[0])
    clock[0] = BASE + timedelta(minutes=3)
    first = poll()
    assert first["news"] == [] and first["capture"] is None          # the first report is the baseline
    clock[0] = BASE + timedelta(minutes=18)
    second = poll()
    assert [g["event_id"] for g in second["news"]] == [EVENT] and second["news"][0]["flip"]
    assert second["capture"]["requested"] and second["capture"]["stored_quotes"] == 12   # 3 books x 2 sides x 2
    assert provider.calls[0]["eventIds"] == EVENT and provider.calls[0]["markets"] == "spreads,totals"
    assert "pinnacle" in provider.calls[0]["bookmakers"]
    quotes = pe.read_records(path, "prospective_quote", sport="NBA")
    assert {q["sportsbook"] for q in quotes} >= {"pinnacle", "novig", "draftkings"}
    assert {q["market_family"] for q in quotes} == {"SPREAD", "TOTAL"}
    clock[0] = BASE + timedelta(minutes=23)
    assert poll()["capture"] is None                                 # the same report: nothing new, recheck not due
    clock[0] = BASE + timedelta(minutes=29)
    recheck = poll()
    assert recheck["capture"]["requested"] and len(provider.calls) == 2
    clock[0] = BASE + timedelta(minutes=40)
    assert poll()["capture"] is None and len(provider.calls) == 2
    assert budget.usage("ODDS")["daily"] == 2 * nc.CREDITS_PER_REQUEST    # two requests, 4 credits each


def test_news_after_the_start_or_for_unstored_games_spends_nothing(tmp_path, clock, monkeypatch):
    path = _store(tmp_path)
    other = dict(_row("Brown, Jaylen", "Questionable"), matchup="MIA@NYK", team="New York Knicks")
    et = lambda t: t.astimezone(reports.ET)  # noqa: E731
    fetch = _reports(monkeypatch, {et(BASE): [other], et(BASE + timedelta(minutes=15)): [dict(other, status="Out")]})
    provider, (budget, _) = _Provider(clock), _budget(clock)
    watcher = nc.Watcher()
    for minutes in (3, 18):
        clock[0] = BASE + timedelta(minutes=minutes)
        report = watcher.poll(path, fetch=fetch, key="k", request_get=provider, budget=budget, clock=lambda: clock[0])
    assert report["news"] and report["news"][0]["event_id"] is None and provider.calls == []


def test_the_daily_cap_and_an_unexpected_charge_stop_capture(tmp_path, clock):
    path = _store(tmp_path)
    events = {EVENT: pe.read_records(path, "prospective_event", sport="NBA")[0]}
    budget, state = _budget(clock)
    for _ in range(nc.NEWS_LIMITS["daily"] // nc.CREDITS_PER_REQUEST):
        budget.reserve("ODDS", nc.CREDITS_PER_REQUEST)
    provider = _Provider(clock)
    capped = nc.capture(path, events, key="k", request_get=provider, budget=budget, clock=lambda: clock[0])
    assert capped["blocker"] == "api_budget:ODDS:daily" and provider.calls == []
    fresh, _ = _budget(clock)
    dear = _Provider(clock, spent="8")
    result = nc.capture(path, events, key="k", request_get=dear, budget=fresh, clock=lambda: clock[0])
    assert result["blocker"] == nc.COST_FLAG and fresh.state[nc.COST_FLAG] == "8"
    again = nc.capture(path, events, key="k", request_get=dear, budget=fresh, clock=lambda: clock[0])
    assert again["blocker"] == nc.COST_FLAG and not again["requested"] and len(dear.calls) == 1


def test_loop_is_off_without_the_switch_and_backs_up_after_stored_quotes(tmp_path):
    loop = importlib.import_module("scripts.news_capture_loop")
    assert loop.run(1, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None,
                    setting={}.get)["status"] == "OFF"

    class Watcher:
        def __init__(self):
            self.replies = [{"news": [{"x": 1}], "capture": {"requested": True, "stored_quotes": 4}},
                            {"news": [], "capture": None}]

        def poll(self, path, **kwargs):
            return self.replies.pop(0)

    syncs = []
    monkey_state = {}
    import app_core.news_capture as news
    original = news.load_state, news.save_state
    news.load_state, news.save_state = (lambda client, folder: monkey_state), (lambda *a, **k: None)
    try:
        ticks = iter([0, 0, 400])
        result = loop.run(10, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None,
                          setting={nc.NEWS_ENV: "NBA"}.get, sync=lambda *a: syncs.append(1),
                          clock=lambda: next(ticks), sleep=lambda s: None, watcher=Watcher())
    finally:
        news.load_state, news.save_state = original
    assert result["polls"] == 2 and result["news"] == 1 and result["captures"] == 1 and result["stored_quotes"] == 4
    assert len(syncs) == 2 and result["backups"] == 1                 # the restore, then one backup
