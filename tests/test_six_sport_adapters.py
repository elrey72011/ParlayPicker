"""Regression coverage for the six-sport research registry and new sources."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from app_core import odds_research_adapter as odds
from app_core import prospective_sport_adapters as registry
from app_core import research_scheduler as scheduler
from app_core.odds_market_store import for_sport
from app_core.research_api_budget import Budget, BudgetLimit


class Cloud:
    def __init__(self):
        self.objects = {}

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        return [{"Contents": [{"Key": key} for key in self.objects if key.startswith(kwargs["Prefix"])]}]

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise ValueError("duplicate remote key")
        self.objects[key] = kwargs["Body"]

    def get_object(self, **kwargs):
        return {"Body": BytesIO(self.objects[kwargs["Key"]])}


class Response:
    def __init__(self, body, *, status=200, credits=None):
        self.body = body
        self.status_code = status
        self.headers = {} if credits is None else {"x-requests-last": str(credits)}

    def json(self):
        return self.body


@pytest.mark.parametrize("sport,markets", [
    ("NFL", ("SPREAD", "TOTAL")), ("NCAAF", ("SPREAD", "TOTAL")),
    ("NBA", ("SPREAD", "TOTAL")), ("NCAAB", ("SPREAD", "TOTAL")),
    ("MLB", ("RUN_LINE", "TOTAL")), ("NHL", ("PUCK_LINE", "TOTAL")),
])
def test_registry_contract_and_market_scope(sport, markets):
    assert tuple(registry.ADAPTERS) == ("NFL", "NCAAF", "NBA", "NCAAB", "MLB", "NHL")
    adapter = registry.get_adapter(sport)
    assert adapter.sport == sport and adapter.supported_markets == markets
    for name in ("restore", "upcoming_events", "capture_pregame", "capture_closes",
                 "grade", "backup", "audit", "health", "run_cycle"):
        assert callable(getattr(adapter, name))
    assert adapter.path_name.endswith(".sqlite3")


@pytest.mark.parametrize("bad", ["", ",", "NBA,,NHL", "NBA,NBA", "WNBA", "NFL,WNBA", "NHL,", " , "])
def test_scheduler_rejects_empty_duplicate_or_unsupported_sets(bad):
    with pytest.raises(ValueError, match="Invalid sports"):
        registry.parse_sports(bad)
    assert registry.parse_sports("nfl,ncaaf,nba,ncaab,mlb,nhl") == list(registry.DEFAULT_SPORTS)


def fixture_payload(sport, now):
    key = odds.SPORT_KEYS[sport]
    start = now + timedelta(minutes=30)
    names = ("Home Team", "Away Team")
    event = {"id": "abc123", "sport_key": key, "home_team": names[0],
             "away_team": names[1], "commence_time": start.isoformat()}
    participants = [{"id": "home1", "full_name": names[0]},
                    {"id": "away1", "full_name": names[1]}]
    line = 1.5 if sport == "NHL" else 4.5
    odds_event = {**event, "bookmakers": [{"key": "book1", "markets": [
        {"key": "spreads", "last_update": (now - timedelta(minutes=2)).isoformat(),
         "outcomes": [{"name": names[0], "point": -line, "price": -110},
                      {"name": names[1], "point": line, "price": -110}]},
        {"key": "totals", "last_update": (now - timedelta(minutes=2)).isoformat(),
         "outcomes": [{"name": "Over", "point": 221.5 if sport != "NHL" else 5.5, "price": -115},
                      {"name": "Under", "point": 221.5 if sport != "NHL" else 5.5, "price": -105}]}
    ]}]}
    score = {**event, "completed": True,
             "last_update": (start + timedelta(hours=3)).isoformat(),
             "scores": [{"name": names[0], "score": "5"}, {"name": names[1], "score": "3"}]}
    return event, participants, odds_event, score


def provider_for(event, participants, odds_event, score=None):
    calls = []
    def get(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append(endpoint)
        bodies = {"events": [event], "participants": participants,
                  "odds": [odds_event], "scores": [] if score is None else [score]}
        return Response(bodies[endpoint])
    return get, calls


@pytest.mark.parametrize("sport", ["NBA", "NCAAB", "NHL"])
def test_new_sport_capture_score_and_verified_remote_roundtrip(tmp_path, monkeypatch, sport):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    clock = [now]
    monkeypatch.setattr(odds, "utcnow", lambda: clock[0])
    event, participants, odds_event, score = fixture_payload(sport, now)
    get, calls = provider_for(event, participants, odds_event, score)
    adapter = registry.get_adapter(sport)
    path = tmp_path / adapter.path_name
    cloud = Cloud()
    session = {}
    assert adapter.restore(path, cloud, "folder", session)["records_verified"] == 0
    backup = lambda: adapter.backup(path, cloud, "folder", session)
    captured = adapter.capture_pregame(path, {}, "key", get, backup)
    assert captured["captured"] == 1 and captured["captured_quote_rows"] == 4
    assert calls == ["events", "participants", "odds"]
    record = adapter.store.records(path)[0]["data"]
    source = record["participants_source"]
    assert source["source_hash"] == odds.digest(source["raw_source"])
    assert {row["full_name"]: row["id"] for row in source["raw_source"]}["Home Team"] == "home1"
    captured_event = record["events"][0]
    assert captured_event["home_team_id"] == "home1" and captured_event["away_team_id"] == "away1"
    assert captured_event["discovery_source"]["source_hash"] == odds.digest(event)
    assert {q["market_family"] for q in captured_event["quotes"]} == set(adapter.supported_markets)
    assert all(q["quote_verified"] and q["source_hash"] == odds.digest(q["raw_source"])
               for q in captured_event["quotes"])
    assert captured_event["model_id"] is None and captured_event["production_eligible"] is False
    close = adapter.capture_closes(path, "key", get, backup)
    assert close == {"close_candidates": 1, "verified_closes": 0,
                     "close_blocker": "NO_VERIFIED_CLOSE_QUOTES", "deferred_close_events": 0,
                     "close_identity_rejections": 0, "close_price_unavailable_events": 0}
    clock[0] = now + timedelta(hours=4)
    grade = adapter.grade(path, "key", get, backup)
    assert grade["graded"] == 1
    assert adapter.audit(path)["settled_events"] == 1
    restored = tmp_path / "restored" / adapter.path_name
    restored_session = {}
    sync = adapter.restore(restored, cloud, "folder", restored_session)
    assert sync["records_verified"] == 3
    assert adapter.store.records(restored) == adapter.store.records(path)
    with adapter.store.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM records")


@pytest.mark.parametrize("sport", ["NBA", "NCAAB", "NHL"])
def test_zero_event_slate_skips_paid_provider_calls_and_succeeds(tmp_path, sport):
    adapter = registry.get_adapter(sport)
    calls = []
    def get(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        return Response([])
    path = tmp_path / adapter.path_name
    result = adapter.capture_pregame(path, {}, "key", get, lambda: pytest.fail("no mutation"))
    assert result["captured"] == result["due"] == result["discovered"] == 0
    assert calls == ["events"]
    assert adapter.grade(path, "key", get, lambda: None)["graded"] == 0
    assert adapter.capture_closes(path, "key", get, lambda: None)["verified_closes"] == 0


def test_missing_participant_identity_keeps_research_quote_unverified(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr(odds, "utcnow", lambda: now)
    event, participants, odds_event, _ = fixture_payload("NCAAB", now)
    get, _ = provider_for(event, participants[:1], odds_event)
    adapter = registry.get_adapter("NCAAB")
    result = adapter.capture_pregame(tmp_path / adapter.path_name, {}, "key", get, lambda: None)
    assert result["identity_blocked"] == 1
    stored = adapter.store.records(tmp_path / adapter.path_name)[0]["data"]["events"][0]
    assert stored["away_team_id"] is None
    assert all(not row["quote_verified"] for row in stored["quotes"])
    assert stored["production_eligible"] is False


def test_score_correction_appends_revision_and_preserves_first_source(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    clock = [now]
    monkeypatch.setattr(odds, "utcnow", lambda: clock[0])
    event, participants, odds_event, score = fixture_payload("NBA", now)
    get, _ = provider_for(event, participants, odds_event, score)
    adapter = registry.get_adapter("NBA")
    path = tmp_path / adapter.path_name
    adapter.capture_pregame(path, {}, "key", get, lambda: None)
    attempts = {}
    clock[0] = now + timedelta(hours=4)
    assert adapter.grade(path, "key", get, lambda: None, attempts)["graded"] == 1
    original = [row for row in adapter.store.records(path) if row["kind"] == "scores"][0]
    score["scores"][0]["score"] = "6"
    score["last_update"] = (now + timedelta(hours=9)).isoformat()
    clock[0] = now + timedelta(hours=10)
    grade = adapter.grade(path, "key", get, lambda: None, attempts)
    assert grade["graded"] == 0 and grade["corrected"] == 1
    rows = [row for row in adapter.store.records(path) if row["kind"] == "scores"]
    assert len(rows) == 2 and rows[0] == original
    assert rows[1]["data"]["event"]["revises_score_record_id"] == original["id"]
    assert rows[1]["data"]["event"]["grading_version"] == 2
    assert odds.scores(rows)["abc123"]["home_score"] == 6


def test_future_stale_post_start_and_incomplete_quotes_fail_closed():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    event, _, odds_event, _ = fixture_payload("NBA", now)
    for stamp in (now + timedelta(minutes=1), now - timedelta(hours=1),
                  now + timedelta(minutes=31)):
        row = json.loads(json.dumps(odds_event))
        for market in row["bookmakers"][0]["markets"]:
            market["last_update"] = stamp.isoformat()
        valid, rejected = odds.quotes("NBA", row, now)
        assert valid == [] and rejected["missing_future_or_stale_market_timestamp"] == 2
    row = json.loads(json.dumps(odds_event))
    row["bookmakers"][0]["markets"][0]["outcomes"][0].pop("point")
    valid, rejected = odds.quotes("NBA", row, now)
    assert len(valid) == 2 and rejected["invalid_market_pair"] == 1
    with pytest.raises(ValueError, match="provider_event_identity"):
        odds.identity("NBA", {**event, "id": ""})


def test_provider_auth_never_retries_and_transient_retry_is_bounded(monkeypatch):
    monkeypatch.setattr(odds.time, "sleep", lambda _: None)
    calls = []
    def auth(*args, **kwargs):
        calls.append(1)
        return Response([], status=401)
    with pytest.raises(odds.ProviderError, match="provider_auth_blocked"):
        odds.fetch("NBA", "events", "key", auth)
    assert len(calls) == 1
    calls.clear()
    def transient(*args, **kwargs):
        calls.append(1)
        raise __import__("requests").ReadTimeout("secret URL")
    with pytest.raises(odds.ProviderError, match="provider_transient_exhausted"):
        odds.fetch("NBA", "events", "key", transient)
    assert len(calls) == 3


def test_new_sport_budget_is_separate_but_within_shared_provider_cap(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("app_core.research_api_budget.is_open", lambda at: True)
    from app_core import research_api_budget as api
    state = {}
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return Response([], credits=2 if url.endswith("/odds") else 0)
    budget = Budget(state, lambda: None, clock=lambda: now, get=get)
    for sport in ("basketball_nba", "basketball_ncaab"):
        budget.request("https://api.the-odds-api.com/v4/sports/" + sport + "/odds",
                       params={"regions": "us", "markets": "spreads,totals", "oddsFormat": "american"})
    assert budget.usage("ODDS")["daily"] == 4
    assert budget.report()["by_sport"]["NBA"]["provider_units"]["daily"] == 2
    with pytest.raises(ValueError, match="unbudgeted_odds_markets"):
        budget.request("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds",
                       params={"regions": "us,uk", "markets": "spreads,totals", "oddsFormat": "american"})
    limited = Budget(state, lambda: None, clock=lambda: now, get=get,
                     limits={"ODDS": {"daily": 4, "rolling_31_days": 4}})
    with pytest.raises(BudgetLimit, match="api_budget:ODDS:daily"):
        limited.request("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds",
                        params={"regions": "us", "markets": "spreads,totals", "oddsFormat": "american"})
    assert len(calls) == 2


def test_new_sport_credit_cap_pauses_without_provider_call(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("app_core.research_api_budget.is_open", lambda at: True)
    calls = []
    budget = Budget({}, lambda: None, clock=lambda: now,
                    get=lambda url, **kwargs: calls.append(url) or Response([], credits=2))
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {"regions": "us", "markets": "spreads,totals", "oddsFormat": "american"}
    for _ in range(20):
        budget.request(url, params=params)
    with pytest.raises(BudgetLimit, match="api_budget:NBA:credits:daily"):
        budget.request(url, params=params)
    assert len(calls) == 20
    assert budget.report()["by_sport"]["NBA"]["provider_units"]["daily"] == 40


def test_all_six_restore_before_cycle_and_verified_backup_after(tmp_path, monkeypatch):
    cloud = Cloud()
    calls = []
    for sport, adapter in registry.ADAPTERS.items():
        monkeypatch.setattr(adapter, "restore", lambda path, client, folder, session, name=sport:
                            calls.append((name, "restore")) or {"records_verified": 0})
        monkeypatch.setattr(adapter, "run_cycle", lambda *args, name=sport:
                            calls.append((name, "cycle")) or {"captured": 0, "graded": 0, "errors": []})
        monkeypatch.setattr(adapter, "backup", lambda path, client, folder, session, name=sport:
                            calls.append((name, "backup")) or {"records_verified": 0})
    report = scheduler.run(list(registry.DEFAULT_SPORTS), tmp_path, cloud, "folder")
    assert report["requested_slate_success"] is True and not report["errors"]
    assert report["production_eligible"] is False
    for sport in registry.DEFAULT_SPORTS:
        assert [stage for name, stage in calls if name == sport] == ["restore", "cycle", "backup"]
        health = report["health"][sport]
        assert health["restore"] == health["capture"] == health["grade"] == health["backup"] == "success"
        assert health["verified_backup"] is True


def test_workflow_default_and_cli_fail_on_partial_requested_slate(tmp_path, monkeypatch):
    import yaml
    from scripts import run_research_scheduler as cli
    workflow = yaml.safe_load(Path(".github/workflows/research-scheduler.yml").read_text())
    job = workflow["jobs"]["research"]
    assert ",".join(registry.DEFAULT_SPORTS) in job["env"]["RESEARCH_SPORTS"]
    monkeypatch.setattr(cli, "is_open", lambda: True)
    monkeypatch.setattr(cli, "settings", lambda: ("folder", None))
    monkeypatch.setattr(cli, "DriveStore", lambda folder: object())
    monkeypatch.setattr(cli, "run", lambda *args: {"errors": [], "requested_slate_success": False})
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("RESEARCH_SPORTS", ",".join(registry.DEFAULT_SPORTS))
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "test-only-placeholder")
    monkeypatch.setenv("ODDS_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("CFBD_API_KEY", "test-only-placeholder")
    monkeypatch.delenv("PARLAYPICKER_NETLIFY_SITE_ID", raising=False)
    assert cli.main() == 1
    assert '"requested_slate_success": false' in (tmp_path / "summary.md").read_text()


def test_requested_sport_failure_is_explicit_and_later_sport_continues(tmp_path, monkeypatch):
    cloud = Cloud()
    monkeypatch.setattr(scheduler, "is_open", lambda: True)
    nba = registry.get_adapter("NBA")
    nhl = registry.get_adapter("NHL")
    monkeypatch.setattr(nba.store, "sync", lambda *a, **kw: {"records_verified": 0})
    monkeypatch.setattr(nhl.store, "sync", lambda *a, **kw: {"records_verified": 0})
    monkeypatch.setattr(nba, "run_cycle", lambda *a: {"captured": 0, "graded": 0, "errors": []})
    monkeypatch.setattr(nhl, "run_cycle", lambda *a: (_ for _ in ()).throw(ValueError("missing_provider_keys")))
    result = scheduler.run(["NHL", "NBA"], tmp_path, cloud, "folder")
    assert result["failure_stages"]["NHL"] == {"stage": "capture_and_grade", "code": "missing_provider_keys"}
    assert result["health"]["NBA"]["verified_backup"] is True
    assert result["requested_slate_success"] is False
    assert result["production_eligible"] is False


def test_health_keeps_last_success_after_new_failure_and_scopes_errors(tmp_path, monkeypatch):
    cloud = Cloud()
    adapter = registry.get_adapter("NBA")
    monkeypatch.setattr(adapter, "restore", lambda *args: {"records_verified": 0})
    monkeypatch.setattr(adapter, "backup", lambda *args: {"records_verified": 0})
    monkeypatch.setattr(adapter, "run_cycle", lambda *args: {"captured": 0, "graded": 0, "errors": []})
    first = scheduler.run(["NBA"], tmp_path, cloud, "folder")
    assert first["requested_slate_success"] is True
    prior = first["health"]["NBA"]["last_successful_capture"]
    monkeypatch.setattr(adapter, "run_cycle", lambda *args: {"captured": 0, "graded": 0,
                                                            "errors": ["capture_failed"],
                                                            "budget_paused": True})
    second = scheduler.run(["NBA"], tmp_path, cloud, "folder")
    assert second["errors"] == ["NBA:capture_failed"]
    assert second["requested_slate_success"] is False
    assert second["health"]["NBA"]["last_successful_capture"] == prior
    monkeypatch.setattr(adapter, "run_cycle", lambda *args: {"captured": 0, "graded": 0,
                                                            "errors": [], "budget_paused": True})
    third = scheduler.run(["NBA"], tmp_path, cloud, "folder")
    assert third["errors"] == [] and third["requested_slate_success"] is False
