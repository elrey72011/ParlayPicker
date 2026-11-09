"""Fixes from the 2026-11-09 repo audit (high-severity findings). SYNTHETIC stores and providers."""
from datetime import datetime, timedelta, timezone
import importlib

import pytest
import requests

from app_core import decision_capture as dc
from app_core import market_skill_validation as ms
from app_core import news_capture as nc
from app_core import pinnacle_model as pm
from app_core import prospective_evidence as pe
from tests.test_news_capture import BASE, EVENT, START, _Reply, _store  # noqa: F401
from tests.test_news_capture import clock  # noqa: F401  (fixture)


def _book(key, stamp, home, away, line=-4.5):
    return {"key": key, "markets": [{"key": "spreads", "last_update": stamp.isoformat(), "outcomes": [
        {"name": "Los Angeles Lakers", "point": line, "price": home},
        {"name": "Boston Celtics", "point": -line, "price": away}]}]}


def _payload(*books):
    return [{"id": EVENT, "sport_key": "basketball_nba", "home_team": "Los Angeles Lakers",
             "away_team": "Boston Celtics", "commence_time": START.isoformat(), "bookmakers": list(books)}]


def _events(path):
    return {e["provider_event_id"]: e for e in pe.read_records(path, "prospective_event", sport="NBA")}


# 1. Every provider attempt is charged -------------------------------------------------------------

def test_a_retried_request_is_charged_for_every_attempt(tmp_path, clock, monkeypatch):  # noqa: F811
    monkeypatch.setattr("app_core.odds_research_adapter.time.sleep", lambda s: None)
    path = _store(tmp_path)
    calls = []

    def flaky(url, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise requests.Timeout()
        return _Reply(_payload(_book("pinnacle", BASE, -110, -110)))

    budget = nc.budget_for({}, lambda: None, clock=lambda: clock[0])
    result = nc.capture(path, _events(path), key="k", request_get=flaky, budget=budget, clock=lambda: clock[0])
    assert len(calls) == 3 and result["stored_quotes"] == 2
    assert budget.usage("ODDS")["daily"] == 3 * nc.CREDITS_PER_REQUEST       # not 1 x 4
    # A retry the allowance cannot cover ends the request before calling the provider again.
    tight = nc.budget_for({}, lambda: None, clock=lambda: clock[0])
    tight.reserve("ODDS", nc.NEWS_LIMITS["daily"] - nc.CREDITS_PER_REQUEST - 1)
    calls.clear()
    stopped = nc.capture(path, _events(path), key="k", request_get=flaky, budget=tight, clock=lambda: clock[0])
    assert stopped["blocker"] == "api_budget:ODDS:daily" and len(calls) == 1


# 2. Each capture stores the full snapshot; the other side comes from the same response -----------

def test_an_unchanged_stake_price_is_stored_again_and_meets_the_moved_pinnacle_price(tmp_path, clock):  # noqa: F811
    path = _store(tmp_path)
    events = _events(path)
    first, second = BASE, BASE + timedelta(minutes=15)
    clock[0] = first
    dc.store_quotes(path, "NBA", "SPREAD", events,
                    _payload(_book("pinnacle", first, -110, -110), _book("novig", first, -110, -110)),
                    clock=lambda: clock[0], reference=True)
    clock[0] = second
    # News: Pinnacle moves to about 54/46 on the home side; Novig has not moved (same last_update).
    stored, _ = dc.store_quotes(path, "NBA", "SPREAD", events,
                                _payload(_book("pinnacle", second, -130, +110), _book("novig", first, -110, -110)),
                                clock=lambda: clock[0], reference=True)
    assert stored == 4                                            # Novig's unchanged price is stored again
    quotes = pe.read_records(path, "prospective_quote", sport="NBA")
    novig = [q for q in quotes if q["sportsbook"] == "novig" and q["selection"] == "Los Angeles Lakers"]
    assert {pm.capture_time(events[EVENT], q) for q in novig} == {first, second}
    # The other side is the same response's, for the selector and for validation scoring.
    later = next(q for q in novig if pm.capture_time(events[EVENT], q) == second)
    assert pm.capture_time(events[EVENT], pm._other_side(later, quotes)) == second
    from contextlib import closing
    with closing(pe.connect(path)) as db:
        other = ms.opposite(db, later)
    assert other is not None and other["selection"] == "Boston Celtics"
    # First-edge now sees the lag: Novig at -110 against Pinnacle's moved price, at the later capture.
    by_id = {e["event_id"]: e for e in events.values()}
    chosen = pm.plan_quotes(by_id, quotes, second + timedelta(minutes=1), entry_rule=ms.FIRST_EDGE_CAPTURE,
                            min_edge=0.02)[events[EVENT]["event_id"]]
    assert chosen["sportsbook"] == "novig" and pm.capture_time(events[EVENT], chosen) == second


def test_two_loops_never_write_the_same_quote_id_for_different_responses(tmp_path, clock):  # noqa: F811
    path = _store(tmp_path)
    events = _events(path)
    payload = _payload(_book("draftkings", BASE, -110, -110))
    for minutes in (1, 2):  # the same unchanged price, received by two loops a minute apart
        clock[0] = BASE + timedelta(minutes=minutes)
        dc.store_quotes(path, "NBA", "SPREAD", events, payload, clock=lambda: clock[0])
    ids = [q["quote_id"] for q in pe.read_records(path, "prospective_quote", sport="NBA")]
    assert len(ids) == len(set(ids)) == 4


# 3. A certified close is requested once per scope -----------------------------------------------

def test_a_book_missing_from_the_close_snapshot_is_not_requested_every_cycle(tmp_path, monkeypatch):
    from tests import test_prospective_closes as tpc
    from app_core import prospective_closes as pcl
    from app_core.research_api_budget import Budget
    state = [tpc.BASE]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"   # the same synthetic store as tests/test_prospective_closes.py
    pe.insert_event(path, dict(event_id="SYNTHETIC-E1", sport="NBA", game_id="SYNTHETIC-E1",
                    provider_namespace="THE_ODDS_API", provider_event_id="SYNTHETIC-E1", home_team="Home",
                    away_team="Away", home_team_id="provider:1", away_team_id="provider:2",
                    scheduled_start=tpc.at(1), observed_at=tpc.at(0), source_id="event-SYNTHETIC-E1",
                    raw_source=tpc.provider_event()))
    pe.insert_quote(path, dict(quote_id="SYNTHETIC-Q1", event_id="SYNTHETIC-E1", sport="NBA",
                    market_family="SPREAD", selection="Home", line=-2.5, american_odds=-110,
                    decimal_odds=1.909090909, sportsbook="TestBook", quote_timestamp=tpc.at(0),
                    quote_source="THE_ODDS_API", quote_verified=True,
                    source_id=f"SYNTHETIC-E1:TestBook:spreads:Home:{tpc.at(0)}", raw_source=tpc.market(0)))
    snap = tpc.snapshot()
    snap["data"]["bookmakers"][0]["key"] = "OtherBook"          # the quoted book pulled its line before tip
    provider = tpc.Provider(snap)
    now = tpc.BASE + timedelta(hours=1, minutes=5)
    state[0] = now
    budget = Budget({}, lambda: None, clock=lambda: now)
    first = pcl.record(path, now=now, key="K", request_get=provider, budget=budget, sports=("NBA",))
    assert first["requests"] == 1 and first["stored"] == 0
    again = pcl.record(path, now=now + timedelta(minutes=30), key="K", request_get=provider, budget=budget,
                       sports=("NBA",))
    assert again["requests"] == 0 and again["already_requested"] == 1 and len(provider.calls) == 1
    assert budget.usage("NBA", budget.close_ledger)["daily"] == first["credits"]   # charged once


# 4. The exposure ledger restore fails closed ------------------------------------------------------

def test_a_transient_drive_error_never_restores_part_of_the_ledger(tmp_path, monkeypatch):
    from app_core import exposure_ledger_remote as remote
    from core.exposure_ledger import append
    from tests.test_exposure_ledger_remote import CONFIG, NOW, Drive, _wager
    drive, path = Drive(), tmp_path / "exposure.sqlite3"
    append(path, CONFIG, confirmed=True, now=NOW)
    append(path, _wager("bet-1"), confirmed=True, now=NOW)
    remote.sync(path, drive)

    class Flaky(Drive):
        def get_object(self, *, Key):
            if Key == remote.key(2):
                raise RuntimeError("Drive 503")
            return super().get_object(Key=Key)

    flaky = Flaky()
    flaky.objects = dict(drive.objects)
    rebooted = tmp_path / "rebooted" / "exposure.sqlite3"
    with pytest.raises(RuntimeError):
        remote.remote_events(flaky)
    on = {remote.REMOTE_ENV: "true"}.get
    result = remote.before_read(rebooted, setting=on, client=flaky)
    assert result["status"] == "SYNC_FAILED" and remote.unsafe(result)
    assert not remote.unsafe({"status": "RESTORED"}) and not remote.unsafe({"status": "OFF"})


def test_the_live_contract_reports_no_exposure_when_the_remote_restore_failed(tmp_path, monkeypatch):
    from app_core import exposure_ledger_remote as remote
    from core import live_wager_contract as contract
    from core.exposure_ledger import append
    from tests.test_exposure_ledger_remote import CONFIG, NOW
    ledger = tmp_path / "exposure.sqlite3"
    append(ledger, CONFIG, confirmed=True, now=NOW)
    monkeypatch.setenv("PARLAYPICKER_EXPOSURE_LEDGER", str(ledger))
    monkeypatch.setattr(remote, "before_read", lambda path: {"status": "IN_SYNC"})
    assert contract.load_configuration(NOW)[1]["exposure"]               # the ledger is read
    monkeypatch.setattr(remote, "before_read", lambda path: {"status": "SYNC_FAILED", "error": "HTTPError"})
    assert contract.load_configuration(NOW)[1]["exposure"] == {}         # fail closed: no stake can be sized


# 5. A first-edge bet must be bettable when it is predicted ----------------------------------------

def test_a_first_edge_bet_predicted_from_a_stale_capture_is_not_a_bet(monkeypatch):
    captured = datetime(2030, 1, 1, 18, tzinfo=timezone.utc)
    event = {"event_id": "e", "scheduled_start": "2030-01-02T00:00:00+00:00",
             "observed_at": "2030-01-01T12:00:00+00:00"}
    entry = {"quote_id": "q", "event_id": "e", "market_family": "SPREAD", "selection": "Home", "line": -3.5,
             "decimal_odds": 2.0, "quote_verified": 1, "quote_timestamp": captured.isoformat(),
             "raw_source": pe.quote_source({"id": "e"}, {"key": "spreads"}, captured.isoformat())}
    made = {}

    def stored(db, table, column, key):
        if table == "prospective_prediction":
            return {"prediction_timestamp": made["at"].isoformat(), "quote_id": "q", "mean_probability": 0.6,
                    "loss_probability": 0.4, "selection": "Home"}
        return entry

    class Db:
        def execute(self, *args):
            return type("C", (), {"fetchall": lambda self: [("p1",)]})()

    monkeypatch.setattr(pe, "_event", lambda db, event_id: event)
    monkeypatch.setattr(pe, "_stored", stored)
    monkeypatch.setattr(pe, "_latest_result", lambda *a: {"outcome": "WIN"})
    monkeypatch.setattr(pe, "_outcome_at_quote_line", lambda *a: "WIN")
    monkeypatch.setattr(ms, "opposite", lambda db, quote: dict(entry, selection="Away", decimal_odds=1.8))
    plan = {"model_id": "m", "calibration_id": "c"}
    made["at"] = captured + timedelta(minutes=5)          # predicted while the price was still there
    bet = ms.first_edge_bet(Db(), plan, entry, {"min_edge": 0.02}, None)
    assert bet is not None and bet[0] == pytest.approx(0.2)
    made["at"] = captured + timedelta(minutes=ms.FIRST_EDGE_MAX_ENTRY_AGE_MINUTES + 1)   # found too late
    assert ms.first_edge_bet(Db(), plan, entry, {"min_edge": 0.02}, None) is None


# 6. Scheduled scheduler / Stage 1 runs step aside for a research loop ----------------------------

class _Api:
    def __init__(self, active):
        self.active, self.calls = set(active), []

    def __call__(self, url, params=None, **kwargs):
        self.calls.append(params["status"])
        count = 1 if params["status"] in self.active else 0
        return type("R", (), {"status_code": 200, "json": lambda self: {"total_count": count}})()


def test_a_scheduled_run_skips_while_a_research_loop_is_queued_or_running():
    gate = importlib.import_module("scripts.research_loop_active")
    env = {"GITHUB_EVENT_NAME": "schedule", "GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}.get
    assert gate.loop_active(_Api({"pending"}), env) is True
    assert gate.loop_active(_Api({"in_progress"}), env) is True
    assert gate.loop_active(_Api(set()), env) is False
    manual = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}.get
    api = _Api({"in_progress"})
    assert gate.loop_active(api, manual) is False and api.calls == []        # manual runs are never skipped

    def broken(url, **kwargs):
        raise requests.ConnectionError()
    assert gate.loop_active(broken, env) is False                            # an API error never blocks research


def test_the_gated_workflows_join_the_group_only_after_the_check():
    import yaml
    for name, job in (("research-scheduler", "research"), ("football-stage1", "football-stage1")):
        workflow = yaml.safe_load(open(f".github/workflows/{name}.yml", encoding="utf-8"))
        assert "concurrency" not in workflow                                  # the check runs outside the group
        assert workflow["permissions"]["actions"] == "read"
        assert "concurrency" not in workflow["jobs"]["loop-check"]
        gated = workflow["jobs"][job]
        assert gated["concurrency"]["group"] == "research-capture-and-grading"
        assert "loop-check" in (gated["needs"] if isinstance(gated["needs"], list) else [gated["needs"]])
        assert "needs.loop-check.outputs.active != 'true'" in gated["if"]
