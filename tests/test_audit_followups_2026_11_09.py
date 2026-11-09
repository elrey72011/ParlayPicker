"""Medium and low findings of the 2026-11-09 repo audit. SYNTHETIC stores, providers and clocks."""
from datetime import datetime, timedelta, timezone
import importlib

import pandas as pd
import pytest

from app_core import odds_research_adapter as odds
from app_core import research_api_budget as rab
from tests.test_six_sport_adapters import fixture_payload, provider_for

NOW = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)
NCAAB_ODDS = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
PIN = {"bookmakers": rab.PINNACLE_BOOKMAKERS, "markets": "spreads,totals", "oddsFormat": "american"}


# A. Budgets ---------------------------------------------------------------------------------------

def test_a_pinnacle_request_the_plan_wide_cap_cannot_hold_charges_nothing():
    state = {"api_budget_v1": {"ODDS": {NOW.date().isoformat(): rab.LIMITS["ODDS"]["daily"] - 3}}}
    calls = []
    budget = rab.Budget(state, lambda: None, clock=lambda: NOW, get=lambda *a, **kw: calls.append(1))
    with pytest.raises(rab.BudgetLimit, match="PINNACLE"):            # so capture retries US-only (2 credits)
        budget.request(NCAAB_ODDS, params=PIN)
    assert not calls and budget.usage("ODDS")["daily"] == rab.LIMITS["ODDS"]["daily"] - 3
    assert budget.usage("NCAAB", budget.sport_ledger)["daily"] == 0


def test_a_provider_cost_lock_fails_the_research_run(monkeypatch, tmp_path):
    from app_core import research_scheduler as scheduler
    from tests.test_scheduler_backlog import _run
    locked = {"api_budget_cost_mismatch": True}
    monkeypatch.setattr(scheduler, "checkpoint", lambda client, folder, state=None: locked)
    result = _run(monkeypatch, tmp_path, {"errors": [], "grade_status": "success", "blockers": []})
    assert "scheduler:PROVIDER_COST_REVIEW_REQUIRED" in result["errors"]
    assert result["requested_slate_success"] is False and result["execution_state"] == "FAILED"


def test_an_overcharge_in_decision_or_news_capture_is_counted():
    from app_core import decision_capture as dc

    class Reply:
        headers = {"x-requests-last": "9"}
    budget = dc.budget_for({}, lambda: None, clock=lambda: NOW)
    budget.reserve("ODDS", 4)
    result = {}
    dc.charging(lambda url, **kw: Reply(), budget, 4, "flag", result)("u")
    assert budget.usage("ODDS")["daily"] == 9 and budget.state["flag"] == "9" and result["blocker"] == "flag"


def test_certified_closes_stop_while_a_cost_review_is_pending(tmp_path):
    from app_core import prospective_closes as pcl
    calls = []
    for flag in (pcl.COST_FLAG, "api_budget_cost_mismatch"):
        budget = rab.Budget({flag: True}, lambda: None, clock=lambda: NOW)
        report = pcl.record(tmp_path / "none.sqlite3", now=NOW, key="K", request_get=lambda *a, **k: calls.append(1),
                            budget=budget, sports=("NBA",))
        assert report["skipped"] == ["PROVIDER_COST_REVIEW_REQUIRED"] and not calls


def _expired_game(tmp_path, monkeypatch):
    from app_core import prospective_sport_adapters as registry
    now = datetime.now(timezone.utc).replace(microsecond=0)
    clock = [now]
    monkeypatch.setattr(odds, "utcnow", lambda: clock[0])
    event, participants, odds_event, _ = fixture_payload("NCAAB", now)
    get, _ = provider_for(event, participants, odds_event, None)
    adapter = registry.get_adapter("NCAAB")
    path = tmp_path / adapter.path_name
    adapter.capture_pregame(path, {}, "key", get, lambda: None)
    monkeypatch.setattr(adapter, "capture_pregame", lambda *a, **k: {"identity_blocked": 0,
                        "price_unavailable_events": 0, "deferred_due_events": 0})
    monkeypatch.setattr(adapter, "capture_closes", lambda *a, **k: {"deferred_close_events": 0,
                        "close_identity_rejections": 0, "close_price_unavailable_events": 0})
    clock[0] = now + timedelta(days=4)
    return adapter, path


def test_an_expiry_found_while_grading_is_paused_still_fails_one_later_cycle(tmp_path, monkeypatch):
    adapter, path = _expired_game(tmp_path, monkeypatch)
    budget = type("NoCalls", (), {"request": None})()
    state = {}

    def paused(*a, **k):
        raise rab.BudgetLimit("api_budget:NCAAB:credits:daily")
    monkeypatch.setattr(adapter, "grade", paused)
    first = adapter.run_cycle(path, state, None, "key", lambda: None, budget)
    assert first["grade_status"] == "budget_paused" and not state.get("ncaab_expired_reported")
    monkeypatch.setattr(adapter, "grade", lambda *a, **k: {"graded": 0, "corrected": 0, "score_rejections": {},
                                                            "deferred_grade_events": 0})
    second = adapter.run_cycle(path, state, None, "key", lambda: None, budget)
    assert "SCORE_UNAVAILABLE_AFTER_PROVIDER_WINDOW" in second["blockers"]   # the one failing cycle
    assert state["ncaab_expired_reported"] == ["abc123"]


def test_score_check_outcome_is_set_only_after_the_provider_answers_and_old_attempts_are_pruned(tmp_path,
                                                                                                monkeypatch):
    from app_core import prospective_sport_adapters as registry
    now = datetime.now(timezone.utc).replace(microsecond=0)
    clock = [now]
    monkeypatch.setattr(odds, "utcnow", lambda: clock[0])
    event, participants, odds_event, _ = fixture_payload("NCAAB", now)
    get, _ = provider_for(event, participants, odds_event, None)              # the scores answer is empty
    adapter = registry.get_adapter("NCAAB")
    path = tmp_path / adapter.path_name
    adapter.capture_pregame(path, {}, "key", get, lambda: None)
    clock[0] = now + timedelta(hours=4)
    state = {"ncaab_score_check_attempts": {"old-game": (now - timedelta(days=30)).isoformat()}}
    adapter.grade(path, "key", get, lambda: None, state)
    assert state["ncaab_score_check_status"] == {"abc123": "not_returned"}
    assert "old-game" not in state["ncaab_score_check_attempts"]


# B. Loops and workflows ---------------------------------------------------------------------------

def test_a_failed_backup_is_retried_and_never_ends_the_closing_loop():
    loop = importlib.import_module("scripts.closing_capture_loop")
    now, results, backups = [0.0], iter([{"verified": 1}, {"verified": 0}, {"verified": 0}]), []

    def backup(path):
        backups.append(1)
        if len(backups) == 1:
            raise RuntimeError("Drive 503")
    totals = loop.run("db", minutes=11, interval=5, capture=lambda path: next(results), restore=lambda p: None,
                      backup=backup, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    assert totals["cycles"] == 3 and totals["backup_errors"] == 1 and totals["backups"] == 1 and len(backups) == 2


def test_the_news_loop_retries_a_failed_backup_and_restores_again_every_hour(tmp_path):
    loop = importlib.import_module("scripts.news_capture_loop")
    import app_core.news_capture as news
    syncs, replies = [], iter([{"capture": {"requested": True, "stored_quotes": 4}}] + [{}] * 20)

    class Watcher:
        def poll(self, path, **kwargs):
            return next(replies)

    def sync(*args):
        syncs.append(args[3] is not None)
        if len(syncs) == 2:
            raise RuntimeError("readback")
    ticks = iter([0] + [i * 15 * 60 for i in range(1, 8)])
    original = news.load_state, news.save_state
    news.load_state, news.save_state = (lambda client, folder: {}), (lambda *a, **k: None)
    try:
        totals = loop.run(100, 15, root=tmp_path, client=None, folder="f", key="k", request_get=None,
                          setting={news.NEWS_ENV: "NBA"}.get, sync=sync, clock=lambda: next(ticks),
                          sleep=lambda s: None, watcher=Watcher())
    finally:
        news.load_state, news.save_state = original
    # restore, failed backup, retried backup, then the hourly restore
    assert totals["backup_errors"] == 1 and totals["backups"] == 1 and len(syncs) == 4


def test_the_news_loop_knows_when_the_slate_is_over():
    loop = importlib.import_module("scripts.news_capture_loop")
    assert loop.slate_over(datetime(2026, 11, 10, 8, tzinfo=timezone.utc))          # 3 AM ET
    assert not loop.slate_over(datetime(2026, 11, 10, 1, tzinfo=timezone.utc))      # 8 PM ET


def test_the_research_loop_ends_when_the_window_closes(tmp_path):
    rl = importlib.import_module("scripts.research_loop")
    states = iter([(0, "COMPLETE"), (0, "COMPLETE"), (0, "SKIPPED"), (0, "SKIPPED")])
    now = [0.0]
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=lambda n: (lambda c, s: (c, {"execution_state": s}))(*next(states)),
                     waiting=lambda: False, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    assert totals["cycles"] == 3 and totals["ended"] == "window_closed"


def test_qualification_filter_covers_the_driver_and_template():
    text = open(".github/workflows/qualification-operations.yml", encoding="utf-8").read()
    assert "tools/qualification/" in text and "snapshot-acquisition" in text and '<<< "$changed"' in text


# C. Plan chain --------------------------------------------------------------------------------------

def test_first_edge_skips_decision_predictions_and_an_unsettled_side(monkeypatch):
    from app_core import market_skill_validation as ms
    from app_core import prospective_evidence as pe
    captured = datetime(2030, 1, 1, 18, tzinfo=timezone.utc)
    event = {"event_id": "e", "scheduled_start": "2030-01-02T00:00:00+00:00", "observed_at": captured.isoformat()}
    entry = {"quote_id": "q", "event_id": "e", "market_family": "SPREAD", "selection": "Home", "line": -3.5,
             "decimal_odds": 2.0, "quote_verified": 1, "quote_timestamp": captured.isoformat(),
             "raw_source": pe.quote_source({"id": "e"}, {"key": "spreads"}, captured.isoformat())}
    preds = {"p1": "Away", "p2": "Home"}
    seen = {}

    class Db:
        def execute(self, sql, *args):
            seen["sql"] = sql
            return type("C", (), {"fetchall": lambda self: [("p1",), ("p2",)]})()

    def stored(db, table, column, key):
        if table == "prospective_prediction":
            return {"prediction_timestamp": (captured + timedelta(minutes=1)).isoformat(), "quote_id": "q",
                    "mean_probability": 0.6, "loss_probability": 0.4, "selection": preds[key]}
        return entry
    monkeypatch.setattr(pe, "_event", lambda db, event_id: event)
    monkeypatch.setattr(pe, "_stored", stored)
    monkeypatch.setattr(pe, "_latest_result", lambda db, e, f, selection, at: None if selection == "Away" else {"outcome": "WIN"})
    monkeypatch.setattr(pe, "_outcome_at_quote_line", lambda *a: "WIN")
    monkeypatch.setattr(ms, "opposite", lambda db, quote: dict(entry, selection="Away", decimal_odds=1.8))
    bet = ms.first_edge_bet(Db(), {"model_id": "m", "calibration_id": "c"}, entry, {"min_edge": 0.02}, None)
    assert bet is not None                                   # the unsettled first side no longer drops the game
    assert "NOT LIKE 'decision-prediction:%'" in seen["sql"]


def test_bound_predictions_use_a_fresh_clock(monkeypatch):
    from app_core import prospective_calibration as pc
    from app_core import prospective_evidence as pe
    early, late = NOW, NOW + timedelta(minutes=9)
    used = []
    monkeypatch.setattr(pc, "settlement_enabled", lambda sport: True)
    monkeypatch.setattr(pc, "settle_scope", lambda *a, **k: {})
    monkeypatch.setattr(pc, "candidate_model", lambda *a, **k: None)
    monkeypatch.setattr(pc, "pinnacle_cycle", lambda *a, **k: {})
    monkeypatch.setattr(pc, "predict_bound", lambda path, sport, market, now: used.append(now) or {})
    monkeypatch.setattr(pe, "_clock", lambda: late)
    pc.run_cycle("db", "NBA", now=early)
    assert used and all(at == late for at in used)


def test_a_quote_received_after_now_is_not_a_decision_quote():
    from app_core import decision_capture as dc
    from app_core import prospective_evidence as pe
    event = {"event_id": "e", "scheduled_start": (NOW + timedelta(minutes=30)).isoformat()}
    quote = {"event_id": "e", "quote_verified": 1, "line": -2.5, "sportsbook": "draftkings", "selection": "H",
             "quote_timestamp": NOW.isoformat(), "decimal_odds": 1.9, "quote_id": "q",
             "raw_source": pe.quote_source({"id": "e"}, {"key": "spreads"}, (NOW + timedelta(minutes=2)).isoformat())}
    assert dc.decision_quotes({"e": event}, [quote], NOW) == {}


def test_plan_progress_waits_before_projecting():
    from app_core import plan_progress as pp
    from tests.test_plan_progress import PLAN, _evaluation
    now = datetime(2030, 1, 1, 6, tzinfo=timezone.utc)                       # 2.5% through validation
    row = pp.progress(PLAN, _evaluation(games=30, graded=0, pending=30, bets=0, with_close=0), now=now)[0]
    assert row["projected_games"] is None and row["on_pace"] is None
    assert "too early" in "\n".join(pp.markdown([row]))


# D. Go-live -----------------------------------------------------------------------------------------

def test_an_approved_pick_with_a_stale_quote_shows_as_a_pass():
    from app.ui.daily_dashboard import daily_board
    now = datetime(2026, 11, 9, 23, tzinfo=timezone.utc)
    board = daily_board(pd.DataFrame([
        {"best_pick": "Fresh", "Play_Stake": 5, "production_eligible": True, "Bet_Decision": "BET",
         "odds_recorded_at": (now - timedelta(minutes=10)).isoformat()},
        {"best_pick": "Stale", "Play_Stake": 5, "production_eligible": True, "Bet_Decision": "BET",
         "odds_recorded_at": (now - timedelta(hours=2)).isoformat()}]), now=now)
    assert board.Bettable.tolist() == [True, False] and board.Play_Stake.tolist() == [5, 0]
    assert "refresh picks" in board.loc[1, "Production_Gate_Reason"]


def test_a_stale_recommendation_is_not_offered_to_mark_as_placed():
    from app.ui.activation_panel import _stale
    now = datetime(2026, 11, 9, 23, tzinfo=timezone.utc)
    assert _stale({"quote_timestamp": (now - timedelta(hours=1)).isoformat()}, now=now)
    assert not _stale({"quote_timestamp": (now - timedelta(minutes=5)).isoformat()}, now=now)
    assert not _stale({}, now=now)


def test_one_malformed_prediction_drops_its_own_bridge_row(monkeypatch):
    from app_core import bridge_stake_fields
    from app_core import qualified_bridge as qb
    from app_core import prospective_evidence as pe
    start = (NOW + timedelta(hours=2)).isoformat()
    records = {
        "prospective_event": [{"event_id": "e1", "scheduled_start": start}, {"event_id": "e2", "scheduled_start": start}],
        "prospective_quote": [{"quote_id": "q1", "sportsbook": "draftkings"}, {"quote_id": "q2", "sportsbook": "draftkings"}],
        "prospective_prediction": [{"event_id": "e1", "quote_id": "q1", "prediction_timestamp": "not a time"},
                                   {"event_id": "e2", "quote_id": "q2", "prediction_timestamp": NOW.isoformat()}]}
    monkeypatch.setattr(pe, "read_records", lambda path, table, **k: records[table])
    monkeypatch.setattr(qb, "_rank", lambda p, q: 0)
    monkeypatch.setattr(qb, "row", lambda p, q, e, v=None: {"start": e["scheduled_start"], "sport": "NBA",
                                                            "game_id": e["event_id"]})
    monkeypatch.setattr(bridge_stake_fields, "market_context", lambda *a, **k: {})
    monkeypatch.setattr(bridge_stake_fields, "fields", lambda *a, **k: {})
    rows = qb.rows("db", {("NBA", "SPREAD")}, now=NOW)
    assert [r["game_id"] for r in rows] == ["e2"]
