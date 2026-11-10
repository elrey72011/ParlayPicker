"""market-skill-v2 (owner D26), pooling (D27) and the frozen junk-price term (D29). SYNTHETIC store."""
import importlib

import pytest

from app_core import market_skill_validation as ms
from app_core import prospective_evidence as pe
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    at, calibration, clock, event, model, prediction, quote, result)

GOOD = {"market_reference_coverage": 0.95, "model_brier": 0.251, "market_brier": 0.249, "model_log_loss": 0.70,
        "market_log_loss": 0.691, "bets": 80, "bet_roi": -0.04, "bet_close_coverage": 0.9, "bet_mean_clv": 0.01,
        "bet_clv_t": 3.1}


def test_v2_rules_lead_with_clv_and_report_the_probability_scores():
    rules = ms.methodology_v2()
    assert rules["version"] == ms.VERSION_V2 and rules["min_bets"] == 75 and rules["min_clv_t"] == 2.0
    assert rules["min_bet_roi"] == -0.25 and rules["min_bet_close_coverage"] == 0.8
    assert rules["push_rule"] == ms.PUSH_ZERO_RETURN and "pool" not in rules and "max_entry_overround" not in rules
    method = {ms.KEY: rules}
    # Worse than the market on Brier and log loss, and a small loss: none of these block in v2.
    assert ms.blockers("HOLDOUT", {}, GOOD, method) == []
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, bet_clv_t=1.9), method) == [
        "HOLDOUT_BET_CLV_NOT_2_STANDARD_ERRORS_ABOVE_ZERO"]
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, bet_clv_t=None), method) == [
        "HOLDOUT_BET_CLV_NOT_2_STANDARD_ERRORS_ABOVE_ZERO"]
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, bets=70), method) == ["HOLDOUT_NEED_5_MORE_BETS"]
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, bet_roi=-0.3), method) == ["HOLDOUT_BET_ROI_BELOW_LOSS_FLOOR"]
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, bet_close_coverage=0.7), method) == ["HOLDOUT_BET_CLOSE_COVERAGE_TOO_LOW"]
    assert ms.blockers("HOLDOUT", {}, dict(GOOD, market_reference_coverage=0.5), method) == [
        "HOLDOUT_MARKET_REFERENCE_COVERAGE_TOO_LOW"]
    # The same result under the v1 rules still fails on the probability gate: v1 plans are unchanged.
    assert "HOLDOUT_BRIER_WORSE_THAN_MARKET" in ms.blockers("HOLDOUT", {}, GOOD, {ms.KEY: ms.methodology()})


def test_v2_choices_are_limited_and_the_template_sizes_by_bets():
    for bad in (dict(min_bets=40), dict(min_edge=0.03), dict(max_entry_overround=1.0),
                dict(pool=["NBA:SPREAD"]), dict(pool=["NBA:SPREAD", "NHL:SPREAD"]),
                dict(pool=["NBA:SPREAD", "NCAAB:TOTAL"]), dict(pool=["NBA:SPREAD", "NBA:TOTAL"])):
        with pytest.raises(ValueError):
            ms.methodology_v2(**bad)
    with pytest.raises(ValueError):
        ms.pool_scopes("NHL", "TOTAL")
    base = {"probability_thresholds": {"max_brier": 0.24, "max_log_loss": 0.68, "max_calibration_error": 0.08,
                                       "min_coverage": 0.9},
            "clv_policy": {"required": True, "min_comparable_close_coverage": 0.8},
            "model_scope": {"cross_sport_or_market_pooling": "PROHIBITED"},
            "minimum_independent_sample": 200, "minimum_effective_sample": 200}
    alone = ms.template_v2(base, min_bets=100, max_entry_overround=1.15)
    assert alone["probability_thresholds"] == dict(ms.V2_THRESHOLDS, min_coverage=0.9)
    assert alone["minimum_independent_sample"] == alone["minimum_effective_sample"] == 100
    assert alone["clv_policy"] == {"required": True, "min_comparable_close_coverage": 0.0}
    assert alone["model_scope"]["cross_sport_or_market_pooling"] == "PROHIBITED"
    assert alone[ms.KEY]["max_entry_overround"] == 1.15 and ms.is_v2(alone)
    pooled = ms.template_v2(base, pool=ms.pool_scopes("NCAAB", "SPREAD"))
    assert pooled[ms.KEY]["pool"]["scopes"] == ["NBA:SPREAD", "NCAAB:SPREAD"]
    assert pooled["minimum_independent_sample"] == ms.MIN_SCOPE_BETS
    assert pooled["model_scope"]["cross_sport_or_market_pooling"] != "PROHIBITED"
    assert base["minimum_independent_sample"] == 200 and not ms.is_v2({ms.KEY: ms.methodology()})


def test_a_pooled_plan_is_gated_on_the_pools_bets_and_each_scopes_minimum():
    method = {ms.KEY: ms.methodology_v2(pool=ms.pool_scopes("NBA", "SPREAD"))}
    own = dict(GOOD, bets=20, bet_clv_t=0.4)          # too few bets and a weak t on its own
    pool = {"complete": True, "bets": 90, "bet_roi": 0.01, "bet_close_coverage": 0.9, "bet_clv_t": 2.6}
    assert ms.blockers("HOLDOUT", {}, own, method, pool=pool) == []
    assert ms.blockers("HOLDOUT", {}, dict(own, bets=4), method, pool=pool) == ["HOLDOUT_NEED_6_MORE_BETS_IN_THIS_SCOPE"]
    assert ms.blockers("HOLDOUT", {}, own, method, pool=dict(pool, bet_clv_t=1.2)) == [
        "HOLDOUT_BET_CLV_NOT_2_STANDARD_ERRORS_ABOVE_ZERO"]
    assert ms.blockers("HOLDOUT", {}, own, method, pool=dict(pool, bets=60)) == ["HOLDOUT_NEED_15_MORE_BETS"]
    assert ms.blockers("HOLDOUT", {}, own, method, pool=dict(pool, complete=False)) == ["HOLDOUT_POOL_INCOMPLETE"]
    assert ms.blockers("HOLDOUT", {}, own, method, pool=None) == ["HOLDOUT_POOL_INCOMPLETE"]


IDS = {"NBA": ("model", "calibration"), "NCAAB": ("model-cbb", "calibration-cbb")}
CLOSES = {}


def _trained(path, clock, sport):  # noqa: F811
    tag = sport.lower()
    model_id, calibration_id = IDS[sport]
    clock(hours=0)
    event(path, sport=sport, event_id=f"{tag}-training", start=1)
    clock(hours=2)
    result(path, sport=sport, event_id=f"{tag}-training", result_id=f"{tag}-training-result")
    clock(hours=3)
    model(path, sport=sport, model_id=model_id, result_ids=(f"{tag}-training-result",))
    event(path, sport=sport, event_id=f"{tag}-fit", start=5, clock_hour=3)
    clock(hours=6)
    result(path, sport=sport, event_id=f"{tag}-fit", result_id=f"{tag}-fit-result", observed=6, available=6)
    clock(hours=7)
    calibration(path, sport=sport, model_id=model_id, calibration_id=calibration_id, result_ids=(f"{tag}-fit-result",))


def _freeze(path, sport, rules, *, holdout_end=25):
    model_id, calibration_id = IDS[sport]
    spec = dict(validation_plan_id=f"plan-{sport.lower()}", sport=sport, market_family="SPREAD", version=1,
                supersedes_plan_id=None, model_id=model_id, calibration_id=calibration_id, training_cutoff=at(8),
                validation_start=at(9), validation_end=at(15), holdout_start=at(16), holdout_end=at(holdout_end),
                minimum_independent_sample=1, minimum_effective_sample=1,
                probability_thresholds=dict(max_brier=0.26, max_log_loss=0.715, max_calibration_error=1, min_coverage=1),
                calibration_requirements=dict(required=True),
                price_evidence_requirements=dict(min_verified_entry_coverage=1),
                clv_policy=dict(required=True, min_comparable_close_coverage=0),
                value_roi_policy=dict(min_paper_roi=0),
                deployment_criteria=dict(target_state="PROVISIONAL_VALIDATED", maturity_rules={"synthetic_only": True}),
                independence_method="ONE_EARLIEST_PREDICTION_PER_EVENT_V1")
    spec[ms.KEY] = rules
    return pe.freeze_validation_plan(path, spec)


def _game(path, clock, sport, name, start, close_odds):  # noqa: F811
    model_id, calibration_id = IDS[sport]
    clock(hours=start - 2)
    event(path, sport=sport, event_id=name, start=start, clock_hour=start - 2)
    quote(path, sport=sport, event_id=name, quote_id=f"{name}-home", stamp=start - 2)
    quote(path, sport=sport, event_id=name, quote_id=f"{name}-away", stamp=start - 2, selection="Away", line=2.5,
          raw_line=-2.5)
    prediction(path, sport=sport, event_id=name, quote_id=f"{name}-home", observation_id=f"{name}-pred",
               stamp=start - 2, model_id=model_id, calibration_id=calibration_id)
    CLOSES[name] = close_odds
    clock(hours=start + 1)
    result(path, sport=sport, event_id=name, result_id=f"{name}-result", observed=start + 1, available=start + 1)


def _rules(**changes):
    """The real v2 rules, with the sample minimums cut down to a synthetic handful of games."""
    rules = ms.methodology_v2(pool=ms.pool_scopes("NBA", "SPREAD"), max_entry_overround=1.15)
    rules.update(min_bets=4, **changes)
    rules["pool"] = dict(rules["pool"], min_scope_bets=2)
    return rules


@pytest.fixture
def closes(monkeypatch):
    CLOSES.clear()
    monkeypatch.setattr(pe, "_comparable_close", lambda db, quote, as_of: (
        {"decimal_odds": CLOSES[quote["event_id"]], "close_id": "close-" + quote["event_id"]}
        if CLOSES.get(quote["event_id"]) else None))


def _season(path, clock, nba_closes, cbb_closes):  # noqa: F811
    for sport, odds in (("NBA", nba_closes), ("NCAAB", cbb_closes)):
        tag = sport.lower()
        for index, close_odds in enumerate(odds):
            _game(path, clock, sport, f"{tag}-val-{index}", 10 + index, close_odds)
            _game(path, clock, sport, f"{tag}-hold-{index}", 17 + index, close_odds)


def test_pooled_plans_pass_together_on_bets_neither_scope_has_alone(tmp_path, clock, closes):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    for sport in ("NBA", "NCAAB"):
        _trained(path, clock, sport)
    clock(hours=8)
    for sport in ("NBA", "NCAAB"):
        _freeze(path, sport, _rules())
    _season(path, clock, [1.80, 1.84], [1.82, 1.86])       # every bet beats its close, by different amounts
    clock(hours=26)
    for sport in ("NBA", "NCAAB"):
        report = pe.evaluate_validation_plan(path, f"plan-{sport.lower()}")
        pool = report["market_skill_pool"]
        assert report["market_skill"]["holdout"]["bets"] == 2             # below min_bets 4 on its own
        assert pool["holdout"]["bets"] == 4 and pool["holdout"]["complete"] and pool["holdout"]["problems"] == []
        assert set(pool["holdout"]["scopes"]) == {"NBA:SPREAD", "NCAAB:SPREAD"}
        assert pool["holdout"]["scopes"][f"{sport}:SPREAD"]["bets"] == 2  # each scope's own results are reported
        assert pool["holdout"]["bet_clv_t"] > 2 and pool["validation"]["bet_clv_t"] > 2
        assert report["blockers"] == [] and report["status"] == "VALIDATION_PASSED"
        # The staking input is the statistic the gate passed on: the pool's holdout lower bound.
        assert report["metrics"]["price_clv_lower_95"] == pool["holdout"]["bet_clv_lower_95"] > 0


def test_a_pool_whose_bets_do_not_beat_the_close_fails_in_every_scope(tmp_path, clock, closes):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    for sport in ("NBA", "NCAAB"):
        _trained(path, clock, sport)
    clock(hours=8)
    for sport in ("NBA", "NCAAB"):
        _freeze(path, sport, _rules())
    _season(path, clock, [1.80, 2.05], [2.00, 1.95])       # mean CLV near zero
    clock(hours=26)
    for sport in ("NBA", "NCAAB"):
        report = pe.evaluate_validation_plan(path, f"plan-{sport.lower()}")
        assert report["status"] == "UNVALIDATED"
        assert {"VALIDATION_BET_CLV_NOT_2_STANDARD_ERRORS_ABOVE_ZERO",
                "HOLDOUT_BET_CLV_NOT_2_STANDARD_ERRORS_ABOVE_ZERO"} <= set(report["blockers"])


def test_a_companion_with_other_rules_or_windows_or_no_plan_leaves_the_pool_incomplete(tmp_path, clock, closes):  # noqa: F811
    for case, companion in (("rules", dict(rules=_rules(min_edge=0.01))), ("windows", dict(rules=_rules(), holdout_end=30)),
                            ("missing", None)):
        path = tmp_path / f"{case}.sqlite3"
        for sport in ("NBA", "NCAAB"):
            _trained(path, clock, sport)
        clock(hours=8)
        _freeze(path, "NBA", _rules())
        if companion:
            _freeze(path, "NCAAB", companion["rules"], holdout_end=companion.get("holdout_end", 25))
        _season(path, clock, [1.80, 1.84], [1.82, 1.86])
        clock(hours=31)
        report = pe.evaluate_validation_plan(path, "plan-nba")
        assert report["market_skill_pool"]["holdout"]["complete"] is False, case
        assert {"VALIDATION_POOL_INCOMPLETE", "HOLDOUT_POOL_INCOMPLETE"} <= set(report["blockers"]), case
        assert report["status"] == "UNVALIDATED"


def test_v1_plan_reports_are_unchanged_by_v2(tmp_path, clock, closes):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    _trained(path, clock, "NBA")
    clock(hours=8)
    _freeze(path, "NBA", ms.methodology())
    _game(path, clock, "NBA", "nba-val-0", 10, 1.80)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan-nba")
    assert "market_skill_pool" not in report and "bet_clv_t" not in report["market_skill"]["validation"]
    assert "VALIDATION_NEED_49_MORE_BETS" in report["blockers"]


def test_v2_draft_carries_its_choices_through_the_freeze_inputs():
    from app_core import prospective_calibration as pc
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2026-11-20T00:00:00+00:00", "validation_end": "2026-12-20T00:00:00+00:00",
            "holdout_start": "2026-12-20T00:00:00+00:00", "holdout_end": "2027-01-19T00:00:00+00:00",
            ms.KEY: ms.methodology_v2(min_edge=0.01, entry_rule=ms.FIRST_EDGE_CAPTURE, min_bets=100,
                                      entry_books=["Novig"], max_entry_overround=1.15,
                                      pool=ms.pool_scopes("NCAAB", "SPREAD"))}
    inputs = pc.freeze_inputs(spec)
    assert inputs["methodology"] == "market_skill_v2" and inputs["min_bets"] == 100 and inputs["min_edge"] == 0.01
    assert inputs["pool"] == ["NBA:SPREAD", "NCAAB:SPREAD"] and inputs["max_entry_overround"] == 1.15
    assert inputs["entry_rule"] == ms.FIRST_EDGE_CAPTURE and inputs["entry_books"] == ["Novig"]
    assert "push_rule" not in inputs and inputs["validation_days"] == 30 and inputs["holdout_days"] == 30


def test_the_design_meets_its_acceptance_check():
    """Roadmap M3: a true 1% CLV passes at least 90% of the time, no edge at most 5% (stylized simulation)."""
    power = importlib.import_module("scripts.plan_gate_power")
    assert power.v2_plan(ms.DEFAULT_MIN_BETS, 0.01, sims=1500) >= 0.90
    assert power.v2_plan(ms.DEFAULT_MIN_BETS, 0.0, sims=1500) <= 0.05


def test_draft_and_freeze_a_v2_plan_sized_by_bets(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app_core import prospective_calibration as pc
    from app_core import prospective_research_models as models
    from app_core.prospective_validation_plans import freeze_current_validation_plans
    from tests.test_prospective_research_models import _at, _event
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, state, "NBA", index)
    state[0] = _at(71)
    model_id = models.train_scope(path, "NBA", "SPREAD")["model_id"]
    for index in range(72, 120):
        _event(path, state, "NBA", index)
        state[0] = _at(index, 1)
        pc.predict_candidate(path, "NBA", "SPREAD", now=state[0])
    state[0] = _at(121)
    pc.settle_scope(path, "NBA", "SPREAD")
    assert pc.fit_calibration(path, "NBA", "SPREAD", model_id, now=state[0])["status"] == "FITTED"
    now = state[0]
    state[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
    freeze_current_validation_plans(path, source_commit="b" * 40)
    state[0] = now
    start = (now + timedelta(days=1)).isoformat()
    draft = lambda **kwargs: pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=start,  # noqa: E731
                                                 methodology="market_skill_v2", **kwargs)
    with pytest.raises(ValueError, match="only a Pinnacle plan may pool"):
        draft(rate=2, pool=ms.pool_scopes("NBA", "SPREAD"))
    with pytest.raises(ValueError, match="give validation_days"):
        draft()                                            # a research model has no bet-rate estimate
    with pytest.raises(ValueError, match="always counts a push"):
        draft(rate=2, push_rule=ms.PUSH_ZERO_RETURN)
    with pytest.raises(ValueError, match="apply only to the market_skill_v2"):
        pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=start, rate=7, methodology="market_skill",
                            min_bets=50)
    spec = draft(rate=2, min_bets=50)                      # 2 bets a day
    assert spec[ms.KEY]["version"] == ms.VERSION_V2 and spec[ms.KEY]["min_bets"] == 50
    assert spec["minimum_independent_sample"] == 50 and spec["probability_thresholds"]["max_brier"] == 0.26
    days = (datetime.fromisoformat(spec["validation_end"]) - datetime.fromisoformat(spec["validation_start"])).days
    assert days == pc.window_days(50, 2) == 40             # sized by bets, not by the 200-game template
    inputs = pc.freeze_inputs(spec)
    frozen = pc.freeze_reviewed_plan(path, expected_hash=pc.plan_hash(spec), now=now, **inputs)
    report = pe.evaluate_validation_plan(path, frozen["validation_plan_id"], as_of=now)
    assert report["status"] == "UNVALIDATED" and "VALIDATION_NEED_50_MORE_BETS" in report["blockers"]
    assert not any("WORSE_THAN_MARKET" in b for b in report["blockers"])


def test_a_plan_frozen_with_the_junk_term_never_selects_a_junk_price(monkeypatch):
    import json
    from app_core import pinnacle_model
    from app_core import prospective_calibration as pc
    rules = ms.methodology_v2(max_entry_overround=1.15)
    plan = {"validation_plan_id": "p", "model_id": "m", "calibration_id": "c", "payload": json.dumps({ms.KEY: rules})}
    monkeypatch.setattr(pc, "bound_plan", lambda path, sport, market: plan)
    monkeypatch.setattr(pc, "runtime_blocker", lambda plan, *a: None)
    monkeypatch.setattr(pc.evidence, "load_record", lambda path, table, key: {"payload": "{}"})
    seen = {}
    monkeypatch.setattr(pc, "_predict", lambda *a, select=None, **k: seen.update(select=select) or {"predictions": 0})
    monkeypatch.setattr(pinnacle_model, "plan_quotes", lambda events, quotes, at, **k: seen.update(quotes=quotes) or {})

    def q(name, selection, line, odds):
        return {"quote_id": name, "event_id": "g", "sportsbook": "novig", "selection": selection, "line": line,
                "decimal_odds": odds, "market_family": "SPREAD", "quote_timestamp": "t", "quote_verified": 1}
    quotes = [q("ok-h", "H", -3.5, 1.95), q("ok-a", "A", 3.5, 1.93), q("junk-h", "H", -9.5, 1.4), q("junk-a", "A", 9.5, 1.4)]
    pc.predict_bound("path", "NBA", "SPREAD")
    seen["select"]({}, quotes, None)
    assert [x["quote_id"] for x in seen["quotes"]] == ["ok-h", "ok-a"]
    # A plan frozen without the term (every plan before D29) still sees every quote.
    plan["payload"] = json.dumps({ms.KEY: dict(ms.methodology(entry_books=["Novig"]))})
    pc.predict_bound("path", "NBA", "SPREAD")
    seen["select"]({}, quotes, None)
    assert len(seen["quotes"]) == 4


def test_nhl_joins_a_pool_only_when_named_and_never_mixes_sides_with_totals():
    """Owner 2026-11-09: NHL may widen the lag pool. Its side market is the puck line."""
    assert ms.pool_scopes("NBA", "SPREAD") == ["NBA:SPREAD", "NCAAB:SPREAD"]                       # the default is unchanged
    assert ms.pool_scopes("NBA", "SPREAD", ["NBA", "NCAAB", "NHL"]) == ["NBA:SPREAD", "NCAAB:SPREAD", "NHL:PUCK_LINE"]
    assert ms.pool_scopes("NHL", "PUCK_LINE", ["NBA", "NHL"]) == ["NBA:SPREAD", "NHL:PUCK_LINE"]
    assert ms.pool_scopes("NHL", "TOTAL", ["NBA", "NCAAB", "NHL"]) == ["NBA:TOTAL", "NCAAB:TOTAL", "NHL:TOTAL"]
    for bad in (("NHL", "PUCK_LINE", None), ("NHL", "SPREAD", ["NBA", "NHL"]), ("NBA", "PUCK_LINE", ["NBA", "NHL"]),
                ("MLB", "TOTAL", ["NBA", "MLB"]), ("NBA", "TOTAL", ["NBA"])):
        with pytest.raises(ValueError):
            ms.pool_scopes(*bad)
    rules = ms.methodology_v2(pool=ms.pool_scopes("NCAAB", "TOTAL", ["NBA", "NCAAB", "NHL"]))
    assert rules["pool"]["scopes"] == ["NBA:TOTAL", "NCAAB:TOTAL", "NHL:TOTAL"]
    for bad in (["NBA:SPREAD", "NHL:TOTAL"], ["NBA:TOTAL", "NHL:PUCK_LINE"], ["NHL:PUCK_LINE"],
                ["NBA:SPREAD", "NHL:SPREAD"], ["NBA:SPREAD", "MLB:RUN_LINE"], ["NBA:SPREAD", "NBA:TOTAL"]):
        with pytest.raises(ValueError):
            ms.methodology_v2(pool=bad)
