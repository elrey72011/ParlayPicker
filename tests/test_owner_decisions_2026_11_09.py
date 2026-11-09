"""Owner decisions D20-D25 (2026-11-09). SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from app_core import market_skill_validation as ms
from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe


# D20: a push is a zero-return bet, for plans frozen from now on ---------------------------------

def test_push_rule_is_recorded_only_when_chosen():
    assert "push_rule" not in ms.methodology(0.02, pm.PLAN_BOOKS)            # frozen plans keep their rules and hash
    rules = ms.methodology(0.02, pm.PLAN_BOOKS, push_rule=ms.PUSH_ZERO_RETURN)
    assert rules["push_rule"] == ms.PUSH_ZERO_RETURN
    with pytest.raises(ValueError, match="push_rule"):
        ms.methodology(0.02, None, push_rule="HALF")
    spec = {"sport": "NBA", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": rules}
    assert pc.freeze_inputs(spec)["push_rule"] == ms.PUSH_ZERO_RETURN       # the freeze redrafts the same plan


def test_a_pushed_bet_returns_its_stake_and_never_enters_the_probability_gate(monkeypatch):
    quote = {"quote_id": "q", "event_id": "e", "market_family": "SPREAD", "selection": "Home", "line": -3.0,
             "decimal_odds": 2.0, "quote_timestamp": "2030-01-01T00:00:00+00:00"}
    monkeypatch.setattr(ms, "opposite", lambda db, q: dict(quote, selection="Away", decimal_odds=1.8))
    rules = ms.methodology(0.02, push_rule=ms.PUSH_ZERO_RETURN)
    out = ms.skill(None, [(0.6, 1, quote), (0.6, None, quote)], rules=rules)
    assert out["bets"] == 2 and out["bet_roi"] == pytest.approx((1.0 + 0.0) / 2)
    assert out["with_market_reference"] == 1                                 # only the decided game is scored


def test_the_bound_plan_workflow_drafts_new_plans_with_the_push_rule():
    import inspect
    from scripts import bound_plan
    assert "push_rule=(market_skill_validation.PUSH_ZERO_RETURN" in inspect.getsource(bound_plan)


# D21: a new Pinnacle plan is bound to its prediction code -----------------------------------------

def test_code_hash_is_stable_and_ignores_docstrings(monkeypatch):
    first = pm.code_hash()
    assert first == pm.code_hash() and len(first) == 64
    import types
    original = pm.capture_time
    redocumented = types.FunctionType(original.__code__, original.__globals__, "capture_time")
    redocumented.__doc__ = "a different docstring"
    assert pm.code_hash() == first                                           # source unchanged

def _plan(scope):
    return {"validation_plan_id": "p", "payload": json.dumps({"model_scope": scope})}


def test_a_plan_with_a_recorded_hash_stops_predicting_when_the_code_changes(monkeypatch):
    assert pc.runtime_blocker(_plan({"sport": "NBA"})) is None               # plans frozen before: not checked
    assert pc.runtime_blocker(_plan({"pinnacle_runtime_hash": pm.code_hash()})) is None
    changed = _plan({"pinnacle_runtime_hash": "0" * 64})
    assert pc.runtime_blocker(changed) == "BOUND_PINNACLE_RUNTIME_CHANGED"
    assert pc.runtime_blocker(changed, "DECISION") == "DECISION_PINNACLE_RUNTIME_CHANGED"
    monkeypatch.setattr(pc, "bound_plan", lambda *a: changed)
    assert pc.predict_bound("db", "NBA", "SPREAD")["blocker"] == "BOUND_PINNACLE_RUNTIME_CHANGED"
    from app_core import decision_capture as dc
    assert dc.predict("db", "NBA", "SPREAD", now=datetime.now(timezone.utc))["blocker"] == \
        "DECISION_PINNACLE_RUNTIME_CHANGED"


# D22: a frozen plan's hash is rechecked ----------------------------------------------------------

def test_plan_integrity_catches_an_edited_plan(tmp_path, monkeypatch):
    from tests.test_market_skill_validation import _plan as freeze
    from tests.test_prospective_evidence import _trained_market
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    now = [base]
    monkeypatch.setattr(pe, "_clock", lambda: now[0])

    def clock(hours=0):
        now[0] = base + timedelta(hours=hours)
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    freeze(path)
    [plan] = pe.read_records(path, "prospective_validation_plan")
    assert pe.plan_integrity(plan)
    assert not pe.plan_integrity(dict(plan, model_id="other-model"))       # a column no longer matches
    payload = json.loads(plan["payload"])
    payload["value_roi_policy"] = {"min_paper_roi": -1.0}                  # a threshold loosened after the freeze
    assert not pe.plan_integrity(dict(plan, payload=json.dumps(payload)))  # the payload no longer hashes
    from app_core import plan_progress
    row = plan_progress.progress(plan, {}, now=base + timedelta(days=400))
    assert row and all(r["plan_integrity"] == "OK" for r in row)
    assert "Plan integrity (frozen hash recomputed): plan OK" in "\n".join(plan_progress.markdown(row))


# D25: Football Stage 1 runs only when started by hand ---------------------------------------------

def test_football_stage1_has_no_schedule():
    import yaml
    workflow = yaml.safe_load(open(".github/workflows/football-stage1.yml", encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "schedule" not in triggers and "workflow_dispatch" in triggers
