"""A first-edge plan frozen to enter only close to the start (owner 2026-11-11).

The 80% close gate needs a certified close at the bet's own book and line. In the week of
2026-11-09, 64% of NCAAB spread bets at Novig had one at any entry time and 80% within the last
60 minutes, so the owner asked for a plan that enters only then. SYNTHETIC; not live evidence.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from app_core import market_skill_validation as ms
from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from tests.test_first_edge_entry import _first_edge_game, _plan_with
from tests.test_pinnacle_plan_selection import EVENT, LATER, both
from tests.test_prospective_evidence import _trained_market, clock  # noqa: F401  (clock is a fixture)

LATE = "2030-01-01T19:10:00+00:00"          # 50 minutes before the 20:00 start
LATE_STAMP = "2030-01-01T19:08:00+00:00"
LAST = "2030-01-01T19:40:00+00:00"          # 20 minutes before the start
LAST_STAMP = "2030-01-01T19:38:00+00:00"
NOW = datetime(2030, 1, 1, 19, 50, tzinfo=timezone.utc)


def test_the_window_is_frozen_only_when_chosen_and_only_with_first_edge():
    plain = ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75)
    assert "entry_within_minutes" not in plain                        # plans frozen before keep their hash
    assert ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75, entry_within_minutes=None) == plain
    late = ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75, entry_within_minutes=60)
    assert late == dict(plain, entry_within_minutes=60)
    with pytest.raises(ValueError, match="entry_within_minutes"):
        ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75, entry_within_minutes=30)
    with pytest.raises(ValueError, match="FIRST_EDGE_CAPTURE"):
        ms.methodology_v2(0.01, ["Novig"], None, 75, entry_within_minutes=60)
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": late}
    assert pc.freeze_inputs(spec)["entry_within_minutes"] == 60        # the freeze redrafts the same window
    assert "entry_within_minutes" not in pc.freeze_inputs(dict(spec, market_skill_methodology=plain))


def test_only_late_captures_can_be_the_entry():
    # 16:00: Pinnacle moves and FanDuel lags (an edge, four hours early). 19:10: no edge. 19:40: an edge again.
    quotes = (both("pinnacle", -3.5, -110, -110) + both("FanDuel", -3.5, -110, -110)
              + both("pinnacle", -3.5, -130, +110, received=LATER) + both("FanDuel", -3.5, -110, -110, received=LATER)
              + both("pinnacle", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP)
              + both("FanDuel", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP)
              + both("pinnacle", -3.5, -130, +110, received=LAST, stamp=LAST_STAMP)
              + both("FanDuel", -3.5, -110, -110, received=LAST, stamp=LAST_STAMP))
    events = {"e": EVENT}
    rule = dict(entry_rule=ms.FIRST_EDGE_CAPTURE, min_edge=0.02)
    any_time = pm.plan_quotes(events, quotes, NOW, **rule)["e"]
    assert pm.capture_time(EVENT, any_time) == pe._time(LATER, "t")              # the early edge
    kept = pm.late_captures(events, quotes, 60)
    assert {q["raw_source"].get("response_received_at") for q in kept} == {LATE, LAST}
    late = pm.plan_quotes(events, kept, NOW, **rule)["e"]
    assert pm.capture_time(EVENT, late) == pe._time(LAST, "t")                   # the first late edge
    # Before the late edge appears, the plan predicts at the first late capture (no bet yet).
    before = datetime(2030, 1, 1, 19, 20, tzinfo=timezone.utc)
    assert pm.capture_time(EVENT, pm.plan_quotes(events, kept, before, **rule)["e"]) == pe._time(LATE, "t")
    # More than an hour out, the plan predicts nothing at all.
    assert pm.plan_quotes(events, pm.late_captures(events, quotes[:8], 60), NOW, **rule) == {}


def test_bound_predictions_read_only_the_late_captures(monkeypatch):
    rules = ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75, entry_within_minutes=60)
    plan = {"validation_plan_id": "p", "model_id": "m", "calibration_id": "c", "payload": json.dumps({ms.KEY: rules})}
    monkeypatch.setattr(pc, "bound_plan", lambda path, sport, market: plan)
    monkeypatch.setattr(pc, "runtime_blocker", lambda plan, *a: None)
    monkeypatch.setattr(pc.evidence, "load_record", lambda path, table, key: {"payload": "{}"})
    seen = {}
    monkeypatch.setattr(pc, "_predict", lambda *a, select=None, **k: seen.update(select=select) or {"predictions": 0})
    monkeypatch.setattr(pm, "plan_quotes", lambda events, quotes, at, **k: seen.update(quotes=quotes, rules=k) or {})
    quotes = (both("pinnacle", -3.5, -110, -110, received=LATER) + both("novig", -3.5, -110, -110, received=LATER)
              + both("pinnacle", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP)
              + both("novig", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP))
    pc.predict_bound("path", "NCAAB", "SPREAD")
    seen["select"]({"e": EVENT}, quotes, NOW)
    assert len(seen["quotes"]) == 4
    assert {q["raw_source"]["response_received_at"] for q in seen["quotes"]} == {LATE}
    assert seen["rules"]["books"] == frozenset({"Novig"}) and seen["rules"]["entry_rule"] == ms.FIRST_EDGE_CAPTURE
    # The same plan without the window reads every capture.
    plan["payload"] = json.dumps({ms.KEY: ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75)})
    pc.predict_bound("path", "NCAAB", "SPREAD")
    seen["select"]({"e": EVENT}, quotes, NOW)
    assert len(seen["quotes"]) == 8


def test_a_bet_entered_before_the_window_is_not_the_plans_bet(tmp_path, clock):  # noqa: F811
    from contextlib import closing
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan_with(path, ms.FIRST_EDGE_CAPTURE)
    _first_edge_game(path, clock, "validation", 12)      # the clearing prediction is 60 minutes before the start
    clock(hours=26)
    plan = pe.load_record(path, "prospective_validation_plan", "plan")
    rules = json.loads(plan["payload"])[ms.KEY]
    with closing(pe._reader(path)) as db:
        first = pe._stored(db, "prospective_quote", "quote_id", "validation-home")
        assert ms.first_edge_bet(db, plan, first, rules, pe._clock()) is not None
        assert ms.first_edge_bet(db, plan, first, dict(rules, entry_within_minutes=60), pe._clock()) is not None
        assert ms.first_edge_bet(db, plan, first, dict(rules, entry_within_minutes=59), pe._clock()) is None


def test_replay_estimate_counts_only_late_captures(monkeypatch):
    from app_core import prospective_research_models as research
    quotes = (both("pinnacle", -3.5, -130, +110, received=LATER) + both("FanDuel", -3.5, -110, -110, received=LATER)
              + both("pinnacle", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP)
              + both("FanDuel", -3.5, -110, -110, received=LATE, stamp=LATE_STAMP))
    for index, item in enumerate(quotes):
        item["quote_id"] = f"q{index}"
    monkeypatch.setattr(research, "_read_scope", lambda path, sport, family: ({"e": EVENT}, quotes, None))
    after = datetime(2030, 1, 1, 21, tzinfo=timezone.utc)

    def first_edge_bets(**kwargs):
        out = pm.replay_estimate("db", "NCAAB", "SPREAD", now=after, edges=(0.02,), **kwargs)
        row = next(v for v in out["variants"] if v["entry_rule"] == ms.FIRST_EDGE_CAPTURE)
        return out, row["by_edge"]["2%"]["bets"]
    out, bets = first_edge_bets()
    assert bets == 1 and out["entry_within_minutes"] is None        # the early edge
    out, bets = first_edge_bets(entry_within_minutes=60)
    assert bets == 0 and out["entry_within_minutes"] == 60 and out["plan_games"] == 1


def test_draft_and_workflow_carry_the_window():
    import importlib
    bound_plan = importlib.import_module("scripts.bound_plan")
    rules = ms.methodology_v2(0.01, ["Novig"], ms.FIRST_EDGE_CAPTURE, 75, entry_within_minutes=60)
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/bound-plan.yml").read_text(encoding="utf-8")
    assert "options: [any, last_60_minutes]" in text and "--entry-within-minutes 60" in text
    source = Path(bound_plan.__file__).read_text(encoding="utf-8")
    assert "entry_within_minutes=args.entry_within_minutes" in source and "- entry window: only captures" in source
    assert rules["entry_within_minutes"] in ms.ALLOWED_ENTRY_WITHIN_MINUTES
