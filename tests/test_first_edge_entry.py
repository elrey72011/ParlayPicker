"""FIRST_EDGE_CAPTURE: a Pinnacle plan enters at the first capture that clears its bet threshold (owner 2026-11-08)."""
from datetime import datetime, timezone

import pytest

from app_core import market_skill_validation as ms
from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from tests.test_pinnacle_plan_selection import EVENT, LATER, NOW, both
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    _trained_market, at, clock, event, prediction, provider_event, quote, result)

LATEST = "2030-01-01T18:00:00+00:00"
LATEST_STAMP = "2030-01-01T17:58:00+00:00"


def _quotes():
    # First capture: Pinnacle and bookA agree (no edge). Later: Pinnacle moves to about 54/46 on the
    # home side, FanDuel lags at -110 / -110 (home EV about +3.6%). Latest: both back at -110 / -110.
    return (both("pinnacle", -3.5, -110, -110) + both("FanDuel", -3.5, -110, -110)
            + both("pinnacle", -3.5, -130, +110, received=LATER) + both("FanDuel", -3.5, -110, -110, received=LATER)
            + both("pinnacle", -3.5, -110, -110, received=LATEST, stamp=LATEST_STAMP)
            + both("FanDuel", -3.5, -110, -110, received=LATEST, stamp=LATEST_STAMP))


def test_first_edge_rule_enters_at_the_first_capture_that_clears_the_threshold():
    quotes = _quotes()
    earliest = pm.plan_quotes({"e": EVENT}, quotes, NOW)["e"]
    assert pm.capture_time(EVENT, earliest) == pe._time(EVENT["observed_at"], "t")      # the default: no edge
    first = pm.plan_quotes({"e": EVENT}, quotes, NOW, entry_rule=ms.FIRST_EDGE_CAPTURE, min_edge=0.02)["e"]
    assert pm.capture_time(EVENT, first) == pe._time(LATER, "t") and first["sportsbook"] == "FanDuel"
    # Nothing clears 10%: the rule falls back to the earliest shared-line capture, so every game is predicted.
    fallback = pm.plan_quotes({"e": EVENT}, quotes, NOW, entry_rule=ms.FIRST_EDGE_CAPTURE, min_edge=0.10)["e"]
    assert pm.capture_time(EVENT, fallback) == pe._time(EVENT["observed_at"], "t")
    # The plan's probability decides: a calibration that pulls everything to 50% removes the edge.
    flat = pm.plan_quotes({"e": EVENT}, quotes, NOW, entry_rule=ms.FIRST_EDGE_CAPTURE, min_edge=0.02,
                          adjust=lambda p: 0.5)["e"]
    assert pm.capture_time(EVENT, flat) == pe._time(EVENT["observed_at"], "t")
    # Prices captured after ``now`` are never used.
    early = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
    assert pm.capture_time(EVENT, pm.plan_quotes({"e": EVENT}, quotes, early, entry_rule=ms.FIRST_EDGE_CAPTURE,
                                                 min_edge=0.02)["e"]) == pe._time(EVENT["observed_at"], "t")


def test_entry_rule_is_frozen_only_when_chosen_so_frozen_plans_keep_their_hash():
    assert "entry_rule" not in ms.methodology(0.02, pm.PLAN_BOOKS)
    assert ms.methodology(0.02, pm.PLAN_BOOKS, ms.EARLIEST_SHARED_LINE) == ms.methodology(0.02, pm.PLAN_BOOKS)
    rules = ms.methodology(0.02, pm.PLAN_BOOKS, ms.FIRST_EDGE_CAPTURE)
    assert rules["entry_rule"] == "FIRST_EDGE_CAPTURE"
    with pytest.raises(ValueError, match="entry_rule"):
        ms.methodology(0.02, None, "LATEST")
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": rules}
    assert pc.freeze_inputs(spec)["entry_rule"] == "FIRST_EDGE_CAPTURE"   # the freeze redrafts the same plan
    assert "entry_rule" not in pc.freeze_inputs(dict(spec, market_skill_methodology=ms.methodology(0.02)))


def test_replay_shows_each_entry_rule_on_the_raw_and_calibrated_probability(monkeypatch):
    now = datetime(2030, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr("app_core.prospective_research_models._read_scope",
                        lambda *a: ({"e": EVENT}, _quotes(), [["never read"]]))
    out = pm.replay_estimate("db", "NCAAB", "SPREAD", now=now, calibration={"a": 0.0, "b": 0.0, "decided": 40})
    assert out["by_edge"]["2%"]["bets"] == 0 and out["calibration"]["decided"] == 40
    found = {(v["entry_rule"], v["probability"]): v["by_edge"] for v in out["variants"]}
    assert found[("EARLIEST_SHARED_LINE", "PINNACLE_NO_VIG")]["2%"]["bets"] == 0
    assert found[("FIRST_EDGE_CAPTURE", "PINNACLE_NO_VIG")]["2%"]["bets"] == 1
    # b = 0 maps every probability to 50%: no bet at -110 under either rule.
    assert found[("FIRST_EDGE_CAPTURE", "CALIBRATED")]["2%"]["bets"] == 0
    assert "CALIBRATED" not in str(pm.replay_estimate("db", "NCAAB", "SPREAD", now=now)["variants"])


def test_draft_shows_the_entry_rule_and_the_variant_table():
    import importlib
    bound_plan = importlib.import_module("scripts.bound_plan")
    rules = ms.methodology(0.02, pm.PLAN_BOOKS, ms.FIRST_EDGE_CAPTURE)
    plan = {"validation_plan_id": "p3", "supersedes_plan_id": "p2", "model_id": "m", "calibration_id": "c",
            "validation_start": "s", "validation_end": "e", "holdout_start": "s", "holdout_end": "e",
            "minimum_independent_sample": 200, "market_skill_methodology": rules}
    replay = {"sport": "NCAAB", "market_family": "SPREAD", "status": "ESTIMATED", "days_observed": 2.4,
              "games_started": 150, "plan_games": 41, "plan_share": 0.27, "plan_games_per_day": 17.4,
              "days_for_min_sample": 11.5, "min_sample": 200, "min_bets": 50,
              "by_edge": {"2%": {"bets": 0, "bets_per_day": 0.0, "days_for_min_bets": None, "days_per_window": None}},
              "calibration": {"a": 0.01, "b": 0.97, "decided": 40},
              "variants": [{"entry_rule": "FIRST_EDGE_CAPTURE", "probability": "CALIBRATED", "by_edge": {
                  "2%": {"bets": 12, "bets_per_day": 5.0, "days_for_min_bets": 10.0, "days_per_window": 11.5}}}]}
    text = bound_plan.markdown({"mode": "DRAFT_READ_ONLY", "markets": [], "replay_estimate": replay, "draft": {
        "sport": "NCAAB", "market_family": "SPREAD", "status": "DRAFTED_NOT_FROZEN", "plan": plan,
        "plan_hash": "h", "freeze_inputs": {"entry_rule": "FIRST_EDGE_CAPTURE"}}})
    assert "- entry rule: the first capture where one of its books posts Pinnacle's line and clears" in text
    assert "| FIRST_EDGE_CAPTURE | CALIBRATED | 2% | 12 | 5.0 | 10.0 | 11.5 |" in text
    assert "a = 0.01, b = 0.97" in text


def _later_quote(path, event_id, quote_id, stamp, selection, line, start):
    """A quote from a later capture of a stored event: it carries its own provider response."""
    full = provider_event(event_id, start, stamp)
    raw = {"event_response": full, "market": full["bookmakers"][0]["markets"][0], "response_received_at": at(stamp)}
    return pe.insert_quote(path, dict(quote_id=quote_id, event_id=event_id, sport="NBA", market_family="SPREAD",
        selection=selection, line=line, american_odds=-110, decimal_odds=1.909090909, sportsbook="TestBook",
        quote_timestamp=at(stamp), quote_source="THE_ODDS_API", quote_verified=True,
        source_id=f"{event_id}:TestBook:spreads:{selection}:{at(stamp)}", raw_source=raw))


def _first_edge_game(path, clock, name, start):  # noqa: F811
    clock(hours=start - 3)
    event(path, event_id=name, start=start, clock_hour=start - 3)
    # First prediction: 52% at -110 (EV -0.7%, no bet). Later prediction at a later capture: 56% (EV +6.9%).
    for suffix, stamp, mean in (("", start - 3, .52), ("-later", start - 1, .56)):
        clock(hours=stamp)
        if not suffix:
            quote(path, event_id=name, quote_id=f"{name}-home", stamp=stamp)
            quote(path, event_id=name, quote_id=f"{name}-away", stamp=stamp, selection="Away", line=2.5,
                  raw_line=-2.5)
        else:
            _later_quote(path, name, f"{name}-home{suffix}", stamp, "Home", -2.5, start)
            _later_quote(path, name, f"{name}-away{suffix}", stamp, "Away", 2.5, start)
        prediction(path, event_id=name, quote_id=f"{name}-home{suffix}", observation_id=f"{name}-pred{suffix}",
                   stamp=stamp, mean=mean, loss=1 - mean)
    clock(hours=start + 1)
    result(path, event_id=name, result_id=f"{name}-result", observed=start + 1, available=start + 1, outcome="WIN")


def _plan_with(path, entry_rule):
    from tests.test_market_skill_validation import _plan
    import tests.test_market_skill_validation as module
    original = module.ms.template
    module.ms.template = lambda spec: original(spec, 0.02, None, entry_rule)
    try:
        return _plan(path)
    finally:
        module.ms.template = original


@pytest.mark.parametrize("entry_rule, bets", [(None, 0), (ms.FIRST_EDGE_CAPTURE, 1)])
def test_first_edge_plans_bet_on_the_first_prediction_that_clears_the_edge(tmp_path, clock, entry_rule, bets):  # noqa: F811
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan_with(path, entry_rule)
    _first_edge_game(path, clock, "validation", 12)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    skill = report["market_skill"]["validation"]
    # The probability gate always scores the first prediction (52%); only the bets differ.
    assert skill["with_market_reference"] == 1 and skill["model_brier"] == pytest.approx(0.48 ** 2)
    assert skill["bets"] == bets
    if bets:
        assert skill["bet_roi"] == pytest.approx(0.909090909)                # home at -110, won
