"""Pinnacle's no-vig price as a registered model. SYNTHETIC store; no model-quality claim."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from tests.test_prospective_research_models import TEAMS, _at, _event

PIN_HOME, PIN_AWAY = -125, 105            # Pinnacle: home -2.5 at -125 / away +2.5 at +105


def _decimal(american):
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def _pin_event(path, clock, index, *, pin_line=-2.5):
    """A game with the US book's home quote and Pinnacle's two sides in the same capture."""
    home, away = TEAMS[index % 4], TEAMS[(index + 1) % 4]
    start = _at(index, 2)
    observed = start - timedelta(hours=1)
    clock[0] = observed
    us = {"key": "spreads", "last_update": observed.isoformat(), "outcomes": [
        {"name": home[0], "point": -2.5, "price": -110}, {"name": away[0], "point": 2.5, "price": -110}]}
    pin = {"key": "spreads", "last_update": observed.isoformat(), "outcomes": [
        {"name": home[0], "point": pin_line, "price": PIN_HOME}, {"name": away[0], "point": -pin_line, "price": PIN_AWAY}]}
    provider_id = f"provider-NBA-{index}"
    raw = {"id": provider_id, "sport_key": pe.ODDS_API_SPORT_KEYS["NBA"], "home_team": home[0], "away_team": away[0],
           "commence_time": start.isoformat(),
           "bookmakers": [{"key": "book", "markets": [us]}, {"key": "pinnacle", "markets": [pin]}]}
    event_id = f"event-NBA-{index}"
    pe.insert_event(path, {"event_id": event_id, "sport": "NBA", "game_id": provider_id,
                           "provider_namespace": "THE_ODDS_API", "provider_event_id": provider_id,
                           "home_team": home[0], "away_team": away[0], "home_team_id": home[1],
                           "away_team_id": away[1], "scheduled_start": start.isoformat(),
                           "observed_at": observed.isoformat(), "source_id": f"capture-{provider_id}", "raw_source": raw})
    quotes = [("book", home[0], -2.5, -110, us), ("pinnacle", home[0], pin_line, PIN_HOME, pin),
              ("pinnacle", away[0], -pin_line, PIN_AWAY, pin)]
    for book, selection, line, price, market in quotes:
        pe.insert_quote(path, {"quote_id": f"q-{event_id}-{book}-{selection}", "event_id": event_id, "sport": "NBA",
                               "market_family": "SPREAD", "selection": selection, "line": line, "american_odds": price,
                               "decimal_odds": _decimal(price), "sportsbook": book,
                               "quote_timestamp": observed.isoformat(), "quote_source": "THE_ODDS_API",
                               "quote_verified": True,
                               "source_id": f"{provider_id}:{book}:spreads:{selection}:{observed.isoformat()}",
                               "raw_source": market})
    clock[0] = start + timedelta(hours=3)
    raw_score = {"id": provider_id, "sport_key": pe.ODDS_API_SPORT_KEYS["NBA"], "home_team": home[0],
                 "away_team": away[0], "commence_time": start.isoformat(), "completed": True,
                 "last_update": (start + timedelta(hours=2)).isoformat(),
                 "scores": [{"name": home[0], "score": str(101 + index % 17)},
                            {"name": away[0], "score": str(96 + index % 13)}]}
    pe.insert_result(path, {"result_id": f"score-NBA-{index}", "event_id": event_id, "sport": "NBA",
                            "market_family": None, "selection": None, "result_source": "THE_ODDS_API",
                            "result_source_id": provider_id, "observed_at": clock[0].isoformat(),
                            "available_at": clock[0].isoformat(), "home_score": 101 + index % 17,
                            "away_score": 96 + index % 13, "outcome": None, "grading_version": 1,
                            "raw_source": raw_score})
    return event_id


@pytest.fixture
def clock(monkeypatch):
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    return state


def _research_market(path, clock):
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    return models.train_scope(path, "NBA", "SPREAD")["model_id"]


def test_fair_price_needs_the_same_line_the_window_and_no_look_ahead(tmp_path, clock):
    path = tmp_path / "SYNTHETIC.sqlite3"
    same = _pin_event(path, clock, 1)
    other = _pin_event(path, clock, 2, pin_line=-3.0)
    events = {e["event_id"]: e for e in pe.read_records(path, "prospective_event", sport="NBA")}
    quotes = pe.read_records(path, "prospective_quote", sport="NBA", market_family="SPREAD")
    us = {q["event_id"]: q for q in quotes if q["sportsbook"] == "book"}
    home, away = 1 / _decimal(PIN_HOME), 1 / _decimal(PIN_AWAY)
    expected = home / (home + away)
    late = datetime(2031, 1, 1, tzinfo=timezone.utc)
    assert pm.fair(events[same], us[same], quotes, now=late) == pytest.approx(expected)
    assert pm.fair(events[other], us[other], quotes, now=late) is None             # Pinnacle at -3, US at -2.5
    early = pe._time(us[same]["quote_timestamp"], "t") - timedelta(minutes=1)
    assert pm.fair(events[same], us[same], quotes, now=early) is None              # not yet quoted
    away_quote = dict(us[same], selection=events[same]["away_team"], line=2.5)
    assert pm.fair(events[same], away_quote, quotes, now=late) == pytest.approx(1 - expected)


def test_register_once_never_the_research_candidate(tmp_path, clock):
    path = tmp_path / "SYNTHETIC.sqlite3"
    research_id = _research_market(path, clock)
    model = pm.register(path, "NBA", "SPREAD", now=clock[0])
    assert pm.register(path, "NBA", "SPREAD", now=clock[0])["model_id"] == model["model_id"]
    assert pm.is_pinnacle(model) and model["independent_event_count"] == 0
    assert pc.candidate_model(path, "NBA", "SPREAD")["model_id"] == research_id
    with pytest.raises(ValueError):
        pm.register(path, "NHL", "PUCK_LINE", now=clock[0])


def test_calibrate_from_settled_games_then_predict_only_through_a_bound_plan(tmp_path, clock):
    path = tmp_path / "SYNTHETIC.sqlite3"
    _research_market(path, clock)
    model = pm.register(path, "NBA", "SPREAD", now=clock[0])
    for index in range(72, 72 + 48):
        _pin_event(path, clock, index)
        clock[0] = _at(index, 1)
        pc.predict_candidate(path, "NBA", "SPREAD", now=clock[0])    # the research cycle's own predictions
    clock[0] = _at(121)
    pc.settle_scope(path, "NBA", "SPREAD")
    assert pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0], min_decided=10_000)["status"] == \
        "INSUFFICIENT_EVIDENCE"
    fitted = pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0])
    assert fitted["status"] == "FITTED" and fitted["decided"] == 48
    assert not [p for p in pe.read_records(path, "prospective_prediction", sport="NBA", market_family="SPREAD")
                if p["model_id"] == model["model_id"]]                     # calibration recorded no predictions
    # Bind it with a market-skill plan (the owner's reviewed freeze), then the cycle predicts with it.
    now = clock[0]
    clock[0] = datetime(2026, 10, 1, tzinfo=timezone.utc)
    from app_core.prospective_validation_plans import freeze_current_validation_plans
    freeze_current_validation_plans(path, source_commit="b" * 40)
    clock[0] = now
    spec = pc.draft_bound_plan(path, "NBA", "SPREAD", validation_start=(now + timedelta(days=1)).isoformat(),
                               rate=7, methodology="market_skill", model_id=model["model_id"])
    assert spec["calibration_id"] == fitted["calibration_id"]
    pc.freeze_reviewed_plan(path, expected_hash=pc.plan_hash(spec), now=now, **pc.freeze_inputs(spec))
    future = 72 + 48 + 3
    event_id = _pin_event(path, clock, future)
    clock[0] = _at(future, 1)
    made = pc.predict_bound(path, "NBA", "SPREAD", now=clock[0])
    assert made["predictions"] == 1 and made["blocker"] is None
    row = next(p for p in pe.read_records(path, "prospective_prediction", sport="NBA", market_family="SPREAD")
               if p["event_id"] == event_id and p["model_id"] == model["model_id"])
    assert row["calibration_id"] == fitted["calibration_id"] and 0 < row["mean_probability"] < 1
    assert row["mean_probability"] + row["loss_probability"] == pytest.approx(1)


def test_cycle_registers_and_calibrates_only_where_pinnacle_is_captured(tmp_path, clock, monkeypatch):
    path = tmp_path / "SYNTHETIC.sqlite3"
    monkeypatch.delenv("PARLAYPICKER_PINNACLE_SPORTS", raising=False)
    assert pc.pinnacle_cycle(path, "NBA", "SPREAD", now=_at(1)) == {"status": "OFF"}
    monkeypatch.setenv("PARLAYPICKER_PINNACLE_SPORTS", "NBA,NCAAB")
    clock[0] = _at(1)
    out = pc.pinnacle_cycle(path, "NBA", "SPREAD", now=_at(1))
    assert out["status"] == "INSUFFICIENT_EVIDENCE" and out["model_id"].startswith("pinnacle-model:NBA:SPREAD")
    assert pc.pinnacle_cycle(path, "NHL", "PUCK_LINE", now=_at(1)) == {"status": "OFF"}
