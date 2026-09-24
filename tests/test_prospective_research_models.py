"""Synthetic provider-shaped chronology checks; never live model evidence."""
from datetime import datetime, timedelta, timezone
from copy import deepcopy
import json
import math

import pytest

from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core.odds_market_store import for_sport
from app_core import odds_research_adapter as odds


BASE = datetime(2030, 1, 1, tzinfo=timezone.utc)
TEAMS = (("A", "team-a"), ("B", "team-b"), ("C", "team-c"), ("D", "team-d"))


def _at(days, hours=0):
    return BASE + timedelta(days=days, hours=hours)


def _event(path, clock, sport, index):
    home = TEAMS[index % 4]
    away = TEAMS[(index + 1) % 4]
    start = _at(index, 2)
    observed = start - timedelta(hours=1)
    clock[0] = observed
    side = "PUCK_LINE" if sport == "NHL" else "SPREAD"
    side_line = -1.5 if sport == "NHL" else -2.5
    total_line = 5.5 if sport == "NHL" else 202.5
    markets = [
        {"key": "spreads", "last_update": observed.isoformat(), "outcomes": [
            {"name": home[0], "point": side_line, "price": -110},
            {"name": away[0], "point": -side_line, "price": -110}]},
        {"key": "totals", "last_update": observed.isoformat(), "outcomes": [
            {"name": "Over", "point": total_line, "price": -110},
            {"name": "Under", "point": total_line, "price": -110}]},
    ]
    provider_id = f"provider-{sport}-{index}"
    raw_event = {"id": provider_id, "sport_key": pe.ODDS_API_SPORT_KEYS[sport],
                 "home_team": home[0], "away_team": away[0],
                 "commence_time": start.isoformat(),
                 "bookmakers": [{"key": "book", "markets": markets}]}
    event_id = f"event-{sport}-{index}"
    pe.insert_event(path, {"event_id": event_id, "sport": sport,
        "game_id": provider_id, "provider_namespace": "THE_ODDS_API",
        "provider_event_id": provider_id, "home_team": home[0], "away_team": away[0],
        "home_team_id": home[1], "away_team_id": away[1],
        "scheduled_start": start.isoformat(), "observed_at": observed.isoformat(),
        "source_id": f"capture-{provider_id}", "raw_source": raw_event})
    _quote(path, sport, event_id, provider_id, home[0], observed, side, side_line, markets[0])
    _quote(path, sport, event_id, provider_id, "Over", observed, "TOTAL", total_line, markets[1])
    clock[0] = start + timedelta(hours=3)
    home_score = (3 + index % 3) if sport == "NHL" else (101 + index % 17)
    away_score = (2 + index % 2) if sport == "NHL" else (96 + index % 13)
    raw_score = {"id": provider_id, "sport_key": pe.ODDS_API_SPORT_KEYS[sport],
                 "home_team": home[0], "away_team": away[0],
                 "commence_time": start.isoformat(), "completed": True,
                 "last_update": (start + timedelta(hours=2)).isoformat(),
                 "scores": [{"name": home[0], "score": str(home_score)},
                            {"name": away[0], "score": str(away_score)}]}
    pe.insert_result(path, {"result_id": f"score-{sport}-{index}", "event_id": event_id,
        "sport": sport, "market_family": None, "selection": None,
        "result_source": "THE_ODDS_API", "result_source_id": provider_id,
        "observed_at": clock[0].isoformat(), "available_at": clock[0].isoformat(),
        "home_score": home_score, "away_score": away_score, "outcome": None,
        "grading_version": 1, "raw_source": raw_score})
    return event_id


def _quote(path, sport, event_id, provider_id, selection, observed, market, line, raw):
    key = "totals" if market == "TOTAL" else "spreads"
    pe.insert_quote(path, {"quote_id": f"quote-{event_id}-{market}", "event_id": event_id,
        "sport": sport, "market_family": market, "selection": selection,
        "line": line, "american_odds": -110, "decimal_odds": 1.909090909,
        "sportsbook": "book", "quote_timestamp": observed.isoformat(),
        "quote_source": "THE_ODDS_API", "quote_verified": True,
        "source_id": f"{provider_id}:book:{key}:{selection}:{observed.isoformat()}",
        "raw_source": raw})


@pytest.fixture
def clock(monkeypatch):
    state = [BASE]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    return state


@pytest.mark.parametrize("sport", ["NBA", "NCAAB", "NHL"])
def test_six_scoped_score_models_preserve_chronology_and_zero_stake(tmp_path, clock, sport):
    path = tmp_path / "evidence.sqlite3"
    for index in range(70):
        _event(path, clock, sport, index)
    clock[0] = _at(71)
    for market in models.SCOPES[sport]:
        fit = models.train_scope(path, sport, market)
        assert fit["status"] == "FITTED_RESEARCH_ONLY"
        assert fit["independent_events"] >= models.MIN_TRAIN_EVENTS
        assert fit["production_eligible"] is False and fit["recommended_stake"] == 0
        model = pe.load_record(path, "prospective_model", fit["model_id"])
        artifact = json.loads(model["payload"])["model_artifact"]
        assert model["independent_event_count"] == fit["independent_events"]
        assert artifact["sport"] == sport and artifact["market_family"] == market
        assert artifact["market_settlement_certified"] is False
        assert artifact["validation_diagnostics"]["independent_events"] >= 18
        assert all(_at(0) <= models._time(row["available_at"]) <=
                   models._time(model["training_cutoff"]) for row in artifact["training_sources"])
        for row in artifact["training_sources"]:
            quote = pe.load_record(path, "prospective_quote", row["quote_id"])
            assert all(models._time(pe.load_record(path, "prospective_result", result_id)["available_at"])
                       <= models._time(pe.load_record(path, "prospective_event", row["event_id"])["observed_at"])
                       for result_id, _ in row["feature_lineage"])
            assert models._time(quote["quote_timestamp"]) < models._time(row["start"])
    assert not pe.read_records(path, "prospective_calibration", sport=sport)
    assert not pe.read_records(path, "prospective_result", sport=sport,
                               market_family=models.SCOPES[sport][0])


def test_future_prediction_has_persisted_features_and_no_cross_scope_model(tmp_path, clock):
    path = tmp_path / "evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    fit = models.train_scope(path, "NBA", "SPREAD")
    assert fit["status"] == "FITTED_RESEARCH_ONLY"
    registered = pe.load_record(path, "prospective_model", fit["model_id"])
    registered_payload = json.loads(registered["payload"])
    tampered = deepcopy(registered_payload["model_artifact"])
    tampered["training_sources"][0]["result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="training source replay failed"):
        pe.insert_model(path, {"model_id": fit["model_id"] + "-tampered",
            "sport": "NBA", "market_family": "SPREAD",
            "model_version": models.MODEL_VERSION,
            "training_start": registered["training_start"],
            "training_cutoff": registered["training_cutoff"],
            "training_observation_count": registered["training_observation_count"],
            "independent_event_count": registered["independent_event_count"],
            "feature_version": models.FEATURE_VERSION,
            "training_code_commit": registered["training_code_commit"],
            "created_at": clock[0].isoformat(), "available_at": clock[0].isoformat(),
            "artifact_hash": models._digest(tampered), "model_artifact": tampered,
            "training_target_kind": models.TARGET_KIND,
            "training_result_ids": registered_payload["training_result_ids"]})
    # The model exists only for NBA spread. An NBA total cannot borrow it.
    assert models.predict_scope(path, "NBA", "TOTAL")["blocker"] == "MISSING_SPORT_MARKET_MODEL"
    future_id = _event(path, clock, "NBA", 73)
    # _event recorded a final score for construction convenience; it is after
    # its start, so move the synthetic clock back to the quote's pregame time.
    clock[0] = _at(73, 1)
    result = models.predict_scope(path, "NBA", "SPREAD")
    assert result["predictions"] == 1
    predictions = pe.read_records(path, "prospective_prediction", sport="NBA", market_family="SPREAD")
    prediction = next(row for row in predictions if row["event_id"] == future_id)
    payload = json.loads(prediction["payload"])
    assert payload["feature_snapshot"]["provider_quote_hash"]
    assert payload["uncertainty"]["market_settlement_certified"] is False
    assert math.isclose(prediction["mean_probability"] + prediction["push_probability"] +
                        prediction["loss_probability"], 1, abs_tol=1e-8)
    assert prediction["conservative_probability"] <= prediction["mean_probability"]
    assert prediction["push_probability"] == 0
    assert pe.market_readiness(path, "NBA", "SPREAD")["recommended_stake"] == 0


def test_score_probability_integer_push_and_half_point_zero_push():
    whole = models.score_probabilities(5.0, 2.0, "PUCK_LINE", -1.0)
    half = models.score_probabilities(5.0, 2.0, "PUCK_LINE", -1.5)
    opposite = models.score_probabilities(5.0, 2.0, "PUCK_LINE", -1.0,
                                          selection="opposite")
    assert whole["push"] > 0 and half["push"] == 0
    assert math.isclose(sum(whole.values()), 1)
    assert math.isclose(opposite["win"], whole["loss"])
    assert math.isclose(opposite["push"], whole["push"])
    with pytest.raises(ValueError):
        models.score_probabilities(5, 2, "TOTAL", 5.25)


def test_small_samples_do_not_register_models(tmp_path, clock):
    path = tmp_path / "evidence.sqlite3"
    for index in range(8):
        _event(path, clock, "NHL", index)
    clock[0] = _at(9)
    for market in models.SCOPES["NHL"]:
        fit = models.train_scope(path, "NHL", market)
        assert fit["status"] == "INSUFFICIENT_EVIDENCE"
        assert fit["model_id"] is None
    assert not pe.read_records(path, "prospective_model", sport="NHL")


def test_post_capture_hook_reports_insufficient_evidence_without_fake_model(tmp_path, clock):
    native = tmp_path / "nba-market.sqlite3"
    canonical = tmp_path / "prospective.sqlite3"
    report = models.run_sport_model_cycle(native, canonical, "NBA")
    assert report["canonical_events"] == 0
    assert report["canonical_quotes"] == 0
    assert report["canonical_results"] == 0
    assert report["models_fitted"] == 0 and report["predictions"] == 0
    assert "SPREAD_INSUFFICIENT_EVIDENCE" in report["blockers"]
    assert "TOTAL_MISSING_SPORT_MARKET_MODEL" in report["blockers"]
    assert report["production_eligible"] is False and report["recommended_stake"] == 0


def test_native_capture_and_final_score_append_without_market_settlement(tmp_path, clock):
    native_path = tmp_path / "nba-market.sqlite3"
    canonical_path = tmp_path / "prospective.sqlite3"
    observed, start = _at(1, 1), _at(1, 2)
    market = {"key": "spreads", "last_update": observed.isoformat(),
              "outcomes": [{"name": "A", "point": -2.5, "price": -110},
                           {"name": "B", "point": 2.5, "price": -110}]}
    provider = {"id": "native-game", "sport_key": "basketball_nba",
                "home_team": "A", "away_team": "B", "commence_time": start.isoformat(),
                "bookmakers": [{"key": "book", "markets": [market]}]}
    participants = [{"full_name": "A", "id": "team-a"},
                    {"full_name": "B", "id": "team-b"}]
    quoted, rejected = odds.quotes("NBA", provider, observed)
    assert not rejected and len(quoted) == 2
    capture = {"sport": "NBA", "protocol": odds.PROTOCOL,
               "participants_source": {"source_id": "nba:participants",
                  "source_hash": odds.digest(participants),
                  "observed_at": observed.isoformat(), "raw_source": participants},
               "events": [{"event_id": "native-game", "home": "A", "away": "B",
                  "start": start.isoformat(), "provider_namespace": "THE_ODDS_API",
                  "provider_event_id": "native-game", "home_team_id": "team-a",
                  "away_team_id": "team-b", "quotes": quoted,
                  "discovery_source": {"source_id": "native-game",
                    "source_hash": odds.digest(provider),
                    "observed_at": observed.isoformat(), "raw_source": provider},
                  "response_received_at": observed.isoformat(),
                  "source_id": "native-game", "source_hash": odds.digest(provider),
                  "raw_source": provider}]}
    for_sport("NBA").save("capture", capture, native_path)
    clock[0] = observed
    report = models.ingest_native(native_path, canonical_path, "NBA")
    assert report["canonical_events"] == 1 and report["canonical_quotes"] == 2
    assert report["canonical_results"] == 0
    raw_score = {"id": "native-game", "sport_key": "basketball_nba",
                 "home_team": "A", "away_team": "B", "commence_time": start.isoformat(),
                 "completed": True, "last_update": (start + timedelta(hours=2)).isoformat(),
                 "scores": [{"name": "A", "score": "110"},
                            {"name": "B", "score": "105"}]}
    score_observed = start + timedelta(hours=3)
    for_sport("NBA").save("scores", {"sport": "NBA", "protocol": odds.PROTOCOL,
        "event": {"event_id": "native-game", "observed_at": score_observed.isoformat(),
                  "available_at": score_observed.isoformat(), "home_score": 110,
                  "away_score": 105, "source_hash": odds.digest(raw_score),
                  "raw_source": raw_score, "grading_version": 1}}, native_path)
    clock[0] = score_observed
    report = models.ingest_native(native_path, canonical_path, "NBA")
    assert report["canonical_events"] == 0 and report["canonical_quotes"] == 0
    assert report["canonical_results"] == 1
    score_rows = pe.read_records(canonical_path, "prospective_result", sport="NBA")
    assert len(score_rows) == 1 and score_rows[0]["market_family"] is None
    assert score_rows[0]["outcome"] is None
    assert not pe.read_records(canonical_path, "prospective_result", sport="NBA",
                               market_family="SPREAD")
