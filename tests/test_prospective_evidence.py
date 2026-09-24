"""Synthetic chronology and scope tests; these do not assert live validation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from app_core import prospective_evidence as pe


BASE = datetime(2030, 1, 1, tzinfo=timezone.utc)


def at(hours=0, minutes=0):
    return (BASE + timedelta(hours=hours, minutes=minutes)).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def provider_market(stamp, *, price=-110, line=-2.5, market="SPREAD"):
    if market == "TOTAL":
        return {"key": "totals", "last_update": at(stamp), "outcomes": [
            {"name": "Over", "point": line, "price": price},
            {"name": "Under", "point": line, "price": price}]}
    return {"key": "spreads", "last_update": at(stamp), "outcomes": [
        {"name": "Home", "point": line, "price": price},
        {"name": "Away", "point": -line, "price": price}]}


def provider_event(event_id, start, stamp, *, sport="NBA", market="SPREAD",
                   book="TestBook", price=-110, line=-2.5):
    return {"id": event_id, "sport_key": pe.ODDS_API_SPORT_KEYS[sport], "home_team": "Home",
            "away_team": "Away", "commence_time": at(start),
            "bookmakers": [{"key": book, "markets": [provider_market(stamp, price=price,
                                                                        line=line, market=market)]}]}


@pytest.fixture
def clock(monkeypatch):
    state = [BASE]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])

    def set_clock(hours=0, minutes=0):
        state[0] = BASE + timedelta(hours=hours, minutes=minutes)

    return set_clock


def event(path, *, sport="NBA", event_id="game", start=1, clock_hour=0,
          home_team_id="provider:1", away_team_id="provider:2", market="SPREAD",
          line=-2.5):
    return pe.insert_event(path, dict(event_id=event_id, sport=sport, game_id=event_id,
        provider_namespace="THE_ODDS_API", provider_event_id=event_id,
        home_team="Home", away_team="Away", home_team_id=home_team_id,
        away_team_id=away_team_id, scheduled_start=at(start), observed_at=at(clock_hour),
        source_id=f"event-{event_id}",
        raw_source=provider_event(event_id, start, clock_hour, sport=sport, market=market,
                                  line=line)))


def quote(path, *, sport="NBA", market="SPREAD", event_id="game", quote_id="quote",
          stamp=0, selection="Home", line=-2.5, price=-110, book="TestBook",
          verified=True, raw_line=None):
    raw = provider_market(stamp, market=market,
                          line=(5.5 if market == "TOTAL" else -2.5)
                          if raw_line is None and line is None else
                          line if raw_line is None else raw_line)
    market_key = "totals" if market == "TOTAL" else "spreads"
    return pe.insert_quote(path, dict(quote_id=quote_id, event_id=event_id, sport=sport,
        market_family=market, selection=selection, line=line, american_odds=price,
        decimal_odds=1.909090909, sportsbook=book, quote_timestamp=at(stamp),
        quote_source="THE_ODDS_API", quote_verified=verified,
        source_id=f"{event_id}:{book}:{market_key}:{selection}:{at(stamp)}", raw_source=raw))


def close(path, *, event_id, close_id, stamp, book="TestBook", line=-2.5,
          selection="Home", verified=False):
    full = provider_event(event_id, 20, stamp, book=book, price=-120, line=line)
    raw = {"event_response": full, "market": full["bookmakers"][0]["markets"][0],
           "response_received_at": at(stamp)}
    return pe.insert_close(path, dict(close_id=close_id, event_id=event_id,
        sport="NBA", market_family="SPREAD", selection=selection, line=line,
        american_odds=-120, decimal_odds=1.833333333, sportsbook=book,
        close_timestamp=at(stamp), close_source="THE_ODDS_API",
        close_verified=verified,
        source_id=f"{event_id}:{book}:spreads:{selection}:{at(stamp)}",
        raw_source=raw))


def result(path, *, sport="NBA", market="SPREAD", event_id="game", result_id="result",
           observed=2, available=2, outcome="WIN", selection="Home", version=1,
           revises=None):
    return pe.insert_result(path, dict(result_id=result_id, event_id=event_id, sport=sport,
        market_family=market, selection=selection, result_source="SYNTHETIC_TEST",
        result_source_id=result_id, observed_at=at(observed), available_at=at(available),
        home_score=104, away_score=98, outcome=outcome, grading_version=version,
        revises_result_id=revises, raw_source={"synthetic": True, "result": result_id}))


def model(path, *, sport="NBA", market="SPREAD", result_ids=("training-result",),
          model_id="model"):
    return pe.insert_model(path, dict(model_id=model_id, sport=sport, market_family=market,
        model_version="test-model-v1", training_start=at(-10), training_cutoff=at(2),
        training_observation_count=len(result_ids), independent_event_count=len(result_ids),
        feature_version="test-features-v1", training_code_commit="synthetic-commit",
        created_at=at(3), available_at=at(3), artifact_hash=digest(model_id),
        training_result_ids=list(result_ids)))


def calibration(path, *, sport="NBA", market="SPREAD", model_id="model",
                calibration_id="calibration", result_ids=("fit-result",)):
    return pe.insert_calibration(path, dict(calibration_id=calibration_id, sport=sport,
        market_family=market, calibration_version="test-cal-v1", model_id=model_id,
        fit_start=at(4), fit_end=at(6), method="synthetic-isotonic",
        created_at=at(7), available_at=at(7), artifact_hash=digest(calibration_id),
        fit_result_ids=list(result_ids)))


def prediction(path, *, event_id, quote_id, observation_id, stamp, model_id="model",
               calibration_id="calibration", sport="NBA", market="SPREAD",
               mean=.65, push=0, loss=.35, semantics="win_push_loss"):
    return pe.insert_prediction(path, dict(observation_id=observation_id,
        event_id=event_id, sport=sport, market_family=market, selection="Home",
        quote_id=quote_id, model_id=model_id, calibration_id=calibration_id,
        feature_version="test-features-v1", feature_snapshot_id=observation_id,
        feature_frozen_at=at(stamp), policy_version="synthetic-policy-v1",
        prediction_timestamp=at(stamp), mean_probability=mean,
        conservative_probability=min(.58, mean), push_probability=push,
        loss_probability=loss, probability_semantics=semantics,
        evidence_snapshot_id=observation_id,
        evidence_hash=digest(observation_id), runtime_hash=digest("runtime"),
        source_commit="synthetic-commit"))


def plan(path, *, model_id="model", calibration_id="calibration", version=1,
         plan_id="plan", supersedes=None, max_brier=1):
    return pe.freeze_validation_plan(path, dict(validation_plan_id=plan_id,
        sport="NBA", market_family="SPREAD", version=version,
        supersedes_plan_id=supersedes, model_id=model_id,
        calibration_id=calibration_id, training_cutoff=at(8),
        validation_start=at(9), validation_end=at(15),
        holdout_start=at(16), holdout_end=at(25),
        minimum_independent_sample=1, minimum_effective_sample=1,
        probability_thresholds=dict(max_brier=max_brier, max_log_loss=2,
                                    max_calibration_error=1, min_coverage=1),
        calibration_requirements=dict(required=True),
        price_evidence_requirements=dict(min_verified_entry_coverage=1),
        clv_policy=dict(required=False, min_comparable_close_coverage=0),
        value_roi_policy=dict(min_paper_roi=0),
        deployment_criteria=dict(target_state="PROVISIONAL_VALIDATED",
                                 maturity_rules={"synthetic_only": True}),
        independence_method="ONE_EARLIEST_PREDICTION_PER_EVENT_V1"))


def test_all_twelve_markets_start_unvalidated_and_zero_stake(tmp_path):
    path = tmp_path / "prospective.sqlite3"
    rows = pe.all_market_readiness(path)
    assert not path.exists()
    assert len(rows) == 12
    assert {(r["sport"], r["market_family"]) for r in rows} == {
        (sport, market) for sport, markets in pe.SPORT_MARKETS.items() for market in markets}
    assert all(r["deployment_state"] == "UNVALIDATED" and
               r["production_eligible"] is False and r["recommended_stake"] == 0 and
               r["next_blocker"] == "NO_PROSPECTIVE_EVENTS" for r in rows)


@pytest.mark.parametrize(("sport", "market"), [
    ("NFL", "SPREAD"), ("NFL", "TOTAL"),
    ("NCAAF", "SPREAD"), ("NCAAF", "TOTAL"),
    ("MLB", "RUN_LINE"), ("MLB", "TOTAL"),
])
def test_new_sport_market_offer_replays_without_creating_wager_authority(
        tmp_path, clock, sport, market):
    path = tmp_path / "prospective.sqlite3"
    line = 5.5 if market == "TOTAL" else -1.5 if market == "RUN_LINE" else -2.5
    selection = "Over" if market == "TOTAL" else "Home"
    event(path, sport=sport, market=market, line=line)
    quote(path, sport=sport, market=market, line=line, selection=selection)
    saved_event = pe.load_record(path, "prospective_event", "game")
    saved_quote = pe.load_record(path, "prospective_quote", "quote")
    assert saved_quote["quote_verified"] == 1
    assert pe.verify_provider_offer(saved_event, saved_quote)
    state = pe.deployment_state(path, sport, market)
    assert state["deployment_state"] == "UNVALIDATED"
    assert state["production_eligible"] is False
    assert state["recommended_stake"] == 0
    if sport == "MLB":
        # This proves the bookmaker offer, not a Stats API game crosswalk.
        assert saved_event["provider_namespace"] == "THE_ODDS_API"
        assert saved_event["provider_event_id"] == "game"


def test_research_evidence_roundtrips_with_missing_team_ids_and_append_only(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    event(path, home_team_id=None)
    quote(path, verified=False)
    assert pe.read_records(path, "prospective_event")[0]["home_team_id"] is None
    assert pe.read_records(path, "prospective_quote")[0]["source_hash"] == digest(
        json.dumps(provider_market(0), sort_keys=True, separators=(",", ":")))
    with pe.connect(path) as db:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("UPDATE prospective_event SET home_team_id='fake' WHERE event_id='game'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("DELETE FROM prospective_quote")
    with pytest.raises(pe.EvidenceConflict):
        event(path, home_team_id="changed")


def test_price_and_prediction_chronology_fail_closed(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    event(path)
    with pytest.raises(ValueError, match="future or stale"):
        quote(path, stamp=1)
    with pytest.raises(ValueError, match="requires exact"):
        quote(path, line=None)
    with pytest.raises(ValueError, match="requires exact"):
        quote(path, book=None)
    with pytest.raises(ValueError, match="requires exact"):
        quote(path, price=None)
    clock(hours=0, minutes=20)
    with pytest.raises(ValueError, match="future or stale"):
        quote(path)
    clock(hours=1)
    with pytest.raises(ValueError, match="future or stale|pregame"):
        quote(path, stamp=1)
    with pytest.raises(ValueError, match="pregame|backdated"):
        prediction(path, event_id="game", quote_id=None, observation_id="late", stamp=0,
                   model_id=None, calibration_id=None)


def test_result_corrections_append_and_training_rejects_unavailable_outcomes(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    event(path)
    clock(hours=2)
    result(path)
    with pytest.raises(ValueError, match="cutoff"):
        pe.insert_model(path, dict(model_id="bad", sport="NBA", market_family="SPREAD",
            model_version="v1", training_start=at(-10), training_cutoff=at(1),
            training_observation_count=1, independent_event_count=1,
            feature_version="v1", training_code_commit="test", created_at=at(2),
            available_at=at(2), artifact_hash=digest("bad"),
            training_result_ids=["result"]))
    clock(hours=3)
    result(path, result_id="correction", observed=2, available=3, outcome="LOSS",
           version=2, revises="result")
    assert len(pe.read_records(path, "prospective_result")) == 2
    with pytest.raises(ValueError, match="advance version"):
        result(path, result_id="bad-correction", observed=2, available=3,
               version=2, revises="correction")


def _trained_market(path, clock):
    event(path, event_id="training", start=1)
    clock(hours=2)
    result(path, event_id="training", result_id="training-result")
    clock(hours=3)
    model(path)
    event(path, event_id="fit", start=5, clock_hour=3)
    clock(hours=6)
    result(path, event_id="fit", result_id="fit-result", observed=6, available=6)
    clock(hours=7)
    calibration(path)


def test_model_and_calibration_are_exact_market_and_versioned(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    with pytest.raises(ValueError, match="scope mismatch"):
        calibration(path, market="TOTAL", calibration_id="bad")
    with pytest.raises(ValueError, match="scope mismatch"):
        pe.insert_model(path, dict(model_id="other", sport="NCAAB", market_family="SPREAD",
            model_version="v1", training_start=at(-10), training_cutoff=at(2),
            training_observation_count=1, independent_event_count=1,
            feature_version="v1", training_code_commit="test", created_at=at(3),
            available_at=at(3), artifact_hash=digest("other"),
            training_result_ids=["training-result"]))
    assert pe.deployment_state(path, "NBA", "TOTAL")["validation_id"] is None
    assert pe.deployment_state(path, "NCAAB", "SPREAD")["validation_id"] is None


def test_frozen_plan_metrics_and_review_do_not_activate_stakes(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    plan(path)
    with pytest.raises(ValueError, match="frozen_at"):
        pe.freeze_validation_plan(path, dict(frozen_at=at(1)))
    assert pe.evaluate_validation_plan(path, "plan")["status"] == "UNVALIDATED"
    clock(hours=9)
    event(path, event_id="validation", start=11, clock_hour=9)
    quote(path, event_id="validation", quote_id="validation-quote", stamp=9)
    prediction(path, event_id="validation", quote_id="validation-quote",
               observation_id="validation-pred", stamp=9)
    clock(hours=12)
    result(path, event_id="validation", result_id="validation-result", observed=12, available=12)
    clock(hours=18)
    event(path, event_id="holdout", start=20, clock_hour=18)
    quote(path, event_id="holdout", quote_id="holdout-quote", stamp=18)
    prediction(path, event_id="holdout", quote_id="holdout-quote",
               observation_id="holdout-pred", stamp=18)
    clock(hours=19.5)
    with pytest.raises(ValueError, match="not replayable"):
        close(path, event_id="holdout", close_id="uncertified-close", stamp=19.5,
              verified=True)
    close(path, event_id="holdout", close_id="other-book-close", stamp=19.5, book="OtherBook")
    close(path, event_id="holdout", close_id="matching-close", stamp=19.5)
    clock(hours=21)
    result(path, event_id="holdout", result_id="holdout-result", observed=21, available=21)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    assert report["status"] == "VALIDATION_PASSED"
    assert report["holdout"]["raw_predictions"] == 1
    assert report["holdout"]["unique_events"] == 1
    assert report["holdout"]["outcomes"]["WIN"] == 1
    assert report["holdout"]["brier"] == pytest.approx(.1225)
    assert report["holdout"]["comparable_close_count"] == 0
    assert report["holdout"]["average_valid_clv"] is None
    assert report["holdout"]["paper_roi"] == pytest.approx(.909090909)
    assert report["holdout"]["accepted_wager_roi"] is None
    pe.create_validation_artifact(path, "plan", "artifact")
    pe.record_deployment_review(path, dict(deployment_id="review", artifact_id="artifact",
        validation_id="test-only-validation", sport="NBA", market_family="SPREAD",
        deployment_state="PROVISIONAL_VALIDATED", reviewer_id="synthetic-reviewer"))
    state = pe.deployment_state(path, "NBA", "SPREAD")
    assert state["validation_id"] == "test-only-validation"
    assert state["model_id"] == "model" and state["calibration_id"] == "calibration"
    assert state["owner_authorized"] is False and state["production_eligible"] is False
    assert state["recommended_stake"] == 0
    assert pe.deployment_state(path, "NBA", "TOTAL")["deployment_state"] == "UNVALIDATED"
    clock(hours=27)
    result(path, event_id="holdout", result_id="holdout-correction", observed=21,
           available=27, outcome="LOSS", version=2, revises="holdout-result")
    assert pe.deployment_state(path, "NBA", "SPREAD")["deployment_state"] == "UNVALIDATED"


def test_plan_versions_cannot_be_overwritten_or_backdated(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    clock(hours=8)
    plan(path, model_id=None, calibration_id=None)
    with pytest.raises(pe.EvidenceConflict):
        plan(path, model_id=None, calibration_id=None, max_brier=.8)
    with pytest.raises(ValueError, match="supersede"):
        plan(path, model_id=None, calibration_id=None, version=2, plan_id="plan-2")
    plan(path, model_id=None, calibration_id=None, version=2, plan_id="plan-2",
         supersedes="plan")
    clock(hours=17)
    with pytest.raises(ValueError, match="before holdout"):
        plan(path, model_id=None, calibration_id=None, version=3, plan_id="late",
             supersedes="plan-2")


def test_missing_validation_sample_cannot_be_rescued_by_holdout(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    plan(path)
    clock(hours=18)
    event(path, event_id="holdout-only", start=20, clock_hour=18)
    quote(path, event_id="holdout-only", quote_id="holdout-only-quote", stamp=18)
    prediction(path, event_id="holdout-only", quote_id="holdout-only-quote",
               observation_id="holdout-only-pred", stamp=18)
    clock(hours=21)
    result(path, event_id="holdout-only", result_id="holdout-only-result",
           observed=21, available=21)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    assert report["holdout"]["effective_observations"] == 1
    assert report["validation"]["effective_observations"] == 0
    assert report["status"] == "UNVALIDATED"
    assert "NEED_1_MORE_EFFECTIVE_VALIDATION_EVENTS" in report["blockers"]
    assert "NEED_1_MORE_SETTLED_VALIDATION_EVENTS" in report["blockers"]


def test_push_semantics_score_decided_win_probability(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    plan(path)
    clock(hours=9)
    event(path, event_id="validation", start=11, clock_hour=9)
    quote(path, event_id="validation", quote_id="validation-quote", stamp=9)
    prediction(path, event_id="validation", quote_id="validation-quote",
               observation_id="validation-pred", stamp=9,
               mean=.55, push=.1, loss=.35)
    clock(hours=12)
    result(path, event_id="validation", result_id="validation-result", observed=12, available=12)
    clock(hours=18)
    event(path, event_id="holdout", start=20, clock_hour=18)
    quote(path, event_id="holdout", quote_id="holdout-quote", stamp=18)
    prediction(path, event_id="holdout", quote_id="holdout-quote",
               observation_id="holdout-pred", stamp=18,
               mean=.55, push=.1, loss=.35)
    clock(hours=21)
    result(path, event_id="holdout", result_id="holdout-result", observed=21, available=21)
    clock(hours=26)
    report = pe.evaluate_validation_plan(path, "plan")
    assert report["status"] == "VALIDATION_PASSED"
    conditional = .55 / (.55 + .35)
    assert report["holdout"]["brier"] == pytest.approx((1 - conditional) ** 2)
    assert report["holdout"]["mean_predicted_probability"] == .55
    assert report["holdout"]["mean_predicted_decided_win_probability"] == pytest.approx(conditional)


def test_source_and_nonfinite_and_orphans_rejected(tmp_path, clock):
    path = tmp_path / "prospective.sqlite3"
    event(path, home_team_id=None)
    with pytest.raises(pe.EvidenceConflict, match="source_hash"):
        pe.insert_quote(path, dict(quote_id="bad-source", event_id="game", sport="NBA",
            market_family="SPREAD", selection="Home -2.5", line=-2.5,
            american_odds=-110, decimal_odds=1.909090909, sportsbook="TestBook",
            quote_timestamp=at(0), quote_source="SYNTHETIC_TEST", quote_verified=True,
            source_id="bad-source", source_hash="0" * 64,
            raw_source={"actual": "different"}))
    with pytest.raises(ValueError, match="not replayable"):
        pe.insert_quote(path, dict(quote_id="fabricated-offer", event_id="game", sport="NBA",
            market_family="SPREAD", selection="Home", line=-2.5,
            american_odds=-110, decimal_odds=1.909090909, sportsbook="TestBook",
            quote_timestamp=at(0), quote_source="THE_ODDS_API", quote_verified=True,
            source_id=f"game:TestBook:spreads:Home:{at(0)}",
            raw_source={"arbitrary": "not a provider market"}))
    with pytest.raises(ValueError, match="not replayable"):
        quote(path, quote_id="missing-team-id")
    event(path, event_id="valid")
    with pytest.raises(ValueError, match="not replayable"):
        quote(path, event_id="valid", quote_id="changed-line", line=-1.5,
              raw_line=-2.5)
    with pytest.raises(ValueError, match="finite"):
        quote(path, line=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        quote(path, line=float("inf"))
    assert pe.market_readiness(path, "NBA", "SPREAD")["next_blocker"] == "NO_MARKET_QUOTES_OR_PREDICTIONS"
    quote(path, verified=False)
    assert pe.market_readiness(path, "NBA", "SPREAD")["next_blocker"] == "MISSING_STABLE_TEAM_IDS"
    loaded = pe.load_record(path, "prospective_quote", "quote")
    assert loaded["source_hash"] == digest(loaded["raw_source"].decode())
    assert not pe.verify_provider_offer(pe.load_record(path, "prospective_event", "game"), loaded)
    assert pe.load_record(path, "prospective_quote", "missing") is None
    with pe.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO prospective_model_training_result VALUES ('orphan','result')")
