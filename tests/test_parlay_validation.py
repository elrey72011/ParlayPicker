"""Synthetic prospective fixtures exercise mechanics, never live validation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3

import pytest

from app_core import parlay_persistence as store
from app_core import parlay_validation as validation
from app_core.parlay_ticket_quotes import bind_ticket_request
from core.true_parlay_engine import leg_hash, ticket_hash


BASE = datetime(2030, 1, 1, tzinfo=timezone.utc)


def at(hours=0, minutes=0):
    return (BASE + timedelta(hours=hours, minutes=minutes)).isoformat()


def plan(product="STANDARD_PARLAY", plan_id=None, *, version=1, supersedes=None):
    method = {"STANDARD_PARLAY": "INDEPENDENCE_PRODUCT",
              "SAME_GAME_PARLAY": "COPULA",
              "CROSS_GAME_PARLAY": "JOINT_COMPONENT_MODEL"}[product]
    scope = {"joint_model_id": "synthetic-joint", "joint_model_version": "v1",
             "calibration_version": "synthetic-cal-v1", "method": method}
    if product == "SAME_GAME_PARLAY":
        scope["correlation_method"] = "synthetic-copula"
    if product == "CROSS_GAME_PARLAY":
        scope.update(preserve_sgp_blocks=True,
                     component_dependence_method="synthetic-block-dependence",
                     shared_factor_method="synthetic-shared-factor",
                     final_calibration_id="synthetic-final-calibration",
                     final_calibration_version="v1")
    return {
        "plan_id": plan_id or f"synthetic-{product}-{version}", "product_type": product,
        "version": version, "supersedes_plan_id": supersedes,
        "product_policy_version": "synthetic-policy-v1", "model_scope": scope,
        "sport_market_scope": [{"sport": "MLB", "market": "spread_away"}],
        "training_cutoff": at(minutes=-10),
        "training_manifest": {"source_manifest_hash": "a"*64,
                              "latest_outcome_available_at": at(minutes=-20),
                              "latest_feature_available_at": at(minutes=-20)},
        "windows": {"validation": {"start": at(hours=1), "end": at(hours=4)},
                    "holdout": {"start": at(hours=5), "end": at(hours=9)}},
        "minimum_independent_units": {"validation": 2, "holdout": 2},
        "probability_thresholds": {"max_brier": .25, "max_log_loss": .8,
                                   "max_calibration_error": .3, "min_coverage": .8,
                                   "coverage_denominator": "ALL_CANDIDATES"},
        "betting_thresholds": {"min_accepted_wagers": 1, "min_roi": 0.0},
        "price_evidence_requirements": {"require_verified_exact_quote": True},
        "clv_policy": {"required": False, "comparable_closing_only": True},
        "roi_reporting_policy": {"accepted_wagers_only": True,
                                 "include_void_in_denominator": True},
        "deployment_state_criteria": {"target_state": "PROVISIONAL_VALIDATED"},
        "independence_method": validation.INDEPENDENCE_METHOD,
    }


def leg(number, *, event=None, start=2):
    row = {"candidate_id": f"synthetic-c{number}", "game_id": event or f"g{number}",
           "sport": "MLB", "provider_namespace": "synthetic-feed",
           "provider_event_id": event or f"synthetic-event-{number}",
           "provider_market_id": f"market-{number}",
           "provider_selection_id": f"selection-{number}",
           "market_type": "spread_away", "selection": f"Away {number}",
           "line": 1.5, "sportsbook": "SyntheticBook", "game_start_at": at(hours=start),
           "model_trained_through": at(minutes=-60),
           "model_available_at": at(minutes=-5),
           "calibration_available_at": at(minutes=-5),
           "evidence_frozen_at": at(minutes=-5)}
    row["leg_hash"] = leg_hash(row)
    return row


PROVIDER_FIELDS = (
    "provider", "product_type", "sportsbook", "quote_id", "provider_ticket_id",
    "provider_response_id", "selection_bindings", "component_hashes", "sgp_components",
    "american_odds", "decimal_odds", "quoted_at", "expires_at", "settlement_rules_id",
)


def raw_response(quote):
    return {**{name: quote[name] for name in PROVIDER_FIELDS},
            "price_origin": "EXECUTABLE_TICKET"}


def attach(path, decision_id):
    quote = store.load_decision(path, decision_id)["quote"]
    return validation.attach_quote_source(path, decision_id,
                                          raw_response=raw_response(quote))


def result_source(path, decision_id, result, available_at):
    with validation.connect(path) as db:
        stored = db.execute("SELECT payload,payload_hash FROM parlay_validation_candidate "
                            "WHERE decision_id=?",(decision_id,)).fetchone()
    candidate = validation._verified(*stored)
    raw = {"source_type":"SPORTSBOOK_SETTLEMENT" if result["result_kind"] == "ACCEPTED_WAGER"
           else "OFFICIAL_RESULT", "source_id":result["source_id"],
           "provider_response_id":f"synthetic-result-response-{decision_id[:8]}",
           "ticket_hash":candidate["ticket_hash"],"event_keys":candidate["event_keys"],
           "outcome":result["outcome"],"leg_outcomes":result["leg_outcomes"],
           "settlement_rule":result["settlement_rule"],"available_at":available_at}
    if result["result_kind"] == "ACCEPTED_WAGER":
        raw.update(accepted_wager_id=result["accepted_wager_id"],
                   net_return=result["net_return"],accepted_stake=result["accepted_stake"],
                   accepted_decimal_odds=result["accepted_decimal_odds"])
    result["result_evidence_hash"] = validation._hash(raw)
    return raw


def decision(path, *, product="STANDARD_PARLAY", number=1, starts=2,
             games=None, quote=True, decision_time=15):
    rows = [leg(number*10+i, event=(games[i] if games else None), start=starts)
            for i in range(2)]
    if product == "SAME_GAME_PARLAY":
        rows[1]["game_id"] = rows[0]["game_id"]
        rows[1]["provider_event_id"] = rows[0]["provider_event_id"]
        rows[1]["leg_hash"] = leg_hash(rows[1])
    identity = ticket_hash(product, rows, "SyntheticBook")
    data = {
        "parlay_id": f"synthetic-{product}-{number}", "product_type": product,
        "status": "UNVALIDATED", "sportsbook": "SyntheticBook",
        "ticket_hash": identity, "decision_at": at(minutes=decision_time),
        "recommended_stake_dollars": 0.0, "production_eligible": False,
        "blockers": ["PRODUCT_UNVALIDATED"],
        "policy_version": "synthetic-policy-v1",
        "joint_model_id": "synthetic-joint", "joint_model_version": "v1",
        "calibration_version": "synthetic-cal-v1",
        "probability_method": {"STANDARD_PARLAY": "INDEPENDENCE_PRODUCT",
                               "SAME_GAME_PARLAY": "COPULA",
                               "CROSS_GAME_PARLAY": "JOINT_COMPONENT_MODEL"}[product],
        "joint_model_trained_through": at(minutes=-60),
        "joint_model_available_at": at(minutes=-5),
        "joint_calibration_available_at": at(minutes=-5),
        "joint_generated_at": at(minutes=10),
        "joint_evidence_frozen_at": at(minutes=9),
        "probability_mean": .6, "probability_conservative": .55,
        "probability_push": 0.0, "probability_partial": 0.0,
        "break_even_probability": 1/3.5,
        "conservative_ev": .55*3.5-1, "conservative_edge": .55-1/3.5,
        "dependence_status": "INDEPENDENT_VERIFIED",
        "validation_state": "UNVALIDATED",
    }
    if product == "SAME_GAME_PARLAY":
        data["joint_correlation_method"] = "synthetic-copula"
    if product == "CROSS_GAME_PARLAY":
        data.update(joint_component_dependence_method="synthetic-block-dependence",
                    joint_shared_factor_method="synthetic-shared-factor",
                    joint_final_calibration_id="synthetic-final-calibration",
                    joint_final_calibration_version="v1")
    quote_data = None
    if quote:
        binding = bind_ticket_request({"product_type": product, "sportsbook": "SyntheticBook",
                                       "components": rows})
        quote_data = {
            **binding, "quote_id": f"synthetic-q-{product}-{number}-{decision_time}",
            "provider": "synthetic-adapter", "provider_ticket_id": "synthetic-provider-ticket",
            "provider_response_id": "synthetic-provider-response",
            "american_odds": 250, "decimal_odds": 3.5,
            "quoted_at": at(minutes=decision_time-1),
            "expires_at": at(minutes=decision_time+10),
            "verification_state": "VERIFIED", "source": "SPORTSBOOK",
            "source_type": "PROVIDER_API", "settlement_rules_id": "synthetic-rules",
        }
        quote_data["raw_evidence_hash"] = validation._hash(raw_response(quote_data))
    return store.save_decision(path, data, rows, quote=quote_data)


def freeze(monkeypatch, when):
    monkeypatch.setattr(validation, "_clock", lambda: when)


def test_product_plan_is_frozen_versioned_and_uses_shared_database(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    standard = validation.freeze_plan(path, plan())
    assert standard["artifact_hash"] == validation._hash({k: v for k, v in standard.items()
                                                          if k != "artifact_hash"})
    assert validation.load_plan(path, standard["plan_id"]) == standard
    with pytest.raises(store.EvidenceConflict, match="new version"):
        validation.freeze_plan(path, plan())
    with pytest.raises(ValueError, match="new product plan"):
        validation.freeze_plan(path, plan(version=2))
    newer = plan(version=2, supersedes=standard["plan_id"])
    newer["windows"] = {"validation": {"start": at(hours=10), "end": at(hours=12)},
                        "holdout": {"start": at(hours=13), "end": at(hours=15)}}
    assert validation.freeze_plan(path, newer)["version"] == 2
    with validation.connect(path) as db:
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM parlay_validation_plan").fetchone() == (2,)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE parlay_validation_plan SET version=3")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO parlay_validation_candidate "
                       "(decision_id,plan_id,parlay_id,product_type,cohort,frozen_at,first_event_start,"
                       "ticket_hash,probability_admissible,event_keys,payload,payload_hash) "
                       "VALUES ('missing','missing','missing','STANDARD_PARLAY','validation',?,?,'x',0,'[]','{}','x')",
                       (at(), at(hours=2)))


def test_product_plans_reject_straight_scope_and_independent_sgp(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    with pytest.raises(ValueError, match="invalid parlay product"):
        validation.freeze_plan(path, {**plan(), "product_type": "STRAIGHT"})
    sgp = plan("SAME_GAME_PARLAY")
    sgp["model_scope"]["method"] = "INDEPENDENCE_PRODUCT"
    with pytest.raises(ValueError, match="independent multiplication"):
        validation.freeze_plan(path, sgp)
    cross = plan("CROSS_GAME_PARLAY")
    del cross["model_scope"]["shared_factor_method"]
    with pytest.raises(ValueError, match="shared_factor_method"):
        validation.freeze_plan(path, cross)
    late = plan()
    late["training_manifest"]["latest_outcome_available_at"] = at()
    with pytest.raises(ValueError, match="training inputs"):
        validation.freeze_plan(path, late)
    for product in validation.PRODUCTS:
        assert validation.product_readiness(path, product)["recommended_stake"] == 0


def test_candidate_chronology_quotes_and_independent_units(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    plan_id = validation.freeze_plan(path, plan())["plan_id"]
    ids = [decision(path, number=1, games=["shared", "g2"]),
           decision(path, number=2, games=["shared", "g3"]),
           decision(path, number=3, games=["g4", "g5"], quote=False)]
    freeze(monkeypatch, BASE + timedelta(minutes=20))
    for item in ids[:2]:
        attach(path, item)
    candidates = [validation.freeze_candidate(path, plan_id, item) for item in ids]
    assert all(item["cohort"] == "validation" for item in candidates)
    assert all(item["probability_admissible"] for item in candidates)
    assert "PRICE_UNAVAILABLE" in candidates[-1]["blockers"]
    assert validation.freeze_candidate(path, plan_id, ids[0]) == candidates[0]
    report = validation.evaluate_plan(path, plan_id)
    assert report["cohorts"]["validation"]["raw_candidates"] == 3
    assert report["cohorts"]["validation"]["independent_units"] == 2
    assert report["cohorts"]["validation"]["outcomes"]["PENDING"] == 3
    assert report["cohorts"]["validation"]["value"]["price_available"] == 2
    assert report["status"] == "UNVALIDATED"
    assert report["recommended_stake"] == 0
    with pytest.raises(ValueError, match="product-scoped"):
        validation.freeze_candidate(path, plan_id, decision(
            path, product="SAME_GAME_PARLAY", number=9))


def test_repricing_and_shared_events_do_not_inflate_independent_units(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    plan_id = validation.freeze_plan(path, plan())["plan_id"]
    first = decision(path, number=1, games=["shared", "g2"])
    snapshot = store.load_decision(path, first)
    revised = dict(snapshot["decision"], decision_at=at(minutes=16))
    quote = dict(snapshot["quote"], quote_id="synthetic-reprice", quoted_at=at(minutes=15),
                 decimal_odds=3.4, american_odds=240)
    second = store.save_decision(path, revised, snapshot["legs"], quote=quote)
    third = decision(path, number=2, games=["shared", "g3"])
    freeze(monkeypatch, BASE + timedelta(minutes=20))
    for identifier in (first, second, third):
        if identifier != second:
            attach(path, identifier)
        validation.freeze_candidate(path, plan_id, identifier)
    report = validation.evaluate_plan(path, plan_id)
    cohort = report["cohorts"]["validation"]
    assert cohort["raw_candidates"] == 3
    assert cohort["unique_tickets"] == 2
    assert cohort["independent_units"] == 1
    assert cohort["value"]["price_available"] == 2


def test_sgp_and_cross_game_scope_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    sgp_id = validation.freeze_plan(path, plan("SAME_GAME_PARLAY"))["plan_id"]
    cross_id = validation.freeze_plan(path, plan("CROSS_GAME_PARLAY"))["plan_id"]
    sgp = decision(path, product="SAME_GAME_PARLAY", number=1)
    cross = decision(path, product="CROSS_GAME_PARLAY", number=2)
    cross_snapshot = store.load_decision(path, cross)
    changed = dict(cross_snapshot["decision"], parlay_id="synthetic-cross-missing-factor",
                   joint_shared_factor_method=None, decision_at=at(minutes=16))
    changed_id = store.save_decision(path, changed, cross_snapshot["legs"],
                                     quote={**cross_snapshot["quote"],
                                            "quote_id": "synthetic-cross-missing-factor-quote"})
    freeze(monkeypatch, BASE + timedelta(minutes=20))
    attach(path, sgp)
    attach(path, cross)
    assert validation.freeze_candidate(path, sgp_id, sgp)["probability_admissible"] is True
    assert validation.freeze_candidate(path, cross_id, cross)["probability_admissible"] is True
    blocked = validation.freeze_candidate(path, cross_id, changed_id)
    assert "CROSS_DEPENDENCE_OR_CALIBRATION_UNVERIFIED" in blocked["blockers"]
    assert blocked["probability_admissible"] is False


def test_shared_event_errors_cannot_cancel_in_probability_metrics():
    base = {"ticket_hash": "same", "event_keys": ["MLB|feed|event"],
            "probability_admissible": True}
    rows = [{**base, "outcome": "WIN", "probability_mean": .1},
            {**base, "outcome": "LOSS", "probability_mean": .9}]
    metrics = validation._binary_metrics(rows)
    assert metrics["scored_independent_units"] == 1
    assert metrics["brier"] == pytest.approx(.81)
    assert metrics["calibration_error"] == pytest.approx(.9)


def test_synthetic_validation_pass_never_activates_stake(tmp_path, monkeypatch):
    """Fixture data stays in a temporary DB; it never supplies live validation."""
    path = tmp_path / "synthetic-only.sqlite3"
    freeze(monkeypatch, BASE)
    frozen = plan()
    frozen["probability_thresholds"]["max_brier"] = .3
    plan_id = validation.freeze_plan(path, frozen)["plan_id"]
    ids = [decision(path, number=number, starts=start)
           for number,start in ((1,2),(2,2),(3,6),(4,6))]
    freeze(monkeypatch, BASE + timedelta(minutes=20))
    for item in ids:
        assert attach(path, item)
        candidate = validation.freeze_candidate(path, plan_id, item)
        assert candidate["probability_admissible"]
        assert "PRICE_UNAVAILABLE" not in candidate["blockers"]
    early = validation.evaluate_plan(path, plan_id)
    assert "HOLDOUT_WINDOW_OPEN" in early["blockers"]
    assert early["status"] == "UNVALIDATED"
    freeze(monkeypatch, BASE + timedelta(hours=10))
    for index,item in enumerate(ids):
        win = index in (0,2)
        kind = "ACCEPTED_WAGER" if index == 2 else "SELECTION"
        result = {"grading_version":1,"result_kind":kind,
                  "outcome":"WIN" if win else "LOSS", "settlement":"SETTLED",
                  "settlement_rule":"synthetic-rules", "source_id":f"synthetic-final-{index}",
                  "graded_at":at(hours=4 if index < 2 else 8),
                  "leg_outcomes":["WIN","WIN"] if win else ["LOSS","WIN"]}
        if kind == "ACCEPTED_WAGER":
            acceptance = {"acceptance_state":"ACCEPTED",
                          "provider_response_id":"synthetic-acceptance-response",
                          "provider":"synthetic-adapter", "sportsbook":"SyntheticBook",
                          "ticket_hash":store.load_decision(path,item)["decision"]["ticket_hash"],
                          "quote_id":store.load_decision(path,item)["quote"]["quote_id"],
                          "accepted_wager_id":"synthetic-receipt-1", "decimal_odds":3.5,
                          "stake":1.0,"placed_at":at(minutes=16)}
            result.update(accepted_wager_id="synthetic-receipt-1",accepted_decimal_odds=3.5,
                          accepted_stake=1.0,placed_at=at(minutes=16),net_return=2.5,
                          accepted_wager_evidence_hash=validation._hash(acceptance))
        else:
            result["selection_return"] = 2.5 if win else -1.0
        available_at = at(hours=3 if index < 2 else 7)
        raw_final = result_source(path,item,result,available_at)
        assert store.append_result(path,item,result)
        if kind == "ACCEPTED_WAGER":
            assert validation.attach_accepted_wager_source(path,item,1,acceptance)
        assert validation.attach_result_source(path,item,1,raw_final)
        assert validation.append_outcome(path,item,1,
                                         source_id=f"synthetic-final-{index}",
                                         available_at=available_at,
                                         observed_at=at(hours=3 if index < 2 else 7,minutes=5))
    report = validation.evaluate_plan(path,plan_id)
    assert report["status"] == "VALIDATION_PASSED"
    assert report["activation_status"] == "ACTIVATION_PENDING"
    assert report["recommended_stake"] == 0
    assert report["cohorts"]["holdout"]["value"]["accepted_wagers"] == 1
    artifact = validation.freeze_report(path,plan_id)
    reviewed = validation.record_deployment_review(
        path,artifact_id=artifact["artifact_id"],validation_id="synthetic-review-only",
        reviewer_id="synthetic-reviewer",validation_state="PROVISIONAL_VALIDATED")
    assert reviewed["activation_status"] == "ACTIVATION_PENDING"
    status = validation.product_readiness(path,"STANDARD_PARLAY")
    assert status["validation_state"] == "PROVISIONAL_VALIDATED"
    assert status["validation_id"] == "synthetic-review-only"
    assert status["production_eligible"] is False
    assert status["recommended_stake"] == 0.0
    assert validation.product_readiness(path,"SAME_GAME_PARLAY")["validation_id"] is None


def test_late_model_availability_and_inferred_price_do_not_count(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    plan_id = validation.freeze_plan(path, plan())["plan_id"]
    identifier = decision(path, number=7)
    with store.connect(path) as db:
        snapshot = store.load_decision(path, identifier)
    altered = deepcopy(snapshot["decision"])
    altered["parlay_id"] = "synthetic-late-model"
    altered["joint_model_available_at"] = at(hours=1)
    altered["decision_at"] = at(minutes=20)
    # A new identity is required for a changed immutable candidate.
    altered_id = store.save_decision(path, altered, snapshot["legs"],
                                     quote={**snapshot["quote"], "quote_id": "synthetic-late-q"})
    freeze(monkeypatch, BASE + timedelta(minutes=30))
    result = validation.freeze_candidate(path, plan_id, altered_id)
    assert "JOINT_MODEL_CHRONOLOGY_INVALID" in result["blockers"]
    assert not result["probability_admissible"]
    # A research candidate may be frozen, but its stored leg-odds estimate is not price evidence.
    unquoted = decision(path, number=8, quote=False)
    assert "PRICE_UNAVAILABLE" in validation.freeze_candidate(path, plan_id, unquoted)["blockers"]
    freeze(monkeypatch, BASE + timedelta(hours=3))
    with pytest.raises(ValueError, match="predate every event"):
        validation.freeze_candidate(path, plan_id, decision(path, number=10))


def test_outcome_availability_void_reporting_and_artifact_boundary(tmp_path, monkeypatch):
    path = tmp_path / "parlay.sqlite3"
    freeze(monkeypatch, BASE)
    plan_id = validation.freeze_plan(path, plan())["plan_id"]
    val_id = decision(path, number=1)
    hold_id = decision(path, number=2, starts=6)
    freeze(monkeypatch, BASE + timedelta(minutes=20))
    attach(path, val_id)
    attach(path, hold_id)
    validation.freeze_candidate(path, plan_id, val_id)
    validation.freeze_candidate(path, plan_id, hold_id)
    freeze(monkeypatch, BASE + timedelta(hours=3))
    result = {
        "grading_version": 1, "result_kind": "SELECTION", "outcome": "VOID",
        "settlement": "VOID", "settlement_rule": "synthetic-rules",
        "selection_return": 0.0, "source_id": "synthetic-final",
        "graded_at": at(hours=2, minutes=30), "leg_outcomes": ["VOID", "VOID"]}
    raw = result_source(path,val_id,result,at(hours=2,minutes=10))
    assert store.append_result(path,val_id,result)
    assert validation.attach_result_source(path,val_id,1,raw)
    with pytest.raises(ValueError, match="chronology"):
        validation.append_outcome(path, val_id, 1, source_id="synthetic-final",
                                  available_at=at(hours=2, minutes=20),
                                  observed_at=at(hours=2, minutes=10))
    assert validation.append_outcome(path, val_id, 1, source_id="synthetic-final",
                                     available_at=at(hours=2, minutes=10),
                                     observed_at=at(hours=2, minutes=20))
    report = validation.evaluate_plan(path, plan_id)
    assert report["cohorts"]["validation"]["outcomes"]["VOID"] == 1
    assert report["cohorts"]["holdout"]["outcomes"]["PENDING"] == 1
    assert report["cohorts"]["validation"]["probability"]["scored_raw"] == 0
    artifact = validation.freeze_report(path, plan_id)
    assert artifact["report"]["status"] == "UNVALIDATED"
    with pytest.raises(ValueError, match="passed immutable"):
        validation.record_deployment_review(path, artifact_id=artifact["artifact_id"],
                                            validation_id="synthetic-validation", reviewer_id="synthetic-reviewer",
                                            validation_state="PROVISIONAL_VALIDATED")
    for product in validation.PRODUCTS:
        status = validation.product_readiness(path, product)
        assert status["validation_state"] == "UNVALIDATED"
        assert status["production_eligible"] is False
        assert status["recommended_stake"] == 0.0
