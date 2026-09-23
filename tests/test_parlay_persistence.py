"""Synthetic persistence fixtures; they never represent validated live wagers."""
from copy import deepcopy
from pathlib import Path
import runpy
import sqlite3

import pytest

from app_core import parlay_persistence as store


AT = "2026-09-23T15:00:00+00:00"


def sample(*, actionable=False):
    combined_leg_hash = store._digest(["leg-1", "leg-2"])
    decision = {
        "parlay_id": "synthetic-parlay-1",
        "product_type": "STANDARD_PARLAY",
        "status": "ACTIONABLE" if actionable else "UNVALIDATED",
        "sportsbook": "SyntheticBook",
        "leg_hash": combined_leg_hash,
        "ticket_hash": "synthetic-ticket-hash",
        "decision_at": AT,
        "production_eligible": actionable,
        "recommended_stake_dollars": 5.0 if actionable else 0.0,
        "blockers": [] if actionable else ["PRODUCT_UNVALIDATED"],
    }
    legs = [
        {"candidate_id": "synthetic-1", "game_id": "g1", "sport": "MLB", "market": "total",
         "selection": "Over", "line": 8.5, "sportsbook": "SyntheticBook", "leg_hash": "leg-1"},
        {"candidate_id": "synthetic-2", "game_id": "g2", "sport": "MLB", "market": "spread",
         "selection": "Home", "line": -1.5, "sportsbook": "SyntheticBook", "leg_hash": "leg-2"},
    ]
    quote = {
        "quote_id": "synthetic-quote-1", "sportsbook": "SyntheticBook", "leg_hash": decision["leg_hash"],
        "ticket_hash": decision["ticket_hash"], "quote_source": "synthetic-fixture",
        "provider_ticket_id": "synthetic-provider-ticket",
        "odds_decimal": 3.5, "odds_american": 250, "quoted_at": "2026-09-23T14:55:00+00:00",
        "expires_at": "2026-09-23T15:15:00+00:00",
        "verification_state": "VERIFIED" if actionable else "UNVERIFIED",
    }
    if actionable:
        decision.update(
            validation_id="synthetic-validation", validation_state="STANDARD_VALIDATED",
            policy_id="synthetic-policy", policy_version="v1", model_id="synthetic-model",
            model_version="v1", calibration_id="synthetic-calibration",
            calibration_version="v1", evidence_id="synthetic-frozen-evidence",
            probability_mean=0.40, probability_conservative=0.35, probability_push=0.0,
            conservative_ev=0.225,
        )
    return decision, legs, quote


def test_migration_preserves_existing_data_and_prevents_orphans(tmp_path):
    path = tmp_path / "shared.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE existing_straight_evidence (id TEXT PRIMARY KEY, payload TEXT)")
        db.execute("INSERT INTO existing_straight_evidence VALUES ('old', 'untouched')")
    with store.connect(path) as db:
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert db.execute("SELECT payload FROM existing_straight_evidence WHERE id='old'").fetchone() == ("untouched",)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO parlay_result (decision_id,parlay_id,grading_version,result_kind,graded_at,recorded_at,input_hash,payload,payload_hash) "
                       "VALUES ('missing','missing',1,'SELECTION',?,?,?,'{}',?)", (AT, AT, "x", "x"))
    # Connecting again must not replace an earlier migration or existing data.
    with store.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM existing_straight_evidence").fetchone() == (1,)


def test_failed_snapshot_write_rolls_back_all_rows(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, quote = sample()
    with pytest.raises(ValueError, match="gate_name"):
        store.save_decision(path, decision, legs, quote=quote,
                            gates=[{"gate_name": "", "passed": False, "blockers": ["MISSING"]}])
    with store.connect(path) as db:
        for table in ("parlay_identity", "parlay_ticket", "parlay_leg", "parlay_quote", "parlay_gate"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_research_snapshot_is_exact_immutable_and_reprice_appends(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, quote = sample()
    first = store.save_decision(path, decision, legs, quote=quote,
                                gates=[{"gate_name": "product_validation", "passed": False,
                                        "blockers": ["PRODUCT_UNVALIDATED"], "evaluated_at": AT}])
    assert store.save_decision(path, decision, legs, quote=quote,
                               gates=[{"gate_name": "product_validation", "passed": False,
                                       "blockers": ["PRODUCT_UNVALIDATED"], "evaluated_at": AT}]) == first
    frozen = store.load_decision(path, first)
    assert frozen["decision"] == decision
    assert frozen["quote"] == quote
    assert frozen["gates"][0]["blockers"] == ["PRODUCT_UNVALIDATED"]
    with store.connect(path) as db:
        assert db.execute("SELECT model_id,calibration_id,validation_id FROM parlay_ticket").fetchone() == (None, None, None)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE parlay_ticket SET status='ACTIONABLE' WHERE decision_id=?", (first,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM parlay_leg WHERE decision_id=?", (first,))
    repriced = deepcopy(quote)
    repriced.update(quote_id="synthetic-quote-2", odds_decimal=3.4, odds_american=240)
    later = dict(decision, decision_at="2026-09-23T15:02:00+00:00")
    second = store.save_decision(path, later, legs, quote=repriced)
    assert second != first
    assert store.decision_history(path, decision["parlay_id"]) == [first, second]
    assert store.load_decision(path, first)["quote"]["odds_decimal"] == 3.5
    assert store.load_decision(path, second)["quote"]["odds_decimal"] == 3.4
    with pytest.raises(store.EvidenceConflict, match="quote identity conflict"):
        store.save_decision(path, dict(decision, decision_at="2026-09-23T15:03:00+00:00"),
                            legs, quote=dict(quote, odds_decimal=2.0))
    with pytest.raises(store.EvidenceConflict, match="identity"):
        changed_legs = deepcopy(legs)
        changed_legs[0]["leg_hash"] = "changed-leg"
        changed_hash = store._digest(sorted(leg["leg_hash"] for leg in changed_legs))
        store.save_decision(path, dict(decision, leg_hash=changed_hash,
                                       decision_at="2026-09-23T15:04:00+00:00"),
                            changed_legs, quote=dict(repriced, quote_id="synthetic-quote-3",
                                                     leg_hash=changed_hash))


def test_actionable_snapshot_requires_exact_quote_and_validated_facts(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, quote = sample(actionable=True)
    decision_id = store.save_decision(path, decision, legs, quote=quote)
    assert store.load_decision(path, decision_id)["decision"]["recommended_stake_dollars"] == 5
    for change in (
        {"production_eligible": False}, {"recommended_stake_dollars": 0},
        {"validation_id": None}, {"calibration_version": None}, {"conservative_ev": 0},
    ):
        rejected = dict(decision, parlay_id="other-" + str(change), **change)
        with pytest.raises(ValueError):
            store.save_decision(path, rejected, legs, quote=dict(quote, quote_id="other-" + str(change),
                                                                  parlay_id=rejected["parlay_id"]))
    with pytest.raises(ValueError, match="quote ticket_hash mismatch"):
        store.save_decision(path, dict(decision, parlay_id="other-hash"), legs,
                            quote=dict(quote, quote_id="other-hash", ticket_hash="changed"))
    with pytest.raises(ValueError, match="quote is not current"):
        store.save_decision(path, dict(decision, parlay_id="other-expiry"), legs,
                            quote=dict(quote, quote_id="other-expiry", expires_at="2026-09-23T14:59:00+00:00"))


def test_grading_revisions_are_append_only_and_original_decision_is_unchanged(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, quote = sample()
    decision_id = store.save_decision(path, decision, legs, quote=quote)
    first = {"grading_version": 1, "result_kind": "SELECTION",
             "graded_at": "2026-09-24T15:00:00+00:00", "outcome": "PUSH",
             "settlement": "REFUND", "selection_return": 0.0, "source_id": "synthetic-final-1",
             "leg_outcomes": ["WIN", "PUSH"]}
    assert store.append_result(path, decision_id, first)
    assert not store.append_result(path, decision_id, dict(first, graded_at="2026-09-24T16:00:00+00:00"))
    with pytest.raises(store.EvidenceConflict):
        store.append_result(path, decision_id, dict(first, outcome="LOSS"))
    with pytest.raises(ValueError, match="unchanged grading inputs"):
        store.append_result(path, decision_id,
                            dict(first, grading_version=2, graded_at="2026-09-25T15:00:00+00:00"))
    correction = dict(first, grading_version=2, graded_at="2026-09-25T15:00:00+00:00",
                      outcome="VOID", settlement="VOID", source_id="synthetic-final-correction")
    assert store.append_result(path, decision_id, correction)
    assert store.result_revisions(path, decision_id) == [first, correction]
    assert store.load_decision(path, decision_id)["decision"] == decision
    with pytest.raises(ValueError, match="selection grading cannot claim"):
        store.append_result(path, decision_id, dict(correction, grading_version=3, net_return=10))
    with store.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE parlay_result SET outcome='WIN' WHERE decision_id=?", (decision_id,))


def test_validation_evidence_scopes_product_and_is_immutable(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, _ = sample()
    decision_id = store.save_decision(path, decision, legs)
    evidence = {"evidence_id": "synthetic-validation-evidence", "decision_id": decision_id,
                "parlay_id": decision["parlay_id"], "product_type": "STANDARD_PARLAY",
                "validation_state": "UNVALIDATED", "evidence_at": AT,
                "metrics": {"sample_size": 0}}
    assert store.append_validation_evidence(path, evidence)
    assert not store.append_validation_evidence(path, evidence)
    assert store.validation_evidence(path, "STANDARD_PARLAY") == [evidence]
    assert store.validation_evidence(path, "SAME_GAME_PARLAY") == []
    with pytest.raises(store.EvidenceConflict):
        store.append_validation_evidence(path, dict(evidence, validation_state="STANDARD_VALIDATED"))
    with pytest.raises(ValueError, match="identity mismatch"):
        store.append_validation_evidence(path, dict(evidence, evidence_id="wrong-product",
                                                     product_type="SAME_GAME_PARLAY"))
    with pytest.raises(ValueError, match="unknown parlay_id"):
        store.append_validation_evidence(path, dict(evidence, evidence_id="orphan",
                                                     decision_id=None, parlay_id="missing"))
    with store.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM parlay_validation_evidence")


def test_accepted_return_requires_actual_receipt_and_price(tmp_path):
    path = tmp_path / "parlays.sqlite3"
    decision, legs, quote = sample(actionable=True)
    decision_id = store.save_decision(path, decision, legs, quote=quote)
    result = {"grading_version": 1, "result_kind": "ACCEPTED_WAGER", "graded_at": "2026-09-24T15:00:00+00:00",
              "outcome": "WIN", "settlement": "SETTLED", "net_return": 12.5,
              "accepted_wager_id": "synthetic-receipt-1", "accepted_decimal_odds": 3.5,
              "accepted_stake": 5.0, "placed_at": "2026-09-23T15:01:00+00:00",
              "settlement_rule": "SYNTHETIC_STANDARD_RULE"}
    with pytest.raises(ValueError, match="accepted wager requires receipt"):
        store.append_result(path, decision_id, dict(result, accepted_wager_id=None))
    with pytest.raises(ValueError, match="separate from selection return"):
        store.append_result(path, decision_id, dict(result, selection_return=12.5))
    assert store.append_result(path, decision_id, result)
    with store.connect(path) as db:
        row = db.execute("SELECT result_kind,selection_return,net_return,accepted_wager_id,accepted_decimal_odds,accepted_stake,placed_at "
                         "FROM parlay_result WHERE decision_id=?", (decision_id,)).fetchone()
    assert row == ("ACCEPTED_WAGER", None, 12.5, "synthetic-receipt-1", 3.5, 5.0,
                   "2026-09-23T15:01:00+00:00")
    other_decision = dict(decision, parlay_id="synthetic-parlay-2", ticket_hash="synthetic-ticket-hash-2")
    other_quote = dict(quote, quote_id="synthetic-quote-2", ticket_hash="synthetic-ticket-hash-2")
    other_id = store.save_decision(path, other_decision, legs, quote=other_quote)
    with pytest.raises(sqlite3.IntegrityError, match="receipt belongs"):
        store.append_result(path, other_id, result)


def test_synthetic_engine_actionable_is_frozen_then_graded(tmp_path):
    """Reuse the engine's mechanical positive control; it is not live validation."""
    fixtures = runpy.run_path(str(Path(__file__).with_name("test_true_parlay_engine_unittest.py")))
    from core.true_parlay_engine import evaluate_ticket

    inputs = fixtures["case"]()
    evaluation = evaluate_ticket(**inputs)
    assert evaluation["status"] == "ACTIONABLE"
    assert evaluation["blockers"] == []
    path = tmp_path / "parlays.sqlite3"
    decision_id = store.save_decision(
        path, evaluation, evaluation["legs"], quote=inputs["quote"],
        gates=[{"gate_name": "engine", "passed": True, "blockers": [],
                "evaluated_at": fixtures["NOW"].isoformat()}],
    )
    frozen = store.load_decision(path, decision_id)
    assert frozen["decision"] == evaluation
    assert frozen["quote"] == inputs["quote"]
    assert frozen["legs"] == evaluation["legs"]
    with store.connect(path) as db:
        stored_legs = db.execute(
            "SELECT model_version,calibration_version,evidence_id,model_available_at,"
            "analysis_at,quote_at,probability_conservative FROM parlay_leg ORDER BY leg_index"
        ).fetchall()
    assert len(stored_legs) == len(evaluation["legs"])
    assert all(row[0] == "m1" and row[1] == "c1" and row[2] == "e1" and
               row[3] == fixtures["at"](-100) and row[4] == fixtures["at"](-2) and
               row[5] == fixtures["at"](-1) and row[6] == 0.55 for row in stored_legs)
    with pytest.raises(ValueError, match="decision legs mismatch"):
        store.save_decision(path, evaluation, evaluation["legs"][:-1], quote=inputs["quote"])
    with pytest.raises(ValueError, match="quote quoted_decimal_odds mismatch"):
        store.save_decision(path, dict(evaluation, quoted_decimal_odds=2.5), evaluation["legs"],
                            quote=inputs["quote"])
    result = {"grading_version": 1, "result_kind": "SELECTION", "graded_at": fixtures["at"](180),
              "outcome": "WIN", "settlement": "SETTLED", "selection_return": 10.0,
              "source_id": "synthetic-result-fixture"}
    assert store.append_result(path, decision_id, result)
    assert store.result_revisions(path, decision_id) == [result]
    assert store.load_decision(path, decision_id) == frozen
