from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from app_core.prediction_engine import PredictionEngine, VERTEX_FEATURE_COLUMNS
from core import probability_calibration as pc
from core.price_value import price_value
from core.probability_semantics import unconditional_from_conditional
from scripts import fit_calibration
from services.subscriber.contracts import Recommendation
from scripts.trace_probability_path import MARKET_SCOPES, build_runtime_trace


NOW = "2026-09-03T00:00:00Z"
SCOPE = {"exact_sport": "NFL", "exact_market_family": "SPREAD"}


def _manifest() -> dict:
    rows = [
        {
            "source_record_id": f"record-{index}",
            "source_path": "isolated-fixture.json",
            "source_hash": str(index + 1) * 64,
            "canonical_event_id": f"event-{index}",
            "exact_sport": "NFL",
            "exact_market_family": "SPREAD",
            "source_predictor_version": "predictor-1",
            "input_probability": probability,
            "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
            "outcome_class": outcome,
            "event_time": f"2026-0{index + 8}-01T00:00:00Z",
            "outcome_available_at": f"2026-0{index + 8}-01T06:00:00Z",
            "sample_weight": 1.0,
            "fit_included": True,
            "exclusion_reason": "INCLUDED",
        }
        for index, (probability, outcome) in enumerate(((0.1, "LOSS"), (0.9, "WIN")))
    ]
    value = {
        "schema_version": 1,
        "fit_target": pc.CONDITIONAL_FIT_TARGET,
        "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
        "source_predictor_version": "predictor-1",
        "training_scope": SCOPE,
        "observation_count": 2,
        "fitted_row_count": 2,
        "independent_event_count": 2,
        "rows": rows,
    }
    value["manifest_hash"] = pc._manifest_digest(value)
    return value


def _payload(**overrides) -> dict:
    meta = {
        "schema_version": pc.CALIBRATION_SCHEMA_VERSION,
        "calibration_method": "isotonic",
        "fitting_implementation_version": pc.FITTING_IMPLEMENTATION_VERSION,
        "artifact_status": "PRODUCTION_CANDIDATE",
        "calibration_trained_through": "2026-09-01T00:00:00Z",
        "calibration_available_at": "2026-09-02T00:00:00Z",
        "outcome_available_through": "2026-09-01T06:00:00Z",
        "source": "isolated-fixture",
        "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
        "fit_target": pc.CONDITIONAL_FIT_TARGET,
        "output_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
        "push_conversion": pc.PER_CANDIDATE_PUSH_CONVERSION,
        "source_predictor_version": "predictor-1",
        "training_scope": SCOPE,
        "fit_manifest": _manifest(),
        "validation": {
            "promotable": True,
            "train_end": "2026-08-01T00:00:00Z",
            "test_start": "2026-08-02T00:00:00Z",
        },
    }
    meta.update(overrides)
    payload = {"knots": [[0.1, 0.2], [0.9, 0.8]], "meta": meta}
    meta["calibration_version"] = pc.calibration_digest(payload)
    return payload


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("version", [999, "3", True, None])
def test_a01_unknown_and_malformed_schema_never_gets_production_acceptance(
    tmp_path, monkeypatch, version
):
    artifact = tmp_path / "artifact.json"
    payload = _payload(schema_version=version)
    payload["meta"]["calibration_version"] = pc.calibration_digest(payload)
    _write(artifact, payload)
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)

    report = pc.inspect_calibration_artifact(
        now=NOW,
        expected_predictor_version="predictor-1",
        expected_training_scope=SCOPE,
    )

    assert report["acceptance_state"] != "PRODUCTION_ACCEPTED"
    assert any(reason.startswith("SCHEMA_VERSION_") for reason in report["rejection_reasons"])


def test_a02_a03_metadata_removal_rejection_digest_predictor_and_scope_fail_closed(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.json"
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    cases = []
    missing_version = _payload()
    missing_version["meta"].pop("schema_version")
    missing_version["meta"]["calibration_version"] = pc.calibration_digest(missing_version)
    cases.append((missing_version, "SCHEMA_VERSION_MISSING"))
    rejected = _payload(artifact_status="REJECTED")
    rejected["meta"]["calibration_version"] = pc.calibration_digest(rejected)
    cases.append((rejected, "ARTIFACT_NOT_PRODUCTION_CANDIDATE"))
    bad_digest = _payload()
    bad_digest["meta"]["calibration_version"] = "0" * 64
    cases.append((bad_digest, "CALIBRATION_DIGEST_MISMATCH"))

    for payload, reason in cases:
        _write(artifact, payload)
        report = pc.inspect_calibration_artifact(
            now=NOW,
            expected_predictor_version="predictor-1",
            expected_training_scope=SCOPE,
        )
        assert report["acceptance_state"] != "PRODUCTION_ACCEPTED"
        assert reason in report["rejection_reasons"]

    _write(artifact, _payload())
    predictor = pc.inspect_calibration_artifact(
        now=NOW,
        expected_predictor_version="predictor-other",
        expected_training_scope=SCOPE,
    )
    scope = pc.inspect_calibration_artifact(
        now=NOW,
        expected_predictor_version="predictor-1",
        expected_training_scope={"exact_sport": "NFL", "exact_market_family": "TOTAL"},
    )
    assert "SOURCE_PREDICTOR_MISMATCH" in predictor["rejection_reasons"]
    assert "TRAINING_SCOPE_MISMATCH" in scope["rejection_reasons"]
    assert pc.inspect_calibration_artifact(artifact)["acceptance_state"] == "RESEARCH_ONLY"
    accepted = pc.inspect_calibration_artifact(
        now=NOW,
        expected_predictor_version="predictor-1",
        expected_training_scope=SCOPE,
    )
    assert accepted["acceptance_state"] == "PRODUCTION_ACCEPTED"


def test_a04_loader_reads_and_validates_one_snapshot(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    first = _payload(artifact_status="RESEARCH_CANDIDATE_ONLY")
    second = _payload()
    reads = [json.dumps(first).encode(), json.dumps(second).encode()]
    calls = []

    def changing_read_bytes(_self):
        calls.append(1)
        return reads[min(len(calls) - 1, 1)]

    monkeypatch.setattr(pc.Path, "read_bytes", changing_read_bytes)
    loaded = pc.load_calibration(artifact)

    assert len(calls) == 1
    assert loaded.payload["meta"]["artifact_status"] == "RESEARCH_CANDIDATE_ONLY"
    assert loaded.acceptance["acceptance_state"] == "RESEARCH_ONLY"


def test_a05_mutation_invalidates_snapshot_binding(tmp_path):
    artifact = tmp_path / "artifact.json"
    _write(artifact, _payload())
    loaded = pc.load_calibration(artifact)
    assert loaded.trusted_snapshot_valid()
    loaded.acceptance["acceptance_state"] = "PRODUCTION_ACCEPTED"
    assert not loaded.trusted_snapshot_valid()
    assert pc.calibration_provenance(loaded) == {}
    assert pc.calibrated_unconditional_mass(0.6, 0.1, loaded) is None


def test_a06_explicit_research_identity_is_not_production_authority(tmp_path):
    artifact = tmp_path / "artifact.json"
    _write(artifact, _payload())
    loaded = pc.load_calibration(artifact)
    facts = pc.calibration_provenance(loaded, now=NOW)
    assert facts["calibration_version"]
    assert facts["acceptance_state"] == "RESEARCH_ONLY"
    assert bool(loaded) is False


def _recommendation(mass, *, conditional_lower=0.55, odds=2.0):
    conservative = unconditional_from_conditional(conditional_lower, mass["p_push"])
    return Recommendation.model_validate(
        {
            "schema_version": 2,
            "recommendation_id": "rec-push",
            "exact_sport": "NFL",
            "exact_market_family": "SPREAD",
            "canonical_event_id": "event-1",
            "selection": "Home -3",
            "line": -3,
            "sportsbook_id": "fixture",
            "odds_american": 100,
            "odds_decimal": odds,
            "quote_id": "quote-1",
            "quote_observed_at": "2026-09-01T10:00:00Z",
            "analysis_generated_at": "2026-09-01T10:01:00Z",
            "event_start_utc": "2026-09-01T12:00:00Z",
            "expiry_at": "2026-09-01T11:00:00Z",
            "model_id": "fixture",
            "model_artifact_hash": "a" * 64,
            "model_target_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
            "calibration_id": "fixture",
            "validation_artifact_id": "fixture",
            "policy_id": "fixture",
            "activation_reference": "fixture",
            "probability_semantics": "win_unconditional_with_push",
            **mass,
            "mean_ev_per_unit": price_value(mass["p_win"], mass["p_push"], odds)["expected_value"],
            "p_win_conservative": conservative["p_win"],
            "conservative_ev_per_unit": price_value(
                conservative["p_win"], conservative["p_push"], odds
            )["expected_value"],
            "uncertainty_method": "fixed_push_lower_win_bound",
            "minimum_acceptable_decimal_odds": 2.01,
            "disclosure_version": "fixture",
        }
    )


def test_a08_a10_conditional_push_chain_and_subscriber_ev_contract(tmp_path):
    artifact = tmp_path / "artifact.json"
    payload = _payload()
    payload["knots"] = [[0.0, 0.0], [1.0, 1.0]]
    payload["meta"]["calibration_version"] = pc.calibration_digest(payload)
    _write(artifact, payload)
    table = pc.load_calibration(artifact)

    mass = pc.calibrated_unconditional_mass(0.60, 0.10, table)
    assert mass == pytest.approx({"p_win": 0.54, "p_push": 0.10, "p_loss": 0.36})
    assert price_value(mass["p_win"], mass["p_push"], 2.0)["expected_value"] == pytest.approx(0.18)
    assert price_value(mass["p_win"], mass["p_push"], 1.5)["expected_value"] == pytest.approx(-0.09)
    parsed = _recommendation(mass)
    assert parsed.mean_ev_per_unit == pytest.approx(0.18)
    assert parsed.p_win_conservative == pytest.approx(0.495)
    assert parsed.conservative_ev_per_unit == pytest.approx(0.09)

    no_push = unconditional_from_conditional(0.60, 0.0)
    parsed_no_push = _recommendation(no_push)
    assert parsed_no_push.mean_ev_per_unit == pytest.approx(0.20)
    assert parsed_no_push.conservative_ev_per_unit == pytest.approx(0.10)


def test_a08_trainer_preserves_pushes_while_fit_rate_stays_conditional():
    frame = pd.DataFrame(
        {
            "effective_win_probability": [0.6] * 100,
            "W/L": ["WIN"] * 54 + ["PUSH"] * 10 + ["LOSS"] * 36,
        }
    )
    extracted = fit_calibration._extract(frame)
    fitted = extracted[extracted["fit_included"]]
    assert extracted["outcome_class"].value_counts().to_dict() == {
        "WIN": 54, "LOSS": 36, "PUSH": 10
    }
    assert fitted["win"].mean() == pytest.approx(0.60)
    assert (extracted["outcome_class"] == "WIN").mean() == pytest.approx(0.54)
    assert (
        extracted.loc[extracted["outcome_class"].eq("PUSH"), "exclusion_reason"]
        .eq("PUSH_EXCLUDED_FROM_CONDITIONAL_BINARY_FIT")
        .all()
    )


def test_a11_push_evidence_is_never_invented():
    assert unconditional_from_conditional(0.60, None) is None
    assert unconditional_from_conditional(0.60, 1.0) is None
    assert unconditional_from_conditional(float("nan"), 0.1) is None
    assert unconditional_from_conditional(0.60, 0.0) == pytest.approx(
        {"p_win": 0.60, "p_push": 0.0, "p_loss": 0.40}
    )


def test_a12_a14_manifest_cutoffs_endpoints_weights_and_refit_identity(tmp_path):
    count = 202
    dates = pd.date_range("2025-01-01", periods=count, freq="D", tz="UTC")
    available = dates + pd.Timedelta(hours=6)
    frame = pd.DataFrame(
        {
            "game_date": dates,
            "outcome_available_at": available,
            "effective_win_probability": ([0.2, 0.8] * 100) + [0.0, 1.0],
            "W/L": (["LOSS", "WIN"] * 100) + ["LOSS", "WIN"],
            "sample_weight": [1.0] * 201 + [3.0],
            "canonical_event_id": [f"event-{index // 2}" for index in range(count)],
            "exact_sport": "NFL",
            "exact_market_family": "SPREAD",
            "source_predictor_version": "predictor-1",
            "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
            "source_record_id": [f"record-{index}" for index in range(count)],
        }
    )
    source = tmp_path / "graded.csv"
    first_out = tmp_path / "candidate-1.json"
    second_out = tmp_path / "candidate-2.json"
    frame.to_csv(source, index=False)
    args = [
        str(tmp_path), str(first_out), "--source-predictor-version", "predictor-1",
        "--exact-sport", "NFL", "--exact-market-family", "SPREAD",
        "--training-scope", "NFL spread fixture",
    ]
    assert fit_calibration.main(args) == 0
    first = json.loads(first_out.read_text())
    meta = first["meta"]
    rows = meta["fit_manifest"]["rows"]
    assert meta["calibration_trained_through"] == dates.max().isoformat()
    assert meta["outcome_available_through"] == available.max().isoformat()
    assert rows[-2]["input_probability"] == 0.0 and rows[-2]["fit_included"]
    assert rows[-1]["input_probability"] == 1.0 and rows[-1]["sample_weight"] == 3.0
    assert meta["fit_manifest"]["independent_event_count"] == count // 2

    frame.loc[len(frame)] = deepcopy(frame.iloc[-1])
    frame.loc[len(frame) - 1, "source_record_id"] = "record-refit"
    frame.loc[len(frame) - 1, "canonical_event_id"] = "event-refit"
    frame.loc[len(frame) - 1, "game_date"] = dates.max() + pd.Timedelta(days=1)
    frame.loc[len(frame) - 1, "outcome_available_at"] = available.max() + pd.Timedelta(days=1)
    frame.to_csv(source, index=False)
    args[1] = str(second_out)
    assert fit_calibration.main(args) == 0
    second = json.loads(second_out.read_text())
    assert second["meta"]["calibration_version"] != meta["calibration_version"]
    assert second["meta"]["calibration_trained_through"] > meta["calibration_trained_through"]


class _InvariantModel:
    def predict_proba(self, frame):
        values = np.full(len(frame), 0.57, dtype=float)
        return np.column_stack([1.0 - values, values])


class _BrokenModel:
    def predict_proba(self, _frame):
        raise RuntimeError("fixture model unavailable")


def _prediction_rows(count=1):
    base = {column: 0.0 for column in VERTEX_FEATURE_COLUMNS}
    rows = []
    for index in range(count):
        row = dict(base)
        row.update(
            league="NFL",
            home_team=f"Home {index}",
            away_team=f"Away {index}",
            game_date="2026-09-01",
            matchup_id=f"event-{index}",
            market_type="spread_home",
            feature_stats_fallback=False,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _engine(model):
    engine = PredictionEngine(model_path="models/does-not-exist.json")
    engine.model = model
    engine.use_fallback = False
    return engine


def test_a15_a17_batch_composition_order_and_ties_do_not_change_probabilities():
    alone = _engine(_InvariantModel()).predict_batch(_prediction_rows(1))[0]
    batch_engine = _engine(_InvariantModel())
    batch = batch_engine.predict_batch(_prediction_rows(8))
    reordered = _engine(_InvariantModel()).predict_batch(_prediction_rows(8).iloc[::-1])

    assert alone == pytest.approx(0.57)
    assert batch == pytest.approx([0.57] * 8)
    assert reordered == pytest.approx([0.57] * 8)
    assert batch_engine._last_metrics["batch_compression_detected"] is True
    assert batch_engine._last_metrics["hybrid_fallback_triggered"] is False
    assert len(set(batch)) == 1  # no hash epsilon is injected into probability


def test_a18_actual_model_failure_remains_explicitly_unavailable():
    engine = _engine(_BrokenModel())
    assert engine.predict_batch(_prediction_rows(3)) == [None, None, None]


def test_a19_a20_runtime_trace_executes_all_market_routes_and_reconciles_values():
    report = build_runtime_trace()
    assert report["market_scope_count"] == 12
    assert {(row["sport"], row["market"]) for row in report["records"]} == set(MARKET_SCOPES)
    for row in report["records"]:
        assert row["raw_probability_conditional"] == pytest.approx(0.60)
        assert row["mean_mass"] == pytest.approx(
            {"p_win": 0.54, "p_push": 0.10, "p_loss": 0.36}
        )
        assert row["mean_ev"] == pytest.approx(0.18)
        assert row["conservative_ev"] == pytest.approx(0.09)
        assert row["exported_subscriber_values"]["p_win"] == pytest.approx(0.54)
        assert row["subscriber_contract_status"] == "EXERCISED_PASS"
        assert row["production_decision"] == "BLOCKED"
        assert row["route_status"] == {
            "strict_decision": "EXPECTED_BLOCK_VERIFIED",
            "controlled_research": "EXERCISED_PASS",
            "public_board": "EXPECTED_BLOCK_VERIFIED",
            "subscriber": "EXERCISED_PASS",
            "parlay_consumer": "EXPECTED_BLOCK_VERIFIED",
        }


def test_a21_changed_calibration_identity_cannot_match_old_review_binding():
    first = _payload()
    second = deepcopy(first)
    second["knots"][1][1] = 0.79
    second["meta"]["calibration_version"] = pc.calibration_digest(second)
    assert first["meta"]["calibration_version"] != second["meta"]["calibration_version"]
    mass = unconditional_from_conditional(0.60, 0.10)
    recommendation = _recommendation(mass)
    assert recommendation.calibration_id == "fixture"
    assert recommendation.calibration_id not in {
        first["meta"]["calibration_version"], second["meta"]["calibration_version"]
    }
