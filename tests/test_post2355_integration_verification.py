from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from app_core.lean_card import score_best_picks_rows
from core import probability_calibration as pc
from scripts import trace_probability_path as tracer


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


def _payload() -> dict:
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
    payload = {"knots": [[0.1, 0.2], [0.9, 0.8]], "meta": meta}
    meta["calibration_version"] = pc.calibration_digest(payload)
    return payload


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "NFL",
                "Home": "Home One",
                "Away": "Away One",
                "market_type": "spread_home",
                "source_predictor_version": "predictor-1",
                "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
                "push_probability": 0.0,
                "line": -2.0,
                "effective_win_probability": 0.60,
                "effective_expected_value": 0.20,
                "effective_edge": 0.10,
                "odds_american": 100,
                "Pick_Status": "Actionable",
                "consensus_agreement": "Agrees",
                "best_pick": "Home One -2",
                "qualified_pick": False,
                "game_already_started_flag": False,
            },
            {
                "league": "NBA",
                "Home": "Home Two",
                "Away": "Away Two",
                "market_type": "total_over",
                "source_predictor_version": "predictor-1",
                "probability_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
                "push_probability": 0.0,
                "line": 220.0,
                "effective_win_probability": 0.60,
                "effective_expected_value": 0.20,
                "effective_edge": 0.10,
                "odds_american": 100,
                "Pick_Status": "Actionable",
                "consensus_agreement": "Agrees",
                "best_pick": "Over 220",
                "qualified_pick": False,
                "game_already_started_flag": False,
            },
        ]
    )


def test_n05_static_trace_returns_inventory_and_known_call_site(tmp_path):
    source = tmp_path / "consumer.py"
    source.write_text("def run():\n    return load_calibration()\n", encoding="utf-8")

    report = tracer.build_trace(tmp_path)

    assert report["trace_kind"] == "static_python_call_sites"
    assert report["calls"]["load_calibration"] == [
        {"path": "consumer.py", "line": 2}
    ]


@pytest.mark.parametrize("mode", ["static", "runtime", "both"])
def test_n06_all_tracer_cli_modes_emit_requested_evidence(tmp_path, mode):
    output = tmp_path / f"{mode}.json"
    assert tracer.main(["--mode", mode, "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    if mode in {"static", "both"}:
        static = payload if mode == "static" else payload["static_inventory"]
        assert static["trace_kind"] == "static_python_call_sites"
        assert static["calls"]
    if mode in {"runtime", "both"}:
        runtime = payload if mode == "runtime" else payload["runtime_trace"]
        assert runtime["trace_kind"] == "runtime_probability_value_trace"
        assert runtime["records"]


def test_n06_requested_missing_evidence_is_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(tracer, "build_trace", lambda root=tracer.ROOT: None)
    assert tracer.main(["--mode", "static", "--out", str(tmp_path / "bad.json")]) != 0


def test_n07_n08_runtime_routes_are_invoked_and_legal():
    report = tracer.build_runtime_trace()
    assert report["route_invocation_counts"] == {
        "controlled_research": 12,
        "parlay_consumer": 12,
        "public_board": 12,
        "strict_decision": 12,
        "subscriber": 12,
    }
    for row in report["records"]:
        assert set(row["route_assertions"]) == set(report["route_invocation_counts"])
        assert all(item["invoked"] is True for item in row["route_assertions"].values())
        if row["mean_mass"]["p_push"] > 0:
            assert float(row["line"]).is_integer()


def test_n11_n12_real_lean_consumer_uses_matching_nonidentity_curve_only(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)

    scored = score_best_picks_rows(_rows(), bucket_stats=None)

    assert scored.loc[0, "Calib_Win%"] == pytest.approx(0.575)
    assert scored.loc[0, "Calibration_Consumer_Status"] == "PRODUCTION_CALIBRATION_APPLIED"
    assert scored.loc[0, "Calibration_Version"] == _payload()["meta"]["calibration_version"]
    assert scored.loc[1, "Calib_Win%"] == pytest.approx(0.60)
    assert scored.loc[1, "Calibration_Consumer_Status"] == "CALIBRATION_REJECTED"

    wrong_predictor = _rows().iloc[[0]].copy()
    wrong_predictor["source_predictor_version"] = "other-predictor"
    rejected = score_best_picks_rows(wrong_predictor, bucket_stats=None)
    assert rejected.iloc[0]["Calib_Win%"] == pytest.approx(0.60)
    assert rejected.iloc[0]["Calibration_Consumer_Status"] == "CALIBRATION_REJECTED"


def test_n12_post_load_mutation_still_blocks_application(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    loaded = pc.load_calibration(
        now=NOW,
        expected_predictor_version="predictor-1",
        expected_training_scope=SCOPE,
    )
    assert loaded is not None
    loaded[0][1] = 0.99
    assert pc.apply_calibration(pd.Series([0.60]), loaded).isna().all()


def test_n03_additive_postgres_workflow_enforces_all_completion_cases():
    workflow = Path(".github/workflows/subscriber-completion-postgres.yml").read_text(
        encoding="utf-8"
    )
    assert "tests/paid_launch/case_subscriber_completion.py" in workflow
    for node in (
        "test_offer_and_account_preferences_are_server_owned",
        "test_results_are_complete_and_scoped_to_entitled_product",
        "test_cancellation_remains_available_when_sales_are_disabled",
    ):
        assert node in workflow


def test_n12_rejected_artifact_never_changes_consumer_value(tmp_path, monkeypatch):
    payload = deepcopy(_payload())
    payload["meta"]["source_predictor_version"] = "wrong"
    payload["meta"]["calibration_version"] = pc.calibration_digest(payload)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)

    scored = score_best_picks_rows(_rows().iloc[[0]], bucket_stats=None)
    assert scored.iloc[0]["Calib_Win%"] == pytest.approx(0.60)
    assert scored.iloc[0]["Calibration_Consumer_Status"] == "CALIBRATION_REJECTED"


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"push_conversion": "unsupported-push-conversion"},
        {"probability_semantics": "win_unconditional"},
    ],
)
def test_n12_legacy_or_semantically_incompatible_artifact_is_not_applied(
    tmp_path, monkeypatch, mutation
):
    payload = _payload()
    payload["meta"].update(mutation)
    payload["meta"]["calibration_version"] = pc.calibration_digest(payload)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)

    scored = score_best_picks_rows(_rows().iloc[[0]], bucket_stats=None)

    assert scored.iloc[0]["Calib_Win%"] == pytest.approx(0.60)
    assert scored.iloc[0]["Calibration_Consumer_Status"] == "CALIBRATION_REJECTED"


def test_n12_missing_predictor_context_is_explicitly_uncalibrated(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    rows = _rows().iloc[[0]].drop(columns="source_predictor_version")

    scored = score_best_picks_rows(rows, bucket_stats=None)

    assert scored.iloc[0]["Calib_Win%"] == pytest.approx(0.60)
    assert scored.iloc[0]["Calibration_Consumer_Status"] == "CONSUMER_CONTEXT_MISSING"


def test_n13_conditional_calibration_converts_push_before_downstream_value(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    row = _rows().iloc[[0]].copy()
    row["push_probability"] = 0.10

    scored = score_best_picks_rows(row, bucket_stats=None)

    assert scored.iloc[0]["Calib_Win%"] == pytest.approx(0.575 * 0.90)
    assert scored.iloc[0]["Calibration_Consumer_Status"] == "PRODUCTION_CALIBRATION_APPLIED"


def test_n08_n12_nonzero_push_on_half_point_line_is_rejected(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    row = _rows().iloc[[0]].copy()
    row["line"] = -2.5
    row["push_probability"] = 0.10

    scored = score_best_picks_rows(row, bucket_stats=None)

    assert scored.iloc[0]["Calib_Win%"] == pytest.approx(0.60)
    assert scored.iloc[0]["Calibration_Consumer_Status"] == (
        "PUSH_SUPPORT_MISSING_OR_INVALID"
    )
