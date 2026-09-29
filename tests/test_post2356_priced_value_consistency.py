from __future__ import annotations

import json

import pandas as pd
import pytest

from app_core.export_scope import label_wager_export
from app_core.lean_card import score_best_picks_rows
from core import probability_calibration as pc
from core.production_gate import evaluate_absolute_production_gate
from scripts.trace_probability_path import build_nonidentity_priced_value_trace
from scripts import trace_probability_path as tracer
from services.subscriber.contracts import Recommendation


SCOPE = {"exact_sport": "NFL", "exact_market_family": "SPREAD"}


def _manifest() -> dict:
    rows = [
        {
            "source_record_id": f"record-{index}",
            "source_path": "isolated-fixture.json",
            "source_hash": str(index + 1) * 64,
            "canonical_event_id": f"training-event-{index}",
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
        for index, (probability, outcome) in enumerate(
            ((0.1, "LOSS"), (0.9, "WIN"))
        )
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


def _row() -> pd.DataFrame:
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
            }
        ]
    )


def _priced_row(tmp_path, monkeypatch, **updates):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    row = _row()
    row["canonical_event_id"] = "event-p01"
    row["quote_id"] = "quote-p01"
    row["quote_observed_at"] = "2026-09-28T12:00:00Z"
    row["decimal_odds"] = 2.0
    row["push_probability"] = 0.10
    row["push_probability_source"] = "candidate-supported-push-v1"
    row["conservative_probability"] = 0.55
    row["conservative_probability_semantics"] = (
        pc.CONDITIONAL_PROBABILITY_SEMANTICS
    )
    for key, value in updates.items():
        row[key] = value
    return score_best_picks_rows(row, bucket_stats=None).iloc[[0]]


def test_p01_actual_lean_gate_uses_one_push_aware_price_contract(
    tmp_path, monkeypatch
):
    priced = _priced_row(tmp_path, monkeypatch).iloc[0]

    assert priced["Final_P_Win"] == pytest.approx(0.5175)
    assert priced["Final_P_Push"] == pytest.approx(0.10)
    assert priced["Final_P_Loss"] == pytest.approx(0.3825)
    assert priced["Price_Break_Even"] == pytest.approx(0.45)
    assert priced["Absolute_Edge"] == pytest.approx(0.0675)
    assert priced["Mean_EV_Per_Unit"] == pytest.approx(0.135)
    assert priced["P_Win_Conservative"] == pytest.approx(0.495)
    assert priced["Conservative_EV_Per_Unit"] == pytest.approx(0.09)
    assert priced["Upstream_Model_EV"] == pytest.approx(0.20)
    assert priced["Calibrated_EV"] == pytest.approx(0.135)
    assert priced["Quote_ID"] == "quote-p01"
    assert priced["Calibration_Version"] == _payload()["meta"][
        "calibration_version"
    ]


def test_p03_p04_gate_uses_mean_ev_not_break_even_ratio():
    gate = evaluate_absolute_production_gate(
        0.5175,
        model_expected_value=0.20,
        push_probability=0.10,
        decimal_odds=2.0,
        conservative_probability=0.495,
    ).iloc[0]

    assert gate["final_p_win"] == pytest.approx(0.5175)
    assert gate["final_p_push"] == pytest.approx(0.10)
    assert gate["final_p_loss"] == pytest.approx(0.3825)
    assert gate["sportsbook_break_even_probability"] == pytest.approx(0.45)
    assert gate["absolute_production_edge"] == pytest.approx(0.0675)
    assert gate["mean_expected_value_per_unit"] == pytest.approx(0.135)
    assert gate["conservative_expected_value_per_unit"] == pytest.approx(0.09)
    assert 0.5175 / 0.45 - 1.0 == pytest.approx(0.15)
    assert gate["mean_expected_value_per_unit"] != pytest.approx(0.15)
    assert bool(gate["production_gate_pass"])


def test_s05_no_push_mean_and_conservative_ev_remain_distinct():
    gate = evaluate_absolute_production_gate(
        0.60,
        model_expected_value=0.20,
        push_probability=0.0,
        decimal_odds=2.0,
        conservative_probability=0.55,
    ).iloc[0]

    assert gate["mean_expected_value_per_unit"] == pytest.approx(0.20)
    assert gate["conservative_expected_value_per_unit"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("p_win", "p_push", "decimal", "expected_status", "expected_ev"),
    [
        (0.60, 0.0, 2.0, "PUSH_AWARE_VERIFIED", 0.20),
        (0.50, 0.0, 2.0, "PUSH_AWARE_VERIFIED", 0.0),
        (0.40, 0.0, 2.0, "PUSH_AWARE_VERIFIED", -0.20),
        (0.55, None, 2.0, "INVALID", None),
        (0.55, -0.1, 2.0, "INVALID", None),
        (0.55, 0.1, 1.0, "INVALID", None),
    ],
)
def test_s06_price_contract_covers_positive_zero_negative_and_invalid_inputs(
    p_win, p_push, decimal, expected_status, expected_ev
):
    gate = evaluate_absolute_production_gate(
        p_win,
        model_expected_value=0.20,
        push_probability=p_push,
        decimal_odds=decimal,
    ).iloc[0]

    assert gate["pricing_contract_status"] == expected_status
    if expected_ev is None:
        assert pd.isna(gate["mean_expected_value_per_unit"])
        assert not bool(gate["production_gate_pass"])
    else:
        assert gate["mean_expected_value_per_unit"] == pytest.approx(expected_ev)


def test_s06_declared_contract_rejects_missing_push_and_mismatched_quote(
    tmp_path, monkeypatch
):
    missing_push = _priced_row(tmp_path, monkeypatch).copy()
    source = _row().drop(columns="push_probability")
    source["canonical_event_id"] = "event-missing-push"
    source["quote_id"] = "quote-missing-push"
    source["decimal_odds"] = 2.0
    scored_missing = score_best_picks_rows(source, bucket_stats=None).iloc[0]
    assert scored_missing["Value_Contract_Status"] == "INVALID"
    assert not bool(scored_missing["Production_Gate_Pass"])

    mismatched = _priced_row(tmp_path, monkeypatch, decimal_odds=1.90).iloc[0]
    assert mismatched["Value_Contract_Status"] == "QUOTE_PRICE_MISMATCH"
    assert not bool(mismatched["Production_Gate_Pass"])
    assert len(missing_push) == 1


@pytest.mark.parametrize(
    ("line", "push_probability", "expected_mean_ev"),
    [(-2.0, 0.10, 0.135), (-2.5, 0.0, 0.15)],
)
def test_s06_integer_and_half_point_lines_keep_explicit_push_semantics(
    tmp_path,
    monkeypatch,
    line,
    push_probability,
    expected_mean_ev,
):
    priced = _priced_row(
        tmp_path,
        monkeypatch,
        line=line,
        push_probability=push_probability,
    ).iloc[0]

    assert priced["Final_P_Push"] == pytest.approx(push_probability)
    assert priced["Mean_EV_Per_Unit"] == pytest.approx(expected_mean_ev)
    assert priced["Value_Contract_Status"] == "PUSH_AWARE_VERIFIED"


def test_s07_s08_selection_export_and_subscriber_share_final_values(
    tmp_path, monkeypatch
):
    priced = _priced_row(tmp_path, monkeypatch)
    exported = label_wager_export(priced).iloc[0]

    recommendation = Recommendation.model_validate(
        {
            "schema_version": 2,
            "recommendation_id": str(exported["Candidate_ID"]),
            "exact_sport": "NFL",
            "exact_market_family": "SPREAD",
            "canonical_event_id": str(exported["Candidate_ID"]),
            "selection": str(exported["Pick"]),
            "line": -2.0,
            "sportsbook_id": "fixture-book",
            "odds_american": int(exported["Odds_American"]),
            "odds_decimal": float(exported["Odds_Decimal"]),
            "quote_id": str(exported["Quote_ID"]),
            "quote_observed_at": str(exported["Quote_Observed_At"]),
            "analysis_generated_at": "2026-09-28T12:01:00Z",
            "event_start_utc": "2026-09-30T12:00:00Z",
            "expiry_at": "2026-09-30T11:00:00Z",
            "model_id": "predictor-1",
            "model_artifact_hash": "a" * 64,
            "model_target_semantics": pc.CONDITIONAL_PROBABILITY_SEMANTICS,
            "calibration_id": str(exported["Calibration_Version"]),
            "validation_artifact_id": "isolated-fixture-not-authority",
            "policy_id": "trace-only",
            "activation_reference": "not-activated",
            "probability_semantics": str(exported["Probability_Semantics"]),
            "p_win": float(exported["Final_P_Win"]),
            "p_push": float(exported["Final_P_Push"]),
            "p_loss": float(exported["Final_P_Loss"]),
            "mean_ev_per_unit": float(exported["Mean_EV_Per_Unit"]),
            "p_win_conservative": float(exported["P_Win_Conservative"]),
            "conservative_ev_per_unit": float(
                exported["Conservative_EV_Per_Unit"]
            ),
            "uncertainty_method": "fixed_push_lower_win_bound",
            "minimum_acceptable_decimal_odds": float(
                exported["Minimum_Acceptable_Decimal_Odds"]
            ),
            "disclosure_version": "trace-fixture-v1",
        }
    )
    customer = recommendation.customer_projection()

    assert exported["Upstream_Model_EV"] == pytest.approx(0.20)
    assert exported["Mean_EV_Per_Unit"] == pytest.approx(0.135)
    assert exported["EV_Field_Semantics"].startswith("legacy alias")
    assert recommendation.calibration_id == exported["Calibration_Version"]
    assert customer["quote_id"] == exported["Quote_ID"]
    assert customer["odds_decimal"] == exported["Odds_Decimal"]
    assert customer["p_win"] == exported["Final_P_Win"]
    assert customer["p_push"] == exported["Final_P_Push"]
    assert customer["p_loss"] == exported["Final_P_Loss"]
    assert customer["mean_ev_per_unit"] == exported["Mean_EV_Per_Unit"]
    assert customer["conservative_ev_per_unit"] == exported[
        "Conservative_EV_Per_Unit"
    ]
    assert not bool(exported["Bettable"])


def test_s08_bucket_tilt_has_separate_identity_and_no_inherited_authority(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)
    row = _row()
    row["decimal_odds"] = 2.0
    row["push_probability"] = 0.10
    row["qualified_pick"] = True
    row["wager_approved"] = True
    row["production_eligible"] = True
    row["production_bet_amount"] = 5.0
    bucket_stats = {
        "overall": {"n": 100, "win_rate": 0.50},
        "buckets": {"NFL:side:Agrees": {"n": 100, "wins": 80}},
    }

    priced = score_best_picks_rows(row, bucket_stats=bucket_stats).iloc[0]

    assert priced["Calibration_Consumer_Status"] == (
        "CALIBRATION_POST_TRANSFORM_UNAUTHORIZED"
    )
    assert priced["Calibration_Post_Transform"] == "BUCKET_TILT_RESEARCH_ONLY"
    assert priced["Calibration_Post_Transform_Identity"].startswith(
        "bucket-conditional-tilt-v1:"
    )
    assert priced["Value_Contract_Status"] == (
        "CALIBRATION_OR_TRANSFORM_NOT_AUTHORIZED"
    )
    assert not bool(priced["Production_Gate_Pass"])


def test_s08_nonidentity_runtime_trace_reconciles_every_actual_consumer():
    report = build_nonidentity_priced_value_trace()

    assert report["market_scope_count"] == 12
    assert report["route_invocation_counts"] == {
        "controlled_research": 12,
        "parlay_consumer": 12,
        "public_board": 12,
        "strict_decision": 12,
        "subscriber": 12,
    }
    for row in report["records"]:
        assert row["mean_mass"] == pytest.approx(
            {"p_win": 0.5175, "p_push": 0.10, "p_loss": 0.3825}
        )
        assert row["sportsbook_break_even_probability"] == pytest.approx(0.45)
        assert row["absolute_edge"] == pytest.approx(0.0675)
        assert row["mean_ev"] == pytest.approx(0.135)
        assert row["conservative_ev"] == pytest.approx(0.09)
        assert row["lean_value_contract"]["quote_id"] == row["quote_id"]
        assert row["lean_value_contract"]["calibration_id"] == row[
            "calibration_content_identity"
        ]
        assert all(
            status != "EXERCISED_FAIL" for status in row["route_status"].values()
        )


def test_s08_priced_trace_cli_writes_validated_evidence(tmp_path):
    output = tmp_path / "priced.json"

    assert tracer.main(["--mode", "priced", "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"] == {"status": "PASS", "errors": []}
    assert payload["trace_kind"] == "post2356_nonidentity_priced_value_trace"
