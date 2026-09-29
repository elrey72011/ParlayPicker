#!/usr/bin/env python3
"""Emit static inventory plus deterministic producer-to-customer runtime traces."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.prediction_engine import PredictionEngine, VERTEX_FEATURE_COLUMNS  # noqa: E402
from app_core.export_scope import label_wager_export, production_wagers  # noqa: E402
from app_core.lean_card import attach_play_stakes, score_best_picks_rows  # noqa: E402
from core.parlay_engine import generate_parlays  # noqa: E402
from core.price_value import price_value  # noqa: E402
from core.probability_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA_VERSION,
    CONDITIONAL_FIT_TARGET,
    CONDITIONAL_PROBABILITY_SEMANTICS,
    FITTING_IMPLEMENTATION_VERSION,
    PER_CANDIDATE_PUSH_CONVERSION,
    CalibrationTable,
    calibrated_unconditional_mass,
    calibration_digest,
)
from core.production_gate import evaluate_absolute_production_gate  # noqa: E402
from core.probability_semantics import unconditional_from_conditional  # noqa: E402
from services.subscriber.contracts import Recommendation  # noqa: E402
from services.subscriber.launch_gate import release_decision  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TRACKED_SYMBOLS = frozenset(
    {
        "apply_bucket_calibration",
        "apply_calibration",
        "calibration_provenance",
        "conditional_probabilities",
        "fit_isotonic_calibration",
        "load_calibration",
        "price_value",
        "verify_reviewed_submission",
    }
)


def build_trace(root: Path = ROOT) -> dict:
    calls: dict[str, list[dict[str, object]]] = {
        symbol: [] for symbol in sorted(TRACKED_SYMBOLS)
    }
    parse_errors: list[dict[str, str]] = []
    excluded_parts = {
        ".git", ".venv", "archive", "node_modules", "outputs", "site-packages",
        "test-results",
    }
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if excluded_parts.intersection(relative.parts) or any(
            part.startswith(".") for part in relative.parts
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append(
                {"path": relative.as_posix(), "error": type(exc).__name__}
            )
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in calls:
                calls[name].append(
                    {
                        "path": relative.as_posix(),
                        "line": int(node.lineno),
                    }
                )
    return {
        "schema_version": 1,
        "trace_kind": "static_python_call_sites",
        "read_only": True,
        "tracked_symbols": sorted(TRACKED_SYMBOLS),
        "calls": calls,
        "parse_errors": parse_errors,
        "qualification_authority": "integrations/subscriber_release/authority.py",
        "subscriber_contract": "services/subscriber/contracts.py",
        "calibration_runtime": "core/probability_calibration.py",
        "notes": [
            "This trace locates source-level consumers; it does not activate calibration or markets.",
            "Dynamic imports and non-Python consumers require separate runtime evidence.",
        ],
    }


MARKET_SCOPES = (
    ("NFL", "SPREAD"), ("NFL", "TOTAL"),
    ("NCAAF", "SPREAD"), ("NCAAF", "TOTAL"),
    ("NBA", "SPREAD"), ("NBA", "TOTAL"),
    ("NCAAB", "SPREAD"), ("NCAAB", "TOTAL"),
    ("MLB", "RUN_LINE"), ("MLB", "TOTAL"),
    ("NHL", "PUCK_LINE"), ("NHL", "TOTAL"),
)


class _FrozenTraceModel:
    """External-I/O replacement; the real application inference method still runs."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.full(len(frame), 0.60, dtype=float)
        return np.column_stack([1.0 - values, values])


def _source_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def _artifact(knots: list[list[float]] | None = None) -> CalibrationTable:
    manifest = {
        "schema_version": 1,
        "fit_target": CONDITIONAL_FIT_TARGET,
        "probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "source_predictor_version": "trace-model-v1",
        "training_scope": {"exact_sport": "MULTI", "exact_market_family": "MULTI"},
        "observation_count": 2,
        "fitted_row_count": 2,
        "independent_event_count": 2,
        "rows": [],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_hash"] = hashlib.sha256(encoded).hexdigest()
    meta = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_method": "isotonic",
        "fitting_implementation_version": FITTING_IMPLEMENTATION_VERSION,
        "artifact_status": "RESEARCH_CANDIDATE_ONLY",
        "source": "ISOLATED_FIXTURE",
        "source_predictor_version": "trace-model-v1",
        "training_scope": manifest["training_scope"],
        "probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "fit_target": CONDITIONAL_FIT_TARGET,
        "output_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "push_conversion": PER_CANDIDATE_PUSH_CONVERSION,
        "fit_manifest": manifest,
        "calibration_trained_through": "2026-09-20T00:00:00Z",
        "calibration_available_at": "2026-09-21T00:00:00Z",
        "validation": {
            "promotable": False,
            "train_end": "2026-09-01T00:00:00Z",
            "test_start": "2026-09-02T00:00:00Z",
        },
    }
    payload = {"knots": knots or [[0.0, 0.0], [1.0, 1.0]], "meta": meta}
    meta["calibration_version"] = calibration_digest(payload)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    acceptance = {
        "requested_mode": "explicit_research",
        "acceptance_state": "RESEARCH_ONLY",
        "rejection_reasons": ["ISOLATED_FIXTURE_NOT_PRODUCTION_AUTHORITY"],
        "canonical_artifact_digest": meta["calibration_version"],
        "artifact_raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return CalibrationTable(
        payload,
        acceptance=acceptance,
        raw_sha256=acceptance["artifact_raw_sha256"],
    )


def _runtime_rows() -> pd.DataFrame:
    rows = []
    for index, (sport, market) in enumerate(MARKET_SCOPES):
        base = {column: 0.0 for column in VERTEX_FEATURE_COLUMNS}
        if market in {"SPREAD", "RUN_LINE", "PUCK_LINE"}:
            market_type, line = "spread_home", -2.0
        else:
            market_type, line = "total_over", 46.0
        base.update(
            league=sport,
            home_team=f"{sport} Home",
            away_team=f"{sport} Away",
            game_date="2026-09-30",
            matchup_id=f"trace-{sport.lower()}-{market.lower()}",
            market_type=market_type,
            line=line,
            odds_american=100,
            feature_stats_fallback=False,
            quote_id=f"trace-quote-{index}",
        )
        rows.append(base)
    return pd.DataFrame(rows)


def build_runtime_trace(
    root: Path = ROOT,
    *,
    calibration_artifact: CalibrationTable | None = None,
) -> dict:
    """Execute actual inference, calibration, price, gate, and subscriber code."""
    started = datetime.now(timezone.utc)
    source_sha = _source_sha(root)
    frame = _runtime_rows()
    engine = PredictionEngine(model_path="models/does-not-exist.json")
    engine.model = _FrozenTraceModel()
    engine.use_fallback = False
    raw_probabilities = engine.predict_batch(frame)
    artifact = (
        calibration_artifact
        if calibration_artifact is not None
        else _artifact()
    )
    traces: list[dict[str, object]] = []
    for index, ((sport, market), (_, row), raw_probability) in enumerate(
        zip(MARKET_SCOPES, frame.iterrows(), raw_probabilities)
    ):
        candidate_id = str(row["matchup_id"])
        snapshot = {
            "candidate_id": candidate_id,
            "sport": sport,
            "market": market,
            "line": float(row["line"]),
            "quote_id": str(row["quote_id"]),
        }
        snapshot_hash = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        mean = calibrated_unconditional_mass(raw_probability, 0.10, artifact)
        conservative = unconditional_from_conditional(0.55, 0.10)
        mean_price = price_value(mean["p_win"], mean["p_push"], 2.0)
        conservative_price = price_value(
            conservative["p_win"], conservative["p_push"], 2.0
        )
        gate = evaluate_absolute_production_gate(
            mean["p_win"],
            model_expected_value=0.20,
            push_probability=mean["p_push"],
            decimal_odds=2.0,
            conservative_probability=conservative["p_win"],
        ).iloc[0]
        consumer_frame = pd.DataFrame(
            [
                {
                    "league": sport,
                    "Home": str(row["home_team"]),
                    "Away": str(row["away_team"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "matchup_id": candidate_id,
                    "canonical_event_id": candidate_id,
                    "market_type": str(row["market_type"]),
                    "source_predictor_version": "trace-model-v1",
                    "effective_win_probability": raw_probability,
                    "effective_expected_value": 0.20,
                    "effective_edge": 0.10,
                    "expected_value": 0.20,
                    "probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
                    "push_probability": 0.10,
                    "push_probability_source": "fixture_per_candidate_supported_push_v1",
                    "conservative_probability": 0.55,
                    "conservative_probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
                    "odds_american": 100,
                    "decimal_odds": 2.0,
                    "Pick_Status": "Actionable",
                    "consensus_agreement": "Agrees",
                    "best_pick": f"{sport} trace selection",
                    "line": float(row["line"]),
                    "quote_id": str(row["quote_id"]),
                    "quote_observed_at": "2026-09-28T12:00:00Z",
                    "qualified_pick": False,
                    "wager_approved": False,
                    "production_eligible": False,
                    "Kelly_Bet_Size": 0.0,
                    "game_already_started_flag": False,
                }
            ]
        )
        route_assertions: dict[str, dict[str, object]] = {}
        research_card = score_best_picks_rows(
            consumer_frame, calibration=artifact, bucket_stats=None
        )
        research_ok = (
            len(research_card) == 1
            and float(research_card.iloc[0]["Final_P_Win"]) == mean["p_win"]
            and float(research_card.iloc[0]["Final_P_Push"]) == mean["p_push"]
            and float(research_card.iloc[0]["Final_P_Loss"]) == mean["p_loss"]
            and float(research_card.iloc[0]["Price_Break_Even"])
            == mean_price["break_even"]
            and float(research_card.iloc[0]["Mean_EV_Per_Unit"])
            == mean_price["expected_value"]
            and str(research_card.iloc[0]["Quote_ID"]) == str(row["quote_id"])
            and str(research_card.iloc[0]["Calibration_Version"])
            == artifact.payload["meta"]["calibration_version"]
        )
        route_assertions["controlled_research"] = {
            "invoked": True,
            "status": "EXERCISED_PASS" if research_ok else "EXERCISED_FAIL",
            "consumer": "app_core.lean_card.score_best_picks_rows",
            "assertion": "one research row preserves the calibrated unconditional win value",
        }
        staked_card = attach_play_stakes(research_card, unit=1.0)
        public_input = staked_card.copy()
        public_board = label_wager_export(public_input)
        public_blocked = (
            len(public_board) == 1
            and not bool(public_board.iloc[0]["Bettable"])
            and float(public_board.iloc[0]["Suggested_Stake"]) == 0.0
            and float(public_board.iloc[0]["Mean_EV_Per_Unit"])
            == mean_price["expected_value"]
            and str(public_board.iloc[0]["Quote_ID"]) == str(row["quote_id"])
        )
        route_assertions["public_board"] = {
            "invoked": True,
            "status": "EXPECTED_BLOCK_VERIFIED" if public_blocked else "EXERCISED_FAIL",
            "consumer": "app_core.export_scope.label_wager_export",
            "assertion": "research-only fixture is exported as a zero-stake non-wager",
        }
        strict_rows = production_wagers(public_input)
        route_assertions["strict_decision"] = {
            "invoked": True,
            "status": "EXPECTED_BLOCK_VERIFIED" if strict_rows.empty else "EXERCISED_FAIL",
            "consumer": "app_core.export_scope.production_wagers",
            "assertion": "unapproved fixture cannot enter the strict wager set",
        }
        parlay_rows = generate_parlays(public_input)
        route_assertions["parlay_consumer"] = {
            "invoked": True,
            "status": "EXPECTED_BLOCK_VERIFIED" if parlay_rows.empty else "EXERCISED_FAIL",
            "consumer": "core.parlay_engine.generate_parlays",
            "assertion": "a single unapproved research row cannot create a parlay",
        }
        recommendation = Recommendation.model_validate(
            {
                "schema_version": 2,
                "recommendation_id": candidate_id,
                "exact_sport": sport,
                "exact_market_family": market,
                "canonical_event_id": candidate_id,
                "selection": f"{sport} trace selection",
                "line": float(row["line"]),
                "sportsbook_id": "fixture-book",
                "odds_american": 100,
                "odds_decimal": 2.0,
                "quote_id": str(row["quote_id"]),
                "quote_observed_at": "2026-09-28T12:00:00Z",
                "analysis_generated_at": "2026-09-28T12:01:00Z",
                "event_start_utc": "2026-09-30T12:00:00Z",
                "expiry_at": "2026-09-30T11:00:00Z",
                "model_id": "trace-model-v1",
                "model_artifact_hash": "a" * 64,
                "model_target_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
                "calibration_id": artifact.payload["meta"]["calibration_version"],
                "validation_artifact_id": "isolated-fixture-not-authority",
                "policy_id": "trace-only",
                "activation_reference": "not-activated",
                "probability_semantics": str(
                    public_board.iloc[0]["Probability_Semantics"]
                ),
                "p_win": float(public_board.iloc[0]["Final_P_Win"]),
                "p_push": float(public_board.iloc[0]["Final_P_Push"]),
                "p_loss": float(public_board.iloc[0]["Final_P_Loss"]),
                "mean_ev_per_unit": float(
                    public_board.iloc[0]["Mean_EV_Per_Unit"]
                ),
                "p_win_conservative": float(
                    public_board.iloc[0]["P_Win_Conservative"]
                ),
                "conservative_ev_per_unit": float(
                    public_board.iloc[0]["Conservative_EV_Per_Unit"]
                ),
                "uncertainty_method": "fixed_push_lower_win_bound",
                "minimum_acceptable_decimal_odds": float(
                    public_board.iloc[0]["Minimum_Acceptable_Decimal_Odds"]
                ),
                "disclosure_version": "trace-fixture-v1",
            }
        )
        customer = recommendation.customer_projection()
        subscriber_ok = (
            customer["recommendation_id"] == candidate_id
            and customer["quote_id"] == str(row["quote_id"])
            and customer["p_win"] == mean["p_win"]
            and customer["mean_ev_per_unit"] == mean_price["expected_value"]
            and customer["conservative_ev_per_unit"]
            == conservative_price["expected_value"]
        )
        route_assertions["subscriber"] = {
            "invoked": True,
            "status": "EXERCISED_PASS" if subscriber_ok else "EXERCISED_FAIL",
            "consumer": "services.subscriber.contracts.Recommendation.customer_projection",
            "assertion": "subscriber v2 projection preserves mean and conservative EV separately",
        }
        release_gate = release_decision(
            authority={
                "revoked": False,
                "upstream_gate_result": "RESEARCH_ONLY",
                "market_status": "RESEARCH",
                "effective_at": started - timedelta(days=1),
                "expires_at": started + timedelta(days=1),
            },
            exact_markets=[f"{sport}:{market}"],
            commercially_enabled_markets=[],
            expiry_at=datetime.fromisoformat(customer["expiry_at"]),
            event_starts=[datetime.fromisoformat(customer["event_start_utc"])],
            reviewed_hash_matches=False,
            rights_present=False,
            now=started,
        )
        traces.append(
            {
                "trace_id": f"runtime-{index + 1:02d}",
                "source_sha": source_sha,
                "runtime": f"python-{sys.version_info.major}.{sys.version_info.minor}",
                "evidence_kind": "ISOLATED_FIXTURE",
                **snapshot,
                "input_snapshot_hash": snapshot_hash,
                "model_predictor_version": "trace-model-v1",
                "raw_probability_conditional": raw_probability,
                "fallback_used": False,
                "fallback_reason": None,
                "blend_version": "prediction_engine.predict_batch",
                "blend_output_conditional": raw_probability,
                "calibration_content_identity": artifact.payload["meta"]["calibration_version"],
                "calibration_raw_sha256": artifact.raw_sha256,
                "calibration_load_mode": "explicit_research_fixture",
                "calibration_acceptance": artifact.acceptance,
                "calibration_input_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
                "calibration_output_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
                "calibration_output_conditional": (
                    mean["p_win"] / (1.0 - mean["p_push"])
                ),
                "push_source": "fixture_per_candidate_supported_push_v1",
                "push_conversion": PER_CANDIDATE_PUSH_CONVERSION,
                "mean_mass": mean,
                "conservative_mass": conservative,
                "conservative_method": "fixed_push_lower_win_bound",
                "mean_ev": mean_price["expected_value"],
                "conservative_ev": conservative_price["expected_value"],
                "sportsbook_break_even_probability": mean_price["break_even"],
                "absolute_edge": mean_price["edge"],
                "upstream_model_ev_diagnostic": 0.20,
                "lean_value_contract": {
                    "status": str(research_card.iloc[0]["Value_Contract_Status"]),
                    "p_win": float(research_card.iloc[0]["Final_P_Win"]),
                    "p_push": float(research_card.iloc[0]["Final_P_Push"]),
                    "p_loss": float(research_card.iloc[0]["Final_P_Loss"]),
                    "break_even": float(research_card.iloc[0]["Price_Break_Even"]),
                    "edge": float(research_card.iloc[0]["Absolute_Edge"]),
                    "mean_ev_per_unit": float(
                        research_card.iloc[0]["Mean_EV_Per_Unit"]
                    ),
                    "conservative_ev_per_unit": float(
                        research_card.iloc[0]["Conservative_EV_Per_Unit"]
                    ),
                    "quote_id": str(research_card.iloc[0]["Quote_ID"]),
                    "calibration_id": str(
                        research_card.iloc[0]["Calibration_Version"]
                    ),
                    "calibration_raw_sha256": str(
                        research_card.iloc[0]["Calibration_Artifact_SHA256"]
                    ),
                },
                "value_source": "core.price_value.price_value",
                "selection_status": str(research_card.iloc[0]["Tier"]),
                "selection_rank": 1,
                "selection_tie_break_key": candidate_id,
                "production_numeric_gate_pass": bool(gate["production_gate_pass"]),
                "production_decision": "BLOCKED",
                "production_blockers": release_gate.reason_codes
                + artifact.acceptance["rejection_reasons"],
                "subscriber_contract_status": route_assertions["subscriber"]["status"],
                "exported_subscriber_values": customer,
                "route_assertions": route_assertions,
                "route_status": {
                    name: assertion["status"]
                    for name, assertion in route_assertions.items()
                },
            }
        )
    route_names = (
        "controlled_research",
        "parlay_consumer",
        "public_board",
        "strict_decision",
        "subscriber",
    )
    return {
        "schema_version": 2,
        "trace_kind": "runtime_probability_value_trace",
        "read_only": True,
        "source_sha": source_sha,
        "runtime": f"python-{sys.version_info.major}.{sys.version_info.minor}",
        "evidence_kind": "ISOLATED_FIXTURE",
        "external_io": "FROZEN_STUB",
        "model_quality": "NOT_VERIFIED",
        "market_scope_count": len(MARKET_SCOPES),
        "route_invocation_counts": {
            name: sum(
                bool(record["route_assertions"][name]["invoked"]) for record in traces
            )
            for name in route_names
        },
        "records": traces,
    }


def build_nonidentity_priced_value_trace(root: Path = ROOT) -> dict:
    """Exercise the actual consumers with the Post-#2356 nonidentity example."""

    report = build_runtime_trace(
        root,
        calibration_artifact=_artifact([[0.1, 0.2], [0.9, 0.8]]),
    )
    return {
        **report,
        "schema_version": 1,
        "trace_kind": "post2356_nonidentity_priced_value_trace",
        "counterexample": {
            "raw_conditional_probability": 0.60,
            "calibrated_conditional_probability": 0.575,
            "push_probability": 0.10,
            "decimal_odds": 2.0,
            "required_final": {
                "p_win": 0.5175,
                "p_push": 0.10,
                "p_loss": 0.3825,
                "break_even": 0.45,
                "edge": 0.0675,
                "mean_ev_per_unit": 0.135,
                "conservative_ev_per_unit": 0.09,
            },
        },
    }


def _validation_errors(payload: object, mode: str) -> list[str]:
    errors: list[str] = []
    static = payload.get("static_inventory") if mode == "both" and isinstance(payload, dict) else payload
    runtime = payload.get("runtime_trace") if mode == "both" and isinstance(payload, dict) else payload
    if mode in {"static", "both"}:
        if not isinstance(static, dict) or static.get("trace_kind") != "static_python_call_sites":
            errors.append("STATIC_INVENTORY_MISSING")
        elif not any(static.get("calls", {}).values()):
            errors.append("STATIC_CALL_SITE_EVIDENCE_MISSING")
    if mode in {"runtime", "both", "priced"}:
        accepted_trace_kinds = {"runtime_probability_value_trace"}
        if mode == "priced":
            accepted_trace_kinds.add("post2356_nonidentity_priced_value_trace")
        if not isinstance(runtime, dict) or runtime.get("trace_kind") not in accepted_trace_kinds:
            errors.append("RUNTIME_TRACE_MISSING")
        else:
            records = runtime.get("records")
            if not isinstance(records, list) or not records:
                errors.append("RUNTIME_RECORDS_MISSING")
            else:
                for record in records:
                    assertions = record.get("route_assertions", {})
                    if not assertions or any(
                        not value.get("invoked")
                        or value.get("status") in {"EXERCISED_FAIL", "NOT_EXERCISED"}
                        for value in assertions.values()
                    ):
                        errors.append(
                            f"ROUTE_EVIDENCE_INCOMPLETE:{record.get('trace_id', 'UNKNOWN')}"
                        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--mode", choices=("static", "runtime", "priced", "both"), default="both"
    )
    args = parser.parse_args(argv)
    payload = (
        build_trace()
        if args.mode == "static"
        else build_runtime_trace()
        if args.mode == "runtime"
        else build_nonidentity_priced_value_trace()
        if args.mode == "priced"
        else {
            "schema_version": 2,
            "static_inventory": build_trace(),
            "runtime_trace": build_runtime_trace(),
        }
    )
    errors = _validation_errors(payload, args.mode)
    if isinstance(payload, dict):
        payload["validation"] = {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
