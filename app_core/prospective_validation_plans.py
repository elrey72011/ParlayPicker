"""Prospectively frozen, exact-market validation methodology for six sports.

The release creates policy plans, not model or calibration evidence. An unbound
plan remains UNVALIDATED; binding a fitted model requires an explicit next plan
version before that version's validation cohort is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from app_core import prospective_evidence as evidence


PLAN_POLICY_VERSION = "six-sport-market-2026-09-23-v1"
TRAINING_CUTOFF = "2026-10-31T23:59:59+00:00"
VALIDATION_START = "2026-11-01T00:00:00+00:00"
VALIDATION_END = "2027-11-01T00:00:00+00:00"
HOLDOUT_START = "2027-11-01T00:00:00+00:00"
HOLDOUT_END = "2028-11-01T00:00:00+00:00"
MINIMUM_INDEPENDENT_EVENTS = 200
MINIMUM_EFFECTIVE_EVENTS = 200

EXPECTED_MARKETS = (
    ("NFL", "SPREAD"), ("NFL", "TOTAL"),
    ("NCAAF", "SPREAD"), ("NCAAF", "TOTAL"),
    ("NBA", "SPREAD"), ("NBA", "TOTAL"),
    ("NCAAB", "SPREAD"), ("NCAAB", "TOTAL"),
    ("MLB", "RUN_LINE"), ("MLB", "TOTAL"),
    ("NHL", "PUCK_LINE"), ("NHL", "TOTAL"),
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _plan_id(sport: str, market: str) -> str:
    return f"prospective-{sport.lower()}-{market.lower()}-2026-09-23-v1"


def _template(sport: str, market: str) -> dict:
    """Version 1 is intentionally unbound until exact-scope models exist."""
    return {
        "validation_plan_id": _plan_id(sport, market),
        "sport": sport,
        "market_family": market,
        "version": 1,
        "model_id": None,
        "calibration_id": None,
        "plan_policy_version": PLAN_POLICY_VERSION,
        "model_scope": {
            "sport": sport,
            "market_family": market,
            "identity": "EXACT_SPORT_MARKET_MODEL_AND_CALIBRATION",
            "initial_binding": "UNBOUND_REQUIRES_NEW_VERSION_BEFORE_VALIDATION",
            "cross_sport_or_market_pooling": "PROHIBITED",
            "market_implied_probability_as_model": "PROHIBITED",
        },
        "training_cutoff": TRAINING_CUTOFF,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "training_cutoff_policy": {
            "result_availability": "RESULT_AVAILABLE_AT_ON_OR_BEFORE_TRAINING_CUTOFF",
            "feature_availability": "BEFORE_PREDICTION_AND_EVENT_START",
            "market_quote_availability": "STRICTLY_PREGAME",
            "model_selection": "TRAINING_AND_VALIDATION_ONLY",
            "holdout_used_for_training_or_threshold_selection": False,
        },
        "minimum_independent_sample": MINIMUM_INDEPENDENT_EVENTS,
        "minimum_effective_sample": MINIMUM_EFFECTIVE_EVENTS,
        "independence_method": "ONE_EARLIEST_PREDICTION_PER_EVENT_V1",
        "independence_policy": {
            "unit": "ONE_PROVIDER_BOUND_EVENT_PER_EXACT_SPORT_MARKET",
            "repricing_or_repeated_capture_adds_unit": False,
            "duplicate_provider_adds_unit": False,
            "corrected_result_adds_unit": False,
            "spread_and_total_are_separate_families": True,
            "report_raw_and_independent_counts": True,
        },
        "probability_thresholds": {
            "max_brier": 0.24,
            "max_log_loss": 0.68,
            "max_calibration_error": 0.08,
            "min_coverage": 0.90,
            "coverage_denominator": "ONE_FIRST_PREDICTION_PER_EVENT",
            "decided_probability": "P_WIN_DIVIDED_BY_P_WIN_PLUS_P_LOSS",
        },
        "calibration_requirements": {
            "required": True,
            "exact_sport_market_and_model_scope": True,
            "available_before_validation_start": True,
            "fit_outcomes_available_before_fit_cutoff": True,
            "insufficient_evidence_status": "INSUFFICIENT_EVIDENCE",
        },
        "price_evidence_requirements": {
            "min_verified_entry_coverage": 1.0,
            "exact_event_selection_line_sportsbook_price_time": True,
            "provider_response_replay_required": True,
            "post_start_or_synthetic_quote_accepted": False,
        },
        "clv_policy": {
            "required": True,
            "min_comparable_close_coverage": 0.80,
            "same_event_market_selection_book_and_line": True,
            "certified_replayable_close_only": True,
            "near_start_candidate_is_verified_close": False,
            "missing_close_status": "UNAVAILABLE",
        },
        "value_roi_policy": {
            "min_paper_roi": 0.0,
            "paper_stake_per_qualified_prediction_units": 1.0,
            "accepted_wager_roi_until_execution_evidence": "UNAVAILABLE",
            "selection_fixed_before_result": True,
        },
        "push_void_policy": {
            "win_loss_scoring": "DECIDED_EVENTS_ONLY",
            "push": "REFUND_PAPER_UNIT_EXCLUDE_FROM_DECIDED_SCORE",
            "void": "REFUND_PAPER_UNIT_EXCLUDE_FROM_DECIDED_SCORE",
            "pending_or_needs_review": "BLOCK_VALIDATION",
            "half_point_push": "ZERO_UNLESS_VERIFIED_SETTLEMENT_EXCEPTION",
        },
        "deployment_criteria": {
            "target_state": "PROVISIONAL_VALIDATED",
            "maturity_rules": {
                "exact_market_model_and_calibration": True,
                "independent_validation_review": True,
                "separate_owner_activation": True,
                "no_automatic_wager_or_stake": True,
            },
        },
        "promotion_criteria": {
            "validation_and_holdout_both_meet_frozen_thresholds": True,
            "certified_comparable_closing_prices": True,
            "separate_deployment_review": True,
            "separate_owner_activation_and_exposure_authority": True,
            "automatic_promotion": False,
            "stake_until_separate_activation": 0,
        },
    }


def plan_specs(*, source_commit: str) -> list[dict]:
    """Return 12 immutable policy inputs, each bound to a source commit."""
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if tuple((sport, market) for sport, markets in evidence.SPORT_MARKETS.items()
             for market in markets) != EXPECTED_MARKETS:
        raise ValueError("canonical sport/market registry changed; create a reviewed plan version")
    rows = []
    for sport, market in EXPECTED_MARKETS:
        row = _template(sport, market)
        row["policy_source_hash"] = _digest(row)
        row["source_commit"] = source_commit
        rows.append(row)
    return rows


def _verify_existing(row: dict, spec: dict) -> dict:
    payload = json.loads(row["payload"])
    artifact_hash = payload.pop("artifact_hash", None)
    if artifact_hash != row["artifact_hash"] or _digest(payload) != artifact_hash:
        raise evidence.EvidenceConflict("frozen validation plan artifact hash mismatch")
    if (row["validation_plan_id"] != spec["validation_plan_id"] or row["version"] != 1 or
            payload.get("policy_source_hash") != spec["policy_source_hash"] or
            payload.get("plan_policy_version") != PLAN_POLICY_VERSION or
            payload.get("model_id") is not None or payload.get("calibration_id") is not None):
        raise evidence.EvidenceConflict("current validation plan differs from frozen v1 policy")
    return row


def _verify_chain(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: row["version"])
    for version, row in enumerate(ordered, start=1):
        if row["version"] != version or (version > 1 and
                row["supersedes_plan_id"] != ordered[version - 2]["validation_plan_id"]):
            raise evidence.EvidenceConflict("validation plan version chain is inconsistent")
        payload = json.loads(row["payload"])
        artifact_hash = payload.pop("artifact_hash", None)
        if artifact_hash != row["artifact_hash"] or _digest(payload) != artifact_hash:
            raise evidence.EvidenceConflict("frozen validation plan artifact hash mismatch")
    return ordered[-1]


def freeze_current_validation_plans(path: str | Path | None, *, source_commit: str) -> list[dict]:
    """Idempotently install v1 plans, preserving any explicit newer versions.

    Restore the authenticated canonical store first. This function neither
    reads outcomes nor creates models, reviews, activation, or wagers.
    """
    specs = plan_specs(source_commit=source_commit)
    existing = evidence.read_records(path, "prospective_validation_plan")
    by_scope: dict[tuple[str, str], list[dict]] = {}
    for row in existing:
        scope = row["sport"], row["market_family"]
        by_scope.setdefault(scope, []).append(row)
    expected = set(EXPECTED_MARKETS)
    if set(by_scope) - expected:
        raise evidence.EvidenceConflict("unknown existing sport/market validation plan")
    for spec in specs:
        history = by_scope.get((spec["sport"], spec["market_family"]), [])
        if history:
            _verify_chain(history)
            initial = next((row for row in history if row["version"] == 1), None)
            if initial is None:
                raise evidence.EvidenceConflict("first validation plan version is missing")
            _verify_existing(initial, spec)
    frozen = []
    for spec in specs:
        history = by_scope.get((spec["sport"], spec["market_family"]), [])
        if not history:
            evidence.freeze_validation_plan(path, spec)
            row = evidence.load_record(path, "prospective_validation_plan", spec["validation_plan_id"])
            if row is None:
                raise evidence.EvidenceConflict("validation plan write not readable")
            _verify_existing(row, spec)
            current = row
        else:
            current = _verify_chain(history)
        if current is None:
            raise evidence.EvidenceConflict("validation plan write not readable")
        current_payload = json.loads(current["payload"])
        deployment = evidence.deployment_state(path, spec["sport"], spec["market_family"])
        frozen.append({
            "sport": spec["sport"], "market_family": spec["market_family"],
            "validation_plan_id": current["validation_plan_id"], "version": current["version"],
            "frozen_at": current["frozen_at"], "source_commit": current_payload.get("source_commit"),
            "policy_source_hash": current_payload.get("policy_source_hash"),
            "artifact_hash": current["artifact_hash"], "model_id": current["model_id"],
            "calibration_id": current["calibration_id"],
            "validation_state": deployment["validation_state"],
            "production_eligible": deployment["production_eligible"],
            "recommended_stake": deployment["recommended_stake"],
        })
    if len(frozen) != 12 or {(r["sport"], r["market_family"]) for r in frozen} != expected:
        raise evidence.EvidenceConflict("exactly 12 current validation plans were not frozen")
    return frozen
