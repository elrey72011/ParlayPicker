"""Frozen, fail-closed football validation V2 planning and replay gates.

No source record is converted to prospective evidence by this module.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from app_core import prospective_evidence as evidence
from app_core import prospective_validation_plans as v1

POLICY_VERSION = "football-short-season-2026-09-24-v2"
SCOPES = (("NFL", "SPREAD"), ("NFL", "TOTAL"), ("NCAAF", "SPREAD"), ("NCAAF", "TOTAL"))
ALPHA = 0.05
LOG_PROBABILITY_FLOOR = 0.01
CALIBRATION_BINS = 10
# Precision budgets are frozen before outcomes. They express the maximum
# uncertainty we will tolerate, rather than an assumed football win rate.
PRECISION = {"brier": 0.05, "log_loss": 0.10, "calibration_error": 0.05,
             "coverage": 0.05}
METRIC_COHORT_TESTS = 8  # Four metrics in two untouched prospective cohorts.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _time(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def uncertainty_radius(n: int, *, metric: str) -> float:
    """Simultaneous fixed-sample Hoeffding/McDiarmid radius at 95% familywise confidence.

    ECE adds a sqrt(bins/n) finite-bin plug-in bias allowance. The union bound
    covers four metrics across confirmation and holdout. Checkpoints are fixed
    at the frozen minimum; repeated peeking is not permitted.
    """
    if n <= 0 or metric not in PRECISION:
        raise ValueError("positive n and known metric required")
    common = math.sqrt(math.log(2 * METRIC_COHORT_TESTS / ALPHA) / (2 * n))
    if metric == "log_loss":
        return -math.log(LOG_PROBABILITY_FLOOR) * common
    if metric == "calibration_error":
        return common + math.sqrt(CALIBRATION_BINS / n)
    return common


def calculated_minimum() -> dict:
    """Find the first n satisfying every frozen distribution-free precision budget.

    One event is one independent trial. No sport-specific variance, coverage or
    push rate is invented from the supplied one-game CSV samples.
    """
    lo, hi = 1, 1
    def enough(n):
        return all(uncertainty_radius(n, metric=k) <= v for k, v in PRECISION.items())
    while not enough(hi):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if enough(mid):
            hi = mid
        else:
            lo = mid + 1
    return {"effective_decided_events": lo,
            "minimum_raw_events_at_full_coverage": lo,
            "radii": {k: uncertainty_radius(lo, metric=k) for k in PRECISION},
            "confidence": 1 - ALPHA,
            "assumption": "independent events, fixed checkpoint, no assumed sport variance or missingness"}


def conservative_probability(mean_probability: float, interval: Mapping) -> dict:
    """Use a registered exact-scope calibration interval; never infer one from cohort n."""
    p = float(mean_probability)
    if (not math.isfinite(p) or not 0 <= p <= 1 or not isinstance(interval, Mapping) or
            interval.get("method_version") != POLICY_VERSION or not interval.get("calibration_id")):
        raise ValueError("frozen calibrated uncertainty interval required")
    try:
        lower, upper = float(interval["lower"]), float(interval["upper"])
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("uncertainty interval bounds required") from exc
    if not all(math.isfinite(x) for x in (lower, upper)) or not 0 <= lower <= p <= upper <= 1:
        raise ValueError("uncertainty interval must contain the mean")
    return {"mean_probability": p, "uncertainty_interval": [lower, upper],
            "conservative_probability": lower}


def historical_replay_eligible(row: Mapping) -> dict:
    """Assess reproducible as-of history; passing never grants prospective status."""
    blockers = []
    event_id = row.get("provider_event_id") or row.get("event_id")
    start = _time(row.get("game_start_utc") or row.get("scheduled_start"))
    prediction = _time(row.get("prediction_generated_at") or row.get("prediction_timestamp"))
    quote = _time(row.get("odds_recorded_at") or row.get("quote_timestamp"))
    result = _time(row.get("result_available_at") or row.get("outcome_recorded_at"))
    features = _time(row.get("features_generated_at") or row.get("feature_frozen_at"))
    model_available = _time(row.get("model_available_at"))
    train_cutoff = _time(row.get("model_trained_through") or row.get("training_cutoff"))
    latest_training_input = _time(row.get("training_inputs_latest_available_at"))
    if (not event_id or not start or (row.get("sport"), row.get("market_family")) not in SCOPES or
            not row.get("home_team_id") or not row.get("away_team_id") or
            row.get("home_team_id") == row.get("away_team_id")):
        blockers.append("EVENT_IDENTITY_AMBIGUOUS")
    if row.get("line") is None:
        blockers.append("EXACT_MARKET_LINE_UNAVAILABLE")
    try:
        valid_price = math.isfinite(float(row.get("decimal_odds"))) and float(row["decimal_odds"]) > 1
    except (TypeError, ValueError):
        valid_price = False
    if not row.get("sportsbook") or not valid_price or not quote or row.get("quote_verified") is not True:
        blockers.append("HISTORICAL_PRICE_UNVERIFIED")
    if not prediction or not start or prediction >= start:
        blockers.append("PREDICTION_CHRONOLOGY_INVALID")
    if quote and prediction and quote > prediction:
        blockers.append("QUOTE_AFTER_PREDICTION")
    if quote and start and quote >= start:
        blockers.append("QUOTE_AFTER_START")
    if (not features or not prediction or features > prediction or
            not row.get("feature_snapshot_id") or row.get("feature_replay_verified") is not True):
        blockers.append("FEATURE_ASOF_UNAVAILABLE")
    if (not row.get("model_id") or not row.get("model_artifact_hash") or
            not row.get("runtime_hash") or not model_available or not train_cutoff or
            not latest_training_input or latest_training_input > train_cutoff or
            row.get("model_replay_verified") is not True or
            not prediction or model_available > prediction or train_cutoff > prediction):
        blockers.append("MODEL_REPLAY_NOT_REPRODUCIBLE")
    if (not result or not prediction or result <= prediction or (start and result < start) or
            row.get("result_verified") is not True):
        blockers.append("RESULT_LEAKAGE_RISK")
    return {"eligible": not blockers, "classification": "HISTORICAL_RESEARCH_ONLY" if blockers else
            "REPLAYABLE_HISTORICAL", "role": None, "prospective_count": 0,
            "blockers": sorted(set(blockers))}


def _plan_spec(sport: str, market: str, *, source_commit: str,
               validation_start: datetime) -> dict:
    base = copy.deepcopy(v1.plan_specs(source_commit=source_commit)[v1.EXPECTED_MARKETS.index((sport, market))])
    n = calculated_minimum()["effective_decided_events"]
    base.update(validation_plan_id=f"prospective-{sport.lower()}-{market.lower()}-football-v2",
                version=2, supersedes_plan_id=v1._plan_id(sport, market),
                plan_policy_version=POLICY_VERSION, model_id=None, calibration_id=None,
                training_cutoff=(validation_start - timedelta(seconds=1)).isoformat(),
                validation_start=validation_start.isoformat(),
                validation_end="2027-04-01T00:00:00+00:00",
                holdout_start="2027-04-01T00:00:00+00:00",
                holdout_end="2028-04-01T00:00:00+00:00",
                minimum_independent_sample=n, minimum_effective_sample=n)
    base["clv_policy"]["v2_mode"] = "CERTIFIED_CLOSE_REQUIRED"
    base["clv_policy"]["rationale"] = "Retain V1's 80% certified comparable-close requirement; near-start quotes are diagnostic only."
    base["value_roi_policy"]["validation_gate"] = False  # Proper scores and uncertainty decide validation.
    base["football_v2_methodology"] = {
        "status": "UNBOUND_MODEL_AND_CALIBRATION", "scopes_are_independent": True,
        "historical_roles": ["HISTORICAL_TRAIN", "HISTORICAL_MODEL_SELECTION", "HISTORICAL_CALIBRATION"],
        "prospective_roles": ["PROSPECTIVE_CONFIRMATION", "PROSPECTIVE_HOLDOUT"],
        "historical_replay_gate": "football_validation_v2.historical_replay_eligible",
        "historical_cutoff": base["training_cutoff"],
        "cohort_boundary": "event and prediction strictly after freeze and validation_start",
        "decision_method": "FIXED_SINGLE_CHECKPOINT", "peek_policy": "NO_EARLY_PASS",
        "confidence": 1 - ALPHA, "familywise_tests": METRIC_COHORT_TESTS,
        "probability_floor_for_log_loss": LOG_PROBABILITY_FLOOR,
        "calibration_bins": CALIBRATION_BINS, "precision_budget": PRECISION,
        "calculated_minimum": calculated_minimum(),
        "independence_unit": "one first provider-bound event per exact sport and market",
        "required_price_coverage": 1.0, "required_certified_close_coverage": 0.8,
        "full_slate_coverage_status": "UNVERIFIED_REQUIRES_NEW_PLAN_VERSION",
        "push_void": "refund; excluded from decided scores; unresolved blocks",
        "conservative_probability_rule": "registered exact-scope calibration interval lower bound; missing interval blocks",
        "provisional": "confirmation meets every upper/lower bound after window closes and independent review",
        "standard": "untouched holdout separately meets every bound after holdout window closes and independent review",
        "premium": "unavailable until new prospectively frozen policy",
        "automatic_activation": False, "recommended_stake": 0,
    }
    base["policy_source_hash"] = _hash({k: v for k, v in base.items() if k not in {"policy_source_hash", "source_commit"}})
    return base


def freeze_plans(path: str | Path | None, *, source_commit: str) -> list[dict]:
    """Install four immutable V2 policy plans after V1; never rewrite a version."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a Git SHA")
    current = evidence.read_records(path, "prospective_validation_plan")
    by_scope = {(r["sport"], r["market_family"], r["version"]): r for r in current}
    now = evidence._clock()
    start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    frozen = []
    for sport, market in SCOPES:
        initial = by_scope.get((sport, market, 1))
        if initial is None:
            raise evidence.EvidenceConflict("V1 plan must be frozen before V2")
        existing = by_scope.get((sport, market, 2))
        if existing:
            payload = json.loads(existing["payload"])
            if (existing["validation_plan_id"] != f"prospective-{sport.lower()}-{market.lower()}-football-v2" or
                    payload.get("plan_policy_version") != POLICY_VERSION or
                    payload.get("artifact_hash") != existing["artifact_hash"] or
                    _hash({k: v for k, v in payload.items() if k != "artifact_hash"}) != existing["artifact_hash"]):
                raise evidence.EvidenceConflict("V2 football plan conflicts with existing version")
            frozen.append(existing)
            continue
        if start >= datetime(2027, 4, 1, tzinfo=timezone.utc):
            raise ValueError("2026 football confirmation window has passed")
        spec = _plan_spec(sport, market, source_commit=source_commit, validation_start=start)
        evidence.freeze_validation_plan(path, spec)
        frozen.append(evidence.load_record(path, "prospective_validation_plan", spec["validation_plan_id"]))
    return [{"sport": r["sport"], "market_family": r["market_family"],
             "validation_plan_id": r["validation_plan_id"], "version": r["version"],
             "frozen_at": r["frozen_at"], "validation_start": r["validation_start"],
             "artifact_hash": r["artifact_hash"], "production_eligible": False,
             "recommended_stake": 0} for r in frozen]


def evaluate(plan: dict, method: dict, confirmation: dict, holdout: dict,
             cutoff: datetime) -> dict:
    """V2 fixed-checkpoint decision, independent of V1's 200+200 evaluator."""
    blockers = []
    if not plan["model_id"]: blockers.append("MISSING_MODEL")
    if not plan["calibration_id"]: blockers.append("MISSING_CALIBRATION")
    if method["football_v2_methodology"].get("full_slate_coverage_status") != "VERIFIED":
        blockers.append("FULL_SLATE_COVERAGE_UNVERIFIED")
    if cutoff < _time(plan["validation_end"]): blockers.append("CONFIRMATION_WINDOW_OPEN")
    thresholds = method["probability_thresholds"]
    n_min = plan["minimum_effective_sample"]
    cohort_blockers = {}
    for label, cohort in (("CONFIRMATION", confirmation), ("HOLDOUT", holdout)):
        current = []
        if label == "HOLDOUT" and cutoff < _time(plan["holdout_end"]):
            current.append("HOLDOUT_WINDOW_OPEN")
        n = cohort["effective_observations"]
        if n < n_min: current.append(f"{label}_INSUFFICIENT_EFFECTIVE_EVENTS")
        if cohort["unique_events"] < plan["minimum_independent_sample"]:
            current.append(f"{label}_INSUFFICIENT_INDEPENDENT_EVENTS")
        for key in ("missing_provenance_count", "missing_stable_identity_count",
                    "unsupported_probability_semantics_count", "wrong_model_or_calibration_count",
                    "missing_calibrated_uncertainty_count"):
            if cohort.get(key): current.append(f"{label}_{key.upper()}")
        if cohort["outcomes"]["PENDING"] or cohort["outcomes"]["NEEDS_REVIEW"]:
            current.append(f"{label}_UNRESOLVED_OUTCOMES")
        if cohort["verified_entry_coverage"] < 1.0:
            current.append(f"{label}_EXACT_PRICE_INCOMPLETE")
        if cohort["comparable_close_coverage"] < 0.8:
            current.append(f"{label}_CERTIFIED_CLOSE_INCOMPLETE")
        for metric, threshold in (("brier", "max_brier"), ("log_loss", "max_log_loss"),
                                  ("calibration_error", "max_calibration_error")):
            if n == 0 or cohort[metric] is None or cohort[metric] + uncertainty_radius(n, metric=metric) > thresholds[threshold]:
                current.append(f"{label}_{metric.upper()}_UPPER_BOUND_NOT_MET")
        raw_n = cohort["unique_events"]
        if raw_n == 0 or cohort["coverage"] - uncertainty_radius(raw_n, metric="coverage") < thresholds["min_coverage"]:
            current.append(f"{label}_COVERAGE_LOWER_BOUND_NOT_MET")
        cohort_blockers[label] = current
    blockers.extend(cohort_blockers["CONFIRMATION"])
    standard_blockers = blockers + cohort_blockers["HOLDOUT"]
    return {"blockers": blockers,
            "status": "UNVALIDATED" if blockers else "VALIDATION_PASSED",
            "tier": "UNVALIDATED" if blockers else "PROVISIONAL_VALIDATED",
            "standard_status": "BLOCKED" if standard_blockers else "ELIGIBLE_FOR_SEPARATE_STANDARD_REVIEW",
            "standard_blockers": standard_blockers,
            "metric_uncertainty": {label: {metric: uncertainty_radius(cohort["effective_observations"], metric=metric)
                                   if cohort["effective_observations"] else None for metric in PRECISION}
                                   for label, cohort in (("confirmation", confirmation), ("holdout", holdout))}}


def timeline(path: str | Path | None) -> list[dict]:
    """Read-only V2 progress; unknown provider coverage stays unknown."""
    rows = []
    for sport, market in SCOPES:
        plan_id = f"prospective-{sport.lower()}-{market.lower()}-football-v2"
        plan = evidence.load_record(path, "prospective_validation_plan", plan_id)
        if not plan:
            rows.append({"sport": sport, "market_family": market, "plan_version": None,
                         "status": "UNVALIDATED", "next_blocker": "V2_PLAN_NOT_FROZEN"})
            continue
        report = evidence.evaluate_validation_plan(path, plan_id)
        cohort = report["validation"]
        required = plan["minimum_independent_sample"]
        rows.append({"sport": sport, "market_family": market, "plan_version": 2,
                     "validation_plan_id": plan_id, "artifact_hash": plan["artifact_hash"],
                     "frozen_at": plan["frozen_at"], "prospective_start": plan["validation_start"],
                     "historical_replay_eligible": None, "historical_train_count": None,
                     "historical_calibration_count": None,
                     "prospective_confirmation_count": cohort["unique_events"],
                     "effective_count": cohort["effective_observations"], "required_count": required,
                     "remaining_count": max(0, required - cohort["effective_observations"]),
                     "metric_uncertainty": report.get("metric_uncertainty", {}),
                     "next_checkpoint": plan["validation_end"],
                     "remaining_season_capacity": None, "full_season_capacity_upper_bound":
                     272 if sport == "NFL" else None, "expected_games_per_week": None,
                     "estimated_weeks_to_checkpoint": None,
                     "status": report.get("tier", "UNVALIDATED"),
                     "next_blocker": report["blockers"][0] if report["blockers"] else "INDEPENDENT_REVIEW_REQUIRED",
                     "production_eligible": False, "recommended_stake": 0,
                     "coverage_status": "FULL_SLATE_NOT_VERIFIED"})
    return rows


def evidence_inventory(directory: str | Path) -> list[dict]:
    """Read existing local sources without treating absence as a verified remote zero."""
    root = Path(directory)
    rows = []
    for sport, market in SCOPES:
        source_path = root / ("nfl-market.sqlite3" if sport == "NFL" else "ncaaf-prospective.sqlite3")
        if source_path.is_file():
            from app_core.prospective_source_view import source_evidence
            source = [r for r in source_evidence(sport, source_path)
                      if r.get("market_family") == market]
        else:
            source = []
        events = {str(r.get("provider_event_id") or r.get("game_id")) for r in source
                  if r.get("provider_event_id") or r.get("game_id")}
        replay = [historical_replay_eligible(r) for r in source]
        blockers = sorted({b for verdict in replay for b in verdict["blockers"]})
        rows.append({"sport": sport, "market_family": market,
                     "source_status": "LOCAL_PRESENT_REMOTE_NOT_VERIFIED" if source_path.is_file()
                                      else "LOCAL_ABSENT_REMOTE_UNKNOWN",
                     "raw_source_records": len({str(r.get("source_record_id")) for r in source
                                                if r.get("source_record_id")}),
                     "research_quote_rows": len(source), "unique_source_events": len(events),
                     "pregame_predictions": sum(bool(_time(r.get("prediction_timestamp")) and
                         _time(r.get("scheduled_start")) and
                         _time(r.get("prediction_timestamp")) < _time(r.get("scheduled_start")))
                         for r in source),
                     "verified_exact_quote_rows": sum(r.get("quote_verified") is True for r in source),
                     "settled_event_count": len({str(r.get("provider_event_id") or r.get("game_id"))
                         for r in source if r.get("result_outcome") in {"WIN", "LOSS", "PUSH", "VOID"}}),
                     "result_availability_known": sum(bool(_time(r.get("result_available_at"))) for r in source),
                     "model_ids": sorted({str(r["model_id"]) for r in source if r.get("model_id")}),
                     "model_training_cutoffs": sorted({str(r["model_trained_through"]) for r in source
                                                        if r.get("model_trained_through")}),
                     "model_availability_timestamps": sorted({str(r["model_available_at"]) for r in source
                                                               if r.get("model_available_at")}),
                     "calibration_ids": sorted({str(r["calibration_id"]) for r in source if r.get("calibration_id")}),
                     "calibration_availability_timestamps": sorted({str(r["calibration_available_at"])
                         for r in source if r.get("calibration_available_at")}),
                     "feature_asof_known": sum(bool(_time(r.get("feature_frozen_at"))) for r in source),
                     "historical_seasons": sorted({_time(r["scheduled_start"]).year for r in source
                         if _time(r.get("scheduled_start")) and _time(r["scheduled_start"]).year < 2026}),
                     "source_2026_rows": sum(bool(_time(r.get("scheduled_start")) and
                         _time(r["scheduled_start"]).year == 2026) for r in source),
                     "authentic_v2_prospective_count": None,
                     "historical_replay_eligible": sum(v["eligible"] for v in replay),
                     "historical_train_count": 0, "historical_calibration_count": 0,
                     "replay_blockers": blockers,
                     "coverage_audit": {
                         "scheduled": None, "discovered": len(events),
                         "exact_odds_events": len({str(r.get("provider_event_id") or r.get("game_id"))
                             for r in source if r.get("quote_verified") is True}),
                         "predicted_events": len({str(r.get("provider_event_id") or r.get("game_id"))
                             for r in source if _time(r.get("prediction_timestamp")) and
                             _time(r.get("scheduled_start")) and
                             _time(r.get("prediction_timestamp")) < _time(r.get("scheduled_start"))}),
                         "excluded_reasons": blockers,
                         "full_slate_verified": False,
                         "population": "NFL_ALL_REGULAR_SEASON_GAMES" if sport == "NFL" else
                                       "CFBD_FBS_CLASSIFICATION_ONLY_FCS_NOT_COVERED",
                         "conference_coverage": None, "bye_impact": None,
                         "remaining_season_capacity": None,
                         "full_season_capacity_upper_bound": 272 if sport == "NFL" else None}})
    return rows
