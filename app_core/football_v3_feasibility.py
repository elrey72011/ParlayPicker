"""Authenticated, read-only football V3 feasibility audit.

This module never creates a V3 plan. Replay and statistical feasibility must be
established before a future, independently reviewed plan can be proposed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, variance

from app_core import football_validation_v2 as v2
from app_core import prospective_evidence as evidence

SCOPES = v2.SCOPES
CONCLUSION = "NO_DEFENSIBLE_SHORT_SEASON_V3"


def _time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _event_id(row):
    value = row.get("provider_event_id") or row.get("game_id")
    return str(value) if value is not None else None


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class ReadOnlyEvidenceStore:
    """Permit verified Drive reads; a pending upload is an audit failure."""
    def __init__(self, source):
        self.source = source

    def read_objects(self, *, Prefix):
        return self.source.read_objects(Prefix=Prefix)

    def get_object(self, **kwargs):
        return self.source.get_object(**kwargs)

    def put_object(self, **kwargs):
        raise RuntimeError("football_v3_audit_remote_write_prohibited")


def _scope_rows(canonical, source, sport, market):
    src = [r for r in source if r.get("sport") == sport and r.get("market_family") == market]
    relevant = {table: evidence.read_records(canonical, table, sport=sport, market_family=market)
                for table in ("prospective_quote", "prospective_close", "prospective_result",
                              "prospective_model", "prospective_calibration", "prospective_prediction",
                              "prospective_validation_plan")}
    return src, relevant


def _safe_models(rows):
    return [dict(model_id=r["model_id"], version=r.get("model_version"),
                 artifact_hash=r.get("artifact_hash"), training_cutoff=r.get("training_cutoff"),
                 available_at=r.get("available_at"), feature_version=r.get("feature_version"),
                 training_observation_count=r.get("training_observation_count"),
                 independent_event_count=r.get("independent_event_count"),
                 sport=r["sport"], market_family=r["market_family"])
            for r in rows]


def _safe_calibrations(rows):
    return [dict(calibration_id=r["calibration_id"], model_id=r.get("model_id"),
                 method=r.get("method"), version=r.get("calibration_version"),
                 artifact_hash=r.get("artifact_hash"), fit_start=r.get("fit_start"),
                 fit_cutoff=r.get("fit_end"),
                 available_at=r.get("available_at"), sport=r["sport"],
                 market_family=r["market_family"])
            for r in rows]


def _bound_models(models, calibrations):
    """Identify chronologically valid exact-scope artifact pairs, never infer one."""
    pairs = []
    for model in models:
        training, model_at = _time(model["training_cutoff"]), _time(model["available_at"])
        if not (model["artifact_hash"] and model["feature_version"] and training and
                model_at and training <= model_at):
            continue
        for calibration in calibrations:
            fit, calibration_at = _time(calibration["fit_cutoff"]), _time(calibration["available_at"])
            if (calibration["model_id"] == model["model_id"] and
                    calibration["artifact_hash"] and fit and calibration_at and
                    model_at <= fit <= calibration_at):
                pairs.append({"model_id": model["model_id"],
                              "calibration_id": calibration["calibration_id"]})
    return pairs


def _legal_replay_blockers(row):
    """Additional V3 statistical requirements after the unchanged V2 as-of gate."""
    blockers = []
    probability = row.get("mean_probability")
    if (isinstance(probability, bool) or not isinstance(probability, (int, float)) or
            not math.isfinite(probability) or not 0 < probability < 1):
        blockers.append("REPLAY_PROBABILITY_UNAVAILABLE")
    if row.get("result_outcome") not in {"WIN", "LOSS", "PUSH", "VOID"}:
        blockers.append("REPLAY_SETTLEMENT_UNAVAILABLE")
    if not row.get("evidence_hash") or not row.get("source_record_id"):
        blockers.append("REPLAY_SOURCE_HASH_UNAVAILABLE")
    return blockers


def _replay_manifest(rows):
    """Bind one earliest as-of offer per event without assigning prospective roles."""
    first = {}
    for row in rows:
        event = _event_id(row)
        at = _time(row.get("prediction_generated_at") or row.get("prediction_timestamp"))
        if event and at and (event not in first or at < _time(
                first[event].get("prediction_generated_at") or first[event].get("prediction_timestamp"))):
            first[event] = row
    fields = ("provider_event_id", "game_id", "prediction_timestamp", "scheduled_start",
              "line", "decimal_odds", "sportsbook", "quote_timestamp", "feature_snapshot_id",
              "model_id", "model_artifact_hash", "runtime_hash", "model_trained_through",
              "mean_probability", "result_outcome", "result_available_at", "evidence_hash",
              "source_record_id")
    return [dict(role="UNASSIGNED_HISTORICAL_REPLAY", **{k: row.get(k) for k in fields})
            for _, row in sorted(first.items())]


def cluster_study(rows, *, seed=20260924, draws=512):
    """Deterministic season/week block bootstrap on legal, independent replay.

    This is exploratory only. It cannot select a V3 checkpoint without a
    sufficient multi-season replay dataset and frozen external review.
    """
    if draws < 100:
        raise ValueError("at least 100 resamples required")
    first = {}
    for row in rows:
        event = _event_id(row)
        at = _time(row.get("prediction_generated_at") or row.get("prediction_timestamp"))
        if event and at and (event not in first or at < _time(
                first[event].get("prediction_generated_at") or first[event].get("prediction_timestamp"))):
            first[event] = row
    scored = []
    for row in first.values():
        start = _time(row.get("scheduled_start"))
        p = row.get("mean_probability")
        outcome = row.get("result_outcome")
        if not start or outcome not in {"WIN", "LOSS"}:
            continue
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p < 1:
            continue
        y = int(outcome == "WIN")
        iso = start.isocalendar()
        scored.append((f"{iso.year}-W{iso.week:02d}", (p-y)**2,
                       -(y*math.log(max(p, .01))+(1-y)*math.log(max(1-p, .01))),
                       row.get("home_team_id"), row.get("away_team_id")))
    clusters = defaultdict(list)
    for row in scored:
        clusters[row[0]].append(row)
    settled = sum(r.get("result_outcome") in {"WIN", "LOSS", "PUSH", "VOID"}
                  for r in first.values())
    push_void = sum(r.get("result_outcome") in {"PUSH", "VOID"} for r in first.values())
    if len(scored) < 30 or len(clusters) < 8:
        return {"status": "INSUFFICIENT_INDEPENDENT_REPLAY_CLUSTERS",
                "decided_events": len(scored), "season_week_clusters": len(clusters),
                "settled_events": settled, "push_void_events": push_void,
                "push_void_rate": push_void / settled if settled else None,
                "brier_variance": None, "log_loss_variance": None,
                "calibration_uncertainty": None, "within_week_dependence": None,
                "team_repeat_dependence": None, "effective_sample": None}
    rng = random.Random(seed)
    keys = sorted(clusters)
    n = len(scored)
    brier = [x[1] for x in scored]
    logloss = [x[2] for x in scored]
    samples = {"brier": [], "log_loss": []}
    for _ in range(draws):
        taken = [x for key in (rng.choice(keys) for _ in keys) for x in clusters[key]]
        samples["brier"].append(mean(x[1] for x in taken))
        samples["log_loss"].append(mean(x[2] for x in taken))
    def metric(name, scores):
        iid = variance(scores)
        means = sorted(samples[name])
        clustered_mean_var = variance(means)
        effective = min(n, iid / clustered_mean_var) if clustered_mean_var > 0 else None
        return {"mean": mean(scores), "event_variance": iid,
                "clustered_mean_variance": clustered_mean_var,
                "mean_interval_95": [means[int(.025*draws)], means[min(draws-1, int(.975*draws))]],
                "effective_events_estimate": effective}
    team_counts = Counter(team for x in scored for team in x[3:] if team is not None)
    return {"status": "EXPLORATORY_CLUSTER_ESTIMATES_ONLY", "decided_events": n,
            "season_week_clusters": len(clusters),
            "settled_events": settled, "push_void_events": push_void,
            "push_void_rate": push_void / settled if settled else None,
            "teams_repeated": sum(count > 1 for count in team_counts.values()),
            "brier": metric("brier", brier), "log_loss": metric("log_loss", logloss),
            "calibration_uncertainty": None,
            "within_week_dependence": "PARTIALLY_REFLECTED_IN_WEEK_BLOCKS",
            "team_repeat_dependence": "NOT_ESTIMATED",
            "seed": seed, "resamples": draws,
            "limitation": "Week block bootstrap does not alone resolve team-repeat or season dependence."}


def _methods(replay_count):
    unavailable = "NOT_ESTIMABLE_FROM_LEGAL_REPLAY" if replay_count == 0 else "NOT_YET_JUSTIFIED"
    return [
        {"method": "V2_DISTRIBUTION_FREE_FIXED", "status": "BENCHMARK_NOT_SHORT_SEASON_FEASIBLE",
         "initial_minimum": v2.calculated_minimum()["effective_decided_events"],
         "expected_minimum": None, "worst_case_minimum": None,
         "assumptions": "Independent decided events; frozen one-time checkpoint",
         "error_control": "95% simultaneous fixed-checkpoint distribution-free bounds",
         "calibration_sensitivity": "Ten-bin ECE allowance dominates sample requirement",
         "failure_mode": "NFL full-season capacity below calculated minimum",
         "complexity": "LOW", "auditability": "HIGH"},
        {"method": "CLUSTER_BOOTSTRAP_FIXED", "status": unavailable,
         "initial_minimum": None, "expected_minimum": None, "worst_case_minimum": None,
         "assumptions": "Many exchangeable independent season/week clusters; fixed resampling rule",
         "error_control": "Requires prospectively fixed cluster bootstrap and sufficiently many seasons/clusters",
         "calibration_sensitivity": "Calibration bins and cluster imbalance unresolved",
         "failure_mode": "Sparse clusters, nonstationarity, team-repeat dependence, incomplete calibration",
         "complexity": "MEDIUM", "auditability": "MEDIUM"},
        {"method": "SEQUENTIAL_CONFIDENCE", "status": unavailable,
         "initial_minimum": None, "expected_minimum": None, "worst_case_minimum": None,
         "assumptions": "Anytime-valid bounds under proven dependence model; frozen checkpoint rule",
         "error_control": "Requires anytime-valid boundaries and frozen checkpoints/stopping rules",
         "calibration_sensitivity": "Sequential ECE bound and changing bins unresolved",
         "failure_mode": "Optional stopping, dependence, unstable calibration bins",
         "complexity": "HIGH", "auditability": "MEDIUM"},
        {"method": "BAYESIAN_CALIBRATION", "status": "NO_INDEPENDENTLY_JUSTIFIED_PRIOR",
         "initial_minimum": None, "expected_minimum": None, "worst_case_minimum": None,
         "assumptions": "Independent defensible prior and frozen posterior decision rule",
         "error_control": "Prior and posterior decision rule would require independent pre-freeze justification",
         "calibration_sensitivity": "High and unquantified without legal replay and prior sensitivity study",
         "failure_mode": "Prior sensitivity and hidden prior tuning",
         "complexity": "HIGH", "auditability": "LOW"},
    ]


def build_report(root, *, authenticated=False, restore_counts=None, as_of=None):
    """Read-only, bounded audit. Local counts are never called authenticated."""
    root = Path(root)
    canonical = root / "prospective-evidence.sqlite3"
    stamp = as_of or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    source = []
    for sport, filename in (("NFL", "nfl-market.sqlite3"),
                            ("NCAAF", "ncaaf-prospective.sqlite3")):
        path = root / filename
        if path.is_file():
            from app_core.prospective_source_view import source_evidence
            source.extend(source_evidence(sport, path))
    market_rows = []
    for sport, market in SCOPES:
        src, canonical_rows = _scope_rows(canonical, source, sport, market)
        replay = [(r, v2.historical_replay_eligible(r)) for r in src]
        v2_eligible = [r for r, result in replay if result["eligible"]]
        legal = [r for r in v2_eligible if not _legal_replay_blockers(r)]
        blockers = Counter(code for _, result in replay for code in result["blockers"])
        blockers.update(code for r in v2_eligible for code in _legal_replay_blockers(r))
        replay_manifest = _replay_manifest(legal)
        src_events = {_event_id(r) for r in src if _event_id(r)}
        priced_events = {_event_id(r) for r in src if _event_id(r) and
                         r.get("quote_verified") is True and
                         _time(r.get("quote_timestamp")) and _time(r.get("scheduled_start")) and
                         _time(r["quote_timestamp"]) < _time(r["scheduled_start"])}
        canonical_models = _safe_models(canonical_rows["prospective_model"])
        canonical_calibrations = _safe_calibrations(canonical_rows["prospective_calibration"])
        bound_pairs = _bound_models(canonical_models, canonical_calibrations)
        canonical_predictions = canonical_rows["prospective_prediction"]
        plan_rows = [{"id": p["validation_plan_id"], "version": p["version"],
                      "artifact_hash": p["artifact_hash"], "frozen_at": p["frozen_at"]}
                     for p in canonical_rows["prospective_validation_plan"]]
        primary = ("AUTHENTICATED_RESTORE_REQUIRED" if not authenticated else
                   "MISSING_EXACT_SCOPE_MODEL_CALIBRATION" if not bound_pairs else
                   "NO_LEGAL_REPLAY" if not legal else
                   "FULL_SLATE_COVERAGE_UNVERIFIED")
        market_rows.append({"sport": sport, "market_family": market,
                            "evidence_status": "AUTHENTICATED_RESTORED" if authenticated else
                                               "LOCAL_ONLY_REMOTE_UNKNOWN",
                            "source_quote_rows": len(src), "unique_source_games": len(src_events),
                            "source_research_model_ids": sorted({str(r["model_id"]) for r in src
                                if r.get("model_id")}),
                            "source_runtime_hashes": sorted({str(r["runtime_hash"]) for r in src
                                if r.get("runtime_hash")}),
                            "historical_seasons": sorted({_time(r["scheduled_start"]).year for r in src
                                if _time(r.get("scheduled_start")) and
                                _time(r["scheduled_start"]).year < stamp.year}),
                            "verified_pregame_price_games": len(priced_events),
                            "price_coverage_of_captured_games": len(priced_events) / len(src_events)
                                if src_events else None,
                            "feature_snapshot_rows": sum(bool(r.get("feature_snapshot_id")) for r in src),
                            "replay_verified_feature_rows": sum(r.get("feature_replay_verified") is True for r in src),
                            "verified_result_availability_rows": sum(bool(_time(r.get("result_available_at")))
                                and r.get("result_verified") is True for r in src),
                            "source_close_candidates": sum(r.get("close_status") not in
                                {None, "NO_VALID_CLOSE_QUOTES", "NO_COMPARABLE_CLOSING_PROXY"}
                                for r in src),
                            "canonical_certified_closes": sum(r.get("close_verified") == 1
                                for r in canonical_rows["prospective_close"]),
                            "canonical_verified_quotes": sum(r.get("quote_verified") == 1
                                for r in canonical_rows["prospective_quote"]),
                            "canonical_predictions": len(canonical_predictions),
                            "prospective_count": None,
                            "prospective_count_status": "UNVERIFIED_COHORT_BOUNDARY_AND_AUTHORITY",
                            "canonical_2026_predictions": sum(bool(_time(r.get("prediction_timestamp")) and
                                _time(r["prediction_timestamp"]).year == 2026) for r in canonical_predictions),
                            "v2_gate_eligible_games": len({_event_id(r) for r in v2_eligible}),
                            "replay_eligible_games": len(replay_manifest),
                            "legal_replay_manifest": replay_manifest,
                            "legal_replay_manifest_sha256": _sha(replay_manifest),
                            "replay_blockers": dict(sorted(blockers.items())),
                            "model_inventory": canonical_models,
                            "calibration_inventory": canonical_calibrations,
                            "bound_model_calibration_pairs": bound_pairs,
                            "frozen_plan_inventory": plan_rows,
                            "research_plan_version": max((p["version"] for p in plan_rows), default=None),
                            "model_calibration_readiness": "BOUND_ARTIFACT_PAIR_PRESENT" if bound_pairs
                                else "MISSING_EXACT_SCOPE_MODEL_CALIBRATION",
                            "empirical_study": cluster_study(legal),
                            "designs": _methods(len(legal)),
                            "v2_required_effective_events": v2.calculated_minimum()["effective_decided_events"],
                            "v3_required_confirmation": None,
                            "v3_required_additional_evidence": None,
                            "v3_effective_sample": None,
                            "expected_eligible_games_per_week": None,
                            "remaining_2026_capacity": None,
                            "earliest_v3_checkpoint": None,
                            "sensitivity_range": None,
                            "full_slate": {"scheduled": None, "discovered": len(src_events),
                                "priced": len(priced_events), "predicted": len({_event_id(r) for r in src
                                    if _event_id(r) and r.get("prediction_timestamp")}),
                                "excluded": None, "exclusion_reasons": dict(sorted(blockers.items())),
                                "status": "UNVERIFIED",
                                "population": "NFL_REGULAR_SEASON" if sport == "NFL" else
                                              "CFBD_FBS_ONLY_FCS_EXCLUDED_UNTIL_REVIEW",
                                "conference_coverage": None, "bye_impact": None,
                                "full_season_upper_bound": 272 if sport == "NFL" else None},
                            "primary_blocker": primary,
                            "production_eligible": False, "recommended_stake": 0})
    immutable = sorted((r["sport"], r["market_family"], r["version"], r["artifact_hash"])
                       for r in evidence.read_records(canonical, "prospective_validation_plan")
                       if r["sport"] in {"NFL", "NCAAF"} and r["version"] <= 2)
    report = {"schema": "football-v3-feasibility-audit-v1",
              "generated_at": stamp.astimezone(timezone.utc).isoformat(),
              "authenticated_remote_verified": bool(authenticated),
              "restore_counts": restore_counts if authenticated else None,
              "result": CONCLUSION,
              "method_selection_uses_prospective_outcomes": False,
              "v3_plans_created": 0,
              "v1_v2_plan_artifact_hashes": immutable,
              "markets": market_rows,
              "no_wager_or_activation": True}
    report["audit_sha256"] = _sha(report)
    return report


def restore_and_audit(root, client, folder):
    """Restore all canonical and football source objects with remote writes denied."""
    from app_core import prospective_remote, nfl_market_store, ncaaf_prospective_store
    readonly = ReadOnlyEvidenceStore(client)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    counts["canonical"] = prospective_remote.sync(root / "prospective-evidence.sqlite3",
                                                   readonly, folder, {})
    counts["nfl"] = nfl_market_store.sync(root / "nfl-market.sqlite3",
                                           client=readonly, folder=folder, session={})
    counts["ncaaf"] = ncaaf_prospective_store.sync(root / "ncaaf-prospective.sqlite3",
                                                     client=readonly, folder=folder, session={})
    if any(item.get("new_records_verified") or item.get("new_records_verified", 0)
           for item in counts.values()):
        raise RuntimeError("football_v3_audit_unexpected_remote_write")
    return build_report(root, authenticated=True, restore_counts=counts)
