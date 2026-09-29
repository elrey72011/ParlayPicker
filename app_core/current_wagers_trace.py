"""Private candidate-to-output diagnostics for one saved board run.

This module consumes already-produced candidate evidence.  It performs no
provider request, creates no authority, and is deliberately excluded from the
public package.  Unknown or short-circuited stages remain explicit.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any, Mapping

import pandas as pd

from app_core.quote_freshness import package_age_minutes


VERSION = "current-wagers-private-trace-v1"
STAGES = (
    "parsed_candidate", "identity_market", "pregame_fresh_quote",
    "model_calibration", "price_gate", "authority_review", "finalist_selection",
    "packaged_output", "release_preflight", "currently_usable_wager",
)


def _text(row: Mapping, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).strip():
            return str(value).strip()
    return ""


def _number(row: Mapping, *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, bool):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _truth(row: Mapping, *names: str) -> bool | None:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().casefold()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or parsed.tzinfo is None:
        return None
    return parsed.tz_convert("UTC").to_pydatetime()


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    return str(value)


def _line(row: Mapping) -> float | None:
    market = _text(row, "market_type")
    return _number(
        row, "line", "market_line_used", "selected_line",
        "total_line" if market.startswith("total") else "spread_line",
    )


def _candidate_identity(row: Mapping) -> tuple[str, dict]:
    facts = {
        "run_id": _text(row, "export_run_id", "run_id"),
        "source_candidate_id": _text(row, "candidate_id"),
        "event_id": _text(row, "canonical_event_id", "matchup_id", "game_id"),
        "sport": _text(row, "exact_sport", "sport", "league", "League").upper(),
        "market": _text(row, "market_type"),
        "selection": _text(row, "selection", "best_pick", "display_pick", "Pick"),
        "line": _line(row),
        "sportsbook": _text(row, "quote_bookmaker", "book", "opposing_odds_source", "odds_source"),
        "quote_id": _text(row, "quote_id", "prospective_quote_id"),
        "odds_american": _number(row, "odds_american", "american_odds", "odds"),
        "quote_timestamp": _text(row, "quote_observed_at", "odds_recorded_at", "quote_time", "quote_timestamp"),
    }
    raw = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(raw.encode("utf-8")).hexdigest(), facts


def _stage(status: str, reason: str, *, executed: bool = True) -> dict:
    return {"status": status, "reason": reason, "executed": executed}


def _model_stage(row: Mapping) -> dict:
    explicit = _truth(row, "production_model_eligible", "model_eligible")
    calibration = _text(row, "Calibration_Consumer_Status", "calibration_consumer_status")
    predictor = _text(row, "source_predictor_version", "predictor_version", "model_version")
    if explicit is False:
        return _stage("BLOCK", "MODEL_NOT_PRODUCTION_ELIGIBLE")
    rejected = {"CALIBRATION_REJECTED", "PROBABILITY_SEMANTICS_INCOMPATIBLE",
                "PUSH_SUPPORT_MISSING_OR_INVALID", "CONSUMER_CONTEXT_MISSING",
                "CALIBRATION_OR_TRANSFORM_NOT_AUTHORIZED"}
    if calibration in rejected:
        return _stage("BLOCK", calibration)
    if explicit is True and predictor and calibration in {
        "PRODUCTION_CALIBRATION_APPLIED", "CONTROLLED_EMPIRICAL_OVERRIDE",
    }:
        return _stage("PASS", "MODEL_AND_CALIBRATION_SUPPORTED")
    return _stage("UNKNOWN", "MODEL_OR_CALIBRATION_EXECUTION_NOT_RECORDED", executed=False)


def _price_stage(row: Mapping) -> dict:
    passed = _truth(row, "Production_Gate_Pass", "production_gate_pass")
    reason = _text(row, "Production_Gate_Reason", "production_gate_reason")
    contract = row.get("wager_contract")
    if passed is True:
        return _stage("PASS", reason or "RECORDED_PRICE_GATE_PASS")
    if passed is False:
        return _stage("BLOCK", reason or "RECORDED_PRICE_GATE_BLOCK")
    if isinstance(contract, dict) and contract.get("production_eligible") is True:
        return _stage("PASS", "FROZEN_WAGER_CONTRACT_PRICE_PASS")
    return _stage("NOT_EVALUATED", "PRICE_GATE_RESULT_NOT_RECORDED", executed=False)


def _authority_stage(row: Mapping) -> dict:
    strict = row.get("wager_contract")
    trial = row.get("controlled_trial_contract")
    if isinstance(strict, dict):
        if strict.get("production_eligible") is True and (_number(strict, "production_bet_amount") or 0) > 0:
            return _stage("PASS", "FROZEN_STRICT_AUTHORITY_AND_ALLOCATION")
        return _stage("BLOCK", "FROZEN_STRICT_AUTHORITY_BLOCKED")
    if isinstance(trial, dict):
        if trial.get("trial_eligible") is True and (_number(trial, "recommended_bet_amount") or 0) > 0:
            return _stage("PASS", "FROZEN_TRIAL_AUTHORITY_AND_ALLOCATION")
        return _stage("BLOCK", "FROZEN_TRIAL_AUTHORITY_BLOCKED")
    eligible = _truth(row, "production_eligible", "wager_approved")
    stake = _number(row, "production_bet_amount", "Play_Stake", "Kelly_Bet_Size") or 0
    if eligible is True and stake > 0:
        return _stage("PASS", "RECORDED_AUTHORITY_AND_ALLOCATION")
    if eligible is False or stake == 0 and any(name in row for name in (
            "production_eligible", "wager_approved", "production_bet_amount", "Play_Stake")):
        return _stage("BLOCK", _text(row, "production_gate_reason", "Production_Gate_Reason") or
                      "AUTHORITY_OR_ALLOCATION_BLOCKED")
    return _stage("UNKNOWN", "AUTHORITY_EXECUTION_NOT_RECORDED", executed=False)


def _same_time(left: object, right: object) -> bool:
    left_time, right_time = _time(left), _time(right)
    if left_time is not None and right_time is not None:
        return left_time == right_time
    return str(left or "").strip() == str(right or "").strip()


def _same_text(left: object, right: object, *, folded: bool = False) -> bool:
    left_text, right_text = str(left or "").strip(), str(right or "").strip()
    return (left_text.casefold() == right_text.casefold()) if folded else left_text == right_text


def _selected_output_records(package: dict) -> list[dict]:
    outputs = package.get("games", {}).get("overall", [])
    traces = (package.get("board_diagnostics") or {}).get("traces") or []
    records = []
    for position, output in enumerate(outputs):
        trace = traces[position] if position < len(traces) and isinstance(traces[position], dict) else {}
        contract = output.get("wager_contract")
        if not isinstance(contract, dict):
            contract = output.get("controlled_trial_contract")
        contract = contract if isinstance(contract, dict) else {}
        records.append({
            "section": "overall",
            "position": position,
            "status": output.get("status"),
            "source_candidate_id": str(trace.get("source_candidate_id") or ""),
            "event_id": str(trace.get("game_id") or contract.get("matchup_id") or
                            contract.get("game_id") or ""),
            "run_id": str(output.get("as_of") or ""),
            "sport": str(trace.get("sport") or output.get("sport") or contract.get("sport") or ""),
            "market": str(trace.get("market_type") or output.get("market") or
                          contract.get("market_type") or ""),
            "selection": str(trace.get("selection") or output.get("pick") or
                             contract.get("selection") or ""),
            "line": (_number(trace, "line") if _number(trace, "line") is not None
                     else _number(contract, "line")),
            "sportsbook": str(trace.get("sportsbook") or output.get("quote_source") or
                              contract.get("sportsbook") or ""),
            "quote_id": str(trace.get("quote_id") or contract.get("quote_id") or ""),
            "odds_american": (_number(trace, "odds") if _number(trace, "odds") is not None
                              else _number(output, "odds")),
            "quote_timestamp": str(trace.get("quote_timestamp") or output.get("quote_time") or
                                   contract.get("quote_timestamp") or ""),
        })
    return records


def _identity_conflicts(candidate: Mapping, selected: Mapping) -> list[str]:
    conflicts = []
    for field, folded in (
        ("event_id", False), ("sport", True), ("market", True),
        ("selection", False), ("sportsbook", True), ("quote_id", False),
    ):
        left, right = candidate.get(field), selected.get(field)
        if left not in {None, ""} and right not in {None, ""} and not _same_text(
                left, right, folded=folded):
            conflicts.append(field)
    for field in ("line", "odds_american"):
        left, right = candidate.get(field), selected.get(field)
        if left is not None and right is not None and not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-9):
            conflicts.append(field)
    for field in ("run_id", "quote_timestamp"):
        left, right = candidate.get(field), selected.get(field)
        if left not in {None, ""} and right not in {None, ""} and not _same_time(left, right):
            conflicts.append(field)
    return conflicts


def _match_result(status: str, reason: str, record: Mapping | None = None) -> dict:
    output = None
    selected_identity = None
    if record is not None:
        output = {key: record.get(key) for key in ("section", "position", "status")}
        selected_identity = {key: record.get(key) for key in (
            "source_candidate_id", "event_id", "run_id", "sport", "market",
            "selection", "line", "sportsbook", "quote_id", "odds_american",
            "quote_timestamp",
        )}
    return {
        "status": status,
        "reason": reason,
        "output": output,
        "selected_identity": selected_identity,
    }


def _output_match(row: Mapping, package: dict) -> dict:
    """Bind a candidate to one exact selected diagnostic position.

    Explicit candidate IDs are authoritative: a conflict or an unselected ID
    never falls back to display text. ID-less legacy rows require complete
    event/run/quote evidence, and ambiguity remains unresolved.
    """

    _, candidate = _candidate_identity(row)
    records = _selected_output_records(package)
    source_id = candidate["source_candidate_id"]
    if source_id:
        selected = [record for record in records
                    if record["source_candidate_id"] == source_id]
        if not selected:
            return _match_result("UNRESOLVED", "EXPLICIT_CANDIDATE_ID_NOT_SELECTED")
        if len(selected) != 1:
            return _match_result("UNRESOLVED", "DUPLICATE_SELECTED_CANDIDATE_ID")
        conflicts = _identity_conflicts(candidate, selected[0])
        if conflicts:
            return _match_result(
                "UNRESOLVED", "EXPLICIT_IDENTITY_CONFLICT:" + ",".join(conflicts)
            )
        return _match_result("MATCHED", "EXACT_SELECTED_CANDIDATE_ID", selected[0])

    required = (
        "event_id", "run_id", "sport", "market", "selection", "line",
        "sportsbook", "odds_american", "quote_timestamp",
    )
    missing = [field for field in required if candidate.get(field) in {None, ""}]
    if missing:
        return _match_result(
            "UNRESOLVED", "LEGACY_CANDIDATE_IDENTITY_INCOMPLETE:" + ",".join(missing)
        )
    complete_records = [record for record in records
                        if all(record.get(field) not in {None, ""} for field in required)]
    matches = [record for record in complete_records
               if not _identity_conflicts(candidate, record)]
    if len(matches) == 1:
        return _match_result("MATCHED", "EXACT_LEGACY_EVENT_QUOTE_IDENTITY", matches[0])
    if len(matches) > 1:
        return _match_result("UNRESOLVED", "AMBIGUOUS_LEGACY_OUTPUT_IDENTITY")
    if len(complete_records) != len(records):
        return _match_result("UNRESOLVED", "SELECTED_OUTPUT_IDENTITY_INCOMPLETE")
    return _match_result("NOT_PRESENT", "NO_EXACT_OUTPUT_IDENTITY_MATCH")


def _quote_stage(row: Mapping, at: datetime, options: Mapping, minutes: int) -> tuple[dict, dict]:
    from app_core.per_game_boards import public_quote
    quote = public_quote(
        row, bool(options.get("college_fallback")),
        nfl_fallback=bool(options.get("nfl_fallback")),
        research_fallback=bool(options.get("research_fallback")),
    )
    start = _time(_text(row, "game_start_utc", "start", "game_time_est"))
    quoted = _time(quote[1]) if quote else None
    timing = {
        "event_start_utc": start.isoformat() if start else None,
        "quote_timestamp": quoted.isoformat() if quoted else None,
        "quote_age_seconds": (at - quoted).total_seconds() if quoted else None,
    }
    if start is None:
        return _stage("BLOCK", "START_TIME_UNAVAILABLE"), timing
    if start <= at:
        return _stage("BLOCK", "GAME_STARTED"), timing
    if quote is None or quoted is None:
        return _stage("BLOCK", "EXACT_SUPPORTED_QUOTE_UNAVAILABLE"), timing
    age = (at - quoted).total_seconds()
    if age < 0:
        return _stage("BLOCK", "QUOTE_TIME_FUTURE"), timing
    if age > minutes * 60:
        return _stage("BLOCK", "QUOTE_EXPIRED_AT_PACKAGE_BUILD"), timing
    return _stage("PASS", "EXACT_SUPPORTED_QUOTE_CURRENT"), timing


def build_private_candidate_trace(candidates: pd.DataFrame | None, package: dict, *,
                                  evaluated_at: datetime | str | None = None,
                                  current_at: datetime | str | None = None,
                                  selection_options: Mapping | None = None) -> dict:
    """Build a sanitized private trace from the actual candidate frame."""
    from app_core.public_board import validate_package
    from app_core.release_preflight import evaluate_release
    validate_package(package)
    built = _time(package.get("built_at"))
    at = _time(evaluated_at) if isinstance(evaluated_at, str) else evaluated_at
    at = at or built
    if not isinstance(at, datetime) or at.tzinfo is None:
        raise ValueError("Candidate trace evaluation time must be timezone-aware")
    at = at.astimezone(timezone.utc)
    current = _time(current_at) if isinstance(current_at, str) else current_at
    current = (current or at).astimezone(timezone.utc)
    options = dict(selection_options or {})
    preflight = evaluate_release(package, at=current, validate=False)
    frame = candidates if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        unavailable = {
            "candidate_count": None, "unique_game_count": None,
            "candidate_ids": None, "status": "UNAVAILABLE",
            "reason": "Candidate audit was not supplied; this stage cannot be reconstructed from selected public rows.",
        }
        funnel = {
            "discovered_events": dict(unavailable),
            "raw_quote_records": dict(unavailable),
            "raw_candidate_rows": dict(unavailable),
            "deduplicated_candidates": dict(unavailable),
        }
        funnel.update({name: dict(unavailable) for name in STAGES})
        funnel["packaged_output"].update({
            "output_row_count": len(package.get("games", {}).get("overall", [])),
            "reason": "Selected output rows are recorded, but their full candidate-stage denominator and IDs are unavailable.",
        })
        return {
            "schema_version": VERSION, "trace_status": "HISTORICAL_CANDIDATES_UNAVAILABLE",
            "run_id": None, "evaluated_at": at.isoformat(), "current_at": current.isoformat(),
            "all_market_candidate_count": None,
            "all_market_candidate_count_unavailable_reason": "Candidate audit was not supplied; selected public rows are not the full candidate set.",
            "funnel": funnel, "primary_blocker_counts": {}, "overlapping_blocker_counts": {},
            "candidates": [], "release_preflight": preflight,
        }
    records_by_id: dict[str, dict] = {}
    raw_count = len(frame)
    for _, series in frame.iterrows():
        row = series.to_dict()
        trace_id, identity = _candidate_identity(row)
        if trace_id in records_by_id:
            records_by_id[trace_id]["exact_duplicate_count"] += 1
            continue
        from core.market_policy import production_market
        required = all(identity[key] not in {None, ""} for key in (
            "run_id", "event_id", "sport", "market", "selection", "line", "odds_american",
        ))
        identity_stage = _stage(
            "PASS" if required and production_market(identity["market"]) else "BLOCK",
            "EXACT_IDENTITY_AND_MARKET_VALID" if required and production_market(identity["market"])
            else "IDENTITY_OR_MARKET_INVALID",
        )
        quote_stage, timing = _quote_stage(row, at, options, package_age_minutes(package))
        output_resolution = _output_match(row, package)
        output = output_resolution["output"]
        finalist = _truth(row, "best_available_selected", "selected_for_finalist")
        finalist_stage = _stage(
            "PASS" if finalist is True else "BLOCK" if finalist is False else "UNKNOWN",
            "RECORDED_FINALIST" if finalist is True else "RECORDED_NOT_FINALIST" if finalist is False
            else "FINALIST_DECISION_NOT_RECORDED", executed=finalist is not None,
        )
        if output_resolution["status"] == "MATCHED":
            output_stage = _stage("PASS", output_resolution["reason"])
        elif output_resolution["status"] == "UNRESOLVED":
            output_stage = _stage("UNKNOWN", output_resolution["reason"])
        else:
            output_stage = _stage("BLOCK", output_resolution["reason"])
        release_row = next((item for item in preflight.get("rows", [])
                            if output and item.get("section") == output["section"] and
                            item.get("position") == output["position"]), None)
        if output and output.get("status") == "PASS":
            release_stage = _stage("NOT_APPLICABLE", "RESEARCH_OUTPUT_NOT_ACTIONABLE")
        elif release_row and release_row.get("current_status") == "CURRENT_ACTIONABLE":
            release_stage = _stage("PASS", "CURRENT_ACTIONABLE_RELEASE")
        elif release_row:
            release_stage = _stage(
                "BLOCK", str(release_row.get("current_primary_reason") or
                             "MATCHED_OUTPUT_RELEASE_BLOCKED")
            )
        elif output_resolution["status"] == "UNRESOLVED":
            release_stage = _stage("UNKNOWN", "OUTPUT_MAPPING_UNRESOLVED", executed=False)
        else:
            release_stage = _stage("BLOCK", "NO_MATCHED_ACTIONABLE_OUTPUT", executed=False)
        stages = {
            "parsed_candidate": _stage("PASS", "CANDIDATE_ROW_PARSED"),
            "identity_market": identity_stage,
            "pregame_fresh_quote": quote_stage,
            "model_calibration": _model_stage(row),
            "price_gate": _price_stage(row),
            "authority_review": _authority_stage(row),
            "finalist_selection": finalist_stage,
            "packaged_output": output_stage,
            "release_preflight": release_stage,
        }
        usable = stages["release_preflight"]["status"] == "PASS"
        stages["currently_usable_wager"] = _stage(
            "PASS" if usable else "BLOCK", "CURRENTLY_USABLE" if usable else "NOT_CURRENTLY_USABLE",
            executed=bool(output),
        )
        blockers = [stage["reason"] for stage in stages.values() if stage["status"] in {"BLOCK", "UNKNOWN", "NOT_EVALUATED"}]
        records_by_id[trace_id] = {
            "trace_id": trace_id, "exact_duplicate_count": 1, **identity,
            "model_id": _text(row, "model_id", "source_predictor_version", "predictor_version"),
            "model_version": _text(row, "model_version", "source_predictor_version", "predictor_version"),
            "calibration_id": _text(row, "calibration_id", "Calibration_Version", "calibration_version"),
            "calibration_artifact_sha256": _text(row, "Calibration_Artifact_SHA256", "calibration_artifact_sha256"),
            "probability_semantics": _text(row, "Probability_Semantics", "probability_semantics"),
            "p_win": _number(row, "Final_P_Win", "win_probability_unconditional"),
            "p_push": _number(row, "Final_P_Push", "push_probability"),
            "p_loss": _number(row, "Final_P_Loss", "loss_probability_unconditional"),
            "break_even": _number(row, "Price_Break_Even", "sportsbook_break_even_probability"),
            "mean_ev": _number(row, "Mean_EV_Per_Unit", "mean_expected_value_per_unit"),
            "conservative_probability": _number(row, "P_Win_Conservative", "conservative_probability"),
            "conservative_ev": _number(row, "Conservative_EV_Per_Unit", "conservative_ev"),
            "edge": _number(row, "Absolute_Edge", "absolute_production_edge"),
            "upstream_model_ev": _number(row, "Upstream_Model_EV", "effective_expected_value", "expected_value"),
            "output": output, "output_resolution": output_resolution,
            "package_actionable_release_allowed": preflight["actionable_release_allowed"],
            "stages": stages, "primary_blocker": blockers[0] if blockers else "NONE",
            "overlapping_blockers": list(dict.fromkeys(blockers)), **timing,
        }
    records = sorted(records_by_id.values(), key=lambda item: item["trace_id"])
    funnel: dict[str, dict] = {
        "discovered_events": {"count": None, "status": "UNAVAILABLE", "reason": "Upstream discovery inventory was not supplied."},
        "raw_quote_records": {"count": None, "status": "UNAVAILABLE", "reason": "Raw provider responses are intentionally not reconstructed from candidate rows."},
        "raw_candidate_rows": {"count": raw_count, "status": "RECORDED"},
        "deduplicated_candidates": {"count": len(records), "status": "RECORDED", "candidate_ids": [r["trace_id"] for r in records]},
    }
    for name in STAGES:
        counts = Counter(record["stages"][name]["status"] for record in records)
        survivors = [record for record in records if record["stages"][name]["status"] == "PASS"]
        funnel[name] = {
            "counts": dict(sorted(counts.items())),
            "candidate_count": len(survivors),
            "unique_game_count": len({record["event_id"] for record in survivors if record["event_id"]}),
            "candidate_ids": [record["trace_id"] for record in survivors],
        }
    primary = Counter(record["primary_blocker"] for record in records)
    overlapping = Counter(reason for record in records for reason in record["overlapping_blockers"])
    run_ids = sorted({record["run_id"] for record in records if record["run_id"]})
    return {
        "schema_version": VERSION, "trace_status": "CONTROLLED_REPLAY" if len(run_ids) != 1 else "RECORDED_CANDIDATE_AUDIT",
        "run_id": run_ids[0] if len(run_ids) == 1 else None,
        "evaluated_at": at.isoformat(), "current_at": current.isoformat(),
        "all_market_candidate_count": len(records),
        "all_market_candidate_count_unavailable_reason": None,
        "funnel": funnel,
        "primary_blocker_counts": dict(sorted(primary.items())),
        "overlapping_blocker_counts": dict(sorted(overlapping.items())),
        "candidates": records, "release_preflight": preflight,
    }


def sanitized_summary(report: dict) -> dict:
    """Small private-UI summary; never return raw provider payloads."""
    return {key: report.get(key) for key in (
        "schema_version", "trace_status", "run_id", "evaluated_at", "current_at",
        "all_market_candidate_count", "all_market_candidate_count_unavailable_reason",
        "funnel", "primary_blocker_counts", "overlapping_blocker_counts",
        "release_preflight",
    )}
