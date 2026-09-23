"""Public, producer-time diagnostics for the selected overall game board.

This is a snapshot of the selected rows, not a reconstruction of every market
candidate considered upstream. Browser-time expiry is evaluated separately.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import re

from app_core.public_quote_policy import supported_quote


VERSION = "selected-overall-v1"
TRACE_FIELDS = frozenset("""selected_row_id source_candidate_id game_id sport market_type selection line
    sportsbook odds quote_timestamp analysis_timestamp evaluated_at expires_at
    identity_status quote_status feature_status model_version calibration_version
    evidence_snapshot_id policy_version maturity probability_semantics p_win p_push
    p_loss conservative_probability strict_gate_status strict_blockers
    trial_gate_status trial_blockers review_status review_timestamp
    authorization_status allocation_status saved_status producer_primary_reason
    producer_blockers""".split())
ROOT_FIELDS = frozenset("""schema_version built_at selected_rows_hash selected_game_count
    all_market_candidate_count traces""".split())
UPSTREAM_STRICT_REASONS = frozenset("""sport_policy_mismatch unsupported_production_market
    unverified_mapping invalid_exact_price missing_exact_line unverified_alternate_quote
    started_or_missing_start stale_or_missing_quote unvalidated_sport_policy
    unvalidated_model unvalidated_calibration missing_or_future_evidence
    critical_features_unverified missing_or_invalid_conservative_probability
    missing_or_invalid_push_probability nonpositive_conservative_ev
    insufficient_conservative_edge deployment_state_below_maturity
    missing_effective_evidence insufficient_evidence_for_maturity
    zero_validated_allocation unvalidated_maturity_or_cap gemini_hold
    wait_for_material_news""".split())
BLOCKER_CODES = frozenset("""QUOTE_UNAVAILABLE QUOTE_UPDATE_TIME_UNVERIFIED
    QUOTE_TIME_UNAVAILABLE QUOTE_FUTURE QUOTE_EXPIRED QUOTE_EXPLICITLY_UNVERIFIED
    ANALYSIS_TIME_UNAVAILABLE ANALYSIS_TIME_FUTURE ANALYSIS_EXPIRED
    START_TIME_UNAVAILABLE GAME_STARTED STRICT_TRACE_UNAVAILABLE
    IDENTITY_UNVERIFIED EXACT_QUOTE_UNVERIFIED SAVED_QUOTE_NOT_FRESH
    FEATURES_UNVERIFIED MODEL_VERSION_MISSING CALIBRATION_VERSION_MISSING
    EVIDENCE_VERSION_MISSING POLICY_VERSION_MISSING
    CONSERVATIVE_PROBABILITY_MISSING NONPOSITIVE_CONSERVATIVE_EV
    STRICT_GATE_REJECTED SAVED_SELECTION_NOT_APPROVED NO_VALIDATED_ALLOCATION
    TRIAL_TRACE_UNAVAILABLE TRIAL_IDENTITY_UNVERIFIED TRIAL_QUOTE_UNVERIFIED
    TRIAL_QUOTE_NOT_FRESH TRIAL_REVIEW_NOT_APPROVED TRIAL_GATE_REJECTED
    SAVED_SELECTION_NOT_TRIAL NO_TRIAL_ALLOCATION""".split()) | {
        "UPSTREAM_" + code.upper() for code in UPSTREAM_STRICT_REASONS
    }


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _scalar(value):
    if isinstance(value, (str, bool, int, float)) and not (
            isinstance(value, float) and not math.isfinite(value)):
        return value
    return None


def _field(obj, key):
    return _scalar(obj.get(key)) if hasattr(obj, "get") else None


def _identifier(obj, key):
    value = _field(obj, key)
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9:._|/-]{1,128}", value) else None


def _truth_status(value):
    return "VERIFIED" if value is True else "FAILED" if value is False else "UNKNOWN"


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _quote_status(row, contract, at, minutes):
    if not supported_quote(row):
        return "UNAVAILABLE"
    if row.get("quote_time_basis"):
        return "UPDATE_TIME_UNVERIFIED"
    quoted = _time(row.get("quote_time"))
    if quoted is None:
        return "TIME_UNAVAILABLE"
    if quoted > at:
        return "FUTURE"
    if at - quoted > timedelta(minutes=minutes):
        return "EXPIRED"
    if contract and contract.get("quote_verified") is False:
        return "EXPLICITLY_UNVERIFIED"
    return "CURRENT"


def _strict_blockers(contract, saved_status):
    if not contract:
        return ["STRICT_TRACE_UNAVAILABLE"]
    codes = []
    for field, code in (
        ("identity_verified", "IDENTITY_UNVERIFIED"),
        ("quote_verified", "EXACT_QUOTE_UNVERIFIED"),
        ("quote_fresh", "SAVED_QUOTE_NOT_FRESH"),
    ):
        if contract.get(field) is False:
            codes.append(code)
    if contract.get("data_quality_status") not in {None, "VERIFIED"}:
        codes.append("FEATURES_UNVERIFIED")
    for field, code in (
        ("model_version", "MODEL_VERSION_MISSING"),
        ("calibration_version", "CALIBRATION_VERSION_MISSING"),
        ("evidence_version", "EVIDENCE_VERSION_MISSING"),
        ("sport_policy_version", "POLICY_VERSION_MISSING"),
        ("conservative_probability", "CONSERVATIVE_PROBABILITY_MISSING"),
    ):
        if contract.get(field) is None or contract.get(field) == "":
            codes.append(code)
    ev = contract.get("conservative_ev")
    if not isinstance(ev, (int, float)) or isinstance(ev, bool) or not math.isfinite(ev) or ev <= 0:
        codes.append("NONPOSITIVE_CONSERVATIVE_EV")
    if contract.get("production_eligible") is False:
        codes.append("STRICT_GATE_REJECTED")
    for reason in str(contract.get("production_gate_reason") or "").split(";"):
        if reason.strip() in UPSTREAM_STRICT_REASONS:
            codes.append("UPSTREAM_" + reason.strip().upper())
    if contract.get("production_eligible") is True and saved_status != "APPROVED":
        codes.append("SAVED_SELECTION_NOT_APPROVED")
    if not _positive(contract.get("production_bet_amount")):
        codes.append("NO_VALIDATED_ALLOCATION")
    return list(dict.fromkeys(codes))


def _trial_blockers(contract, saved_status):
    if not contract:
        return ["TRIAL_TRACE_UNAVAILABLE"] if saved_status == "TRIAL" else []
    codes = []
    for field, code in (("identity_verified", "TRIAL_IDENTITY_UNVERIFIED"),
                        ("quote_verified", "TRIAL_QUOTE_UNVERIFIED"),
                        ("quote_fresh", "TRIAL_QUOTE_NOT_FRESH")):
        if contract.get(field) is False:
            codes.append(code)
    if contract.get("gemini_review_status") != "APPROVE":
        codes.append("TRIAL_REVIEW_NOT_APPROVED")
    if contract.get("trial_eligible") is False:
        codes.append("TRIAL_GATE_REJECTED")
    if contract.get("trial_eligible") is True and saved_status != "TRIAL":
        codes.append("SAVED_SELECTION_NOT_TRIAL")
    if not _positive(contract.get("recommended_bet_amount")):
        codes.append("NO_TRIAL_ALLOCATION")
    return codes


def _trace(source, row, built_at, minutes):
    strict = row.get("wager_contract") if isinstance(row.get("wager_contract"), dict) else None
    trial = row.get("controlled_trial_contract") if isinstance(row.get("controlled_trial_contract"), dict) else None
    active = trial if row["status"] == "TRIAL" else strict
    quote_status = _quote_status(row, active, built_at, minutes)
    quote_codes = [] if quote_status == "CURRENT" else ["QUOTE_" + quote_status]
    analysis = _time(row.get("as_of"))
    timing = []
    if analysis is None:
        timing.append("ANALYSIS_TIME_UNAVAILABLE")
    elif analysis > built_at:
        timing.append("ANALYSIS_TIME_FUTURE")
    elif built_at - analysis > timedelta(minutes=minutes):
        timing.append("ANALYSIS_EXPIRED")
    start = _time(row.get("start"))
    if start is None:
        timing.append("START_TIME_UNAVAILABLE")
    elif start <= built_at:
        timing.append("GAME_STARTED")
    strict_codes = _strict_blockers(strict, row["status"])
    trial_codes = _trial_blockers(trial, row["status"])
    # A saved trial has its own authority. Strict rejection remains visible but
    # does not hide a current trial decision as the primary producer reason.
    ordered = quote_codes + timing + (trial_codes + strict_codes if row["status"] == "TRIAL" else strict_codes + trial_codes)
    blockers = list(dict.fromkeys(ordered))
    primary = blockers[0] if blockers else {
        "APPROVED": "SAVED_VALIDATED_APPROVAL",
        "TRIAL": "SAVED_CONTROLLED_TRIAL",
        "PASS": "SAVED_RESEARCH_PASS",
    }[row["status"]]
    probability_semantics = _field(trial, "probability_semantics") if trial else None
    p_win = _field(trial, "estimated_probability") if trial else None
    p_push = _field(trial, "push_probability") if trial else None
    p_loss = _field(trial, "loss_probability") if trial else None
    expiry = (min(_time(row.get("quote_time")), analysis) + timedelta(minutes=minutes)).isoformat() if _time(row.get("quote_time")) and analysis else None
    identity = _field(active, "identity_verified")
    feature = _field(active, "data_quality_status")
    if feature == "VERIFIED" or feature == "NO_DEGRADED_FLAGS":
        feature_status = "VERIFIED"
    elif feature is None:
        feature_status = "UNKNOWN"
    else:
        feature_status = "FAILED"
    return {
        "selected_row_id": _digest(row), "source_candidate_id": _identifier(source, "candidate_id"),
        "game_id": _identifier(source, "matchup_id"), "sport": row["sport"],
        "market_type": row["market"], "selection": row["pick"],
        "line": _field(active, "line"), "sportsbook": row.get("quote_source"),
        "odds": row["odds"], "quote_timestamp": row.get("quote_time"),
        "analysis_timestamp": row["as_of"], "evaluated_at": built_at.isoformat(),
        "expires_at": expiry, "identity_status": _truth_status(identity),
        "quote_status": quote_status, "feature_status": feature_status,
        "model_version": _field(strict, "model_version"),
        "calibration_version": _field(strict, "calibration_version"),
        "evidence_snapshot_id": _field(strict, "evidence_version"),
        "policy_version": _field(strict, "sport_policy_version"),
        "maturity": _field(strict, "maturity") or row.get("maturity"),
        "probability_semantics": probability_semantics,
        "p_win": p_win, "p_push": p_push, "p_loss": p_loss,
        "conservative_probability": _field(strict, "conservative_probability"),
        "strict_gate_status": "PASS" if strict and strict.get("production_eligible") is True else "BLOCKED" if strict and strict.get("production_eligible") is False else "UNKNOWN",
        "strict_blockers": strict_codes,
        "trial_gate_status": "PASS" if trial and trial.get("trial_eligible") is True else "BLOCKED" if trial and trial.get("trial_eligible") is False else "UNKNOWN",
        "trial_blockers": trial_codes,
        "review_status": _field(active, "gemini_review_status") or row.get("gemini_review_status"),
        "review_timestamp": row.get("gemini_reviewed_at"),
        "authorization_status": "CONTRACT_CLAIMED" if row["status"] == "TRIAL" and trial else "UNKNOWN",
        "allocation_status": "POSITIVE_VALIDATED" if strict and strict.get("production_eligible") is True and _positive(strict.get("production_bet_amount")) else "POSITIVE_TRIAL" if trial and trial.get("trial_eligible") is True and _positive(trial.get("recommended_bet_amount")) else "NONE_RECORDED",
        "saved_status": row["status"], "producer_primary_reason": primary,
        "producer_blockers": blockers,
    }


def build_selected_diagnostics(source_rows, public_rows, built_at, stale_after_minutes):
    """Freeze one diagnostic per selected overall game; candidate count stays unknown."""
    if len(source_rows) != len(public_rows):
        raise ValueError("Selected overall source and public row counts differ")
    traces = [_trace(source, row, built_at, stale_after_minutes)
              for source, row in zip(source_rows, public_rows)]
    return {"schema_version": VERSION, "built_at": built_at.isoformat(),
            "selected_rows_hash": _digest(public_rows),
            "selected_game_count": len(public_rows),
            "all_market_candidate_count": None, "traces": traces}


def validate_selected_diagnostics(diagnostics, public_rows, built_at, stale_after_minutes):
    """Reject edited/injected diagnostics while accepting packages without this key."""
    if not isinstance(diagnostics, dict) or set(diagnostics) != ROOT_FIELDS:
        raise ValueError("Invalid selected-board diagnostic schema")
    if diagnostics["schema_version"] != VERSION or diagnostics["built_at"] != built_at:
        raise ValueError("Unsupported or mismatched selected-board diagnostic version")
    if diagnostics["all_market_candidate_count"] is not None:
        raise ValueError("Candidate count is unavailable from selected game rows")
    if diagnostics["selected_rows_hash"] != _digest(public_rows):
        raise ValueError("Selected-board diagnostics do not match saved game rows")
    if diagnostics["selected_game_count"] != len(public_rows) or not isinstance(diagnostics["traces"], list) or len(diagnostics["traces"]) != len(public_rows):
        raise ValueError("Every selected overall game needs one diagnostic trace")
    for row, trace in zip(public_rows, diagnostics["traces"]):
        if not isinstance(trace, dict) or set(trace) != TRACE_FIELDS:
            raise ValueError("Invalid selected-game trace fields")
        if trace["selected_row_id"] != _digest(row) or trace["saved_status"] != row["status"] or trace["sport"] != row["sport"] or trace["market_type"] != row["market"] or trace["selection"] != row["pick"] or trace["odds"] != row["odds"] or trace["analysis_timestamp"] != row["as_of"] or trace["quote_timestamp"] != row.get("quote_time") or trace["sportsbook"] != row.get("quote_source") or trace["evaluated_at"] != built_at:
            raise ValueError("Selected-game trace does not match saved selection")
        if not isinstance(trace["producer_blockers"], list) or not isinstance(trace["strict_blockers"], list) or not isinstance(trace["trial_blockers"], list) or any(code not in BLOCKER_CODES for code in trace["producer_blockers"] + trace["strict_blockers"] + trace["trial_blockers"]):
            raise ValueError("Invalid selected-game blocker codes")
        if len(set(trace["producer_blockers"])) != len(trace["producer_blockers"]) or (trace["producer_blockers"] and trace["producer_primary_reason"] != trace["producer_blockers"][0]):
            raise ValueError("Invalid selected-game primary reason")
        if any(value is not None and not isinstance(value, (str, int, float, bool)) for key, value in trace.items() if key not in {"producer_blockers", "strict_blockers", "trial_blockers"}):
            raise ValueError("Invalid selected-game diagnostic value")
        rebuilt = _trace({"matchup_id": trace["game_id"], "candidate_id": trace["source_candidate_id"]},
                         row, _time(built_at), stale_after_minutes)
        if trace != rebuilt:
            raise ValueError("Selected-game trace is not the frozen producer decision")
