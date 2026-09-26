"""Point-in-time and model-authority facts for candidate ranking exports.

These are evidence labels, not a way to grant wager authority.  In particular,
an attractive model probability or price cannot make an old quote current.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.selector_validation import timestamp


CHRONOLOGY_FIELDS = (
    "quote_chronology_status", "selected_quote_recorded_at",
    "minutes_to_start_at_quote", "pregame_quote_valid", "candidate_context",
)
AUTHORITY_FIELDS = (
    "independent_model_available", "probability_authority",
    "production_model_eligible", "model_scope_status", "model_validation_status",
    "market_validation_status",
)


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def _present(value):
    if value is None or isinstance(value, (list, dict)):
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip() not in {"", "nan", "None", "<NA>"}


def _time(value):
    return timestamp(value) if _present(value) else pd.NaT


def _bool(value):
    return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


def classify(row: Mapping, *, as_of) -> dict:
    """Bind the selected price to its retained quote and compare UTC instants."""
    from app_core.prediction_evidence import matching_quotes

    start = _time(row.get("game_start_utc"))
    selected = _time(row.get("selected_quote_recorded_at"))
    if pd.isna(selected):
        selected = _time(row.get("odds_recorded_at"))
    if pd.isna(selected):
        selected = _time(row.get("quote_time"))
    matches = matching_quotes(row) if _present(row.get("provider_quotes")) else []
    # An exact retained quote is the authority for the selected price.  A
    # supplied timestamp may be later (capture time), but never earlier than
    # the provider update.  No date-only or game-date substitution is allowed.
    provider = _time(matches[0].get("recorded_at")) if len(matches) == 1 else pd.NaT
    last_update = _time(row.get("provider_last_update"))
    if pd.isna(selected) and len(matches) == 1:
        selected = provider
    if pd.isna(start):
        status = "GAME_START_MISSING"
    elif pd.isna(selected):
        status = "QUOTE_TIME_MISSING"
    elif _present(row.get("provider_quotes")) and len(matches) != 1:
        status = "QUOTE_TIME_MISSING"
    elif _present(row.get("provider_quotes")) and pd.isna(provider):
        status = "QUOTE_TIME_MISSING"
    elif _present(row.get("provider_last_update")) and pd.isna(_time(row.get("provider_last_update"))):
        status = "PROVIDER_TIME_INCONSISTENT"
    elif (not pd.isna(provider) and provider > selected) or (not pd.isna(last_update) and last_update > selected):
        status = "PROVIDER_TIME_INCONSISTENT"
    elif selected >= start:
        status = "POST_START_QUOTE"
    else:
        status = "VERIFIED_PREGAME"

    observed = _time(as_of)
    context = ("POST_START_DIAGNOSTIC" if not pd.isna(start) and not pd.isna(selected) and selected >= start
               else "CURRENT_PREGAME" if status == "VERIFIED_PREGAME" and not pd.isna(observed) and observed < start
               else "HISTORICAL_BACKTEST")
    return {
        "quote_chronology_status": status,
        "selected_quote_recorded_at": selected.isoformat() if not pd.isna(selected) else None,
        "minutes_to_start_at_quote": (start - selected).total_seconds() / 60 if not pd.isna(start) and not pd.isna(selected) else None,
        "pregame_quote_valid": status == "VERIFIED_PREGAME",
        "candidate_context": context,
    }


def model_authority(row: Mapping) -> dict:
    league = str(row.get("league", row.get("sport", ""))).strip().upper()
    if league == "NCAAF":
        return {"independent_model_available": False, "probability_authority": "RESEARCH_BLEND_ONLY",
                "production_model_eligible": False, "model_scope_status": "NO_INDEPENDENT_MARKET_MODEL",
                "model_validation_status": "UNVALIDATED", "market_validation_status": "UNVALIDATED"}
    if league == "MLB":
        return {"independent_model_available": False, "probability_authority": "NOT_PRODUCTION_AUTHORITY",
                "production_model_eligible": False, "model_scope_status": "RESEARCH_MODEL",
                "model_validation_status": "UNVALIDATED", "market_validation_status": "UNVALIDATED"}
    validated = _bool(row.get("model_validated")) and _bool(row.get("calibration_validated"))
    market = str(row.get("market_type", "")).strip().lower()
    family = "spread" if market.startswith("spread") else "total" if market.startswith("total") else ""
    market_validated = validated and (
        str(row.get("market_validation_status", "")).upper() == "VALIDATED"
        or (family and str(row.get("validated_evidence_family", "")).strip().lower() == family)
    )
    return {"independent_model_available": validated,
            "probability_authority": "VALIDATED_EXACT_MARKET" if validated else "NOT_PRODUCTION_AUTHORITY",
            "production_model_eligible": validated,
            "model_scope_status": "VALIDATED_MODEL" if validated else "UNVERIFIED_MODEL",
            "model_validation_status": "VALIDATED" if validated else "UNVALIDATED",
            "market_validation_status": "VALIDATED" if market_validated else "UNVALIDATED"}


def annotate(frame: pd.DataFrame, *, as_of) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    facts = [dict(classify(row, as_of=as_of), **model_authority(row)) for row in frame.to_dict("records")]
    result = frame.copy()
    for name in CHRONOLOGY_FIELDS + AUTHORITY_FIELDS:
        result[name] = [fact[name] for fact in facts]
    return result


def contradictions(frame: pd.DataFrame) -> dict[str, int]:
    """Count contradictions in an export; absent safety evidence fails closed."""
    if frame.empty:
        return {"chronology": 0, "semantics": 0}
    def flag(name):
        return frame[name].map(_bool) if name in frame else pd.Series(False, index=frame.index)
    def text(name):
        return frame[name].astype("string").fillna("") if name in frame else pd.Series("", index=frame.index)
    current = text("candidate_context").eq("CURRENT_PREGAME")
    pregame = flag("pregame_quote_valid")
    approved = flag("wager_approved")
    selected = flag("best_available_selected")
    finalist = flag("best_available_finalist")
    final_valid = flag("final_pick_valid")
    live_reason = text("final_pick_valid_reason").eq("validated_live_line")
    chronology = ((selected | finalist) & ~current | (selected & current & ~pregame) |
                  (live_reason & ~pregame) | (final_valid & ~pregame) |
                  (approved & (~pregame | ~current | ~final_valid)))
    semantics = approved & (~flag("production_model_eligible") |
                            ~text("market_validation_status").eq("VALIDATED"))
    return {"chronology": int(chronology.sum()), "semantics": int(semantics.sum())}


def assert_integrity(frame: pd.DataFrame) -> None:
    counts = contradictions(frame)
    if any(counts.values()):
        raise ValueError(f"CANDIDATE_AUDIT_INTEGRITY_FAILURE: {counts}")
