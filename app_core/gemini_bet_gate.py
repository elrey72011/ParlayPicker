"""Deterministic, fail-closed Gemini review gates for wager cards.

Gemini is a secondary reviewer, not a probability model.  It may confirm a
quantitatively qualified pick, reduce its stake, or hold it.  It can never turn
an otherwise ineligible row into a wager or synthesize an opposing ticket whose
price/line was not validated by the deterministic pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, MutableMapping

import pandas as pd


APPROVED_CONFIDENCE = frozenset({"HIGH", "MEDIUM"})
HARD_BLOCKING_FLAGS = frozenset(
    {
        "incomplete_data",
        "missing_data",
        "missing_live_stats",
        "missing_odds",
        "no_value_at_price",
        "picked_opposing_side",
        "stale_data",
    }
)


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _pick_key(value: Any) -> str:
    """Normalize display text while preserving spread/total signs and decimals."""
    normalized = re.sub(r"\s+", "", _text(value).casefold())
    return re.sub(r"[^a-z0-9+\-.]", "", normalized)


def _flag_set(value: Any) -> set[str]:
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                values = parsed if isinstance(parsed, list) else [raw]
            except (TypeError, ValueError, json.JSONDecodeError):
                values = re.split(r"[|,]", raw)
        else:
            values = re.split(r"[|,]", raw)
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        values = []
    return {
        re.sub(r"[^a-z0-9]+", "_", _text(item).casefold()).strip("_")
        for item in values
        if _text(item)
    }


def classify_gemini_review(row: pd.Series | dict[str, Any]) -> tuple[str, str, float]:
    """Return ``(status, reason, stake_multiplier)`` for one Gemini review."""
    get = row.get
    selected = _pick_key(get("best_pick"))
    recommended_raw = _text(get("gemini_pick"))
    recommended = _pick_key(recommended_raw)
    confidence = _text(get("gemini_confidence")).upper()
    flags = _flag_set(get("gemini_flags"))
    explanation = _text(get("gemini_explanation"))
    risk_notes = _text(get("gemini_risk_notes"))
    reviewed_marker = _text(get("gemini_reviewed"))
    unavailable = (
        not recommended
        or recommended_raw.casefold() in {
            "no gemini pick",
            "gemini analysis unavailable",
        }
        or explanation.casefold() == "gemini analysis unavailable"
        or risk_notes.casefold() == "gemini analysis unavailable"
        or (
            bool(reviewed_marker)
            and reviewed_marker.casefold() not in {"true", "1", "yes"}
        )
    )
    if unavailable:
        return "UNAVAILABLE", "Gemini review unavailable; wager held at $0", 0.0
    if recommended_raw.casefold() in {"none", "abstain", "pass"}:
        return "ABSTAIN", "Gemini abstained; wager held at $0", 0.0
    if not selected or recommended != selected:
        return (
            "OPPOSE",
            "Gemini selected a different side; unpriced flips are prohibited and the wager is held at $0",
            0.0,
        )
    if confidence not in APPROVED_CONFIDENCE:
        return "LOW_CONFIDENCE", "Gemini confidence is below MEDIUM; wager held at $0", 0.0

    combined_notes = f"{explanation} {risk_notes}".casefold()
    if "league-average fallbacks" in combined_notes or "missing live stats" in combined_notes:
        flags.add("missing_live_stats")
    blocking = sorted(flags.intersection(HARD_BLOCKING_FLAGS))
    if blocking:
        return (
            "HOLD",
            f"Gemini raised blocking risk flag(s): {', '.join(blocking)}; wager held at $0",
            0.0,
        )
    multiplier = 1.0 if confidence == "HIGH" else 0.75
    return (
        "APPROVE",
        "Gemini independently confirmed the selected side"
        + (" with a 75% stake multiplier" if multiplier < 1.0 else ""),
        multiplier,
    )


def apply_gemini_bet_gate(
    frame: pd.DataFrame,
    *,
    enabled: bool,
    product: str,
    diagnostics: MutableMapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach audited Gemini verdicts and fail closed on potential wagers.

    Existing deterministic eligibility is only narrowed.  This function never
    grants eligibility and never increases a stake.
    """
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    out["gemini_gate_enabled"] = bool(enabled)

    if enabled:
        classified = [classify_gemini_review(row) for _, row in out.iterrows()]
        out["gemini_review_status"] = [item[0] for item in classified]
        out["gemini_gate_reason"] = [item[1] for item in classified]
        out["gemini_stake_multiplier"] = [item[2] for item in classified]
        out["gemini_approved"] = out["gemini_review_status"].eq("APPROVE")
        out["gemini_reviewed"] = ~out["gemini_review_status"].eq("UNAVAILABLE")
    else:
        out["gemini_review_status"] = "DISABLED"
        out["gemini_gate_reason"] = "Gemini wager gate disabled"
        out["gemini_stake_multiplier"] = 1.0
        out["gemini_approved"] = False
        out["gemini_reviewed"] = False

    gate_ok = ~out["gemini_gate_enabled"] | out["gemini_approved"]
    if "production_eligible" in out.columns:
        existing = pd.Series(out["production_eligible"], index=out.index).fillna(False).astype(bool)
        out["production_eligible"] = existing & gate_ok

    # Defense in depth for frames that already carry dollars when the review is
    # attached (player props).  Final game-card sizing applies the multiplier in
    # the portfolio allocator after all other caps.
    for column in (
        "Kelly_Bet_Size",
        "Play_Stake",
        "production_bet_amount",
        "recommended_bet",
        "Suggested_Stake",
    ):
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
            out.loc[~gate_ok, column] = 0.0
            if product == "prop":
                out.loc[gate_ok, column] = (
                    values.loc[gate_ok]
                    * pd.to_numeric(
                        out.loc[gate_ok, "gemini_stake_multiplier"], errors="coerce"
                    ).fillna(0.0)
                ).round(2)

    if enabled and product == "prop":
        held = ~gate_ok
        if "production_gate_reason" not in out.columns:
            out["production_gate_reason"] = ""
        out.loc[held, "production_gate_reason"] = out.loc[held, "gemini_gate_reason"]
        if "Status_Reason" not in out.columns:
            out["Status_Reason"] = ""
        out.loc[held, "Status_Reason"] = out.loc[held, "gemini_gate_reason"]
        if "Wager_Instruction" in out.columns:
            out.loc[held, "Wager_Instruction"] = "DO NOT BET - GEMINI REVIEW HOLD / $0"

    if diagnostics is not None:
        prefix = "gemini_prop" if product == "prop" else "gemini_best_pick"
        diagnostics[f"{prefix}_gate_enabled"] = bool(enabled)
        diagnostics[f"{prefix}_reviewed_count"] = int(
            (~out["gemini_review_status"].isin({"DISABLED", "UNAVAILABLE"})).sum()
        )
        diagnostics[f"{prefix}_approved_count"] = int(out["gemini_approved"].sum())
        diagnostics[f"{prefix}_held_count"] = int((enabled & ~out["gemini_approved"]).sum())
        diagnostics[f"{prefix}_status_counts"] = out["gemini_review_status"].value_counts().to_dict()
    return out
