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


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"true", "1", "yes", "y"}


def _price_expected_value(row: pd.Series | dict[str, Any]) -> float | None:
    """Return the final exact-price EV authority available on the row."""
    for column in (
        "effective_expected_value",
        "production_expected_value",
        "expected_value",
    ):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


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
    response_error = _text(get("gemini_response_error"))
    response_valid = _strict_bool(get("gemini_response_valid"))
    reviewed = _strict_bool(get("gemini_reviewed"))
    if not response_valid or not reviewed:
        detail = response_error or "Gemini returned an incomplete structured response"
        return "INVALID_RESPONSE", f"{detail}; wager held at $0", 0.0
    unavailable = (
        not recommended
        or recommended_raw.casefold() in {
            "no gemini pick",
            "gemini analysis unavailable",
        }
        or explanation.casefold() == "gemini analysis unavailable"
        or risk_notes.casefold() == "gemini analysis unavailable"
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

    price_ev = _price_expected_value(row)
    if price_ev is None:
        return (
            "MISSING_PRICE_EV",
            "Exact-price expected value is unavailable; wager held at $0",
            0.0,
        )
    if price_ev <= 0.0:
        return (
            "NO_VALUE_AT_PRICE",
            f"Exact-price expected value is non-positive ({price_ev:+.4f}); wager held at $0",
            0.0,
        )

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
    else:
        out["gemini_review_status"] = "DISABLED"
        out["gemini_gate_reason"] = "Gemini wager gate disabled"
        out["gemini_stake_multiplier"] = 1.0
        out["gemini_approved"] = False

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

    if enabled:
        held = ~gate_ok
        for reason_column in (
            "production_gate_reason",
            "Production_Gate_Reason",
            "Status_Reason",
        ):
            if (
                reason_column in out.columns
                or reason_column == "production_gate_reason"
                or product == "prop"
            ):
                if reason_column not in out.columns:
                    out[reason_column] = ""
                out.loc[held, reason_column] = out.loc[held, "gemini_gate_reason"]
        if "Wager_Instruction" in out.columns:
            out.loc[held, "Wager_Instruction"] = "DO NOT BET - GEMINI REVIEW HOLD / $0"

    if diagnostics is not None:
        prefix = "gemini_prop" if product == "prop" else "gemini_best_pick"
        diagnostics[f"{prefix}_gate_enabled"] = bool(enabled)
        diagnostics[f"{prefix}_reviewed_count"] = int(
            pd.Series(out.get("gemini_reviewed", False), index=out.index)
            .fillna(False)
            .astype(bool)
            .sum()
        )
        diagnostics[f"{prefix}_approved_count"] = int(out["gemini_approved"].sum())
        diagnostics[f"{prefix}_held_count"] = int((enabled & ~out["gemini_approved"]).sum())
        diagnostics[f"{prefix}_status_counts"] = out["gemini_review_status"].value_counts().to_dict()
    return out
