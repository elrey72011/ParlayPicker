"""Unambiguous wager instructions for full-board and research exports."""

from __future__ import annotations

import pandas as pd


_CANONICAL_EXPORT_COLUMNS = ("Bettable", "Export_Scope", "Wager_Instruction")


def _strict_bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    """Parse public authorization flags without treating the string ``False`` as true."""

    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def _drop_case_insensitive_export_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one canonical spelling for public wager-scope columns.

    Candidate diagnostics historically carried a lowercase
    ``wager_instruction`` field.  Adding the public ``Wager_Instruction``
    column beside it produced a CSV that pandas could write but common
    case-insensitive readers (including PowerShell ``Import-Csv``) could not
    load.  The canonical public label supersedes those internal aliases.
    """

    canonical_by_fold = {
        column.casefold(): column for column in _CANONICAL_EXPORT_COLUMNS
    }
    aliases = [
        column
        for column in frame.columns
        if column.casefold() in canonical_by_fold
        and column != canonical_by_fold[column.casefold()]
    ]
    if not aliases:
        return frame
    return frame.drop(columns=aliases)


def _positive_stake(frame: pd.DataFrame) -> pd.Series:
    stake_columns = [
        column
        for column in (
            "Play_Stake",
            "production_bet_amount",
            "Kelly_Bet_Size",
            "recommended_bet",
            "Suggested_Stake",
        )
        if column in frame.columns
    ]
    if not stake_columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    stakes = pd.concat(
        [pd.to_numeric(frame[column], errors="coerce").fillna(0.0) for column in stake_columns],
        axis=1,
    )
    return stakes.max(axis=1).gt(0.0)


def label_wager_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Label and reconcile every public wager field from one funded mask.

    The export is a betting artifact, so a stale upstream ``BET`` tier or stake must
    not coexist with ``Bettable=False``. Every non-funded row is forced to zero and
    receives the matching pass/lean label; every funded row is explicitly marked BET.
    Controlled Value wagers remain distinguishable from strict Premium production bets.
    """

    if frame is None:
        return frame
    out = _drop_case_insensitive_export_aliases(frame.copy())
    funded = _positive_stake(out)
    controlled_value = _strict_bool_series(
        out, "controlled_card_recovery"
    )

    if "production_eligible" in out.columns:
        funded &= _strict_bool_series(out, "production_eligible")
    for approval_column in ("wager_approved", "Wager_Approved"):
        if approval_column in out.columns:
            funded &= _strict_bool_series(out, approval_column)
    if "Bet_Decision" in out.columns:
        decision = out["Bet_Decision"].fillna("").astype(str).str.upper().str.strip()
        funded &= decision.eq("BET")
    if "Stake_Status" in out.columns:
        stake_status = out["Stake_Status"].fillna("").astype(str).str.casefold().str.strip()
        if stake_status.ne("").any():
            funded &= stake_status.eq("funded")

    qualified = (
        _strict_bool_series(out, "qualified_pick")
        if "qualified_pick" in out.columns
        else pd.Series(False, index=out.index, dtype=bool)
    )
    qualified_pass = qualified & ~funded
    best_available_pass = ~qualified & ~funded

    # Reconcile every existing public approval/stake field before adding scope labels.
    # Preserve explicit STARTED/UNAVAILABLE blockers instead of relabeling them AVOID.
    decision = out.get(
        "Bet_Decision", pd.Series("", index=out.index, dtype="object")
    ).astype("string").fillna("").str.strip().str.upper()
    tier = out.get(
        "Play_Tier", pd.Series("", index=out.index, dtype="object")
    ).astype("string").fillna("").str.strip().str.upper()
    unavailable = decision.isin({"STARTED", "UNAVAILABLE"}) | tier.isin(
        {"STARTED", "UNAVAILABLE"}
    )

    if "Bet_Decision" in out.columns:
        out.loc[funded, "Bet_Decision"] = "BET"
        out.loc[qualified_pass & ~unavailable, "Bet_Decision"] = "QUALIFIED LEAN - PASS"
        out.loc[best_available_pass & ~unavailable, "Bet_Decision"] = (
            "BEST AVAILABLE - PASS" if "qualified_pick" in out.columns else "PASS"
        )
    if "Play_Tier" in out.columns:
        out.loc[funded, "Play_Tier"] = "BET"
        out.loc[qualified_pass & ~unavailable, "Play_Tier"] = "LEAN"
        out.loc[best_available_pass & ~unavailable, "Play_Tier"] = "AVOID"
    for stake_column in (
        "Play_Stake",
        "Play_Units",
        "production_bet_amount",
        "Kelly_Bet_Size",
        "recommended_bet",
        "Suggested_Stake",
    ):
        if stake_column in out.columns:
            out.loc[~funded, stake_column] = 0.0
    for approval_column in ("Wager_Approved", "wager_approved", "All_Row_Bet"):
        if approval_column in out.columns:
            out[approval_column] = funded

    out["Bettable"] = funded
    out["Export_Scope"] = "COVERAGE / RESEARCH"
    out["Wager_Instruction"] = "DO NOT BET - $0 PASS / RESEARCH"
    if "qualified_pick" in out.columns:
        out.loc[~qualified, "Export_Scope"] = "BEST AVAILABLE PICK / RESEARCH"
        out.loc[~qualified, "Wager_Instruction"] = (
            "DO NOT BET - BEST AVAILABLE PICK DOES NOT CLEAR THE WAGER GATE"
        )
        out.loc[qualified_pass, "Export_Scope"] = "QUALIFIED LEAN / RESEARCH"
        out.loc[qualified_pass, "Wager_Instruction"] = (
            "DO NOT BET - QUALIFIED LEAN HAS NO APPROVED STAKE"
        )
        if "Pick_Status" in out.columns:
            out.loc[~qualified, "Pick_Status"] = "Best Available / Pass"
            out.loc[qualified_pass, "Pick_Status"] = "Qualified Lean / Pass"
        for quality_column in ("Pick_Quality", "Pick Quality"):
            if quality_column in out.columns:
                out.loc[~qualified, quality_column] = "No Bet - Best Available"
                out.loc[qualified_pass, quality_column] = "No Bet - Qualified Lean"
    out.loc[funded, "Export_Scope"] = "PRODUCTION BET"
    out.loc[funded, "Wager_Instruction"] = "BET - APP APPROVED"
    out.loc[funded & controlled_value, "Export_Scope"] = "CONTROLLED VALUE BET"
    out.loc[funded & controlled_value, "Wager_Instruction"] = (
        "BET - CONTROLLED VALUE CARD / SMALL STAKE / NOT PREMIUM"
    )
    return out


def production_wagers(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows carrying an app-approved positive production stake."""

    labeled = label_wager_export(frame)
    if labeled is None or labeled.empty:
        return labeled
    return labeled[labeled["Bettable"]].copy()
