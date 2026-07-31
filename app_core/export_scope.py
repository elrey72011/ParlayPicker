"""Unambiguous wager instructions for full-board and research exports."""

from __future__ import annotations

import pandas as pd


_CANONICAL_EXPORT_COLUMNS = ("Bettable", "Export_Scope", "Wager_Instruction")


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
    """Label every row as a funded wager or non-bet coverage/research row."""

    if frame is None:
        return frame
    out = _drop_case_insensitive_export_aliases(frame.copy())
    funded = _positive_stake(out)

    if "production_eligible" in out.columns:
        funded &= out["production_eligible"].fillna(False).astype(bool)
    if "Bet_Decision" in out.columns:
        decision = out["Bet_Decision"].fillna("").astype(str).str.upper().str.strip()
        funded &= decision.eq("BET")
    if "Stake_Status" in out.columns:
        stake_status = out["Stake_Status"].fillna("").astype(str).str.casefold().str.strip()
        if stake_status.ne("").any():
            funded &= stake_status.eq("funded")

    out["Bettable"] = funded
    out["Export_Scope"] = "COVERAGE / RESEARCH"
    out["Wager_Instruction"] = "DO NOT BET - $0 PASS / RESEARCH"
    out.loc[funded, "Export_Scope"] = "PRODUCTION BET"
    out.loc[funded, "Wager_Instruction"] = "BET - APP APPROVED"
    return out


def production_wagers(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows carrying an app-approved positive production stake."""

    labeled = label_wager_export(frame)
    if labeled is None or labeled.empty:
        return labeled
    return labeled[labeled["Bettable"]].copy()
