"""Keep audited games visible when no current pregame pick can be ranked.

The candidate audit is a schedule/market observation, not wager authority.  Its
unranked games may be displayed as coverage only; they must never acquire a
selection, price, probability, or stake from a rejected candidate.
"""
from __future__ import annotations

import pandas as pd


def _reason(rows: pd.DataFrame) -> str:
    contexts = set(rows["candidate_context"].fillna("").astype(str)) if "candidate_context" in rows else set()
    statuses = set(rows["quote_chronology_status"].fillna("").astype(str)) if "quote_chronology_status" in rows else set()
    if "POST_START_DIAGNOSTIC" in contexts or "POST_START_QUOTE" in statuses:
        return "Game started before a verifiable pregame selection was captured"
    if "QUOTE_TIME_MISSING" in statuses:
        return "Sportsbook quote update time is unavailable; no verified pregame selection"
    if "HISTORICAL_BACKTEST" in contexts:
        return "Game already started; historical candidates are not current picks"
    return "No verified current pregame selection is available"


def unranked_games(best_picks: pd.DataFrame, candidate_audit: pd.DataFrame) -> pd.DataFrame:
    """Return one fail-closed display row per audited game absent from Best Picks.

    Both inputs must belong to the same analysis run.  Conflicting game identity
    within the audit fails closed instead of silently choosing one matchup.
    """
    if not isinstance(candidate_audit, pd.DataFrame) or candidate_audit.empty:
        return pd.DataFrame()
    best_picks = best_picks if isinstance(best_picks, pd.DataFrame) else pd.DataFrame()
    required = {"export_run_id", "matchup_id", "league", "home_team", "away_team", "game_time_est"}
    if not required.issubset(candidate_audit.columns):
        raise ValueError("Candidate audit lacks game coverage identity")
    audit = candidate_audit.copy()
    audit["export_run_id"] = audit["export_run_id"].fillna("").astype(str)
    audit["matchup_id"] = audit["matchup_id"].fillna("").astype(str)
    runs = set(audit["export_run_id"])
    if "" in runs or len(runs) != 1 or audit["matchup_id"].eq("").any():
        raise ValueError("Candidate audit has missing or mixed run/game identity")
    run = next(iter(runs))
    if not best_picks.empty:
        if not {"export_run_id", "matchup_id"}.issubset(best_picks.columns):
            raise ValueError("Best Picks lacks run/game identity")
        best_runs = set(best_picks["export_run_id"].fillna("").astype(str))
        if best_runs != {run}:
            raise ValueError("Best Picks and candidate audit are from different runs")
        ranked_ids = set(best_picks["matchup_id"].fillna("").astype(str))
    else:
        ranked_ids = set()

    rows = []
    for matchup_id, group in audit.groupby("matchup_id", sort=False):
        if matchup_id in ranked_ids:
            continue
        identity = ("league", "home_team", "away_team", "game_time_est")
        if any(group[field].fillna("").astype(str).nunique() != 1 for field in identity):
            raise ValueError("Conflicting candidate audit identity for a game")
        first = group.iloc[0]
        start = str(first["game_time_est"])
        if not start or start.lower() in {"nan", "none"}:
            raise ValueError("Audited game has no start time")
        day = start[:10]
        rows.append({
            "export_run_id": run,
            "matchup_id": matchup_id,
            "league": str(first["league"]),
            "Home": str(first["home_team"]),
            "Away": str(first["away_team"]),
            "Local Date": day,
            "Commence (Local)": start,
            "Bettable": False,
            "Play_Stake": 0.0,
            "production_eligible": False,
            "wager_approved": False,
            "coverage_reason": _reason(group),
        })
    return pd.DataFrame(rows).sort_values(["Commence (Local)", "league", "Away", "Home"]).reset_index(drop=True) if rows else pd.DataFrame()


def publication_games(best_picks: pd.DataFrame, candidate_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append coverage rows while retaining the exact ranked Best Picks export."""
    best = best_picks if isinstance(best_picks, pd.DataFrame) else pd.DataFrame()
    coverage = unranked_games(best, candidate_audit)
    if coverage.empty:
        return best.copy(), coverage
    return pd.concat([best, coverage], ignore_index=True), coverage
