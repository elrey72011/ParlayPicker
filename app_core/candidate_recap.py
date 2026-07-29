"""Grade every best-pick candidate against final scores and summarize rank quality.

The selected best-picks export contains only one row per game.  The candidate audit
contains every side/total alternative that survived the data-integrity gates.  Grading
that wider audit is what lets the app learn whether rank 1 is actually beating rank 2,
and whether a market family is carrying or hurting the card.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


_OUTCOMES = {"WIN", "LOSS", "PUSH"}


def _series(frame: pd.DataFrame, names: Iterable[str], default: object = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index, dtype="object")


def _canonical_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _event_key(frame: pd.DataFrame) -> pd.Series:
    league = _series(frame, ("league", "League"), "").map(_canonical_text)
    home = _series(frame, ("home_team", "Home Team", "Home"), "").map(_canonical_text)
    away = _series(frame, ("away_team", "Away Team", "Away"), "").map(_canonical_text)
    return league + "|" + away + "|" + home


def _parse_number(text: object, pattern: str) -> float | None:
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _grade_candidate(row: pd.Series) -> str:
    home_score = pd.to_numeric(pd.Series([row.get("actual_home_score")]), errors="coerce").iloc[0]
    away_score = pd.to_numeric(pd.Series([row.get("actual_away_score")]), errors="coerce").iloc[0]
    if pd.isna(home_score) or pd.isna(away_score):
        return "N/A"

    home_score = float(home_score)
    away_score = float(away_score)
    if home_score == 0 and away_score == 0:
        return "N/A"

    market_type = str(row.get("market_type", "") or "").strip().casefold()
    pick = str(row.get("best_pick", "") or "").strip()
    pick_folded = pick.casefold()

    if market_type.startswith("total_") or "over" in pick_folded or "under" in pick_folded:
        line = _parse_number(pick, r"(?:over|under)\s*(\d+(?:\.\d+)?)")
        if line is None:
            return "N/A"
        total = home_score + away_score
        side = "over" if market_type == "total_over" or "over" in pick_folded else "under"
        if total == line:
            return "PUSH"
        if side == "over":
            return "WIN" if total > line else "LOSS"
        return "WIN" if total < line else "LOSS"

    if market_type.startswith("spread_") or re.search(r"[+-]\d", pick):
        line = _parse_number(pick, r"([+-]\d+(?:\.\d+)?)")
        if line is None:
            return "N/A"
        if market_type == "spread_home":
            margin = home_score - away_score
        elif market_type == "spread_away":
            margin = away_score - home_score
        else:
            pick_team = re.sub(r"\s*[+-]\d+(?:\.\d+)?\s*$", "", pick).strip()
            pick_key = _canonical_text(pick_team)
            home_key = _canonical_text(row.get("home_team", row.get("Home", "")))
            away_key = _canonical_text(row.get("away_team", row.get("Away", "")))
            if pick_key == home_key:
                margin = home_score - away_score
            elif pick_key == away_key:
                margin = away_score - home_score
            else:
                return "N/A"
        result = margin + line
        return "PUSH" if result == 0 else ("WIN" if result > 0 else "LOSS")

    if market_type in {"moneyline_home", "ml_home"}:
        margin = home_score - away_score
    elif market_type in {"moneyline_away", "ml_away"}:
        margin = away_score - home_score
    else:
        pick_team = re.sub(r"\s+ml\s*$", "", pick, flags=re.IGNORECASE).strip()
        pick_key = _canonical_text(pick_team)
        home_key = _canonical_text(row.get("home_team", row.get("Home", "")))
        away_key = _canonical_text(row.get("away_team", row.get("Away", "")))
        if pick_key == home_key:
            margin = home_score - away_score
        elif pick_key == away_key:
            margin = away_score - home_score
        else:
            return "N/A"
    return "PUSH" if margin == 0 else ("WIN" if margin > 0 else "LOSS")


def _ledger_key(frame: pd.DataFrame) -> pd.Series:
    date_value = _series(frame, ("game_date", "slate_date"), "").fillna("").astype(str)
    matchup = _series(frame, ("matchup_id",), "").fillna("").astype(str)
    slate_identity = date_value.where(date_value.str.strip().ne(""), matchup)
    slate_identity = slate_identity.map(_canonical_text)
    market = _series(frame, ("market_type",), "").map(_canonical_text)
    pick = _series(frame, ("best_pick", "Pick Taken", "Best Pick"), "").map(_canonical_text)
    return slate_identity + "|" + _event_key(frame) + "|" + market + "|" + pick


def grade_candidate_audit(
    candidate_audit: pd.DataFrame,
    scored_picks: pd.DataFrame,
) -> pd.DataFrame:
    """Attach final scores to every candidate row and grade its offered line."""

    if candidate_audit is None or candidate_audit.empty:
        return pd.DataFrame() if candidate_audit is None else candidate_audit.copy()
    if scored_picks is None or scored_picks.empty:
        out = candidate_audit.copy()
        out["actual_home_score"] = pd.NA
        out["actual_away_score"] = pd.NA
        out["candidate_outcome"] = "N/A"
        out["candidate_graded"] = False
        out["candidate_ledger_key"] = _ledger_key(out)
        return out

    out = candidate_audit.copy()
    rename = {}
    for source, target in (
        ("League", "league"),
        ("Home Team", "home_team"),
        ("Home", "home_team"),
        ("Away Team", "away_team"),
        ("Away", "away_team"),
        ("Pick Taken", "best_pick"),
        ("Best Pick", "best_pick"),
    ):
        if source in out.columns and target not in out.columns:
            rename[source] = target
    out = out.rename(columns=rename)
    out = out.drop(
        columns=[
            "actual_home_score",
            "actual_away_score",
            "candidate_outcome",
            "candidate_graded",
            "candidate_ledger_key",
        ],
        errors="ignore",
    )
    out["_candidate_event_key"] = _event_key(out)

    score_frame = scored_picks.copy()
    score_frame["_candidate_event_key"] = _event_key(score_frame)
    score_frame["actual_home_score"] = pd.to_numeric(
        _series(score_frame, ("actual_home_score", "Home Score", "home_score")),
        errors="coerce",
    )
    score_frame["actual_away_score"] = pd.to_numeric(
        _series(score_frame, ("actual_away_score", "Away Score", "away_score")),
        errors="coerce",
    )
    score_lookup = (
        score_frame[
            ["_candidate_event_key", "actual_home_score", "actual_away_score"]
        ]
        .dropna(subset=["actual_home_score", "actual_away_score"])
        .drop_duplicates("_candidate_event_key", keep="last")
    )

    out = out.merge(score_lookup, on="_candidate_event_key", how="left")
    out["candidate_outcome"] = out.apply(_grade_candidate, axis=1)
    out["candidate_graded"] = out["candidate_outcome"].isin(_OUTCOMES)
    out["candidate_ledger_key"] = _ledger_key(out)
    return out.drop(columns=["_candidate_event_key"], errors="ignore")


def merge_candidate_ledgers(
    current: pd.DataFrame,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append current grades to a prior ledger, replacing duplicate candidate rows."""

    frames = []
    if prior is not None and not prior.empty:
        frames.append(prior.copy())
    if current is not None and not current.empty:
        frames.append(current.copy())
    if not frames:
        return pd.DataFrame()

    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger["candidate_outcome"] = _series(
        ledger, ("candidate_outcome", "Outcome"), "N/A"
    ).fillna("N/A").astype(str).str.upper()
    ledger["candidate_graded"] = ledger["candidate_outcome"].isin(_OUTCOMES)
    ledger["candidate_ledger_key"] = _ledger_key(ledger)
    return ledger.drop_duplicates("candidate_ledger_key", keep="last").reset_index(drop=True)


def _summarize(ledger: pd.DataFrame, group_column: str, output_column: str) -> pd.DataFrame:
    columns = [
        output_column,
        "Candidates",
        "Graded",
        "Wins",
        "Losses",
        "Pushes",
        "Hit Rate",
        "Selected Rows",
        "Avg Probability",
        "Avg EV",
    ]
    if ledger is None or ledger.empty or group_column not in ledger.columns:
        return pd.DataFrame(columns=columns)

    work = ledger.copy()
    work["candidate_outcome"] = _series(
        work, ("candidate_outcome", "Outcome"), "N/A"
    ).fillna("N/A").astype(str).str.upper()
    work["_group"] = work[group_column].where(work[group_column].notna(), "Unknown")
    work["_probability"] = pd.to_numeric(
        _series(work, ("selection_probability_used",)), errors="coerce"
    )
    work["_ev"] = pd.to_numeric(_series(work, ("expected_value",)), errors="coerce")
    selected = _series(work, ("best_available_selected",), False)
    work["_selected"] = selected.fillna(False).astype(bool)

    rows = []
    for value, group in work.groupby("_group", dropna=False, sort=False):
        outcomes = group["candidate_outcome"]
        wins = int(outcomes.eq("WIN").sum())
        losses = int(outcomes.eq("LOSS").sum())
        pushes = int(outcomes.eq("PUSH").sum())
        decisions = wins + losses
        rows.append(
            {
                output_column: value,
                "Candidates": int(len(group)),
                "Graded": wins + losses + pushes,
                "Wins": wins,
                "Losses": losses,
                "Pushes": pushes,
                "Hit Rate": (wins / decisions) if decisions else 0.0,
                "Selected Rows": int(group["_selected"].sum()),
                "Avg Probability": group["_probability"].mean(),
                "Avg EV": group["_ev"].mean(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_candidate_performance(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return cumulative performance views by global rank, family rank, and market."""

    rank = _summarize(ledger, "best_available_rank", "Candidate Rank")
    if not rank.empty:
        rank["_sort"] = pd.to_numeric(rank["Candidate Rank"], errors="coerce")
        rank = rank.sort_values("_sort", na_position="last").drop(columns="_sort").reset_index(drop=True)

    family_rank = _summarize(ledger, "best_available_family_rank", "Family Rank")
    if not family_rank.empty:
        family_rank["_sort"] = pd.to_numeric(family_rank["Family Rank"], errors="coerce")
        family_rank = (
            family_rank.sort_values("_sort", na_position="last")
            .drop(columns="_sort")
            .reset_index(drop=True)
        )

    market_family = _summarize(ledger, "market_family", "Market Family")
    if not market_family.empty:
        market_family = market_family.sort_values("Market Family").reset_index(drop=True)

    return {
        "rank": rank,
        "family_rank": family_rank,
        "market_family": market_family,
    }
