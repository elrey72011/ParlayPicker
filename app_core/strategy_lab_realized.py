from typing import Any, Dict, Tuple

import pandas as pd


VALID_RESULTS = {"WIN", "LOSS", "PUSH"}

STATUS_BUCKET_ACTIONABLE = "Actionable"
STATUS_BUCKET_HIGH_VARIANCE = "High Variance/Speculative"
STATUS_BUCKET_BELOW_THRESHOLD = "Below Threshold"
STATUS_BUCKET_NO_PLAY = "No Play"

REALIZED_MODE_ACTIONABLE_ONLY = "Actionable only"
REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE = "Actionable + High Variance/Speculative"
REALIZED_MODE_ALL_GRADED = "All graded bets"
REALIZED_MODE_CUSTOM = "Custom status filter"

REALIZED_STRATEGY_MODE_ORDER = [
    REALIZED_MODE_ACTIONABLE_ONLY,
    REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
    REALIZED_MODE_ALL_GRADED,
    REALIZED_MODE_CUSTOM,
]

STATUS_BUCKET_ORDER = [
    STATUS_BUCKET_ACTIONABLE,
    STATUS_BUCKET_HIGH_VARIANCE,
    STATUS_BUCKET_BELOW_THRESHOLD,
    STATUS_BUCKET_NO_PLAY,
]


def _first_present(row: pd.Series, columns: list[str]) -> Any:
    for col in columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return row[col]
    return pd.NA


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _matchup_key(row: pd.Series) -> str:
    league = _normalize_text(_first_present(row, ["league", "League"]))
    home = _normalize_text(_first_present(row, ["home_team", "Home", "Home Team"]))
    away = _normalize_text(_first_present(row, ["away_team", "Away", "Away Team"]))
    return f"{league}::{home}::{away}"


def _american_to_decimal(odds: Any) -> float | None:
    try:
        val = float(odds)
    except (TypeError, ValueError):
        return None
    if val == 0:
        return None
    if val > 0:
        return 1.0 + (val / 100.0)
    return 1.0 + (100.0 / abs(val))


def _normalize_status(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return STATUS_BUCKET_ACTIONABLE
    if "actionable" in text:
        return STATUS_BUCKET_ACTIONABLE
    if "high variance" in text or "speculative" in text:
        return STATUS_BUCKET_HIGH_VARIANCE
    if "below" in text and "threshold" in text:
        return STATUS_BUCKET_BELOW_THRESHOLD
    if "no play" in text:
        return STATUS_BUCKET_NO_PLAY
    return STATUS_BUCKET_NO_PLAY


def get_strategy_mode_statuses(mode: str, custom_statuses: list[str] | None = None) -> set[str]:
    if mode == REALIZED_MODE_ACTIONABLE_ONLY:
        return {STATUS_BUCKET_ACTIONABLE}
    if mode == REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE:
        return {STATUS_BUCKET_ACTIONABLE, STATUS_BUCKET_HIGH_VARIANCE}
    if mode == REALIZED_MODE_ALL_GRADED:
        return set(STATUS_BUCKET_ORDER)
    if mode == REALIZED_MODE_CUSTOM:
        return set(custom_statuses or [])
    return {STATUS_BUCKET_ACTIONABLE}


def _summarize_subset(df: pd.DataFrame) -> Dict[str, float]:
    wins = int((df["Result"] == "WIN").sum())
    losses = int((df["Result"] == "LOSS").sum())
    pushes = int((df["Result"] == "PUSH").sum())
    decisions = wins + losses
    total_staked = float(df["Stake"].sum())
    gross_returned = float(df["Gross Return"].sum())
    net_pl = float(df["Net Profit"].sum())
    hit_rate = (wins / decisions) if decisions > 0 else 0.0
    roi = (net_pl / total_staked) if total_staked > 0 else 0.0
    return {
        "Bet Count": int(len(df)),
        "Wins": wins,
        "Losses": losses,
        "Pushes": pushes,
        "Hit Rate": hit_rate,
        "Total Staked": total_staked,
        "Gross Returned": gross_returned,
        "Net P/L": net_pl,
        "ROI": roi,
    }


def compute_status_bucket_summary(realized_df: pd.DataFrame) -> pd.DataFrame:
    eligible = realized_df[realized_df["Excluded Reason"] == ""]
    rows = []
    for bucket in STATUS_BUCKET_ORDER:
        subset = eligible[eligible["Status Bucket"] == bucket]
        row = {"Status Bucket": bucket}
        row.update(_summarize_subset(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_mode_comparison(realized_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in [
        REALIZED_MODE_ACTIONABLE_ONLY,
        REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
        REALIZED_MODE_ALL_GRADED,
    ]:
        mode_statuses = get_strategy_mode_statuses(mode)
        subset = realized_df[(realized_df["Excluded Reason"] == "") & (realized_df["Status Bucket"].isin(mode_statuses))]
        summary = _summarize_subset(subset)
        rows.append(
            {
                "Mode": mode,
                "Bet Count": summary["Bet Count"],
                "Hit Rate": summary["Hit Rate"],
                "Net P/L": summary["Net P/L"],
                "ROI": summary["ROI"],
            }
        )
    return pd.DataFrame(rows)


def build_realized_strategy_lab(
    graded_df: pd.DataFrame,
    strategy_source_df: pd.DataFrame | None = None,
    *,
    default_stake: float = 1.0,
    starting_bankroll: float = 0.0,
    strategy_mode: str = REALIZED_MODE_ACTIONABLE_ONLY,
    custom_statuses: list[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, pd.DataFrame | int]]:
    """
    Build realized Strategy Lab rows strictly from a graded source (same as recap),
    while validating pick consistency against an optional strategy source dataframe.
    """
    if graded_df is None or graded_df.empty:
        empty = pd.DataFrame()
        summary = {
            "Total Staked": 0.0,
            "Gross Returned": 0.0,
            "Net P/L": 0.0,
            "ROI": 0.0,
            "Win Count": 0,
            "Loss Count": 0,
            "Push Count": 0,
            "Hit Rate": 0.0,
            "Mismatch Count": 0,
        }
        diagnostics = {
            "pick_mismatches": empty,
            "missing_scores": empty,
            "missing_odds": empty,
            "excluded_rows": empty,
            "excluded_count": 0,
        }
        return empty, summary, diagnostics

    realized = graded_df.copy()
    realized["matchup_key"] = realized.apply(_matchup_key, axis=1)
    realized["Recap Pick"] = realized.apply(
        lambda row: _first_present(row, ["Pick Taken", "Best Pick", "best_pick", "Pick"]),
        axis=1,
    )

    if strategy_source_df is not None and not strategy_source_df.empty:
        src = strategy_source_df.copy()
        src["matchup_key"] = src.apply(_matchup_key, axis=1)
        src["Strategy Pick"] = src.apply(
            lambda row: _first_present(row, ["Pick Taken", "Best Pick", "best_pick", "Pick"]),
            axis=1,
        )
        src = src[["matchup_key", "Strategy Pick"]].drop_duplicates(subset=["matchup_key"], keep="first")
        realized = realized.merge(src, on="matchup_key", how="left")
    else:
        realized["Strategy Pick"] = pd.NA

    realized["Result"] = realized.get("Pick_Outcome", "N/A").astype(str).str.upper().str.strip()
    realized["Result"] = realized["Result"].where(realized["Result"].isin(VALID_RESULTS), "UNGRADED")
    realized["Status Bucket"] = realized.apply(
        lambda row: _normalize_status(_first_present(row, ["Status", "status", "Recommendation Tier", "recommendation_tier"])),
        axis=1,
    )

    realized["Pick Match"] = True
    strategy_pick_norm = realized["Strategy Pick"].map(_normalize_text)
    recap_pick_norm = realized["Recap Pick"].map(_normalize_text)
    has_strategy_pick = strategy_pick_norm != ""
    realized.loc[has_strategy_pick, "Pick Match"] = strategy_pick_norm[has_strategy_pick] == recap_pick_norm[has_strategy_pick]
    realized["Result"] = realized["Result"].where(realized["Pick Match"], "MISMATCH")

    realized["Stake"] = pd.to_numeric(realized.get("Stake", realized.get("recommended_bet", default_stake)), errors="coerce").fillna(default_stake)
    realized["American Odds"] = pd.to_numeric(realized.get("American Odds", realized.get("odds_american", pd.NA)), errors="coerce")
    realized["Decimal Odds"] = pd.to_numeric(realized.get("Decimal Odds", realized.get("decimal_odds", pd.NA)), errors="coerce")

    missing_decimal = realized["Decimal Odds"].isna()
    realized.loc[missing_decimal, "Decimal Odds"] = realized.loc[missing_decimal, "American Odds"].map(_american_to_decimal)

    realized["Excluded Reason"] = ""
    score_missing = realized.get("actual_home_score", pd.Series([pd.NA] * len(realized))).isna() | realized.get("actual_away_score", pd.Series([pd.NA] * len(realized))).isna()
    realized.loc[realized["Result"] == "MISMATCH", "Excluded Reason"] = "MISMATCH"
    realized.loc[(realized["Result"] == "UNGRADED") & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "UNGRADED"
    realized.loc[(score_missing) & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "MISSING_SCORES"
    realized.loc[(realized["Result"] == "WIN") & (realized["Decimal Odds"].isna()) & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "MISSING_ODDS"

    core_eligible = realized["Excluded Reason"] == ""
    selected_statuses = get_strategy_mode_statuses(strategy_mode, custom_statuses)
    realized["Mode Included"] = realized["Status Bucket"].isin(selected_statuses)
    realized["Include In Totals"] = core_eligible & realized["Mode Included"]

    realized["Gross Return"] = 0.0
    realized["Net Profit"] = 0.0

    include = realized["Include In Totals"]
    win_mask = core_eligible & (realized["Result"] == "WIN")
    loss_mask = core_eligible & (realized["Result"] == "LOSS")
    push_mask = core_eligible & (realized["Result"] == "PUSH")

    realized.loc[win_mask, "Gross Return"] = realized.loc[win_mask, "Stake"] * realized.loc[win_mask, "Decimal Odds"]
    realized.loc[win_mask, "Net Profit"] = realized.loc[win_mask, "Gross Return"] - realized.loc[win_mask, "Stake"]
    realized.loc[loss_mask, "Gross Return"] = 0.0
    realized.loc[loss_mask, "Net Profit"] = -realized.loc[loss_mask, "Stake"]
    realized.loc[push_mask, "Gross Return"] = realized.loc[push_mask, "Stake"]
    realized.loc[push_mask, "Net Profit"] = 0.0

    running_net = realized["Net Profit"].where(include, 0.0).cumsum()
    realized["Running Bankroll"] = starting_bankroll + running_net

    included_df = realized[include]
    mode_summary = _summarize_subset(included_df)

    summary = {
        "Total Staked": mode_summary["Total Staked"],
        "Gross Returned": mode_summary["Gross Returned"],
        "Net P/L": mode_summary["Net P/L"],
        "ROI": mode_summary["ROI"],
        "Win Count": mode_summary["Wins"],
        "Loss Count": mode_summary["Losses"],
        "Push Count": mode_summary["Pushes"],
        "Hit Rate": mode_summary["Hit Rate"],
        "Bet Count": mode_summary["Bet Count"],
        "Mismatch Count": int((realized["Result"] == "MISMATCH").sum()),
        "Strategy Mode": strategy_mode,
    }

    status_summary = compute_status_bucket_summary(realized)
    mode_comparison = compute_mode_comparison(realized)

    diagnostics = {
        "pick_mismatches": realized[realized["Result"] == "MISMATCH"].copy(),
        "missing_scores": realized[realized["Excluded Reason"] == "MISSING_SCORES"].copy(),
        "missing_odds": realized[realized["Excluded Reason"] == "MISSING_ODDS"].copy(),
        "excluded_rows": realized[realized["Excluded Reason"] != ""].copy(),
        "excluded_count": int((realized["Excluded Reason"] != "").sum()),
        "outside_mode_rows": realized[(realized["Excluded Reason"] == "") & (~realized["Mode Included"])].copy(),
        "status_bucket_summary": status_summary,
        "mode_comparison": mode_comparison,
        "ungraded_count": int((realized["Result"] == "UNGRADED").sum()),
        "missing_odds_count": int((realized["Excluded Reason"] == "MISSING_ODDS").sum()),
        "mismatch_count": int((realized["Result"] == "MISMATCH").sum()),
        "outside_mode_count": int(((realized["Excluded Reason"] == "") & (~realized["Mode Included"])).sum()),
    }

    return realized, summary, diagnostics
