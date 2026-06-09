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
REALIZED_MODE_PRODUCTION_CARD = "Production Card"
REALIZED_MODE_CUSTOM = "Custom status filter"
PERFORMANCE_GUARD_ACTION_ZERO_KELLY_OR_WATCHLIST = "zero_kelly_or_watchlist"

REALIZED_STRATEGY_MODE_ORDER = [
    REALIZED_MODE_ACTIONABLE_ONLY,
    REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
    REALIZED_MODE_ALL_GRADED,
    REALIZED_MODE_PRODUCTION_CARD,
    REALIZED_MODE_CUSTOM,
]

STATUS_BUCKET_ORDER = [
    STATUS_BUCKET_ACTIONABLE,
    STATUS_BUCKET_HIGH_VARIANCE,
    STATUS_BUCKET_BELOW_THRESHOLD,
    STATUS_BUCKET_NO_PLAY,
]
NON_PRODUCTION_BUCKETS = [STATUS_BUCKET_HIGH_VARIANCE, STATUS_BUCKET_BELOW_THRESHOLD, STATUS_BUCKET_NO_PLAY]


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


def _first_numeric(row: pd.Series, columns: list[str]) -> float | None:
    for col in columns:
        if col in row:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                return float(val)
    return None


def _extract_market_profile(row: pd.Series, pick_value: Any) -> Dict[str, Any]:
    pick_text = str(pick_value) if pd.notna(pick_value) else ""
    pick_norm = _normalize_text(pick_text)
    lower_pick = pick_text.lower()
    home_norm = _normalize_text(_first_present(row, ["home_team", "Home", "Home Team"]))
    away_norm = _normalize_text(_first_present(row, ["away_team", "Away", "Away Team"]))
    line_value = _first_numeric(row, ["line_value", "Line", "line", "Total Line", "spread_line"])

    market_family = "unknown"
    market_type = "unknown"
    direction = "unknown"

    if "over" in lower_pick or "under" in lower_pick:
        market_family = "total"
        if "over" in lower_pick:
            market_type = "total_over"
            direction = "Over"
        else:
            market_type = "total_under"
            direction = "Under"
    elif " ml" in f" {lower_pick}" or lower_pick.endswith("ml"):
        market_family = "moneyline"
        market_type = "moneyline"
        direction = "home_side" if home_norm and home_norm in pick_norm else "away_side"
    elif any(ch.isdigit() for ch in pick_text) and ("+" in pick_text or "-" in pick_text):
        market_family = "spread"
        direction = "home_side" if home_norm and home_norm in pick_norm else "away_side"
        market_type = "spread_home" if direction == "home_side" else "spread_away"

    if line_value is None:
        import re

        match = re.search(r"([+-]?\d+(?:\.\d+)?)", pick_text)
        if match:
            line_value = float(match.group(1))

    if line_value is not None:
        line_value = round(line_value, 4)

    return {
        "market_family": market_family,
        "market_type": market_type,
        "pick_direction": direction,
        "line_value": line_value,
        "best_pick_normalized": pick_norm,
    }


def _canonical_pick_key(row: pd.Series, pick_col: str) -> str:
    league = _normalize_text(_first_present(row, ["league", "League"]))
    home = _normalize_text(_first_present(row, ["home_team", "Home", "Home Team"]))
    away = _normalize_text(_first_present(row, ["away_team", "Away", "Away Team"]))
    game_date = _normalize_text(_first_present(row, ["game_date", "Game Date", "date"]))
    pick_value = _first_present(row, [pick_col])
    market = _extract_market_profile(row, pick_value)
    line_source = _normalize_text(_first_present(row, ["market_line_source", "line_source"]))
    return "::".join(
        [
            league,
            home,
            away,
            game_date,
            market["market_family"],
            market["market_type"],
            market["pick_direction"],
            str(market["line_value"]),
            market["best_pick_normalized"],
            str(market["line_value"]),
            line_source,
        ]
    )


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
    if mode == REALIZED_MODE_PRODUCTION_CARD:
        return {STATUS_BUCKET_ACTIONABLE}
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


def summarize_recap_tiers(recap_df: pd.DataFrame) -> pd.DataFrame:
    """Tiered W/L summary of a graded Performance Recap, keyed to what is *staked*.

    A single all-rows hit rate blends in the Below Threshold / No Play picks the system
    declined to stake (which trend toward ~50% by construction), masking staked
    performance. This reports the staked tiers separately so the headline reflects the
    card you would actually bet:

      * Actionable (production) — the staked card.
      * Actionable + High Variance — staked + speculative.
      * All graded rows — every graded pick incl. unstaked (reference only).

    Accepts a recap frame with a ``Status`` column and either a boolean ``Win`` column or
    an ``Outcome`` column (WIN/LOSS or W/L). Returns rows of
    ``Tier, Wins, Losses, Total, Hit Rate`` (Hit Rate is a 0..1 float; 0.0 when empty).
    """
    df = recap_df.copy()
    if "Win" in df.columns:
        win = df["Win"].fillna(False).astype(bool)
    else:
        outcome = df.get("Outcome", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
        graded = outcome.isin(["WIN", "LOSS", "W", "L"])
        df = df[graded]
        win = outcome[graded].isin(["WIN", "W"])

    bucket = df.get("Status", pd.Series("", index=df.index)).apply(_normalize_status)
    actionable = bucket.eq(STATUS_BUCKET_ACTIONABLE)
    staked_spec = bucket.isin([STATUS_BUCKET_ACTIONABLE, STATUS_BUCKET_HIGH_VARIANCE])
    all_rows = pd.Series(True, index=df.index)

    def _rec(tier: str, mask: pd.Series) -> Dict[str, Any]:
        w = int(win[mask].sum())
        n = int(mask.sum())
        return {"Tier": tier, "Wins": w, "Losses": n - w, "Total": n,
                "Hit Rate": (w / n) if n else 0.0}

    return pd.DataFrame([
        _rec("Actionable (production)", actionable),
        _rec("Actionable + High Variance", staked_spec),
        _rec("All graded rows (incl. unstaked)", all_rows),
    ])



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


def _group_summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + ["Bet Count", "Wins", "Losses", "Pushes", "Hit Rate", "Total Staked", "Gross Returned", "Net P/L", "ROI"])
    return df.groupby(keys, dropna=False).apply(_summarize_subset).apply(pd.Series).reset_index()


def build_realized_strategy_lab(
    graded_df: pd.DataFrame,
    strategy_source_df: pd.DataFrame | None = None,
    *,
    default_stake: float = 1.0,
    starting_bankroll: float = 0.0,
    strategy_mode: str = REALIZED_MODE_ACTIONABLE_ONLY,
    custom_statuses: list[str] | None = None,
    enable_performance_guard: bool = True,
    min_guard_sample_size: int = 5,
    performance_guard_min_win_rate: float = 0.50,
    performance_guard_min_roi: float = 0.00,
    performance_guard_action: str = PERFORMANCE_GUARD_ACTION_ZERO_KELLY_OR_WATCHLIST,
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
    realized["recap_outcome"] = pd.Series(realized.get("Pick_Outcome", pd.NA), index=realized.index).astype(str).str.upper().str.strip()
    realized["grading_match_source"] = "none"
    realized["grading_conflict_flag"] = False
    realized["grading_conflict_reason"] = ""

    if strategy_source_df is not None and not strategy_source_df.empty:
        src = strategy_source_df.copy()
        src["matchup_key"] = src.apply(_matchup_key, axis=1)
        src["Strategy Pick"] = src.apply(
            lambda row: _first_present(row, ["Pick Taken", "Best Pick", "best_pick", "Pick"]),
            axis=1,
        )
        src["strategy_pick_key"] = src.apply(lambda row: _canonical_pick_key(row, "Strategy Pick"), axis=1)
        for col in ["export_run_id", "pick_id", "canonical_pick_key", "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag"]:
            if col not in src.columns:
                src[col] = pd.NA
        src = src[["matchup_key", "Strategy Pick", "strategy_pick_key", "export_run_id", "pick_id", "canonical_pick_key", "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag"]].drop_duplicates(subset=["matchup_key"], keep="first")
        realized = realized.merge(src, on="matchup_key", how="left")
    else:
        realized["Strategy Pick"] = pd.NA

    existing_result = pd.Series(realized.get("Result", pd.NA), index=realized.index).astype(str).str.upper().str.strip()
    recap_result = realized["recap_outcome"]
    conflict_mask = existing_result.isin(VALID_RESULTS) & recap_result.isin(VALID_RESULTS) & (existing_result != recap_result)
    realized.loc[conflict_mask, "grading_conflict_flag"] = True
    realized.loc[conflict_mask, "grading_conflict_reason"] = "Strategy Lab result disagreed with Performance Recap; overwritten."
    realized["Result"] = recap_result
    realized["Result"] = realized["Result"].where(realized["Result"].isin(VALID_RESULTS), "UNGRADED")
    realized["Status Bucket"] = realized.apply(
        lambda row: _normalize_status(_first_present(row, ["Status", "status", "Recommendation Tier", "recommendation_tier"])),
        axis=1,
    )

    for col in ["export_run_id", "pick_id", "canonical_pick_key", "market_line_used", "market_line_source"]:
        if col not in realized.columns:
            realized[col] = pd.NA
    realized["recap_pick_taken"] = realized["Recap Pick"]
    realized["strategy_lab_pick"] = realized["Strategy Pick"]
    realized["recap_pick_key"] = realized.apply(lambda row: _canonical_pick_key(row, "Recap Pick"), axis=1)
    if "strategy_pick_key" not in realized.columns:
        realized["strategy_pick_key"] = ""
    realized["strategy_pick_key"] = realized["canonical_pick_key"].where(realized["canonical_pick_key"].notna() & realized["canonical_pick_key"].astype(str).str.strip().ne(""), realized["strategy_pick_key"])
    realized["pick_identity_match"] = realized["strategy_pick_key"] == realized["recap_pick_key"]
    realized["pick_identity_match"] = realized["pick_identity_match"] | realized["Strategy Pick"].isna()
    realized.loc[realized["pick_identity_match"], "grading_match_source"] = "canonical_pick_key_or_pick_identity"
    fallback_match = (~realized["pick_identity_match"]) & (realized["Recap Pick"].astype(str).str.strip() == realized["Strategy Pick"].astype(str).str.strip())
    realized.loc[fallback_match, "grading_match_source"] = "league+teams+exact_pick"
    recap_profile = realized.apply(lambda row: _extract_market_profile(row, row["Recap Pick"]), axis=1, result_type="expand")
    strategy_profile = realized.apply(lambda row: _extract_market_profile(row, row["Strategy Pick"]), axis=1, result_type="expand")
    realized["line_match_flag"] = (recap_profile["line_value"] == strategy_profile["line_value"]) | realized["Strategy Pick"].isna()
    realized["direction_match_flag"] = (recap_profile["pick_direction"] == strategy_profile["pick_direction"]) | realized["Strategy Pick"].isna()
    realized["export_run_match_flag"] = True
    if "export_run_id" in realized.columns and "export_run_id" in graded_df.columns:
        realized["export_run_match_flag"] = realized["export_run_id"].astype(str).eq(graded_df.get("export_run_id", pd.Series(dtype=str)).astype(str))
    realized["realized_grade_source"] = "performance_recap"
    realized["grading_excluded_reason"] = ""
    realized.loc[~realized["pick_identity_match"] & (~realized["line_match_flag"]), "grading_excluded_reason"] = (
        "Line mismatch: Strategy Lab "
        + realized["Strategy Pick"].fillna("").astype(str)
        + " vs Recap "
        + realized["Recap Pick"].fillna("").astype(str)
    )
    realized.loc[
        ~realized["pick_identity_match"] & (realized["grading_excluded_reason"] == "") & (~realized["direction_match_flag"]),
        "grading_excluded_reason",
    ] = (
        "Direction mismatch: Strategy Lab "
        + realized["Strategy Pick"].fillna("").astype(str)
        + " vs Recap "
        + realized["Recap Pick"].fillna("").astype(str)
    )
    realized.loc[realized["pick_identity_match"], "grading_excluded_reason"] = "Exact pick match"
    realized.loc[~realized["export_run_match_flag"], "grading_excluded_reason"] = "Export run mismatch"
    realized["Result"] = realized["Result"].where(realized["pick_identity_match"], "MISMATCH")

    default_stake_by_status = {
        STATUS_BUCKET_ACTIONABLE: default_stake,
        STATUS_BUCKET_HIGH_VARIANCE: 0.0 if strategy_mode == REALIZED_MODE_PRODUCTION_CARD else default_stake,
        STATUS_BUCKET_BELOW_THRESHOLD: 0.0,
        STATUS_BUCKET_NO_PLAY: 0.0,
    }
    raw_stake_source = realized["Stake"] if "Stake" in realized.columns else realized.get("recommended_bet", pd.Series([pd.NA] * len(realized)))
    raw_stake = pd.to_numeric(raw_stake_source, errors="coerce")
    fallback_stake = realized["Status Bucket"].map(default_stake_by_status).fillna(default_stake)
    realized["Stake"] = raw_stake.fillna(fallback_stake)
    default_for_row = realized["Status Bucket"].map(default_stake_by_status).fillna(default_stake)
    non_prod_mask = realized["Status Bucket"].isin(NON_PRODUCTION_BUCKETS)
    realized["manual_non_production_stake_flag"] = non_prod_mask & raw_stake.notna() & (raw_stake > default_for_row)
    realized["stake_warning"] = ""
    realized.loc[realized["manual_non_production_stake_flag"], "stake_warning"] = "Non-production row has manual stake"
    bankroll = max(float(starting_bankroll), 0.0)
    max_slate = bankroll * 0.25
    max_pick = bankroll * 0.04
    if bankroll > 0:
        realized["Stake"] = realized["Stake"].clip(upper=max_pick)
        if realized["Stake"].sum() > max_slate:
            scale = max_slate / realized["Stake"].sum()
            realized["Stake"] = realized["Stake"] * scale
    realized["American Odds"] = pd.to_numeric(realized.get("American Odds", realized.get("odds_american", pd.NA)), errors="coerce")
    realized["Decimal Odds"] = pd.to_numeric(realized.get("Decimal Odds", realized.get("decimal_odds", pd.NA)), errors="coerce")

    missing_decimal = realized["Decimal Odds"].isna()
    realized.loc[missing_decimal, "Decimal Odds"] = realized.loc[missing_decimal, "American Odds"].map(_american_to_decimal)

    realized["Excluded Reason"] = ""
    score_missing = realized.get("actual_home_score", pd.Series([pd.NA] * len(realized))).isna() | realized.get("actual_away_score", pd.Series([pd.NA] * len(realized))).isna()
    realized.loc[realized["Result"] == "MISMATCH", "Excluded Reason"] = realized["grading_excluded_reason"]
    realized.loc[(~realized["export_run_match_flag"]) & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "Export run mismatch"
    realized.loc[(realized["Result"] == "UNGRADED") & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "UNGRADED"
    realized.loc[(score_missing) & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "MISSING_SCORES"
    realized.loc[(realized["Result"] == "WIN") & (realized["Decimal Odds"].isna()) & (realized["Excluded Reason"] == ""), "Excluded Reason"] = "MISSING_ODDS"

    core_eligible = realized["Excluded Reason"] == ""
    selected_statuses = get_strategy_mode_statuses(strategy_mode, custom_statuses)
    realized["Mode Included"] = realized["Status Bucket"].isin(selected_statuses)
    realized["Include In Totals"] = core_eligible & realized["Mode Included"]
    realized["production_pnl_included"] = core_eligible & realized["Status Bucket"].eq(STATUS_BUCKET_ACTIONABLE)
    realized["research_pnl_included"] = core_eligible & non_prod_mask

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
    production_summary = _summarize_subset(realized[realized["production_pnl_included"]])
    research_summary = _summarize_subset(realized[realized["research_pnl_included"]])
    all_rows_summary = _summarize_subset(realized[core_eligible])
    eligible_with_family = realized[realized["Excluded Reason"] == ""].assign(
        market_family=recap_profile["market_family"],
        market_direction=recap_profile["market_type"],
    )
    by_league_family = _group_summary(eligible_with_family, ["league", "market_family"])
    by_pick_status = _group_summary(realized[realized["Excluded Reason"] == ""], ["Status Bucket"])
    by_league = _group_summary(realized[realized["Excluded Reason"] == ""], ["league"])
    by_market_family = _group_summary(eligible_with_family, ["market_family"])

    # The performance guard groups by direction (market_type, e.g. total_over vs
    # total_under) rather than by market_family (total). Pooling Over and Under into one
    # "total" bucket let a bleeding Over bucket be masked by winning Unders, so a
    # direction that is structurally failing (6 Jun MLB Overs 0-4) never tripped the
    # guard while the Unders it was averaged with were winning. Direction granularity
    # lets the guard zero exposure to the failing side alone.
    by_league_direction = _group_summary(eligible_with_family, ["league", "market_direction"])
    guard_table = by_league_direction.copy()
    guard_table["performance_guard_warning"] = ""
    guard_table["performance_guard_flag"] = False
    guard_table["performance_guard_reason"] = ""
    guard_table["production_eligible"] = True
    if enable_performance_guard and not guard_table.empty:
        insufficient = guard_table["Bet Count"] < min_guard_sample_size
        guard_table.loc[insufficient, "performance_guard_warning"] = "Insufficient sample, monitor only"
        failing = (~insufficient) & ((guard_table["Hit Rate"] < performance_guard_min_win_rate) | (guard_table["ROI"] < performance_guard_min_roi))
        guard_table.loc[failing, "performance_guard_flag"] = True
        guard_table.loc[failing, "performance_guard_reason"] = "Trailing league+market family performance below configured thresholds."
        guard_table.loc[failing, "production_eligible"] = False
        if performance_guard_action == PERFORMANCE_GUARD_ACTION_ZERO_KELLY_OR_WATCHLIST:
            for _, row in guard_table[failing].iterrows():
                mask = (realized["league"] == row["league"]) & (recap_profile["market_type"] == row["market_direction"])
                if "Kelly_Bet_Size" in realized.columns:
                    realized.loc[mask, "Kelly_Bet_Size"] = 0.0
                realized.loc[mask, "production_eligible"] = False
                realized.loc[mask, "performance_guard_flag"] = True
                realized.loc[mask, "performance_guard_reason"] = row["performance_guard_reason"]
        warn_mask = insufficient
        for _, row in guard_table[warn_mask].iterrows():
            mask = (realized["league"] == row["league"]) & (recap_profile["market_type"] == row["market_direction"])
            realized.loc[mask, "performance_guard_warning"] = "Insufficient sample, monitor only"

    diagnostics = {
        "pick_mismatches": realized[realized["Result"] == "MISMATCH"].copy(),
        "missing_scores": realized[realized["Excluded Reason"] == "MISSING_SCORES"].copy(),
        "missing_odds": realized[realized["Excluded Reason"] == "MISSING_ODDS"].copy(),
        "excluded_rows": realized[realized["Excluded Reason"] != ""].copy(),
        "excluded_count": int((realized["Excluded Reason"] != "").sum()),
        "outside_mode_rows": realized[(realized["Excluded Reason"] == "") & (~realized["Mode Included"])].copy(),
        "status_bucket_summary": status_summary,
        "mode_comparison": mode_comparison,
        "league_market_family_summary": by_league_family,
        "league_market_direction_summary": by_league_direction,
        "status_summary": by_pick_status,
        "league_summary": by_league,
        "market_family_summary": by_market_family,
        "production_pnl_summary": production_summary,
        "research_pnl_summary": research_summary,
        "all_rows_pnl_summary": all_rows_summary,
        "performance_guard_summary": guard_table,
        "exact_matched_rows": int(realized["pick_identity_match"].sum()),
        "mismatched_rows": int((~realized["pick_identity_match"]).sum()),
        "excluded_stake_mismatch": float(realized.loc[~realized["pick_identity_match"], "Stake"].sum()),
        "mismatches_by_reason": realized.loc[~realized["pick_identity_match"], "grading_excluded_reason"].value_counts().to_dict(),
        "ungraded_count": int((realized["Result"] == "UNGRADED").sum()),
        "missing_odds_count": int((realized["Excluded Reason"] == "MISSING_ODDS").sum()),
        "mismatch_count": int((realized["Result"] == "MISMATCH").sum()),
        "outside_mode_count": int(((realized["Excluded Reason"] == "") & (~realized["Mode Included"])).sum()),
        "excluded_stake_non_production": float(realized.loc[non_prod_mask, "Stake"].sum()),
        "production_card_stake_total": float(realized.loc[realized["Status Bucket"] == STATUS_BUCKET_ACTIONABLE, "Stake"].sum()),
    }

    return realized, summary, diagnostics
