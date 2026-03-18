import re
import sys

file_path = "core/streamlit_pipeline.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. First, insert _expand_live_odds_to_bet_rows before run_analysis_pipeline
if "_expand_live_odds_to_bet_rows" not in content:
    new_function = """
def _expand_live_odds_to_bet_rows(live_odds_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Expands the wide live_odds_df (1 row per game) into 4 market rows per game
    (spread_home, spread_away, total_over, total_under).
    \"\"\"
    if live_odds_df is None or live_odds_df.empty:
        return pd.DataFrame()

    out_rows = []

    # Required identity columns
    id_cols = ["league", "home_team", "away_team", "game_date", "matchup_id"]
    # Check for game_time_est if exists
    if "game_time_est" in live_odds_df.columns:
        id_cols.append("game_time_est")

    for _, row in live_odds_df.iterrows():
        base_dict = {col: row.get(col) for col in id_cols}

        # We need to map novig prices and points to the expanded rows
        market_mappings = [
            ("spread_home", "novig_home_price", "novig_home_point", "odds_source_spread"),
            ("spread_away", "novig_away_price", "novig_away_point", "odds_source_spread"),
            ("total_over", "novig_over_price", "novig_over_point", "odds_source_total"),
            ("total_under", "novig_under_price", "novig_under_point", "odds_source_total")
        ]

        for market_type, price_col, point_col, source_col in market_mappings:
            market_dict = base_dict.copy()
            market_dict["market_type"] = market_type

            # Map pricing
            price_val = pd.to_numeric(row.get(price_col), errors="coerce")
            if pd.isna(price_val):
                market_dict["odds_american"] = -110.0
                market_dict["odds_source"] = "fallback_novig"
            else:
                market_dict["odds_american"] = float(price_val)
                market_dict["odds_source"] = "odds_api" # As requested by instructions, setting to odds_api

            # Map lines based on market type
            point_val = pd.to_numeric(row.get(point_col), errors="coerce")
            if market_type in ["spread_home", "spread_away"]:
                market_dict["spread_line"] = float(point_val) if pd.notna(point_val) else pd.NA
                market_dict["total_line"] = pd.NA
            else:
                market_dict["spread_line"] = pd.NA
                market_dict["total_line"] = float(point_val) if pd.notna(point_val) else pd.NA

            out_rows.append(market_dict)

    expanded_df = pd.DataFrame(out_rows)
    return expanded_df

def run_analysis_pipeline(
"""
    content = content.replace("def run_analysis_pipeline(", new_function)


# 2. Now replace the massive body of run_analysis_pipeline up to the `merged["odds_american"] = _numeric_series...` part
pattern = re.compile(r'def run_analysis_pipeline\(.*?merged\["odds_american"\] = _numeric_series\(merged, "odds_american", pd\.NA\)', re.DOTALL)

new_code = """def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:

    # 1. Expand TheOdds API into the Master Slate
    live_odds_df = fetch_live_odds_dataframe(sports)

    if not live_odds_df.empty:
        live_odds_df = _normalize_identity_strings(live_odds_df, ["league", "home_team", "away_team"])
        live_odds_df["league"] = _string_series(live_odds_df, "league").str.upper().replace(LEAGUE_ALIASES)
        live_odds_df["home_team"] = _string_series(live_odds_df, "home_team").map(normalize_team_name)
        live_odds_df["away_team"] = _string_series(live_odds_df, "away_team").map(normalize_team_name)

        fallback_day = _et_day_string(_game_dates(live_odds_df))
        today_et_day = pd.Series([_game_date_fallback().strftime("%Y-%m-%d")] * len(live_odds_df), index=live_odds_df.index, dtype="string")
        live_odds_df["game_date"] = _et_day_string(live_odds_df.get("game_date", pd.Series([pd.NA] * len(live_odds_df), index=live_odds_df.index)))
        live_odds_df["game_date"] = live_odds_df["game_date"].fillna(fallback_day).fillna(today_et_day)
        live_odds_df["matchup_id"] = _matchup_id(live_odds_df)

    master_slate = _expand_live_odds_to_bet_rows(live_odds_df)
    if master_slate.empty:
        logger.warning("Master slate is empty after odds expansion. Falling back to an empty DataFrame.")
        master_slate = pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "matchup_id", "market_type", "odds_american", "odds_source"])

    # 2. Build the enrichment frame (TheOver)
    theover_rows = build_theover_bet_rows(spreads_df, totals_df, sports)

    raw_base_df = load_base_data()
    odds_schedule_loaded = not raw_base_df.empty

    # Ensure identity string dtypes on theover_rows
    theover_rows = _infer_missing_league_from_team_sets(theover_rows, sports)
    theover_rows = _restore_missing_ncaab_league_priority(theover_rows)
    theover_rows = _recover_ncaab_league_labels(theover_rows)
    theover_rows = _enforce_identity_string_dtype(theover_rows, ["league", "home_team", "away_team"])
    theover_rows = _preprocess_bet_rows_for_league_bridge(theover_rows)
    theover_rows = _normalize_identity_strings(theover_rows, ["league", "home_team", "away_team"])

    stale = is_stale_schedule(raw_base_df, theover_rows)
    base_df = raw_base_df.copy()
    stale_base_rows_removed = 0

    theover_rows = _enforce_identity_string_dtype(theover_rows, ["league", "home_team", "away_team"])
    theover_rows = _normalize_identity_strings(theover_rows, ["league", "home_team", "away_team"])
    theover_rows["league"] = _string_series(theover_rows, "league").str.upper().replace(LEAGUE_ALIASES)
    theover_rows["home_team"] = _string_series(theover_rows, "home_team").map(normalize_team_name)
    theover_rows["away_team"] = _string_series(theover_rows, "away_team").map(normalize_team_name)
    theover_rows["game_date"] = _et_day_string(_game_dates(theover_rows))
    theover_rows["matchup_id"] = _matchup_id(theover_rows)

    if not theover_rows.empty and not base_df.empty:
        base_dates = base_df.copy()
        base_dates = _normalize_identity_strings(base_dates, ["league", "home_team", "away_team"])
        base_dates["league"] = _string_series(base_dates, "league").str.upper().replace(LEAGUE_ALIASES)
        base_dates["home_team"] = _string_series(base_dates, "home_team").map(normalize_team_name)
        base_dates["away_team"] = _string_series(base_dates, "away_team").map(normalize_team_name)
        base_dates["date"] = _et_day_string(_game_dates(base_dates))
        base_dates["matchup_id"] = _matchup_id(base_dates)

        date_lookup = base_dates[["league", "matchup_id", "date"]].drop_duplicates(["league", "matchup_id"])
        merged_dates = theover_rows.merge(date_lookup, on=["league", "matchup_id"], how="left")
        theover_rows["game_date"] = theover_rows["game_date"].fillna(merged_dates["date"])

    theover_rows, date_stats = _fill_missing_game_dates_from_base(theover_rows, base_df)
    theover_rows = _dedupe_inverted_matchups(theover_rows)

    merge_keys = ["league", "home_team", "away_team", "game_date", "fuzzy_team_match>=85"]

    # Primary ingestion baseline: master_slate (from Odds API) is the master slate frame.
    merged = master_slate.copy()

    # 3. Invert the Merge (Odds API is Base, TheOver is Enrichment)
    if not theover_rows.empty and not merged.empty:
        # Standardize both sides of the merge to ET day boundaries before join.
        merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.floor("D")
        theover_rows["game_date"] = pd.to_datetime(theover_rows["game_date"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.floor("D")
        fallback_merge_day = pd.Timestamp.now(tz="America/New_York").floor("D")
        merged["game_date"] = merged["game_date"].fillna(fallback_merge_day)
        theover_rows["game_date"] = theover_rows["game_date"].fillna(fallback_merge_day)

        # Merge theover enrichment columns
        theover_cols_to_merge = ["matchup_id", "market_type", "theover_probability", "ml_probability"]
        # Only merge columns that exist
        theover_cols_to_merge = [c for c in theover_cols_to_merge if c in theover_rows.columns]

        merged = merged.merge(
            theover_rows[theover_cols_to_merge].drop_duplicates(["matchup_id", "market_type"]),
            on=["matchup_id", "market_type"],
            how="left"
        )

        # 4. Fuzzy Matching Fallback
        # After the strict join, identify any rows in master_slate where theover_probability is still NaN.
        # Ensure we only try fuzzy match if theover_rows has probability columns
        if "theover_probability" in theover_rows.columns or "ml_probability" in theover_rows.columns:
            needs_fuzzy = pd.Series([False]*len(merged), index=merged.index)
            if "theover_probability" in merged.columns:
                needs_fuzzy = needs_fuzzy | merged["theover_probability"].isna()
            if "ml_probability" in merged.columns:
                needs_fuzzy = needs_fuzzy | merged["ml_probability"].isna()

            if needs_fuzzy.any():
                logger.info(f"Attempting fuzzy match for {needs_fuzzy.sum()} rows missing enrichment from TheOver.")
                theover_schedule = theover_rows.drop_duplicates(["league", "home_team", "away_team", "market_type"])

                for idx in merged.index[needs_fuzzy]:
                    row_market = merged.at[idx, "market_type"]
                    # We need to filter theover_schedule to the same market type
                    market_schedule = theover_schedule[theover_schedule["market_type"] == row_market]

                    if market_schedule.empty:
                        continue

                    match = _fuzzy_match_schedule_row(merged.loc[idx], market_schedule, threshold=85)
                    if match.empty:
                        continue

                    # Patch missing columns
                    if "theover_probability" in merged.columns and pd.isna(merged.at[idx, "theover_probability"]) and pd.notna(match.get("theover_probability")):
                        merged.at[idx, "theover_probability"] = match.get("theover_probability")

                    if "ml_probability" in merged.columns and pd.isna(merged.at[idx, "ml_probability"]) and pd.notna(match.get("ml_probability")):
                        merged.at[idx, "ml_probability"] = match.get("ml_probability")

    # Ensure identity columns survive master-frame merges.
    if "league" not in merged.columns or _string_series(merged, "league").str.len().eq(0).all():
        if not theover_rows.empty and "league" in theover_rows.columns and "matchup_id" in merged.columns and "game_date" in merged.columns:
            league_lookup = (
                theover_rows[[c for c in ["matchup_id", "game_date", "league"] if c in theover_rows.columns]]
                .dropna(subset=["matchup_id", "game_date"])
                .drop_duplicates(["matchup_id", "game_date"], keep="last")
            )
            if not league_lookup.empty:
                merged = merged.merge(
                    league_lookup.rename(columns={"league": "league_from_bets"}),
                    on=["matchup_id", "game_date"],
                    how="left",
                )
                merged["league"] = _string_series(merged, "league").where(
                    _string_series(merged, "league").str.len().gt(0),
                    _string_series(merged, "league_from_bets"),
                )
                merged = merged.drop(columns=["league_from_bets"], errors="ignore")
    if "league" not in merged.columns:
        merged["league"] = ""
    merged["league"] = _string_series(merged, "league").str.upper().replace(LEAGUE_ALIASES)

    # 5. Eliminate the Fallback Artifacts
    # The clunky np.select code and novig_home_price checks have been completely removed from here
    # since we already mapped odds_american properly during the expand step!

    # Do not set a fallback odds_source
    if "odds_source" not in merged.columns:
        merged["odds_source"] = pd.NA

    uploaded_odds = _numeric_series(merged, "odds_american")
    merged.loc[uploaded_odds.notna() & (uploaded_odds != -110), "odds_source"] = "uploaded"

    if not base_df.empty:
        base_schedule = base_df.copy()
        base_schedule["league"] = _string_series(base_schedule, "league").str.upper().replace(LEAGUE_ALIASES)
        base_schedule["home_team"] = _string_series(base_schedule, "home_team").map(normalize_team_name)
        base_schedule["away_team"] = _string_series(base_schedule, "away_team").map(normalize_team_name)
        base_schedule["date"] = _force_utc_datetime(_game_dates(base_schedule))
        base_schedule["merge_date_utc"] = _et_day_string(base_schedule["date"])
        # Backward-compat safety key: older merge paths referenced game_date_key directly.
        # Keep it aligned with merge_date_utc to prevent KeyError in mixed/stale runtime code paths.
        base_schedule["game_date_key"] = base_schedule["merge_date_utc"]

        base_schedule["home_team_lower"] = clean_team_name(base_schedule["home_team"])
        base_schedule["away_team_lower"] = clean_team_name(base_schedule["away_team"])
        base_schedule["matchup_key"] = _canonical_matchup_teams_key(base_schedule)
        base_schedule["matchup_id"] = _matchup_id(base_schedule)
        base_schedule["date_day"] = _date_join_key(base_schedule["date"])

        base_merge_columns = ["league", "matchup_id", "merge_date_utc"] + [
            col for col in ["date", "game_time_est", "is_neutral"]
            if col in base_schedule.columns
        ]

        merged["home_team_lower"] = clean_team_name(_string_series(merged, "home_team").map(normalize_team_name))
        merged["away_team_lower"] = clean_team_name(_string_series(merged, "away_team").map(normalize_team_name))
        merged["matchup_key"] = _canonical_matchup_teams_key(merged)
        merged["matchup_id"] = _matchup_id(merged)
        merged["merge_date_utc"] = _et_day_string(merged.get("game_date"))

        # Primary join uses explicit UTC day keys and canonical matchup keys (order-insensitive).
        merged = merged.merge(
            base_schedule[base_merge_columns].drop_duplicates(["league", "matchup_id", "merge_date_utc"]),
            on=["league", "matchup_id", "merge_date_utc"],
            how="left",
            suffixes=("", "_base"),
        )

        merged["game_date"] = _game_dates(merged)
        merged["game_date"] = merged["game_date"].fillna(merged["date"])

        if "game_time_est_base" in merged.columns:
            merged["game_time_est"] = _string_series(merged, "game_time_est").where(
                _string_series(merged, "game_time_est").str.len().gt(0),
                _string_series(merged, "game_time_est_base"),
            )
            merged = merged.drop(columns=["game_time_est_base"])

        if "is_neutral_base" in merged.columns:
            merged["is_neutral"] = merged["is_neutral"].fillna(merged["is_neutral_base"]) if "is_neutral" in merged.columns else merged["is_neutral_base"]
            merged = merged.drop(columns=["is_neutral_base"])

        if "merge_date_utc" in merged.columns:
            merged = merged.drop(columns=["merge_date_utc"])

    logger.info(f"Number of live Novig games fetched: {len(live_odds_df)}")

    # We still need to ensure odds_american and odds_source are correctly typed.
    merged["odds_american"] = _numeric_series(merged, "odds_american", pd.NA)"""

content, num_subs = pattern.subn(new_code, content, count=1)
if num_subs == 0:
    print("WARNING: Regex pattern failed to match.")
    sys.exit(1)

with open(file_path, "w") as f:
    f.write(content)

print("Final clean block patched into core/streamlit_pipeline.py.")
