from datetime import date, timedelta

import pandas as pd
import streamlit as st
from app_core.results_fetcher import fetch_yesterdays_results
from app_core.results_ingestion import attach_results


def determine_display_outcome(row):
    # 1. Respect explicit manual dropdown overrides
    if 'Outcome' in row and pd.notna(row['Outcome']) and str(row['Outcome']) != 'N/A':
        return row['Outcome']

    pick = str(row.get('best_pick', '')).lower().strip()
    if not pick:
         return 'N/A'

    # Unresolved-line placeholders (e.g. "Total line unresolved") carry no
    # gradeable side/line — they must stay N/A, never WIN/LOSS. The 10 Jun
    # recap graded one as LOSS, polluting downstream calibration data.
    if 'unresolved' in pick:
         return 'N/A'

    # 2. Auto-grade based on scores if they exist in the grid
    h_score = row.get('actual_home_score')
    a_score = row.get('actual_away_score')

    if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
        try:
            h = float(h_score)
            a = float(a_score)

            # A 0-0 final does not exist in MLB/NBA/NHL — it means the game was
            # postponed or not played. Grading it WIN/LOSS poisons the
            # calibration data (6 Jun NYY/BOS and 11 Jun CWS/ATL were both
            # graded LOSS at 0-0). Void instead.
            if h == 0 and a == 0:
                return 'N/A'

            # Evaluate TOTALS (Over/Under)
            if 'over' in pick or 'under' in pick:
                import re
                m = re.search(r'(over|under)\s*(\d+\.?\d*)', pick, re.IGNORECASE)
                if m:
                    side = m.group(1).lower()
                    line = float(m.group(2))
                    total = h + a
                    if side == 'over':
                        return 'WIN' if total > line else ('LOSS' if total < line else 'PUSH')
                    elif side == 'under':
                        return 'WIN' if total < line else ('LOSS' if total > line else 'PUSH')

            # Evaluate SPREADS (+/-)
            elif '+' in pick or '-' in pick:
                import re
                from difflib import SequenceMatcher
                m = re.search(r'([+-]\d+\.?\d*)\s*(?:\(.*\))?$', pick)
                if m:
                    line = float(m.group(1))
                    pick_team = pick[:m.start()].strip()
                    # Safely get team names checking both 'home_team'/'away_team' and 'Home'/'Away'
                    home_team = str(row.get('home_team', row.get('Home', ''))).lower()
                    away_team = str(row.get('away_team', row.get('Away', ''))).lower()

                    home_ratio = SequenceMatcher(None, pick_team, home_team).ratio() if home_team else 0.0
                    away_ratio = SequenceMatcher(None, pick_team, away_team).ratio() if away_team else 0.0

                    is_home = (home_ratio > away_ratio) and home_ratio > 0.6
                    is_away = (away_ratio > home_ratio) and away_ratio > 0.6

                    if is_home:
                        margin = h - a
                        return 'WIN' if margin + line > 0 else ('LOSS' if margin + line < 0 else 'PUSH')
                    elif is_away:
                        margin = a - h
                        return 'WIN' if margin + line > 0 else ('LOSS' if margin + line < 0 else 'PUSH')

            # Evaluate MONEYLINE
            else:
                from difflib import SequenceMatcher
                pick_team = pick.replace(' ml', '').strip()
                # Safely get team names checking both 'home_team'/'away_team' and 'Home'/'Away'
                home_team = str(row.get('home_team', row.get('Home', ''))).lower()
                away_team = str(row.get('away_team', row.get('Away', ''))).lower()

                home_ratio = SequenceMatcher(None, pick_team, home_team).ratio() if home_team else 0.0
                away_ratio = SequenceMatcher(None, pick_team, away_team).ratio() if away_team else 0.0

                is_home = (home_ratio > away_ratio) and home_ratio > 0.6
                is_away = (away_ratio > home_ratio) and away_ratio > 0.6

                if is_home:
                    return 'WIN' if h > a else ('LOSS' if h < a else 'PUSH')
                elif is_away:
                    return 'WIN' if a > h else ('LOSS' if a < h else 'PUSH')

        except Exception:
            pass # If math fails, drop down to the fallback

    # 3. Fallback to ingestion logic (if no manual scores are entered yet)
    if 'Pick_Outcome' in row and pd.notna(row['Pick_Outcome']) and str(row['Pick_Outcome']) != 'N/A':
        return row['Pick_Outcome']

    if 'over' in pick or 'under' in pick:
         return row.get('total_result', 'N/A')
    elif '+' in pick or '-' in pick:
         return row.get('spread_result', 'N/A')
    else:
         return row.get('ml_result', 'N/A')


def funded_prop_rows(card: pd.DataFrame) -> pd.DataFrame:
    """Return only player props the production card actually funded."""

    if card is None or card.empty:
        return pd.DataFrame(columns=[] if card is None else card.columns)

    out = card.copy()
    if "Stake_Status" in out.columns:
        stake_status = out["Stake_Status"].fillna("").astype(str).str.strip()
        if stake_status.ne("").any():
            return out[stake_status.str.casefold().eq("funded")].copy()

    pick_status = out.get("Pick_Status", pd.Series("", index=out.index))
    status_mask = pick_status.fillna("").astype(str).str.strip().str.casefold().eq("actionable")
    stake = pd.to_numeric(out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    return out[status_mask & stake.gt(0)].copy()


def summarize_prop_results(results: pd.DataFrame) -> dict:
    """Summarize funded prop results without counting unresolved rows."""

    empty = {
        "wins": 0, "losses": 0, "pushes": 0, "unresolved": 0,
        "staked": 0.0, "pnl": 0.0, "roi": 0.0, "win_rate": 0.0,
    }
    if results is None or results.empty or "result" not in results.columns:
        return empty

    outcome = results["result"].fillna("").astype(str).str.upper()
    settled_mask = outcome.isin(["WIN", "LOSS", "PUSH"])
    settled = results[settled_mask]
    settled_outcome = outcome[settled_mask]
    stake = pd.to_numeric(settled.get("stake", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)
    profit = pd.to_numeric(settled.get("profit", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)

    wins = int(settled_outcome.eq("WIN").sum())
    losses = int(settled_outcome.eq("LOSS").sum())
    pushes = int(settled_outcome.eq("PUSH").sum())
    decisions = wins + losses
    staked = float(stake.sum())
    pnl = float(profit.sum())
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "unresolved": int((~settled_mask).sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": (pnl / staked) if staked else 0.0,
        "win_rate": (wins / decisions) if decisions else 0.0,
    }


def _render_prop_results_recap() -> None:
    """Upload, grade, and recap the funded MLB player-prop card."""

    st.markdown("#### Player Prop Performance")
    uploaded = st.file_uploader(
        "Upload Yesterday's MLB Player Props Export",
        type=["csv"],
        key="perf_props_uploader",
    )
    if uploaded is None:
        st.caption("Upload the player-props export to grade only the bets marked Funded.")
        return

    file_identifier = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("perf_props_file_identifier") != file_identifier:
        st.session_state["perf_props_file_identifier"] = file_identifier
        st.session_state.pop("perf_props_results", None)

    try:
        uploaded.seek(0)
        card = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Error reading player-props export: {exc}")
        return

    funded = funded_prop_rows(card)
    st.caption(f"{len(funded)} funded prop(s) will be graded; unfunded research rows are excluded.")
    game_date = st.date_input(
        "Player-prop slate date",
        value=date.today() - timedelta(days=1),
        key="perf_props_date",
    )

    if st.button("Fetch MLB Player Prop Results", key="perf_props_fetch", disabled=funded.empty):
        with st.spinner("Fetching MLB player results and grading the funded card..."):
            try:
                from scripts.grade_props import (
                    _build_name_to_id,
                    fetch_actual_batter,
                    fetch_actual_ks,
                    grade_card,
                )

                date_text = game_date.isoformat()
                season = game_date.year
                name_to_id = _build_name_to_id(date_text, funded)
                rows = grade_card(
                    funded,
                    date_text,
                    name_to_id,
                    lambda player_id, participant_type: (
                        fetch_actual_batter(player_id, date_text, season)
                        if participant_type == "batter"
                        else fetch_actual_ks(player_id, date_text, season)
                    ),
                )
                st.session_state["perf_props_results"] = pd.DataFrame(rows)
            except Exception as exc:
                st.error(f"Error fetching or grading player props: {exc}")

    results = st.session_state.get("perf_props_results")
    if not isinstance(results, pd.DataFrame) or results.empty:
        if funded.empty:
            st.info("This export contains no funded player props.")
        return

    summary = summarize_prop_results(results)
    cols = st.columns(4)
    cols[0].metric("Funded Prop Record", f"{summary['wins']}-{summary['losses']}-{summary['pushes']}")
    cols[1].metric("Funded Prop Win Rate", f"{summary['win_rate']:.1%}")
    cols[2].metric("Funded Prop P&L", f"${summary['pnl']:+.2f}")
    cols[3].metric("Funded Prop ROI", f"{summary['roi']:+.1%}", f"${summary['staked']:.2f} staked")

    if summary["unresolved"]:
        st.warning(
            f"{summary['unresolved']} funded prop(s) could not be resolved from MLB StatsAPI "
            "and are excluded from the record and ROI."
        )

    if "market_type" in results.columns and "result" in results.columns:
        market_recap = pd.crosstab(
            results["market_type"].fillna("unknown"),
            results["result"].fillna("UNRESOLVED"),
        ).reset_index()
        st.markdown("##### Results by Prop Market")
        st.dataframe(market_recap, width="stretch", hide_index=True)

    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download Player Prop Performance Recap",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="player_prop_performance_recap.csv",
        mime="text/csv",
        key="perf_props_download",
    )


def render_results_dashboard(picks_df: pd.DataFrame) -> None:
    # 1. File Uploader for Yesterday's Picks at the very top
    uploaded_picks_file = st.file_uploader("Upload Yesterday's Best Picks Export", type=["csv"], key="perf_picks_uploader")

    st.subheader("Prior Day Performance")

    _render_prop_results_recap()
    st.divider()
    st.markdown("#### Game Performance")

    restricted_leagues = st.session_state.get("restricted_leagues", set())
    if restricted_leagues:
        league_str = ", ".join(sorted(list(restricted_leagues)))
        st.warning(f"Results for [{league_str}] are currently unavailable due to API plan restrictions. These picks must be verified manually.")

    # Re-evaluate the source data based on the explicit uploader first
    new_upload_detected = False
    if uploaded_picks_file is not None:
        file_identifier = f"{uploaded_picks_file.name}_{uploaded_picks_file.size}"
        if st.session_state.get("perf_picks_file_identifier") != file_identifier:
            new_upload_detected = True
            st.session_state["perf_picks_file_identifier"] = file_identifier

        try:
            # We must reset the file pointer to 0 because the file might have been read before
            uploaded_picks_file.seek(0)
            picks_df = pd.read_csv(uploaded_picks_file)

            # Map common pretty/export columns to canonical names
            if "League" in picks_df.columns and "league" not in picks_df.columns:
                picks_df["league"] = picks_df["League"]
            if "Home Team" in picks_df.columns and "home_team" not in picks_df.columns:
                picks_df["home_team"] = picks_df["Home Team"]
            elif "Home" in picks_df.columns and "home_team" not in picks_df.columns:
                picks_df["home_team"] = picks_df["Home"]
            if "Away Team" in picks_df.columns and "away_team" not in picks_df.columns:
                picks_df["away_team"] = picks_df["Away Team"]
            elif "Away" in picks_df.columns and "away_team" not in picks_df.columns:
                picks_df["away_team"] = picks_df["Away"]

            if "Pick Taken" in picks_df.columns and "best_pick" not in picks_df.columns:
                picks_df["best_pick"] = picks_df["Pick Taken"]
            elif "Best Pick" in picks_df.columns and "best_pick" not in picks_df.columns:
                picks_df["best_pick"] = picks_df["Best Pick"]

            st.success("Successfully loaded uploaded picks.")
        except Exception as e:
            st.error(f"Error reading uploaded picks file: {e}")
            picks_df = None

    if picks_df is None or picks_df.empty:
        st.info("No picks data available for the previous day. Please upload a file above.")
        return

    # Initialize editable picks in session state if not present, OR if a new file was uploaded
    if "perf_edited_picks" not in st.session_state or new_upload_detected:
        st.session_state["perf_edited_picks"] = picks_df.copy()

    # Get the latest dataframe state (could be edited previously)
    current_df = st.session_state["perf_edited_picks"]

    # 2. Toggle to fetch scores from API
    fetch_api = st.toggle("Fetch Scores from API", value=False, key="perf_api_toggle")

    # 3. File Uploader for Manual Results
    uploaded_manual_results = st.file_uploader("Upload Manual Results CSV (If API restricted)", type=["csv"], key="perf_manual_results_uploader")

    has_results = False

    if uploaded_manual_results is not None:
        try:
            uploaded_manual_results.seek(0)
            manual_results_df = pd.read_csv(uploaded_manual_results)
            current_df = attach_results(current_df, manual_results_df)
            st.session_state["perf_edited_picks"] = current_df
            st.success("Successfully applied manual results.")
            has_results = True
        except Exception as e:
            st.error(f"Error reading manual results file: {e}")
    elif fetch_api:
        with st.spinner("Fetching yesterday's results from API..."):
            leagues_in_picks = current_df['league'].unique().tolist() if 'league' in current_df.columns else []
            # Filter out restricted leagues
            restricted = st.session_state.get("restricted_leagues", set())
            allowed_leagues = [l for l in leagues_in_picks if l not in restricted]

            if allowed_leagues:
                try:
                    fetched_results_df = fetch_yesterdays_results(allowed_leagues)
                    if not fetched_results_df.empty:
                        current_df = attach_results(current_df, fetched_results_df)
                        st.session_state["perf_edited_picks"] = current_df
                        st.success("Successfully fetched and applied API results.")
                        has_results = True
                    else:
                        st.warning("No results found from API for the given leagues.")
                        current_df = attach_results(current_df, pd.DataFrame()) # Add empty columns
                        st.session_state["perf_edited_picks"] = current_df
                except Exception as e:
                    st.error(f"Error fetching results from API: {e}")
            else:
                st.warning("No allowed leagues found in picks to fetch results for (or all are restricted).")

    # We consider it having results if either Pick_Outcome is present or actual_home_score is present
    if not has_results and ('Pick_Outcome' not in current_df.columns and 'actual_home_score' not in current_df.columns):
         st.warning("Results have not been attached to these picks yet. Please upload manual results, fetch from API, or edit manually below.")

    # Process all rows for the display table first so we have the Outcome logic for the top metrics
    display_df = current_df.copy()

    # Ensure actual scores exist as columns for the editor
    for col in ['actual_home_score', 'actual_away_score', 'Outcome']:
         if col not in display_df.columns:
              display_df[col] = pd.NA

    display_df['Outcome'] = display_df.apply(determine_display_outcome, axis=1)

    def calculate_metrics(df):
        wins = len(df[df['Outcome'] == 'WIN'])
        losses = len(df[df['Outcome'] == 'LOSS'])
        pushes = len(df[df['Outcome'] == 'PUSH'])

        total_decisions = wins + losses
        win_rate = (wins / total_decisions) if total_decisions > 0 else 0.0

        # Calculate Net Profit based on odds_american assuming 1 unit bet
        net_profit = 0.0
        for idx, row in df.iterrows():
            outcome = row['Outcome']
            if outcome == 'WIN':
                 odds = pd.to_numeric(row.get('odds_american'), errors='coerce')
                 if pd.isna(odds):
                      odds = -110 # default

                 if abs(odds) < 1.0:
                      odds = -110

                 # Calculate profit for 1 unit (e.g. $100 bet)
                 if odds < 0:
                      profit = 1.0 / (abs(odds) / 100.0)
                 else:
                      profit = odds / 100.0
                 net_profit += profit
            elif outcome == 'LOSS':
                 net_profit -= 1.0

        return wins, losses, pushes, win_rate, net_profit

    # Owner bets every valid game row, so the all-row settled record is the
    # production headline. N/A/postponed rows are excluded from the denominator.
    wins_all, losses_all, pushes_all, win_rate_all, net_profit_all = calculate_metrics(display_df)
    settled_all = wins_all + losses_all + pushes_all
    st.markdown("#### All-Row Betting Performance")
    _all_cols = st.columns(4)
    _all_cols[0].metric(
        "Settled Record",
        f"{wins_all}-{losses_all}-{pushes_all}",
    )
    _all_cols[1].metric("All-Row Win Rate", f"{win_rate_all:.1%}")
    _all_cols[2].metric("Estimated Flat-Bet P&L (Units)", f"{net_profit_all:+.2f}")
    _all_cols[3].metric("Rows Settled", f"{settled_all}/{len(display_df)}")
    st.caption(
        "This is the headline because every valid game row is bet. N/A, postponed, "
        "started, and unresolved-line rows do not count as settled decisions. P&L uses "
        "the exported odds when present and -110 only as a fallback when odds are missing."
    )

    # Keep strict production-edge tiers as a diagnostic beneath the owner's
    # all-row headline. They show where the model had genuine priced edge versus
    # where the full-board policy forced the best available valid selection.
    from app_core.strategy_lab_realized import summarize_recap_tiers
    _tier_src = display_df.rename(columns={"Pick_Status": "Status"}) if "Pick_Status" in display_df.columns else display_df
    tiers = summarize_recap_tiers(_tier_src)
    st.markdown("#### Staked Performance by Tier")
    _tcols = st.columns(3)
    for _col, (_, _row) in zip(_tcols, tiers.iterrows()):
        _col.metric(
            _row["Tier"],
            f"{_row['Hit Rate']:.1%}",
            f"{int(_row['Wins'])}-{int(_row['Losses'])} ({int(_row['Total'])})",
        )
    st.caption(
        "Judge the system by the staked tiers (Actionable / Actionable + High Variance). "
        "'All graded rows' includes Below Threshold and No Play picks that were never "
        "staked, so it trends toward ~50% by construction and understates a good staked day."
    )

    # Net profit on the staked card (Actionable or High Variance with a positive stake).
    STAKED_STATUSES = {"Actionable", "High Variance/Speculative"}
    if "Pick_Status" in display_df.columns:
        _staked_mask = display_df["Pick_Status"].astype(str).isin(STAKED_STATUSES)
        if "Kelly_Bet_Size" in display_df.columns:
            _staked_mask &= pd.to_numeric(display_df["Kelly_Bet_Size"], errors="coerce").fillna(0) > 0
        staked_df = display_df[_staked_mask].copy()
    else:
        staked_df = pd.DataFrame()

    if staked_df.empty:
        st.info(
            "No rows cleared the strict production-edge stake gate. The all-row "
            "best-available record above still represents the owner's placed bets."
        )
    else:
        wins_s, losses_s, pushes_s, win_rate_s, net_profit_s = calculate_metrics(staked_df)
        scol1, scol2 = st.columns(2)
        scol1.metric("Net Profit - Staked (Units)", f"{net_profit_s:+.2f}")
        scol2.metric("Staked Picks Evaluated", len(staked_df))

    st.markdown("###### Production-Edge Reference")
    st.caption(
        "The tier metrics above remain useful for comparing strict modeled edges with "
        "forced best-available rows, but they no longer replace the all-row headline."
    )

    st.divider()

    # Ensure team columns map correctly if the export used capitalized 'Home'/'Away'
    if 'Home' in display_df.columns and 'home_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Home': 'home_team'})
    if 'Away' in display_df.columns and 'away_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Away': 'away_team'})
    
    # Select columns to show. export_run_id is carried through so a downloaded
    # recap stays traceable to the pipeline run that produced the card —
    # scripts/grade_slate.py warns when a recap's run id doesn't match the
    # export being graded (11 Jun: a recap built from a stale morning card
    # graded lines the final card never played).
    cols_to_show = ['league', 'home_team', 'away_team', 'best_pick', 'actual_home_score', 'actual_away_score', 'Outcome', 'Pick_Status', 'export_run_id']
    # Filter to only existing columns
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]

    # Rename columns for presentation
    rename_map = {
         'league': 'League',
         'home_team': 'Home',
         'away_team': 'Away',
         'best_pick': 'Pick Taken',
         'Pick_Status': 'Status'
    }

    table_df = display_df[cols_to_show].rename(columns=rename_map)

    # Render st.data_editor
    edited_df = st.data_editor(
        table_df,
        width="stretch",
        disabled=["League", "Home", "Away", "Pick Taken", "Status", "export_run_id"],
        column_config={
            "actual_home_score": st.column_config.NumberColumn(
                "Home Score",
                min_value=0,
                step=1,
                format="%d",
                required=False
            ),
            "actual_away_score": st.column_config.NumberColumn(
                "Away Score",
                min_value=0,
                step=1,
                format="%d",
                required=False
            ),
            "Outcome": st.column_config.SelectboxColumn(
                "Outcome",
                options=["WIN", "LOSS", "PUSH", "N/A"],
                required=True
            )
        },
        key="perf_data_editor"
    )

    # Check for changes and update session state
    if not edited_df.equals(table_df):
        # We need to map edited columns back to our session state current_df
        # Create mapping of presentation names to original names
        reverse_rename_map = {v: k for k, v in rename_map.items()}
        for col in edited_df.columns:
            orig_col = reverse_rename_map.get(col, col)
            current_df[orig_col] = edited_df[col]

        # Save back to session state
        st.session_state["perf_edited_picks"] = current_df

        # Trigger a rerun so metrics update immediately
        st.rerun()
