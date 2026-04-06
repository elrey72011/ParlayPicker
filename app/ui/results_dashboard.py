import pandas as pd
import streamlit as st
from app_core.results_fetcher import fetch_yesterdays_results
from app_core.results_ingestion import attach_results

def render_results_dashboard(picks_df: pd.DataFrame) -> None:
    # 1. File Uploader for Yesterday's Picks at the very top
    uploaded_picks_file = st.file_uploader("Upload Yesterday's Best Picks Export", type=["csv"], key="perf_picks_uploader")

    st.subheader("Prior Day Performance")

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

    def determine_display_outcome(row):
        # 1. Respect explicit manual dropdown overrides
        if 'Outcome' in row and pd.notna(row['Outcome']) and str(row['Outcome']) != 'N/A':
            return row['Outcome']
            
        pick = str(row.get('best_pick', '')).lower().strip()
        if not pick:
             return 'N/A'
             
        # 2. Auto-grade based on scores if they exist in the grid
        h_score = row.get('actual_home_score')
        a_score = row.get('actual_away_score')
        
        if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
            try:
                h = float(h_score)
                a = float(a_score)
                
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

    # Only evaluate "Actionable" picks for top-line metrics based on current display_df
    actionable_df = display_df[display_df.get('Pick_Status', '') == 'Actionable'].copy()

    if actionable_df.empty:
         st.info("No 'Actionable' picks were found in yesterday's export.")
    else:
        wins_act, losses_act, pushes_act, win_rate_act, net_profit_act = calculate_metrics(actionable_df)

        # Assume 1 unit = 1 for the metric, maybe display in Units or $ assuming $100/u
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Win Rate (Actionable)", f"{win_rate_act:.1%}", f"{wins_act}-{losses_act}-{pushes_act}")
        col2.metric("Total Net Profit - Actionable (Units)", f"{net_profit_act:+.2f}")
        col3.metric("Actionable Picks Evaluated", len(actionable_df))

    # Calculate metrics for all picks in display_df
    wins_all, losses_all, pushes_all, win_rate_all, net_profit_all = calculate_metrics(display_df)

    col4, col5, col6 = st.columns(3)
    col4.metric("Overall Win Rate (All Picks)", f"{win_rate_all:.1%}", f"{wins_all}-{losses_all}-{pushes_all}")
    col5.metric("Total Net Profit - All Picks (Units)", f"{net_profit_all:+.2f}")
    col6.metric("Total Picks Evaluated", len(display_df))

    st.divider()

    # Ensure team columns map correctly if the export used capitalized 'Home'/'Away'
    if 'Home' in display_df.columns and 'home_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Home': 'home_team'})
    if 'Away' in display_df.columns and 'away_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Away': 'away_team'})
    
    # Select columns to show
    cols_to_show = ['league', 'home_team', 'away_team', 'best_pick', 'actual_home_score', 'actual_away_score', 'Outcome', 'Pick_Status']
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
        disabled=["League", "Home", "Away", "Pick Taken", "Status"],
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
