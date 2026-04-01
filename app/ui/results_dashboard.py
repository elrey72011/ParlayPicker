import pandas as pd
import streamlit as st
from app_core.results_fetcher import fetch_yesterdays_results
from app_core.results_ingestion import attach_results

def render_results_dashboard(picks_df: pd.DataFrame) -> None:
    st.subheader("Prior Day Performance")

    restricted_leagues = st.session_state.get("restricted_leagues", set())
    if restricted_leagues:
        league_str = ", ".join(sorted(list(restricted_leagues)))
        st.warning(f"Results for [{league_str}] are currently unavailable due to API plan restrictions. These picks must be verified manually.")

    # 1. File Uploader for Yesterday's Picks
    uploaded_picks_file = st.file_uploader("Upload Yesterday's Best Picks Export", type=["csv"])
    if uploaded_picks_file is not None:
        try:
            picks_df = pd.read_csv(uploaded_picks_file)
            st.success("Successfully loaded uploaded picks.")
        except Exception as e:
            st.error(f"Error reading uploaded picks file: {e}")

    if picks_df is None or picks_df.empty:
        st.info("No picks data available for the previous day. Please upload a file above.")
        return

    # 2. Toggle to fetch scores from API
    fetch_api = st.toggle("Fetch Scores from API", value=False)

    # 3. File Uploader for Manual Results
    uploaded_manual_results = st.file_uploader("Upload Manual Results CSV (If API restricted)", type=["csv"])

    has_results = False

    if uploaded_manual_results is not None:
        try:
            manual_results_df = pd.read_csv(uploaded_manual_results)
            picks_df = attach_results(picks_df, manual_results_df)
            st.success("Successfully applied manual results.")
            has_results = True
        except Exception as e:
            st.error(f"Error reading manual results file: {e}")
    elif fetch_api:
        with st.spinner("Fetching yesterday's results from API..."):
            leagues_in_picks = picks_df['league'].unique().tolist() if 'league' in picks_df.columns else []
            # Filter out restricted leagues
            restricted = st.session_state.get("restricted_leagues", set())
            allowed_leagues = [l for l in leagues_in_picks if l not in restricted]

            if allowed_leagues:
                try:
                    fetched_results_df = fetch_yesterdays_results(allowed_leagues)
                    if not fetched_results_df.empty:
                        picks_df = attach_results(picks_df, fetched_results_df)
                        st.success("Successfully fetched and applied API results.")
                        has_results = True
                    else:
                        st.warning("No results found from API for the given leagues.")
                        picks_df = attach_results(picks_df, pd.DataFrame()) # Add empty columns
                except Exception as e:
                    st.error(f"Error fetching results from API: {e}")
            else:
                st.warning("No allowed leagues found in picks to fetch results for (or all are restricted).")

    # We consider it having results if either Pick_Outcome is present or actual_home_score is present
    if not has_results and ('Pick_Outcome' not in picks_df.columns and 'actual_home_score' not in picks_df.columns):
         st.warning("Results have not been attached to these picks yet. Please upload manual results or fetch from API.")
         # Continue to show table of picks anyway, just without outcomes

    # Only evaluate "Actionable" picks for top-line metrics
    actionable_df = picks_df[picks_df.get('Pick_Status', '') == 'Actionable'].copy()

    if actionable_df.empty:
         st.info("No 'Actionable' picks were found in yesterday's export.")
    else:
        # Evaluate Outcome
        # picks_df comes from attach_results, which populates spread_result, total_result, ml_result
        def determine_outcome(row):
            if 'Pick_Outcome' in row and pd.notnull(row['Pick_Outcome']) and row['Pick_Outcome'] != 'N/A':
                return row['Pick_Outcome']

            pick = str(row.get('best_pick', '')).lower()
            if not pick:
                 return 'N/A'
            if 'over' in pick or 'under' in pick:
                 return row.get('total_result', 'N/A')
            elif '+' in pick or '-' in pick:
                 return row.get('spread_result', 'N/A')
            else:
                 return row.get('ml_result', 'N/A')

        actionable_df['Outcome'] = actionable_df.apply(determine_outcome, axis=1)

        wins = len(actionable_df[actionable_df['Outcome'] == 'WIN'])
        losses = len(actionable_df[actionable_df['Outcome'] == 'LOSS'])
        pushes = len(actionable_df[actionable_df['Outcome'] == 'PUSH'])

        total_decisions = wins + losses
        win_rate = (wins / total_decisions) if total_decisions > 0 else 0.0

        # Calculate Net Profit based on odds_american assuming 1 unit bet
        net_profit = 0.0
        for idx, row in actionable_df.iterrows():
            outcome = row['Outcome']
            if outcome == 'WIN':
                 odds = pd.to_numeric(row.get('odds_american'), errors='coerce')
                 if pd.isna(odds):
                      odds = -110 # default

                 # Calculate profit for 1 unit (e.g. $100 bet)
                 if odds < 0:
                      profit = 1.0 / (abs(odds) / 100.0)
                 else:
                      profit = odds / 100.0
                 net_profit += profit
            elif outcome == 'LOSS':
                 net_profit -= 1.0

        # Assume 1 unit = 1 for the metric, maybe display in Units or $ assuming $100/u
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Win Rate (Actionable)", f"{win_rate:.1%}", f"{wins}-{losses}-{pushes}")
        col2.metric("Total Net Profit (Units)", f"{net_profit:+.2f}")
        col3.metric("Actionable Picks Evaluated", len(actionable_df))

    st.divider()

    # Process all rows for the display table
    display_df = picks_df.copy()

    def determine_display_outcome(row):
            if 'Pick_Outcome' in row and pd.notnull(row['Pick_Outcome']) and row['Pick_Outcome'] != 'N/A':
                return row['Pick_Outcome']

            pick = str(row.get('best_pick', '')).lower()
            if not pick:
                 return 'N/A'
            if 'over' in pick or 'under' in pick:
                 return row.get('total_result', 'N/A')
            elif '+' in pick or '-' in pick:
                 return row.get('spread_result', 'N/A')
            else:
                 return row.get('ml_result', 'N/A')

    display_df['Outcome'] = display_df.apply(determine_display_outcome, axis=1)

    def format_final_score(row):
         h = row.get('actual_home_score')
         a = row.get('actual_away_score')
         if pd.isna(h) or pd.isna(a):
             return "Manual Result" if 'Pick_Outcome' in row and pd.notnull(row['Pick_Outcome']) and row['Pick_Outcome'] != 'N/A' else "TBD"
         return f"{int(h)} - {int(a)}"

    display_df['Final Score (H - A)'] = display_df.apply(format_final_score, axis=1)

    # Select columns to show
    cols_to_show = ['league', 'home_team', 'away_team', 'best_pick', 'Final Score (H - A)', 'Outcome', 'Pick_Status']
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

    def color_outcome(val):
         color = ''
         if val == 'WIN':
              color = 'green'
         elif val == 'LOSS':
              color = 'red'
         elif val == 'PUSH':
              color = 'orange'
         return f'color: {color}'

    st.dataframe(table_df.style.map(color_outcome, subset=['Outcome']), width="stretch")
