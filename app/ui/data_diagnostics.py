import streamlit as st


def _safe_len(df):
    if df is not None and not df.empty:
        return len(df)
    return 0


import pandas as pd
from app_core.odds_api import TheOddsAPIClient, OddsAPIAuthError

def show_data_diagnostics(
    odds_df,
    theover_df,
    kalshi_df,
    gemini_df,
):
    st.subheader("Raw API Data Tools")
    st.write("Fetch raw, unmerged data from The Odds API for the sports currently selected in the sidebar.")

    if "raw_odds_csv_data" not in st.session_state:
        st.session_state.raw_odds_csv_data = None
        st.session_state.raw_odds_fetch_msg = ""
        st.session_state.raw_odds_fetch_success = False

    if st.button("Fetch Raw Odds API Data", key="fetch_raw_odds_api"):
        import os
        api_key = st.secrets.get("ODDS_API_KEY") or os.environ.get("ODDS_API_KEY", "")

        if not api_key:
            st.session_state.raw_odds_fetch_msg = "The Odds API key is missing. Please verify your credentials in Streamlit secrets."
            st.session_state.raw_odds_fetch_success = False
            st.session_state.raw_odds_csv_data = None
        else:
            try:
                # We need to grab the sports from the current session state/sidebar controls if possible
                sports = st.session_state.get("selected_sports", ["NBA", "NHL", "NCAAB", "NFL", "NCAAF"])

                SPORT_KEYS = {
                    "NBA": "basketball_nba",
                    "NHL": "icehockey_nhl",
                    "NCAAB": "basketball_ncaab",
                    "NFL": "americanfootball_nfl",
                    "NCAAF": "americanfootball_ncaaf"
                }

                sports_to_fetch = [s for s in sports if s.upper() in SPORT_KEYS]
                if not sports_to_fetch:
                    sports_to_fetch = ["NBA", "NHL", "NCAAB", "NFL", "NCAAF"]

                with st.spinner(f"Fetching raw Odds API data for {sports_to_fetch}..."):
                    client = TheOddsAPIClient(
                        api_key=api_key,
                        regions="us_ex,us",
                        markets="h2h,spreads,totals",
                        bookmakers="novig,draftkings,fanduel",
                        oddsFormat="american"
                    )

                    all_games = []
                    for sport in sports_to_fetch:
                        sport_key = SPORT_KEYS.get(sport.upper())
                        if sport_key:
                            games = client.get_odds(sport_key)
                            if games:
                                all_games.extend(games)

                    if not all_games:
                        st.session_state.raw_odds_fetch_msg = f"No games returned from Odds API for {sports_to_fetch}"
                        st.session_state.raw_odds_fetch_success = False
                        st.session_state.raw_odds_csv_data = None
                    else:
                        from app_core.odds_api import export_raw_odds_api
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            filename = tmp.name

                        # Use export_raw_odds_api which parses the complex JSON structure into a flat CSV
                        export_raw_odds_api(all_games, filename=filename)

                        # Read the file to byte stream for download
                        with open(filename, "rb") as f:
                            st.session_state.raw_odds_csv_data = f.read()

                        st.session_state.raw_odds_fetch_success = True
                        st.session_state.raw_odds_fetch_msg = f"Successfully fetched {len(all_games)} raw games from Odds API!"

            except OddsAPIAuthError:
                st.session_state.raw_odds_fetch_msg = "The Odds API key is invalid or revoked. Please verify your credentials."
                st.session_state.raw_odds_fetch_success = False
                st.session_state.raw_odds_csv_data = None
            except Exception as e:
                st.session_state.raw_odds_fetch_msg = f"Error fetching raw Odds API data: {e}"
                st.session_state.raw_odds_fetch_success = False
                st.session_state.raw_odds_csv_data = None

    if st.session_state.get("raw_odds_fetch_msg"):
        if st.session_state.get("raw_odds_fetch_success"):
            st.success(st.session_state.raw_odds_fetch_msg)
        else:
            st.error(st.session_state.raw_odds_fetch_msg)

    if st.session_state.get("raw_odds_csv_data") is not None:
        st.download_button(
            label="Export Raw Odds CSV",
            data=st.session_state.raw_odds_csv_data,
            file_name=f"odds_api_raw_export.csv",
            mime="text/csv",
            key="download_raw_odds_csv"
        )

    st.divider()

    st.subheader("Data Diagnostics")

    st.write("TheOdds rows:", _safe_len(odds_df))
    st.write("TheOver rows:", _safe_len(theover_df))
    st.write("Kalshi rows:", _safe_len(kalshi_df))
    st.write("Gemini rows:", _safe_len(gemini_df))

    data = {
        "Source": [
            "TheOdds",
            "TheOver",
            "Kalshi",
            "Gemini",
        ],
        "Rows": [
            _safe_len(odds_df),
            _safe_len(theover_df),
            _safe_len(kalshi_df),
            _safe_len(gemini_df),
        ],
    }

    st.dataframe(data)
