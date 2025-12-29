import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union

# CRITICAL: Force discovery of local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Backend Imports
from app_core.apisports import APISportsBasketballClient, APISportsFootballClient, APISportsHockeyClient, get_key
from app_core.sportsdata import SportsDataNBAClient, SportsDataNCAABClient, SportsDataNFLClient, SportsDataNCAAFClient, SportsDataNHLClient
from app_core.feature_processing import run_roi_pipeline_validation, TeamNameMatcher
from app_core.sentiment_pipeline import SentimentPipeline
from app_core.vertex_ai_endpoint import is_vertex_prediction_configured, predict_win_probabilities, VERTEX_FEATURE_COLUMNS
from zoneinfo import ZoneInfo
from app_core.kalshi_integrator import KalshiIntegrator
from parlay_optimizer import ParlayOptimizer
from vertex_master_analyzer import blended_win_prob

# Helper imports from app_core
try:
    from app_core.kalshi_integrator import league_game_prefix, team_code_for_league
except ImportError:
    def league_game_prefix(league: str) -> str:
        return f"KX{league.upper()}GAME"
    def team_code_for_league(league: str, team: str) -> str:
        return team[:3].upper()

# --- Global Definitions ---
CLIENT_MAPPING = {
    "NBA": APISportsBasketballClient,
    "NFL": APISportsFootballClient,
    "NHL": APISportsHockeyClient,
    "NCAAB": SportsDataNCAABClient,
    "NCAAF": SportsDataNCAAFClient,
}

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_local_tz() -> str:
    """Return the local timezone string."""
    return "America/New_York"

def parse_commence_to_utc(date_str: str) -> datetime:
    """Parse a commence time string to UTC datetime."""
    try:
        if not date_str:
            return datetime.now(timezone.utc)
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def kalshi_date_token_from_local(dt: Optional[datetime]) -> str:
    """Generate Kalshi date token from local time."""
    if not dt:
        return "UNKNOWN"
    return dt.strftime("%y%b%d").upper()

def team_code_candidates(league: str, team: Any) -> List[str]:
    """Get candidate team codes for a team."""
    if not team:
        return []
    code = team_code_for_league(league, str(team))
    return [code] if code else []

def enrich_game_context(games_data: List[Dict], api_clients: Dict[str, Any], games_api: List[Dict] = None) -> List[Dict]:
    """
    Enrich game data with stats using fuzzy matching as the only way to match teams.
    Ref: Unlock Real Win Rates - Mandatory Fuzzy Logic.
    """
    if not games_data:
        return []

    # 1. Fetch all available stats into a lookup dataframe
    from app_core.feature_processing import fetch_and_process_standings

    stats_df = fetch_and_process_standings(api_clients)

    # 2. Create a lookup dictionary: normalized_name -> stats_row
    stats_lookup = {}
    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            norm_name = row.get('team_norm')
            if norm_name:
                stats_lookup[norm_name] = row.to_dict()

    enriched_games = []

    for game in games_data:
        g = game.copy()

        home_raw = g.get('home_team')
        away_raw = g.get('away_team')

        # 3. Mandatory Fuzzy Logic
        # Force fuzzy matching to bridge naming gaps
        home_norm = TeamNameMatcher.normalize(home_raw)
        away_norm = TeamNameMatcher.normalize(away_raw)

        matched = None
        for g_api in (games_api or []):
            api_teams = g_api.get("teams", {})
            h_api = api_teams.get("home", {}).get("name", "")
            a_api = api_teams.get("away", {}).get("name", "")

            if TeamNameMatcher.match_team(home_norm, [h_api], threshold=0.75) and \
               TeamNameMatcher.match_team(away_norm, [a_api], threshold=0.75):
                matched = g_api
                break

        home_stats = {}
        away_stats = {}

        if matched:
            # Use matched game to retrieve canonical names for stats lookup
            h_api_name = matched.get("teams", {}).get("home", {}).get("name")
            a_api_name = matched.get("teams", {}).get("away", {}).get("name")

            # Match these canonical names to our stats_lookup (which uses normalized names)
            matched_home = TeamNameMatcher.match_team(TeamNameMatcher.normalize(h_api_name), list(stats_lookup.keys()))
            home_stats = stats_lookup.get(matched_home) if matched_home else {}

            matched_away = TeamNameMatcher.match_team(TeamNameMatcher.normalize(a_api_name), list(stats_lookup.keys()))
            away_stats = stats_lookup.get(matched_away) if matched_away else {}
        else:
            # Fallback if no game match (use original names)
            matched_home = TeamNameMatcher.match_team(home_norm, list(stats_lookup.keys()))
            home_stats = stats_lookup.get(matched_home) if matched_home else {}

            matched_away = TeamNameMatcher.match_team(away_norm, list(stats_lookup.keys()))
            away_stats = stats_lookup.get(matched_away) if matched_away else {}

        # Inject stats with defaults if missing
        g['home_win_pct'] = home_stats.get('win_pct', 0.5)
        g['home_ppg'] = home_stats.get('ppg', 50.0)
        g['home_oppg'] = home_stats.get('oppg', 50.0)
        g['home_streak'] = home_stats.get('streak', 0.0)

        g['away_win_pct'] = away_stats.get('win_pct', 0.5)
        g['away_ppg'] = away_stats.get('ppg', 50.0)
        g['away_oppg'] = away_stats.get('oppg', 50.0)
        g['away_streak'] = away_stats.get('streak', 0.0)

        enriched_games.append(g)

    return enriched_games

def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
    home_team: Any = None,
    away_team: Any = None,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flexible filtering for game markets with title-based fallback."""
    try:
        tz_name = get_local_tz()
        local_tz = None
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = None

        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_commence_to_utc(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)
        game_local = game_dt.astimezone(local_tz) if (game_dt and local_tz) else game_dt
        date_token = game_local.strftime("%y%b%d").upper() if game_local else kalshi_date_token_from_local(game_time_utc)
        date_token = date_token or "UNKNOWN"

        league_upper = (league or "").upper()
        winner_prefix = league_game_prefix(league_upper)
        prefix_overrides = (st.session_state.get("kalshi_game_prefix_map") or {}).get(
            league_upper
        )
        allowed_prefixes = [p for p in [winner_prefix, prefix_overrides] if p]

        home_codes = []
        away_codes = []
        if home_code:
            home_codes.append(str(home_code).upper())
        home_codes.extend(team_code_candidates(league, home_team))
        if away_code:
            away_codes.append(str(away_code).upper())
        away_codes.extend(team_code_candidates(league, away_team))

        def ticker_upper(market: Dict[str, Any]) -> str:
            return str(market.get("event_ticker") or market.get("ticker") or "").upper()

        allowed_date_tokens: List[str] = []
        if game_local:
            base_date = game_local.date()
            for delta in (-1, 0, 1):
                allowed_date_tokens.append(
                    (base_date + timedelta(days=delta)).strftime("%y%b%d").upper()
                )
        
        home_codes_set = set(c.upper() for c in home_codes if c)
        away_codes_set = set(c.upper() for c in away_codes if c)

        filtered = []
        for m in markets:
            ticker = ticker_upper(m)
            
            if league == "NCAAB" and "GAME" in ticker and "NCAA" in ticker:
                 has_any_team = any(c in ticker for c in home_codes_set.union(away_codes_set))
                 if has_any_team:
                      filtered.append(m)
                      continue

            if not any(p in ticker for p in allowed_prefixes):
                continue

            date_match = False
            for dt_tok in allowed_date_tokens:
                if dt_tok in ticker:
                    date_match = True
                    break
            
            if not date_match:
                 continue

            has_home = any(c in ticker for c in home_codes_set)
            has_away = any(c in ticker for c in away_codes_set)
            
            if has_home and has_away:
                filtered.append(m)
        
        return filtered

    except Exception:
        return []

def load_games(client, selected_date: datetime, league: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch games and normalize them.
    Returns (raw_games, normalized_games).
    """
    with st.spinner(f"Fetching {league} games for {selected_date}..."):
        # Ensure date format if needed or pass date object directly if client supports it
        # Clients usually take datetime or string. Assuming datetime is fine based on previous code.
        games_raw = client.get_games_by_date(selected_date)

    if not games_raw:
        return [], []

    # Normalize games for enrichment
    # We need a flat structure: home_team, away_team, commence_time, game_id
    games_flat = []
    for g in games_raw:
        norm_game = {}
        # APISports vs SportsData handling
        if "teams" in g and "home" in g["teams"]: # APISports
            norm_game["home_team"] = g["teams"]["home"]["name"]
            norm_game["away_team"] = g["teams"]["away"]["name"]
            norm_game["commence_time"] = g["game"]["date"]
            norm_game["game_id"] = g["game"]["id"]
        elif "HomeTeam" in g or "HomeTeamName" in g: # SportsData
            norm_game["home_team"] = g.get("HomeTeamName") or g.get("HomeTeam")
            norm_game["away_team"] = g.get("AwayTeamName") or g.get("AwayTeam")
            norm_game["commence_time"] = g.get("DateTime") or g.get("Date")
            norm_game["game_id"] = g.get("GameKey") or g.get("GlobalGameID")

        if norm_game.get("home_team"):
            # Attach original raw data if needed for later reference
            norm_game.update(g)
            games_flat.append(norm_game)

    return games_raw, games_flat


if __name__ == "__main__":
    st.set_page_config(page_title="ParlayDesk", layout="wide")
    st.title("ParlayDesk")

    # -------------------------------------------------------------------------
    # 1. Initialize Sidebar Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("Configuration")
    selected_date = st.sidebar.date_input("Game Date", datetime.now())
    selected_league = st.sidebar.selectbox("League", list(CLIENT_MAPPING.keys()), index=0)

    # -------------------------------------------------------------------------
    # 2. Initialize Data Clients
    # -------------------------------------------------------------------------
    client_cls = CLIENT_MAPPING.get(selected_league)

    # 2a. Secure Key Loading
    # Try the wrapper first, then fallback to st.secrets['general']
    api_key = get_key(selected_league)
    if not api_key:
         try:
             # APISports vs SportsData key names in secrets
             if selected_league in ["NBA", "NFL", "NHL"]:
                 api_key = st.secrets["general"]["api_sports_key"]
             else:
                 api_key = st.secrets["general"]["sportsdata_key"]
         except Exception:
             pass

    if not client_cls or not api_key:
        st.error(f"Configuration missing for {selected_league}. Please check secrets.")
        st.stop()

    client = client_cls(api_key=api_key)
    if not client.is_configured():
        st.error(f"Failed to configure client for {selected_league}.")
        st.stop()

    # Initialize Kalshi
    kalshi = KalshiIntegrator()
    if not kalshi.api_key:
        st.warning("Kalshi API key not found. Market data will be unavailable.")

    # -------------------------------------------------------------------------
    # 3. Data Fetching & Processing
    # -------------------------------------------------------------------------

    # Load Games
    games_raw, games_flat = load_games(client, selected_date, selected_league)

    if not games_raw:
        st.info(f"No games found for {selected_league} on {selected_date}.")
        with st.expander("Debug Info"):
            st.write(getattr(client, 'last_error', 'No error logged'))

        # Display tabs even if no games, to show debug info if needed, but mostly stop
        tab_shotgun, tab_master, tab_games, tab_kalshi, tab_sentiment, tab_debug = st.tabs(
            ["🚀 Shotgun Mode", "Master Analysis", "Games & Odds", "Kalshi", "Sentiment", "Debug"]
        )
        with tab_debug:
             st.write("Client Last Error:", getattr(client, 'last_error', 'None'))
        st.stop()

    # Load TheOver.ai Odds CSV if available
    odds_df = pd.DataFrame()
    try:
        csv_path = f"data/theover/{selected_date.strftime('%Y-%m-%d')}.csv"
        # Fallback to test file if date specific not found for dev
        if not os.path.exists(csv_path) and os.path.exists("data/theover/test_odds.csv"):
             # Only strictly use test file if user requested (or for this dev session)
             # But for automation requirement, we check the date.
             pass

        if os.path.exists(csv_path):
             odds_df = pd.read_csv(csv_path)
             st.toast(f"Loaded TheOver.ai Odds from {csv_path}", icon="✅")
        elif os.path.exists("data/theover/test_odds.csv"):
             # For dev verification purposes in this environment
             odds_df = pd.read_csv("data/theover/test_odds.csv")
             st.toast("Loaded Test Odds Data (Dev Mode)", icon="⚠️")

    except Exception as e:
        logger.error(f"Failed to load odds CSV: {e}")

    # Enrichment
    with st.spinner("Enriching game context with stats..."):
        enriched_games = enrich_game_context(games_flat, {selected_league: client}, games_api=games_raw)

    # Fetch Kalshi Markets
    kalshi_markets = []
    if kalshi.api_key:
        with st.spinner("Fetching Kalshi markets..."):
            kalshi_markets = kalshi.get_markets_for_league(selected_league)

    # Sentiment Analysis
    with st.spinner("Analyzing Sentiment..."):
        news_key = st.secrets["general"].get("news_api_key")
        sentiment_map, sentiment_meta, sentiment_debug = SentimentPipeline.build_team_sentiment_map(
            news_key, enriched_games, selected_league
        )

    # -------------------------------------------------------------------------
    # 4. Master Analysis Loop
    # -------------------------------------------------------------------------
    rows_out = []
    validation_rows = []

    progress_bar = st.progress(0)
    total_games = len(enriched_games)

    # Store Vertex features to batch predict later
    vertex_rows = []

    for i, game in enumerate(enriched_games):
        # A. Validation
        game_df_dict = {
            "home_team": [game.get("home_team")],
            "away_team": [game.get("away_team")],
            "commence_time": [game.get("commence_time")],
            "feature_home_ppg": [game.get("home_ppg")],
            "feature_home_win_pct": [game.get("home_win_pct")],
            "feature_away_ppg": [game.get("away_ppg")],
            "feature_away_win_pct": [game.get("away_win_pct")],
            "League": [selected_league]
        }
        val_df = pd.DataFrame(game_df_dict)
        validation_rows.append(val_df)
        run_roi_pipeline_validation(val_df)

        # B. Kalshi Matching
        matched_markets = filter_kalshi_game_markets(
            kalshi_markets,
            game_time_utc=game.get("commence_time"),
            league=selected_league,
            home_team=game.get("home_team"),
            away_team=game.get("away_team")
        )

        kalshi_match = None
        kalshi_ticker = "N/A"
        kalshi_prob = None
        kalshi_matched = False

        if matched_markets:
            kalshi_match = matched_markets[0] # Best match
            kalshi_ticker = kalshi_match.get("ticker") or kalshi_match.get("event_ticker")
            kalshi_matched = True
            try:
                price = kalshi_match.get("yes_bid") or kalshi_match.get("last_price") or 50
                kalshi_prob = float(price) / 100.0
            except:
                kalshi_prob = 0.5

        # C. Sentiment Injection
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        home_sent = sentiment_map.get(home_team, 0.0)
        away_sent = sentiment_map.get(away_team, 0.0)
        # Handle None if sentiment returns None
        if home_sent is None: home_sent = 0.0
        if away_sent is None: away_sent = 0.0

        sentiment_diff = home_sent - away_sent

        # D. Construct Output Row
        # Use enriched features for prediction
        home_win_pct = game.get("home_win_pct", 0.5)
        away_win_pct = game.get("away_win_pct", 0.5)
        home_ppg = game.get("home_ppg", 50.0)
        away_ppg = game.get("away_ppg", 50.0)
        home_streak = game.get("home_streak", 0.0)
        away_streak = game.get("away_streak", 0.0)

        # E. Odds Matching (TheOver.ai)
        pick_odds = None
        market_prob = None

        if not odds_df.empty:
            # Fuzzy match team names
            home_norm = TeamNameMatcher.normalize(home_team)

            # Simple exact match on normalized names or loop through odds_df
            # This is O(N*M), but acceptable for small N
            matched_odds = None
            for _, o_row in odds_df.iterrows():
                o_home = TeamNameMatcher.normalize(o_row.get('HomeTeam', ''))
                o_away = TeamNameMatcher.normalize(o_row.get('AwayTeam', ''))

                # Check match
                if TeamNameMatcher.match_team(home_norm, [o_row.get('HomeTeam', '')], threshold=0.8):
                    matched_odds = o_row
                    break

            if matched_odds is not None:
                # Decide implied probability from odds
                # Assuming Moneyline for simplicity in this bridge
                h_ml = matched_odds.get('HomeMoneyLine')
                a_ml = matched_odds.get('AwayMoneyLine')

                # Assign odds based on predicted winner (pre-vertex)
                # We will refine this after Vertex
                # For now, store both
                pass

        predicted_winner = home_team if home_win_pct >= away_win_pct else away_team
        win_prob = max(home_win_pct, away_win_pct)

        # ROI Calculation (pre-Vertex)
        roi = 0.0
        if kalshi_prob and kalshi_prob > 0:
            # Simplified Assumption: Kalshi Prob is for Home Win
            implied_prob = kalshi_prob
            my_prob = home_win_pct
            roi = (my_prob - implied_prob) / implied_prob

        row_data = {
            "Home Team": home_team,
            "Away Team": away_team,
            "Predicted Winner": predicted_winner,
            "Win Prob": win_prob, # Float for now
            "ROI": roi,
            "Kalshi Ticker": kalshi_ticker,
            "kalshi_prob": kalshi_prob,
            "kalshi_matched": kalshi_matched,
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "home_ppg": home_ppg,
            "away_ppg": away_ppg,
            "home_streak": home_streak,
            "away_streak": away_streak,
            "sentiment_home": home_sent,
            "sentiment_away": away_sent,
            "sentiment_diff": sentiment_diff,
            "commence_time": game.get("commence_time"),
            # Store matched odds data if any
            "odds_home_ml": matched_odds.get('HomeMoneyLine') if 'matched_odds' in locals() and matched_odds is not None else None,
            "odds_away_ml": matched_odds.get('AwayMoneyLine') if 'matched_odds' in locals() and matched_odds is not None else None,
            "game": f"{away_team} @ {home_team}"
        }

        # Prepare Vertex Row
        # Needs to match VERTEX_FEATURE_COLUMNS
        v_row = {
            "implied_home_prob": kalshi_prob if kalshi_prob else 0.5,
            "sentiment_diff": sentiment_diff,
            "kalshi_prob": kalshi_prob if kalshi_prob else 0.5,
            "feature_home_win_pct": home_win_pct,
            "feature_home_ppg": home_ppg,
            "feature_home_streak": home_streak,
            "feature_away_win_pct": away_win_pct,
            "feature_away_ppg": away_ppg,
            "feature_away_streak": away_streak,
            # Add other required columns with defaults
            "injuries_home_count": 0.0,
            "injuries_away_count": 0.0,
            "weather_flag": 0.0,
            "feature_home_home_win_pct": home_win_pct, # Proxy
            "feature_home_last5_win_pct": home_win_pct, # Proxy
            "feature_home_oppg": 50.0, # Default
            "feature_away_away_win_pct": away_win_pct, # Proxy
            "feature_away_last5_win_pct": away_win_pct, # Proxy
            "feature_away_oppg": 50.0, # Default
            "feature_diff_win_pct": home_win_pct - away_win_pct,
            "feature_diff_ppg": home_ppg - away_ppg,
            "feature_diff_oppg": 0.0,
            "feature_diff_last5": 0.0,
            "feature_diff_streak": home_streak - away_streak,
            "feature_commence_hour": 19.0, # Default
            "feature_commence_day_of_week": 0.0, # Default
            "feature_home_rest_days": 3.0, # Default
            "feature_away_rest_days": 3.0, # Default
        }
        vertex_rows.append(v_row)

        progress_bar.progress((i + 1) / total_games)
        rows_out.append(row_data)

    progress_bar.empty()

    # -------------------------------------------------------------------------
    # 5. Post-Loop Processing (Vertex & Persistence)
    # -------------------------------------------------------------------------

    # Create Master DataFrame
    df = pd.DataFrame(rows_out)

    # Run Vertex Predictions if configured
    if not df.empty and is_vertex_prediction_configured():
        with st.spinner("Running Vertex AI Predictions..."):
            vertex_df = pd.DataFrame(vertex_rows)
            preds = predict_win_probabilities(vertex_df)

            # Attach predictions to df
            if len(preds) == len(df):
                df["vertex_home_win_prob"] = preds
            else:
                df["vertex_home_win_prob"] = np.nan

    # --- Blended Probability Calculation ---
    if not df.empty:
        # Calculate final_win_prob using blended logic
        def calc_blended(row):
            v_prob = row.get("vertex_home_win_prob")
            k_prob = row.get("kalshi_prob")
            sent = row.get("sentiment_diff")

            # Use market prob from odds if available, else 0.5 default
            m_prob = 0.5
            if row.get("odds_home_ml"):
                # Convert ML to prob
                def ml_to_prob(ml):
                    try:
                        ml = float(ml)
                        return 100/(ml+100) if ml > 0 else abs(ml)/(abs(ml)+100)
                    except: return 0.5
                m_prob = ml_to_prob(row["odds_home_ml"])

            # Determine selection based on vertex or basic stats
            home_prob = v_prob if pd.notnull(v_prob) else row.get("home_win_pct", 0.5)
            selection = "home" if home_prob >= 0.5 else "away"

            final_prob = blended_win_prob(
                market_prob=m_prob,
                vertex_prob=v_prob if pd.notnull(v_prob) else None,
                theover_prob=None, # Not explicitly parsed yet
                kalshi_prob=k_prob if row.get("kalshi_matched") else None,
                sentiment_diff=sent,
                selection=selection
            )
            return pd.Series([final_prob, selection])

        df[["final_win_prob", "Pick_Side"]] = df.apply(calc_blended, axis=1)

        # Set final pick text and odds
        df["Pick"] = df.apply(lambda r: r["Home Team"] if r["Pick_Side"] == "home" else r["Away Team"], axis=1)
        df["pick_odds"] = df.apply(lambda r: r["odds_home_ml"] if r["Pick_Side"] == "home" else r["odds_away_ml"], axis=1)

    # Persist
    st.session_state["master_df"] = df

    # -------------------------------------------------------------------------
    # 6. Tab UI Implementation
    # -------------------------------------------------------------------------

    tab_shotgun, tab_master, tab_games, tab_kalshi, tab_sentiment, tab_debug = st.tabs(
        ["🚀 Shotgun Mode", "Master Analysis", "Games & Odds", "Kalshi", "Sentiment", "Debug"]
    )

    # --- Tab 1: Shotgun Mode ---
    with tab_shotgun:
        st.subheader("🚀 Shotgun Allocation")
        if not df.empty:
             optimizer = ParlayOptimizer(model_dir=".", min_edge=0.01)
             # Calculate allocation
             allocation = optimizer.get_shotgun_allocation(df)

             c1, c2, c3 = st.columns(3)

             with c1:
                 st.markdown("### 🎯 $3 Snipers")
                 st.caption("High Confidence, Positive EV")
                 if not allocation['snipers'].empty:
                     for _, row in allocation['snipers'].iterrows():
                         st.success(f"**{row['Pick']}**")
                         st.write(f"Prob: {row['final_win_prob']:.1%} | EV: {row['ev']:.1%}")
                         if pd.notnull(row.get('pick_odds')):
                             st.caption(f"Odds: {row['pick_odds']}")
                         st.divider()
                 else:
                     st.info("No Sniper plays found.")

             with c2:
                 st.markdown("### ♟️ $2 Strategy")
                 st.caption("3-Leg Parlays, Edge > 5%")
                 if not allocation['strategy'].empty:
                     for _, row in allocation['strategy'].iterrows():
                         st.warning(f"**{row['Odds']}**")
                         st.write(f"{row['Legs']}")
                         st.write(f"EV: {row['EV']:.1%}")
                         st.divider()
                 else:
                     st.info("No Strategy plays found.")

             with c3:
                 st.markdown("### 🚀 $1 Longshots")
                 st.caption("4-5 Legs, High Upside")
                 if not allocation['longshots'].empty:
                     for _, row in allocation['longshots'].iterrows():
                         st.error(f"**{row['Odds']}**")
                         st.write(f"{row['Legs']}")
                         st.write(f"Kelly Growth: {row['Kelly ROI']:.2%}")
                         st.divider()
                 else:
                     st.info("No Longshot plays found.")

        else:
            st.info("No data available for Shotgun Mode.")

    # --- Tab 3: Games & Odds ---
    with tab_games:
        st.subheader(f"Games for {selected_date}")
        if not df.empty:
            # Simplified view
            display_cols = ["Home Team", "Away Team", "Predicted Winner", "Win Prob", "Kalshi Ticker", "ROI"]
            # Formatting
            st.dataframe(
                df[display_cols].style.format({
                    "Win Prob": "{:.1%}",
                    "ROI": "{:.1%}"
                }),
                use_container_width=True
            )

            st.markdown("### Game Cards")
            for idx, row in df.iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.5, 1, 1.5])
                    with c1:
                        st.markdown(f"**{row['Home Team']}**")
                        st.caption(f"Win%: {row['home_win_pct']:.1%}")
                    with c2:
                        st.markdown(f"**VS**")
                        if row['vertex_home_win_prob'] and not pd.isna(row['vertex_home_win_prob']):
                             st.metric("Vertex Home %", f"{row['vertex_home_win_prob']:.1%}")
                    with c3:
                        st.markdown(f"**{row['Away Team']}**")
                        st.caption(f"Win%: {row['away_win_pct']:.1%}")

        else:
            st.info("No games processed.")

    # --- Tab 2: Master Analysis ---
    with tab_master:
        st.subheader("Master Analysis DataFrame")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data available.")

    # --- Tab 3: Kalshi ---
    with tab_kalshi:
        st.subheader("Kalshi Market Data")
        if kalshi_markets:
             st.write(f"Found {len(kalshi_markets)} markets for {selected_league}")
             st.dataframe(pd.DataFrame(kalshi_markets).head(50), use_container_width=True)
        else:
             st.info("No Kalshi markets found or API key missing.")

    # --- Tab 4: Sentiment ---
    with tab_sentiment:
        st.subheader("Sentiment Analysis")

        c1, c2 = st.columns(2)
        with c1:
            st.write("### Sentiment Map")
            st.json(sentiment_map)
        with c2:
            st.write("### Debug Info")
            st.json(sentiment_debug)

    # --- Tab 5: Debug ---
    with tab_debug:
        st.subheader("System Debug")
        st.write("### Client Info")
        st.write(f"Client: {client_cls.__name__}")
        st.write(f"Last Error: {getattr(client, 'last_error', 'None')}")

        st.write("### Session State")
        st.write(st.session_state)

        if validation_rows:
            st.write("### Validation Data")
            st.dataframe(pd.concat(validation_rows), use_container_width=True)

