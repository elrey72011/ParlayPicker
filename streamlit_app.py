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
from zoneinfo import ZoneInfo
from app_core.kalshi_integrator import KalshiIntegrator

# Helper imports from app_core
try:
    from app_core.kalshi_integrator import league_game_prefix, team_code_for_league
except ImportError:
    def league_game_prefix(league: str) -> str:
        return f"KX{league.upper()}GAME"
    def team_code_for_league(league: str, team: str) -> str:
        return team[:3].upper()

# --- Global Definitions ---
# 4. Verify Global Definitions
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

        # 3. Mandatory Fuzzy Logic (Line 2145 in original, now here)

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

if __name__ == "__main__":
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

    kalshi = KalshiIntegrator()
    if not kalshi.api_key:
        st.warning("Kalshi API key not found. Market data will be unavailable.")

    # -------------------------------------------------------------------------
    # 3. Data Fetching
    # -------------------------------------------------------------------------
    with st.spinner(f"Fetching {selected_league} games for {selected_date}..."):
        games_raw = client.get_games_by_date(selected_date)

    if not games_raw:
        st.info(f"No games found for {selected_league} on {selected_date}.")
        with st.expander("Debug Info"):
            st.write(getattr(client, 'last_error', 'No error logged'))
        st.stop()

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
            # Attach original raw data if needed
            norm_game.update(g)
            games_flat.append(norm_game)

    # -------------------------------------------------------------------------
    # 4. Enrichment
    # -------------------------------------------------------------------------
    with st.spinner("Enriching game context with stats..."):
        enriched_games = enrich_game_context(games_flat, {selected_league: client}, games_api=games_raw)

    # Fetch Kalshi Markets globally for the league to avoid rate limits
    kalshi_markets = []
    if kalshi.api_key:
        with st.spinner("Fetching Kalshi markets..."):
            kalshi_markets = kalshi.get_markets_for_league(selected_league)

    # -------------------------------------------------------------------------
    # 5. Main Analysis Loop
    # -------------------------------------------------------------------------
    rows_out = []
    processed_games = []
    validation_rows = []

    progress_bar = st.progress(0)
    total_games = len(enriched_games)

    for i, game in enumerate(enriched_games):
        # A. Validation
        # Create a single-row DataFrame for validation as requested.
        # We populate it with the stats we already enriched via enrich_game_context.
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

        # Run validation (logging only) - also will be run in ROI tab
        run_roi_pipeline_validation(val_df)

        # B. Kalshi Matching
        # Filter markets for this specific game
        matched_markets = filter_kalshi_game_markets(
            kalshi_markets,
            game_time_utc=game.get("commence_time"),
            league=selected_league,
            home_team=game.get("home_team"),
            away_team=game.get("away_team")
        )

        # Pick the best market (e.g., Moneyline/Winner)
        # filter_kalshi_game_markets returns a list. We take the first one if available.
        # Ideally we want the "Winner" market.
        kalshi_match = None
        kalshi_ticker = "N/A"
        kalshi_prob = None

        if matched_markets:
            kalshi_match = matched_markets[0] # Best match
            kalshi_ticker = kalshi_match.get("ticker") or kalshi_match.get("event_ticker")

            # Extract probability (approximate from yes/ask/bid)
            # KalshiIntegrator helpers might be useful, or use raw
            # filter_kalshi_game_markets returns dicts.
            # Let's try to get probability from 'yes_bid' or 'last_price'
            try:
                # Simple logic: use yes_bid / 100 or last_price / 100
                price = kalshi_match.get("yes_bid") or kalshi_match.get("last_price") or 50
                kalshi_prob = float(price) / 100.0
            except:
                kalshi_prob = 0.5

        # C. Construct Output Row
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        # Use enriched features for prediction
        home_win_pct = game.get("home_win_pct", 0.5)
        away_win_pct = game.get("away_win_pct", 0.5)

        predicted_winner = home_team if home_win_pct >= away_win_pct else away_team
        win_prob = max(home_win_pct, away_win_pct)

        # ROI Calculation
        # ROI = (Win Prob - Implied Prob) / Implied Prob
        roi = 0.0
        if kalshi_prob and kalshi_prob > 0:
            roi = (win_prob - kalshi_prob) / kalshi_prob

        row_data = {
            "Home Team": home_team,
            "Away Team": away_team,
            "Predicted Winner": predicted_winner,
            "Win Prob": f"{win_prob:.1%}",
            "ROI": f"{roi:.1%}",
            "Kalshi Ticker": kalshi_ticker
        }

        # Store rich data for Tabs
        game_display_data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_streak": game.get("home_streak", 0.0),
            "away_streak": game.get("away_streak", 0.0),
            "home_ppg": game.get("home_ppg", 0.0),
            "away_ppg": game.get("away_ppg", 0.0),
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "kalshi_ticker": kalshi_ticker,
            "kalshi_prob": kalshi_prob,
            "win_prob": win_prob,
            "roi": roi,
            "predicted_winner": predicted_winner,
            "commence_time": game.get("commence_time")
        }
        processed_games.append(game_display_data)

        # CRUCIAL REQUIREMENT: Append at the absolute bottom
        progress_bar.progress((i + 1) / total_games)
        rows_out.append(row_data)

    progress_bar.empty()

    # -------------------------------------------------------------------------
    # 6. Display Results
    # -------------------------------------------------------------------------
    if rows_out:
        # Create Tabs
        tab_pred, tab_insights, tab_roi = st.tabs(["🎯 Predictions", "📊 Team Insights", "📈 ROI Analysis"])

        # --- TAB 1: Predictions (Game Cards) ---
        with tab_pred:
            st.subheader(f"Game Cards ({len(processed_games)})")

            for g in processed_games:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.5, 1, 1.5])

                    # Helper for hot/cold
                    def get_trend_str(streak_val):
                        if streak_val > 0: return f"🔥 Hot (W{int(streak_val)})"
                        if streak_val < 0: return f"❄️ Cold (L{int(abs(streak_val))})"
                        return "Neutral"

                    # Column 1: Home
                    with c1:
                        st.markdown(f"### 🏠 {g['home_team']}")
                        st.caption(f"PPG: {g['home_ppg']:.1f} | Win%: {g['home_win_pct']:.1%}")
                        st.markdown(f"**Trend:** {get_trend_str(g['home_streak'])}")

                    # Column 2: VS / Info
                    with c2:
                        st.markdown("#### VS")
                        st.metric("Win Prob", f"{g['win_prob']:.1%}", delta=f"ROI: {g['roi']:.1%}")
                        st.caption(f"Winner: {g['predicted_winner']}")
                        if g['kalshi_ticker'] != "N/A":
                            st.caption(f"Kalshi: {g['kalshi_ticker']}")

                    # Column 3: Away
                    with c3:
                        st.markdown(f"### ✈️ {g['away_team']}")
                        st.caption(f"PPG: {g['away_ppg']:.1f} | Win%: {g['away_win_pct']:.1%}")
                        st.markdown(f"**Trend:** {get_trend_str(g['away_streak'])}")

        # --- TAB 2: Team Insights ---
        with tab_insights:
            st.subheader("Deep Metrics")
            if enriched_games:
                # Construct a clean DataFrame for insights
                insight_rows = []
                for g in enriched_games:
                    # Home row
                    insight_rows.append({
                        "Team": g.get("home_team"),
                        "Role": "Home",
                        "PPG": g.get("home_ppg"),
                        "OPPG": g.get("home_oppg"),
                        "Win Pct": g.get("home_win_pct"),
                        "Streak": g.get("home_streak"),
                    })
                    # Away row
                    insight_rows.append({
                        "Team": g.get("away_team"),
                        "Role": "Away",
                        "PPG": g.get("away_ppg"),
                        "OPPG": g.get("away_oppg"),
                        "Win Pct": g.get("away_win_pct"),
                        "Streak": g.get("away_streak"),
                    })
                st.dataframe(pd.DataFrame(insight_rows), use_container_width=True)
            else:
                st.info("No insights available.")

        # --- TAB 3: ROI Analysis ---
        with tab_roi:
            st.subheader("Pipeline Validation Breakdown")
            if validation_rows:
                master_val_df = pd.concat(validation_rows, ignore_index=True)
                val_results = run_roi_pipeline_validation(master_val_df)

                # Display validation results nicely
                st.write("### Validation Status")
                for k, v in val_results.items():
                    if "❌" in v:
                        st.error(f"**{k}**: {v}")
                    elif "⚠️" in v:
                        st.warning(f"**{k}**: {v}")
                    else:
                        st.success(f"**{k}**: {v}")

                st.markdown("---")
                st.write("### Raw Validation Data")
                st.dataframe(master_val_df, use_container_width=True)
            else:
                st.warning("No validation data collected.")

    else:
        st.warning("No games processed.")
