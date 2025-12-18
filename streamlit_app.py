import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from app_core.kalshi_integrator import KalshiIntegrator, LEAGUE_SERIES_MAP
from app_core.sentiment_pipeline import build_team_sentiment_map
from vertex_master_analyzer import blended_win_prob

# 1. INITIAL CONFIGURATION
st.set_page_config(page_title="ParlayDesk", layout="wide")

# -----------------
# 2. HELPER UTILITIES
# -----------------

def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets: return st.secrets[name]
    except Exception: pass
    return os.getenv(name, default)

def american_to_implied_prob(odds: Any) -> Optional[float]:
    if odds is None: return None
    try:
        o = float(odds)
        if o == 0: return None
        return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)
    except Exception: return None

def get_local_tz() -> str:
    return st.secrets.get("APP_TIMEZONE", "America/New_York")

def fmt_local_time(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else ""

def nba_abbrev(team_name: str) -> Optional[str]:
    mapping = {
        "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
        "charlotte hornets": "CHA", "chicago bulls": "CHI", "cleveland cavaliers": "CLE",
        "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET",
        "golden state warriors": "GSW", "houston rockets": "HOU", "indiana pacers": "IND",
        "los angeles clippers": "LAC", "la clippers": "LAC", "los angeles lakers": "LAL",
        "la lakers": "LAL", "memphis grizzlies": "MEM", "miami heat": "MIA",
        "milwaukee bulls": "MIL", "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP",
        "new york knicks": "NYK", "oklahoma city thunder": "OKC", "orlando magic": "ORL",
        "philadelphia 76ers": "PHI", "phoenix suns": "PHX", "portland trail blazers": "POR",
        "sacramento kings": "SAC", "san antonio spurs": "SAS", "toronto raptors": "TOR",
        "utah jazz": "UTA", "washington wizards": "WAS"
    }
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(team_name or "").lower()).strip()
    for key, code in mapping.items():
        if key in cleaned: return code
    return None

# -----------------
# 3. API CLIENTS & STATE
# -----------------

SPORT_KEYS = {"NBA": "basketball_nba", "NFL": "americanfootball_nfl", "NHL": "icehockey_nhl"}
odds_api_key = read_secret("ODDS_API_KEY")
news_api_key = read_secret("NEWS_API_KEY")
vertex_endpoint_id = read_secret("VERTEX_ENDPOINT_ID")

if "kalshi_integrator" not in st.session_state:
    k_key = read_secret("KALSHI_API_KEY")
    k_secret = read_secret("KALSHI_API_SECRET")
    if k_key and k_secret:
        st.session_state["kalshi_integrator"] = KalshiIntegrator(k_key, k_secret)
    else:
        st.session_state["kalshi_integrator"] = None

def get_vertex_prob(game: Dict[str, Any]) -> Optional[float]:
    # Placeholder for actual Vertex AI logic
    return 0.52 if vertex_endpoint_id else None

# -----------------
# 4. MAIN INTERFACE
# -----------------

st.sidebar.header("Controls")
selected_league = st.sidebar.selectbox("League", list(SPORT_KEYS.keys()))
if st.sidebar.button("Load Games", use_container_width=True):
    # Simulated load logic for brevity
    st.session_state["games"] = [{"home_team": "Lakers", "away_team": "Celtics", "league": "NBA"}]

# --- 1. DEFINE TABS ONCE (Fixes NameError and Duplicate ID) ---
tab_master, tab_kalshi, tab_debug = st.tabs(["Master Analysis", "Kalshi Status", "Debug"])

with tab_master:
    st.header("Best Bets Analysis")
    games = st.session_state.get("games", [])
    
    # Use a unique key if you ever need another button with this label
    if st.button("Run Master Analysis", use_container_width=True, key="master_analysis_btn"):
        if not games:
            st.warning("No games loaded. Please load games from the sidebar first.")
        else:
            # Gather global auxiliary data once per run
            ki = st.session_state.get("kalshi_integrator")
            k_markets = ki.get_sports_markets(selected_league) if ki else []
            sentiment_map, _ = build_team_sentiment_map(news_api_key, games, selected_league) if news_api_key else ({}, {})
            
            rows_out = []
            master_stats = {"market_rows_out": 0}

            for idx, g in enumerate(games):
                # A. SETUP VARIABLES (Sequencing is critical here)
                league_name = g.get("league") or selected_league
                home, away = g.get("home_team"), g.get("away_team")
                h_code, a_code = nba_abbrev(home), nba_abbrev(away)
                
                # B. CALCULATE EXTERNAL DATA (Define variables before fallback)
                vertex_prob_home = get_vertex_prob(g)
                home_sent, away_sent = sentiment_map.get(home, 0.0), sentiment_map.get(away, 0.0)
                sentiment_diff = home_sent - away_sent
                
                # Kalshi Matching Logic
                filtered_k = filter_kalshi_game_markets(
                    k_markets, g.get("commence_time_utc"), league_name,
                    home, away, h_code, a_code
                )
                k_match_results, _ = match_kalshi_market(g, filtered_k)
                
                kalshi_winner = k_match_results.get("winner", {})
                kalshi_spread = k_match_results.get("spread", {})
                kalshi_total = k_match_results.get("total", {})

                # C. PROBABILITY HELPERS
                home_ml, away_ml = g.get("home_ml_price"), g.get("away_ml_price")
                implied_h = american_to_implied_prob(home_ml)
                implied_a = american_to_implied_prob(away_ml)
                market_home_prob = implied_h if implied_h else (1.0 - implied_a if implied_a else 0.5)

                def blended_for_selection(team_side: str, m_prob_val: Optional[float], k_prob_val: Optional[float]) -> float:
                    return blended_win_prob(
                        market_prob=m_prob_val, vertex_prob=vertex_prob_home,
                        theover_prob=None, kalshi_prob=k_prob_val,
                        sentiment_diff=sentiment_diff, selection=team_side
                    )

                # D. RESTORE DATA ROWS (Moneyline, Spread, Total)
                if not (home_ml or g.get("home_spread_point") or g.get("total_point")):
                    rows_out.append({
                        "Game": f"{away} @ {home}", "Market": "None", "AI Win Prob": vertex_prob_home,
                        "Kalshi": "No Match", "Sentiment": sentiment_diff
                    })
                    continue

                if home_ml is not None:
                    pick = home if (implied_h or 0) >= (implied_a or 0) else away
                    rows_out.append({
                        "Game": f"{away} @ {home}", "Market": "Moneyline", "Pick": pick,
                        "AI Win Prob": blended_for_selection("home" if pick == home else "away", market_home_prob, kalshi_winner.get("kalshi_prob")),
                        "Kalshi Match": kalshi_winner.get("kalshi_matched")
                    })
                    master_stats["market_rows_out"] += 1

                if g.get("home_spread_point") is not None:
                    rows_out.append({
                        "Game": f"{away} @ {home}", "Market": "Spread", "Line": g.get("home_spread_point"),
                        "AI Win Prob": blended_for_selection("home", market_home_prob, kalshi_spread.get("kalshi_prob")),
                        "Kalshi Match": kalshi_spread.get("kalshi_matched")
                    })
                    master_stats["market_rows_out"] += 1

                if g.get("total_point") is not None:
                    rows_out.append({
                        "Game": f"{away} @ {home}", "Market": "Total", "Line": g.get("total_point"),
                        "AI Win Prob": blended_for_selection("home", market_home_prob, kalshi_total.get("kalshi_prob")),
                        "Kalshi Match": kalshi_total.get("kalshi_matched")
                    })
                    master_stats["market_rows_out"] += 1

            st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

with tab_kalshi:
    st.header("API Connectivity")
    if st.session_state.get("kalshi_integrator"):
        st.success("Kalshi Online")
    else:
        st.error("Kalshi Offline")

with tab_debug:
    st.write("Loaded Games Count:", len(st.session_state.get("games", [])))
