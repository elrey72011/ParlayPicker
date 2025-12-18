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

# DEFINE TABS FIRST (Fixes NameError)
tab_master, tab_kalshi, tab_debug = st.tabs(["Master Analysis", "Kalshi Status", "Debug"])

with tab_master:
    st.header("Best Bets Analysis")
    games = st.session_state.get("games", [])
    
    if st.button("Run Master Analysis", use_container_width=True):
        if not games:
            st.warning("No games loaded.")
        else:
            # Gather global data
            ki = st.session_state.get("kalshi_integrator")
            k_markets = ki.get_sports_markets(selected_league) if ki else []
            sentiment_map, _ = build_team_sentiment_map(news_api_key, games, selected_league) if news_api_key else ({}, {})
            
            rows_out = []
            for g in games:
                # 1. Setup game variables immediately
                home, away = g.get("home_team"), g.get("away_team")
                h_code, a_code = nba_abbrev(home), nba_abbrev(away)
                
                # 2. Calculate Brain dependencies
                v_prob = get_vertex_prob(g)
                h_sent, a_sent = sentiment_map.get(home, 0.0), sentiment_map.get(away, 0.0)
                sent_diff = h_sent - a_sent
                
                # 3. Probability Blending
                final_prob = blended_win_prob(
                    market_prob=0.5, # Placeholder baseline
                    vertex_prob=v_prob,
                    theover_prob=None,
                    kalshi_prob=None,
                    sentiment_diff=sent_diff,
                    selection="home"
                )
                
                rows_out.append({
                    "Game": f"{away} @ {home}",
                    "AI Win Prob": final_prob,
                    "Sentiment Diff": sent_diff
                })
            
            st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

with tab_kalshi:
    st.header("API Connectivity")
    if st.session_state.get("kalshi_integrator"):
        st.success("Kalshi Online")
    else:
        st.error("Kalshi Offline")

with tab_debug:
    st.write("Loaded Games Count:", len(st.session_state.get("games", [])))
