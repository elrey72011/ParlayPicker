import json
import os
import re
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import streamlit as st
from app_core.kalshi_integrator import KalshiIntegrator, LEAGUE_SERIES_MAP
from app_core.sentiment_pipeline import build_team_sentiment_map
from vertex_master_analyzer import blended_win_prob

# Must be the first Streamlit call
st.set_page_config(page_title="ParlayDesk", layout="wide")

# ------------------------------------------------------------
# Unified Kalshi Health Check
# ------------------------------------------------------------
kalshi_integrator: Optional[KalshiIntegrator] = None

def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """Unified check to replace redundant definitions."""
    try:
        ki = st.session_state.get("kalshi_integrator")
        if ki is None:
            return {"configured": False, "ok": False, "error": "Integrator not initialized."}

        markets = ki.get_sports_markets(selected_league) or []
        game_markets = [m for m in markets if str(m.get("ticker", "")).startswith("KXNBAGAME")]
        
        return {
            "configured": True,
            "ok": True,
            "market_count": len(markets),
            "game_market_count": len(game_markets),
            "has_game_markets": len(game_markets) > 0
        }
    except Exception as e:
        return {"configured": True, "ok": False, "error": str(e)}

# -----------------
# Helper Utilities
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
        if o > 0: return 100.0 / (o + 100.0)
        if o < 0: return (-o) / ((-o) + 100.0)
    except Exception: pass
    return None

def safe_iso(value: Any) -> Optional[str]:
    if value is None: return None
    if isinstance(value, datetime):
        if value.tzinfo is None: value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)

def get_local_tz() -> str:
    return st.secrets.get("APP_TIMEZONE", "America/New_York")

def parse_commence_to_utc(value: Any) -> Optional[datetime]:
    if value is None: return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            s = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception: return None
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def normalize_commence_times(games: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tz_name = get_local_tz()
    local_tz = ZoneInfo(tz_name)
    parsed, failed = 0, 0
    for g in games:
        raw_time = g.get("commence_time") or g.get("commence_time_iso")
        dt_utc = parse_commence_to_utc(raw_time)
        if dt_utc is None:
            failed += 1
            g.update({"commence_time_utc": None, "commence_date_local": None})
        else:
            parsed += 1
            g["commence_time_utc"] = dt_utc
            g["commence_time_iso_utc"] = dt_utc.isoformat().replace("+00:00", "Z")
            dt_local = dt_utc.astimezone(local_tz)
            g["commence_time_local"] = dt_local
            g["commence_date_local"] = dt_local.strftime("%Y-%m-%d")
    return games, {"parsed": parsed, "failed": failed, "timezone": tz_name}

def nba_abbrev(team_name: str) -> Optional[str]:
    mapping = {
        "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
        "charlotte hornets": "CHA", "chicago bulls": "CHI", "cleveland cavaliers": "CLE",
        "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET",
        "golden state warriors": "GSW", "houston rockets": "HOU", "indiana pacers": "IND",
        "los angeles clippers": "LAC", "la clippers": "LAC", "los angeles lakers": "LAL",
        "la lakers": "LAL", "memphis grizzlies": "MEM", "miami heat": "MIA",
        "milwaukee bucks": "MIL", "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP",
        "new york knicks": "NYK", "oklahoma city thunder": "OKC", "orlando magic": "ORL",
        "philadelphia 76ers": "PHI", "phoenix suns": "PHX", "portland trail blazers": "POR",
        "sacramento kings": "SAC", "san antonio spurs": "SAS", "toronto raptors": "TOR",
        "utah jazz": "UTA", "washington wizards": "WAS"
    }
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(team_name or "").lower()).strip()
    for key, code in mapping.items():
        if key in cleaned: return code
    return None

def fmt_local_time(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else ""

def extract_best_market(game: Dict[str, Any]) -> Dict[str, Any]:
    home, away = game.get("home_team"), game.get("away_team")
    results = {"best_ml_book": None, "home_ml_price": None, "away_ml_price": None}
    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if market["key"] == "h2h" and home in outcomes and away in outcomes:
                results.update({"best_ml_book": bm["title"], "home_ml_price": outcomes[home]["price"], "away_ml_price": outcomes[away]["price"]})
            elif market["key"] == "spreads" and home in outcomes:
                results.update({"best_spread_book": bm["title"], "home_spread_point": outcomes[home]["point"], "home_spread_price": outcomes[home]["price"]})
            elif market["key"] == "totals":
                over = outcomes.get("Over")
                if over: results.update({"best_total_book": bm["title"], "total_point": over["point"], "over_price": over["price"]})
    return results

# -----------------
# API CONFIG & STATE
# -----------------

SPORT_KEYS = {"NBA": "basketball_nba", "NFL": "americanfootball_nfl", "NHL": "icehockey_nhl", "MLB": "baseball_mlb"}
odds_api_key = read_secret("ODDS_API_KEY")
news_api_key = read_secret("NEWS_API_KEY")
vertex_endpoint_id = read_secret("VERTEX_ENDPOINT_ID")
kalshi_api_key = read_secret("KALSHI_API_KEY")
kalshi_api_secret = read_secret("KALSHI_API_SECRET")

if "kalshi_integrator" not in st.session_state:
    if kalshi_api_key and kalshi_api_secret:
        st.session_state["kalshi_integrator"] = KalshiIntegrator(kalshi_api_key, kalshi_api_secret)
    else: st.session_state["kalshi_integrator"] = None

def get_vertex_prob(game: Dict[str, Any]) -> Optional[float]:
    """Stub for Vertex score."""
    return None

# -----------------
# MAIN INTERFACE
# -----------------

st.sidebar.header("Controls")
league = st.sidebar.selectbox("League", list(SPORT_KEYS.keys()))
if st.sidebar.button("Load Games"):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEYS[league]}/odds/"
    resp = requests.get(url, params={"apiKey": odds_api_key, "regions": "us", "markets": "h2h,spreads,totals"})
    if resp.status_code == 200:
        games, _ = normalize_commence_times(resp.json())
        for g in games: g.update(extract_best_market(g))
        st.session_state["games"] = games

tab_master, tab_kalshi, tab_debug = st.tabs(["Master Analysis", "Kalshi Status", "Debug"])

with tab_master:
    st.header("Master Analysis")
    games = st.session_state.get("games", [])
    if st.button("Run Master Analysis"):
        sentiment_map, _ = build_team_sentiment_map(news_api_key, games, league) if news_api_key else ({}, {})
        kalshi_markets = st.session_state["kalshi_integrator"].get_sports_markets(league) if st.session_state["kalshi_integrator"] else []
        
        rows_out = []
        master_stats = {"market_rows_out": 0}

        # --- CORRECTED MASTER ANALYSIS LOOP ---
        for idx, g in enumerate(games):
            warnings: List[str] = list(g.get("warnings") or [])
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
        
            # 1. INITIALIZE DATA IMMEDIATELY (Prevents NameErrors)
            h_code = nba_abbrev(home)
            a_code = nba_abbrev(away)
            
            # Standardize display strings
            commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
            commence_local = fmt_local_time(g.get("commence_time_local"))
            commence_date_local = g.get("commence_date_local") or ""
            
            # Calculate all external dependencies before row generation
            vertex_prob_home = get_vertex_prob(g)
            home_sent = sentiment_map.get(home, 0.0)
            away_sent = sentiment_map.get(away, 0.0)
            sentiment_diff = home_sent - away_sent
        
            # Kalshi Matching
            filtered_markets = filter_kalshi_game_markets(
                kalshi_markets, g.get("commence_time_utc"), league_name,
                home, away, h_code, a_code
            )
            deduped = {m.get("event_ticker") or m.get("ticker"): m for m in filtered_markets}
            winner_reason_override = "winner_not_in_fetched_markets" if not filtered_markets else None
            kalshi_matches, _ = match_kalshi_market(g, list(deduped.values()), winner_reason_override)
            
            kalshi_winner = kalshi_matches.get("winner", {})

            # 2. GENERATE ROWS (Now safely uses pre-calculated variables)
            if not (g.get("home_ml_price") or g.get("home_spread_point") or g.get("total_point")):
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso,
                    "Market": "None",
                    "AI_Prob": vertex_prob_home,
                    "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                    "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1
                continue

            # Moneyline logic
            home_ml = g.get("home_ml_price")
            if home_ml is not None:
                market_home_prob = american_to_implied_prob(home_ml)
                ai_prob_row = blended_win_prob(
                    market_prob=market_home_prob, vertex_prob=vertex_prob_home,
                    kalshi_prob=kalshi_winner.get("kalshi_prob"),
                    sentiment_diff=sentiment_diff, selection="home"
                )
                
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Market": "Moneyline", "AI_Prob": ai_prob_row,
                    "Kalshi_Prob": kalshi_winner.get("kalshi_prob"),
                    "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1

        st.dataframe(pd.DataFrame(rows_out))

with tab_kalshi:
    st.json(kalshi_health_check(league))
