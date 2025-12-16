import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Must be the first Streamlit call
st.set_page_config(page_title="ParlayDesk", layout="wide")

# -----------------
# Helper utilities
# -----------------

def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read from st.secrets then env vars."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def american_to_implied(odds: Any) -> Optional[float]:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)


def safe_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        try:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return None


def extract_h2h_prices(game: Dict[str, Any]) -> Dict[str, Any]:
    home = game.get("home_team")
    away = game.get("away_team")
    for bm in game.get("bookmakers") or []:
        for market in bm.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes") or []
            prices = {o.get("name"): o.get("price") for o in outcomes if o.get("name")}
            if home in prices and away in prices:
                return {
                    "home_odds": prices.get(home),
                    "away_odds": prices.get(away),
                    "book": bm.get("title") or bm.get("key"),
                }
    return {"home_odds": None, "away_odds": None, "book": None}


def league_from_sport_key(sk: Optional[str]) -> Optional[str]:
    if not sk:
        return None
    if sk == "basketball_nba":
        return "NBA"
    if sk == "basketball_ncaab":
        return "NCAAB"
    if sk == "americanfootball_nfl":
        return "NFL"
    if sk == "americanfootball_ncaaf":
        return "NCAAF"
    if sk == "icehockey_nhl":
        return "NHL"
    if sk == "baseball_mlb":
        return "MLB"
    return sk.upper()


def normalize_game(game: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(game)
    normalized["league"] = league_from_sport_key(game.get("sport_key")) or "UNKNOWN"
    normalized["commence_time_iso"] = safe_iso(game.get("commence_time")) or game.get(
        "commence_time_iso"
    )

    home = game.get("home_team") or "UNKNOWN_HOME"
    away = game.get("away_team")
    warnings: List[str] = []
    if not away:
        for bm in game.get("bookmakers") or []:
            for m in bm.get("markets") or []:
                if m.get("key") != "h2h":
                    continue
                names = [o.get("name") for o in m.get("outcomes") or [] if o.get("name")]
                uniq = list({n for n in names if n and n.lower() not in {"over", "under"}})
                if len(uniq) == 2:
                    if home in uniq:
                        other = uniq[0] if uniq[1] == home else uniq[1]
                        away = other
                    else:
                        home, away = uniq[0], uniq[1]
                    break
        if not away:
            away = "UNKNOWN_AWAY"
            warnings.append("missing_away_team")
    normalized["home_team"] = home
    normalized["away_team"] = away
    normalized.setdefault("warnings", warnings)
    return normalized


# -----------------
# API Clients & config
# -----------------

SPORT_KEYS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}

odds_api_key = read_secret("ODDS_API_KEY")
news_api_key = read_secret("NEWS_API_KEY")
project_id = read_secret("GCP_PROJECT_ID", "elite-hangar-479017-m8")
location = read_secret("GCP_LOCATION", "us-central1")
vertex_endpoint_id = read_secret("VERTEX_ENDPOINT_ID")
kalshi_api_key = read_secret("KALSHI_API_KEY")
kalshi_api_secret = read_secret("KALSHI_API_SECRET")


@st.cache_data(ttl=60)
def fetch_odds_games(sport_key: str) -> List[Dict[str, Any]]:
    if not odds_api_key or not sport_key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": odds_api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_news() -> List[Dict[str, Any]]:
    if not news_api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "NBA basketball",
        "sortBy": "publishedAt",
        "pageSize": 3,
        "apiKey": news_api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


# -----------------
# Vertex stub
# -----------------

def get_vertex_prob(game: Dict[str, Any]) -> Optional[float]:
    """Stubbed Vertex call: return None if not configured or on error."""
    if not vertex_endpoint_id:
        return None
    try:
        return None
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return None


# -----------------
# Kalshi stubs
# -----------------

def kalshi_health_check() -> Dict[str, Any]:
    configured = bool(kalshi_api_key and kalshi_api_secret)
    if not configured:
        return {
            "configured": False,
            "ok": False,
            "market_count": 0,
            "sample_market": None,
            "error": "Kalshi is required but not configured.",
        }
    # Placeholder: real API integration to be added; treat as available for now.
    return {
        "configured": True,
        "ok": True,
        "market_count": 0,
        "sample_market": None,
        "error": None,
    }


def match_kalshi_market(game: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kalshi_available": False,
        "kalshi_label": "no_match",
        "kalshi_event_ticker": None,
        "kalshi_reason": "Kalshi matching not yet implemented",
    }


# -----------------
# Session defaults
# -----------------

if "last_exception" not in st.session_state:
    st.session_state["last_exception"] = None
if "last_rows_out" not in st.session_state:
    st.session_state["last_rows_out"] = 0
if "games" not in st.session_state:
    st.session_state["games"] = []
if "league" not in st.session_state:
    st.session_state["league"] = "NBA"


# -----------------
# Data loading helpers
# -----------------

def load_games(selected_league: str) -> List[Dict[str, Any]]:
    sport_key = SPORT_KEYS.get(selected_league)
    if not sport_key:
        st.session_state["last_exception"] = f"Unknown league: {selected_league}"
        return []
    try:
        games_raw = fetch_odds_games(sport_key)
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []
    normalized = [normalize_game({**g, "sport_key": sport_key}) for g in games_raw]
    st.session_state["games"] = normalized
    return normalized


# -----------------
# Sidebar
# -----------------

st.sidebar.header("Controls")
league = st.sidebar.selectbox("League", list(SPORT_KEYS.keys()), index=list(SPORT_KEYS.keys()).index(st.session_state.get("league", "NBA")))
st.session_state["league"] = league
if st.sidebar.button("Load Games", use_container_width=True):
    load_games(league)

st.sidebar.markdown("---")
st.sidebar.subheader("Status")
badges = {
    "OddsAPI": bool(odds_api_key),
    "Vertex": bool(vertex_endpoint_id),
    "News": bool(news_api_key),
    "API-Sports": False,
    "SportsData": False,
    "Kalshi": bool(kalshi_api_key and kalshi_api_secret),
}
for name, ok in badges.items():
    color = "green" if ok else "red"
    st.sidebar.markdown(f"**{name}:** :{color}[{'OK' if ok else 'Missing'}]")


# -----------------
# Tabs
# -----------------

tab_games, tab_master, tab_kalshi, tab_sentiment, tab_debug = st.tabs(
    ["Games & Odds", "Master Analysis", "Kalshi", "Sentiment", "Debug"]
)


with tab_games:
    st.header("Games & Odds")
    games = st.session_state.get("games", [])
    if not games:
        st.info("Load games from the sidebar to begin.")
    else:
        rows = []
        for g in games:
            markets = set()
            for bm in g.get("bookmakers") or []:
                for m in bm.get("markets") or []:
                    if m.get("key"):
                        markets.add(m.get("key"))
            rows.append(
                {
                    "League": g.get("league"),
                    "Home": g.get("home_team"),
                    "Away": g.get("away_team"),
                    "Commence (UTC)": safe_iso(g.get("commence_time_iso")),
                    "Books": len(g.get("bookmakers") or []),
                    "MarketsAvailable": ", ".join(sorted(markets)),
                }
            )
        st.dataframe(pd.DataFrame(rows))


with tab_master:
    st.header("Master Analysis")
    kalshi_status = kalshi_health_check()
    if not kalshi_status.get("ok"):
        st.error(kalshi_status.get("error") or "Kalshi is required but unavailable.")
        st.info("Master Analysis is disabled until Kalshi is available.")
    run_master = st.button(
        "Run Master Analysis",
        key="run_master",
        disabled=not kalshi_status.get("ok"),
        help="Requires Kalshi availability",
    )
    games = st.session_state.get("games", [])
    if run_master:
        rows_out: List[Dict[str, Any]] = []
        master_stats = {
            "games_in": len(games),
            "rows_out": 0,
            "h2h_found": 0,
            "exceptions": 0,
        }
        for g in games:
            warnings: List[str] = []
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
            commence_iso = safe_iso(g.get("commence_time_iso"))

            h2h = extract_h2h_prices(g)
            if h2h.get("home_odds") is not None and h2h.get("away_odds") is not None:
                master_stats["h2h_found"] += 1
            home_p = american_to_implied(h2h.get("home_odds"))
            away_p = american_to_implied(h2h.get("away_odds"))
            implied_home = home_p
            implied_away = away_p
            if implied_home is not None and implied_away is not None:
                if implied_home >= implied_away:
                    pick = home
                    implied_pick = implied_home
                else:
                    pick = away
                    implied_pick = implied_away
            elif implied_home is not None:
                pick = home
                implied_pick = implied_home
            elif implied_away is not None:
                pick = away
                implied_pick = implied_away
            else:
                pick = home
                implied_pick = None

            kalshi_match = match_kalshi_market(g)
            if not kalshi_match.get("kalshi_available"):
                warnings.append("kalshi_no_match")

            try:
                ai_prob = get_vertex_prob(g)
            except Exception:
                ai_prob = None
                warnings.append("vertex_error")
                st.session_state["last_exception"] = traceback.format_exc()

            rows_out.append(
                {
                    "League": league_name,
                    "Home": home,
                    "Away": away,
                    "Commence (UTC)": commence_iso,
                    "Market": "Moneyline",
                    "Book": h2h.get("book"),
                    "Home_ML": h2h.get("home_odds"),
                    "Away_ML": h2h.get("away_odds"),
                    "Pick": pick,
                    "Implied_Prob": implied_pick,
                    "AI_Prob": ai_prob,
                    "Warnings": ";".join(warnings),
                    "kalshi_available": kalshi_match.get("kalshi_available"),
                    "kalshi_label": kalshi_match.get("kalshi_label"),
                    "kalshi_event_ticker": kalshi_match.get("kalshi_event_ticker"),
                }
            )

        df = pd.DataFrame(rows_out)
        master_stats["rows_out"] = len(df)
        st.session_state["last_rows_out"] = len(df)
        st.session_state["master_stats"] = master_stats

        if master_stats["games_in"] > 0 and master_stats["rows_out"] == 0:
            st.error("Master analysis produced 0 rows; see debug stats below.")
            st.json(master_stats)
        elif not games:
            st.warning("No games loaded. Use the sidebar to load games first.")
        else:
            st.success(f"Produced {len(df)} rows from {len(games)} games")
            st.dataframe(df)
            st.caption(
                f"rows_out/games_in = {master_stats['rows_out']} / {master_stats['games_in']}"
            )
    elif not games:
        st.info("Load games from the sidebar, then run Master Analysis.")


with tab_kalshi:
    st.header("Kalshi Health")
    kalshi_status = kalshi_health_check()
    st.json(kalshi_status)
    if not kalshi_status.get("configured"):
        st.error("Kalshi is required but not configured.")
    elif not kalshi_status.get("ok"):
        st.error("Kalshi is configured but unavailable. Fix keys/API and retry.")
    else:
        st.success("Kalshi credentials detected. Full integration coming soon.")


with tab_sentiment:
    st.header("Sentiment (Stub)")
    st.info("Sentiment analysis integration coming soon.")


with tab_debug:
    st.header("Debug")
    flags = {
        "odds_api": bool(odds_api_key),
        "news_api": bool(news_api_key),
        "vertex_configured": bool(vertex_endpoint_id),
        "kalshi_configured": bool(kalshi_api_key and kalshi_api_secret),
    }
    st.subheader("Config Flags")
    st.json({**flags, "project_id": project_id, "location": location})

    games = st.session_state.get("games", [])
    st.subheader("Counts")
    st.json(
        {
            "games_loaded_raw": len(games),
            "games_normalized": len(games),
            "last_rows_out": st.session_state.get("last_rows_out", 0),
        }
    )

    if games:
        st.subheader("Sample normalized game")
        st.code(json.dumps(games[0], indent=2))

    st.subheader("Kalshi health")
    st.json(kalshi_health_check())

    if "master_stats" in st.session_state:
        st.subheader("Master analysis stats")
        st.json(st.session_state["master_stats"])

    if st.session_state.get("last_exception"):
        st.subheader("Last exception")
        st.code(st.session_state["last_exception"])
