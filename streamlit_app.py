import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


st.set_page_config(title="ParlayDesk — Clean Core", layout="wide")


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
        # derive from h2h
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
# API Clients
# -----------------

odds_api_key = read_secret("ODDS_API_KEY")
news_api_key = read_secret("NEWS_API_KEY")
project_id = read_secret("GCP_PROJECT_ID", "elite-hangar-479017-m8")
location = read_secret("GCP_LOCATION", "us-central1")
vertex_endpoint_id = read_secret("VERTEX_ENDPOINT_ID")


@st.cache_data(ttl=60)
def fetch_odds_games() -> List[Dict[str, Any]]:
    if not odds_api_key:
        return []
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
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
    # Placeholder for real Vertex prediction. Keep resilient.
    try:
        # TODO: integrate real Vertex endpoint invocation if available.
        return None
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return None


# -----------------
# Session defaults
# -----------------

if "last_exception" not in st.session_state:
    st.session_state["last_exception"] = None
if "last_rows_out" not in st.session_state:
    st.session_state["last_rows_out"] = 0


# -----------------
# Data loading
# -----------------

try:
    all_games_raw = fetch_odds_games()
except Exception:
    st.session_state["last_exception"] = traceback.format_exc()
    all_games_raw = []

all_games_norm = [normalize_game(g) for g in all_games_raw]


# -----------------
# Tabs
# -----------------

tab_games, tab_vertex, tab_news, tab_debug = st.tabs([
    "Games & Odds",
    "Vertex Analysis",
    "News",
    "Debug",
])


with tab_games:
    st.header("Games & Odds")
    st.write(f"Loaded {len(all_games_norm)} NBA games")
    rows = []
    for g in all_games_norm:
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
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No games available. Check Odds API configuration.")


with tab_vertex:
    st.header("Vertex Analysis")
    run = st.button("Run Vertex Analysis", key="run_vertex")
    if run:
        master_stats = {
            "games_in": len(all_games_norm),
            "rows_out": 0,
            "h2h_found": 0,
            "vertex_called": 0,
            "vertex_none": 0,
            "exceptions": 0,
        }
        rows_out: List[Dict[str, Any]] = []
        for g in all_games_norm:
            warnings: List[str] = []
            league = g.get("league")
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
            # Decide pick deterministically
            if implied_home is not None and implied_away is not None:
                if implied_home >= implied_away:
                    pick = home
                    implied_pick = implied_home
                else:
                    pick = away
                    implied_pick = implied_away
            else:
                pick = home
                implied_pick = implied_home

            ai_prob = None
            try:
                master_stats["vertex_called"] += 1
                ai_prob = get_vertex_prob(g)
                if ai_prob is None:
                    master_stats["vertex_none"] += 1
                    warnings.append("vertex_none")
            except Exception:
                master_stats["exceptions"] += 1
                st.session_state["last_exception"] = traceback.format_exc()
                warnings.append("vertex_error")

            ai_edge = None
            if ai_prob is not None and implied_pick is not None:
                ai_edge = ai_prob - implied_pick

            rows_out.append(
                {
                    "League": league,
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
                    "AI_Edge": ai_edge,
                    "Warnings": ";".join(warnings),
                }
            )

        df = pd.DataFrame(rows_out)
        master_stats["rows_out"] = len(df)
        st.session_state["last_rows_out"] = len(df)

        if master_stats["games_in"] > 0 and master_stats["rows_out"] == 0:
            st.error("Master analysis produced 0 rows; see debug stats below.")
            st.json(master_stats)
        else:
            st.success(f"Produced {len(df)} rows from {len(all_games_norm)} games")
            st.dataframe(df)
            st.caption(f"rows_out/games_in = {master_stats['rows_out']} / {master_stats['games_in']}")

        st.session_state["master_stats"] = master_stats


with tab_news:
    st.header("News")
    if news_api_key:
        try:
            articles = fetch_news()
            if not articles:
                st.info("No news articles returned.")
            for art in articles:
                title = art.get("title")
                source = (art.get("source") or {}).get("name")
                published = art.get("publishedAt") or art.get("published_at")
                st.subheader(title or "(untitled)")
                st.caption(f"{source or 'Unknown'} — {published}")
                if art.get("url"):
                    st.markdown(f"[Read more]({art['url']})")
                st.write("---")
        except Exception:
            st.session_state["last_exception"] = traceback.format_exc()
            st.error("Failed to fetch news.")
    else:
        st.warning("NEWS_API_KEY not configured. Skipping news fetch.")


with tab_debug:
    st.header("Debug")
    flags = {
        "odds_api": bool(odds_api_key),
        "news_api": bool(news_api_key),
        "vertex_configured": bool(vertex_endpoint_id),
    }
    st.subheader("Config Flags")
    st.json({**flags, "project_id": project_id, "location": location})

    st.subheader("Counts")
    st.json(
        {
            "games_loaded_raw": len(all_games_raw),
            "games_normalized": len(all_games_norm),
            "last_rows_out": st.session_state.get("last_rows_out", 0),
        }
    )

    if all_games_norm:
        st.subheader("Sample normalized game")
        st.code(json.dumps(all_games_norm[0], indent=2))

    if "master_stats" in st.session_state:
        st.subheader("Master analysis stats")
        st.json(st.session_state["master_stats"])

    if st.session_state.get("last_exception"):
        st.subheader("Last exception")
        st.code(st.session_state["last_exception"])

