import json
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from app_core.kalshi_integrator import (
    KalshiIntegrator,
    LEAGUE_SERIES_MAP,
    match_game_to_kalshi,
    price_to_prob,
)

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


def american_to_implied_prob(odds: Any) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None


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


def get_local_tz() -> str:
    tz_name = None
    try:
        tz_name = st.secrets.get("APP_TIMEZONE")
    except Exception:
        tz_name = None
    if not tz_name:
        tz_name = "America/New_York"
    return tz_name


def parse_commence_to_utc(value: Any) -> Optional[datetime]:
    raw = value
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            s = str(raw)
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_commence_times(games: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tz_name = get_local_tz()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = None
    parsed = 0
    failed = 0
    for g in games:
        warnings = list(g.get("warnings") or [])
        raw_time = g.get("commence_time") or g.get("commence_time_iso")
        dt_utc = parse_commence_to_utc(raw_time)
        if dt_utc is None:
            failed += 1
            warnings.append("commence_parse_failed")
            g["commence_time_utc"] = None
            g["commence_time_iso_utc"] = None
            g["commence_time_local"] = None
            g["commence_time_iso_local"] = None
            g["commence_date_local"] = None
        else:
            parsed += 1
            g["commence_time_utc"] = dt_utc
            iso_utc = dt_utc.isoformat().replace("+00:00", "Z")
            g["commence_time_iso_utc"] = iso_utc
            if local_tz:
                dt_local = dt_utc.astimezone(local_tz)
                g["commence_time_local"] = dt_local
                g["commence_time_iso_local"] = dt_local.isoformat()
                g["commence_date_local"] = dt_local.strftime("%Y-%m-%d")
            else:
                g["commence_time_local"] = None
                g["commence_time_iso_local"] = None
                g["commence_date_local"] = None
        g["warnings"] = warnings
    stats = {"parsed": parsed, "failed": failed, "timezone": tz_name}
    return games, stats


def fmt_local_time(dt: Optional[datetime]) -> str:
    try:
        if dt is None:
            return ""
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


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


def _parse_last_update(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = str(value)
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def extract_best_market(game: Dict[str, Any]) -> Dict[str, Any]:
    home = game.get("home_team")
    away = game.get("away_team")
    bookmakers = game.get("bookmakers") or []
    warnings: List[str] = list(game.get("warnings") or [])
    if not bookmakers:
        warnings.append("missing_bookmakers")

    best_ml = None
    best_spread = None
    best_total = None

    for bm in bookmakers:
        bm_name = bm.get("title") or bm.get("key")
        last_update = _parse_last_update(bm.get("last_update"))
        for market in bm.get("markets") or []:
            key = market.get("key")
            outcomes = market.get("outcomes") or []
            if key == "h2h":
                prices = {o.get("name"): o.get("price") for o in outcomes if o.get("name")}
                home_price = prices.get(home)
                away_price = prices.get(away)
                if home_price is None or away_price is None:
                    continue
                quality = max(abs(float(home_price)), abs(float(away_price))) if home_price and away_price else 0
                candidate = {
                    "book": bm_name,
                    "home_price": home_price,
                    "away_price": away_price,
                    "quality": quality,
                    "last_update": last_update,
                }
                if not best_ml or quality > best_ml["quality"]:
                    best_ml = candidate
                elif best_ml and quality == best_ml.get("quality"):
                    if last_update and best_ml.get("last_update"):
                        if last_update > best_ml["last_update"]:
                            best_ml = candidate
            elif key == "spreads":
                price_map = {o.get("name"): (o.get("point"), o.get("price")) for o in outcomes if o.get("name")}
                if home in price_map and away in price_map:
                    home_point, home_price = price_map.get(home)
                    away_point, away_price = price_map.get(away)
                    if home_point is None or away_point is None:
                        continue
                    quality = max(
                        abs(float(home_price)) if home_price is not None else 0,
                        abs(float(away_price)) if away_price is not None else 0,
                    )
                    candidate = {
                        "book": bm_name,
                        "home_point": home_point,
                        "home_price": home_price,
                        "away_point": away_point,
                        "away_price": away_price,
                        "quality": quality,
                        "last_update": last_update,
                    }
                    if not best_spread or quality > best_spread["quality"]:
                        best_spread = candidate
                    elif best_spread and quality == best_spread.get("quality"):
                        if last_update and best_spread.get("last_update"):
                            if last_update > best_spread["last_update"]:
                                best_spread = candidate
            elif key == "totals":
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if over and under:
                    over_point = over.get("point")
                    under_point = under.get("point")
                    if over_point is None or under_point is None or over_point != under_point:
                        continue
                    over_price = over.get("price")
                    under_price = under.get("price")
                    quality = max(
                        abs(float(over_price)) if over_price is not None else 0,
                        abs(float(under_price)) if under_price is not None else 0,
                    )
                    candidate = {
                        "book": bm_name,
                        "point": over_point,
                        "over_price": over_price,
                        "under_price": under_price,
                        "quality": quality,
                        "last_update": last_update,
                    }
                    if not best_total or quality > best_total["quality"]:
                        best_total = candidate
                    elif best_total and quality == best_total.get("quality"):
                        if last_update and best_total.get("last_update"):
                            if last_update > best_total["last_update"]:
                                best_total = candidate

    if not best_ml:
        warnings.append("missing_h2h")
    if not best_spread:
        warnings.append("missing_spreads")
    if not best_total:
        warnings.append("missing_totals")

    return {
        "best_ml_book": best_ml.get("book") if best_ml else None,
        "home_ml_price": best_ml.get("home_price") if best_ml else None,
        "away_ml_price": best_ml.get("away_price") if best_ml else None,
        "implied_prob_home": american_to_implied_prob(best_ml.get("home_price")) if best_ml else None,
        "implied_prob_away": american_to_implied_prob(best_ml.get("away_price")) if best_ml else None,
        "best_spread_book": best_spread.get("book") if best_spread else None,
        "home_spread_point": best_spread.get("home_point") if best_spread else None,
        "home_spread_price": best_spread.get("home_price") if best_spread else None,
        "away_spread_point": best_spread.get("away_point") if best_spread else None,
        "away_spread_price": best_spread.get("away_price") if best_spread else None,
        "best_total_book": best_total.get("book") if best_total else None,
        "total_point": best_total.get("point") if best_total else None,
        "over_price": best_total.get("over_price") if best_total else None,
        "under_price": best_total.get("under_price") if best_total else None,
        "warnings": warnings,
    }


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
kalshi_api_key = read_secret("KALSHI_API_KEY") or read_secret("kalshi_api_key")
kalshi_api_secret = read_secret("KALSHI_API_SECRET") or read_secret("kalshi_api_secret")
kalshi_integrator: Optional[KalshiIntegrator] = None
try:
    if kalshi_api_key and kalshi_api_secret:
        kalshi_integrator = KalshiIntegrator(kalshi_api_key, kalshi_api_secret)
except Exception:
    st.session_state["last_exception"] = traceback.format_exc()
    kalshi_integrator = None


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
# Kalshi integration
# -----------------


@st.cache_data(ttl=60)
def fetch_kalshi_markets(selected_league: str) -> List[Dict[str, Any]]:
    if not kalshi_integrator:
        return []
    try:
        markets = kalshi_integrator.get_sports_markets()
        prefix = LEAGUE_SERIES_MAP.get((selected_league or "").upper())
        if prefix:
            markets = [
                m for m in markets if str(m.get("ticker") or "").upper().startswith(prefix)
            ]
        return markets or []
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


def pick_sample_game_market(
    markets: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    def parse_iso(dt_value: Any) -> Optional[datetime]:
        try:
            if not dt_value:
                return None
            raw = str(dt_value)
            if raw.endswith("Z"):
                raw = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed
        except Exception:
            return None

    now_utc = datetime.now(timezone.utc)
    window = timedelta(days=7)
    best_market: Optional[Dict[str, Any]] = None
    best_reason: Optional[str] = None
    best_within = False
    best_diff: Optional[timedelta] = None
    for m in markets or []:
        try:
            title = str(m.get("title") or "")
            ticker = str(m.get("event_ticker") or m.get("ticker") or "")
            lower_title = title.lower()
            reason = None
            if any(tok in lower_title for tok in [" vs ", " at ", "@"]):
                reason = "contains_vs"
            elif "game winner" in lower_title:
                reason = "contains_game_winner"
            elif any(tok in lower_title for tok in ["spread", "total"]):
                reason = "contains_spread_total"
            elif ticker and "game" in ticker.lower():
                reason = "contains_game_ticker"
            if not reason:
                continue
            close_dt = parse_iso(m.get("close_time")) or parse_iso(m.get("expiration_time"))
            within_window = False
            time_diff: Optional[timedelta] = None
            if close_dt:
                time_diff = abs(close_dt - now_utc)
                within_window = timedelta(0) <= (close_dt - now_utc) <= window
            reason_with_window = reason
            if within_window:
                reason_with_window = f"{reason}+within_7d"
            if best_market is None:
                best_market = m
                best_reason = reason_with_window
                best_within = within_window
                best_diff = time_diff
                continue
            if within_window and not best_within:
                best_market = m
                best_reason = reason_with_window
                best_within = True
                best_diff = time_diff
                continue
            if within_window == best_within:
                if time_diff is not None and (best_diff is None or time_diff < best_diff):
                    best_market = m
                    best_reason = reason_with_window
                    best_diff = time_diff
        except Exception:
            continue
    return best_market, best_reason


def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    configured = bool(kalshi_integrator and kalshi_api_key and kalshi_api_secret)
    if not configured:
        return {
            "configured": False,
            "ok": False,
            "market_count": 0,
            "sample_market": None,
            "sample_game_market": None,
            "sample_game_market_reason": None,
            "error": "Kalshi is required but not configured.",
        }
    try:
        markets = fetch_kalshi_markets(selected_league)
    except Exception:
        markets = []
    market_count = len(markets)
    ok = market_count > 0
    sample_game_market, sample_reason = pick_sample_game_market(markets)
    return {
        "configured": configured,
        "ok": ok,
        "market_count": market_count,
        "sample_market": markets[0] if markets else None,
        "sample_game_market": sample_game_market,
        "sample_game_market_reason": sample_reason,
        "error": None
        if ok
        else "Kalshi is required but unavailable. Fix keys / API and retry.",
    }


def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
) -> List[Dict[str, Any]]:
    def parse_iso(dt_value: Any) -> Optional[datetime]:
        try:
            if not dt_value:
                return None
            raw = str(dt_value)
            if raw.endswith("Z"):
                raw = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed
        except Exception:
            return None

    window = timedelta(hours=72)
    try:
        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_iso(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)
        filtered: List[Dict[str, Any]] = []
        for m in markets or []:
            try:
                title = str(m.get("title") or "")
                ticker = str(m.get("event_ticker") or m.get("ticker") or "")
                lower_title = title.lower()
                looks_game = any(
                    token in lower_title for token in [" vs ", " at ", "@", " - "]
                ) or (ticker and "game" in ticker.lower())
                if not looks_game:
                    continue
                candidate_time = parse_iso(m.get("close_time")) or parse_iso(
                    m.get("expiration_time")
                )
                if game_dt:
                    if candidate_time and abs(candidate_time - game_dt) <= window:
                        filtered.append(m)
                    elif candidate_time is None:
                        continue
                else:
                    filtered.append(m)
            except Exception:
                continue
        return filtered
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


def match_kalshi_market(game: Dict[str, Any], kalshi_markets: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "kalshi_available": False,
        "kalshi_label": None,
        "kalshi_event_ticker": None,
        "kalshi_reason": "kalshi_not_configured",
        "kalshi_matched": False,
        "kalshi_prob": None,
        "kalshi_market_type": None,
        "kalshi_match_score": None,
    }
    if not kalshi_integrator:
        return base
    try:
        kalshi_integrator._markets_cache = kalshi_markets
        kalshi_integrator._markets_cache_ts = time.time()
        result = match_game_to_kalshi(
            game.get("league"),
            game.get("home_team"),
            game.get("away_team"),
            game.get("commence_time_utc"),
            integrator=kalshi_integrator,
            status="open",
        )
        return {
            "kalshi_available": result.kalshi_available,
            "kalshi_label": result.label,
            "kalshi_event_ticker": result.raw_event_id,
            "kalshi_reason": result.reason,
            "kalshi_matched": result.matched,
            "kalshi_prob": result.probability if result else None,
            "kalshi_market_type": result.market_type,
            "kalshi_match_score": None,
        }
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return {**base, "kalshi_reason": "kalshi_match_error"}


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
if "commence_stats" not in st.session_state:
    st.session_state["commence_stats"] = {"parsed": 0, "failed": 0, "timezone": get_local_tz()}
if "market_counts" not in st.session_state:
    st.session_state["market_counts"] = {
        "moneyline_available_count": 0,
        "spreads_available_count": 0,
        "totals_available_count": 0,
    }


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
    with_times, commence_stats = normalize_commence_times(normalized)
    moneyline_count = 0
    spreads_count = 0
    totals_count = 0
    for g in with_times:
        try:
            best = extract_best_market(g)
            warnings = list(best.pop("warnings", []))
            merged_warnings = list(dict.fromkeys((g.get("warnings") or []) + warnings))
            g.update(best)
            g["warnings"] = merged_warnings
            if g.get("best_ml_book") is not None:
                moneyline_count += 1
            if g.get("best_spread_book") is not None:
                spreads_count += 1
            if g.get("best_total_book") is not None:
                totals_count += 1
        except Exception:
            g["warnings"] = list(g.get("warnings") or []) + ["odds_extract_error"]
            st.session_state["last_exception"] = traceback.format_exc()
    st.session_state["games"] = with_times
    st.session_state["commence_stats"] = commence_stats
    st.session_state["market_counts"] = {
        "moneyline_available_count": moneyline_count,
        "spreads_available_count": spreads_count,
        "totals_available_count": totals_count,
    }
    return with_times


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
                    "Commence (UTC)": g.get("commence_time_iso_utc")
                    or safe_iso(g.get("commence_time_iso")),
                    "Commence (Local)": fmt_local_time(g.get("commence_time_local")),
                    "Local Date": g.get("commence_date_local") or "",
                    "Books": len(g.get("bookmakers") or []),
                    "MarketsAvailable": ", ".join(sorted(markets)),
                    "home_ml_price": g.get("home_ml_price"),
                    "away_ml_price": g.get("away_ml_price"),
                    "implied_prob_home": g.get("implied_prob_home"),
                    "implied_prob_away": g.get("implied_prob_away"),
                    "home_spread_point": g.get("home_spread_point"),
                    "home_spread_price": g.get("home_spread_price"),
                    "total_point": g.get("total_point"),
                    "over_price": g.get("over_price"),
                    "under_price": g.get("under_price"),
                    "warnings": ",".join(g.get("warnings") or []),
                }
            )
        st.dataframe(pd.DataFrame(rows))


with tab_master:
    st.header("Master Analysis")
    kalshi_status = kalshi_health_check(league)
    if not kalshi_status.get("ok"):
        st.error(
            "Kalshi is required and is not healthy (missing keys / 0 markets / auth error). Fix Kalshi first."
        )
        st.info("Master Analysis is disabled until Kalshi is available.")
    run_master = st.button(
        "Run Master Analysis",
        key="run_master",
        disabled=not kalshi_status.get("ok"),
        help="Requires Kalshi availability",
    )
    games = st.session_state.get("games", [])
    if run_master and not kalshi_status.get("ok"):
        st.error("Kalshi is required but unavailable. Fix Kalshi first.")
        st.stop()
    if run_master:
        kalshi_markets = fetch_kalshi_markets(league)
        if not kalshi_markets:
            st.error(
                "Kalshi is required but unavailable. Fix keys / API and retry."
            )
            st.stop()
        filtered_counts: List[int] = []
        sample_filtered_for_first_game: List[Dict[str, Any]] = []
        rows_out: List[Dict[str, Any]] = []
        master_stats = {
            "games_in": len(games),
            "rows_out": 0,
            "h2h_found": 0,
            "exceptions": 0,
            "market_rows_out": 0,
            "kalshi_matches": 0,
            "kalshi_total": len(games),
        }
        kalshi_match_results: List[Dict[str, Any]] = []
        for g in games:
            warnings: List[str] = list(g.get("warnings") or [])
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
            commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
            commence_local = fmt_local_time(g.get("commence_time_local"))
            commence_date_local = g.get("commence_date_local") or ""
            filtered_markets = filter_kalshi_game_markets(
                kalshi_markets, g.get("commence_time_utc"), league_name
            )
            filtered_counts.append(len(filtered_markets))
            if not sample_filtered_for_first_game:
                sample_filtered_for_first_game = [
                    {
                        "title": fm.get("title"),
                        "ticker": fm.get("event_ticker") or fm.get("ticker"),
                    }
                    for fm in filtered_markets[:10]
                ]
            if not filtered_markets:
                kalshi_match = {
                    "kalshi_available": bool(kalshi_integrator),
                    "kalshi_label": None,
                    "kalshi_event_ticker": None,
                    "kalshi_reason": "no_game_like_markets_in_window",
                    "kalshi_matched": False,
                    "kalshi_prob": None,
                    "kalshi_market_type": None,
                    "kalshi_match_score": None,
                }
            else:
                kalshi_match = match_kalshi_market(g, filtered_markets)
            if not kalshi_match.get("kalshi_matched"):
                warnings.append(f"kalshi_{kalshi_match.get('kalshi_reason')}")
            else:
                master_stats["kalshi_matches"] += 1
            kalshi_match_results.append({
                "home": home,
                "away": away,
                "league": league_name,
                **kalshi_match,
            })

            ai_prob = None
            try:
                ai_prob = get_vertex_prob(g)
            except Exception:
                ai_prob = None
                warnings.append("vertex_error")
                st.session_state["last_exception"] = traceback.format_exc()

            moneyline_row_added = False
            spread_row_added = False
            total_row_added = False

            home_ml = g.get("home_ml_price")
            away_ml = g.get("away_ml_price")
            implied_home = american_to_implied_prob(home_ml)
            implied_away = american_to_implied_prob(away_ml)
            if home_ml is not None or away_ml is not None:
                pick = home
                implied_pick = implied_home
                if implied_home is not None and implied_away is not None:
                    if implied_home >= implied_away:
                        pick = home
                        implied_pick = implied_home
                    else:
                        pick = away
                        implied_pick = implied_away
                elif implied_home is None and implied_away is not None:
                    pick = away
                    implied_pick = implied_away
                rows_out.append(
                    {
                        "League": league_name,
                        "Home": home,
                        "Away": away,
                        "Commence (UTC)": commence_iso,
                        "Commence (Local)": commence_local,
                        "Local Date": commence_date_local,
                        "Market": "Moneyline",
                        "Book": g.get("best_ml_book"),
                        "Home_ML": home_ml,
                        "Away_ML": away_ml,
                        "Pick": pick,
                        "Implied_Prob": implied_pick,
                        "AI_Prob": ai_prob,
                        "Warnings": ";".join(warnings),
                        "kalshi_available": kalshi_match.get("kalshi_available"),
                        "kalshi_label": kalshi_match.get("kalshi_label"),
                        "kalshi_event_ticker": kalshi_match.get("kalshi_event_ticker"),
                        "kalshi_matched": kalshi_match.get("kalshi_matched"),
                        "kalshi_prob": kalshi_match.get("kalshi_prob"),
                        "kalshi_reason": kalshi_match.get("kalshi_reason"),
                    }
                )
                master_stats["h2h_found"] += 1
                moneyline_row_added = True
                master_stats["market_rows_out"] += 1

            if (
                g.get("home_spread_point") is not None
                and g.get("away_spread_point") is not None
                and g.get("home_spread_price") is not None
                and g.get("away_spread_price") is not None
            ):
                spread_pick = home
                home_spread_price = g.get("home_spread_price")
                away_spread_price = g.get("away_spread_price")
                spread_pick_price = home_spread_price
                if away_spread_price is not None and home_spread_price is not None:
                    if float(away_spread_price) > float(home_spread_price):
                        spread_pick = away
                        spread_pick_price = away_spread_price
                elif away_spread_price is not None:
                    spread_pick = away
                    spread_pick_price = away_spread_price
                rows_out.append(
                    {
                        "League": league_name,
                        "Home": home,
                        "Away": away,
                        "Commence (UTC)": commence_iso,
                        "Commence (Local)": commence_local,
                        "Local Date": commence_date_local,
                        "Market": "Spread",
                        "Book": g.get("best_spread_book"),
                        "Home_Spread": g.get("home_spread_point"),
                        "Home_Spread_Price": g.get("home_spread_price"),
                        "Away_Spread": g.get("away_spread_point"),
                        "Away_Spread_Price": g.get("away_spread_price"),
                        "Pick": spread_pick,
                        "Implied_Prob": american_to_implied_prob(spread_pick_price),
                        "AI_Prob": ai_prob,
                        "Warnings": ";".join(warnings),
                        "kalshi_available": kalshi_match.get("kalshi_available"),
                        "kalshi_label": kalshi_match.get("kalshi_label"),
                        "kalshi_event_ticker": kalshi_match.get("kalshi_event_ticker"),
                        "kalshi_matched": kalshi_match.get("kalshi_matched"),
                        "kalshi_prob": kalshi_match.get("kalshi_prob"),
                        "kalshi_reason": kalshi_match.get("kalshi_reason"),
                    }
                )
                spread_row_added = True
                master_stats["market_rows_out"] += 1

            if (
                g.get("total_point") is not None
                and g.get("over_price") is not None
                and g.get("under_price") is not None
            ):
                total_pick = "Over"
                over_price = g.get("over_price")
                under_price = g.get("under_price")
                total_pick_price = over_price
                if under_price is not None and over_price is not None:
                    if float(under_price) > float(over_price):
                        total_pick = "Under"
                        total_pick_price = under_price
                elif under_price is not None:
                    total_pick = "Under"
                    total_pick_price = under_price
                rows_out.append(
                    {
                        "League": league_name,
                        "Home": home,
                        "Away": away,
                        "Commence (UTC)": commence_iso,
                        "Commence (Local)": commence_local,
                        "Local Date": commence_date_local,
                        "Market": "Total",
                        "Book": g.get("best_total_book"),
                        "Total_Point": g.get("total_point"),
                        "Over_Price": g.get("over_price"),
                        "Under_Price": g.get("under_price"),
                        "Pick": total_pick,
                        "Implied_Prob": american_to_implied_prob(total_pick_price),
                        "AI_Prob": ai_prob,
                        "Warnings": ";".join(warnings),
                        "kalshi_available": kalshi_match.get("kalshi_available"),
                        "kalshi_label": kalshi_match.get("kalshi_label"),
                        "kalshi_event_ticker": kalshi_match.get("kalshi_event_ticker"),
                        "kalshi_matched": kalshi_match.get("kalshi_matched"),
                        "kalshi_prob": kalshi_match.get("kalshi_prob"),
                        "kalshi_reason": kalshi_match.get("kalshi_reason"),
                    }
                )
                total_row_added = True
                master_stats["market_rows_out"] += 1

            if not (moneyline_row_added or spread_row_added or total_row_added):
                warnings = list(dict.fromkeys(warnings + ["no_markets"]))
                rows_out.append(
                    {
                        "League": league_name,
                        "Home": home,
                        "Away": away,
                        "Commence (UTC)": commence_iso,
                        "Commence (Local)": commence_local,
                        "Local Date": commence_date_local,
                        "Market": "None",
                        "Book": None,
                        "Pick": None,
                        "Implied_Prob": None,
                        "AI_Prob": ai_prob,
                        "Warnings": ";".join(warnings),
                        "kalshi_available": kalshi_match.get("kalshi_available"),
                        "kalshi_label": kalshi_match.get("kalshi_label"),
                        "kalshi_event_ticker": kalshi_match.get("kalshi_event_ticker"),
                        "kalshi_matched": kalshi_match.get("kalshi_matched"),
                        "kalshi_prob": kalshi_match.get("kalshi_prob"),
                        "kalshi_reason": kalshi_match.get("kalshi_reason"),
                    }
                )
                master_stats["market_rows_out"] += 1

        df = pd.DataFrame(rows_out)
        master_stats["rows_out"] = len(df)
        st.session_state["last_rows_out"] = len(df)
        st.session_state["master_stats"] = master_stats
        st.session_state["kalshi_match_results"] = kalshi_match_results
        st.session_state["kalshi_filter_stats"] = {
            "total_markets_fetched": len(kalshi_markets),
            "avg_filtered_markets_per_game": sum(filtered_counts) / len(filtered_counts)
            if filtered_counts
            else 0,
            "filtered_markets_min": min(filtered_counts) if filtered_counts else 0,
            "filtered_markets_max": max(filtered_counts) if filtered_counts else 0,
            "sample_filtered_first_game": sample_filtered_for_first_game,
        }
        matches = master_stats.get("kalshi_matches", 0)
        total_games = master_stats.get("kalshi_total", 0) or 1
        st.caption(
            f"Kalshi matches: {matches}/{total_games} ({matches/total_games:.1%})"
        )

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
    kalshi_status = kalshi_health_check(league)
    st.json(kalshi_status)
    if not kalshi_status.get("configured"):
        st.error("Kalshi is required but not configured.")
    elif not kalshi_status.get("ok"):
        st.error("Kalshi is required but unavailable. Fix keys/API and retry.")
    else:
        st.success("Kalshi credentials detected and markets available.")


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
            "moneyline_available_count": st.session_state.get("market_counts", {}).get(
                "moneyline_available_count", 0
            ),
            "spreads_available_count": st.session_state.get("market_counts", {}).get(
                "spreads_available_count", 0
            ),
            "totals_available_count": st.session_state.get("market_counts", {}).get(
                "totals_available_count", 0
            ),
            "market_rows_out": st.session_state.get("master_stats", {}).get(
                "market_rows_out", 0
            ),
        }
    )

    st.subheader("Timezones")
    commence_stats = st.session_state.get("commence_stats", {})
    st.json({"timezone_used": commence_stats.get("timezone") or get_local_tz()})
    if games:
        samples = []
        for g in games[:3]:
            samples.append(
                {
                    "home": g.get("home_team"),
                    "away": g.get("away_team"),
                    "utc": g.get("commence_time_iso_utc"),
                    "local": g.get("commence_time_iso_local"),
                }
            )
        st.caption("Sample commence conversions (first 3 games)")
        st.json(samples)

    if games:
        st.subheader("Sample normalized game")
        st.code(json.dumps(games[0], indent=2, default=str))

    st.subheader("Kalshi health")
    kalshi_health = kalshi_health_check(league)
    st.json(kalshi_health)
    filter_stats = st.session_state.get("kalshi_filter_stats") or {}
    if filter_stats:
        st.subheader("Kalshi filtering stats")
        st.json(filter_stats)
    if st.session_state.get("kalshi_match_results"):
        matches = st.session_state.get("kalshi_match_results")
        matched = [m for m in matches if m.get("kalshi_matched")]
        non_matches = [m for m in matches if not m.get("kalshi_matched")]
        if matched:
            st.caption("Sample matched market")
            st.json(matched[0])
        if filter_stats.get("sample_filtered_first_game"):
            st.caption("Sample filtered markets for first game")
            st.json(filter_stats.get("sample_filtered_first_game"))
        if non_matches:
            reasons = {}
            for m in non_matches:
                reason = m.get("kalshi_reason") or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
            top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            st.caption("Top non-match reasons")
            st.json(top_reasons)

    if "master_stats" in st.session_state:
        st.subheader("Master analysis stats")
        st.json(st.session_state["master_stats"])

    if st.session_state.get("last_exception"):
        st.subheader("Last exception")
        st.code(st.session_state["last_exception"])
