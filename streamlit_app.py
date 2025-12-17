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

# Must be the first Streamlit call
st.set_page_config(page_title="ParlayDesk", layout="wide")

# ------------------------------------------------------------
# Kalshi globals / shims (must exist before any call sites)
# ------------------------------------------------------------
kalshi_integrator: Optional[KalshiIntegrator] = None

def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """
    MUST NOT crash. Used for UI gating + debug.
    ok=True means "reachable/call succeeded", not "game markets exist".
    """
    try:
        ki = kalshi_integrator
        if ki is None:
            return {
                "configured": False,
                "ok": False,
                "error": "Kalshi integrator not initialized.",
                "market_count": 0,
            }

        markets = ki.get_sports_markets(selected_league) or []
        return {
            "configured": True,
            "ok": True,
            "error": None,
            "market_count": len(markets),
        }
    except Exception as e:
        return {
            "configured": True,
            "ok": False,
            "error": str(e),
            "market_count": 0,
        }

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
st.session_state.setdefault("kalshi_required", False)
kalshi_integrator: Optional[KalshiIntegrator] = None
try:
    if "kalshi_integrator" not in st.session_state:
        if kalshi_api_key and kalshi_api_secret:
            st.session_state["kalshi_integrator"] = KalshiIntegrator(
                kalshi_api_key,
                kalshi_api_secret,
                required=st.session_state.get("kalshi_required", True),
            )
        else:
            st.session_state["kalshi_integrator"] = None
    kalshi_integrator = st.session_state.get("kalshi_integrator")
except Exception:
    st.session_state["last_exception"] = traceback.format_exc()
    kalshi_integrator = None
if kalshi_integrator:
    kalshi_integrator.required = st.session_state.get("kalshi_required", True)

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
@st.cache_data(ttl=300)
def fetch_kalshi_markets(
    selected_league: str, commence_times_utc: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    # Ensure the global exists, but don't crash if not initialized yet
    global kalshi_integrator
    if kalshi_integrator is None:
        return []

    league_upper = (selected_league or "").upper()

    def ticker_upper(market: Dict[str, Any]) -> str:
        return str(market.get("event_ticker") or market.get("ticker") or "").upper()

    def prefix_count(markets: List[Dict[str, Any]]) -> Dict[str, int]:
        tickers = [ticker_upper(m) for m in (markets or [])]
        return {
            "count_prefix_KXNBA": len([t for t in tickers if t.startswith("KXNBA")]),
            "count_prefix_KXNBAGAME": len([t for t in tickers if t.startswith("KXNBAGAME")]),
            "count_prefix_KXNBATOTAL": len([t for t in tickers if t.startswith("KXNBATOTAL")]),
            "count_prefix_KXNBASPREAD": len([t for t in tickers if t.startswith("KXNBASPREAD")]),
            "count_prefix_KXMV": len([t for t in tickers if t.startswith("KXMV")]),
        }

    def date_tokens_from_commence(commence_list: Optional[List[str]]) -> set:
        """Convert commence_time ISO strings -> Kalshi tokens like 25DEC17 using APP_TIMEZONE."""
        if not commence_list:
            return set()

        tz_name = get_local_tz()
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = None

        tokens = set()
        for raw in commence_list:
            dt_utc = parse_commence_to_utc(raw)
            if not dt_utc:
                continue
            dt_local = dt_utc.astimezone(local_tz) if local_tz else dt_utc
            tokens.add(dt_local.strftime("%y%b%d").upper())  # 25DEC17
        return tokens

    def prefix_count(markets: List[Dict[str, Any]]) -> Dict[str, int]:
        tickers = [ticker_upper(m) for m in markets]
        return {
            "count_prefix_KXNBA": len([t for t in tickers if t.startswith("KXNBA")]),
            "count_prefix_KXNBAGAME": len([t for t in tickers if t.startswith("KXNBAGAME")]),
            "count_prefix_KXNBATOTAL": len([t for t in tickers if t.startswith("KXNBATOTAL")]),
            "count_prefix_KXNBASPREAD": len([t for t in tickers if t.startswith("KXNBASPREAD")]),
            "count_prefix_KXMV": len([t for t in tickers if t.startswith("KXMV")]),
        }

    try:
        markets_raw = kalshi_integrator.get_league_markets(
            selected_league,
            min_prefix_hits=20,
            max_pages=5,
        )
        if not markets_raw:
            markets_raw = kalshi_integrator.get_markets_paginated(status=None, max_pages=5)
        markets_raw = markets_raw or []

        raw_counts = prefix_count(markets_raw)
        split = kalshi_integrator.split_market_kinds(markets_raw, selected_league)
        game_pool: List[Dict[str, Any]] = [
            m for m in (split.get("single_game_candidates") or []) if ticker_upper(m).startswith("KXNBAGAME-")
        ]
        game_pool_counts = prefix_count(game_pool)

        wanted_tokens = date_tokens_from_commence(commence_times_utc)
        if wanted_tokens:
            filtered = []
            for m in game_pool:
                t = ticker_upper(m)
                if any(tok in t for tok in wanted_tokens):
                    filtered.append(m)
            if filtered:
                game_pool = filtered
                game_pool_counts = prefix_count(game_pool)
            elif game_pool:
                st.session_state["kalshi_filter_warning"] = "date_filter_removed_all"

        if not game_pool and markets_raw:
            fallback_raw = kalshi_integrator.get_markets_paginated(status=None, max_pages=5)
            split_fb = kalshi_integrator.split_market_kinds(fallback_raw, selected_league)
            if split_fb.get("single_game_candidates"):
                markets_raw = fallback_raw or []
                raw_counts = prefix_count(markets_raw)
                game_pool = [
                    m
                    for m in (split_fb.get("single_game_candidates") or [])
                    if ticker_upper(m).startswith("KXNBAGAME-")
                ]
                game_pool_counts = prefix_count(game_pool)
        if not game_pool and split.get("single_game_candidates"):
            game_pool = [
                m for m in (split.get("single_game_candidates") or []) if ticker_upper(m).startswith("KXNBAGAME-")
            ]
            game_pool_counts = prefix_count(game_pool)

        st.session_state["kalshi_markets_raw"] = markets_raw
        st.session_state["kalshi_markets_game_pool"] = game_pool
        st.session_state["kalshi_all_markets"] = markets_raw
        st.session_state["kalshi_prefix_counts"] = {
            "raw": {"total": len(markets_raw), **raw_counts},
            "game_pool": {"total": len(game_pool), **game_pool_counts},
        }
        samples_game = []
        for m in game_pool:
            evt = ticker_upper(m)
            if "GAME-" in evt:
                samples_game.append(evt)
            if len(samples_game) >= 20:
                break
        st.session_state["kalshi_prefix_samples_game"] = samples_game
        st.session_state["kalshi_game_pool_sample"] = samples_game[:10]
        return game_pool
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    def _ticker(m: Dict[str, Any]) -> str:
        return str(m.get("event_ticker") or m.get("ticker") or "").upper()

    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "game_market_count": 0,
        "futures_market_count": 0,
        "has_game_markets": False,
        "has_futures_markets": False,
        "error": None,
        "warning": None,
        "status_code": None,
        "response_text_snippet": None,
        "request_params": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = kalshi_integrator.get_league_markets(
            selected_league, status="active", max_pages=5
        )
        info = kalshi_integrator.last_error_info or {}
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        markets = markets or []
        if not markets and snippet:
            try:
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = json.loads(snippet or "{}")
                parsed_markets = (data.get("markets") or []) if isinstance(data, dict) else []
                markets = parsed_markets or markets
            except Exception:
                markets = markets
        status["market_count"] = len(markets)
        game_markets = [m for m in markets if _ticker(m).startswith("KXNBAGAME-")]
        futures_markets = [
            m
            for m in markets
            if _ticker(m).startswith("KXNBA") and not _ticker(m).startswith("KXNBAGAME-")
        ]
        status["game_market_count"] = len(game_markets)
        status["futures_market_count"] = len(futures_markets)
        status["has_game_markets"] = bool(game_markets)
        status["has_futures_markets"] = bool(futures_markets)
        status["ok"] = True
        if not status["has_game_markets"] and status["has_futures_markets"]:
            status["warning"] = "Kalshi reachable; only futures markets returned for KXNBA series."
        elif not status["has_game_markets"]:
            status["warning"] = (
                "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = False
        return status


def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    def _ticker(m: Dict[str, Any]) -> str:
        return str(m.get("event_ticker") or m.get("ticker") or "").upper()

    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "game_market_count": 0,
        "futures_market_count": 0,
        "has_game_markets": False,
        "has_futures_markets": False,
        "error": None,
        "warning": None,
        "status_code": None,
        "response_text_snippet": None,
        "request_params": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = kalshi_integrator.get_league_markets(
            selected_league, status="active", max_pages=5
        )
        info = kalshi_integrator.last_error_info or {}
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        markets = markets or []
        if not markets and snippet:
            try:
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = json.loads(snippet or "{}")
                parsed_markets = (data.get("markets") or []) if isinstance(data, dict) else []
                markets = parsed_markets or markets
            except Exception:
                markets = markets
        status["market_count"] = len(markets)
        game_markets = [m for m in markets if _ticker(m).startswith("KXNBAGAME-")]
        futures_markets = [
            m
            for m in markets
            if _ticker(m).startswith("KXNBA") and not _ticker(m).startswith("KXNBAGAME-")
        ]
        status["game_market_count"] = len(game_markets)
        status["futures_market_count"] = len(futures_markets)
        status["has_game_markets"] = bool(game_markets)
        status["has_futures_markets"] = bool(futures_markets)
        status["ok"] = True
        if not status["has_game_markets"] and status["has_futures_markets"]:
            status["warning"] = "Kalshi reachable; only futures markets returned for KXNBA series."
        elif not status["has_game_markets"]:
            status["warning"] = (
                "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = False
        return status

def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    def _ticker(m: Dict[str, Any]) -> str:
        return str(m.get("event_ticker") or m.get("ticker") or "").upper()

    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "has_game_markets": False,
        "has_futures_markets": False,
        "error": None,
        "warning": None,
        "status_code": None,
        "response_text_snippet": None,
        "request_params": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = fetch_kalshi_markets(selected_league, commence_times_utc=None)
        markets = markets or []
        if not markets:
            info = kalshi_integrator.last_error_info or {}
            status_code = info.get("status_code") or kalshi_integrator.last_status_code
            resp_text = info.get("response_text") or kalshi_integrator.last_response_text
            if status_code == 200 and resp_text:
                try:
                    try:
                        data = json.loads(resp_text)
                    except Exception:
                        data = {}
                    parsed_markets = (data.get("markets") or []) if isinstance(data, dict) else []
                    if parsed_markets:
                        markets = parsed_markets
                except Exception:
                    markets = markets
        status["market_count"] = len(markets)
        game_markets = [
            m
            for m in markets
            if _ticker(m).startswith("KXNBAGAME-")
        ]
        futures_markets = [
            m
            for m in markets
            if _ticker(m).startswith("KXNBA")
        ]
        status["has_game_markets"] = bool(game_markets)
        status["has_futures_markets"] = bool(futures_markets)
        status["ok"] = True
        if not status["has_game_markets"] and status["has_futures_markets"]:
            status["warning"] = "Kalshi reachable; only futures markets returned for KXNBA series."
        elif not status["has_game_markets"]:
            status["warning"] = (
                "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )
        info = kalshi_integrator.last_error_info or {}
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = False
        return status

def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "has_game_markets": False,
        "has_futures_markets": False,
        "error": None,
        "warning": None,
        "status_code": None,
        "response_text_snippet": None,
        "request_params": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = fetch_kalshi_markets(selected_league, commence_times_utc=None)
        markets = markets or []
        status["market_count"] = len(markets)
        game_markets = [
            m
            for m in markets
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBAGAME-")
        ]
        futures_markets = [
            m
            for m in markets
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBA-")
        ]
        status["has_game_markets"] = bool(game_markets)
        status["has_futures_markets"] = bool(futures_markets)
        status["ok"] = True
        if not status["has_game_markets"]:
            status["warning"] = (
                "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )
        info = kalshi_integrator.last_error_info or {}
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = False
        return status

def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "error": None,
        "warning": None,
        "status_code": None,
        "response_text_snippet": None,
        "request_params": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = fetch_kalshi_markets(selected_league, commence_times_utc=None)
        markets = markets or []
        status["market_count"] = len(markets)
        status["ok"] = True
        game_markets = [
            m
            for m in markets
            if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith("KXNBAGAME-")
        ]
        if not game_markets:
            status["warning"] = (
                f"No {selected_league} game markets returned for the loaded slate dates."
            )
        info = kalshi_integrator.last_error_info or {}
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        snippet = info.get("response_text") or kalshi_integrator.last_response_text
        if snippet:
            status["response_text_snippet"] = snippet[:500]
        status["request_params"] = kalshi_integrator.last_request_params
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = False
        return status

def kalshi_health_check(selected_league: str) -> Dict[str, Any]:
    status = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "error": None,
        "status_code": None,
        "response_text": None,
    }

    if not kalshi_integrator:
        status["error"] = "Kalshi not configured."
        return status

    try:
        markets = fetch_kalshi_markets(selected_league, commence_times_utc=None)
        markets = markets or []
        status["market_count"] = len(markets)
        status["ok"] = bool(markets)
        if not status["ok"]:
            status["error"] = f"Kalshi reachable but no {selected_league} markets returned."
        return status
    except Exception as exc:
        info = kalshi_integrator.last_error_info or {}
        status["error"] = str(exc)
        status["status_code"] = info.get("status_code")
        status["response_text"] = info.get("response_text")
        cached = st.session_state.get("kalshi_markets_raw") or []
        status["market_count"] = len(cached)
        status["ok"] = bool(cached)
        if cached and not status["error"]:
            status["error"] = "Using cached Kalshi markets"
        return status

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


def kalshi_health(selected_league: str = "NBA") -> Dict[str, Any]:
    def prefix_count_local(tickers: List[str]) -> Dict[str, int]:
        return {
            "count_prefix_KXNBA": len([t for t in tickers if t.startswith("KXNBA")]),
            "count_prefix_KXNBAGAME": len([t for t in tickers if t.startswith("KXNBAGAME")]),
            "count_prefix_KXNBATOTAL": len([t for t in tickers if t.startswith("KXNBATOTAL")]),
            "count_prefix_KXNBASPREAD": len([t for t in tickers if t.startswith("KXNBASPREAD")]),
            "count_prefix_KXMV": len([t for t in tickers if t.startswith("KXMV")]),
        }

    def _ticker(m: Dict[str, Any]) -> str:
        return str(m.get("event_ticker") or m.get("ticker") or "").upper()

    base_health = {
        "configured": bool(kalshi_integrator),
        "ok": False,
        "market_count": 0,
        "game_market_count": 0,
        "futures_market_count": 0,
        "sample_market": None,
        "sample_game_market": None,
        "sample_game_market_reason": None,
        "error": None,
        "status_code": None,
        "response_text": None,
        "request_params": None,
        "has_game_markets": False,
        "has_futures_markets": False,
        "warning": None,
    }

    if not kalshi_integrator:
        base_health["error"] = "Kalshi not configured."
        return base_health

    try:
        prefix_counts = st.session_state.get("kalshi_prefix_counts")
        markets_raw: List[Dict[str, Any]] = []

        if not prefix_counts or not prefix_counts.get("game_pool"):
            markets_raw = kalshi_integrator.get_league_markets(
                selected_league,
                min_prefix_hits=1,
                max_pages=2,
            ) or []
            tickers = [m.get("event_ticker") or m.get("ticker") or "" for m in markets_raw]
            prefix_counts = {
                "raw": prefix_count_local(tickers),
                "game_pool": prefix_count_local([t for t in tickers if t.startswith("KXNBA")]),
            }
            st.session_state["kalshi_prefix_counts"] = prefix_counts

        if not markets_raw:
            markets_raw = st.session_state.get("kalshi_markets_raw") or []

        if not markets_raw:
            info = kalshi_integrator.last_error_info or {}
            status_code = info.get("status_code") or kalshi_integrator.last_status_code
            resp_text = info.get("response_text") or kalshi_integrator.last_response_text
            if status_code == 200 and resp_text:
                try:
                    try:
                        data = json.loads(resp_text)
                    except Exception:
                        data = {}
                    markets_raw = (data.get("markets") or []) if isinstance(data, dict) else []
                except Exception:
                    markets_raw = []

        base_health["market_count"] = len(markets_raw)
        base_health["sample_market"] = markets_raw[0] if markets_raw else None

        game_markets = [m for m in markets_raw if _ticker(m).startswith("KXNBAGAME-")]
        futures_markets = [
            m
            for m in markets_raw
            if _ticker(m).startswith("KXNBA") and not _ticker(m).startswith("KXNBAGAME-")
        ]
        base_health["game_market_count"] = len(game_markets)
        base_health["futures_market_count"] = len(futures_markets)
        base_health["sample_game_market"] = game_markets[0] if game_markets else None
        base_health["has_game_markets"] = bool(game_markets)
        base_health["has_futures_markets"] = bool(futures_markets)
        base_health["ok"] = True
        if not base_health["has_game_markets"] and base_health["has_futures_markets"]:
            base_health["warning"] = "Kalshi reachable; only futures markets returned for KXNBA series."
        elif not base_health["has_game_markets"]:
            base_health["warning"] = (
                "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )
        info = kalshi_integrator.last_error_info or {}
        base_health["status_code"] = info.get("status_code") or kalshi_integrator.last_status_code
        base_health["response_text"] = (
            (info.get("response_text") or kalshi_integrator.last_response_text or "")[:500]
        )
        base_health["request_params"] = kalshi_integrator.last_request_params
        return base_health

    except Exception as e:
        if (kalshi_integrator.last_error_info or {}).get("status_code") == 429:
            cached_markets = st.session_state.get("kalshi_markets_raw") or []
            base_health["market_count"] = len(cached_markets)
            base_health["sample_market"] = cached_markets[0] if cached_markets else None
            game_markets = [
                m
                for m in cached_markets
                if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(
                    "KXNBAGAME-"
                )
            ]
            base_health["sample_game_market"] = game_markets[0] if game_markets else None
            base_health["has_game_markets"] = bool(game_markets)
            base_health["has_futures_markets"] = bool(cached_markets)
            base_health["ok"] = False
            base_health["error"] = "Kalshi rate limited; using cached markets"
            return base_health
        base_health["error"] = f"Kalshi health check failed: {e}"
        return base_health

def kalshi_health_check(selected_league: str = "NBA") -> Dict[str, Any]:
    """
    Backwards-compatible alias.
    Some UI code calls kalshi_health_check(), but the implementation is kalshi_health().
    """
    return kalshi_health(selected_league)

def parse_kalshi_datetime(dt_value: Any) -> Optional[datetime]:
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

def kalshi_market_best_time_utc(m: Dict[str, Any]) -> Optional[datetime]:
    """Return the best available Kalshi timestamp for matching."""
    for key in [
        "expected_expiration_time",
        "latest_expiration_time",
        "close_time",
        "expiration_time",
        "open_time",
    ]:
        candidate = parse_kalshi_datetime(m.get(key))
        if candidate:
            return candidate
    return None


def market_mentions_game_teams(
    market: Dict[str, Any],
    home_team: Any,
    away_team: Any,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> bool:
    try:
        ticker = str(market.get("event_ticker") or market.get("ticker") or "")
        title = str(market.get("title") or "")
        rules_primary = str(market.get("rules") or market.get("rules_primary") or "")
        blob = f"{ticker} {title} {rules_primary}".lower()
        if home_code and away_code:
            if home_code.lower() in blob and away_code.lower() in blob:
                return True

        def nickname_token(name: Any) -> Optional[str]:
            cleaned = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower()).strip()
            if not cleaned:
                return None
            parts = [p for p in cleaned.split() if p]
            return parts[-1] if parts else None

        home_tok = nickname_token(home_team)
        away_tok = nickname_token(away_team)
        if home_tok and away_tok:
            return home_tok in blob and away_tok in blob
        return False
    except Exception:
        return False


def debug_search_markets_for_game(
    markets: List[Dict[str, Any]],
    home_team: Any,
    away_team: Any,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
    limit: int = 15,
) -> Dict[str, Any]:
    def text_blob(parts: List[str]) -> str:
        return " ".join([p for p in parts if p]).lower()

    def word_set(val: str) -> set:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", val.lower())
        return {w for w in cleaned.split() if w}

    home_tokens = team_tokens(home_team)
    away_tokens = team_tokens(away_team)
    matches: List[Dict[str, Any]] = []
    found = {"winner": False, "total": False, "spread": False, "other": False}
    counts = {"winner": 0, "total": 0, "spread": 0, "other": 0}
    for m in markets or []:
        try:
            ticker = str(m.get("event_ticker") or m.get("ticker") or "")
            title = str(m.get("title") or "")
            rules = str(m.get("rules") or m.get("rules_primary") or "")
            blob = text_blob([ticker, title, rules])
            code_match = False
            if home_code and away_code:
                if home_code.lower() in blob and away_code.lower() in blob:
                    code_match = True
            blob_tokens = word_set(blob)
            token_match = bool(home_tokens.intersection(blob_tokens)) and bool(
                away_tokens.intersection(blob_tokens)
            )
            if not (code_match or token_match):
                continue
            ticker_upper = ticker.upper()
            category = "other"
            if ticker_upper.startswith("KXNBAGAME") or "GAME-" in ticker_upper:
                category = "winner"
            elif ticker_upper.startswith("KXNBATOTAL") or "TOTAL" in ticker_upper:
                category = "total"
            elif ticker_upper.startswith("KXNBASPREAD") or "SPREAD" in ticker_upper:
                category = "spread"
            found[category] = True
            counts[category] += 1
            matches.append({"ticker": ticker, "title": title, "category": category})
        except Exception:
            continue
    return {
        "found_any_winner_market_for_game": found["winner"],
        "found_any_total_market_for_game": found["total"],
        "found_any_spread_market_for_game": found["spread"],
        "counts": counts,
        "matches": matches[:limit],
    }


def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
    home_team: Any = None,
    away_team: Any = None,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    window = timedelta(hours=72)
    try:
        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_kalshi_datetime(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)

        def looks_like_game(market: Dict[str, Any]) -> bool:
            title = str(market.get("title") or "").lower()
            rules = str(market.get("rules") or market.get("rules_primary") or "").lower()
            ticker = str(market.get("event_ticker") or market.get("ticker") or "").lower()
            title_match = any(tok in title for tok in [" vs ", " at ", "@"])
            rules_match = any(tok in rules for tok in [" vs ", " at ", "@"])
            ticker_match = "game" in ticker
            return ticker_match or title_match or rules_match

        def candidate_time(market: Dict[str, Any]) -> Optional[datetime]:
            for key in ["close_time", "expected_expiration_time", "latest_expiration_time", "expiration_time"]:
                dt_val = parse_kalshi_datetime(market.get(key))
                if dt_val:
                    return dt_val
            return None

        def team_match(market: Dict[str, Any]) -> bool:
            blob = " ".join(
                [
                    str(market.get("title") or ""),
                    str(market.get("rules") or market.get("rules_primary") or ""),
                    str(market.get("event_ticker") or market.get("ticker") or ""),
                ]
            ).lower()
            code_ok = False
            if home_code and away_code:
                code_ok = home_code.lower() in blob and away_code.lower() in blob
            tokens = team_tokens(home_team)
            tokens_away = team_tokens(away_team)
            blob_tokens = word_set(blob)
            nickname_ok = bool(tokens.intersection(blob_tokens)) and bool(
                tokens_away.intersection(blob_tokens)
            )
            rules_text = str(market.get("rules") or market.get("rules_primary") or "").lower()
            rules_ok = False
            if rules_text:
                rules_tokens = word_set(rules_text)
                rules_ok = bool(tokens.intersection(rules_tokens)) and bool(
                    tokens_away.intersection(rules_tokens)
                )
            return code_ok or nickname_ok or rules_ok

        filtered: List[Dict[str, Any]] = []
        for m in markets or []:
            if not looks_like_game(m):
                continue
            ct = candidate_time(m)
            if game_dt and ct:
                if abs(ct - game_dt) > window:
                    continue
            if game_dt and ct is None:
                pass
            filtered.append(m)

        filtered_by_team: List[Dict[str, Any]] = []
        for m in filtered:
            if team_match(m):
                filtered_by_team.append(m)
        return filtered_by_team
    except Exception:
        st.session_state["last_exception"] = traceback.format_exc()
        return []


def classify_kalshi_market(market: Dict[str, Any]) -> str:
    ticker = str(market.get("ticker") or market.get("event_ticker") or "").upper()
    title = str(market.get("title") or "").lower()
    rules = str(market.get("rules") or "").lower()

    if "GAME-" in ticker or ticker.startswith("KXNBAGAME-") or "GAME" in ticker:
        return "winner"
    if "TOTAL" in ticker:
        return "total"
    if "SPREAD" in ticker:
        return "spread"
    if any(tok in ticker for tok in ["2D", "3D", "TD", "PTS", "REB", "AST"]) or any(
        key in title for key in ["double", "triple"]
    ):
        return "prop"

    if "total points" in title:
        return "total"
    if "spread" in title:
        return "spread"
    if "winner" in title or "win" in title or "wins the game" in rules:
        return "winner"
    return "unknown"


def team_tokens(name: str) -> set:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower())
    tokens = [t for t in cleaned.split() if t]
    stopwords = {
        "the",
        "fc",
        "sc",
        "university",
        "state",
        "college",
        "team",
        "basketball",
        "football",
        "hockey",
        "baseball",
    }
    return {t for t in tokens if t not in stopwords}


def nba_abbrev(team_name: str) -> Optional[str]:
    mapping = {
        "atlanta hawks": "ATL",
        "boston celtics": "BOS",
        "brooklyn nets": "BKN",
        "charlotte hornets": "CHA",
        "chicago bulls": "CHI",
        "cleveland cavaliers": "CLE",
        "dallas mavericks": "DAL",
        "denver nuggets": "DEN",
        "detroit pistons": "DET",
        "golden state warriors": "GSW",
        "houston rockets": "HOU",
        "indiana pacers": "IND",
        "los angeles clippers": "LAC",
        "la clippers": "LAC",
        "los angeles lakers": "LAL",
        "la lakers": "LAL",
        "memphis grizzlies": "MEM",
        "miami heat": "MIA",
        "milwaukee bucks": "MIL",
        "minnesota timberwolves": "MIN",
        "new orleans pelicans": "NOP",
        "new york knicks": "NYK",
        "oklahoma city thunder": "OKC",
        "orlando magic": "ORL",
        "philadelphia 76ers": "PHI",
        "phoenix suns": "PHX",
        "portland trail blazers": "POR",
        "sacramento kings": "SAC",
        "san antonio spurs": "SAS",
        "toronto raptors": "TOR",
        "utah jazz": "UTA",
        "washington wizards": "WAS",
    }
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(team_name or "").lower()).strip()
    for key, code in mapping.items():
        if key in cleaned:
            return code
    return None


def kalshi_date_token_from_local(date_val: Any) -> Optional[str]:
    """Return YYMONDD token (e.g., 25DEC16) for local YYYY-MM-DD date strings."""
    try:
        if not date_val:
            return None
        parsed = datetime.fromisoformat(str(date_val))
        return parsed.strftime("%y%b%d").upper()
    except Exception:
        return None


def kalshi_ticker_team_codes(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ticker = str(market.get("event_ticker") or market.get("ticker") or "")
    match = re.search(r"([A-Z]{6})$", ticker)
    if match:
        segment = match.group(1)
        return segment[:3], segment[3:]
    return None, None


def extract_teams_from_kalshi_text(text: Any) -> Tuple[Optional[str], Optional[str]]:
    content = str(text or "")
    patterns = [
        r"(.+?)\s+at\s+(.+?)(:|\||-|$)",
        r"(.+?)\s+@\s+(.+?)(:|\||-|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(:|\||-|$)",
    ]
    for pat in patterns:
        match = re.search(pat, content, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None, None


def match_kalshi_market(
    game: Dict[str, Any],
    kalshi_markets: List[Dict[str, Any]],
    winner_reason_override: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    def base_result(reason: str, market_type: str) -> Dict[str, Any]:
        return {
            "kalshi_available": bool(kalshi_integrator),
            "kalshi_label": None,
            "kalshi_event_ticker": None,
            "kalshi_reason": reason,
            "kalshi_matched": False,
            "kalshi_prob": None,
            "kalshi_market_type": market_type,
            "kalshi_match_score": None,
            "kalshi_ticker": None,
            "kalshi_line": None,
            "kalshi_title": None,
        }

    def norm_team(name: Any) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).strip()

    def league_from_game(g: Dict[str, Any]) -> str:
        skey = (g.get("sport_key") or g.get("league") or g.get("League") or "").lower()
        mapping = {
            "basketball_nba": "NBA",
            "nba": "NBA",
            "basketball_ncaab": "NCAAB",
            "ncaab": "NCAAB",
            "americanfootball_nfl": "NFL",
            "nfl": "NFL",
            "americanfootball_ncaaf": "NCAAF",
            "ncaaf": "NCAAF",
            "icehockey_nhl": "NHL",
            "nhl": "NHL",
            "baseball_mlb": "MLB",
            "mlb": "MLB",
        }
        return mapping.get(skey, skey.upper())

    def team_score(market: Dict[str, Any], market_type: str) -> float:
        home_norm = norm_team(game.get("home_team"))
        away_norm = norm_team(game.get("away_team"))
        home_tokens = team_tokens(game.get("home_team"))
        away_tokens = team_tokens(game.get("away_team"))
        nickname_tokens = {
            "knicks",
            "spurs",
            "lakers",
            "celtics",
            "bulls",
            "nets",
            "bucks",
            "sixers",
            "warriors",
            "heat",
            "suns",
            "mavs",
            "mavericks",
            "clippers",
            "kings",
            "hawks",
            "raptors",
            "jazz",
            "rockets",
            "thunder",
            "pelicans",
            "nuggets",
            "timberwolves",
            "wolves",
            "grizzlies",
            "magic",
            "pacers",
            "pistons",
            "cavaliers",
            "cavs",
            "blazers",
            "wizards",
            "76ers",
        }
        title_away, title_home = extract_teams_from_kalshi_text(market.get("title"))
        rules_away, rules_home = extract_teams_from_kalshi_text(market.get("rules"))
        guesses = [
            (title_away, title_home),
            (rules_away, rules_home),
        ]
        for away_guess, home_guess in guesses:
            if away_guess and home_guess:
                away_guess_norm = norm_team(away_guess)
                home_guess_norm = norm_team(home_guess)
                if (
                    away_guess_norm == away_norm and home_guess_norm == home_norm
                ) or (away_guess_norm == home_norm and home_guess_norm == away_norm):
                    return 1.0
        market_text = f"{market.get('title') or ''} {market.get('rules') or ''}"
        market_tokens = team_tokens(market_text)
        if (
            away_tokens
            and home_tokens
            and away_tokens.issubset(market_tokens)
            and home_tokens.issubset(market_tokens)
        ):
            return 1.0
        away_city_tokens = {t for t in away_tokens if t not in nickname_tokens}
        home_city_tokens = {t for t in home_tokens if t not in nickname_tokens}
        if (
            away_city_tokens
            and home_city_tokens
            and away_city_tokens.issubset(market_tokens)
            and home_city_tokens.issubset(market_tokens)
        ):
            return 0.85
        if market_type == "winner":
            away_expected = nba_abbrev(game.get("away_team"))
            home_expected = nba_abbrev(game.get("home_team"))
            away_code_market, home_code_market = kalshi_ticker_team_codes(market)
            if (
                away_expected
                and home_expected
                and away_code_market
                and home_code_market
                and (
                    (away_expected == away_code_market and home_expected == home_code_market)
                    or (away_expected == home_code_market and home_expected == away_code_market)
                )
            ):
                return 1.0
        return 0.0

    def time_score(market: Dict[str, Any]) -> float:
        game_dt = game.get("commence_time_utc")
        market_dt = kalshi_market_best_time_utc(market)
        if not isinstance(game_dt, datetime) or market_dt is None:
            return 0.0
        delta_hours = abs((market_dt - game_dt).total_seconds()) / 3600.0
        return max(0.0, 1.0 - min(delta_hours / 36.0, 1.0))

    def extract_prob_and_line(
        market: Dict[str, Any], market_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        prices: List[float] = []
        for val in [market.get("yes_bid"), market.get("yes_ask")]:
            try:
                prices.append(float(val))
            except Exception:
                continue
        selected_price = None
        if len(prices) == 2:
            selected_price = sum(prices) / 2.0
        elif prices:
            selected_price = prices[0]
        prob = (selected_price / 100.0) if selected_price is not None else None
        line = market.get("floor_strike") or market.get("cap_strike")
        if line is not None:
            try:
                line = float(line)
            except Exception:
                line = None
        return prob, line

    def evaluate_partition(
        partition_markets: List[Dict[str, Any]],
        market_type: str,
        no_market_reason: Optional[str] = None,
    ):
        best_market: Optional[Dict[str, Any]] = None
        best_score = -1.0
        candidates: List[Dict[str, Any]] = []
        any_positive_team = False
        if market_type == "winner" and candidate_event_tickers_set:
            for m in partition_markets:
                tick_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
                if tick_upper in candidate_event_tickers_set:
                    best_market = m
                    best_score = 1.0
                    any_positive_team = True
                    candidates.append(
                        {
                            "title": m.get("title"),
                            "ticker": m.get("event_ticker") or m.get("ticker"),
                            "team_score": 1.0,
                            "time_score": time_score(m),
                            "final_score": 1.0,
                        }
                    )
                    break
        for m in partition_markets:
            ts = team_score(m, market_type)
            tms = time_score(m)
            final = 0.8 * ts + 0.2 * tms
            candidates.append(
                {
                    "title": m.get("title"),
                    "ticker": m.get("event_ticker") or m.get("ticker"),
                    "away_code_market": kalshi_ticker_team_codes(m)[0]
                    if market_type == "winner"
                    else None,
                    "home_code_market": kalshi_ticker_team_codes(m)[1]
                    if market_type == "winner"
                    else None,
                    "team_score": ts,
                    "time_score": tms,
                    "final_score": final,
                }
            )
            if final > best_score:
                if market_type != "winner" or ts > 0 or final > 0:
                    any_positive_team = any_positive_team or ts > 0 or final > 0
                    best_score = final
                    best_market = m
        if best_market and any_positive_team:
            prob, line = extract_prob_and_line(best_market, market_type)
            return {
                "kalshi_available": True,
                "kalshi_label": f"matched_{market_type}",
                "kalshi_event_ticker": best_market.get("event_ticker") or best_market.get("ticker"),
                "kalshi_reason": f"matched_{market_type}",
                "kalshi_matched": True,
                "kalshi_prob": prob,
                "kalshi_market_type": market_type,
                "kalshi_match_score": best_score,
                "kalshi_ticker": best_market.get("event_ticker") or best_market.get("ticker"),
                "kalshi_line": line,
                "kalshi_title": best_market.get("title"),
            }, candidates
        no_market_message = no_market_reason or f"no_{market_type}_market"
        if any_positive_team:
            return base_result(no_market_message, market_type), candidates
        if partition_markets:
            return base_result("no_team_match", market_type), candidates
        return base_result(no_market_message, market_type), candidates

    if not kalshi_integrator:
        base = {t: base_result("kalshi_not_configured", t) for t in ["total", "spread", "winner"]}
        return base, {"total": [], "spread": [], "winner": []}
    if not kalshi_markets:
        base = {t: base_result("no_game_like_markets_in_window", t) for t in ["total", "spread", "winner"]}
        return base, {"total": [], "spread": [], "winner": []}

    league_name = league_from_game(game)
    date_token = kalshi_date_token_from_local(game.get("commence_date_local"))
    away_code_expected = nba_abbrev(game.get("away_team")) if league_name == "NBA" else None
    home_code_expected = nba_abbrev(game.get("home_team")) if league_name == "NBA" else None
    searched_prefix = None
    winner_rejections = {"wrong_date": 0, "missing_code": 0}
    candidate_event_tickers: List[str] = []
    date_bucket_event_tickers: List[str] = []
    winner_reason = winner_reason_override or "no_winner_market_for_game"

    totals = [m for m in kalshi_markets if classify_kalshi_market(m) == "total"]
    spreads = [m for m in kalshi_markets if classify_kalshi_market(m) == "spread"]
    winners = [m for m in kalshi_markets if classify_kalshi_market(m) == "winner"]

    # Date-token summary for debug (unique by event_ticker)
    date_token_counts: Dict[str, int] = {}
    for m in kalshi_markets:
        et = str(m.get("event_ticker") or m.get("ticker") or "").upper()
        if et.startswith("KXNBAGAME-") and len(et) >= 16:
            token = et.split("KXNBAGAME-")[1][:7]
            date_token_counts[token] = date_token_counts.get(token, 0) + 1

    candidate_event_tickers_set: set = set()
    if league_name == "NBA":
        searched_prefix = f"KXNBAGAME-{date_token}" if date_token else None
        candidate_event_tickers = []
        if date_token and away_code_expected and home_code_expected:
            candidate_event_tickers = [
                f"KXNBAGAME-{date_token}{away_code_expected}{home_code_expected}",
                f"KXNBAGAME-{date_token}{home_code_expected}{away_code_expected}",
            ]
        else:
            winners = []
            winner_reason = "missing_team_codes_or_date_token"

        if winners and searched_prefix:
            bucket_map: Dict[str, Dict[str, Any]] = {}
            for m in winners:
                event_tick = str(m.get("event_ticker") or "").upper()
                if event_tick.startswith(searched_prefix):
                    if event_tick not in bucket_map:
                        bucket_map[event_tick] = m
            date_bucket = list(bucket_map.values())
            date_bucket_event_tickers = list(bucket_map.keys())
            if not date_bucket:
                winners = []
                winner_reason = "no_kalshi_date_bucket"
            else:
                candidates_set = set(candidate_event_tickers)
                winners = [
                    m
                    for m in date_bucket
                    if str(m.get("event_ticker") or "").upper() in candidates_set
                ]
                if not winners:
                    winner_reason = "no_exact_event_ticker_match_in_bucket"
        elif winners:
            # If we have winners but no searched_prefix/date_token, treat as missing token
            winners = []
            winner_reason = "no_kalshi_date_bucket"
        else:
            date_bucket_event_tickers = []
            if winner_reason == winner_reason_override or winner_reason == "no_winner_market_for_game":
                winner_reason = "no_kalshi_date_bucket"
        candidate_event_tickers = list(dict.fromkeys(candidate_event_tickers))
        candidate_event_tickers_set = {c.upper() for c in candidate_event_tickers}

    total_result, total_candidates = evaluate_partition(totals, "total")
    spread_result, spread_candidates = evaluate_partition(spreads, "spread")
    winner_result, winner_candidates = evaluate_partition(
        winners, "winner", winner_reason
    )

    # Guard against any cross-date matches that slipped through
    if league_name == "NBA" and winner_result.get("kalshi_matched"):
        evt = str(winner_result.get("kalshi_event_ticker") or "").upper()
        if not date_token or (date_token and f"KXNBAGAME-{date_token}" not in evt):
            winner_result = base_result("date_bucket_guard_triggered", "winner")

    match_status = "matched" if winner_result.get("kalshi_matched") else "no_match"
    no_match_reason = None if winner_result.get("kalshi_matched") else winner_result.get("kalshi_reason")
    matched_event_ticker = winner_result.get("kalshi_event_ticker")
    matched_ticker = winner_result.get("kalshi_ticker")

    winner_meta = {
        "expected_date_token": date_token,
        "expected_codes": {"away": away_code_expected, "home": home_code_expected},
        "candidate_event_tickers": candidate_event_tickers[:10],
        "searched_prefix": searched_prefix,
        "date_bucket_counts": date_token_counts,
        "date_bucket_markets_count": len(date_bucket_event_tickers),
        "checked_event_tickers_sample": date_bucket_event_tickers[:10],
        "rejection_counts": winner_rejections,
        "match_status": match_status,
        "no_match_reason": no_match_reason,
        "matched_event_ticker": matched_event_ticker,
        "matched_ticker": matched_ticker,
    }

    return (
        {"total": total_result, "spread": spread_result, "winner": winner_result},
        {
            "total": total_candidates,
            "spread": spread_candidates,
            "winner": winner_candidates,
            "winner_meta": winner_meta,
        },
    )


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
kalshi_required_toggle = st.sidebar.checkbox(
    "Kalshi required", value=st.session_state.get("kalshi_required", True)
)
st.session_state["kalshi_required"] = kalshi_required_toggle
if kalshi_integrator:
    kalshi_integrator.required = kalshi_required_toggle
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
    if not kalshi_status.get("configured"):
        error_detail = kalshi_status.get("error") or "Kalshi is required and missing keys."
        if kalshi_status.get("status_code"):
            error_detail = f"{error_detail} (status {kalshi_status.get('status_code')}: {kalshi_status.get('response_text_snippet')})"
        st.error(error_detail)
        st.info("Master Analysis is disabled until Kalshi is available.")
    else:
        if kalshi_status.get("error") and not kalshi_status.get("ok"):
            warn_detail = kalshi_status.get("error") or "Kalshi reachable but returned no markets; proceeding without Kalshi data."
            st.warning(warn_detail)
        if kalshi_status.get("warning"):
            st.warning(kalshi_status.get("warning"))
    run_master = st.button(
        "Run Master Analysis",
        key="run_master",
        disabled=(not kalshi_status.get("configured")) and st.session_state.get("kalshi_required", True),
        help="Requires Kalshi availability",
    )
    games = st.session_state.get("games", [])
    if run_master and (not kalshi_status.get("configured")):
        st.error("Kalshi is required but unavailable. Fix Kalshi first.")
        st.stop()
    if run_master:
        if st.session_state.get("kalshi_required", True) and kalshi_integrator:
            try:
                kalshi_integrator.assert_available()
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        commence_times_utc = [
            g.get("commence_time_iso_utc")
            or g.get("commence_time")
            or g.get("commence_time_iso")
            for g in games
            if g.get("commence_time_iso_utc")
            or g.get("commence_time")
            or g.get("commence_time_iso")
        ]
        try:
            kalshi_markets = fetch_kalshi_markets(league, commence_times_utc)
        except RuntimeError as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                st.error("Kalshi rate-limited. Please retry in ~X seconds.")
            else:
                st.error(msg)
            st.stop()
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        if not kalshi_markets:
            st.warning(
                "Kalshi markets could not be fetched; proceeding with cached/empty set."
            )
            kalshi_markets = st.session_state.get("kalshi_markets_raw") or []
        st.session_state["kalshi_all_markets"] = st.session_state.get(
            "kalshi_markets_raw", kalshi_markets
        )
        winner_refetch_attempted = False
        full_search_first_game: Optional[Dict[str, Any]] = None
        if games:
            try:
                fg = games[0]
                full_search_first_game = debug_search_markets_for_game(
                    kalshi_markets,
                    fg.get("home_team"),
                    fg.get("away_team"),
                    nba_abbrev(fg.get("home_team")),
                    nba_abbrev(fg.get("away_team")),
                )
            except Exception:
                st.session_state["last_exception"] = traceback.format_exc()
        if (
            full_search_first_game
            and league == "NBA"
            and not full_search_first_game.get("found_any_winner_market_for_game")
            and (
                full_search_first_game.get("found_any_total_market_for_game")
                or full_search_first_game.get("matches")
            )
        ):
            winner_refetch_attempted = True
            try:
                refreshed = kalshi_integrator.get_sports_markets() if kalshi_integrator else []
                prefix = LEAGUE_SERIES_MAP.get((league or "").upper())
                if prefix:
                    refreshed = [
                        m
                        for m in refreshed
                        if str(m.get("ticker") or "").upper().startswith(prefix)
                    ]
                if refreshed:
                    kalshi_markets = refreshed
                    st.session_state["kalshi_all_markets"] = refreshed
                    if games:
                        try:
                            fg = games[0]
                            full_search_first_game = debug_search_markets_for_game(
                                refreshed,
                                fg.get("home_team"),
                                fg.get("away_team"),
                                nba_abbrev(fg.get("home_team")),
                                nba_abbrev(fg.get("away_team")),
                            )
                        except Exception:
                            st.session_state["last_exception"] = traceback.format_exc()
            except Exception:
                st.session_state["last_exception"] = traceback.format_exc()
        filtered_counts: List[int] = []
        per_game_kalshi_debug: List[Dict[str, Any]] = []
        first_game_full_search = full_search_first_game
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
        for idx, g in enumerate(games):
            warnings: List[str] = list(g.get("warnings") or [])
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
            commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
            commence_local = fmt_local_time(g.get("commence_time_local"))
            commence_date_local = g.get("commence_date_local") or ""
            away_code = nba_abbrev(away)
            home_code = nba_abbrev(home)
            filtered_markets = filter_kalshi_game_markets(
                kalshi_markets,
                g.get("commence_time_utc"),
                league_name,
                home,
                away,
                home_code,
                away_code,
            )
            deduped: Dict[str, Dict[str, Any]] = {}
            for fm in filtered_markets:
                key = fm.get("event_ticker") or fm.get("ticker") or str(id(fm))
                if key not in deduped:
                    deduped[key] = fm
            filtered_markets = list(deduped.values())
            filtered_counts.append(len(filtered_markets))
            partition_counts = {
                "total": len([m for m in filtered_markets if classify_kalshi_market(m) == "total"]),
                "spread": len([m for m in filtered_markets if classify_kalshi_market(m) == "spread"]),
                "winner": len([m for m in filtered_markets if classify_kalshi_market(m) == "winner"]),
                "prop": len([m for m in filtered_markets if classify_kalshi_market(m) == "prop"]),
            }
            winner_reason_override = None
            if (
                idx == 0
                and first_game_full_search
                and not first_game_full_search.get("found_any_winner_market_for_game")
            ):
                winner_reason_override = "winner_not_in_fetched_markets"
            kalshi_matches, candidate_debug = match_kalshi_market(
                g, filtered_markets, winner_reason_override
            )
            winner_sample = []
            if candidate_debug and isinstance(candidate_debug.get("winner"), list):
                winner_sample = candidate_debug.get("winner", [])[:3]
            per_game_kalshi_debug.append(
                {
                    "game_index": idx,
                    "game_home": home,
                    "game_away": away,
                    "game_commence_utc": commence_iso,
                    "kalshi_date_token_used": (candidate_debug or {})
                    .get("winner_meta", {})
                    .get("expected_date_token"),
                    "expected_codes": (candidate_debug or {})
                    .get("winner_meta", {})
                    .get("expected_codes"),
                    "away_code": away_code,
                    "home_code": home_code,
                    "strict_filtered_count": len(filtered_markets),
                    "strict_filtered_sample": [
                        {
                            "title": fm.get("title"),
                            "ticker": fm.get("event_ticker") or fm.get("ticker"),
                        }
                        for fm in filtered_markets[:3]
                    ],
                    "winner_candidates_count": len(winner_sample),
                    "winner_candidates_sample": winner_sample,
                    "matched_ticker": kalshi_matches.get("winner", {}).get(
                        "kalshi_event_ticker"
                    ),
                    "matched_title": kalshi_matches.get("winner", {}).get(
                        "kalshi_title"
                    ),
                    "kalshi_reason": kalshi_matches.get("winner", {}).get(
                        "kalshi_reason"
                    ),
                }
            )

            matched_any = any(v.get("kalshi_matched") for v in kalshi_matches.values())
            if not matched_any:
                for res in kalshi_matches.values():
                    warnings.append(f"kalshi_{res.get('kalshi_reason')}")
            else:
                master_stats["kalshi_matches"] += 1
            kalshi_match_results.append(
                {
                    "home": home,
                    "away": away,
                    "league": league_name,
                    "matches": kalshi_matches,
                    "candidates": candidate_debug,
                }
            )

            try:
                ai_prob = get_vertex_prob(g)
            except Exception:
                ai_prob = None
                warnings.append("vertex_error")
                st.session_state["last_exception"] = traceback.format_exc()

            moneyline_row_added = False
            spread_row_added = False
            total_row_added = False

            kalshi_total = kalshi_matches.get("total", {})
            kalshi_spread = kalshi_matches.get("spread", {})
            kalshi_winner = kalshi_matches.get("winner", {})

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
                match_ref = kalshi_winner
                if not match_ref.get("kalshi_matched"):
                    warnings.append(f"kalshi_{match_ref.get('kalshi_reason')}")
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
                        "kalshi_available": match_ref.get("kalshi_available"),
                        "kalshi_label": match_ref.get("kalshi_label"),
                        "kalshi_event_ticker": match_ref.get("kalshi_event_ticker"),
                        "kalshi_matched": match_ref.get("kalshi_matched"),
                        "kalshi_prob": match_ref.get("kalshi_prob"),
                        "kalshi_reason": match_ref.get("kalshi_reason"),
                        "kalshi_market_type": match_ref.get("kalshi_market_type"),
                        "kalshi_ticker": match_ref.get("kalshi_ticker"),
                        "kalshi_line": match_ref.get("kalshi_line"),
                        "kalshi_title": match_ref.get("kalshi_title"),
                        "kalshi_match_score": match_ref.get("kalshi_match_score"),
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
                match_ref = kalshi_spread
                if not match_ref.get("kalshi_matched"):
                    warnings.append(f"kalshi_{match_ref.get('kalshi_reason')}")
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
                        "kalshi_available": match_ref.get("kalshi_available"),
                        "kalshi_label": match_ref.get("kalshi_label"),
                        "kalshi_event_ticker": match_ref.get("kalshi_event_ticker"),
                        "kalshi_matched": match_ref.get("kalshi_matched"),
                        "kalshi_prob": match_ref.get("kalshi_prob"),
                        "kalshi_reason": match_ref.get("kalshi_reason"),
                        "kalshi_market_type": match_ref.get("kalshi_market_type"),
                        "kalshi_ticker": match_ref.get("kalshi_ticker"),
                        "kalshi_line": match_ref.get("kalshi_line"),
                        "kalshi_title": match_ref.get("kalshi_title"),
                        "kalshi_match_score": match_ref.get("kalshi_match_score"),
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
                match_ref = kalshi_total
                if not match_ref.get("kalshi_matched"):
                    warnings.append(f"kalshi_{match_ref.get('kalshi_reason')}")
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
                        "kalshi_available": match_ref.get("kalshi_available"),
                        "kalshi_label": match_ref.get("kalshi_label"),
                        "kalshi_event_ticker": match_ref.get("kalshi_event_ticker"),
                        "kalshi_matched": match_ref.get("kalshi_matched"),
                        "kalshi_prob": match_ref.get("kalshi_prob"),
                        "kalshi_reason": match_ref.get("kalshi_reason"),
                        "kalshi_market_type": match_ref.get("kalshi_market_type"),
                        "kalshi_ticker": match_ref.get("kalshi_ticker"),
                        "kalshi_line": match_ref.get("kalshi_line"),
                        "kalshi_title": match_ref.get("kalshi_title"),
                        "kalshi_match_score": match_ref.get("kalshi_match_score"),
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
                        "kalshi_available": kalshi_winner.get("kalshi_available"),
                        "kalshi_label": kalshi_winner.get("kalshi_label"),
                        "kalshi_event_ticker": kalshi_winner.get("kalshi_event_ticker"),
                        "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                        "kalshi_prob": kalshi_winner.get("kalshi_prob"),
                        "kalshi_reason": kalshi_winner.get("kalshi_reason"),
                        "kalshi_market_type": kalshi_winner.get("kalshi_market_type"),
                        "kalshi_ticker": kalshi_winner.get("kalshi_ticker"),
                        "kalshi_line": kalshi_winner.get("kalshi_line"),
                        "kalshi_title": kalshi_winner.get("kalshi_title"),
                        "kalshi_match_score": kalshi_winner.get("kalshi_match_score"),
                    }
                )
                master_stats["market_rows_out"] += 1

        df = pd.DataFrame(rows_out)
        master_stats["rows_out"] = len(df)
        st.session_state["last_rows_out"] = len(df)
        st.session_state["master_stats"] = master_stats
        st.session_state["kalshi_match_results"] = kalshi_match_results
        total_game_markets = len(
            [
                m
                for m in kalshi_markets
                if str(m.get("event_ticker") or m.get("ticker") or "").upper().startswith(
                    "KXNBAGAME-"
                )
            ]
        )
        first_game_meta = per_game_kalshi_debug[0] if per_game_kalshi_debug else {}
        st.session_state["kalshi_filter_stats"] = {
            "total_markets_fetched": len(kalshi_markets),
            "total_game_markets": total_game_markets,
            "avg_filtered_markets_per_game": sum(filtered_counts) / len(filtered_counts)
            if filtered_counts
            else 0,
            "filtered_markets_min": min(filtered_counts) if filtered_counts else 0,
            "filtered_markets_max": max(filtered_counts) if filtered_counts else 0,
            "per_game_debug": per_game_kalshi_debug,
            "first_game_debug": per_game_kalshi_debug[0]
            if per_game_kalshi_debug
            else {},
            "first_game_full_market_search": first_game_full_search,
            "kalshi_winner_refetch_attempted": winner_refetch_attempted,
            "first_game_expected": {
                "expected_date_token": (first_game_meta or {}).get("kalshi_date_token_used"),
                "expected_codes": (first_game_meta or {}).get("expected_codes"),
                "matched_ticker": (first_game_meta or {}).get("matched_ticker"),
                "kalshi_reason": (first_game_meta or {}).get("kalshi_reason"),
            },
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
        if (kalshi_status.get("market_count") or 0) > 0:
            st.success("Kalshi credentials detected and markets available.")
        else:
            st.warning(
                kalshi_status.get("warning")
                or "Kalshi reachable, but no NBA KXNBAGAME markets returned (futures-only or slate not listed)."
            )


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
    if kalshi_integrator and st.checkbox("Show Kalshi market counts", key="kalshi_market_counts_toggle"):
        dbg_markets = kalshi_integrator.get_league_markets(
            league, status="active", max_pages=2, min_prefix_hits=5
        )
        dbg_tickers = [str(m.get("event_ticker") or m.get("ticker") or "").upper() for m in dbg_markets]
        st.json(
            {
                "kalshi_debug_counts": {
                    "kxnbagame": len([t for t in dbg_tickers if t.startswith("KXNBAGAME")]),
                    "kxnba_futures": len([
                        t for t in dbg_tickers if t.startswith("KXNBA") and not t.startswith("KXNBAGAME")
                    ]),
                    "total": len(dbg_tickers),
                }
            }
        )
    filter_stats = st.session_state.get("kalshi_filter_stats") or {}
    if filter_stats:
        st.subheader("Kalshi filtering stats")
        st.json(
            {
                "total_markets_fetched": filter_stats.get("total_markets_fetched"),
                "total_game_markets": filter_stats.get("total_game_markets"),
                "avg_filtered_markets_per_game": filter_stats.get(
                    "avg_filtered_markets_per_game"
                ),
                "first_game": filter_stats.get("first_game_expected"),
            }
        )
        st.json(filter_stats)
    prefix_counts = st.session_state.get("kalshi_prefix_counts")
    if prefix_counts:
        st.subheader("Kalshi ticker prefix counts")
        st.json(prefix_counts)
        samples = st.session_state.get("kalshi_prefix_samples_game") or []
        if samples:
            st.caption("First 20 KXNBAGAME tickers")
            st.json(samples)
    all_markets_debug = st.session_state.get("kalshi_all_markets") or []
    if games and all_markets_debug:
        fg = games[0]
        home_code_dbg = nba_abbrev(fg.get("home_team"))
        away_code_dbg = nba_abbrev(fg.get("away_team"))
        search_results = debug_search_markets_for_game(
            all_markets_debug,
            fg.get("home_team"),
            fg.get("away_team"),
            home_code_dbg,
            away_code_dbg,
            limit=15,
        )
        st.subheader("Kalshi full-market search (first game)")
        st.json(
            {
                "expected_codes": {"home": home_code_dbg, "away": away_code_dbg},
                "found_any_winner_market_for_game": search_results.get(
                    "found_any_winner_market_for_game"
                ),
                "found_any_total_market_for_game": search_results.get(
                    "found_any_total_market_for_game"
                ),
                "found_any_spread_market_for_game": search_results.get(
                    "found_any_spread_market_for_game"
                ),
                "counts": search_results.get("counts"),
                "top_matches": search_results.get("matches"),
            }
        )
    if st.session_state.get("kalshi_match_results"):
        matches = st.session_state.get("kalshi_match_results")
        matched = []
        non_match_reasons: List[str] = []
        for m in matches:
            for res in (m.get("matches") or {}).values():
                if res.get("kalshi_matched"):
                    matched.append(res)
                else:
                    non_match_reasons.append(res.get("kalshi_reason") or "unknown")
        if matched:
            st.caption("Sample matched market")
            st.json(matched[0])
        first_game_debug = filter_stats.get("first_game_debug") or {}
        if first_game_debug:
            st.caption("Kalshi per-game debug (first game)")
            st.json(first_game_debug)
        if filter_stats.get("per_game_debug"):
            st.caption("Kalshi per-game debug (all games)")
            st.json(filter_stats.get("per_game_debug"))
        if non_match_reasons:
            reasons: Dict[str, int] = {}
            for reason in non_match_reasons:
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
