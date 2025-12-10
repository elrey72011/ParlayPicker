"""
Kalshi Integrator with team-aware, league-aware fuzzy matching.
Updated to accept API keys directly in constructor.

Drop this file in: app_core/kalshi_integrator.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    if not name:
        return ""
    return (
        str(name)
        .strip()
        .upper()
        .replace("&", "AND")
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
    )


def price_to_prob(price: Any) -> Optional[float]:
    """
    Convert a Kalshi-style price or probability into a 0–1 float.
    Kalshi prices are usually in [0, 1] already for contracts.
    """
    if price is None:
        return None
    try:
        val = float(price)
    except Exception:
        return None
    if val < 0:
        return None
    # If looks like a decimal prob already, just clamp.
    if 0.0 <= val <= 1.0:
        return max(0.0, min(1.0, val))
    # If given in percent (0–100), convert.
    if 0.0 <= val <= 100.0:
        return max(0.0, min(1.0, val / 100.0))
    return None


# ---------------------------------------------------------------------------
# League / team metadata
# ---------------------------------------------------------------------------

SUPPORTED_LEAGUES = {"NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"}

LEAGUE_SERIES_MAP: Dict[str, str] = {
    "NBA": "KXNBA",
    "NFL": "KXNFL",
    "MLB": "KXMLB",
    "NHL": "KXNHL",
    "NCAAF": "KXNCAAF",
    "NCAAB": "KXNCAAB",
}

# Abbreviation mapping – not exhaustive, but covers major pro teams.
KALSHI_TEAM_ABBREVIATIONS: Dict[str, List[str]] = {
    # NBA
    "ATLANTA HAWKS": ["ATL"],
    "BOSTON CELTICS": ["BOS"],
    "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CLT"],
    "CHICAGO BULLS": ["CHI"],
    "CLEVELAND CAVALIERS": ["CLE"],
    "DALLAS MAVERICKS": ["DAL"],
    "DENVER NUGGETS": ["DEN"],
    "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW"],
    "HOUSTON ROCKETS": ["HOU"],
    "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC"],
    "LOS ANGELES LAKERS": ["LAL"],
    "MEMPHIS GRIZZLIES": ["MEM"],
    "MIAMI HEAT": ["MIA"],
    "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN"],
    "NEW ORLEANS PELICANS": ["NOP"],
    "NEW YORK KNICKS": ["NYK"],
    "OKLAHOMA CITY THUNDER": ["OKC"],
    "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI"],
    "PHOENIX SUNS": ["PHX"],
    "PORTLAND TRAIL BLAZERS": ["POR"],
    "SACRAMENTO KINGS": ["SAC"],
    "SAN ANTONIO SPURS": ["SAS"],
    "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"],
    "WASHINGTON WIZARDS": ["WAS", "WSH"],

    # NFL
    "ARIZONA CARDINALS": ["ARI"],
    "ATLANTA FALCONS": ["ATL"],
    "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"],
    "CAROLINA PANTHERS": ["CAR"],
    "CHICAGO BEARS": ["CHI"],
    "CINCINNATI BENGALS": ["CIN"],
    "CLEVELAND BROWNS": ["CLE"],
    "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"],
    "DETROIT LIONS": ["DET"],
    "GREEN BAY PACKERS": ["GB"],
    "HOUSTON TEXANS": ["HOU"],
    "INDIANAPOLIS COLTS": ["IND"],
    "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC"],
    "LAS VEGAS RAIDERS": ["LV"],
    "LOS ANGELES CHARGERS": ["LAC"],
    "LOS ANGELES RAMS": ["LAR"],
    "MIAMI DOLPHINS": ["MIA"],
    "MINNESOTA VIKINGS": ["MIN"],
    "NEW ENGLAND PATRIOTS": ["NE"],
    "NEW ORLEANS SAINTS": ["NO"],
    "NEW YORK GIANTS": ["NYG"],
    "NEW YORK JETS": ["NYJ"],
    "PHILADELPHIA EAGLES": ["PHI"],
    "PITTSBURGH STEELERS": ["PIT"],
    "SAN FRANCISCO 49ERS": ["SF"],
    "SEATTLE SEAHAWKS": ["SEA"],
    "TAMPA BAY BUCCANEERS": ["TB"],
    "TENNESSEE TITANS": ["TEN"],
    "WASHINGTON COMMANDERS": ["WAS", "WSH"],

    # NHL (subset)
    "BOSTON BRUINS": ["BOS"],
    "TORONTO MAPLE LEAFS": ["TOR"],
    "MONTREAL CANADIENS": ["MTL"],
    "NEW YORK RANGERS": ["NYR"],
    "CHICAGO BLACKHAWKS": ["CHI"],
    "DETROIT RED WINGS": ["DET"],
    "PITTSBURGH PENGUINS": ["PIT"],
    "TAMPA BAY LIGHTNING": ["TBL"],
    "VEGAS GOLDEN KNIGHTS": ["VGK"],
    "SEATTLE KRAKEN": ["SEA"],

    # MLB (subset)
    "ATLANTA BRAVES": ["ATL"],
    "NEW YORK YANKEES": ["NYY"],
    "NEW YORK METS": ["NYM"],
    "LOS ANGELES DODGERS": ["LAD"],
    "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"],
    "ST LOUIS CARDINALS": ["STL"],
    "SAN FRANCISCO GIANTS": ["SF"],
    "HOUSTON ASTROS": ["HOU"],
    "PHILADELPHIA PHILLIES": ["PHI"],
}

# Matching thresholds
TEAM_FUZZY_THRESHOLD = 1.5   # overall score threshold to accept a market
DATE_TOLERANCE_DAYS = 5
DATE_SOFT_PENALTY = 0.10


# ---------------------------------------------------------------------------
# Dataclass for match result
# ---------------------------------------------------------------------------

@dataclass
class KalshiMatchResult:
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: Optional[str] = None
    reason: str = ""
    market_type: Optional[str] = None
    direction: Optional[str] = None
    game_date: Optional[datetime] = None
    kalshi_volume: Optional[float] = None


# ---------------------------------------------------------------------------
# Helper functions for ticker parsing & market metadata
# ---------------------------------------------------------------------------

def _extract_teams_from_ticker(ticker: str) -> List[str]:
    """
    Many Kalshi tickers look like KXNBA-LAL-GSW-YYYYMMDD-...
    We try to pull out the middle tokens as team codes.
    """
    if not ticker:
        return []
    parts = ticker.split("-")
    if len(parts) < 3:
        return []
    # Ignore first prefix and trailing date/extra fields
    middle = parts[1:3]
    return [p.strip().upper() for p in middle if p.strip()]


def _extract_market_type(title: str, ticker: str) -> str:
    t = (title or "").upper()
    if "SPREAD" in t or "POINTS" in t:
        return "spread"
    if "TOTAL" in t or "OVER/UNDER" in t or "O/U" in t:
        return "total"
    if "MONEYLINE" in t or "ML" in t:
        return "moneyline"
    # Fallback
    return "generic"


def _parse_market_metadata(mkt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts: title, approximate date, teams, probability, market_type.
    """
    title = (mkt.get("title") or "").strip()
    if not title:
        return None

    ticker = (
        mkt.get("ticker")
        or mkt.get("event_ticker")
        or mkt.get("series_ticker")
        or ""
    )

    # Market date: from close_time / expected_expiration_time if available
    market_dt: Optional[datetime] = None
    close_raw = (
        mkt.get("close_time")
        or mkt.get("expiration_time")
        or mkt.get("expected_expiration_time")
    )
    if close_raw:
        try:
            market_dt = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
        except Exception:
            market_dt = None

    # Teams – first try the title, then fall back to ticker codes
    upper_title = title.upper()
    teams: List[str] = []

    separators = [" VS ", " VS. ", " V ", " @ ", " AT ", " - ", " / ", " | "]
    for sep in separators:
        if sep in upper_title:
            parts = upper_title.split(sep)
            if len(parts) >= 2:
                teams = [parts[0].strip(), parts[1].strip()]
            break

    if len(teams) < 2 and ticker:
        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) >= 2:
            teams = ticker_teams[:2]

    market_type = _extract_market_type(title, ticker)

    prob_source = (
        mkt.get("yes_price")
        or mkt.get("last_price")
        or mkt.get("last_yes_price")
        or mkt.get("implied_prob")
        or mkt.get("probability")
    )
    prob = price_to_prob(prob_source)

    return {
        "title": title,
        "market_date": market_dt,
        "teams": teams,
        "probability": prob,
        "market_type": market_type,
    }


def _build_team_codes(team_name: str) -> List[str]:
    """
    Return a list of possible codes for a given sportsbook team name,
    using KALSHI_TEAM_ABBREVIATIONS and basic normalization.
    """
    norm = normalize_name(team_name)
    codes: List[str] = []
    for full_name, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
        if normalize_name(full_name) == norm:
            codes.extend(abbrs)
    # Always include short tokens of the name (like NYK, GSW, etc.)
    tokens = norm.split()
    for t in tokens:
        if len(t) >= 2 and t not in codes:
            codes.append(t)
    return list(dict.fromkeys(codes))  # unique


def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
    """
    Score how well a Kalshi team token matches a sportsbook team name.
    """
    if not team_code:
        return 0.0

    clean_code = team_code.strip().upper()

    # 1) Code / abbreviation match (fast path)
    if clean_code in target_codes:
        return 2.0

    # 2) Fallback: simple normalization + word overlap
    norm_code = normalize_name(team_code)

    if norm_code == target_norm and norm_code:
        return 1.5

    if norm_code and target_norm and (norm_code in target_norm or target_norm in norm_code):
        return 1.2

    words_code = set(norm_code.split())
    words_target = set(target_norm.split())
    overlap = words_code & words_target
    if overlap:
        ratio = len(overlap) / max(len(words_target), 1)
        if ratio >= 0.5:
            return 1.0
        return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Core matching function
# ---------------------------------------------------------------------------

def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """
    Match an OddsAPI game to the best Kalshi market using TEAM-FIRST scoring,
    with date as a soft constraint (penalty + hard cutoff).
    """
    league_key = (league or "").upper()
    if league_key not in SUPPORTED_LEAGUES:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_key,
            reason="league_not_supported",
        )

    kalshi = integrator or KalshiIntegrator()
    if kalshi is None:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_key,
            reason="api_error:no_integrator",
        )

    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    markets = kalshi.get_markets(status=status)
    if not markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_key,
            reason="no_kalshi_markets",
        )

    # Normalize game time to UTC
    game_dt: Optional[datetime] = None
    if isinstance(game_time, datetime):
        if game_time.tzinfo is None:
            game_dt = pytz.UTC.localize(game_time)
        else:
            game_dt = game_time.astimezone(pytz.UTC)

    parsed_markets: List[Dict[str, Any]] = []
    for m in markets:
        meta = _parse_market_metadata(m)
        if not meta:
            continue
        m2 = dict(m)
        m2["__meta"] = meta
        parsed_markets.append(m2)

    if not parsed_markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_key,
            reason="no_parsable_markets",
        )

    series_prefix = LEAGUE_SERIES_MAP.get(league_key)

    best_market: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    best_day_diff: Optional[int] = None

    for market in parsed_markets:
        meta = market["__meta"]
        title = meta.get("title", "").upper()
        ticker = (market.get("ticker") or market.get("event_ticker") or "").upper()

        # Filter by series
        if series_prefix and ticker and not ticker.startswith(series_prefix):
            continue

        # Skip clear futures
        futures_noise = [
            "APPROVE",
            "FRANCHISE",
            "DRAFT",
            "MVP",
            "ROOKIE",
            "CHAMPION",
            "WINNER",
            "SEASON",
            "CONFERENCE",
            "DIVISION",
            "REGULAR SEASON WINS",
            "SEASON WINS",
        ]
        if any(bad in title for bad in futures_noise):
            continue

        teams = meta.get("teams") or []
        if len(teams) < 2:
            continue

        # Score both orientations
        score_home_first = (
            _team_score(teams[0], home_norm, home_codes)
            + _team_score(teams[1], away_norm, away_codes)
        )
        score_away_first = (
            _team_score(teams[0], away_norm, away_codes)
            + _team_score(teams[1], home_norm, home_codes)
        )
        team_score = max(score_home_first, score_away_first)
        if team_score <= 0.0:
            continue

        score = team_score

        # Date penalty / cutoff
        day_diff: Optional[int] = None
        market_dt: Optional[datetime] = meta.get("market_date")
        if game_dt and market_dt:
            day_diff = abs((market_dt.date() - game_dt.date()).days)
            if day_diff > DATE_TOLERANCE_DAYS:
                continue
            score -= DATE_SOFT_PENALTY * day_diff

        # Small bump if we have prob
        if meta.get("probability") is not None:
            score += 0.25

        if score > best_score:
            best_score = score
            best_market = market
            best_day_diff = day_diff

    if not best_market:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_key,
            reason="no_candidate_for_game",
        )

    if best_score < TEAM_FUZZY_THRESHOLD:
        reason = f"low_score:{best_score:.2f}"
        if best_day_diff is not None:
            reason += f":day_diff={best_day_diff}"
        return KalshiMatchResult(
            matched=False,
            kalshi_available=True,
            label=best_market["__meta"].get("title", ""),
            probability=None,
            raw_event_id=best_market.get("event_ticker") or best_market.get("ticker"),
            league=league_key,
            reason=reason,
            market_type=best_market["__meta"].get("market_type"),
            game_date=best_market["__meta"].get("market_date"),
            kalshi_volume=best_market.get("volume"),
        )

    best_meta = best_market["__meta"]
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=best_meta.get("title", ""),
        probability=best_meta.get("probability"),
        raw_event_id=best_market.get("event_ticker") or best_market.get("ticker"),
        league=league_key,
        reason="ok",
        market_type=best_meta.get("market_type"),
        direction=None,
        game_date=best_meta.get("market_date"),
        kalshi_volume=best_market.get("volume"),
    )


# ---------------------------------------------------------------------------
# KalshiIntegrator class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    """
    Thin wrapper around the Kalshi API with caching and simple key auth.
    Only the /markets endpoint is used for matching.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        """
        Initialize the integrator.
        Accepts optional arguments to satisfy dependency injection in main app.
        """
        self.api_key: Optional[str] = api_key
        # Note: api_secret is accepted for compatibility but primarily unused
        # in this simplified wrapper which relies on KEY header auth.
        self.api_secret: Optional[str] = api_secret
        
        self.api_url: str = os.getenv(
            "KALSHI_API_URL", "https://trading-api.kalshi.com/trade-api/v2"
        )
        self._markets_cache: List[Dict[str, Any]] = []
        self._markets_cache_ts: float = 0.0
        self.cache_ttl_seconds: int = 60
        self.last_error: Optional[str] = None
        self._auth_ready: bool = False

        if not self.api_key:
            self._load_config()
        else:
            self._auth_ready = True

    # ------------------------------------------------------------------
    # Config / auth
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """
        Load Kalshi API credentials from environment variables or Streamlit secrets.
        """
        try:
            self.api_key = (
                os.getenv("KALSHI_API_KEY")
                or st.secrets.get("KALSHI_API_KEY")  # type: ignore[attr-defined]
            )
        except Exception:
            self.api_key = os.getenv("KALSHI_API_KEY")

        self._auth_ready = bool(self.api_key)

        if not self.api_key:
            logger.warning("Kalshi API key not configured; Kalshi will be unavailable.")
        else:
            logger.info("Kalshi API key loaded.")

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[dict]:
        """
        Make an authenticated Kalshi API request.
        This simplified version uses only the API key header.
        """
        url = f"{self.api_url}{endpoint}"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["KALSHI-ACCESS-KEY"] = self.api_key

        try:
            resp = requests.request(
                method.upper(),
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                self.last_error = f"API error: {resp.status_code}"
                logger.warning(
                    f"Kalshi {method} {endpoint} -> {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi request failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_markets(
        self,
        status: str = "open",
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch Kalshi markets, with simple in-memory caching. We don't filter by
        league here; that is done in match_game_to_kalshi via series prefix.
        """
        now = time.time()
        if (
            not force_refresh
            and self._markets_cache
            and (now - self._markets_cache_ts) < self.cache_ttl_seconds
        ):
            return self._markets_cache

        params: Dict[str, Any] = {}
        if status:
            params["status"] = status

        data = self._make_authenticated_request("GET", "/markets", params=params)
        if not data:
            return []

        markets = data.get("markets") or data.get("data") or data.get("result") or []
        if not isinstance(markets, list):
            logger.warning("Kalshi /markets response did not contain a list.")
            return []

        self._markets_cache = markets
        self._markets_cache_ts = now
        logger.info(f"✅ Fetched {len(markets)} Kalshi markets.")
        return markets

    # Backwards-compat aliases (used by existing code)
    get_sports_markets = get_markets

    def get_game_markets_for_events(self, league: str) -> List[Dict[str, Any]]:
        return self.get_markets()

    def get_sports_series(self) -> List[Dict[str, Any]]:
        return []

    def filter_markets_closing_today(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return markets

    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        data = self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook")
        if not data:
            return {}
        return data.get("orderbook", {}) or {}
