"""
Kalshi Integrator with Proper RSA-PSS Authentication & Improved Matching
This file goes in: app_core/kalshi_integrator.py
"""

import copy
import time
import logging
import re
import string
import uuid
from datetime import datetime, timedelta
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict, Tuple

# Import the improved matcher
try:
    from app_core.team_name_matcher import TeamNameMatcher
except ImportError:
    # Fallback if not found (e.g. running standalone)
    class TeamNameMatcher:
        @staticmethod
        def normalize(s): return s.lower().strip()
        @staticmethod
        def match_team(t, options, threshold=0.8): return None
        @staticmethod
        def similarity_score(a, b): return 0.0

logger = logging.getLogger(__name__)

class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str

SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}
# Map common league codes to Kalshi Series Prefixes
LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

# Slightly lower threshold for Kalshi because their titles can be weird
TEAM_FUZZY_THRESHOLD = 0.65

def price_to_prob(price) -> Optional[float]:
    """Convert a Kalshi price (dollars or fraction) to probability."""
    if price is None or price == "":
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p > 1.01:
        p = p / 100.0
    return max(0.0, min(1.0, p))

def _parse_market_date(raw) -> Optional[datetime]:
    """Parse Kalshi timestamps (ms or iso) into a date."""
    if raw is None or raw == "":
        return None
    try:
        # Kalshi close_time is milliseconds since epoch
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=pytz.UTC)
        # ISO timestamp string
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except Exception:
        return None

# ==============================================================================
# UNIFIED MATCHING LOGIC
# ==============================================================================

def _find_best_market_match(
    home_team: str,
    away_team: str,
    game_dt: Optional[datetime],
    markets: List[Dict[str, Any]],
    league_code: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Shared logic to find the best market for a game using TeamNameMatcher.
    Returns (best_market, reason_code).
    """
    if not markets:
        return None, "no_events"

    # 1. Pre-filter by date (if provided) with a generous window
    date_filtered = markets
    if game_dt:
        date_filtered = []
        # Ensure game_dt is aware
        if game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=pytz.UTC)
        else:
            game_dt = game_dt.astimezone(pytz.UTC)
        
        target_date = game_dt.date()
        
        for m in markets:
            m_dt = _parse_market_date(m.get("close_time") or m.get("event_date"))
            if not m_dt:
                continue # Skip if no date
                
            # Allow same day match (Kalshi markets usually close approx match end)
            # We accept markets closing within +/- 24 hours to handle timezone shifts
            m_date = m_dt.astimezone(pytz.UTC).date()
            if abs((m_date - target_date).days) <= 1:
                date_filtered.append(m)
        
        if not date_filtered:
            return None, "date_mismatch"
    
    # 2. Score matches
    best_market = None
    best_score = 0.0
    
    # Normalize inputs
    norm_home = TeamNameMatcher.normalize(home_team)
    norm_away = TeamNameMatcher.normalize(away_team)
    
    for market in date_filtered:
        # Check series prefix if strict league provided
        series = str(market.get("series_ticker") or "").upper()
        if league_code and league_code in LEAGUE_SERIES_MAP:
            expected_prefix = LEAGUE_SERIES_MAP[league_code]
            if not series.startswith(expected_prefix):
                continue

        # Construct a searchable string from the market
        title = market.get("title") or ""
        ticker = market.get("ticker") or ""
        subtitle = market.get("subtitle") or ""
        
        # Combine title parts for matching: "Los Angeles Lakers vs Golden State Warriors"
        market_text = f"{title} {subtitle} {ticker}"
        market_norm = TeamNameMatcher.normalize(market_text)
        
        # 2a. Check if BOTH teams appear in the market text
        # We check if the "normalized" team name is a substring of the "normalized" market text
        home_in = norm_home in market_norm
        away_in = norm_away in market_norm
        
        match_quality = 0.0
        
        if home_in and away_in:
            match_quality = 1.0
        else:
            # 2b. Fallback: Fuzzy score
            home_score = TeamNameMatcher.similarity_score(norm_home, market_norm)
            away_score = TeamNameMatcher.similarity_score(norm_away, market_norm)
            
            # Simple average score, but ONLY if both have some relevance
            if home_score > 0.4 and away_score > 0.4:
                match_quality = (home_score + away_score) / 2.0
            
        # Boost for "GAME" or "SPREAD" or "TOTAL" tickers to prefer main lines over player props
        if "GAME" in ticker or "SPREAD" in ticker:
            match_quality += 0.1
            
        if match_quality > best_score:
            best_score = match_quality
            best_market = market

    if best_score < TEAM_FUZZY_THRESHOLD:
        return None, "below_threshold"

    return best_market, "ok"


# ==============================================================================
# KALSHI INTEGRATOR CLASS
# ==============================================================================

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        print("="*80)
        print("🚀 KALSHI INTEGRATOR v3.0 - PAGINATION FIX - INITIALIZING")
        print("="*80)
        
        # Use Streamlit secrets if keys not provided directly
        try:
            self.api_key = api_key or st.secrets.get("KALSHI_API_KEY")
            self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET")
        except Exception:
            self.api_key = api_key
            self.api_secret = api_secret

        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.api_url = self.base_url if self.api_key else self.demo_url
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # RSA Auth Setup
        self._private_key = None
        self._auth_ready = False
        
        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                
                key_data = self.api_secret.strip()
                if not key_data.startswith('-----BEGIN'):
                    key_data = f"-----BEGIN RSA PRIVATE KEY-----\n{key_data}\n-----END RSA PRIVATE KEY-----"
                
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(),
                    password=None,
                    backend=default_backend()
                )
                self._auth_ready = True
                logger.info(f"✅ Kalshi RSA key loaded successfully")
            except ImportError:
                logger.warning("cryptography library not installed")
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")

        # Cache
        self._markets_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_duration = 300  # 5 minutes
        self.last_error = None
        
    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Create RSA signature for Kalshi API request"""
        if not self._private_key:
            return ""
        
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            message = f"{timestamp}{method}{path}"
            
            signature = self._private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error signing Kalshi request: {e}")
            return ""
    
    def _make_authenticated_request(self, method: str, endpoint: str, params: dict = None) -> Optional[dict]:
        import time as time_module
        
        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))
        headers = self.headers.copy()
        
        if self._auth_ready:
            path_without_query = endpoint.split('?')[0]
            signature = self._sign_request(method.upper(), path_without_query, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                response = requests.post(url, headers=headers, json=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("⚠️ Kalshi Rate Limit Hit. Sleeping 1s...")
                time.sleep(1.0)
                # Retry once
                return self._make_authenticated_request(method, endpoint, params)
            else:
                logger.error(f"Kalshi API Error {response.status_code}: {response.text}")
                self.last_error = f"API Error {response.status_code}"
                return None
                
        except Exception as e:
            logger.error(f"Kalshi Request Failed: {e}")
            self.last_error = str(e)
            return None

    def get_sports_series(self) -> List[Dict]:
        """Get all available sports series tickers"""
        try:
            # We want specific sports, but fetching all series is safer to filter locally
            endpoint = "/series"
            params = {"limit": 1000} # Max limit
            data = self._make_authenticated_request("GET", endpoint, params=params)
            
            if not data: return []
            
            all_series = data.get("series", [])
            sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'NCAAF', 'NCAAB', 'COLLEGE', 'BASKETBALL', 'FOOTBALL']
            
            # Filter
            sports_series = []
            for s in all_series:
                t = s.get('ticker', '').upper()
                c = s.get('category', '').upper()
                if any(k in t or k in c for k in sports_keywords):
                    sports_series.append(s)
            
            return sports_series
        except Exception as e:
            logger.error(f"Error getting series: {e}")
            return []

    def get_game_markets_for_events(self, league: str = "NBA") -> List[Dict[str, Any]]:
        """Efficiently get markets for a specific league using prefix filtering"""
        league = league.upper()
        # Map common names to prefixes
        prefix_map = {"NBA": "KXNBA", "NFL": "KXNFL", "MLB": "KXMLB", "NHL": "KXNHL", "NCAAB": "KXNCAAB"}
        prefix = prefix_map.get(league, f"KX{league}")
        
        # 1. Fetch Series matching prefix
        all_series = self.get_sports_series()
        target_series = [s for s in all_series if s.get('ticker', '').startswith(prefix)]
        
        if not target_series:
            logger.warning(f"No series found for league {league} (prefix {prefix})")
            return []

        all_markets = []
        
        # 2. Fetch markets for these series (Paginated)
        for series in target_series:
            ticker = series.get('ticker')
            
            # Pagination Loop
            cursor = None
            page_count = 0
            while True:
                params = {
                    "series_ticker": ticker,
                    "limit": 100, # Max per page
                    "status": "open"
                }
                if cursor:
                    params["cursor"] = cursor
                    
                data = self._make_authenticated_request("GET", "/markets", params=params)
                if not data:
                    break
                    
                markets = data.get("markets", [])
                all_markets.extend(markets)
                
                cursor = data.get("cursor")
                page_count += 1
                
                if not cursor:
                    break
                
                # Safety break for huge series
                if page_count > 20: # Limit to 2000 markets per series
                    break
                    
        return all_markets

    def get_markets(self, category: str = "sports", status: Optional[str] = "open") -> List[Dict]:
        """
        Broad fetch for markets. 
        """
        cache_key = f"{category}_{status}"
        now = time.time()
        if cache_key in self._markets_cache and now - self._cache_time.get(cache_key, 0) < self._cache_duration:
            return self._markets_cache[cache_key]

        sports_series = self.get_sports_series()
        all_markets = []
        
        # Prioritize major leagues
        priority = ["KXNBA", "KXNFL", "KXMLB", "KXNHL"]
        sports_series.sort(key=lambda x: next((i for i, p in enumerate(priority) if x.get('ticker','').startswith(p)), 999))

        # Iterate all relevant series (REMOVED the [:10] limit!)
        for series in sports_series:
            ticker = series.get('ticker')
            
            # Fetch first page only for broad search to save time
            params = {"series_ticker": ticker, "limit": 100, "status": status}
            data = self._make_authenticated_request("GET", "/markets", params=params)
            
            if data:
                markets = data.get("markets", [])
                all_markets.extend(markets)

        self._markets_cache[cache_key] = all_markets
        self._cache_time[cache_key] = now
        return all_markets

    def get_game_market(
        self,
        home_team: str,
        away_team: str,
        sport: str = "NBA",
        game_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find and return Kalshi market data for a specific game.
        Uses optimized fetching based on 'sport' if provided.
        """
        
        # 1. Fetch relevant markets efficiently
        if sport and sport.upper() in ["NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF"]:
            markets = self.get_game_markets_for_events(sport)
        else:
            markets = self.get_markets(status="open")

        # 2. Parse game time
        game_dt = None
        if game_time:
            try:
                game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
            except:
                pass

        # 3. Match
        best_market, reason = _find_best_market_match(
            home_team, away_team, game_dt, markets, league_code=sport.lower() if sport else None
        )

        result = {
            "kalshi_available": False,
            "kalshi_prob": None,
            "market_ticker": None,
            "kalshi_match_debug": reason,
            "kalshi_label": None
        }

        if best_market:
            # Extract Probability
            kalshi_prob = None
            
            # Try direct fields
            for field in ["yes_bid_dollars", "last_price", "implied_prob"]:
                if field in best_market:
                    p = price_to_prob(best_market[field])
                    if p is not None:
                        kalshi_prob = p
                        break
            
            result.update({
                "kalshi_available": True,
                "kalshi_prob": kalshi_prob,
                "market_ticker": best_market.get("ticker"),
                "kalshi_label": best_market.get("title"),
                "kalshi_match_debug": "match_found"
            })
            
        return result

    def get_orderbook(self, ticker: str) -> Dict:
        """Helper to get orderbook for a specific ticker"""
        endpoint = f"/markets/{ticker}/orderbook"
        return self._make_authenticated_request("GET", endpoint) or {}

# ==============================================================================
# LEGACY WRAPPERS
# ==============================================================================

def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """Wrapper for legacy calls"""
    kalshi = integrator or KalshiIntegrator()
    
    # Map full league names to codes if needed
    sport_code = None
    for code in SUPPORTED_LEAGUES:
        if code in league.lower():
            sport_code = code
            break
            
    result = kalshi.get_game_market(home_team, away_team, sport=sport_code, game_time=game_time)
    
    return KalshiMatchResult(
        matched=result["kalshi_available"],
        label=result.get("kalshi_label", ""),
        probability=result.get("kalshi_prob"),
        raw_event_id=result.get("market_ticker"),
        league=league,
        reason=result.get("kalshi_match_debug", "unknown")
    )
