"""
Kalshi Integrator with Proper RSA-PSS Authentication
This file goes in: app_core/kalshi_integrator.py or app_core/__init__.py
"""

import copy
import time
import logging
import re
import string
from difflib import SequenceMatcher
from datetime import datetime
import pytz
import requests
import streamlit as st
from typing import Dict, List, Any, Optional, TypedDict

from app_core.team_name_matcher import TeamNameMatcher

logger = logging.getLogger(__name__)


class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str


SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}
LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

TEAM_FUZZY_THRESHOLD = 0.6
MAX_LINE_DIFF = 3.0


def price_to_prob(price) -> Optional[float]:
    """Convert a Kalshi price (dollars or fraction) to probability.

    Returns None when the input cannot be parsed instead of defaulting to 0.5.
    """
    if price is None or price == "":
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p > 1.01:
        p = p / 100.0
    if 0 <= p <= 1:
        return p
    return None


def _normalize_market_text(*parts: str) -> str:
    """Normalize free text for matching: lowercase, strip punctuation, collapse spaces."""

    joined = " ".join([p or "" for p in parts])
    cleaned = joined.lower()
    cleaned = re.sub(r"[\.,\-_/]", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_market_date(raw) -> Optional[datetime]:
    """Parse Kalshi timestamps (ms or iso) into a date for day-level matching."""

    if raw is None or raw == "":
        return None

    try:
        # Kalshi close_time may be milliseconds since epoch
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0)
        # ISO timestamp string
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def normalize_name(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_game_to_kalshi(
    league: str,
    home_team: str,
    away_team: str,
    game_time: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """Attempt to match a game to a Kalshi market with explicit reasons."""

    league_norm = normalize_name(league)
    if league_norm and league_norm not in SUPPORTED_LEAGUES:
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="league_not_supported",
        )

    kalshi = integrator or KalshiIntegrator()
    if kalshi is None:
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="api_error:no_integrator",
        )

    # Normalize game time
    game_dt: Optional[datetime] = None
    if isinstance(game_time, datetime):
        game_dt = game_time
    elif game_time:
        try:
            game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
        except Exception:
            game_dt = None

    try:
        markets = kalshi.get_markets(status=status)
    except Exception as exc:  # pragma: no cover - defensive logging path
        short_err = str(exc)
        if len(short_err) > 80:
            short_err = short_err[:77] + "..."
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason=f"api_error:{short_err}",
        )

    if not markets:
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_events",
        )

    norm_home = normalize_name(home_team)
    norm_away = normalize_name(away_team)
    game_date = game_dt.date() if game_dt else None

    best_market: Optional[Dict[str, Any]] = None
    best_score = 0.0
    any_team_hit = False
    any_date_hit = False

    for market in markets:
        series = str(market.get("series_ticker") or "").upper()
        if league_norm and league_norm in LEAGUE_SERIES_MAP:
            expected = LEAGUE_SERIES_MAP[league_norm]
            if expected and series and not series.startswith(expected):
                continue

        title = market.get("title") or ""
        subtitle = market.get("subtitle") or ""
        ticker = market.get("ticker") or ""
        market_text = normalize_name(" ".join([title, subtitle, ticker]))

        home_ratio = SequenceMatcher(None, norm_home, market_text).ratio() if norm_home else 0.0
        away_ratio = SequenceMatcher(None, norm_away, market_text).ratio() if norm_away else 0.0

        home_hit = bool(norm_home) and (norm_home in market_text or home_ratio >= TEAM_FUZZY_THRESHOLD)
        away_hit = bool(norm_away) and (norm_away in market_text or away_ratio >= TEAM_FUZZY_THRESHOLD)

        team_score = 0.0
        if home_hit:
            team_score += max(1.0, home_ratio)
        if away_hit:
            team_score += max(1.0, away_ratio)

        if team_score > 0:
            any_team_hit = True
        else:
            if max(home_ratio, away_ratio) > 0:
                any_team_hit = True
            continue

        market_dt = _parse_market_date(market.get("close_time") or market.get("event_date"))
        date_ok = False
        if game_date and market_dt:
            try:
                diff_days = abs((market_dt.date() - game_date).days)
                date_ok = diff_days <= 1
            except Exception:
                date_ok = False
        elif not game_date:
            date_ok = True

        if date_ok:
            any_date_hit = True
        else:
            continue

        score = team_score + (0.5 if date_ok else 0)
        if expected := LEAGUE_SERIES_MAP.get(league_norm, ""):
            if series.startswith(expected):
                score += 0.25
        if ticker and "GAME" in ticker.upper():
            score += 0.1

        if score > best_score:
            best_score = score
            best_market = market

    if not best_market:
        reason = "team_mismatch"
        if any_team_hit and not any_date_hit:
            reason = "date_mismatch"
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason=reason,
        )

    if best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
            league=league_norm,
            reason="below_threshold",
        )

    prob_fields = [
        "yes_bid_dollars",
        "yes_ask_dollars",
        "last_price",
        "last_trade_price",
        "yes_bid",
        "yes_ask",
    ]

    probability: Optional[float] = None
    for field in prob_fields:
        candidate = best_market.get(field)
        prob_candidate = price_to_prob(candidate)
        if prob_candidate is not None:
            probability = prob_candidate
            break

    # Fall back to live orderbook if available
    if probability is None and best_market.get("ticker"):
        try:
            order = kalshi.get_orderbook(best_market.get("ticker")) or {}
            yes_levels = None
            for key in ("yes", "orderbook_yes", "levels"):
                level = (order.get(key) or order.get("orderbook", {}).get(key) or {})
                if level:
                    yes_levels = level
                    break
            if isinstance(yes_levels, list) and yes_levels:
                price_val = yes_levels[0].get("price") or yes_levels[0].get("bid")
                if price_val is not None:
                    probability = price_to_prob(price_val)
        except Exception as exc:  # pragma: no cover
            short_err = str(exc)
            if len(short_err) > 80:
                short_err = short_err[:77] + "..."
            return KalshiMatchResult(
                matched=False,
                label="",
                probability=None,
                raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
                league=league_norm,
                reason=f"api_error:{short_err}",
            )

    if probability is None:
        return KalshiMatchResult(
            matched=False,
            label="",
            probability=None,
            raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
            league=league_norm,
            reason="no_price",
        )

    probability = max(0.0, min(1.0, float(probability)))

    result = KalshiMatchResult(
        matched=True,
        label=best_market.get("title") or best_market.get("ticker") or "",
        probability=probability,
        raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
        league=league_norm,
        reason="ok",
    )

    print(
        f"[Kalshi] league={league} game={home_team} vs {away_team} date={game_time} -> {result['reason']}"
    )
    return result


def get_match_for_game(
    league: str,
    home_team: str,
    away_team: str,
    game_date: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> KalshiMatchResult:
    """Compatibility wrapper that delegates to match_game_to_kalshi."""

    return match_game_to_kalshi(
        league=league,
        home_team=home_team,
        away_team=away_team,
        game_time=game_date,
        integrator=integrator,
        status=status,
    )


def fetch_kalshi_for_game(
    home_team: str,
    away_team: str,
    game_date: Optional[datetime],
    integrator: "KalshiIntegrator" = None,
    status: Optional[str] = "open",
) -> Optional[Dict[str, Any]]:
    """Legacy wrapper retained for compatibility. Prefer get_match_for_game."""

    result = get_match_for_game("", home_team, away_team, game_date, integrator, status)
    if result.get("matched"):
        return {
            "kalshi_label": result.get("label"),
            "kalshi_probability": result.get("probability"),
            "kalshi_volume": None,
            "kalshi_match_debug": result.get("reason"),
        }
    return {
        "kalshi_label": None,
        "kalshi_probability": None,
        "kalshi_volume": None,
        "kalshi_match_debug": result.get("reason", "no_match"),
    }

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis
    
    Kalshi uses RSA signature authentication for API requests.
    The API key is a UUID and the secret is an RSA private key.
    
    Keys are loaded from Streamlit secrets:
    - st.secrets["KALSHI_API_KEY"]
    - st.secrets["KALSHI_API_SECRET"]
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        print("="*80)
        print("🚀 KALSHI INTEGRATOR v2.0 - SECRETS VERSION - INITIALIZING")
        print("="*80)
        
        # Use Streamlit secrets if keys not provided directly
        try:
            self.api_key = api_key or st.secrets.get("KALSHI_API_KEY")
            self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET")
            print(f"🔑 KALSHI: API key loaded: {bool(self.api_key)}")
            print(f"🔑 KALSHI: API secret loaded: {bool(self.api_secret)}")
            if self.api_key:
                print(f"🔑 KALSHI: Key preview: {self.api_key[:8]}...")
        except Exception:
            # Fallback if secrets not available (e.g., local testing)
            print(f"⚠️ KALSHI: Secrets not available, using provided values")
            self.api_key = api_key
            self.api_secret = api_secret
            print(f"🔑 KALSHI: Fallback - API key: {bool(self.api_key)}, secret: {bool(self.api_secret)}")
        
        # Kalshi API URLs - use elections subdomain (verified working)
        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        
        # Use production API if we have credentials
        self.api_url = self.base_url if self.api_key else self.demo_url
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # RSA key for signing (parsed from api_secret)
        self._private_key = None
        self._auth_ready = False
        
        if self.api_key and self.api_secret:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
                
                # Clean up the key if needed
                key_data = self.api_secret.strip()
                if not key_data.startswith('-----BEGIN'):
                    key_data = f"-----BEGIN RSA PRIVATE KEY-----\n{key_data}\n-----END RSA PRIVATE KEY-----"
                
                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(),
                    password=None,
                    backend=default_backend()
                )
                self._auth_ready = True
                logger.info(f"✅ Kalshi RSA key loaded successfully (key: {self.api_key[:8]}...)")
            except ImportError:
                logger.warning("cryptography library not installed - Kalshi auth disabled")
                self._private_key = None
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")
                self._private_key = None

        # Synthetic fallback cache when Kalshi API is unavailable
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None
        
        # Cache for API responses
        self._markets_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_duration = 300  # 5 minutes
        self._match_attempts = 0
        self._match_success = 0
        
    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Create RSA signature for Kalshi API request"""
        if not self._private_key:
            return ""
        
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            # Message format: timestamp + method + path
            message = f"{timestamp}{method}{path}"
            
            # Kalshi requires PSS padding, not PKCS1v15
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
        """Make authenticated request to Kalshi API
        
        Important: Per Kalshi docs, we sign ONLY the path (without query params),
        but send the full URL with query params in the actual request.
        """
        import time as time_module
        
        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))
        
        print(f"🌐 KALSHI: Request {method} {url}")
        print(f"🌐 KALSHI: Params: {params}")
        logger.info(f"Kalshi API request: {method} {url} with params: {params}")
        
        headers = self.headers.copy()
        
        if self._auth_ready and self._private_key:
            # CRITICAL: Strip query parameters from endpoint before signing
            # Per Kalshi docs: "Strip query parameters from path before signing"
            path_without_query = endpoint.split('?')[0]
            
            signature = self._sign_request(method.upper(), path_without_query, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
            
            print(f"🔐 KALSHI: Auth configured (key: {self.api_key[:8]}...)")
            logger.debug(f"Signing: {timestamp}{method.upper()}{path_without_query}")
            logger.info(f"Using API URL: {self.api_url}")
        else:
            print("⚠️ KALSHI: No authentication configured!")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                response = requests.post(url, headers=headers, json=params, timeout=15)
            
            print(f"📥 KALSHI: Response Status {response.status_code}")
            logger.info(f"Kalshi API response: Status {response.status_code}, URL: {response.url}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ KALSHI: Response keys: {list(data.keys())}")
                    logger.info(f"Response data keys: {list(data.keys())}")
                    if 'markets' in data:
                        print(f"✅ KALSHI: Markets in response: {len(data.get('markets', []))}")
                        logger.info(f"Markets in response: {len(data.get('markets', []))}")
                    if 'series' in data:
                        print(f"✅ KALSHI: Series in response: {len(data.get('series', []))}")
                    self.last_error = None
                    return data
                except Exception as json_error:
                    print(f"❌ KALSHI: Failed to parse JSON: {json_error}")
                    logger.error(f"Failed to parse JSON response: {json_error}")
                    logger.error(f"Response text: {response.text[:500]}")
                    self.last_error = f"JSON parse error: {json_error}"
                    return None
            elif response.status_code == 401:
                print(f"❌ KALSHI: Authentication failed (401)")
                logger.warning(f"Kalshi API authentication failed - Response: {response.text[:200]}")
                self.last_error = "Authentication failed"
            elif response.status_code == 403:
                print(f"❌ KALSHI: Access forbidden (403)")
                logger.warning(f"Kalshi API access forbidden - Response: {response.text[:200]}")
                self.last_error = "Access forbidden"
            else:
                print(f"❌ KALSHI: API error {response.status_code}")
                logger.warning(f"Kalshi API error: {response.status_code} - {response.text[:200]}")
                self.last_error = f"API error: {response.status_code}"
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ KALSHI: Connection error (network blocked?): {str(e)[:100]}")
            logger.error(f"Kalshi API connection error (network may be blocked): {e}")
            self.last_error = f"Connection blocked - check network settings"
        except requests.exceptions.Timeout:
            print(f"❌ KALSHI: Request timeout")
            logger.warning("Kalshi API timeout")
            self.last_error = "Request timeout"
        except Exception as e:
            print(f"❌ KALSHI: Request failed: {str(e)[:100]}")
            logger.error(f"Kalshi API request failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.last_error = str(e)
        
        return None
    
    def is_configured(self) -> bool:
        """Check if Kalshi is properly configured"""
        return bool(self.api_key and self._auth_ready)

    def get_sports_series(self) -> List[Dict]:
        """Get all available sports series tickers from Kalshi
        
        Returns list of series with tickers like:
        - KXNFL (NFL markets)
        - KXNBA (NBA markets)  
        - KXMLB (MLB markets)
        etc.
        """
        try:
            endpoint = "/series"
            params = {"limit": 200}
            
            print("🔍 KALSHI: Fetching series list...")  # Force output to logs
            logger.info("Fetching Kalshi series list...")
            response_data = self._make_authenticated_request("GET", endpoint, params=params)
            
            if response_data:
                all_series = response_data.get("series", [])
                print(f"🔍 KALSHI: Found {len(all_series)} total series")
                logger.info(f"Found {len(all_series)} total series")
                
                # Filter for sports-related series
                sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS', 
                                  'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY',
                                  'SPORT', 'GAME']
                
                sports_series = []
                for series in all_series:
                    ticker = series.get('ticker', '').upper()
                    title = series.get('title', '').upper()
                    category = series.get('category', '').upper()
                    
                    if any(keyword in ticker or keyword in title or keyword in category 
                           for keyword in sports_keywords):
                        sports_series.append(series)
                        print(f"🏈 KALSHI: Found sports series: {series.get('ticker')} - {series.get('title')}")
                        logger.info(f"Found sports series: {series.get('ticker')} - {series.get('title')}")
                
                print(f"✅ KALSHI: Total {len(sports_series)} sports series found")
                return sports_series
            else:
                print("❌ KALSHI: No response from /series endpoint")
                logger.error("No response from /series endpoint")
            
            return []
            
        except Exception as e:
            print(f"❌ KALSHI: Error fetching sports series: {e}")
            logger.error(f"Error fetching sports series: {e}")
            import traceback
            print(f"❌ KALSHI: Traceback: {traceback.format_exc()}")
            return []
    
    def get_markets(self, category: str = "sports", status: Optional[str] = "open") -> List[Dict]:
        """Fetch available Kalshi markets.

        Args:
            category: 'sports', 'politics', 'economics', etc. (used to filter series)
            status: 'open', 'closed', 'settled'

        Returns:
            List of market dictionaries.
        """
        if self._using_synthetic_data:
            return []

        cache_key = status or "all"
        now = time.time()
        if (
            cache_key in self._markets_cache
            and cache_key in self._cache_time
            and now - self._cache_time[cache_key] < self._cache_duration
        ):
            logger.info("Kalshi get_markets cache hit for status=%s", cache_key)
            return copy.deepcopy(self._markets_cache.get(cache_key, []))

        try:
            # First, get sports series tickers
            sports_series = self.get_sports_series()
            
            if not sports_series:
                logger.warning("No sports series found")
                self.last_error = "No sports series available"
                self._using_synthetic_data = True
                return []
            
            logger.info(f"Found {len(sports_series)} sports series")
            
            # Collect markets from all sports series
            all_markets = []
            
            for series in sports_series[:10]:  # Limit to first 10 series to avoid rate limits
                series_ticker = series.get('ticker')
                if not series_ticker:
                    continue
                
                logger.info(f"Fetching markets for series: {series_ticker}")
                
                endpoint = "/markets"
                params = {
                    "series_ticker": series_ticker,  # CRITICAL: Use series_ticker parameter
                    "limit": 200,
                }
                if status:
                    params["status"] = status
                
                response_data = self._make_authenticated_request("GET", endpoint, params=params)
                
                if response_data:
                    markets = response_data.get("markets", [])
                    logger.info(f"Got {len(markets)} markets from {series_ticker}")
                    all_markets.extend(markets)
                    
                    if len(all_markets) >= 100:
                        # We have enough markets, stop querying
                        break
            
            if all_markets:
                self.last_error = None
                logger.info(f"✅ Loaded {len(all_markets)} total Kalshi sports markets")
                
                # Log sample tickers
                sample_tickers = [m.get('ticker', 'NO_TICKER') for m in all_markets[:5]]
                logger.info(f"Sample tickers: {sample_tickers}")
                
                self._markets_cache[cache_key] = all_markets
                self._cache_time[cache_key] = now
                return all_markets
            else:
                self.last_error = "Kalshi API returned no markets"
                logger.warning("No markets found in any sports series")

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error fetching Kalshi markets: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        # Return empty list if API fails (don't fallback to synthetic)
        self._using_synthetic_data = True
        return []
    
    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets"""
        try:
            all_markets = self.get_markets()
            all_markets = all_markets or []
            logger.info("Kalshi get_sports_markets → %d markets", len(all_markets))
            if all_markets[:3]:
                logger.info("Kalshi sample markets: %s", all_markets[:3])
        except Exception as e:
            logger.warning("Kalshi get_sports_markets error: %s", e)
            self.last_error = str(e)
            return []

        # Filter for sports-related markets
        sports_keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'UFC', 'SOCCER', 'TENNIS',
                          'GOLF', 'FOOTBALL', 'BASKETBALL', 'BASEBALL', 'HOCKEY']
        
        sports_markets = []
        for market in all_markets:
            title = market.get('title', '').upper()
            ticker = market.get('ticker', '').upper()
            
            if any(keyword in title or keyword in ticker for keyword in sports_keywords):
                sports_markets.append(market)
        
        return sports_markets
    
    def _get_today_timestamp_range(self) -> tuple:
        """Get UTC timestamp range for today (midnight to midnight)"""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Kalshi uses milliseconds
        min_ts = int(today_start.timestamp() * 1000)
        max_ts = int(today_end.timestamp() * 1000)
        
        return min_ts, max_ts
    
    def get_game_markets_for_events(self, league: str = "NBA") -> List[Dict[str, Any]]:
        """Get game-related markets for a specific league (NBA, NFL, MLB, NHL, etc.)
        
        Args:
            league: League code like "NBA", "NFL", "MLB", "NHL"
            
        Returns:
            List of enriched market dictionaries with game data
        """
        try:
            print(f"🏀 KALSHI: Fetching {league} game markets...")
            
            # Get all series
            endpoint = "/series"
            params = {"limit": 1000}
            response_data = self._make_authenticated_request("GET", endpoint, params=params)
            
            if not response_data:
                print(f"❌ KALSHI: Failed to get series list")
                return []
            
            all_series = response_data.get("series", [])
            print(f"🔍 KALSHI: Found {len(all_series)} total series")
            
            # Filter to league prefix and game-ish suffixes
            league_prefix = f"KX{league.upper()}"
            game_suffixes = ["GAME", "GAMES", "SPREAD", "TOTAL", "ANYTD", "PASSYDS", "RUSHYDS"]
            
            relevant_series = []
            for series in all_series:
                ticker = series.get('ticker', '').upper()
                if ticker.startswith(league_prefix):
                    if any(suffix in ticker for suffix in game_suffixes):
                        relevant_series.append(series)
                        print(f"  ✅ {ticker}")
            
            print(f"🎯 KALSHI: Found {len(relevant_series)} {league} game series")
            
            # Get markets for each series
            all_markets = []
            for series in relevant_series:
                series_ticker = series.get('ticker')
                if not series_ticker:
                    continue
                
                endpoint = "/markets"
                params = {
                    "series_ticker": series_ticker,
                    "limit": 1000,
                    "status": "open"
                }
                
                response_data = self._make_authenticated_request("GET", endpoint, params=params)
                
                if response_data:
                    markets = response_data.get("markets", [])
                    print(f"  📊 {series_ticker}: {len(markets)} markets")
                    
                    # Enrich each market
                    for m in markets:
                        enriched = {
                            "league": league,
                            "series_ticker": series_ticker,
                            "ticker": m.get("ticker"),
                            "event_ticker": m.get("event_ticker"),
                            "title": m.get("title"),
                            "subtitle": m.get("subtitle"),
                            "yes_bid": m.get("yes_bid"),
                            "yes_ask": m.get("yes_ask"),
                            "no_bid": m.get("no_bid"),
                            "no_ask": m.get("no_ask"),
                            "yes_bid_dollars": m.get("yes_bid_dollars"),
                            "yes_ask_dollars": m.get("yes_ask_dollars"),
                            "no_bid_dollars": m.get("no_bid_dollars"),
                            "no_ask_dollars": m.get("no_ask_dollars"),
                            "close_time": m.get("close_time"),
                            "status": m.get("status"),
                        }
                        all_markets.append(enriched)
            
            print(f"✅ KALSHI: Total {len(all_markets)} {league} game markets")
            return all_markets
            
        except Exception as e:
            print(f"❌ KALSHI: Error fetching game markets: {e}")
            logger.error(f"Error fetching game markets for {league}: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []
    
    def filter_markets_closing_today(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter markets to only those closing today (in UTC)
        
        Args:
            markets: List of market dictionaries with 'close_time' field
            
        Returns:
            Filtered list of markets closing today
        """
        if not markets:
            return []
        
        min_ts, max_ts = self._get_today_timestamp_range()
        
        filtered = []
        for m in markets:
            close_ts = m.get("close_time")
            if isinstance(close_ts, (int, float)) and min_ts <= close_ts <= max_ts:
                filtered.append(m)
        
        return filtered
    
    def group_game_markets_by_event(self, markets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group markets by event_ticker (one event = one game)
        
        Args:
            markets: List of market dictionaries
            
        Returns:
            Dictionary mapping event_ticker to list of markets for that event
        """
        from collections import defaultdict
        
        grouped = defaultdict(list)
        for m in markets:
            event_ticker = m.get("event_ticker")
            if event_ticker:
                grouped[event_ticker].append(m)
        
        return dict(grouped)
    
    @staticmethod
    def price_to_prob(price_dollars: Optional[float]) -> Optional[float]:
        """Convert Kalshi price (0-1 dollars) to implied probability
        
        Args:
            price_dollars: Kalshi contract price in dollars (0.00 to 1.00)
            
        Returns:
            Implied probability (0.0 to 1.0) or None
        """
        if price_dollars is None:
            return None
        return float(price_dollars)  # Kalshi prices are already probabilities
    
    def get_game_market(
        self,
        home_team: str,
        away_team: str,
        sport: str = "NBA",
        game_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find and return Kalshi market data for a specific game.

        The matching is intentionally forgiving (punctuation/spacing-insensitive) and
        attempts partial matches to handle Kalshi's abbreviations.
        """

        result: Dict[str, Any] = {
            "kalshi_available": False,
            "kalshi_prob": None,
            "kalshi_home_prob": None,
            "kalshi_away_prob": None,
            "market_ticker": None,
            "market_title": None,
            "confidence": 0.0,
            "kalshi_match_debug": "no_market_match",
        }

        def normalize_name(name: str) -> str:
            cleaned = TeamNameMatcher.normalize(name)
            cleaned = re.sub(r"[^a-z0-9 ]", " ", (cleaned or "").lower())
            cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
            return " ".join(cleaned.split())

        def normalize_market_text(title: str, subtitle: str = "", ticker: str = "") -> str:
            parts = [title or "", subtitle or "", ticker or ""]
            joined = " ".join(parts)
            joined = joined.replace("_", " ").replace(".", " ")
            return normalize_name(joined)

        def teams_match(bet_team: str, market_text: str) -> bool:
            bet_norm = normalize_name(bet_team)
            if not bet_norm:
                return False
            text_norm = normalize_name(market_text)
            return bet_norm in text_norm or text_norm in bet_norm

        def extract_yes_levels(orderbook: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            candidates = [
                orderbook.get("yes"),
                (orderbook.get("bids") or {}).get("yes")
                if isinstance(orderbook.get("bids"), dict)
                else None,
                orderbook.get("yes_bids"),
                (orderbook.get("orderbook") or {}).get("yes")
                if isinstance(orderbook.get("orderbook"), dict)
                else None,
                (orderbook.get("orderbook", {}).get("bids") or {}).get("yes")
                if isinstance(orderbook.get("orderbook"), dict)
                and isinstance(orderbook.get("orderbook", {}).get("bids"), dict)
                else None,
            ]
            for levels in candidates:
                if levels:
                    return levels
            return None

        try:
            self._match_attempts += 1
            sports_markets = self.get_sports_markets()
            if not sports_markets:
                self.last_error = self.last_error or "No Kalshi sports markets available"
                logger.info(
                    "Kalshi get_game_market: %s vs %s -> %s",
                    home_team,
                    away_team,
                    result,
                )
                return result

            best_market = None
            best_score = 0.0

            # Normalize teams once
            norm_home = normalize_name(home_team)
            norm_away = normalize_name(away_team)

            game_date = None
            if game_time:
                try:
                    dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
                    game_date = dt.date()
                except Exception:
                    game_date = None

            for market in sports_markets:
                title = (market.get("title") or "")
                subtitle = market.get("subtitle") or ""
                ticker = (market.get("ticker") or "")
                market_text = normalize_market_text(title, subtitle, ticker)

                home_match = teams_match(norm_home, market_text)
                away_match = teams_match(norm_away, market_text)
                if not (home_match and away_match):
                    continue

                score = 1.0
                if market.get("series_ticker") and "GAME" in str(market.get("series_ticker")).upper():
                    score += 0.5

                close_time = market.get("close_time")
                if close_time and game_date:
                    try:
                        close_dt = datetime.fromtimestamp(float(close_time) / 1000.0, tz=pytz.UTC)
                        close_et = close_dt.astimezone(pytz.timezone("US/Eastern"))
                        if close_et.date() == game_date:
                            score += 0.4
                    except Exception:
                        pass

                if score > best_score:
                    best_score = score
                    best_market = market

            if not best_market:
                result["kalshi_match_debug"] = "no_market_match"
                logger.info(
                    "Kalshi get_game_market: %s vs %s -> %s",
                    home_team,
                    away_team,
                    result,
                )
                return result

            ticker = best_market.get("ticker")
            title = best_market.get("title")

            kalshi_prob = None
            used_key = None
            price_cents = None

            # 1) Use any implied probability provided directly in the market payload
            for prob_field in [
                "implied_result_prob",
                "implied_prob",
                "yes_bid_dollars",
                "yes_ask_dollars",
                "last_price",
                "last_trade_price",
                "yes_bid",
                "yes_ask",
            ]:
                if prob_field in best_market and best_market.get(prob_field) not in (None, ""):
                    prob_candidate = price_to_prob(best_market.get(prob_field))
                    if prob_candidate is not None:
                        kalshi_prob = prob_candidate
                        used_key = prob_field
                        break

            # 2) Otherwise, inspect the live orderbook for YES bids
            orderbook = {}
            if kalshi_prob is None and ticker:
                orderbook = self.get_orderbook(ticker) or {}
                levels = extract_yes_levels(orderbook)
                if levels:
                    level = levels[0] if isinstance(levels, list) and levels else None
                    if isinstance(level, dict):
                        price_cents = (
                            level.get("price")
                            or level.get("bid")
                            or level.get("px")
                        )
                        used_key = used_key or "orderbook_yes"
                    if price_cents is not None:
                        try:
                            kalshi_prob = float(price_cents) / 100.0
                            kalshi_prob = max(0.0, min(1.0, kalshi_prob))
                            logger.info(
                                "Kalshi YES price for %s: %s -> prob=%.3f",
                                ticker,
                                price_cents,
                                kalshi_prob,
                            )
                        except Exception:
                            kalshi_prob = None

            if kalshi_prob is None:
                logger.warning(
                    "Kalshi: no usable YES price for %s (orderbook/markets shape: %s)",
                    ticker,
                    orderbook,
                )
                result.update(
                    {
                        "kalshi_available": False,
                        "kalshi_prob": None,
                        "kalshi_home_prob": None,
                        "kalshi_away_prob": None,
                        "market_ticker": ticker,
                        "market_title": title,
                        "confidence": 0.0,
                        "kalshi_match_debug": "orderbook_missing_yes_price",
                    }
                )
                logger.info(
                    "Kalshi get_game_market: %s vs %s -> %s",
                    home_team,
                    away_team,
                    result,
                )
                return result

            result.update(
                {
                    "kalshi_available": True,
                    "kalshi_prob": kalshi_prob,
                    "kalshi_home_prob": kalshi_prob,
                    "kalshi_away_prob": 1.0 - kalshi_prob,
                    "market_ticker": ticker,
                    "market_title": title,
                    "kalshi_label": title or ticker,
                    "kalshi_volume": best_market.get("volume")
                    or best_market.get("yes_bid_size")
                    or best_market.get("no_bid_size")
                    or best_market.get("volume_yes"),
                    "confidence": min(1.0, 0.6 + best_score / 3.0)
                    if not best_market.get("synthetic")
                    else 0.5,
                    "kalshi_match_debug": f"matched_ticker={ticker} title={title} prob={kalshi_prob:.3f} used_price_key={used_key}",
                }
            )
            self._match_success += 1
            logger.info(
                "Kalshi get_game_market: %s vs %s -> %s",
                home_team,
                away_team,
                result,
            )
            return result

        except Exception as e:
            logger.warning(f"Error getting Kalshi game market: {e}")
            logger.info(
                "Kalshi get_game_market: %s vs %s -> %s",
                home_team,
                away_team,
                result,
            )
            return result
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name by removing mascots and common suffixes
        
        Examples:
            "Oklahoma Sooners" → "Oklahoma"
            "Los Angeles Lakers" → "Los Angeles"
            "Wake Forest Demon Deacons" → "Wake Forest"
            "St. John's Red Storm" → "St John's"
        """
        if not team_name:
            return ""
        
        # Common mascots to remove (NCAAB + NBA + NFL + NHL)
        mascots = [
            # NCAAB Common
            "Sooners", "Demon Deacons", "Tar Heels", "Wildcats", "Blue Devils",
            "Cavaliers", "Jayhawks", "Spartans", "Wolverines", "Buckeyes",
            "Hoosiers", "Terrapins", "Badgers", "Hawkeyes", "Illini",
            "Boilermakers", "Cornhuskers", "Scarlet Knights", "Golden Gophers",
            "Nittany Lions", "Fighting Irish", "Orange", "Cardinals", "Panthers",
            "Eagles", "Bulldogs", "Tigers", "Bears", "Aggies", "Rebels",
            "Commodores", "Volunteers", "Gamecocks", "Gators", "Crimson Tide",
            "War Eagles", "Razorbacks", "Red Storm", "Friars",
            "Musketeers", "Flyers", "Billikens", "Blue Jays", "Golden Eagles",
            "Pirates", "Hoyas", "Huskies", "Gaels", "Broncos", "Rams",
            "Aztecs", "Wolf Pack", "Aces", "Hilltoppers", "Colonels",
            # NBA
            "Lakers", "Clippers", "Warriors", "Kings", "Suns", "Mavericks",
            "Rockets", "Spurs", "Grizzlies", "Pelicans", "Thunder", "Trail Blazers",
            "Blazers", "Jazz", "Nuggets", "Timberwolves", "Wolves", "Bucks", "Bulls", "Pacers",
            "Pistons", "Celtics", "Nets", "Knicks", "76ers", "Sixers", "Raptors",
            "Heat", "Magic", "Hawks", "Hornets", "Wizards",
            # NFL
            "49ers", "Niners", "Seahawks", "Cardinals", "Rams", "Cowboys", "Giants",
            "Eagles", "Football Team", "Commanders", "Packers", "Vikings", "Lions",
            "Saints", "Falcons", "Buccaneers", "Bucs", "Chiefs", "Chargers",
            "Raiders", "Broncos", "Texans", "Colts", "Jaguars", "Jags", "Titans",
            "Ravens", "Steelers", "Browns", "Bengals", "Bills", "Dolphins",
            "Patriots", "Pats", "Jets",
            # NHL
            "Bruins", "Sabres", "Red Wings", "Panthers", "Canadiens", "Habs",
            "Senators", "Sens", "Lightning", "Bolts", "Maple Leafs", "Leafs",
            "Hurricanes", "Canes", "Blue Jackets", "Jackets",
            "Devils", "Islanders", "Isles", "Rangers", "Flyers", "Penguins", "Pens",
            "Capitals", "Caps", "Blackhawks", "Hawks", "Avalanche", "Avs", "Stars", "Wild",
            "Predators", "Preds", "Blues", "Jets", "Ducks", "Coyotes", "Flames",
            "Oilers", "Canucks", "Nucks", "Golden Knights", "Knights", "Kraken", "Sharks", "Kings"
        ]
        
        # Remove state abbreviations and clean
        team_clean = team_name.strip()
        team_clean = team_clean.replace(" St.", " State")  # Handle St. abbreviation
        team_clean = team_clean.replace(" St ", " State ")
        
        # Split and remove mascot if found
        words = team_clean.split()
        
        # Check if last word(s) is a mascot
        for i in range(len(words), 0, -1):
            potential_mascot = " ".join(words[i-1:])
            if potential_mascot in mascots:
                # Remove mascot, keep everything before it
                result = " ".join(words[:i-1])
                return result if result else team_clean
        
        # No mascot found, return as-is
        return team_clean
