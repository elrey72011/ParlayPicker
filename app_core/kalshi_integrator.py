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


def _debug_log(message: str, *args: Any) -> None:
    if DEBUG_KALSHI_MATCHING:
        logger.info(message, *args)


class KalshiMatchResult(TypedDict, total=False):
    matched: bool
    kalshi_available: bool
    label: str
    probability: Optional[float]
    raw_event_id: Optional[str]
    league: str
    reason: str
    market_type: Optional[str]
    direction: Optional[str]
    game_date: Optional[datetime]


SUPPORTED_LEAGUES = {"nba", "nfl", "mlb", "ncaaf", "ncaab", "nhl"}
LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

FUTURE_EXCLUDE_KEYWORDS = {
    "champions league",
    "ucl",
    "win the league",
    "to win league",
    "to win the league",
    "to win championship",
    "championship",
    "playoffs",
    "division winner",
    "relegation",
    "bottom of table",
    "top of table",
    "season wins",
    "regular season wins",
    "season",
    "wins",
    "champion",
    "exactly",
}

TEAM_FUZZY_THRESHOLD = 2.0
MAX_LINE_DIFF = 3.0
DEBUG_KALSHI_MATCHING = False

# Fuzzy threshold for fallback team name matching
TEAM_NAME_SIMILARITY = 0.80

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
    """Parse Kalshi timestamps (ms or iso) into a timezone-aware datetime.

    Returns None when parsing fails so callers can skip date-based filtering gracefully.
    """

    if raw is None or raw == "":
        return None

    try:
        # Kalshi close_time may be milliseconds since epoch
        if isinstance(raw, (int, float)):
            # Some responses are already in seconds; treat values above year 2050 as ms
            value = float(raw)
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=pytz.UTC)
        # ISO timestamp string
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC)
    except Exception:
        return None


def _build_team_codes(team_name: str) -> List[str]:
    """Generate plausible team codes from a full team name.

    This makes it possible to align Kalshi's compact tickers (e.g., "PHI")
    with ESPN/OddsAPI names such as "Philadelphia 76ers".
    """

    if not team_name:
        return []

    words = [w for w in re.split(r"\s+", team_name) if w]
    codes: List[str] = []
    if words:
        codes.append(words[0][:3].upper())
        codes.append("".join(w[0] for w in words[:3]).upper())
    if len(words) > 1:
        codes.append(words[-1][:3].upper())
    return list({c for c in codes if len(c) >= 2})


def _extract_date_from_ticker(ticker: str) -> Optional[datetime]:
    """Attempt to pull a date from the Kalshi ticker text."""

    if not ticker:
        return None

    # Examples: 20231104, 2023-11-04, 231104
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=pytz.UTC)
        except Exception:
            pass

    match = re.search(r"(\d{4}-\d{2}-\d{2})", ticker)
    if match:
        try:
            return datetime.fromisoformat(match.group(1)).replace(tzinfo=pytz.UTC)
        except Exception:
            pass

    match = re.search(r"(\d{2}[01]\d[0-3]\d)", ticker)
    if match:
        try:
            return datetime.strptime(match.group(1), "%y%m%d").replace(tzinfo=pytz.UTC)
        except Exception:
            pass

    return None


def _extract_market_type(title: str, ticker: str) -> Optional[str]:
    """Determine market type (ML, Spread, Total) from ticker/title."""

    text = ((title or "") + " " + (ticker or "")).upper()
    if "MONEYLINE" in text or re.search(r"\bML\b", text):
        return "ML"
    if "SPREAD" in text or re.search(r"\bSPR?\b", text):
        return "Spread"
    if "TOTAL" in text or "OVER" in text or "UNDER" in text or re.search(r"\bOU\b", text):
        return "Total"
    return None


def _extract_teams_from_ticker(ticker: str) -> List[str]:
    """Extract potential team codes from a ticker string."""

    if not ticker:
        return []

    tokens = re.findall(r"[A-Z]{2,4}", ticker)
    ignore = {"ML", "OU", "OVER", "UNDER", "SP", "SPD", "TOT", "GAME", "VS", "AT"}
    return [t for t in tokens if t not in ignore]


# Alias used in matching snippets
def _parse_kalshi_date(raw) -> Optional[datetime]:
    return _parse_market_date(raw)


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

    def _parse_market_metadata(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = market.get("title") or ""
        subtitle = market.get("subtitle") or ""
        ticker = market.get("ticker") or ""

        text = f"{title} {subtitle} {ticker}".lower()
        if any(key in text for key in FUTURE_EXCLUDE_KEYWORDS):
            return None

        # Do not filter out markets just because they have strikes.
        # Spreads and totals always have strikes and are valid game markets.

        ticker_teams = _extract_teams_from_ticker(ticker)
        if len(ticker_teams) < 2:
            return None

        market_type = _extract_market_type(title, ticker)
        if market_type is None:
            return None

        date_from_ticker = _extract_date_from_ticker(ticker)
        parsed_date = _parse_market_date(market.get("event_date") or market.get("close_time"))
        market_date = parsed_date or date_from_ticker
        if market_date is None:
            return None

        probability = None
        for key in ["last_price_dollars", "last_price", "yes_bid_dollars", "yes_ask_dollars", "last_trade_price", "yes_bid", "yes_ask"]:
            probability = price_to_prob(market.get(key))
            if probability is not None:
                break

        teams_norm = [TeamNameMatcher.normalize(t) for t in ticker_teams]

        return {
            "ticker": ticker,
            "title": title,
            "teams": ticker_teams,
            "teams_norm": teams_norm,
            "market_type": market_type,
            "market_date": market_date,
            "probability": probability,
        }

    league_norm = normalize_name(league)
    _debug_log("[KALSHI MATCH DEBUG] normalized league=%s", league_norm)
    if league_norm and league_norm not in SUPPORTED_LEAGUES:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
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
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="api_error:no_integrator",
        )

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
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason=f"api_error:{short_err}",
        )

    parsed_markets: List[Dict[str, Any]] = []
    for mkt in markets or []:
        parsed = _parse_market_metadata(mkt)
        if parsed:
            parsed_markets.append({"__meta": parsed, **mkt})

    if not parsed_markets:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_valid_game_markets",
        )

    home_norm = TeamNameMatcher.normalize(home_team)
    away_norm = TeamNameMatcher.normalize(away_team)
    home_codes = _build_team_codes(home_team)
    away_codes = _build_team_codes(away_team)

    best_market: Optional[Dict[str, Any]] = None
    best_score: float = 0.0

    for market in parsed_markets:
        meta = market["__meta"]

        if league_norm and league_norm in LEAGUE_SERIES_MAP:
            expected = LEAGUE_SERIES_MAP[league_norm]
            series = str(market.get("series_ticker") or "").upper()
            if expected and series and not series.startswith(expected):
                continue

        if game_dt and meta.get("market_date"):
            day_diff = abs((meta["market_date"].date() - game_dt.date()).days)
            if day_diff > 3:
                continue

        teams = meta.get("teams", [])
        if len(teams) < 2:
            continue

        def _team_score(team_code: str, target_norm: str, target_codes: List[str]) -> float:
            if team_code in target_codes:
                return 2.0
            if TeamNameMatcher.normalize(team_code) == target_norm:
                return 1.5
            if TeamNameMatcher.similarity_score(
                TeamNameMatcher.normalize(team_code), target_norm
            ) >= TEAM_NAME_SIMILARITY:
                return 1.0
            return 0.0

        score_home_first = _team_score(teams[0], home_norm, home_codes) + _team_score(
            teams[1], away_norm, away_codes
        )
        score_away_first = _team_score(teams[0], away_norm, away_codes) + _team_score(
            teams[1], home_norm, home_codes
        )

        score = max(score_home_first, score_away_first)

        if score_home_first < 1.0 and score_away_first < 1.0:
            continue

        if meta.get("probability") is not None:
            score += 0.25

        if score > best_score:
            best_score = score
            best_market = market

    if not best_market or best_score < TEAM_FUZZY_THRESHOLD:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label="",
            probability=None,
            raw_event_id=None,
            league=league_norm,
            reason="no_market_match",
        )

    meta = best_market["__meta"]
    probability = meta.get("probability")
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
                probability = price_to_prob(price_val)
        except Exception:
            probability = None

    if probability is None:
        return KalshiMatchResult(
            matched=False,
            kalshi_available=False,
            label=meta.get("title") or meta.get("ticker"),
            probability=None,
            raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
            league=league_norm,
            reason="no_price",
        )

    direction = "YES" if probability >= 0.5 else "NO"

    result = KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=meta.get("title") or meta.get("ticker"),
        probability=probability,
        raw_event_id=str(best_market.get("ticker") or best_market.get("id")),
        league=league_norm,
        reason="ok",
        market_type=meta.get("market_type"),
        direction=direction,
        game_date=meta.get("market_date"),
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

    match = get_match_for_game(
        league=None,
        home_team=home_team,
        away_team=away_team,
        game_date=game_date,
        integrator=integrator,
        status=status,
    )

    if not match.get("matched"):
        return {
            "kalshi_label": match.get("label"),
            "kalshi_probability": None,
            "kalshi_volume": None,
            "kalshi_match_debug": match.get("reason", "no_market_match"),
            "kalshi_event_ticker": match.get("raw_event_id"),
        }

    return {
        "kalshi_label": match.get("label"),
        "kalshi_probability": match.get("probability"),
        "kalshi_volume": None,
        "kalshi_match_debug": match.get("reason", "ok"),
        "kalshi_event_ticker": match.get("raw_event_id"),
        "kalshi_market_type": match.get("market_type"),
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
        print("=" * 80)
        print("🚀 KALSHI INTEGRATOR v2.0 - SECRETS VERSION - INITIALIZING")
        print("=" * 80)

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
            print("⚠️ KALSHI: Secrets not available, using provided values")
            self.api_key = api_key
            self.api_secret = api_secret
            print(
                f"🔑 KALSHI: Fallback - API key: {bool(self.api_key)}, secret: {bool(self.api_secret)}"
            )

        # Kalshi API URLs
        # If your account is on the main trading host, you can change base_url to:
        # "https://api.kalshi.com/trade-api/v2"
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
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

                key_data = self.api_secret.strip()
                if not key_data.startswith("-----BEGIN"):
                    key_data = (
                        "-----BEGIN RSA PRIVATE KEY-----\n"
                        + key_data
                        + "\n-----END RSA PRIVATE KEY-----"
                    )

                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(),
                    password=None,
                    backend=default_backend(),
                )
                self._auth_ready = True
                logger.info(
                    "✅ Kalshi RSA key loaded successfully (key: %s...)",
                    self.api_key[:8],
                )
            except ImportError:
                logger.warning(
                    "cryptography library not installed - Kalshi auth disabled"
                )
                self._private_key = None
            except Exception as e:
                logger.warning(f"Could not load Kalshi RSA key: {e}")
                self._private_key = None

        # Synthetic fallback cache flags
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, Any]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}

        self.last_error: Optional[str] = None

        # Cache for API responses
        self._markets_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_duration = 300  # 5 minutes

        self._match_attempts = 0
        self._match_success = 0

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Create RSA signature for Kalshi API request."""
        if not self._private_key:
            return ""

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64

            # Message format: timestamp + method + path
            message = f"{timestamp}{method}{path}"

            signature = self._private_key.sign(
                message.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )

            return base64.b64encode(signature).decode("utf-8")
        except Exception as e:
            logger.warning(f"Error signing Kalshi request: {e}")
            return ""

    def _make_authenticated_request(
        self, method: str, endpoint: str, params: dict = None
    ) -> Optional[dict]:
        """Make authenticated request to Kalshi API.

        Per Kalshi docs, we sign ONLY the path (without query params),
        but send the full URL with query params in the actual request.
        """
        import time as time_module

        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))

        print(f"🌐 KALSHI: Request {method} {url}")
        print(f"🌐 KALSHI: Params: {params}")
        logger.info("Kalshi API request: %s %s with params: %s", method, url, params)

        headers = self.headers.copy()

        if self._auth_ready and self._private_key:
            # Strip query parameters from endpoint before signing
            path_suffix = endpoint.split("?", 1)[0]
            path_to_sign = f"/trade-api/v2{path_suffix}"

            signature = self._sign_request(method.upper(), path_to_sign, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

            print(f"🔐 KALSHI: Auth configured (key: {self.api_key[:8]}...)")
            logger.debug("Signing: %s%s%s", timestamp, method.upper(), path_to_sign)
            logger.info("Using API URL: %s", self.api_url)
        else:
            print("⚠️ KALSHI: No authentication configured!")

        try:
            if method.upper() == "GET":
                response = requests.get(
                    url, headers=headers, params=params, timeout=15
                )
            else:
                response = requests.post(
                    url, headers=headers, json=params, timeout=15
                )

            print(f"📥 KALSHI: Response Status {response.status_code}")
            logger.info(
                "Kalshi API response: Status %s, URL: %s",
                response.status_code,
                response.url,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✅ KALSHI: Response keys: {list(data.keys())}")
                    logger.info("Response data keys: %s", list(data.keys()))
                    if "markets" in data:
                        print(
                            f"✅ KALSHI: Markets in response: {len(data.get('markets', []))}"
                        )
                        logger.info(
                            "Markets in response: %d",
                            len(data.get("markets", [])),
                        )
                    if "series" in data:
                        print(
                            f"✅ KALSHI: Series in response: {len(data.get('series', []))}"
                        )
                    self.last_error = None
                    return data
                except Exception as json_error:
                    print(f"❌ KALSHI: Failed to parse JSON: {json_error}")
                    logger.error("Failed to parse JSON response: %s", json_error)
                    logger.error("Response text: %s", response.text[:500])
                    self.last_error = f"JSON parse error: {json_error}"
                    return None

            elif response.status_code == 401:
                print("❌ KALSHI: Authentication failed (401)")
                logger.warning(
                    "Kalshi API authentication failed - Response: %s",
                    response.text[:200],
                )
                self.last_error = "Authentication failed"

            elif response.status_code == 403:
                print("❌ KALSHI: Access forbidden (403)")
                logger.warning(
                    "Kalshi API access forbidden - Response: %s",
                    response.text[:200],
                )
                self.last_error = "Access forbidden"

            else:
                print(f"❌ KALSHI: API error {response.status_code}")
                logger.warning(
                    "Kalshi API error: %s - %s",
                    response.status_code,
                    response.text[:200],
                )
                self.last_error = f"API error: {response.status_code}"

        except requests.exceptions.ConnectionError as e:
            print(
                f"❌ KALSHI: Connection error (network blocked?): {str(e)[:100]}"
            )
            logger.error(
                "Kalshi API connection error (network may be blocked): %s", e
            )
            self.last_error = "Connection blocked - check network settings"

        except requests.exceptions.Timeout:
            print("❌ KALSHI: Request timeout")
            logger.warning("Kalshi API timeout")
            self.last_error = "Request timeout"

        except Exception as e:
            print(f"❌ KALSHI: Request failed: {str(e)[:100]}")
            logger.error("Kalshi API request failed: %s", e)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())
            self.last_error = str(e)

        return None

    def is_configured(self) -> bool:
        """Check if Kalshi is properly configured."""
        return bool(self.api_key and self._auth_ready)

    def get_sports_series(self) -> List[Dict]:
        """Get all available sports series tickers from Kalshi."""
        try:
            endpoint = "/series"
            params = {"limit": 200}

            print("🔍 KALSHI: Fetching series list...")
            logger.info("Fetching Kalshi series list...")
            response_data = self._make_authenticated_request(
                "GET", endpoint, params=params
            )

            if response_data:
                all_series = response_data.get("series", [])
                print(f"🔍 KALSHI: Found {len(all_series)} total series")
                logger.info("Found %d total series", len(all_series))

                sports_keywords = [
                    "NFL",
                    "NBA",
                    "MLB",
                    "NHL",
                    "UFC",
                    "SOCCER",
                    "TENNIS",
                    "GOLF",
                    "FOOTBALL",
                    "BASKETBALL",
                    "BASEBALL",
                    "HOCKEY",
                    "SPORT",
                    "GAME",
                ]

                sports_series = []
                for series in all_series:
                    ticker = series.get("ticker", "").upper()
                    title = series.get("title", "").upper()
                    category = series.get("category", "").upper()

                    if any(
                        kw in ticker or kw in title or kw in category
                        for kw in sports_keywords
                    ):
                        sports_series.append(series)
                        print(
                            f"🏈 KALSHI: Found sports series: {series.get('ticker')} - {series.get('title')}"
                        )
                        logger.info(
                            "Found sports series: %s - %s",
                            series.get("ticker"),
                            series.get("title"),
                        )

                print(f"✅ KALSHI: Total {len(sports_series)} sports series found")
                return sports_series

            print("❌ KALSHI: No response from /series endpoint")
            logger.error("No response from /series endpoint")
            return []

        except Exception as e:
            print(f"❌ KALSHI: Error fetching sports series: {e}")
            logger.error("Error fetching sports series: %s", e)
            import traceback

            print(f"❌ KALSHI: Traceback: {traceback.format_exc()}")
            return []

    def get_markets(
        self, category: str = "sports", status: Optional[str] = "open"
    ) -> List[Dict]:
        """Fetch available Kalshi markets."""
        # We *don't* hard-return synthetic here; we still try live API.
        cache_key = status or "all"
        now = time.time()
        if (
            cache_key in self._markets_cache
            and cache_key in self._cache_time
            and now - self._cache_time[cache_key] < self._cache_duration
        ):
            logger.info("Kalshi get_markets cache hit for status=%s", cache_key)
            return copy.deepcopy(self._markets_cache.get(cache_key, []))

        all_markets: List[Dict[str, Any]] = []

        try:
            sports_series = self.get_sports_series()

            if sports_series:
                logger.info(
                    "Found %d sports series from /series", len(sports_series)
                )

                # Pull markets from first few sports series to avoid rate limits
                for series in sports_series[:10]:
                    series_ticker = series.get("ticker")
                    if not series_ticker:
                        continue

                    logger.info("Fetching markets for series: %s", series_ticker)
                    endpoint = "/markets"
                    params = {
                        "series_ticker": series_ticker,
                        "limit": 200,
                    }
                    if status:
                        params["status"] = status

                    response_data = self._make_authenticated_request(
                        "GET", endpoint, params=params
                    )
                    if response_data:
                        markets = response_data.get("markets", [])
                        logger.info(
                            "Got %d markets from %s",
                            len(markets),
                            series_ticker,
                        )
                        all_markets.extend(markets)
                        if len(all_markets) >= 200:
                            break
            else:
                # Fallback: call /markets directly
                logger.warning(
                    "No sports series found from /series; falling back to /markets"
                )
                endpoint = "/markets"
                params = {"limit": 200}
                if status:
                    params["status"] = status
                response_data = self._make_authenticated_request(
                    "GET", endpoint, params=params
                )
                if response_data:
                    fallback_markets = response_data.get("markets", [])
                    logger.info(
                        "Fallback /markets returned %d markets",
                        len(fallback_markets),
                    )
                    all_markets.extend(fallback_markets)

            if all_markets:
                self.last_error = None
                logger.info(
                    "✅ Loaded %d total Kalshi markets", len(all_markets)
                )
                sample_tickers = [
                    m.get("ticker", "NO_TICKER") for m in all_markets[:5]
                ]
                logger.info("Sample tickers: %s", sample_tickers)
                self._markets_cache[cache_key] = all_markets
                self._cache_time[cache_key] = now
                return all_markets

            self.last_error = "Kalshi API returned no markets"
            logger.warning(
                "No markets found (sports_series=%d)",
                len(sports_series or []),
            )

        except Exception as e:
            self.last_error = str(e)
            logger.error("Error fetching Kalshi markets: %s", e)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

        return []

    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets."""
        try:
            all_markets = self.get_markets()
            all_markets = all_markets or []
            logger.info(
                "Kalshi get_sports_markets → %d markets", len(all_markets)
            )
            if all_markets[:3]:
                logger.info("Kalshi sample markets: %s", all_markets[:3])
        except Exception as e:
            logger.warning("Kalshi get_sports_markets error: %s", e)
            self.last_error = str(e)
            return []

        sports_keywords = [
            "NFL",
            "NBA",
            "MLB",
            "NHL",
            "UFC",
            "SOCCER",
            "TENNIS",
            "GOLF",
            "FOOTBALL",
            "BASKETBALL",
            "BASEBALL",
            "HOCKEY",
        ]

        sports_markets = []
        for market in all_markets:
            title = market.get("title", "").upper()
            ticker = market.get("ticker", "").upper()
            if any(kw in title or kw in ticker for kw in sports_keywords):
                sports_markets.append(market)

        return sports_markets

    def _get_today_datetime_range(self, tz_name: str = "America/New_York"):
        """Get datetime range for *today* in the given timezone, returned in UTC."""
        from datetime import datetime, time as dtime, timezone
        import pytz
    
        # Localize to the given timezone (your app is Eastern)
        tz = pytz.timezone(tz_name)
        now_local = datetime.now(tz)
    
        start_local = tz.localize(
            datetime.combine(now_local.date(), dtime.min)
        )
        end_local = tz.localize(
            datetime.combine(now_local.date(), dtime.max)
        )
    
        # Return as UTC datetimes
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

    def get_game_markets_for_events(self, league: str = "NBA") -> List[Dict[str, Any]]:
        """Get game-related markets for a specific league (NBA, NFL, MLB, NHL, etc.)."""
        try:
            print(f"🏀 KALSHI: Fetching {league} game markets...")

            endpoint = "/series"
            params = {"limit": 1000}
            response_data = self._make_authenticated_request(
                "GET", endpoint, params=params
            )

            if not response_data:
                print("❌ KALSHI: Failed to get series list")
                return []

            all_series = response_data.get("series", [])
            print(f"🔍 KALSHI: Found {len(all_series)} total series")

            league_prefix = f"KX{league.upper()}"
            game_suffixes = [
                "GAME",
                "GAMES",
                "SPREAD",
                "TOTAL",
                "ANYTD",
                "PASSYDS",
                "RUSHYDS",
            ]

            relevant_series = []
            for series in all_series:
                ticker = series.get("ticker", "").upper()
                if ticker.startswith(league_prefix) and any(
                    suffix in ticker for suffix in game_suffixes
                ):
                    relevant_series.append(series)
                    print(f"  ✅ {ticker}")

            print(f"🎯 KALSHI: Found {len(relevant_series)} {league} game series")

            all_markets: List[Dict[str, Any]] = []
            for series in relevant_series:
                series_ticker = series.get("ticker")
                if not series_ticker:
                    continue

                endpoint = "/markets"
                params = {
                    "series_ticker": series_ticker,
                    "limit": 1000,
                    "status": "open",
                }

                response_data = self._make_authenticated_request(
                    "GET", endpoint, params=params
                )
                if response_data:
                    markets = response_data.get("markets", [])
                    print(f"  📊 {series_ticker}: {len(markets)} markets")
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
            logger.error("Error fetching game markets for %s: %s", league, e)
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return []

    def filter_markets_closing_today(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter markets to only those closing today (America/New_York),
        but fall back to all markets if none close today.
        """
        from datetime import datetime, timezone
    
        if not markets:
            return []
    
        # Get today range in UTC based on Eastern "today"
        min_dt, max_dt = self._get_today_datetime_range("America/New_York")
    
        filtered: List[Dict[str, Any]] = []
    
        for m in markets:
            close_raw = m.get("close_time")
            if close_raw is None:
                continue
    
            close_dt = None
    
            # Case 1: ISO 8601 string like "2025-12-25T01:00:00Z"
            if isinstance(close_raw, str):
                try:
                    iso_str = close_raw.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(iso_str)
                    if close_dt.tzinfo is None:
                        close_dt = close_dt.replace(tzinfo=timezone.utc)
                    else:
                        close_dt = close_dt.astimezone(timezone.utc)
                except Exception:
                    continue
    
            # Case 2: numeric timestamp (sec or ms)
            elif isinstance(close_raw, (int, float)):
                try:
                    ts = float(close_raw)
                    if ts > 10**11:  # assume ms
                        ts /= 1000.0
                    close_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    continue
    
            if close_dt is None:
                continue
    
            if min_dt <= close_dt <= max_dt:
                filtered.append(m)
    
        # 🔁 Fallback: if no markets close today, use all markets
        if not filtered:
            logger.info(
                "No Kalshi markets close today; falling back to all markets (%d).",
                len(markets),
            )
            return markets
    
        return filtered

    def group_game_markets_by_event(
        self, markets: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group markets by event_ticker (one event = one game)."""
        from collections import defaultdict

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m in markets:
            event_ticker = m.get("event_ticker")
            if event_ticker:
                grouped[event_ticker].append(m)
        return dict(grouped)

    @staticmethod
    def price_to_prob(price_dollars: Optional[float]) -> Optional[float]:
        """Convert Kalshi price (0–1 dollars) to implied probability."""
        if price_dollars is None:
            return None
        return float(price_dollars)

    # get_game_market and _normalize_team_name methods are the same
    # as in your current file – you can keep them as-is unless you
    # want me to rewrite those too.
    # (If you’d like, I can also clean those up in a follow-up.)

if __name__ == "__main__":
    from datetime import datetime, timezone

    integrator = KalshiIntegrator()
    nba_markets = integrator.get_game_markets_for_events("NBA")
    print("NBA markets returned (all):", len(nba_markets))

    for m in nba_markets[:5]:
        ct = m.get("close_time")
        print("NBA sample close_time raw:", ct)
        try:
            if isinstance(ct, str):
                iso_str = ct.replace("Z", "+00:00")
                dt_utc = datetime.fromisoformat(iso_str)
            elif isinstance(ct, (int, float)):
                ts = float(ct)
                if ts > 10**11:
                    ts /= 1000.0
                dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt_utc = None
            print(" -> parsed UTC:", dt_utc)
        except Exception as e:
            print(" -> parse error:", e)
