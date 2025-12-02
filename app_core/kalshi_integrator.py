import textwrap # <-- Make sure this is at the top of your file
import os
import copy
import time
import logging
import requests
from typing import Dict, List, Any, Optional
# --- FIX: Removed redundant calendar import (not used) ---
import pytz
from datetime import datetime, time, timezone
import json  # ✅ New: for pretty-printing market JSON

# --------------------------------------------
# GLOBAL LOGGER (required by KalshiIntegrator)
# --------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis

    Kalshi uses RSA signature authentication for API requests.
    The API key is a UUID and the secret is an RSA private key.
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        # 1. Ensure keys are read correctly from environment or passed in
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")

        # Kalshi API URLs - FIX: This logic is correct for PROD vs DEMO
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"  # PRODUCTION
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"

        # Use production API if we have credentials
        self.api_url = self.base_url if self.api_key else self.demo_url

        # 2. Add diagnostic print to confirm keys were read
        if self.api_key:
            print(f"✅ KALSHI: API Key read successfully (starts with: {self.api_key[:8]})")
        else:
            print("❌ KALSHI: API Key not found. Defaulting to Demo URL.")

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
                # Ensure correct PEM format for loading
                if not key_data.startswith("-----BEGIN"):
                    # The secret should be the RSA private key string (base64 body only),
                    # so we wrap it in proper PEM headers if needed.
                    key_data = self.api_secret.strip()

                    # No need to wrap; it's already full PEM
                    self._private_key = serialization.load_pem_private_key(
                        key_data.encode(),
                        password=None,
                        backend=default_backend()
                    )

                self._private_key = serialization.load_pem_private_key(
                    key_data.encode(),
                    password=None,
                    backend=default_backend()
                )
                self._auth_ready = True
                logger.info(f"✅ Kalshi RSA key loaded successfully (key: {self.api_key[:8]}...)")
            except ImportError:
                # This happens if 'cryptography' is not installed
                logger.warning("cryptography library not installed - Kalshi auth disabled")
                self._private_key = None
            except Exception as e:
                # This is the most common failure point: key format error
                print(f"❌ KALSHI: RSA Key Load FAILED: {e}. Check your KALSHI_API_SECRET format.")
                logger.warning(f"Could not load Kalshi RSA key: {e}")
                self._private_key = None

        # Synthetic fallback cache when Kalshi API is unavailable
        # ... (rest of the __init__ remains the same) ...

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
        """Make authenticated request to Kalshi API"""
        import time as time_module

        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(time_module.time() * 1000))

        print(f"🌐 KALSHI: Request {method} {url}")
        print(f"🌐 KALSHI: Params: {params}")
        logger.info(f"Kalshi API request: {method} {url} with params: {params}")

        headers = self.headers.copy()

        if self._auth_ready and self._private_key:
            # CRITICAL: Strip query parameters from endpoint before signing
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

    # --- NEW HELPER METHOD ---
    def _get_today_timestamp_range(self) -> tuple[int, int]:
        """
        Override today's date so it's based on real UTC time,
        not your system's incorrect date.
        """
        # Correct, timezone-aware real UTC date
        real_today = datetime.now(timezone.utc).date()

        # Build start + end timestamps for the actual real UTC day
        min_ts_dt = datetime.combine(real_today, time.min, tzinfo=timezone.utc)
        max_ts_dt = datetime.combine(real_today, time.max, tzinfo=timezone.utc)

        min_ts = int(min_ts_dt.timestamp() * 1000)
        max_ts = int(max_ts_dt.timestamp() * 1000)

        return min_ts, max_ts

    # --------------------------

    # ---------- NEW: Retrieve Sports Events (Games) ----------
    def get_game_markets_for_events(self, league: str) -> List[Dict[str, Any]]:
        """
        Pull ML / Spread / Total style markets for a given league by querying
        /markets directly for the game-related series (GAME / SPREAD / TOTAL).

        This aligns with what you see in the Kalshi UI:
        - Moneyline / Game winner
        - Spread
        - Total (Over/Under)
        """
        league = league.upper()
        prefix = f"KX{league}"

        # We care about these “game” series types
        GAME_SERIES_SUFFIXES = ("GAME", "GAMES", "SPREAD", "TOTAL")

        logger.info(f"Pulling game markets by series for league {league}")

        # 1) Get all series and filter to the league + game-related ones
        series_data = self._make_authenticated_request(
            "GET",
            "/series",
            params={"limit": 1000},
        )

        all_series = series_data.get("series", [])
        game_series: List[str] = []

        for s in all_series:
            ticker = s.get("ticker", "")
            if not ticker.startswith(prefix):
                continue

            # Remove league prefix and look at the rest
            suffix = ticker[len(prefix):]
            if any(key in suffix for key in GAME_SERIES_SUFFIXES):
                game_series.append(ticker)

        logger.info(
            "Game-related series for league %s: %s",
            league,
            ", ".join(game_series) if game_series else "NONE",
        )

        # 2) For each game-related series, pull all open markets
        all_markets: List[Dict[str, Any]] = []

        for s_ticker in game_series:
            cursor = None
            while True:
                params = {
                    "limit": 1000,
                    "series_ticker": s_ticker,
                    "status": "open",
                }
                if cursor:
                    params["cursor"] = cursor

                logger.info(
                    f"Kalshi API request: GET /markets for series {s_ticker} with params: {params}"
                )
                data = self._make_authenticated_request(
                    "GET",
                    "/markets",
                    params=params,
                )

                markets = data.get("markets", [])
                logger.info(
                    "Series %s: pulled %d markets (cursor=%s)",
                    s_ticker,
                    len(markets),
                    data.get("cursor"),
                )

                for m in markets:
                    enriched = {
                        "league": league,
                        "series_ticker": s_ticker,
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

                cursor = data.get("cursor")
                if not cursor:
                    break

        logger.info(
            "Total game markets pulled for league %s: %d",
            league,
            len(all_markets),
        )
        return all_markets

    # ---------- NEW: Retrieve Game Markets for a Specific Event ----------
    def get_event_markets(self, event_ticker: str):
        """
        Fetch all markets (ML / Spread / Total, etc.) for a given event_ticker.
        These are the real game markets shown in the Kalshi UI.
        """
        if not event_ticker:
            return []

        params = {
            "event_ticker": event_ticker,
            "limit": 200,
            "status": "open",
        }

        data = self._make_authenticated_request("GET", "/markets", params=params)

        if data and "markets" in data:
            return data["markets"]

        return []

    def is_configured(self) -> bool:
        """Check if Kalshi is properly configured"""
        return bool(self.api_key and self._auth_ready)

    def get_sports_series(self) -> List[Dict]:
        """Get only the specified NFL, NBA, MLB, NCAAF, NCAAB, NHL series tickers."""
        try:
            endpoint = "/series"
            params = {"limit": 200}

            # --- AUTHENTICATED REQUEST --- (This will use the Production URL if authentication succeeds)
            response_data = self._make_authenticated_request("GET", endpoint, params=params)

            if response_data:
                all_series = response_data.get("series", [])

                # --- UPDATED: STRICT FILTER FOR SPECIFIC LEAGUES ONLY ---
                # These are the prefixes for the core series tickers Kalshi uses.
                strict_ticker_starts = ('KXNFL', 'KXNBA', 'KXMLB', 'KXNCAAF', 'KXNCAAB', 'KXNHL', 'NCAAF', 'NCAAB')

                sports_series = []
                for series in all_series:
                    ticker = series.get('ticker', '').upper()

                    # Check if the Ticker starts with one of our desired league prefixes
                    if ticker.startswith(strict_ticker_starts):
                        sports_series.append(series)

                return sports_series

            return []

        except Exception as e:
            # Error handling remains the same
            import traceback
            logger.error(f"Error fetching sports series: {e}\n{traceback.format_exc()}")
            return []

        except Exception as e:
            print(f"❌ KALSHI: Error fetching sports series: {e}")
            logger.error(f"Error fetching sports series: {e}")
            import traceback
            print(f"❌ KALSHI: Traceback: {traceback.format_exc()}")
            return []

    def get_markets(self, category: str = "sports", status: str = "open", closing_today: bool = False) -> List[Dict]:
        """Fetch available Kalshi markets."""
        if getattr(self, "_using_synthetic_data", False):
            return []

        try:
            # First, get sports series tickers
            sports_series = self.get_sports_series()

            if not sports_series:
                logger.warning("No sports series found")
                self.last_error = "No sports series available"
                self._using_synthetic_data = True
                return []

            logger.info(f"Found {len(sports_series)} sports series")

            # --- IMPLEMENT CLOSING TODAY FILTER ---
            min_close_ts, max_close_ts = None, None
            if closing_today:
                min_close_ts, max_close_ts = self._get_today_timestamp_range()
                print(f"🗓️ Filtering markets closing today (UTC): {min_close_ts} to {max_close_ts}")
                logger.info(f"Filtering markets closing today (UTC): {min_close_ts} to {max_close_ts}")
            # --------------------------------------

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
                    "status": status
                }

                # --- ADD TIMESTAMP FILTERS TO PARAMS ---
                if closing_today:
                    params["min_close_ts"] = min_close_ts
                    params["max_close_ts"] = max_close_ts
                # ---------------------------------------

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

    def get_sports_events(self, league: str = "NBA") -> List[Dict[str, Any]]:
        """
        Fetch open Kalshi events for a specific sports league.

        We infer the league from the series ticker prefix:
          NBA  -> KXNBA
          NFL  -> KXNFL
          NHL  -> KXNHL
          NCAAF -> KXNCAAFB
          NCAAB -> KXNCAABB
        Then, for the league we:
          1) Pull all series
          2) Filter series by league prefix AND 'game-ish' series
          3) For each series, pull its open events
        """

        league_prefix_map = {
            "NBA": "KXNBA",
            "NFL": "KXNFL",
            "NHL": "KXNHL",
            "NCAAF": "KXNCAAFB",
            "NCAAB": "KXNCAABB",
        }

        prefix = league_prefix_map.get(league.upper())
        if not prefix:
            logger.warning(f"Unknown league '{league}', returning no events.")
            return []

        # 1. Get all series
        series_data = self._make_authenticated_request(
            "GET",
            "/series",
            params={"limit": 500}
        )
        if not series_data:
            logger.warning("No series data returned from Kalshi.")
            return []

        series_list = series_data.get("series", [])
        logger.info(f"Kalshi returned {len(series_list)} total series.")

        # 2. Filter series for this league and for game-ish series (GAME / SPREAD / TOTAL)
        GAME_KEYWORDS = ("GAME", "GAMES", "SPREAD", "TOTAL", "MONEYLINE")

        league_series: List[Dict[str, Any]] = []
        for s in series_list:
            ticker = (s.get("ticker") or "").upper()
            if not ticker.startswith(prefix):
                continue
            if not any(kw in ticker for kw in GAME_KEYWORDS):
                # Skip long-term futures like WIN TOTAL, MVP, etc
                continue
            league_series.append(s)

        logger.info(
            f"Found {len(league_series)} game-related series for league {league} "
            f"with prefix {prefix}"
        )

        events: List[Dict[str, Any]] = []

        # 3. For each series, pull its open events
        for s in league_series:
            series_ticker = s.get("ticker")
            if not series_ticker:
                continue

            ev_params = {
                "limit": 200,
                "series_ticker": series_ticker,
                "status": "open",
            }
            logger.info(
                f"Kalshi API request: GET /events with params: {ev_params}"
            )
            ev_data = self._make_authenticated_request(
                "GET",
                "/events",
                params=ev_params,
            )
            if not ev_data:
                continue

            ev_list = ev_data.get("events", [])
            logger.info(f"Series {series_ticker}: {len(ev_list)} events")
            events.extend(ev_list)

        logger.info(
            f"Total sports events found for league {league}: {len(events)}"
        )
        return events


    def get_sports_markets(self, closing_today: bool = False) -> List[Dict]:
        """Get all active sports betting markets, optionally filtered to close today."""
        # --- PASS THE NEW FLAG TO GET_MARKETS ---
        all_markets = self.get_markets(closing_today=closing_today)
        # ----------------------------------------

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


# 4. FIX: The __main__ block must be outside the class definition
if __name__ == "__main__":
    import textwrap

    # Your Kalshi API credentials
    MY_API_KEY = "822c8fdc-1cd7-4d54-a613-75ba788f9721"
    MY_API_SECRET_INDENTED = """
    -----BEGIN RSA PRIVATE KEY-----
    MIIEpQIBAAKCAQEApFTQD+Y6iyqArNwsOn+zhBG6rE4bwlptZPe1DCxnU6uhQ2cx
    tOTkj1GlZVRRL5j57wgCAVpBwqYeNuqGtSotFmPFcXgT2gRdKZNUeiFBPLzs1w1V
    AAhxWQIzCWywehcIhNA5HiRhFI1ChRKEcgI1q67hgtmP/Jf4DGVY5YhGa1V6BlrN
    sBu4T9BWm8nn1g1DC1catDRtJlurzRk0kVajTBw2TnAFXxpCbswSoQp7jcNKAzgP
    igwih2fTferkvZRwc8u5l9mtHFx/WWU6a8ORKPm1sNoc4RHny8Ip1fdPyyuzMikR
    5ae71aN83440/3v01Rbhc7/ChV0nALz9vzOIcQIDAQABAoIBAAnQbuSkMVbiYPDx
    7UpJGiWHEYj82TeQQvxGtu9VL46Vr3nhDdzp1qSgkaotkPOwxSx7Y/NEuqyvUxqg
    gw8Kx8qQhtgx/a2FmTLEc8Ufe0/vUI2/ZBYuauaP9RBZB2kjOwUDkoW0Un/xt8lO
    w84zR5VLSniQGcNSKCRFYTvEGcCLL8ndgFWzjoqOdnz2jfkEUVrmrHGXbcJ3qhbb
    AXlkzYzlKfxb/nbZelDOhcUaDTo2ytcQ0nHQ24pmVndVp6MSNuhAqC3a/1YTOrNN
    xQD/HiFQ8MgwYddZhri5ksKFJeK//ivBxWPVC4KU0vbwpPMqkyMl3PDWvJllCNMG
    iGMUy2ECgYEA1uQSrjAwS9KytHMhAn5jX4q29e3YLE9VVVMvkYg6iX0XRgjR8wA9
    leT3bkIsSXr0LBQ9JRUIUC7s9Usis0S4/r1Hp/nnObl18lOMoGAx56L4acCYOSRS
    CQgG64aiaEql4X7C9+HuVE9F974Rs70aqD3PasXGmSv3BEqAWJs+mkUCgYEAw8Sq
    nJvKLnnpG/uXkUIPgXG2CD2J8wNoaSX9f7CplAHbzkq4cSeIOUYlUM5U86iWiXh8
    c8S5sW3w71sx9P0/x11Ujj9sa7qjhi+LZvpsswtdHt15nR25EDH8ru7tzCs9rmXU
    sabzIr3wUkKbg8HSkBu6DKzyqFwKB3+66uOFDj0CgYEAh3ZUtEuaVmHm10vwFiNY
    P2UxOsyJRj1ofJYo2WP4Cq25WyO6PaX8LJ6ev3mlc1i3zYWgmdytVeaO535KzJlY
    yPTG1AP8F+5qhKzYbEzXiL66O+f1zaewsxLFUfmYLYhJE2IWQ66/z5P9dlPx0s01
    nbMBKrysGeiWGbVhPPn4N8UCgYEAgXvL6Oe2E4V45JRFDMOn49MlNlAVtRFU9u80
    u0dK8mVEUC7lzZn7JP67YbYHRF4Gq4hwsFW3CJ8SFA66fTMgAyo86hUTDjIVRISf
    7I3IZagngGm2rW/iXs7hNYc866TSGE6sHpCxEhKVKKN7nusM7VoZdZbSrP6rd4hJ
    RmEUOXUCgYEAirA+fnNpN2T1BA3plXQE1WmYEeLsGt2YwtQYs6bib7OS5I3CpUHS
    T75uTTh7mnPOm3VN6+xJXLfPJjx8mjt+Oc7a0AdPttviOAqAuQQqAsiH5ExeKDve
    EjuRL/7EYWrfTX1F8GvceGfpSOk6C0EDaS5Fdbd4GAAtAjJIgoiNRYY=
    -----END RSA PRIVATE KEY-----
    """
    MY_API_SECRET = textwrap.dedent(MY_API_SECRET_INDENTED)

    integrator = KalshiIntegrator(api_key=MY_API_KEY, api_secret=MY_API_SECRET)

    if not integrator.is_configured():
        print("\n--- ⚠️ WARNING: Integrator not fully configured. Check API key / RSA key. ---")
    else:
        league = "NBA"
        print("\n=====================================")
        print(f"🏀 Pulling {league} game markets")
        print("=====================================\n")

        game_markets = integrator.get_game_markets_for_events(league=league)
        logger.info(f"Total markets pulled for league {league}: {len(game_markets)}")

        print("\n=====================================")
        print(f"🎯 Total Game Markets Pulled: {len(game_markets)}")
        print("=====================================\n")

        if game_markets:
            # Show a few example markets
            for i, market in enumerate(game_markets[:5], start=1):
                title = market.get("title", "No title")
                ticker = market.get("ticker", "N/A")
                yes_ask = market.get("yes_ask_dollars")
                no_ask = market.get("no_ask_dollars")

                print(f"[{i}] {title}")
                print(f"    Ticker: {ticker}")
                print(f"    YES ask: {yes_ask}")
                print(f"    NO ask : {no_ask}")
                print()

        else:
            print("No game markets returned by Kalshi for this league right now.")
