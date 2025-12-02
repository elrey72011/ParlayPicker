# app_core/kalshi_integrator.py

from __future__ import annotations

import os
import logging
import datetime as dt
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Default base URL (can be overridden via env/secrets)
DEFAULT_KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Map your sports leagues to Kalshi series tickers
KALSHI_LEAGUE_MAP: Dict[str, List[str]] = {
    # You can expand these if you add more series
    "NBA": ["KXNBASPREAD", "KXNBAGAME", "KXNBATOTAL"],
    "NFL": ["KXNFLSPREAD", "KXNFLGAME", "KXNFLTOTAL"],
    "NHL": ["KXNHLSPREAD", "KXNHLGAME", "KXNHLTOTAL"],
    "NCAAF": ["KXNCAAFSPREAD", "KXNCAAFGAME", "KXNCAAFTOTAL"],
    "NCAAB": ["KXNCAABSPREAD", "KXNCAABGAME", "KXNCAABTOTAL"],
}


def price_to_prob(price: Optional[float | str]) -> Optional[float]:
    """
    Convert a Kalshi price (like '0.6500' or 65) to a probability 0–1.

    Returns None if the price is missing or invalid.
    """
    if price is None or price == "":
        return None

    try:
        p = float(price)
    except (TypeError, ValueError):
        return None

    # If it's in cents (0–100), normalize
    if p > 1.01:
        p = p / 100.0

    # Sanity check
    if p <= 0.0 or p >= 1.0:
        return None

    return p


class KalshiIntegrator:
    """
    Thin wrapper around the Kalshi REST API, focused on sports markets.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        # API key: prefer explicit arg, then env/secrets
        self.api_key = (
            api_key
            or os.getenv("KALSHI_API_KEY")
            or os.getenv("KALSHI_API_TOKEN")
        )

        # Base URL
        self.api_url = api_url or os.getenv("KALSHI_API_URL") or DEFAULT_KALSHI_API_URL
        self.session = requests.Session()

        self.last_error: Optional[str] = None
        self.last_message: Optional[str] = None

        if self.api_key:
            logger.info("✅ Kalshi API key loaded (starts with %s)", self.api_key[:6])
        else:
            logger.warning("⚠️ No Kalshi API key found. Set KALSHI_API_KEY in secrets/env.")

        logger.info("Using Kalshi API URL: %s", self.api_url)

    # ------------------------------------------------------------------ #
    # Core request helper
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Low-level authenticated request to Kalshi.
        Raises on HTTP error; returns parsed JSON.
        """
        url = self.api_url.rstrip("/") + path

        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("Kalshi API request: %s %s params=%r", method, url, params)

        resp = self.session.request(method, url, headers=headers, params=params, timeout=10)
        logger.info("Kalshi API response: %s %s", resp.status_code, resp.url)

        try:
            data = resp.json()
        except Exception:
            data = {}

        if not resp.ok:
            msg = f"Kalshi HTTP {resp.status_code}: {data or resp.text}"
            self.last_error = msg
            logger.error(msg)
            resp.raise_for_status()

        self.last_error = None
        return data or {}

    # ------------------------------------------------------------------ #
    # Public helpers used by Streamlit app
    # ------------------------------------------------------------------ #
    def get_game_markets_for_events(
        self,
        league: str,
        status: str = "open",
        only_today: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Pull markets for a single league (e.g. 'NBA', 'NFL').

        - status: 'open' (default) or 'all'
        - only_today: if True, filter to markets whose close_time is today (UTC).
        """
        league_key = league.upper()
        series_list = KALSHI_LEAGUE_MAP.get(league_key, [])

        if not series_list:
            msg = f"No series mapping configured for league {league_key}"
            self.last_error = msg
            logger.warning(msg)
            return []

        all_markets: List[Dict[str, Any]] = []

        for series in series_list:
            params = {
                "limit": 1000,
                "series_ticker": series,
                "status": status,
            }
            data = self._request("GET", "/markets", params=params)
            markets = data.get("markets", []) or []

            logger.info(
                "Series %s: pulled %d markets (status=%s)",
                series,
                len(markets),
                status,
            )

            if only_today and markets:
                markets = self.filter_markets_closing_today(markets)

            all_markets.extend(markets)

        logger.info("Total %s markets pulled for league %s", len(all_markets), league_key)
        self.last_message = f"{len(all_markets)} markets pulled for {league_key}"
        return all_markets

    def filter_markets_closing_today(
        self,
        markets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return only markets whose close_time is today (UTC).
        """
        if not markets:
            return []

        today_utc = dt.datetime.utcnow().date()
        filtered: List[Dict[str, Any]] = []

        for m in markets:
            close_str = m.get("close_time")
            if not close_str:
                continue
            try:
                # Example: '2025-12-17T01:00:00Z'
                close_dt = dt.datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                if close_dt.date() == today_utc:
                    filtered.append(m)
            except Exception:
                continue

        logger.info("Filtered to %d markets with close_time today (UTC)", len(filtered))
        return filtered

    def get_sports_markets(
        self,
        leagues: Optional[List[str]] = None,
        only_today: bool = False,
        status: str = "open",
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: pull markets across multiple leagues at once.

        - leagues: list like ['NFL', 'NBA']; if None, uses all known leagues.
        - only_today: pass through to get_game_markets_for_events.
        """
        if leagues is None:
            leagues = list(KALSHI_LEAGUE_MAP.keys())

        all_markets: List[Dict[str, Any]] = []

        for lg in leagues:
            lg_markets = self.get_game_markets_for_events(
                league=lg,
                status=status,
                only_today=only_today,
            )
            all_markets.extend(lg_markets)

        self.last_message = f"Pulled {len(all_markets)} markets from {len(leagues)} league(s)"
        return all_markets
