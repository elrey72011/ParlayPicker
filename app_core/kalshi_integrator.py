"""
Kalshi Integrator with RSA Signing, Auto-Key Formatting, and Fuzzy Matching.
Location: app_core/kalshi_integrator.py
"""

from __future__ import annotations

import logging
import os
import time
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
import requests
import streamlit as st

# Cryptography for RSA Signing (Required for Kalshi v2)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Mappings
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

# ---------------------------------------------------------------------------
# Helper Functions
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
    if price is None:
        return None
    try:
        val = float(price)
    except Exception:
        return None
    if val < 0:
        return None
    if 0.0 <= val <= 1.0:
        return max(0.0, min(1.0, val))
    if 0.0 <= val <= 100.0:
        return max(0.0, min(1.0, val / 100.0))
    return None

def _extract_market_type(title: str, ticker: str) -> str:
    t = (title or "").upper()
    if "SPREAD" in t or "POINTS" in t: return "spread"
    if "TOTAL" in t or "OVER/UNDER" in t or "O/U" in t: return "total"
    if "MONEYLINE" in t or "ML" in t: return "moneyline"
    return "generic"

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
# Parsing Logic
# ---------------------------------------------------------------------------

def _parse_market_metadata(mkt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = (mkt.get("title") or "").strip()
    if not title: return None
    ticker = (mkt.get("ticker") or mkt.get("event_ticker") or "")
    
    market_dt: Optional[datetime] = None
    # Prioritize close_time (Trading Close)
    close_raw = (mkt.get("close_time") or mkt.get("expiration_time") or mkt.get("expected_expiration_time"))
    if close_raw:
        try:
            market_dt = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
        except Exception:
            market_dt = None

    upper_title = title.upper()
    teams: List[str] = []
    separators = [" VS ", " VS. ", " V ", " @ ", " AT ", " - ", " / ", " | "]
    for sep in separators:
        if sep in upper_title:
            parts = upper_title.split(sep)
            if len(parts) >= 2:
                teams = [parts[0].strip(), parts[1].strip()]
            break

    # Fallback to ticker parsing if title fails
    if len(teams) < 2 and ticker:
        parts = ticker.split("-")
        if len(parts) >= 3:
            teams = [p.strip().upper() for p in parts[1:3]]

    market_type = _extract_market_type(title, ticker)
    prob_source = (mkt.get("yes_price") or mkt.get("last_price") or mkt.get("yes_ask") or mkt.get("implied_prob"))
    prob = price_to_prob(prob_source)

    return {"title": title, "market_date": market_dt, "teams": teams, "probability": prob, "market_type": market_type}

def _team_score(team_code: str, target_norm: str) -> float:
    """Simple fuzzy matching score."""
    if not team_code: return 0.0
    norm_code = normalize_name(team_code)
    
    # Exact Match
    if norm_code == target_norm: return 2.0
    # Substring Match
    if norm_code in target_norm or target_norm in norm_code: return 1.5
    
    # Word Overlap
    words_code = set(norm_code.split())
    words_target = set(target_norm.split())
    if words_code & words_target: return 1.0
    
    return 0.0

# ---------------------------------------------------------------------------
# Matcher Function
# ---------------------------------------------------------------------------

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: "KalshiIntegrator" = None, status: Optional[str] = "open") -> KalshiMatchResult:
    """Finds the best Kalshi market match for a given game."""
    league_key = (league or "").upper()
    kalshi = integrator or KalshiIntegrator()
    
    if not kalshi or not kalshi.api_key:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_integrator")

    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)

    markets = kalshi.get_markets(status=status)
    if not markets:
        return KalshiMatchResult(matched=False, kalshi_available=False, label="", probability=None, raw_event_id=None, reason="no_markets_found")

    # Normalize game time to UTC for comparison
    game_dt_utc: Optional[datetime] = None
    if isinstance(game_time, datetime):
        game_dt_utc = game_time.astimezone(pytz.UTC)

    series_prefix = LEAGUE_SERIES_MAP.get(league_key)
    best_market = None
    best_score = 0.0

    for m in markets:
        meta = _parse_market_metadata(m)
        if not meta: continue
        
        ticker = (m.get("ticker") or "").upper()
        if series_prefix and not ticker.startswith(series_prefix):
            continue

        teams = meta.get("teams") or []
        if len(teams) < 2: continue

        # Score both orientations (Home vs Away OR Away vs Home)
        s1 = _team_score(teams[0], home_norm) + _team_score(teams[1], away_norm)
        s2 = _team_score(teams[0], away_norm) + _team_score(teams[1], home_norm)
        score = max(s1, s2)

        # Date penalty (±24 hours tolerance for timezone shifts)
        m_date = meta.get("market_date")
        if game_dt_utc and m_date:
            try:
                # Compare dates
                diff = abs((m_date.date() - game_dt_utc.date()).days)
                if diff > 1: # Strict 1 day tolerance
                     continue 
            except Exception:
                pass 

        if score > best_score:
            best_score = score
            best_market = m
            best_market["__meta"] = meta

    if not best_market or best_score < 1.5: # Threshold of 1.5 ensures at least one strong name match
        return KalshiMatchResult(matched=False, kalshi_available=True, label="", probability=None, raw_event_id=None, reason=f"low_score_{best_score:.1f}")

    meta = best_market["__meta"]
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=meta["title"],
        probability=meta["probability"],
        raw_event_id=best_market.get("ticker"),
        league=league_key,
        reason="matched",
        market_type=meta["market_type"],
        game_date=meta["market_date"]
    )

# ---------------------------------------------------------------------------
# KalshiIntegrator Class (with RSA & Auto-Formatting)
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        # Load keys from args, then secrets, then env
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY", "") or os.getenv("KALSHI_API_KEY", "")
        self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET", "") or os.getenv("KALSHI_API_SECRET", "")
        
        # KEY FIX: Clean the private key string to handle formatting issues
        if self.api_secret:
            # Replace literal "\n" characters with actual newlines if they exist
            self.api_secret = self.api_secret.replace("\\n", "\n").strip()
            # Ensure headers/footers are clean
            if "-----BEGIN RSA PRIVATE KEY-----" not in self.api_secret:
                # Try adding headers if missing (rare but possible)
                self.api_secret = f"-----BEGIN RSA PRIVATE KEY-----\n{self.api_secret}\n-----END RSA PRIVATE KEY-----"

        self.api_url = "https://trading-api.kalshi.com/trade-api/v2"
        self._markets_cache = []
        self._markets_cache_ts = 0
        self.cache_ttl_seconds = 300  # Cache for 5 minutes
        self.last_error = None

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Sign request using RSA-PSS SHA256."""
        if not self.api_secret:
            return ""
        
        msg_string = f"{timestamp}{method}{path}"
        
        try:
            # Load the formatted private key
            private_key = serialization.load_pem_private_key(
                self.api_secret.encode('utf-8'),
                password=None
            )
            
            signature = private_key.sign(
                msg_string.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            self.last_error = f"Signing Error: {e}"
            return ""

    def _make_authenticated_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[dict]:
        url = f"{self.api_url}{endpoint}"
        
        # Path for signature must be relative, e.g., '/trade-api/v2/markets'
        path_for_signing = f"/trade-api/v2{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        signature = self._sign_request(method, path_for_signing, timestamp)
        
        if not signature:
            logger.error("Failed to generate signature. Check private key format.")
            return None

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }
        
        try:
            resp = requests.request(method, url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                self.last_error = f"Status {resp.status_code}: {resp.text}"
                logger.error(f"Kalshi API Error: {self.last_error}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi Connection Exception: {e}")
        return None

    def get_markets(self, status: str = "open") -> List[Dict[str, Any]]:
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self.cache_ttl_seconds:
            return self._markets_cache

        # Fetch markets (limit 2000 to get everything)
        data = self._make_authenticated_request("GET", "/markets", params={"limit": 2000, "status": status})
        
        markets = data.get("markets", []) if data else []
        
        if markets:
            self._markets_cache = markets
            self._markets_cache_ts = now
            logger.info(f"✅ Successfully loaded {len(markets)} Kalshi markets")
        
        return markets

    # Compatibility aliases
    def get_sports_markets(self): return self.get_markets()
    def get_game_markets_for_events(self, league): return self.get_markets()
    def filter_markets_closing_today(self, markets): return markets
    def get_orderbook(self, ticker): return self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook") or {}
