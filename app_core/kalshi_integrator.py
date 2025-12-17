"""
Kalshi Integrator with RSA Signing & Multi-Series NBA Fetching.
Location: app_core/kalshi_integrator.py
"""
from __future__ import annotations

import logging
import os
import time
import base64
import random
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import streamlit as st

# Cryptography for RSA Signing (Required for Kalshi v2)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
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
    debug: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------

SUPPORTED_LEAGUES = {"NBA", "NFL", "MLB", "NHL", "NCAAF", "NCAAB"}
SAFE_STATUS_ALLOWLIST = {"active", "finalized", "settled", "closed"}
NBA_TZ = pytz.timezone("America/New_York")

# Specific series mapping for daily slates vs futures
LEAGUE_SERIES_MAP: Dict[str, Any] = {
    "NBA": [
        "KXNBAGAME",   # Daily Winners
        "KXNBATOTAL",  # Daily Totals
        "KXNBASPREAD", # Daily Spreads
        "KXNBA",       # Generic/Futures
    ],
    "NFL": ["KXNFL", "KXNFLGAME"],
    "MLB": "KXMLB",
    "NHL": "KXNHL",
}

NBA_TEAM_CODE_MAP: Dict[str, str] = {
    "NEW YORK KNICKS": "NYK", "SAN ANTONIO SPURS": "SAS", "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE", "MINNESOTA TIMBERWOLVES": "MIN", "MEMPHIS GRIZZLIES": "MEM",
    "LOS ANGELES CLIPPERS": "LAC", "LA CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL", "LA LAKERS": "LAL",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_status(status: Optional[str]) -> Optional[str]:
    """Kalshi API v2 status filter normalization."""
    if not status: return None
    s = str(status).strip().lower()
    if s == "open": return None 
    return s if s in SAFE_STATUS_ALLOWLIST else None

def price_to_prob(price: Any) -> Optional[float]:
    if price is None: return None
    try:
        val = float(price)
        if 0.0 <= val <= 100.0: return max(0.0, min(1.0, val / 100.0))
    except: pass
    return None

def _nba_date_token(dt: datetime) -> str:
    # Kalshi uses YYMONDD (e.g., 25DEC17) based on local game date
    return dt.astimezone(NBA_TZ).strftime("%y%b%d").upper()

# ---------------------------------------------------------------------------
# KalshiIntegrator Class
# ---------------------------------------------------------------------------

class KalshiIntegrator:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, *, required: bool = False):
        self.api_key = api_key or st.secrets.get("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY")
        raw_secret = api_secret or st.secrets.get("KALSHI_API_SECRET") or os.getenv("KALSHI_API_SECRET")
        self.api_secret_pem = self._normalize_secret(raw_secret)
        self.api_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.session = requests.Session()
        self.required = required

        # --- FIX ATTRIBUTEERROR: Initialize expected properties ---
        self.last_error_info = {}       
        self.last_status_code = None
        self.last_response_text = None
        self.last_request_params = None
        self._markets_cache = []
        self._league_cache = {}

    @staticmethod
    def _normalize_secret(secret_val: Optional[str]) -> Optional[str]:
        if not secret_val: return None
        cleaned = str(secret_val).replace("\\n", "\n").strip()
        if "-----BEGIN" in cleaned: return cleaned
        return f"-----BEGIN PRIVATE KEY-----\n{cleaned}\n-----END PRIVATE KEY-----"

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        if not self.api_secret_pem: return ""
        msg = f"{timestamp}{method}{path}"
        key = serialization.load_pem_private_key(self.api_secret_pem.encode("utf-8"), password=None)
        sig = key.sign(msg.encode("utf-8"), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return base64.b64encode(sig).decode("utf-8")

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url, path_sign, ts = f"{self.api_url}{path}", f"/trade-api/v2{path}", str(int(time.time() * 1000))
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": self._sign_request(method, path_sign, ts),
            "KALSHI-ACCESS-TIMESTAMP": ts
        }
        try:
            resp = self.session.request(method, url, headers=headers, params=params, timeout=10)
            self.last_status_code, self.last_response_text = resp.status_code, resp.text
            self.last_error_info = {"status_code": resp.status_code}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'status_code'):
                self.last_error_info = {"status_code": e.response.status_code}
            raise e

    def get_markets_paginated(self, status: Optional[str] = None, extra_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        all_markets, cursor = [], None
        for _ in range(5):
            p = {"limit": 200, "cursor": cursor}
            if status: p["status"] = normalize_status(status)
            if extra_params: p.update(extra_params)
            try:
                data = self._request("GET", "/markets", params=p)
                markets = data.get("markets", [])
                all_markets.extend(markets)
                cursor = data.get("cursor")
                if not cursor or not markets: break
            except: break
        return all_markets

    def get_league_markets(
        self, 
        league: str, 
        status: Optional[str] = None, 
        min_prefix_hits: int = 20,     # <--- Add this parameter
        max_pages: int = 5,            # <--- Ensure this is also present
        **kwargs                       # <--- Catch any other unexpected arguments
    ) -> List[Dict[str, Any]]:
        """Targeted NBA Fetch with flexible parameters."""
        """Multi-series fetch to find Winners, Totals, and Spreads."""
        league_key = league.upper()
        series_targets = LEAGUE_SERIES_MAP.get(league_key, [])
        if not isinstance(series_targets, list): series_targets = [series_targets]
        
        collected = {}
        for series in series_targets:
            chunk = self.get_markets_paginated(status=status, extra_params={"series_ticker": series})
            for m in chunk:
                ticker = m.get("ticker") or m.get("event_ticker")
                if ticker: collected[ticker] = m
        return list(collected.values())

    def get_sports_markets(self, league: str) -> List[Dict[str, Any]]:
        return self.get_league_markets(league)

    def get_markets_for_date_token(self, league: str, date_token: str, status: Optional[str] = None) -> Dict[str, Any]:
        all_m = self.get_league_markets(league, status=status)
        bucket = [m for m in all_m if date_token in (m.get("event_ticker") or m.get("ticker") or "")]
        return {"bucket": bucket, "all_markets": all_m}

    def _best_market_time(self, m: Dict[str, Any]) -> Optional[datetime]:
        for k in ["expected_expiration_time", "close_time"]:
            val = m.get(k)
            if not val: continue
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                return dt if dt.tzinfo else pytz.utc.localize(dt)
            except: pass
        return None

    def split_market_kinds(self, markets: List[Dict[str, Any]], league: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        single, multi, other = [], [], []
        for m in (markets or []):
            t = (m.get("event_ticker") or m.get("ticker") or "").upper()
            if "GAME-" in t or "KXNBAGAME" in t: single.append(m)
            else: other.append(m)
        return {"single_game_candidates": single, "multivariate_bundles": multi, "other": other}

    def assert_available(self) -> None:
        if not self.api_key or not self.api_secret_pem:
            raise RuntimeError("Kalshi keys missing from secrets.")

def match_game_to_kalshi(league: str, home_team: str, away_team: str, game_time: Optional[datetime], integrator: KalshiIntegrator = None, status: Optional[str] = None) -> KalshiMatchResult:
    l_key = league.upper()
    ki = integrator or KalshiIntegrator()
    if not ki.api_key: return KalshiMatchResult(False, False, "", None, None, reason="no_auth")
    
    if l_key == "NBA":
        if not game_time: return KalshiMatchResult(False, True, "", None, None, reason="no_time")
        date_tok = _nba_date_token(game_time)
        away_c, home_c = NBA_TEAM_CODE_MAP.get(away_team.upper()), NBA_TEAM_CODE_MAP.get(home_team.upper())
        if not away_c or not home_c: return KalshiMatchResult(False, True, "", None, None, reason="no_codes")
        
        matchup = f"{away_c}{home_c}"
        res = ki.get_markets_for_date_token(l_key, date_tok, status=status)
        candidates = [m for m in res['bucket'] if matchup in (m.get("event_ticker") or "")]
        
        if not candidates: return KalshiMatchResult(False, True, "", None, None, reason="no_match")
        exact = candidates[0]
        return KalshiMatchResult(True, True, exact.get("title", ""), price_to_prob(exact.get("last_price")), exact.get("event_ticker"), league=l_key, market_type="winner")
    
    return KalshiMatchResult(False, True, "", None, None, reason="unsupported_league")
