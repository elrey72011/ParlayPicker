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
    class TeamNameMatcher:
        @staticmethod
        def normalize(s): return s.lower().strip()
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
LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

TEAM_FUZZY_THRESHOLD = 0.65

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "": return None
    try:
        p = float(price)
        if p > 1.01: p = p / 100.0
        return max(0.0, min(1.0, p))
    except: return None

def _parse_market_date(raw) -> Optional[datetime]:
    if not raw: return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=pytz.UTC)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except: return None

def _find_best_market_match(
    home_team: str,
    away_team: str,
    game_dt: Optional[datetime],
    markets: List[Dict[str, Any]],
    league_code: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not markets: return None, "no_events"

    date_filtered = markets
    if game_dt:
        date_filtered = []
        if game_dt.tzinfo is None: game_dt = game_dt.replace(tzinfo=pytz.UTC)
        else: game_dt = game_dt.astimezone(pytz.UTC)
        target_date = game_dt.date()
        
        for m in markets:
            m_dt = _parse_market_date(m.get("close_time") or m.get("event_date"))
            if not m_dt: continue
            if abs((m_dt.astimezone(pytz.UTC).date() - target_date).days) <= 1:
                date_filtered.append(m)
        
        if not date_filtered: return None, "date_mismatch"
    
    best_market = None
    best_score = 0.0
    norm_home = TeamNameMatcher.normalize(home_team)
    norm_away = TeamNameMatcher.normalize(away_team)
    
    for market in date_filtered:
        series = str(market.get("series_ticker") or "").upper()
        if league_code and league_code in LEAGUE_SERIES_MAP:
            expected = LEAGUE_SERIES_MAP[league_code]
            if not series.startswith(expected): continue

        title = market.get("title") or ""
        ticker = market.get("ticker") or ""
        market_norm = TeamNameMatcher.normalize(f"{title} {market.get('subtitle','')}")
        
        home_in = norm_home in market_norm
        away_in = norm_away in market_norm
        
        quality = 0.0
        if home_in and away_in: quality = 1.0
        else:
            h_s = TeamNameMatcher.similarity_score(norm_home, market_norm)
            a_s = TeamNameMatcher.similarity_score(norm_away, market_norm)
            if h_s > 0.4 and a_s > 0.4: quality = (h_s + a_s) / 2.0
            
        if "GAME" in ticker or "SPREAD" in ticker: quality += 0.1
        
        if quality > best_score:
            best_score = quality
            best_market = market

    if best_score < TEAM_FUZZY_THRESHOLD: return None, "below_threshold"
    return best_market, "ok"

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        try:
            self.api_key = api_key or st.secrets.get("KALSHI_API_KEY")
            self.api_secret = api_secret or st.secrets.get("KALSHI_API_SECRET")
        except:
            self.api_key = api_key
            self.api_secret = api_secret

        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.api_url = self.base_url if self.api_key else self.demo_url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
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
                    key_data.encode(), password=None, backend=default_backend()
                )
                self._auth_ready = True
            except Exception as e:
                logger.warning(f"Kalshi Auth Error: {e}")

        self._markets_cache = {}
        self._cache_time = {}
    
    def _sign_request(self, method, path, timestamp):
        if not self._private_key: return ""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            msg = f"{timestamp}{method}{path}"
            sig = self._private_key.sign(
                msg.encode('utf-8'),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256()
            )
            return base64.b64encode(sig).decode('utf-8')
        except: return ""

    def _make_authenticated_request(self, method, endpoint, params=None):
        import time as tm
        url = f"{self.api_url}{endpoint}"
        timestamp = str(int(tm.time() * 1000))
        headers = self.headers.copy()
        
        if self._auth_ready:
            path = endpoint.split('?')[0]
            sig = self._sign_request(method.upper(), path, timestamp)
            headers.update({
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            })
            
        try:
            if method == "GET": resp = requests.get(url, headers=headers, params=params, timeout=10)
            else: resp = requests.post(url, headers=headers, json=params, timeout=10)
            
            if resp.status_code == 200: return resp.json()
            if resp.status_code == 429:
                time.sleep(1)
                return self._make_authenticated_request(method, endpoint, params)
        except Exception as e:
            logger.error(f"Kalshi Req Error: {e}")
        return None

    def get_sports_series(self):
        data = self._make_authenticated_request("GET", "/series", {"limit": 1000})
        if not data: return []
        keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'NCAAF', 'NCAAB']
        return [s for s in data.get("series", []) if any(k in s.get('ticker','').upper() for k in keywords)]

    def get_markets(self, category="sports", status="open"):
        cache_key = f"{category}_{status}"
        if cache_key in self._markets_cache: return self._markets_cache[cache_key]

        series_list = self.get_sports_series()
        all_markets = []
        
        priority = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXNCAAB"]
        series_list.sort(key=lambda x: next((i for i,p in enumerate(priority) if x.get('ticker','').startswith(p)), 999))

        for s in series_list:
            ticker = s.get('ticker')
            cursor = None
            while True:
                p = {"series_ticker": ticker, "limit": 100, "status": status}
                if cursor: p["cursor"] = cursor
                
                data = self._make_authenticated_request("GET", "/markets", p)
                if not data: break
                
                ms = data.get("markets", [])
                all_markets.extend(ms)
                
                cursor = data.get("cursor")
                if not cursor: break
                if len(ms) < 100: break

        self._markets_cache[cache_key] = all_markets
        return all_markets

    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        markets = self.get_markets()
        
        game_dt = None
        if game_time:
            try: game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
            except: pass
            
        best, reason = _find_best_market_match(home_team, away_team, game_dt, markets, sport.lower() if sport else None)
        
        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": reason,
            "kalshi_label": None
        }
        
        if best:
            prob = None
            for f in ["yes_bid_dollars", "last_price", "implied_prob"]:
                if f in best:
                    p = price_to_prob(best[f])
                    if p is not None: 
                        prob = p
                        break
            
            res.update({
                "kalshi_available": True,
                "kalshi_prob": prob,
                "kalshi_label": best.get("title"),
                "kalshi_match_debug": "match_found",
                "market_ticker": best.get("ticker")
            })
            
        return res

def match_game_to_kalshi(league, home, away, time, integrator=None, status="open"):
    k = integrator or KalshiIntegrator()
    sport = next((c for c in SUPPORTED_LEAGUES if c in league.lower()), "nba")
    r = k.get_game_market(home, away, sport, time)
    return KalshiMatchResult(
        matched=r["kalshi_available"],
        label=r.get("kalshi_label"),
        probability=r.get("kalshi_prob"),
        reason=r.get("kalshi_match_debug")
    )
