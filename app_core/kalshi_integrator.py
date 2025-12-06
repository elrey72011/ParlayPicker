"""
Kalshi Integrator: Uses Public Data API if no keys provided.
This file goes in: app_core/kalshi_integrator.py
"""

import time
import logging
import re
from datetime import datetime
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
# Map common league codes to Kalshi Series Prefixes
LEAGUE_SERIES_MAP = {
    "nba": "KXNBA",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
    "nhl": "KXNHL",
    "ncaaf": "KXNCAAF",
    "ncaab": "KXNCAAB",
}

# Lowered threshold to catch more games (e.g. "St" vs "State" variations)
TEAM_FUZZY_THRESHOLD = 0.55 

def price_to_prob(price) -> Optional[float]:
    if price is None or price == "": return None
    try:
        p = float(price)
        # Kalshi prices are in cents (1-99), convert to 0.0-1.0
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

    # 1. Date Filter
    date_filtered = markets
    if game_dt:
        date_filtered = []
        if game_dt.tzinfo is None: game_dt = game_dt.replace(tzinfo=pytz.UTC)
        else: game_dt = game_dt.astimezone(pytz.UTC)
        target_date = game_dt.date()
        
        for m in markets:
            m_dt = _parse_market_date(m.get("close_time") or m.get("event_date"))
            if not m_dt: continue
            # Kalshi markets often close a few hours after the game starts
            # Allow matches for same day or next day (UTC shift)
            m_local = m_dt.astimezone(pytz.UTC).date()
            if abs((m_local - target_date).days) <= 1:
                date_filtered.append(m)
        
        if not date_filtered: 
            # If date strict filtering fails, try to return ALL markets if list is small,
            # otherwise return None. This acts as a fallback for timezone edge cases.
            if len(markets) < 500:
                date_filtered = markets
            else:
                return None, "date_mismatch"
    
    # 2. Name Matching
    best_market = None
    best_score = 0.0
    
    norm_home = TeamNameMatcher.normalize(home_team)
    norm_away = TeamNameMatcher.normalize(away_team)
    
    # Pre-calculate series prefix
    expected_series = LEAGUE_SERIES_MAP.get(league_code, "") if league_code else ""

    for market in date_filtered:
        # Series Filter (Soft filter - only if we have a league code)
        if expected_series:
            series = str(market.get("series_ticker") or "").upper()
            # If it's a specific game prop, it might not start with KXNBA, so be careful
            # But generally for game winners it should.
            if not series.startswith(expected_series): 
                # Check if it's a generic ticker match
                pass 

        title = market.get("title") or ""
        subtitle = market.get("subtitle") or ""
        ticker = market.get("ticker") or ""
        
        # Combine title and subtitle for matching
        market_full_text = f"{title} {subtitle}".lower()
        market_norm = TeamNameMatcher.normalize(market_full_text)
        
        # Token overlap check (More robust than exact string match)
        home_in = norm_home in market_norm
        away_in = norm_away in market_norm
        
        quality = 0.0
        
        # Perfect token match
        if home_in and away_in: 
            quality = 1.0
        else:
            # Fuzzy fallback
            h_s = TeamNameMatcher.similarity_score(norm_home, market_norm)
            a_s = TeamNameMatcher.similarity_score(norm_away, market_norm)
            
            # IMPROVED LOGIC: Average score instead of strict AND
            # This allows "Lakers" (1.0) vs "Golden St" (0.4) to still match if average is high
            avg_score = (h_s + a_s) / 2.0
            
            # Boost if at least one team is a very strong match
            if h_s > 0.8 or a_s > 0.8:
                quality = avg_score + 0.1
            else:
                quality = avg_score
            
        # Boost "Game Winner" or "Spread" markets over player props
        # Tickers usually look like 'KXNBA-23NOV24-LAL-GSW'
        if "GAME" in ticker.upper() or len(ticker.split('-')) <= 4: 
            quality += 0.15
        
        if quality > best_score:
            best_score = quality
            best_market = market

    if best_score < TEAM_FUZZY_THRESHOLD: 
        return None, f"below_threshold ({best_score:.2f} < {TEAM_FUZZY_THRESHOLD})"
        
    return best_market, "ok"

class KalshiIntegrator:
    def __init__(self, api_key: str = None, api_secret: str = None):
        print("="*60)
        print("🚀 KALSHI INTEGRATOR v4.1 (Fixed Matching)")
        print("="*60)
        
        # Try to load from Streamlit secrets if not provided
        if not api_key:
            try:
                api_key = st.secrets.get("KALSHI_API_KEY")
                api_secret = st.secrets.get("KALSHI_API_SECRET")
            except:
                pass

        self.api_key = api_key
        self.api_secret = api_secret

        # URL Configuration
        self.prod_url = "https://api.kalshi.com/trade-api/v2"
        self.public_url = "https://api.elections.kalshi.com/trade-api/v2"
        
        if self.api_key and self.api_secret:
            self.api_url = self.prod_url
            print(f"✅ Using Authenticated Production API")
        else:
            self.api_url = self.public_url
            print(f"⚠️ No Keys Found: Using Public Read-Only API (Real Data)")

        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # RSA Setup
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
        self.last_error = None
    
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
        
        if self._auth_ready and self.api_url == self.prod_url:
            path = endpoint.split('?')[0]
            sig = self._sign_request(method.upper(), path, timestamp)
            headers.update({
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": timestamp
            })
            
        try:
            if method == "GET": 
                resp = requests.get(url, headers=headers, params=params, timeout=10)
            else: 
                resp = requests.post(url, headers=headers, json=params, timeout=10)
            
            if resp.status_code == 200: 
                self.last_error = None
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Kalshi Rate Limit. Sleeping...")
                time.sleep(1.0)
                return self._make_authenticated_request(method, endpoint, params)
            else:
                self.last_error = f"{resp.status_code}: {resp.text}"
                logger.error(f"Kalshi API Error: {self.last_error}")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Kalshi Connection Error: {e}")
        return None

    def get_sports_series(self):
        """Fetch all sports series"""
        data = self._make_authenticated_request("GET", "/series", {"limit": 1000})
        if not data: return []
        
        all_series = data.get("series", [])
        keywords = ['NFL', 'NBA', 'MLB', 'NHL', 'NCAAF', 'NCAAB', 'FOOTBALL', 'BASKETBALL']
        
        sports_series = []
        for s in all_series:
            t = s.get('ticker', '').upper()
            c = s.get('category', '').upper()
            if any(k in t or k in c for k in keywords):
                sports_series.append(s)
        return sports_series

    def get_markets(self, category="sports", status="open"):
        """Fetch ALL markets for ALL sports series with pagination"""
        cache_key = f"{category}_{status}"
        now = time.time()
        
        # 5 minute cache
        if cache_key in self._markets_cache and now - self._cache_time.get(cache_key, 0) < 300:
            return self._markets_cache[cache_key]

        series_list = self.get_sports_series()
        if not series_list:
            return []

        all_markets = []
        # Priority sort to fetch major sports first
        priority = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXNCAAB", "KXNCAAF"]
        series_list.sort(key=lambda x: next((i for i,p in enumerate(priority) if x.get('ticker','').startswith(p)), 999))

        for s in series_list:
            ticker = s.get('ticker')
            
            cursor = None
            page = 0
            while True:
                p = {"series_ticker": ticker, "limit": 100, "status": status}
                if cursor: p["cursor"] = cursor
                
                data = self._make_authenticated_request("GET", "/markets", p)
                if not data: break
                
                ms = data.get("markets", [])
                if ms:
                    all_markets.extend(ms)
                
                cursor = data.get("cursor")
                page += 1
                if not cursor or page > 5: break 

        self._markets_cache[cache_key] = all_markets
        self._cache_time[cache_key] = now
        return all_markets

    # --- ALIAS FOR COMPATIBILITY ---
    def get_sports_markets(self):
        """Alias for get_markets to fix Streamlit compatibility"""
        return self.get_markets()
    
    def get_game_markets_for_events(self, league="NBA"):
        """Legacy wrapper for get_markets"""
        return self.get_markets()
        
    def filter_markets_closing_today(self, markets):
        """Filter markets closing soon"""
        if not markets: return []
        today = datetime.now(pytz.UTC).date()
        filtered = []
        for m in markets:
            dt = _parse_market_date(m.get("close_time"))
            if dt and dt.date() == today:
                filtered.append(m)
        return filtered

    def get_orderbook(self, ticker: str):
        """Fetch orderbook for a specific market ticker"""
        data = self._make_authenticated_request("GET", f"/markets/{ticker}/orderbook", {"depth": 5})
        if data and 'orderbook' in data:
            return data['orderbook']
        return None

    def get_game_market(self, home_team, away_team, sport="NBA", game_time=None):
        markets = self.get_markets()
        
        game_dt = None
        if game_time:
            try: game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
            except: pass
            
        best, reason = _find_best_market_match(
            home_team, away_team, game_dt, markets, sport.lower() if sport else None
        )
        
        res = {
            "kalshi_available": False, 
            "kalshi_prob": None, 
            "kalshi_match_debug": reason,
            "kalshi_label": None
        }
        
        if best:
            prob = None
            # Prioritize implied probability, then orderbook
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
                "market_ticker": best.get("ticker"),
                "kalshi_volume": best.get("volume")
            })
            
        return res

# Legacy Wrapper for compatibility
def match_game_to_kalshi(league, home, away, time, integrator=None, status="open"):
    k = integrator or KalshiIntegrator()
    # Normalize league name
    sport_map = {
        'nba': 'nba', 'basketball': 'nba',
        'nfl': 'nfl', 'football': 'nfl',
        'nhl': 'nhl', 'hockey': 'nhl',
        'mlb': 'mlb', 'baseball': 'mlb'
    }
    sport = sport_map.get(league.lower(), "nba")
    
    r = k.get_game_market(home, away, sport, time)
    
    return KalshiMatchResult(
        matched=r["kalshi_available"],
        label=r.get("kalshi_label"),
        probability=r.get("kalshi_prob"),
        reason=r.get("kalshi_match_debug"),
        raw_event_id=r.get("market_ticker"),
        league=sport
    )
