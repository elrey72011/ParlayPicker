"""
ParlayPicker — Full Clean Build (with Kalshi Partial Validation)
---------------------------------------------------------------
This is a self-contained Streamlit app that includes:

- Sidebar:
  • API key inputs (TheOddsAPI, Kalshi, TheOver.ai placeholder)
  • Kalshi Teams Loader
  • Kalshi Debug (fetch counts + market samples)
  • Basic settings

- Tabs:
  • Best Bets (pulls events from TheOddsAPI and previews JSON)
  • Build Parlay (construct legs, validates with Kalshi [strict + partial],
    blends AI/Market/Kalshi, and scores combos; shows Kalshi impact/metrics)

- Sources:
  • TheOddsAPI (moneyline)
  • Kalshi (prediction markets)
  • theOver.ai placeholder hook (non-breaking)

Notes:
- This file is indentation-normalized (4 spaces).
- No external modules beyond Streamlit + requests + pandas + numpy are required.
- All Kalshi logic is defensive; missing/empty orderbooks won’t crash the app.
"""

import os
import re
import json
import math
import itertools
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st

# =========================
# Configuration / Secrets
# =========================

THEODDS_API_KEY = os.getenv("THEODDS_API_KEY") or st.secrets.get("THEODDS_API_KEY", "")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY") or st.secrets.get("KALSHI_API_KEY", "")
KALSHI_API_URL = os.getenv("KALSHI_API_URL") or st.secrets.get("KALSHI_API_URL", "https://trading-api.kalshi.com/v1")
THEOVER_API_KEY = os.getenv("THEOVER_API_KEY") or st.secrets.get("THEOVER_API_KEY", "")  # optional

LOCAL_TZ = timezone(timedelta(hours=0))  # UTC baseline; adjust if needed


# =========================
# Helper Functions
# =========================
def american_to_decimal_safe(odds) -> Optional[float]:
    try:
        o = float(odds)
    except Exception:
        return None
    if o >= 100:
        return 1.0 + o / 100.0
    if o <= -100:
        return 1.0 + 100.0 / abs(o)
    return None


def implied_prob_from_decimal(d: float) -> Optional[float]:
    if d is None or d <= 1.0:
        return None
    return 1.0 / d


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# TheOddsAPI
# =========================
def get_theodds_games(sport_key: str = "basketball_nba", regions: str = "us") -> List[Dict]:
    """Pull games & ML market from TheOddsAPI (requires THEODDS_API_KEY)."""
    if not THEODDS_API_KEY:
        return []
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "oddsFormat": "american",
            "markets": "h2h",
            "apiKey": THEODDS_API_KEY,
        }
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        return r.json() or []
    except Exception:
        return []


def build_legs_from_theodds(events: List[Dict]) -> List[Dict]:
    """
    Create legs (moneyline) from TheOddsAPI events.
    We select the first bookmaker's H2H line for the home team for demo purposes.
    """
    legs: List[Dict] = []
    for ev in events:
        eid = ev.get("id")
        home = ev.get("home_team")
        # Determine away team
        teams = ev.get("teams") or []
        away = None
        for t in teams:
            if t != home:
                away = t
        if not (eid and home and away):
            continue

        # Extract ML odds
        d_odds = None
        p_imp = None
        label = None
        market = "ML"

        bks = ev.get("bookmakers") or []
        if bks:
            mkts = bks[0].get("markets") or []
            if mkts and mkts[0].get("key") == "h2h":
                outcomes = mkts[0].get("outcomes") or []
                # choose the home team outcome
                for out in outcomes:
                    if out.get("name") == home:
                        ap = out.get("price")
                        d = american_to_decimal_safe(ap)
                        p = implied_prob_from_decimal(d) if d else None
                        if d and p:
                            d_odds = d
                            p_imp = p
                            label = f"{home} ML"
                            break

        if d_odds and p_imp and label:
            legs.append({
                "event_id": eid,
                "home": home,
                "away": away,
                "market": market,
                "label": label,
                "d": d_odds,
                "p": p_imp,
                "ai_prob": p_imp,           # start from market implied probability
                "ai_confidence": 0.60,      # placeholder
                "ai_edge": 0.0,             # placeholder
                "sport": "NBA",
            })
    return legs


# =========================
# theOver.ai placeholder
# =========================
def match_theover_to_leg(leg: Dict, theover_json: Any) -> Dict:
    """Placeholder: return neutral impact unless wired to your data."""
    return {"matches": False, "impact": 0.0}


# =========================
# Kalshi Integrator
# =========================
class KalshiIntegrator:
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _list_all_markets(self, status: Optional[str] = "open", limit: int = 100) -> List[Dict]:
        """List markets with pagination; status is optional."""
        all_markets: List[Dict] = []
        cursor = None
        tries = 0
        max_tries = 30
        while tries < max_tries:
            tries += 1
            params = {"limit": limit}
            if status is not None:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{self.api_url}/markets", headers=self.headers, params=params, timeout=12)
                if r.status_code != 200:
                    break
                payload = r.json() or {}
                markets = payload.get("markets", [])
                all_markets.extend(markets)
                if len(markets) < limit:
                    break
                cursor = payload.get("next") or payload.get("next_cursor")
                if not cursor:
                    break
            except Exception:
                break

        # dedupe by ticker/id
        seen = set()
        out: List[Dict] = []
        for m in all_markets:
            key = m.get("ticker") or m.get("id") or id(m)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    def get_all_sports_markets(self, status: Optional[str] = "open") -> List[Dict]:
        sports_keywords = ['NFL','NBA','MLB','NHL','UFC','SOCCER','TENNIS','GOLF','FOOTBALL','BASKETBALL','BASEBALL','HOCKEY']
        all_mkts = self._list_all_markets(status=status, limit=100)
        if not all_mkts:
            all_mkts = self._list_all_markets(status=None, limit=100)
        out = []
        for m in all_mkts:
            title = (m.get("title") or "").upper()
            ticker = (m.get("ticker") or "").upper()
            if any(k in title or k in ticker for k in sports_keywords):
                out.append(m)
        # Fallback: return everything if filter yields nothing
        if not out:
            return all_mkts
        return out

    def get_orderbook(self, ticker: str) -> Dict:
        if not ticker:
            return {}
        try:
            r = requests.get(f"{self.api_url}/markets/{ticker}/orderbook", headers=self.headers, timeout=12)
            if r.status_code != 200:
                return {}
            return r.json() or {}
        except Exception:
            return {}

    # ----- Team extraction for sidebar loader -----
    @staticmethod
    def _normalize_pieces(text: str) -> List[str]:
        s = (text or "").strip()
        if not s:
            return []
        for sep in [" vs ", " @ ", " at ", " v ", "|", ":", "-", ",", "/"]:
            s = s.replace(sep, " ")
        s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.split(" ")

    @staticmethod
    def _clean_team_token(token: str) -> str:
        t = (token or "").strip()
        if not t:
            return ""
        if re.fullmatch(r"[0-9./:%]+", t):
            return ""
        LEAGUE_WORDS = {"NFL","NBA","MLB","NHL","UFC","NCAAF","NCAAB","SOCCER","GOLF","TENNIS","FOOTBALL",
                        "BASKETBALL","BASEBALL","HOCKEY","ODDS","TOTAL","SPREAD","YES","NO"}
        if t.upper() in LEAGUE_WORDS:
            return ""
        if len(t) <= 2 and t.upper() not in {"LA","NY","SF","OKC"}:
            return ""
        return t

    @staticmethod
    def _extract_teams_from_text(text: str) -> List[str]:
        parts = KalshiIntegrator._normalize_pieces(text)
        cleaned = []
        for p in parts:
            c = KalshiIntegrator._clean_team_token(p)
            if c:
                cleaned.append(c)
        joined = []
        current = []
        for w in cleaned:
            if re.match(r"^[A-Za-z][A-Za-z'.-]*$", w):
                current.append(w)
            else:
                if current:
                    joined.append(" ".join(current))
                    current = []
        if current:
            joined.append(" ".join(current))
        return joined or cleaned

    def get_all_teams(self, status: Optional[str] = "open") -> List[str]:
        markets = self.get_all_sports_markets(status=status)
        raw = set()
        for m in markets:
            for txt in (m.get("title",""), m.get("ticker",""), m.get("subtitle","")):
                for t in self._extract_teams_from_text(txt):
                    raw.add(t)
        norm = {}
        for t in raw:
            key = t.lower()
            if key not in norm:
                norm[key] = t
        return sorted(norm.values(), key=lambda x: x.lower())


# =========================
# Kalshi Validation (strict + partial)
# =========================
def validate_with_kalshi(kalshi_integrator, home_team: str, away_team: str,
                         side: str, sportsbook_prob: float, sport: str) -> Dict:
    """
    Validate with Kalshi prediction markets.

    Logic:
      1) STRICT: look for a market that mentions BOTH teams (rare on Kalshi).
      2) PARTIAL: if no strict match, prefer a futures-like market that mentions the bet team;
                  if still none, fall back to ANY market that mentions the bet team.
    """
    def normalize_team_name(team: str) -> List[str]:
        team_upper = (team or "").upper()
        if not team_upper:
            return []
        parts = re.split(r"\s+", team_upper.strip())
        variations: List[str] = [team_upper]

        # progressive prefixes
        current = []
        for p2 in parts:
            current.append(p2)
            variations.append(" ".join(current))

        # suffix-trimmed versions
        for i2 in range(len(parts) - 1, 0, -1):
            variations.append(" ".join(parts[:i2]))

        # heuristics: initials + 3-letter city/team
        initials = "".join(w[0] for w in parts if w and w[0].isalpha())
        if len(initials) >= 2:
            variations.append(initials)
        if len(parts) >= 1:
            variations.append(parts[0][:3])
        if len(parts) >= 2:
            variations.append(parts[-1][:3])
        if len(initials) >= 3:
            variations.append(initials[:3])

        # common abbreviations
        abbreviations = {
            'NEW YORK': ['NY', 'N.Y.', 'NYK', 'BKN'],
            'LOS ANGELES': ['LA', 'L.A.', 'LAL', 'LAC'],
            'GOLDEN STATE': ['GS', 'GSW'],
            'OKLAHOMA CITY': ['OKC'],
            'TORONTO': ['TOR'],
            'BROOKLYN': ['BKN', 'NETS'],
            'SAN FRANCISCO': ['SF', 'S.F.'],
            'WASHINGTON': ['WSH', 'WAS'],
        }
        for city, abbrevs in abbreviations.items():
            if team_upper.startswith(city) or city in team_upper:
                variations.extend(abbrevs)

        return list(dict.fromkeys(variations))

    def teams_match(bet_team: str, market_text: str) -> bool:
        bet_variations = normalize_team_name(bet_team)
        market_upper = (market_text or "").upper()
        for variation in bet_variations:
            if variation and (variation in market_upper or market_upper in variation):
                return True
        return False

    try:
        markets = kalshi_integrator.get_all_sports_markets(status="open") or []
        if not markets:
            markets = kalshi_integrator.get_all_sports_markets(status=None) or []

        bet_team = home_team if side == 'home' else away_team
        other_team = away_team if side == 'home' else home_team

        # STRICT: both teams
        strict_market = None
        for m in markets:
            title = m.get('title', '') or ''
            ticker = m.get('ticker', '') or ''
            subtitle = m.get('subtitle', '') or ''
            txt = f"{title} {ticker} {subtitle}"
            if teams_match(bet_team, txt) and teams_match(other_team, txt):
                strict_market = m
                break

        if strict_market is not None:
            ob = kalshi_integrator.get_orderbook(strict_market.get('ticker', '')) or {}
            yes_bids = ob.get('yes') or []
            no_bids  = ob.get('no') or []
            if not yes_bids and not no_bids:
                raise RuntimeError("Empty orderbook")
            yes_price = (yes_bids[0].get('price', 0) / 100.0) if yes_bids else 0.0
            no_price  = (no_bids[0].get('price', 0) / 100.0) if no_bids else 0.0
            kalshi_prob = max(yes_price, 1.0 - no_price)

            discrepancy = kalshi_prob - sportsbook_prob
            validation = (
                'confirms' if abs(discrepancy) >= 0.05 and kalshi_prob > sportsbook_prob else
                'strong_contradiction' if abs(discrepancy) >= 0.05 and kalshi_prob < sportsbook_prob else
                'agreement' if abs(discrepancy) < 0.05 else
                'market_confirm'
            )
            confidence_boost = clamp(abs(discrepancy) * 0.25, 0.0, 0.10)
            return {
                'kalshi_prob': kalshi_prob,
                'kalshi_available': True,
                'discrepancy': discrepancy,
                'validation': validation,
                'edge': discrepancy,
                'confidence_boost': confidence_boost,
                'market_ticker': strict_market.get('ticker'),
                'market_title': strict_market.get('title', 'Kalshi Market')
            }

        # PARTIAL: futures-like or any team-mentioning market
        futures_keywords = ['PLAYOFF', 'CHAMP', 'DIVISION', 'CONFERENCE', 'WIN TOTAL', 'WINS', 'TITLE', 'TO WIN', 'AWARD', 'MVP']
        partial_market = None
        fallback_market = None
        for m in markets:
            title = m.get('title', '') or ''
            ticker = m.get('ticker', '') or ''
            subtitle = m.get('subtitle', '') or ''
            up = f"{title} {ticker} {subtitle}".upper()
            if not teams_match(bet_team, up):
                continue
            if fallback_market is None:
                fallback_market = m
            if any(k in up for k in futures_keywords):
                partial_market = m
                break
        if partial_market is None:
            partial_market = fallback_market

        if partial_market is not None:
            ob = kalshi_integrator.get_orderbook(partial_market.get('ticker', '')) or {}
            yes_bids = ob.get('yes') or []
            no_bids  = ob.get('no') or []
            if not yes_bids and not no_bids:
                derived = 0.50
            else:
                yes_price = (yes_bids[0].get('price', 0) / 100.0) if yes_bids else 0.0
                no_price  = (no_bids[0].get('price', 0) / 100.0) if no_bids else 0.0
                derived = max(yes_price, 1.0 - no_price)

            strength = abs(derived - 0.50)
            confidence_boost = clamp(0.02 + strength * 0.08, 0.0, 0.06)
            return {
                'kalshi_prob': derived,
                'kalshi_available': True,
                'discrepancy': derived - sportsbook_prob,
                'validation': 'partial_support',
                'edge': (derived - 0.50) * 0.10,
                'confidence_boost': confidence_boost,
                'market_ticker': partial_market.get('ticker'),
                'market_title': partial_market.get('title', 'Kalshi (Partial)'),
            }

        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None
        }
    except Exception:
        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None
        }


# =========================
# Parlay Scoring
# =========================
def score_parlay(legs: List[Dict]) -> Dict[str, Any]:
    """
    Score a parlay combination using simple EV + confidence + edge, with Kalshi influence.
    """
    if not legs:
        return {'score': 0, 'confidence': 0, 'ev_ai': 0, 'kalshi_factor': 1.0, 'kalshi_legs': 0, 'kalshi_boost': 0}

    combined_prob = 1.0
    combined_conf = 1.0
    total_edge = 0.0
    kalshi_boost = 0
    kalshi_legs = 0

    for leg in legs:
        combined_prob *= leg.get('ai_prob', leg.get('p', 0.5))
        combined_conf *= leg.get('ai_confidence', 0.5)
        total_edge += leg.get('ai_edge', 0.0)

        # KALSHI INTEGRATION: Add Kalshi influence
        if 'kalshi_validation' in leg:
            kv = leg['kalshi_validation']
            if kv.get('kalshi_available'):
                kalshi_legs += 1
                kalshi_prob = kv.get('kalshi_prob', 0)
                sportsbook_prob = leg.get('p', 0)
                ai_prob = leg.get('ai_prob', sportsbook_prob)

                # Scale if only partial support
                weight = 0.5 if kv.get('validation') == 'partial_support' else 1.0

                if kalshi_prob > sportsbook_prob and ai_prob > sportsbook_prob:
                    kalshi_boost += int(15 * weight)
                elif kalshi_prob < sportsbook_prob and ai_prob < sportsbook_prob:
                    kalshi_boost -= int(10 * weight)
                elif abs(kalshi_prob - ai_prob) < 0.05:
                    kalshi_boost += int(10 * weight)
                elif abs(kalshi_prob - sportsbook_prob) < 0.03:
                    kalshi_boost += int(5 * weight)
                else:
                    kalshi_boost -= int(5 * weight)

    # Expected value (approx)
    ai_ev = combined_prob * 2.0 - 1.0  # toy EV vs even payout
    correlation_factor = len(set(leg['event_id'] for leg in legs)) / len(legs)

    # Kalshi factor (multiplier around 1.0)
    if kalshi_legs > 0:
        kalshi_factor = clamp(1.0 + (kalshi_boost / 100.0), 0.8, 1.2)
    else:
        kalshi_factor = 1.0

    ev_score = ai_ev * 100
    confidence_score = combined_conf * 50
    edge_score = total_edge * 150

    final_score = (ev_score + confidence_score + edge_score) * kalshi_factor * correlation_factor

    return {
        'score': final_score,
        'confidence': combined_conf,
        'ev_ai': ai_ev,
        'correlation_factor': correlation_factor,
        'kalshi_factor': kalshi_factor,
        'kalshi_legs': kalshi_legs,
        'kalshi_boost': kalshi_boost
    }


# =========================
# UI Helpers
# =========================
def render_kalshi_section(row: Dict[str, Any], combo_legs: List[Dict]):
    """Show Kalshi influence info and legs table."""
    st.markdown("### 📊 Kalshi Prediction Market Status:")
    kalshi_legs_with_data = row.get('kalshi_legs', 0)
    total_legs = len(combo_legs)
    if kalshi_legs_with_data > 0:
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Kalshi Legs", f"{kalshi_legs_with_data}/{total_legs}")
        with col_k2:
            st.metric("Kalshi Boost Points", f"{row.get('kalshi_boost', 0):+d}")
        with col_k3:
            st.metric("Score Multiplier", f"{row.get('kalshi_factor', 1.0):.2f}x")
        with col_k4:
            base_score = row['score'] / row.get('kalshi_factor', 1.0) if row.get('kalshi_factor', 1.0) != 0 else row['score']
            st.metric("Score Impact", f"{(row['score'] - base_score):+.1f} pts")
        st.info("✅ Kalshi validation applied (includes partial futures support where exact game markets are unavailable).")
    else:
        st.warning(f"⚠️ No Kalshi Data Available for this Parlay ({kalshi_legs_with_data}/{total_legs} legs).")

    # Legs table
    table_rows = []
    for j, leg in enumerate(combo_legs, 1):
        kalshi_display = "—"
        if 'kalshi_validation' in leg and leg['kalshi_validation'].get('kalshi_available'):
            kv = leg['kalshi_validation']
            kalshi_display = f"{kv.get('kalshi_prob', 0)*100:.1f}%"
            if kv.get('validation') == 'partial_support':
                kalshi_display = f"🟡 {kalshi_display}"
        table_rows.append({
            "Leg": j,
            "Type": leg.get("market", ""),
            "Selection": leg.get("label", ""),
            "Odds": f"{leg.get('d', 0):.3f}",
            "Market %": f"{leg.get('p', 0)*100:.1f}%",
            "AI % (final)": f"{leg.get('ai_prob', leg.get('p', 0))*100:.1f}%",
            "Kalshi": kalshi_display,
        })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="ParlayPicker + Kalshi (Full Clean)", layout="wide")

st.title("ParlayPicker — Full Clean (Kalshi Partial Validation)")

with st.sidebar:
    st.subheader("Connections")
    THEODDS_API_KEY = st.text_input("TheOddsAPI Key", value=THEODDS_API_KEY, type="password")
    KALSHI_API_KEY = st.text_input("Kalshi API Key", value=KALSHI_API_KEY, type="password")
    KALSHI_API_URL = st.text_input("Kalshi API URL", value=KALSHI_API_URL)
    THEOVER_API_KEY = st.text_input("theOver.ai API Key (optional)", value=THEOVER_API_KEY, type="password")

    # Initialize Kalshi
    if KALSHI_API_KEY and ("kalshi_integrator" not in st.session_state):
        st.session_state["kalshi_integrator"] = KalshiIntegrator(KALSHI_API_KEY, KALSHI_API_URL)

    st.markdown("---")
    st.subheader("Kalshi Teams Loader")
    if st.button("📥 Load Teams from Kalshi"):
        kalshi = st.session_state.get("kalshi_integrator")
        if not kalshi:
            st.error("Kalshi integrator not initialized.")
        else:
            with st.spinner("Fetching teams from Kalshi markets..."):
                teams = kalshi.get_all_teams(status="open")
                if not teams:
                    teams = kalshi.get_all_teams(status=None)
                if teams:
                    st.session_state["analysis_teams"] = teams
                    st.session_state["team_games"] = {t: [] for t in teams}
                    st.success(f"Loaded {len(teams)} teams from Kalshi.")
                else:
                    st.warning("No teams parsed from Kalshi (try again later).")

    with st.expander("Kalshi Debug"):
        if st.button("🔎 Test Kalshi Fetch"):
            kalshi = st.session_state.get("kalshi_integrator")
            if kalshi:
                mkts_open = kalshi.get_all_sports_markets(status="open")
                mkts_all = kalshi.get_all_sports_markets(status=None)
                st.write(f"Open sports markets: {len(mkts_open)} | All (no-status): {len(mkts_all)}")
                for m in (mkts_open or mkts_all)[:15]:
                    st.write({"ticker": m.get("ticker"), "title": m.get("title"), "subtitle": m.get("subtitle")})
            else:
                st.warning("Kalshi integrator not initialized.")

    st.markdown("---")
    st.subheader("Settings")
    st.caption("This demo uses home moneylines for simplicity.")


# Tabs
tab1, tab2 = st.tabs(["Best Bets", "Build Parlay"])

with tab1:
    st.header("Best Bets (TheOddsAPI)")
    sport_key = st.selectbox("Sport", ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"], index=0)
    if st.button("Fetch Games"):
        evs = get_theodds_games(sport_key=sport_key)
        st.session_state["last_events"] = evs
        st.success(f"Fetched {len(evs)} events from TheOddsAPI.")
        st.json((evs[:3] if evs else []))
    st.caption("Tip: after fetching, go to 'Build Parlay' to analyze.")


with tab2:
    st.header("Build Parlay")
    events = st.session_state.get("last_events") or get_theodds_games()
    legs = build_legs_from_theodds(events)

    kalshi = st.session_state.get("kalshi_integrator")
    if kalshi:
        for leg in legs:
            try:
                v = validate_with_kalshi(
                    kalshi_integrator=kalshi,
                    home_team=leg.get("home",""),
                    away_team=leg.get("away",""),
                    side="home",
                    sportsbook_prob=leg.get("p", 0.5),
                    sport=leg.get("sport","")
                )
                leg["kalshi_validation"] = v
                # Blend probabilities when Kalshi is available
                if v.get("kalshi_available"):
                    original_ai = leg["ai_prob"]
                    kalshi_prob = v.get("kalshi_prob", original_ai)
                    blended = original_ai * 0.50 + kalshi_prob * 0.30 + leg.get("p", 0.5) * 0.20
                    leg["ai_prob"] = blended
                    leg["ai_confidence"] = clamp(leg.get("ai_confidence", 0.6) + v.get("confidence_boost", 0.0), 0.0, 0.98)
            except Exception:
                leg["kalshi_validation"] = {"kalshi_available": False}

    st.write(f"Pulled {len(legs)} candidate legs.")
    if not legs:
        st.info("Fetch games first in Best Bets, or set your API keys.")
    else:
        # Build a few 2-leg parlay options
        combos = list(itertools.combinations(legs, 2))[:5]
        for idx, combo in enumerate(combos, 1):
            st.subheader(f"Parlay Option #{idx}")
            metrics = score_parlay(list(combo))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AI EV (approx)", f"{metrics['ev_ai']*100:.1f}%")
            with col2:
                st.metric("Score", f"{metrics['score']:.1f}")
            with col3:
                st.metric("Kalshi Factor", f"{metrics['kalshi_factor']:.2f}x")

            render_kalshi_section(metrics, list(combo))
            st.markdown("---")
