# ParlayDesk_AI_Enhanced.py - v9.1 FIXED
# AI-Enhanced parlay finder with sentiment analysis, ML predictions, and live market data
import os, io, json, itertools, re, copy
from html import escape
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import pytz

from app_core import APISportsFootballClient, RealSentimentAnalyzer, SentimentAnalyzer

# ============ HELPER FUNCTIONS ============
def american_to_decimal_safe(odds) -> float | None:
    """
    Safe American→Decimal conversion.
    Returns None for None/0/invalid odds in (-100, 100) or on parsing errors.
    """
    try:
        if odds is None:
            return None
        o = float(odds)
        if abs(o) < 100:
            return None
        if o >= 100:
            return 1.0 + o/100.0
        else:
            return 1.0 + 100.0/abs(o)
    except Exception:
        return None

APP_CFG: Dict[str, Any] = {
    "title": "ParlayDesk - AI-Enhanced Odds Finder",
    "sports_common": [
        "americanfootball_nfl","americanfootball_ncaaf",
        "basketball_nba","basketball_ncaab",
        "baseball_mlb","icehockey_nhl","mma_mixed_martial_arts",
        "soccer_epl","soccer_uefa_champs_league","tennis_atp_singles"
    ]
}

# Comprehensive mapping of Kalshi team abbreviations → canonical team names.
# The Kalshi markets often reference tickers like "NBA.LAL_GSW" or subtitles using
# short-hands. By centralizing these variations we can translate between
# sportsbook-style names ("Los Angeles Lakers") and Kalshi identifiers ("LAL").
# This dramatically increases the likelihood that we locate the correct Kalshi
# market when validating a parlay leg.
KALSHI_TEAM_ABBREVIATIONS: Dict[str, List[str]] = {
    # ========================= NFL =========================
    "ARIZONA CARDINALS": ["ARI", "ARZ", "AZ"],
    "ATLANTA FALCONS": ["ATL"],
    "BALTIMORE RAVENS": ["BAL"],
    "BUFFALO BILLS": ["BUF"],
    "CAROLINA PANTHERS": ["CAR", "CLT"],
    "CHICAGO BEARS": ["CHI", "CHB"],
    "CINCINNATI BENGALS": ["CIN", "CINC"],
    "CLEVELAND BROWNS": ["CLE"],
    "DALLAS COWBOYS": ["DAL"],
    "DENVER BRONCOS": ["DEN"],
    "DETROIT LIONS": ["DET"],
    "GREEN BAY PACKERS": ["GB", "GBP", "GBE"],
    "HOUSTON TEXANS": ["HOU", "HTX"],
    "INDIANAPOLIS COLTS": ["IND"],
    "JACKSONVILLE JAGUARS": ["JAX", "JAC"],
    "KANSAS CITY CHIEFS": ["KC", "KCC"],
    "LAS VEGAS RAIDERS": ["LV", "LVR"],
    "LOS ANGELES CHARGERS": ["LAC", "LA CHARGERS"],
    "LOS ANGELES RAMS": ["LAR", "LA RAMS"],
    "MIAMI DOLPHINS": ["MIA"],
    "MINNESOTA VIKINGS": ["MIN", "MINN"],
    "NEW ENGLAND PATRIOTS": ["NE", "NEP"],
    "NEW ORLEANS SAINTS": ["NO", "NOS"],
    "NEW YORK GIANTS": ["NYG", "NY GIANTS"],
    "NEW YORK JETS": ["NYJ", "NY JETS"],
    "PHILADELPHIA EAGLES": ["PHI", "PHL", "PHI EAGLES"],
    "PITTSBURGH STEELERS": ["PIT", "PITTSBURGH"],
    "SAN FRANCISCO 49ERS": ["SF", "SFO", "SF 49ERS"],
    "SEATTLE SEAHAWKS": ["SEA", "SEA HAWKS"],
    "TAMPA BAY BUCCANEERS": ["TB", "TBB"],
    "TENNESSEE TITANS": ["TEN", "TENN"],
    "WASHINGTON COMMANDERS": ["WAS", "WSH"],

    # ========================= NBA =========================
    "ATLANTA HAWKS": ["ATL"],
    "BOSTON CELTICS": ["BOS"],
    "BROOKLYN NETS": ["BKN", "BRK"],
    "CHARLOTTE HORNETS": ["CHA", "CHH", "CLT"],
    "CHICAGO BULLS": ["CHI"],
    "CLEVELAND CAVALIERS": ["CLE", "CAVS"],
    "DALLAS MAVERICKS": ["DAL", "MAVS"],
    "DENVER NUGGETS": ["DEN"],
    "DETROIT PISTONS": ["DET"],
    "GOLDEN STATE WARRIORS": ["GSW", "GS"],
    "HOUSTON ROCKETS": ["HOU"],
    "INDIANA PACERS": ["IND"],
    "LOS ANGELES CLIPPERS": ["LAC", "LA CLIPPERS"],
    "LOS ANGELES LAKERS": ["LAL", "LA LAKERS"],
    "MEMPHIS GRIZZLIES": ["MEM"],
    "MIAMI HEAT": ["MIA"],
    "MILWAUKEE BUCKS": ["MIL"],
    "MINNESOTA TIMBERWOLVES": ["MIN", "MINN"],
    "NEW ORLEANS PELICANS": ["NOP", "NO PELICANS"],
    "NEW YORK KNICKS": ["NYK", "NY KNICKS"],
    "OKLAHOMA CITY THUNDER": ["OKC"],
    "ORLANDO MAGIC": ["ORL"],
    "PHILADELPHIA 76ERS": ["PHI", "PHL", "SIXERS"],
    "PHOENIX SUNS": ["PHX"],
    "PORTLAND TRAIL BLAZERS": ["POR", "PTB", "PDX"],
    "SACRAMENTO KINGS": ["SAC"],
    "SAN ANTONIO SPURS": ["SAS", "SA SPURS"],
    "TORONTO RAPTORS": ["TOR"],
    "UTAH JAZZ": ["UTA"],
    "WASHINGTON WIZARDS": ["WAS", "WSH"],

    # ========================= MLB =========================
    "ARIZONA DIAMONDBACKS": ["ARI", "ARZ", "AZ"],
    "ATLANTA BRAVES": ["ATL"],
    "BALTIMORE ORIOLES": ["BAL"],
    "BOSTON RED SOX": ["BOS"],
    "CHICAGO CUBS": ["CHC"],
    "CHICAGO WHITE SOX": ["CWS", "CHW"],
    "CINCINNATI REDS": ["CIN", "CINC"],
    "CLEVELAND GUARDIANS": ["CLE", "CLV"],
    "COLORADO ROCKIES": ["COL"],
    "DETROIT TIGERS": ["DET"],
    "HOUSTON ASTROS": ["HOU"],
    "KANSAS CITY ROYALS": ["KC", "KCR"],
    "LOS ANGELES ANGELS": ["LAA", "LA ANGELS"],
    "LOS ANGELES DODGERS": ["LAD", "LA DODGERS"],
    "MIAMI MARLINS": ["MIA"],
    "MILWAUKEE BREWERS": ["MIL"],
    "MINNESOTA TWINS": ["MIN", "MINN"],
    "NEW YORK METS": ["NYM", "NY METS"],
    "NEW YORK YANKEES": ["NYY", "NY YANKEES"],
    "OAKLAND ATHLETICS": ["OAK"],
    "PHILADELPHIA PHILLIES": ["PHI", "PHL", "PHILS"],
    "PITTSBURGH PIRATES": ["PIT"],
    "SAN DIEGO PADRES": ["SD", "SDP"],
    "SAN FRANCISCO GIANTS": ["SF", "SFG"],
    "SEATTLE MARINERS": ["SEA"],
    "ST. LOUIS CARDINALS": ["STL", "SLC"],
    "TAMPA BAY RAYS": ["TB", "TBR"],
    "TEXAS RANGERS": ["TEX"],
    "TORONTO BLUE JAYS": ["TOR"],
    "WASHINGTON NATIONALS": ["WSH", "WAS"],

    # ========================= NHL =========================
    "ANAHEIM DUCKS": ["ANA"],
    "ARIZONA COYOTES": ["ARI", "ARZ", "AZ"],
    "BOSTON BRUINS": ["BOS"],
    "BUFFALO SABRES": ["BUF"],
    "CALGARY FLAMES": ["CGY"],
    "CAROLINA HURRICANES": ["CAR", "CLT"],
    "CHICAGO BLACKHAWKS": ["CHI"],
    "COLORADO AVALANCHE": ["COL"],
    "COLUMBUS BLUE JACKETS": ["CBJ"],
    "DALLAS STARS": ["DAL"],
    "DETROIT RED WINGS": ["DET"],
    "EDMONTON OILERS": ["EDM"],
    "FLORIDA PANTHERS": ["FLA"],
    "LOS ANGELES KINGS": ["LAK", "LA KINGS"],
    "MINNESOTA WILD": ["MIN", "MINN"],
    "MONTREAL CANADIENS": ["MTL"],
    "NASHVILLE PREDATORS": ["NSH"],
    "NEW JERSEY DEVILS": ["NJD"],
    "NEW YORK ISLANDERS": ["NYI"],
    "NEW YORK RANGERS": ["NYR"],
    "OTTAWA SENATORS": ["OTT"],
    "PHILADELPHIA FLYERS": ["PHI", "PHL"],
    "PITTSBURGH PENGUINS": ["PIT"],
    "SAN JOSE SHARKS": ["SJS"],
    "SEATTLE KRAKEN": ["SEA"],
    "ST. LOUIS BLUES": ["STL"],
    "TAMPA BAY LIGHTNING": ["TB", "TBL"],
    "TORONTO MAPLE LEAFS": ["TOR"],
    "VANCOUVER CANUCKS": ["VAN"],
    "VEGAS GOLDEN KNIGHTS": ["VGK", "VEGAS"],
    "WASHINGTON CAPITALS": ["WSH", "WAS"],
    "WINNIPEG JETS": ["WPG"],
}

# Map teams back to their primary league so we can build league-aware fallbacks
KALSHI_LEAGUE_TEAM_SETS: Dict[str, List[str]] = {
    "NFL": [
        "ARIZONA CARDINALS", "ATLANTA FALCONS", "BALTIMORE RAVENS", "BUFFALO BILLS",
        "CAROLINA PANTHERS", "CHICAGO BEARS", "CINCINNATI BENGALS", "CLEVELAND BROWNS",
        "DALLAS COWBOYS", "DENVER BRONCOS", "DETROIT LIONS", "GREEN BAY PACKERS",
        "HOUSTON TEXANS", "INDIANAPOLIS COLTS", "JACKSONVILLE JAGUARS", "KANSAS CITY CHIEFS",
        "LAS VEGAS RAIDERS", "LOS ANGELES CHARGERS", "LOS ANGELES RAMS", "MIAMI DOLPHINS",
        "MINNESOTA VIKINGS", "NEW ENGLAND PATRIOTS", "NEW ORLEANS SAINTS", "NEW YORK GIANTS",
        "NEW YORK JETS", "PHILADELPHIA EAGLES", "PITTSBURGH STEELERS", "SAN FRANCISCO 49ERS",
        "SEATTLE SEAHAWKS", "TAMPA BAY BUCCANEERS", "TENNESSEE TITANS", "WASHINGTON COMMANDERS"
    ],
    "NBA": [
        "ATLANTA HAWKS", "BOSTON CELTICS", "BROOKLYN NETS", "CHARLOTTE HORNETS",
        "CHICAGO BULLS", "CLEVELAND CAVALIERS", "DALLAS MAVERICKS", "DENVER NUGGETS",
        "DETROIT PISTONS", "GOLDEN STATE WARRIORS", "HOUSTON ROCKETS", "INDIANA PACERS",
        "LOS ANGELES CLIPPERS", "LOS ANGELES LAKERS", "MEMPHIS GRIZZLIES", "MIAMI HEAT",
        "MILWAUKEE BUCKS", "MINNESOTA TIMBERWOLVES", "NEW ORLEANS PELICANS", "NEW YORK KNICKS",
        "OKLAHOMA CITY THUNDER", "ORLANDO MAGIC", "PHILADELPHIA 76ERS", "PHOENIX SUNS",
        "PORTLAND TRAIL BLAZERS", "SACRAMENTO KINGS", "SAN ANTONIO SPURS", "TORONTO RAPTORS",
        "UTAH JAZZ", "WASHINGTON WIZARDS"
    ],
    "MLB": [
        "ARIZONA DIAMONDBACKS", "ATLANTA BRAVES", "BALTIMORE ORIOLES", "BOSTON RED SOX",
        "CHICAGO CUBS", "CHICAGO WHITE SOX", "CINCINNATI REDS", "CLEVELAND GUARDIANS",
        "COLORADO ROCKIES", "DETROIT TIGERS", "HOUSTON ASTROS", "KANSAS CITY ROYALS",
        "LOS ANGELES ANGELS", "LOS ANGELES DODGERS", "MIAMI MARLINS", "MILWAUKEE BREWERS",
        "MINNESOTA TWINS", "NEW YORK METS", "NEW YORK YANKEES", "OAKLAND ATHLETICS",
        "PHILADELPHIA PHILLIES", "PITTSBURGH PIRATES", "SAN DIEGO PADRES", "SAN FRANCISCO GIANTS",
        "SEATTLE MARINERS", "ST. LOUIS CARDINALS", "TAMPA BAY RAYS", "TEXAS RANGERS",
        "TORONTO BLUE JAYS", "WASHINGTON NATIONALS"
    ],
    "NHL": [
        "ANAHEIM DUCKS", "ARIZONA COYOTES", "BOSTON BRUINS", "BUFFALO SABRES",
        "CALGARY FLAMES", "CAROLINA HURRICANES", "CHICAGO BLACKHAWKS", "COLORADO AVALANCHE",
        "COLUMBUS BLUE JACKETS", "DALLAS STARS", "DETROIT RED WINGS", "EDMONTON OILERS",
        "FLORIDA PANTHERS", "LOS ANGELES KINGS", "MINNESOTA WILD", "MONTREAL CANADIENS",
        "NASHVILLE PREDATORS", "NEW JERSEY DEVILS", "NEW YORK ISLANDERS", "NEW YORK RANGERS",
        "OTTAWA SENATORS", "PHILADELPHIA FLYERS", "PITTSBURGH PENGUINS", "SAN JOSE SHARKS",
        "SEATTLE KRAKEN", "ST. LOUIS BLUES", "TAMPA BAY LIGHTNING", "TORONTO MAPLE LEAFS",
        "VANCOUVER CANUCKS", "VEGAS GOLDEN KNIGHTS", "WASHINGTON CAPITALS", "WINNIPEG JETS"
    ],
}

KALSHI_TEAM_LEAGUE_MAP: Dict[str, str] = {
    team: league
    for league, teams in KALSHI_LEAGUE_TEAM_SETS.items()
    for team in teams
}

SPORT_KEY_TO_LEAGUE: Dict[str, str] = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "mma_mixed_martial_arts": "MMA",
    "soccer_epl": "EPL",
    "soccer_uefa_champs_league": "UEFA",
    "tennis_atp_singles": "TENNIS",
}

# ============ REAL SENTIMENT ANALYSIS ENGINE ============
# (moved to app_core.sentiment so it can be reused without importing the
# Streamlit UI. RealSentimentAnalyzer and SentimentAnalyzer are imported above.)

# ============ ML PREDICTION ENGINE ============
class MLPredictor:
    """Machine Learning prediction engine for game outcomes"""
    
    def __init__(self):
        self.model_loaded = False
        self.feature_importance = {}
    
    def predict_game_outcome(self, home_team: str, away_team: str, 
                            home_odds: float, away_odds: float,
                            sentiment_home: float, sentiment_away: float) -> Dict[str, float]:
        """
        Predict game outcome using ensemble ML approach
        Returns adjusted probabilities for home/away
        """
        
        # Calculate base probabilities from odds
        home_implied = self._odds_to_prob(home_odds)
        away_implied = self._odds_to_prob(away_odds)
        
        # Feature engineering
        features = {
            'home_odds': home_odds,
            'away_odds': away_odds,
            'odds_differential': home_odds - away_odds,
            'sentiment_home': sentiment_home,
            'sentiment_away': sentiment_away,
            'sentiment_diff': sentiment_home - sentiment_away,
            'market_efficiency': abs(home_implied + away_implied - 1.0)
        }
        
        # ML Adjustment (in production, this would use trained XGBoost/LightGBM)
        # UPDATED: Sentiment now has real impact (40% vs 15% before)
        sentiment_weight = 0.40  # Increased from 0.15 to 0.40
        market_weight = 0.60     # Decreased from 0.85 to 0.60
        
        # Adjust probabilities based on sentiment
        sentiment_adjustment = (sentiment_home - sentiment_away) * sentiment_weight
        
        home_adjusted = home_implied * market_weight + (0.5 + sentiment_adjustment) * (1 - market_weight)
        away_adjusted = away_implied * market_weight + (0.5 - sentiment_adjustment) * (1 - market_weight)
        
        # Normalize to sum to 1
        total = home_adjusted + away_adjusted
        home_adjusted /= total
        away_adjusted /= total
        
        # Calculate confidence based on agreement between sources
        confidence = self._calculate_confidence(features, home_adjusted, home_implied)
        
        return {
            'home_prob': home_adjusted,
            'away_prob': away_adjusted,
            'confidence': confidence,
            'edge': abs(home_adjusted - home_implied),
            'recommendation': 'home' if home_adjusted > away_adjusted else 'away'
        }
    
    def _odds_to_prob(self, american_odds: float) -> float:
        """Convert American odds to probability"""
        if american_odds > 0:
            return 100.0 / (american_odds + 100.0)
        else:
            return abs(american_odds) / (abs(american_odds) + 100.0)
    
    def _calculate_confidence(self, features: dict, ml_prob: float, market_prob: float) -> float:
        """Calculate prediction confidence score"""
        # Higher confidence when:
        # 1. Sentiment is strong and clear
        # 2. ML and market agree
        # 3. Market efficiency is high
        
        sentiment_strength = abs(features['sentiment_diff'])
        ml_market_agreement = 1.0 - abs(ml_prob - market_prob)
        market_eff = 1.0 - features['market_efficiency']
        
        confidence = (sentiment_strength * 0.3 + 
                     ml_market_agreement * 0.4 + 
                     market_eff * 0.3)
        
        return min(max(confidence, 0.3), 0.95)  # Clamp between 30% and 95%

# ============ AI PARLAY OPTIMIZER ============
class AIOptimizer:
    """Optimizes parlay selection using AI insights"""
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, ml_predictor: MLPredictor):
        self.sentiment = sentiment_analyzer
        self.ml = ml_predictor
    
    def score_parlay(self, legs: List[Dict]) -> Dict[str, float]:
        """
        Score a parlay combination using AI
        Higher scores = better opportunity
        NOW INCLUDES KALSHI VALIDATION
        """
        if not legs:
            return {'score': 0, 'confidence': 0}
        
        # Calculate combined probability (AI-adjusted)
        combined_prob = 1.0
        combined_confidence = 1.0
        total_edge = 0
        kalshi_boost = 0
        kalshi_legs = 0
        apisports_boost = 0
        apisports_legs = 0
        
        for leg in legs:
            combined_prob *= leg.get('ai_prob', leg['p'])
            combined_confidence *= leg.get('ai_confidence', 0.5)
            total_edge += leg.get('ai_edge', 0)
            
            # KALSHI INTEGRATION: Add Kalshi influence
            if 'kalshi_validation' in leg:
                kv = leg['kalshi_validation']
                if kv.get('kalshi_available'):
                    kalshi_legs += 1
                    # Kalshi provides additional probability estimate
                    kalshi_prob = kv.get('kalshi_prob', 0)
                    sportsbook_prob = leg.get('p', 0)
                    data_source = kv.get('data_source', 'kalshi')
                    source_weight = 0.6 if data_source and 'synthetic' in data_source else 1.0

                    # If Kalshi and AI both disagree with sportsbook in same direction
                    # that's a strong signal
                    ai_prob = leg.get('ai_prob', sportsbook_prob)

                    if kalshi_prob > sportsbook_prob and ai_prob > sportsbook_prob:
                        # Both Kalshi and AI see value
                        kalshi_boost += 15 * source_weight  # Strong boost
                    elif kalshi_prob < sportsbook_prob and ai_prob < sportsbook_prob:
                        # Both Kalshi and AI skeptical
                        kalshi_boost -= 10 * source_weight  # Penalty
                    elif abs(kalshi_prob - ai_prob) < 0.05:
                        # Kalshi and AI agree (regardless of sportsbook)
                        kalshi_boost += 10 * source_weight  # Agreement boost
                    elif abs(kalshi_prob - sportsbook_prob) < 0.03:
                        # Kalshi confirms market
                        kalshi_boost += 5 * source_weight  # Small boost for confirmation
                    else:
                        # Kalshi contradicts both AI and market
                        kalshi_boost -= 5 * source_weight  # Small penalty for confusion

            apisports_info = leg.get('apisports')
            if apisports_info:
                apisports_legs += 1
                trend = apisports_info.get('trend')
                if trend == 'hot':
                    apisports_boost += 5
                elif trend == 'cold':
                    apisports_boost -= 5

        # Calculate combined decimal odds
        combined_odds = legs[0]['d']
        for leg in legs[1:]:
            combined_odds *= leg['d']
        
        # AI-enhanced EV
        ai_ev = (combined_prob * combined_odds) - 1.0
        
        # Correlation penalty (same-game parlays are correlated)
        unique_games = len(set(leg['event_id'] for leg in legs))
        correlation_factor = unique_games / len(legs)
        
        # Kalshi validation factor (0.8 to 1.2 multiplier)
        if kalshi_legs > 0:
            kalshi_factor = 1.0 + (kalshi_boost / 100)
            kalshi_factor = max(0.8, min(1.2, kalshi_factor))  # Clamp between 0.8 and 1.2
        else:
            kalshi_factor = 1.0  # No Kalshi data = neutral
        
        # Final score components with KALSHI integration
        ev_score = ai_ev * 100  # EV contribution
        confidence_score = combined_confidence * 50  # Confidence contribution
        edge_score = total_edge * 150  # Edge is most important
        
        # Calculate base score
        base_score = (edge_score * 0.45 +      # 45% edge
                     ev_score * 0.30 +          # 30% EV
                     confidence_score * 0.25)    # 25% confidence
        
        # Apply Kalshi factor, correlation factor, and live data adjustments
        live_data_factor = 1.0
        if apisports_legs:
            live_data_factor += apisports_boost / 100.0
            live_data_factor = max(0.9, min(1.1, live_data_factor))

        final_score = base_score * correlation_factor * kalshi_factor * live_data_factor

        return {
            'score': final_score,
            'ai_ev': ai_ev,
            'confidence': combined_confidence,
            'edge': total_edge,
            'correlation_factor': correlation_factor,
            'kalshi_factor': kalshi_factor,
            'kalshi_legs': kalshi_legs,
            'kalshi_boost': kalshi_boost,
            'apisports_factor': live_data_factor,
            'apisports_legs': apisports_legs,
            'apisports_boost': apisports_boost
        }

# ============ KALSHI INTEGRATION ============

class KalshiIntegrator:
    """Integrates Kalshi prediction market odds and analysis"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.elections.kalshi.com/trade-api/v2"

        # Use demo for testing, production for live
        self.api_url = self.base_url if self.api_key else self.demo_url

        self.headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Synthetic fallback cache when Kalshi API is unavailable (e.g., network blocks)
        self._using_synthetic_data = False
        self._synthetic_markets: List[Dict[str, Any]] = []
        self._synthetic_orderbooks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._synthetic_market_by_team: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None

    # -------------------- Synthetic helpers --------------------
    def _synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Generate a deterministic synthetic probability for a team."""

        team_upper = team.upper()
        base_prob = sportsbook_prob if sportsbook_prob is not None else 0.52

        # Deterministic offset based on team characters (stable pseudo-random)
        ordinal_sum = sum(ord(c) for c in team_upper if c.isalpha())
        offset = ((ordinal_sum % 21) - 10) / 200.0  # -0.05 .. +0.05

        league = KALSHI_TEAM_LEAGUE_MAP.get(team_upper)
        if sport_key and sport_key in SPORT_KEY_TO_LEAGUE and not league:
            league = SPORT_KEY_TO_LEAGUE[sport_key]

        league_bias = {
            "NFL": 0.015,
            "NBA": 0.010,
            "MLB": 0.005,
            "NHL": 0.005,
        }.get(league, 0.0)

        synthetic = base_prob + offset + league_bias
        return max(0.05, min(0.95, synthetic))

    def _synthetic_ticker_for_team(self, team: str, league: Optional[str]) -> str:
        team_key = re.sub(r"[^A-Z0-9]", "", team.upper())
        league_key = league or "SPORTS"
        return f"SIM.{league_key}.{team_key[:8]}"

    def _ensure_synthetic_data(self) -> None:
        if self._synthetic_markets:
            return

        # Build synthetic markets for every team we know about so UI has coverage
        now = datetime.utcnow()
        expiry = (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for team, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            league = KALSHI_TEAM_LEAGUE_MAP.get(team)
            ticker = self._synthetic_ticker_for_team(team, league)
            prob = self._synthetic_probability(team)
            price = int(round(prob * 100))

            market = {
                "ticker": ticker,
                "title": f"{team.title()} confidence (synthetic Kalshi)",
                "subtitle": "Synthetic fallback market generated locally",
                "series_ticker": "SPORTS",
                "status": "open",
                "close_time": expiry,
                "league": league,
                "synthetic": True,
                "team": team,
                "abbreviation": abbrs[0] if abbrs else None,
            }

            self._synthetic_markets.append(market)
            self._synthetic_orderbooks[ticker] = {
                "yes": [{"price": price, "contracts": 100}],
                "no": [{"price": 100 - price, "contracts": 100}],
            }
            self._synthetic_market_by_team[team] = market

    def using_synthetic_data(self) -> bool:
        return self._using_synthetic_data

    def get_synthetic_market_for_team(self, team: str) -> Optional[Dict[str, Any]]:
        self._ensure_synthetic_data()
        return self._synthetic_market_by_team.get(team.upper())

    def synthetic_probability(self, team: str, sport_key: Optional[str] = None,
                               sportsbook_prob: Optional[float] = None) -> float:
        """Public helper to compute synthetic probabilities for validation."""
        self._ensure_synthetic_data()
        return self._synthetic_probability(team, sport_key, sportsbook_prob)
    
    def get_markets(self, category: str = "sports", status: str = "open") -> List[Dict]:
        """
        Fetch available Kalshi markets

        Args:
            category: 'sports', 'politics', 'economics', etc.
            status: 'open', 'closed', 'settled'
        
        Returns:
            List of market dictionaries
        """
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            return copy.deepcopy(self._synthetic_markets)

        try:
            endpoint = f"{self.api_url}/markets"
            params = {
                "limit": 100,
                "status": status
            }

            if category:
                params["series_ticker"] = category.upper()

            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                if markets:
                    self.last_error = None
                    return markets
                else:
                    self.last_error = "Kalshi API returned no markets"
            else:
                self.last_error = f"Kalshi API responded with status {response.status_code}"

        except Exception as e:
            self.last_error = str(e)
            st.warning(f"Error fetching Kalshi markets: {str(e)}")

        # Fallback to synthetic data when API fails or returns nothing
        self._using_synthetic_data = True
        self._ensure_synthetic_data()
        return copy.deepcopy(self._synthetic_markets)
    
    def get_sports_markets(self) -> List[Dict]:
        """Get all active sports betting markets"""
        all_markets = self.get_markets()
        
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
    
    def get_market_details(self, market_ticker: str) -> Dict:
        """Get detailed information about a specific market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            market = next((m for m in self._synthetic_markets if m.get('ticker') == market_ticker), None)
            return copy.deepcopy(market) if market else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("market", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching market details: {str(e)}")
            return {}

    def get_orderbook(self, market_ticker: str) -> Dict:
        """Get current orderbook (bids/asks) for a market"""
        if self._using_synthetic_data:
            self._ensure_synthetic_data()
            orderbook = self._synthetic_orderbooks.get(market_ticker)
            return copy.deepcopy(orderbook) if orderbook else {}

        try:
            endpoint = f"{self.api_url}/markets/{market_ticker}/orderbook"
            response = requests.get(endpoint, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json().get("orderbook", {})
            else:
                return {}

        except Exception as e:
            st.warning(f"Error fetching orderbook: {str(e)}")
            return {}
    
    def compare_with_sportsbook(self, kalshi_market: Dict, sportsbook_odds: Dict) -> Dict:
        """
        Compare Kalshi prediction market with traditional sportsbook
        
        This finds arbitrage and value opportunities
        """
        kalshi_yes_price = kalshi_market.get('yes_bid', 0) / 100  # Convert cents to probability
        kalshi_no_price = kalshi_market.get('no_bid', 0) / 100
        
        # Get implied probability from sportsbook
        sb_odds = sportsbook_odds.get('price')
        if sb_odds:
            sb_prob = implied_p_from_american(sb_odds)
        else:
            sb_prob = None
        
        # Calculate discrepancy
        if sb_prob and kalshi_yes_price:
            discrepancy = abs(kalshi_yes_price - sb_prob)
            
            # Determine which side is better value
            if kalshi_yes_price < sb_prob - 0.05:  # Kalshi underpricing by 5%+
                recommendation = "🟢 BUY YES on Kalshi (underpriced vs sportsbook)"
                edge = sb_prob - kalshi_yes_price
            elif kalshi_yes_price > sb_prob + 0.05:  # Kalshi overpricing by 5%+
                recommendation = "🟢 BUY NO on Kalshi (or take sportsbook)"
                edge = kalshi_yes_price - sb_prob
            else:
                recommendation = "🟡 Prices aligned (no significant edge)"
                edge = discrepancy
            
            return {
                'kalshi_prob': kalshi_yes_price,
                'sportsbook_prob': sb_prob,
                'discrepancy': discrepancy,
                'edge': edge,
                'recommendation': recommendation,
                'has_arbitrage': discrepancy > 0.10  # 10%+ difference
            }
        
        return {
            'kalshi_prob': kalshi_yes_price,
            'sportsbook_prob': sb_prob,
            'discrepancy': 0,
            'edge': 0,
            'recommendation': '⚪ Insufficient data for comparison',
            'has_arbitrage': False
        }
    
    def find_arbitrage_opportunities(self, kalshi_markets: List[Dict], 
                                     sportsbook_events: List[Dict]) -> List[Dict]:
        """
        Find arbitrage opportunities between Kalshi and traditional sportsbooks
        
        Returns list of arbitrage opportunities with expected profit
        """
        arbitrage_opps = []
        
        for kalshi_market in kalshi_markets:
            title = kalshi_market.get('title', '')
            
            # Try to match with sportsbook events
            for sb_event in sportsbook_events:
                # Simple matching logic (can be enhanced)
                home_team = sb_event.get('home_team', '')
                away_team = sb_event.get('away_team', '')
                
                if home_team in title or away_team in title:
                    comparison = self.compare_with_sportsbook(kalshi_market, sb_event)
                    
                    if comparison['has_arbitrage']:
                        arbitrage_opps.append({
                            'kalshi_market': title,
                            'kalshi_ticker': kalshi_market.get('ticker'),
                            'sportsbook_game': f"{away_team} @ {home_team}",
                            'comparison': comparison
                        })
        
        return arbitrage_opps
    
    def analyze_kalshi_market(self, market: Dict, sentiment_score: float = 0, 
                             ml_probability: float = None) -> Dict:
        """
        Comprehensive analysis of a Kalshi market
        
        Combines:
        - Kalshi orderbook data
        - Sentiment analysis
        - ML predictions
        - Value assessment
        """
        yes_bid = market.get('yes_bid', 0) / 100
        yes_ask = market.get('yes_ask', 100) / 100
        no_bid = market.get('no_bid', 0) / 100
        no_ask = market.get('no_ask', 100) / 100
        
        volume = market.get('volume', 0)
        open_interest = market.get('open_interest', 0)
        
        # Market efficiency (tight spread = efficient)
        yes_spread = yes_ask - yes_bid
        no_spread = no_ask - no_bid
        avg_spread = (yes_spread + no_spread) / 2
        
        efficiency = 1 - avg_spread  # Higher = more efficient
        
        # Compare with AI prediction
        kalshi_implied = yes_bid  # Using bid as market consensus
        
        if ml_probability:
            ai_edge = ml_probability - kalshi_implied
            
            if ai_edge > 0.10:
                ai_recommendation = f"🟢 STRONG BUY YES - AI sees {ai_edge*100:.1f}% edge"
            elif ai_edge < -0.10:
                ai_recommendation = f"🟢 STRONG BUY NO - AI sees {abs(ai_edge)*100:.1f}% edge"
            elif abs(ai_edge) < 0.05:
                ai_recommendation = "🟡 FAIR PRICE - AI agrees with market"
            else:
                ai_recommendation = f"🟡 SLIGHT EDGE - AI sees {ai_edge*100:.1f}% edge"
        else:
            ai_edge = 0
            ai_recommendation = "⚪ No AI prediction available"
        
        # Sentiment alignment
        if sentiment_score > 0.3 and kalshi_implied < 0.6:
            sentiment_signal = "🟢 Positive sentiment + underpriced = BUY YES"
        elif sentiment_score < -0.3 and kalshi_implied > 0.4:
            sentiment_signal = "🟢 Negative sentiment + overpriced = BUY NO"
        else:
            sentiment_signal = "🟡 Sentiment neutral or priced in"
        
        # Liquidity assessment
        if volume > 1000 and open_interest > 500:
            liquidity = "🟢 High liquidity - easy to enter/exit"
        elif volume > 100 and open_interest > 50:
            liquidity = "🟡 Moderate liquidity - tradeable"
        else:
            liquidity = "🔴 Low liquidity - be cautious"
        
        return {
            'kalshi_probability': kalshi_implied,
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'spread': avg_spread,
            'efficiency': efficiency,
            'volume': volume,
            'open_interest': open_interest,
            'liquidity': liquidity,
            'ai_edge': ai_edge,
            'ai_recommendation': ai_recommendation,
            'sentiment_score': sentiment_score,
            'sentiment_signal': sentiment_signal,
            'overall_score': self._calculate_overall_score(
                ai_edge, sentiment_score, efficiency, volume
            )
        }
    
    def _calculate_overall_score(self, ai_edge: float, sentiment: float, 
                                 efficiency: float, volume: int) -> float:
        """Calculate overall opportunity score 0-100"""
        # Weight different factors
        edge_score = min(abs(ai_edge) * 200, 50)  # Max 50 points
        sentiment_score = min(abs(sentiment) * 50, 20)  # Max 20 points
        efficiency_score = efficiency * 15  # Max 15 points
        liquidity_score = min(volume / 100, 15)  # Max 15 points
        
        total = edge_score + sentiment_score + efficiency_score + liquidity_score
        return min(total, 100)
    
    def get_best_opportunities(self, min_score: float = 60) -> List[Dict]:
        """
        Find the best Kalshi betting opportunities
        
        Returns markets with high overall scores
        """
        markets = self.get_sports_markets()
        opportunities = []
        
        for market in markets:
            # Get orderbook
            ticker = market.get('ticker')
            orderbook = self.get_orderbook(ticker)
            
            if orderbook:
                # Enhance market data with orderbook
                market['yes_bid'] = orderbook.get('yes', [{}])[0].get('price', 0)
                market['yes_ask'] = orderbook.get('yes', [{}])[-1].get('price', 100)
                market['no_bid'] = orderbook.get('no', [{}])[0].get('price', 0)
                market['no_ask'] = orderbook.get('no', [{}])[-1].get('price', 100)
                
                # Analyze
                analysis = self.analyze_kalshi_market(market)
                
                if analysis['overall_score'] >= min_score:
                    opportunities.append({
                        'market': market,
                        'analysis': analysis
                    })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['analysis']['overall_score'], reverse=True)
        
        return opportunities
    
    def calculate_kelly_for_kalshi(self, kalshi_prob: float, ai_prob: float, 
                                   bankroll: float = 1000) -> Dict:
        """
        Calculate Kelly Criterion for Kalshi market
        
        Kalshi uses probability pricing (0-100 cents), not traditional odds
        """
        if ai_prob <= kalshi_prob:
            return {
                'kelly_percentage': 0,
                'recommended_stake': 0,
                'expected_value': 0,
                'recommendation': '🔴 NO EDGE - AI probability not better than Kalshi price'
            }
        
        # Edge calculation for binary market
        edge = ai_prob - kalshi_prob
        
        # Kelly formula for binary outcome: f = edge / (1 - kalshi_prob)
        if kalshi_prob < 0.95:  # Avoid division issues
            kelly_fraction = edge / (1 - kalshi_prob)
        else:
            kelly_fraction = 0
        
        # Conservative Kelly (0.25x)
        kelly_fraction = kelly_fraction * 0.25
        kelly_fraction = max(0, min(kelly_fraction, 0.10))  # Cap at 10%
        
        kelly_percentage = kelly_fraction * 100
        recommended_stake = bankroll * kelly_fraction
        
        # Expected value
        expected_payout = ai_prob * (1 / kalshi_prob)
        expected_value = expected_payout - 1
        
        if kelly_percentage > 3:
            risk = "High"
        elif kelly_percentage > 1:
            risk = "Medium"
        else:
            risk = "Low"
        
        return {
            'kelly_percentage': kelly_percentage,
            'recommended_stake': recommended_stake,
            'expected_value': expected_value,
            'edge_percentage': edge * 100,
            'risk_level': risk,
            'recommendation': f"🟢 BET {kelly_percentage:.1f}% (${recommended_stake:.0f}) | Edge: {edge*100:+.1f}%"
        }


# ============ ADVANCED AI MODULES ============

# 1. SHARP MONEY DETECTOR
class SharpMoneyDetector:
    """Detects sharp money movement and line steam"""
    
    def __init__(self):
        self.line_history = {}  # Track line movements
        self.steam_threshold = 0.5  # Half point move = steam
    
    def analyze_line_movement(self, game_id: str, current_line: float, 
                              opening_line: float, public_bet_pct: float = None) -> Dict:
        """
        Analyze if line movement suggests sharp money
        
        Returns:
            'movement': Points line moved
            'direction': 'with_public' or 'reverse' or 'neutral'
            'is_steam': True if sharp action detected
            'confidence': How confident we are (0-1)
        """
        movement = current_line - opening_line
        
        # Reverse line movement = sharp money indicator
        # (Line moves opposite to where public is betting)
        is_reverse = False
        if public_bet_pct:
            if public_bet_pct > 65 and movement < -0.3:  # Public on favorite, line moves away
                is_reverse = True
            elif public_bet_pct < 35 and movement > 0.3:  # Public on underdog, line moves other way
                is_reverse = True
        
        # Steam = sudden sharp move (0.5+ points)
        is_steam = abs(movement) >= self.steam_threshold
        
        # Confidence based on movement size and reverse nature
        confidence = min(abs(movement) / 3.0, 0.95)  # Max 95% confidence
        if is_reverse:
            confidence = min(confidence + 0.2, 0.95)  # Boost confidence for reverse moves
        
        direction = 'reverse' if is_reverse else ('with_public' if public_bet_pct else 'neutral')
        
        return {
            'movement': movement,
            'direction': direction,
            'is_steam': is_steam,
            'is_reverse': is_reverse,
            'confidence': confidence,
            'recommendation': self._get_recommendation(movement, is_reverse, is_steam)
        }
    
    def _get_recommendation(self, movement: float, is_reverse: bool, is_steam: bool) -> str:
        """Generate betting recommendation based on sharp action"""
        if is_reverse and abs(movement) > 1.0:
            return "🟢 STRONG SHARP ACTION - Follow the line movement"
        elif is_steam and abs(movement) > 0.75:
            return "🟡 STEAM DETECTED - Sharp bettors moving line"
        elif is_reverse:
            return "🟢 REVERSE LINE MOVEMENT - Fade the public"
        elif abs(movement) < 0.3:
            return "⚪ NO SIGNIFICANT MOVEMENT"
        else:
            return "🟡 MODERATE MOVEMENT - Monitor closely"

# 2. PLAYER IMPACT ANALYZER
class PlayerImpactAnalyzer:
    """Analyzes impact of injuries and player absence"""
    
    def __init__(self):
        # Historical impact data (simplified - in production, use real database)
        self.position_weights = {
            'QB': 0.35,   # Quarterback most important in NFL
            'RB': 0.12,
            'WR': 0.10,
            'TE': 0.08,
            'OL': 0.08,
            'DL': 0.10,
            'LB': 0.08,
            'DB': 0.09,
            # NBA
            'PG': 0.25,
            'SG': 0.20,
            'SF': 0.20,
            'PF': 0.18,
            'C': 0.17
        }
    
    def analyze_injury_impact(self, team: str, injured_players: List[Dict], 
                              team_odds_before: float) -> Dict:
        """
        Calculate how injuries should affect team performance
        
        Args:
            injured_players: [{'name': 'Patrick Mahomes', 'position': 'QB', 'status': 'OUT'}]
        
        Returns:
            'total_impact': -0.15 means team is 15% worse
            'adjusted_odds': What odds should be
            'confidence': How sure we are
        """
        total_impact = 0.0
        key_players_out = []
        
        for player in injured_players:
            position = player.get('position', '').upper()
            status = player.get('status', '').upper()
            
            # Weight impact by position and injury severity
            if status == 'OUT':
                impact = self.position_weights.get(position, 0.05)
                total_impact += impact
                key_players_out.append(player['name'])
            elif status == 'DOUBTFUL':
                impact = self.position_weights.get(position, 0.05) * 0.75
                total_impact += impact
            elif status == 'QUESTIONABLE':
                impact = self.position_weights.get(position, 0.05) * 0.30
                total_impact += impact
        
        # Adjust odds based on impact
        # Negative impact = odds get worse (more positive/less negative)
        if team_odds_before:
            prob_before = implied_p_from_american(team_odds_before)
            prob_adjusted = prob_before * (1 - total_impact)
            
            # Convert back to American odds
            if prob_adjusted > 0.5:
                adjusted_odds = -100 * (prob_adjusted / (1 - prob_adjusted))
            else:
                adjusted_odds = 100 * ((1 - prob_adjusted) / prob_adjusted)
        else:
            adjusted_odds = None
        
        confidence = min(total_impact * 2, 0.9)  # More injuries = more confident
        
        recommendation = self._get_injury_recommendation(total_impact, key_players_out)
        
        return {
            'total_impact': total_impact,
            'adjusted_odds': adjusted_odds,
            'key_players_out': key_players_out,
            'confidence': confidence,
            'recommendation': recommendation
        }
    
    def _get_injury_recommendation(self, impact: float, key_players: List[str]) -> str:
        """Generate recommendation based on injury impact"""
        if impact > 0.25:
            return f"🔴 MAJOR IMPACT - Key players out: {', '.join(key_players[:2])}"
        elif impact > 0.15:
            return f"🟠 SIGNIFICANT IMPACT - Important players questionable"
        elif impact > 0.08:
            return f"🟡 MODERATE IMPACT - Monitor injury reports"
        else:
            return "🟢 MINIMAL IMPACT - Team at full strength"

# 3. WEATHER ANALYZER
class WeatherAnalyzer:
    """Analyzes weather impact on outdoor games"""
    
    def __init__(self, weather_api_key: str = None):
        self.api_key = weather_api_key or os.environ.get("WEATHER_API_KEY")
        self.outdoor_venues = {
            # NFL outdoor stadiums (simplified)
            'Green Bay Packers': True,
            'Chicago Bears': True,
            'Buffalo Bills': True,
            'Cleveland Browns': True,
            'Pittsburgh Steelers': True,
            'Kansas City Chiefs': True,
            'Denver Broncos': True,
            'New England Patriots': True,
            'Philadelphia Eagles': True,
            'Washington Commanders': True,
            'Baltimore Ravens': True,
            'Cincinnati Bengals': True,
            'Tennessee Titans': True,
            'Jacksonville Jaguars': True,
            'Miami Dolphins': True,
            'Carolina Panthers': True,
            'Tampa Bay Buccaneers': True,
            'Seattle Seahawks': True,
            'San Francisco 49ers': True,
            # MLB - all outdoor except dome stadiums
        }
    
    def analyze_weather_impact(self, home_team: str, sport: str, 
                               weather_data: Dict = None) -> Dict:
        """
        Analyze how weather affects the game
        
        weather_data: {
            'temp': 35,  # Fahrenheit
            'wind_speed': 20,  # MPH
            'precipitation': 0.5,  # inches
            'condition': 'snow'
        }
        """
        if not self._is_outdoor(home_team):
            return {
                'is_outdoor': False,
                'impact': 0.0,
                'total_adjustment': 0.0,
                'recommendation': '🏟️ INDOOR - Weather not a factor'
            }
        
        if not weather_data:
            # Try to fetch real weather if API configured
            weather_data = self._fetch_weather(home_team) if self.api_key else {}
        
        if not weather_data:
            return {
                'is_outdoor': True,
                'impact': 0.0,
                'total_adjustment': 0.0,
                'recommendation': '⚪ Weather data unavailable'
            }
        
        temp = weather_data.get('temp', 70)
        wind = weather_data.get('wind_speed', 0)
        precip = weather_data.get('precipitation', 0)
        
        # Calculate impact on scoring (for totals)
        impact = 0.0
        factors = []
        
        # Temperature impact (NFL)
        if sport in ['americanfootball_nfl', 'americanfootball_ncaaf']:
            if temp < 32:
                impact -= 0.15  # Cold reduces scoring by ~15%
                factors.append(f"❄️ Freezing ({temp}°F)")
            elif temp < 45:
                impact -= 0.08
                factors.append(f"🥶 Cold ({temp}°F)")
            elif temp > 95:
                impact -= 0.05  # Extreme heat also reduces scoring
                factors.append(f"🔥 Hot ({temp}°F)")
        
        # Wind impact (huge for passing/kicking)
        if wind > 20:
            impact -= 0.20  # Strong wind kills passing game
            factors.append(f"💨 Strong Wind ({wind} MPH)")
        elif wind > 15:
            impact -= 0.12
            factors.append(f"🌬️ Windy ({wind} MPH)")
        elif wind > 10:
            impact -= 0.05
            factors.append(f"🍃 Breezy ({wind} MPH)")
        
        # Precipitation impact
        if precip > 0.5:
            impact -= 0.15  # Heavy rain/snow significantly reduces scoring
            factors.append(f"🌧️ Heavy Precipitation")
        elif precip > 0.1:
            impact -= 0.08
            factors.append(f"☔ Light Precipitation")
        
        # Total adjustment (can be significant)
        total_adjustment = impact
        
        recommendation = self._get_weather_recommendation(total_adjustment, factors)
        
        return {
            'is_outdoor': True,
            'impact': impact,
            'total_adjustment': total_adjustment,
            'factors': factors,
            'recommendation': recommendation,
            'suggested_play': 'UNDER' if total_adjustment < -0.10 else ('OVER' if total_adjustment > 0.05 else 'NEUTRAL')
        }
    
    def _is_outdoor(self, team: str) -> bool:
        """Check if team plays in outdoor venue"""
        return self.outdoor_venues.get(team, False)
    
    def _fetch_weather(self, team: str) -> Dict:
        """Fetch real weather data (placeholder for API call)"""
        # In production, use OpenWeatherMap or similar
        # For now, return empty to use manual input
        return {}
    
    def _get_weather_recommendation(self, adjustment: float, factors: List[str]) -> str:
        """Generate weather-based recommendation"""
        if adjustment < -0.20:
            return f"🔴 EXTREME CONDITIONS - Strong UNDER play. {', '.join(factors)}"
        elif adjustment < -0.10:
            return f"🟠 ADVERSE CONDITIONS - Lean UNDER. {', '.join(factors)}"
        elif adjustment < -0.05:
            return f"🟡 MINOR IMPACT - Slight UNDER lean. {', '.join(factors)}"
        elif adjustment > 0.05:
            return f"🟢 FAVORABLE CONDITIONS - Possible OVER. {', '.join(factors)}"
        else:
            return "⚪ NEUTRAL - Weather not a major factor"

# 4. KELLY CRITERION CALCULATOR
class KellyCalculator:
    """Calculates optimal bet sizing using Kelly Criterion"""
    
    def calculate_kelly(self, win_probability: float, decimal_odds: float, 
                       bankroll: float = 1000, conservative: bool = True) -> Dict:
        """
        Calculate Kelly Criterion bet size
        
        Args:
            win_probability: Your estimated probability of winning (0-1)
            decimal_odds: Payout odds in decimal format
            bankroll: Total bankroll
            conservative: Use fractional Kelly (recommended)
        
        Returns:
            'kelly_percentage': % of bankroll to bet
            'recommended_stake': Dollar amount to bet
            'expected_value': Expected profit per $1 bet
            'risk_level': 'Low', 'Medium', 'High'
        """
        # Kelly formula: f = (bp - q) / b
        # where f = fraction to bet, b = odds-1, p = win prob, q = lose prob
        
        b = decimal_odds - 1  # Net odds
        p = win_probability
        q = 1 - p
        
        # Calculate edge
        edge = (p * decimal_odds) - 1
        
        if edge <= 0:
            # No edge, don't bet
            return {
                'kelly_percentage': 0.0,
                'recommended_stake': 0.0,
                'expected_value': edge,
                'risk_level': 'NONE',
                'recommendation': '🔴 NO EDGE - Do not bet (negative EV)'
            }
        
        # Full Kelly
        kelly_fraction = (b * p - q) / b
        
        # Clamp to reasonable range
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Never bet more than 25%
        
        # Conservative Kelly (recommended: 0.25x to 0.5x Kelly)
        if conservative:
            kelly_fraction *= 0.25  # Quarter Kelly
        
        kelly_percentage = kelly_fraction * 100
        recommended_stake = bankroll * kelly_fraction
        
        # Risk level based on Kelly %
        if kelly_percentage < 1:
            risk_level = 'Low'
        elif kelly_percentage < 3:
            risk_level = 'Medium'
        elif kelly_percentage < 5:
            risk_level = 'High'
        else:
            risk_level = 'Very High'
        
        recommendation = self._get_kelly_recommendation(kelly_percentage, edge, risk_level)
        
        return {
            'kelly_percentage': kelly_percentage,
            'recommended_stake': recommended_stake,
            'expected_value': edge,
            'edge_percentage': edge * 100,
            'risk_level': risk_level,
            'recommendation': recommendation
        }
    
    def _get_kelly_recommendation(self, kelly_pct: float, edge: float, risk: str) -> str:
        """Generate Kelly-based recommendation"""
        if kelly_pct == 0:
            return "🔴 NO BET - Negative expected value"
        elif kelly_pct < 1:
            return f"🟢 SMALL BET - {kelly_pct:.1f}% of bankroll | {risk} risk | Edge: {edge*100:+.1f}%"
        elif kelly_pct < 3:
            return f"🟡 MEDIUM BET - {kelly_pct:.1f}% of bankroll | {risk} risk | Edge: {edge*100:+.1f}%"
        elif kelly_pct < 5:
            return f"🟠 LARGE BET - {kelly_pct:.1f}% of bankroll | {risk} risk | Edge: {edge*100:+.1f}%"
        else:
            return f"🔴 REDUCE SIZE - {kelly_pct:.1f}% too high, cap at 5% | Edge: {edge*100:+.1f}%"

# 5. MATCHUP ANALYZER
class MatchupAnalyzer:
    """Analyzes historical head-to-head matchups"""
    
    def __init__(self):
        # In production, fetch from real database
        # This is simplified for demonstration
        self.matchup_history = {}
    
    def analyze_matchup(self, home_team: str, away_team: str, 
                       recent_games: int = 10) -> Dict:
        """
        Analyze historical matchup between teams
        
        Returns patterns like:
        - Home team dominance
        - Scoring trends
        - ATS records
        - Over/under trends
        """
        # Simplified analysis (in production, query actual database)
        matchup_key = f"{home_team}_vs_{away_team}"
        
        # Placeholder data structure
        analysis = {
            'games_analyzed': 0,
            'home_wins': 0,
            'away_wins': 0,
            'avg_total': 0,
            'home_ats': 0,
            'over_record': 0,
            'trend': 'neutral',
            'confidence': 0.3,
            'recommendation': '⚪ Insufficient historical data'
        }
        
        # In production, calculate from real data:
        # - Win percentages
        # - Scoring averages
        # - Recent trends
        # - Home/away splits
        
        return analysis
    
    def get_recommendation(self, analysis: Dict) -> str:
        """Generate matchup-based recommendation"""
        if analysis['games_analyzed'] < 3:
            return "⚪ Limited history - rely on other factors"
        
        home_win_pct = analysis['home_wins'] / analysis['games_analyzed']
        
        if home_win_pct > 0.7:
            return "🟢 HOME TEAM DOMINATES - Strong historical advantage"
        elif home_win_pct < 0.3:
            return "🔴 AWAY TEAM OWNS MATCHUP - Fade home team"
        else:
            return "🟡 COMPETITIVE MATCHUP - No clear historical edge"

# 6. ADVANCED STATS INTEGRATOR
class AdvancedStatsIntegrator:
    """Integrates advanced metrics like EPA, DVOA, etc."""
    
    def __init__(self):
        # Weights for different advanced metrics
        self.metric_weights = {
            'epa': 0.30,      # Expected Points Added
            'dvoa': 0.25,     # Defense-adjusted Value Over Average  
            'success_rate': 0.20,
            'explosive_play_rate': 0.15,
            'turnover_rate': 0.10
        }
    
    def calculate_team_strength(self, team: str, sport: str, 
                               advanced_metrics: Dict = None) -> Dict:
        """
        Calculate overall team strength using advanced metrics
        
        advanced_metrics: {
            'offensive_epa': 0.15,
            'defensive_epa': -0.10,
            'offensive_dvoa': 12.5,
            'defensive_dvoa': -8.2,
            ...
        }
        """
        if not advanced_metrics:
            # Return baseline
            return {
                'overall_rating': 0.0,
                'offensive_rating': 0.0,
                'defensive_rating': 0.0,
                'confidence': 0.2,
                'recommendation': '⚪ Advanced stats unavailable'
            }
        
        # Combine metrics using weights
        offensive_score = 0.0
        defensive_score = 0.0
        
        # EPA (Expected Points Added)
        if 'offensive_epa' in advanced_metrics:
            offensive_score += advanced_metrics['offensive_epa'] * self.metric_weights['epa']
        if 'defensive_epa' in advanced_metrics:
            defensive_score += advanced_metrics['defensive_epa'] * self.metric_weights['epa']
        
        overall_rating = offensive_score + defensive_score
        
        recommendation = self._get_advanced_stats_recommendation(overall_rating)
        
        return {
            'overall_rating': overall_rating,
            'offensive_rating': offensive_score,
            'defensive_rating': defensive_score,
            'confidence': 0.75,
            'recommendation': recommendation
        }
    
    def _get_advanced_stats_recommendation(self, rating: float) -> str:
        """Generate recommendation from advanced stats"""
        if rating > 0.20:
            return "🟢 ELITE TEAM - Advanced metrics very strong"
        elif rating > 0.10:
            return "🟡 ABOVE AVERAGE - Good underlying metrics"
        elif rating > -0.10:
            return "⚪ AVERAGE - Neutral metrics"
        elif rating > -0.20:
            return "🟠 BELOW AVERAGE - Concerning metrics"
        else:
            return "🔴 POOR TEAM - Weak underlying metrics"

# 7. SOCIAL MEDIA ANALYZER (Simplified)
class SocialMediaAnalyzer:
    """Analyzes Twitter/Reddit for real-time sentiment"""
    
    def __init__(self, twitter_api_key: str = None):
        self.twitter_key = twitter_api_key or os.environ.get("TWITTER_API_KEY")
        self.keywords = {
            'positive': ['win', 'great', 'amazing', 'clutch', 'dominant', 'beast'],
            'negative': ['lose', 'bad', 'terrible', 'bench', 'cut', 'injured']
        }
    
    def analyze_social_sentiment(self, team: str, player: str = None) -> Dict:
        """
        Analyze social media sentiment
        
        Note: This is a simplified placeholder. Full implementation requires:
        - Twitter API v2 access
        - Reddit API (PRAW) integration
        - Real-time data scraping
        """
        # Placeholder return
        return {
            'sentiment_score': 0.0,
            'tweet_volume': 0,
            'trending': False,
            'confidence': 0.2,
            'recommendation': '⚪ Social media analysis unavailable (API key needed)'
        }
    
    def detect_breaking_news(self, team: str) -> Dict:
        """Detect breaking news that might not be priced in yet"""
        # Placeholder - would monitor Twitter for breaking news keywords
        return {
            'has_breaking_news': False,
            'news_type': None,
            'urgency': 'low'
        }

# ============ KALSHI VALIDATION HELPER ============
def validate_with_kalshi(kalshi_integrator, home_team: str, away_team: str,
                        side: str, sportsbook_prob: float, sport: str) -> Dict:
    """
    IMPROVED: Validate sportsbook odds with Kalshi prediction market
    
    Now handles team name variations like:
    - "Memphis Grizzlies" matches "Memphis"
    - "New York Knicks" matches "New York K"  
    - "Los Angeles Lakers" matches "LA Lakers"
    
    Returns:
        'kalshi_prob': Kalshi market probability
        'discrepancy': Difference between markets
        'validation': 'confirms', 'contradicts', or 'unavailable'
        'edge': Additional edge from Kalshi vs sportsbook
        'confidence_boost': How much to boost confidence (0-0.20)
    """
    
    def normalize_team_name(team: str) -> List[str]:
        """Generate multiple variations of a team name for flexible matching"""
        team_upper = team.upper()
        variations = [team_upper, team_upper.replace(" ", "")]

        # Split into parts and add individual words
        parts = team_upper.split()
        for part in parts:
            if len(part) > 2:  # Skip very short words
                variations.append(part)

        # Special handling for common abbreviations
        abbreviations = {
            'NEW YORK': ['NY', 'NEW YORK K', 'N.Y.'],
            'LOS ANGELES': ['LA', 'L.A.'],
            'SAN FRANCISCO': ['SF', 'S.F.'],
            'GOLDEN STATE': ['GS'],
            'OKLAHOMA CITY': ['OKC'],
            'WASHINGTON': ['WSH'],
        }

        for city, abbrevs in abbreviations.items():
            if team_upper.startswith(city):
                variations.extend(abbrevs)

        # Kalshi-specific abbreviation support
        for canonical, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            canonical_upper = canonical.upper()
            canonical_words = canonical_upper.split()

            # Direct matches (exact, contains, or shared keywords)
            if (
                team_upper == canonical_upper
                or canonical_upper in team_upper
                or team_upper in canonical_upper
                or any(word in canonical_words for word in parts if len(word) > 2)
            ):
                variations.extend(abbrs)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)

        return unique_variations
    
    def teams_match(bet_team: str, market_text: str) -> bool:
        """Check if a bet team matches text in a market without short false-positives."""
        bet_variations = normalize_team_name(bet_team)
        market_upper = re.sub(r"[^A-Z0-9 ]", " ", market_text.upper().replace('_', ' '))
        market_compact = market_upper.replace(' ', '')
        market_tokens = set(re.findall(r"[A-Z0-9]+", market_upper))

        for variation in bet_variations:
            variation_upper = variation.upper()
            variation_clean = re.sub(r"[^A-Z0-9 ]", " ", variation_upper).strip()
            variation_compact = variation_clean.replace(' ', '')

            if not variation_compact:
                continue

            # Longer variations (team names, extended abbreviations) can match anywhere in the text
            if len(variation_compact) >= 4 and variation_compact in market_compact:
                return True

            # Compare token-by-token to avoid matching "LA" with "ATLANTA"
            variation_tokens = re.findall(r"[A-Z0-9]+", variation_clean)
            if variation_tokens and all(token in market_tokens for token in variation_tokens):
                return True

            # Allow short tokens (NY, LA, SF) only on whole-word matches
            if len(variation_compact) <= 3 and variation_clean in market_tokens:
                return True

        return False
    
    def extract_probability(orderbook: Dict[str, Any]) -> Optional[float]:
        if not orderbook:
            return None

        yes_bids = orderbook.get('yes', [])
        no_bids = orderbook.get('no', [])

        if yes_bids:
            price = yes_bids[0].get('price')
            if price is not None:
                return price / 100.0

        if no_bids:
            price = no_bids[0].get('price')
            if price is not None:
                return 1.0 - (price / 100.0)

        return None

    def find_canonical_team_name(team: str) -> Optional[str]:
        team_upper = team.upper()
        squeezed = team_upper.replace(" ", "")

        for canonical, abbrs in KALSHI_TEAM_ABBREVIATIONS.items():
            if canonical == team_upper or canonical.replace(" ", "") == squeezed:
                return canonical
            if canonical in team_upper or team_upper in canonical:
                return canonical
            for abbr in abbrs:
                abbr_clean = abbr.upper().replace(" ", "")
                if abbr_clean and abbr_clean in squeezed:
                    return canonical
        return None

    def build_market_validation(market: Dict[str, Any], scope: str) -> Optional[Dict[str, Any]]:
        if not market:
            return None

        orderbook = kalshi_integrator.get_orderbook(market.get('ticker', '')) if market.get('ticker') else {}
        kalshi_prob = extract_probability(orderbook)

        if kalshi_prob is None:
            return None

        diff = kalshi_prob - sportsbook_prob
        synthetic_market = market.get('synthetic', False)

        mild_threshold = 0.05 if scope == 'head_to_head' else 0.04
        strong_threshold = 0.10 if scope == 'head_to_head' else 0.08
        base_boost = 0.08 if scope == 'head_to_head' else 0.05
        boost_multiplier = 0.6 if synthetic_market else 1.0

        if diff >= strong_threshold:
            validation = 'strong_kalshi_higher'
            confidence_boost = base_boost * 1.2 * boost_multiplier
            edge = diff
        elif diff >= mild_threshold:
            validation = 'kalshi_higher'
            confidence_boost = base_boost * boost_multiplier
            edge = diff
        elif diff <= -strong_threshold:
            validation = 'strong_contradiction'
            confidence_boost = -base_boost * boost_multiplier
            edge = abs(diff)
        elif diff <= -mild_threshold:
            validation = 'kalshi_lower'
            confidence_boost = -base_boost * 0.6 * boost_multiplier
            edge = abs(diff)
        else:
            validation = 'confirms'
            confidence_boost = base_boost * 0.5 * boost_multiplier
            edge = max(diff, 0)

        return {
            'kalshi_prob': kalshi_prob,
            'kalshi_available': True,
            'discrepancy': abs(diff),
            'validation': validation,
            'edge': edge,
            'confidence_boost': confidence_boost,
            'market_ticker': market.get('ticker'),
            'market_title': market.get('title'),
            'market_scope': scope,
            'data_source': 'synthetic' if synthetic_market else 'kalshi'
        }

    try:
        markets = kalshi_integrator.get_sports_markets()

        bet_team = home_team if side == 'home' else away_team
        other_team = away_team if side == 'home' else home_team

        canonical_team = find_canonical_team_name(bet_team) or bet_team.upper()

        matching_market = None
        fallback_market = None

        for market in markets:
            title = market.get('title', '')
            ticker = market.get('ticker', '')
            subtitle = market.get('subtitle', '')
            market_text = f"{title} {ticker} {subtitle}"

            has_bet_team = teams_match(bet_team, market_text)
            has_other_team = teams_match(other_team, market_text)
            is_synthetic_market = market.get('synthetic', False)

            if has_bet_team and has_other_team and not is_synthetic_market:
                matching_market = market
                break

            if has_bet_team and fallback_market is None:
                fallback_market = market

        if matching_market:
            result = build_market_validation(matching_market, 'head_to_head')
            if result:
                return result

        if fallback_market:
            scope = 'head_to_head' if teams_match(other_team, f"{fallback_market.get('title', '')} {fallback_market.get('subtitle', '')}") else 'team_future'
            result = build_market_validation(fallback_market, scope)
            if result:
                return result

        synthetic_market = kalshi_integrator.get_synthetic_market_for_team(canonical_team)
        if synthetic_market:
            result = build_market_validation(synthetic_market, 'synthetic')
            if result:
                return result

        synthetic_prob = kalshi_integrator.synthetic_probability(canonical_team, sport, sportsbook_prob)
        diff = synthetic_prob - sportsbook_prob

        if diff >= 0.06:
            validation = 'kalshi_higher'
            confidence_boost = 0.03
            edge = diff
        elif diff <= -0.06:
            validation = 'strong_contradiction'
            confidence_boost = -0.03
            edge = abs(diff)
        else:
            validation = 'confirms'
            confidence_boost = 0.02
            edge = max(diff, 0)

        return {
            'kalshi_prob': synthetic_prob,
            'kalshi_available': True,
            'discrepancy': abs(diff),
            'validation': validation,
            'edge': edge,
            'confidence_boost': confidence_boost,
            'market_ticker': None,
            'market_title': f"Synthetic confidence for {bet_team}",
            'market_scope': 'synthetic_estimate',
            'data_source': 'synthetic_estimate'
        }

    except Exception as e:
        # Error fetching Kalshi data
        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None,
            'market_scope': 'error',
            'data_source': 'error'
        }

# Helper to apply Kalshi validation to a betting leg in-place
def integrate_kalshi_into_leg(
    leg_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    side: str,
    base_prob: float,
    sport: str,
    use_kalshi: bool,
) -> None:
    """Mutate a leg dictionary with Kalshi validation + probability blending."""

    # Ensure downstream code sees the reason when Kalshi is not active
    if not use_kalshi:
        leg_data.setdefault('kalshi_validation', {
            'kalshi_available': False,
            'validation': 'disabled',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'disabled',
            'data_source': 'disabled'
        })
        return

    kalshi = None
    try:
        kalshi = st.session_state.get('kalshi_integrator')
    except Exception:
        # When Streamlit session state isn't available (e.g. testing), skip gracefully
        pass

    if not kalshi:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'not_initialized',
            'data_source': 'unavailable'
        }
        return

    try:
        kalshi_data = validate_with_kalshi(kalshi, home_team, away_team, side, base_prob, sport)
    except Exception:
        leg_data['kalshi_validation'] = {
            'kalshi_available': False,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_scope': 'error',
            'data_source': 'error'
        }
        return

    leg_data['kalshi_validation'] = kalshi_data

    if not kalshi_data.get('kalshi_available'):
        return

    original_ai_prob = leg_data.get('ai_prob', base_prob)
    kalshi_prob = kalshi_data.get('kalshi_prob', base_prob)

    blended_prob = (
        original_ai_prob * 0.50 +  # AI model
        kalshi_prob * 0.30 +       # Kalshi market
        base_prob * 0.20           # Sportsbook baseline
    )

    leg_data['ai_prob_before_kalshi'] = original_ai_prob
    leg_data['ai_prob'] = blended_prob
    leg_data['kalshi_influence'] = blended_prob - original_ai_prob
    leg_data['kalshi_edge'] = kalshi_data.get('edge', 0)
    leg_data['ai_confidence'] = min(
        leg_data.get('ai_confidence', 0.5) + kalshi_data.get('confidence_boost', 0),
        0.95
    )

# ============ UTILITY FUNCTIONS ============
def american_to_decimal(odds) -> float:
    odds = float(odds)
    if odds >= 100: return 1.0 + odds/100.0
    if odds <= -100: return 1.0 + 100.0/abs(odds)
    raise ValueError("Bad American odds")

def implied_p_from_american(odds) -> float:
    odds = float(odds)
    return 100.0/(odds+100.0) if odds>0 else abs(odds)/(abs(odds)+100.0)

def ev_rate(p: float, d: float) -> float: 
    return p*d - 1.0

def _dig(obj, path, default=None):
    try:
        cur = obj
        for token in path.split('.'):
            if '[' in token and token.endswith(']'):
                name, idx = token[:-1].split('[')
                if name: cur = cur.get(name, {})
                cur = cur[int(idx)]
            else:
                cur = cur.get(token) if isinstance(cur, dict) else None
        return default if cur is None else cur
    except Exception:
        return default

def _odds_api_base():
    return "https://api.the-odds-api.com"


def build_leg_apisports_payload(summary: Any, side: str) -> Dict[str, Any]:
    """Return a compact snapshot of API-Sports data for a parlay leg."""

    if not summary:
        return {}

    def _get(container, attr):
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(attr)
        return getattr(container, attr, None)

    team_obj = _get(summary, 'home' if side == 'home' else 'away')
    opponent_obj = _get(summary, 'away' if side == 'home' else 'home')

    payload = {
        'game_id': _get(summary, 'id'),
        'league': _get(summary, 'league'),
        'season': _get(summary, 'season'),
        'status': _get(summary, 'status'),
        'kickoff': _get(summary, 'kickoff_local'),
        'venue': _get(summary, 'venue'),
        'team_record': _get(team_obj, 'record'),
        'team_form': _get(team_obj, 'form'),
        'trend': _get(team_obj, 'trend'),
        'team_avg_points_for': _get(team_obj, 'average_points_for'),
        'team_avg_points_against': _get(team_obj, 'average_points_against'),
        'opponent_record': _get(opponent_obj, 'record'),
        'opponent_form': _get(opponent_obj, 'form'),
        'opponent_avg_points_for': _get(opponent_obj, 'average_points_for'),
        'opponent_avg_points_against': _get(opponent_obj, 'average_points_against'),
    }

    return {k: v for k, v in payload.items() if v not in (None, '')}

def fetch_oddsapi_snapshot(api_key: str, sport_key: str) -> Dict[str, Any]:
    url = f"{_odds_api_base()}/v4/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        
        # Check if we got valid data
        if not data or not isinstance(data, list):
            st.warning(f"No data returned for {sport_key}")
            return {"events": []}
            
    except requests.exceptions.Timeout:
        st.warning(f"Request timeout for {sport_key} - skipping")
        return {"events": []}
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for {sport_key}: {str(e)}")
        return {"events": []}
    except Exception as e:
        st.error(f"Unexpected error fetching {sport_key}: {str(e)}")
        return {"events": []}
    
    events = []
    for ev in (data or []):
        home, away = ev.get("home_team"), ev.get("away_team")
        markets = {"h2h":{}, "spreads":[], "totals":[]}
        for b in ev.get("bookmakers") or []:
            for m in b.get("markets") or []:
                key = m.get("key")
                if key == "h2h":
                    for o in m.get("outcomes") or []:
                        if o.get("name")==home: 
                            markets["h2h"]["home"] = {"price": o.get("price")}
                        elif o.get("name")==away: 
                            markets["h2h"]["away"] = {"price": o.get("price")}
                elif key == "spreads":
                    for o in m.get("outcomes") or []:
                        markets["spreads"].append({
                            "name": o.get("name"), 
                            "point": o.get("point"), 
                            "price": o.get("price")
                        })
                elif key == "totals":
                    for o in m.get("outcomes") or []:
                        markets["totals"].append({
                            "name": o.get("name"), 
                            "point": o.get("point"), 
                            "price": o.get("price")
                        })
        events.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": home, 
            "away_team": away,
            "markets": markets
        })
    return {"events": events}

def calculate_profit(decimal_odds: float, stake: float = 100) -> float:
    return (decimal_odds - 1.0) * stake

def match_theover_to_leg(leg, theover_data):
    """
    Match a parlay leg with theover.ai data and validate direction
    Returns dict with: {'pick': str, 'matches': bool, 'signal': str} or None
    - pick: theover.ai recommendation (Over/Under)
    - matches: True if leg direction matches theover.ai pick
    - signal: Visual indicator (✅/⚠️/—)
    """
    if theover_data is None or theover_data.empty:
        return None
    
    try:
        # Normalize column names
        theover_df = theover_data.copy()
        theover_df.columns = [c.strip().lower() for c in theover_df.columns]
        
        # Extract info from leg
        leg_label = leg.get('label', '').lower()
        team = leg.get('team', '').lower()
        market_type = leg.get('type', '').lower()
        
        # Check if this is theover.ai format (has awayteam, hometeam, pick columns)
        if 'awayteam' in theover_df.columns and 'hometeam' in theover_df.columns and 'pick' in theover_df.columns:
            # theover.ai format - match by teams and pick type
            for idx, row in theover_df.iterrows():
                away_team = str(row.get('awayteam', '')).lower()
                home_team = str(row.get('hometeam', '')).lower()
                pick = str(row.get('pick', '')).lower()
                
                # Check if both teams match the leg (for totals)
                if market_type == 'total':
                    # Check if the matchup matches
                    if (away_team in leg_label or team in away_team) and \
                       (home_team in leg_label or team in home_team):
                        # Return the pick (Over/Under) with match validation
                        if pick in ['over', 'under']:
                            # Check if the leg direction matches theover.ai pick
                            leg_direction = None
                            if 'over' in leg_label:
                                leg_direction = 'over'
                            elif 'under' in leg_label:
                                leg_direction = 'under'
                            
                            matches = (leg_direction == pick) if leg_direction else None
                            
                            return {
                                'pick': pick.capitalize(),
                                'matches': matches,
                                'signal': '✅' if matches else ('⚠️' if matches == False else '❓')
                            }
                
                # Check for moneyline or spread picks on specific teams
                elif team:
                    team_matches = team in away_team or away_team in team or \
                                 team in home_team or home_team in team
                    
                    if team_matches:
                        # For ML or spread, return the pick if available
                        if pick and pick != 'nan':
                            return {
                                'pick': pick.capitalize(),
                                'matches': None,  # Can't validate ML/spread direction yet
                                'signal': '🎯'
                            }
        
        else:
            # Standard format - original matching logic
            for idx, row in theover_df.iterrows():
                row_team = str(row.get('team', row.get('player', ''))).lower()
                row_market = str(row.get('stat', row.get('market', ''))).lower()
                
                # Check for team match
                if team in row_team or row_team in team:
                    # Check for market type match
                    if ('moneyline' in market_type and ('ml' in row_market or 'win' in row_market)) or \
                       ('spread' in market_type and 'spread' in row_market) or \
                       ('total' in market_type and ('total' in row_market or 'points' in row_market)):
                        projection = row.get('projection', None)
                        if projection is not None:
                            return {
                                'pick': float(projection),
                                'matches': None,
                                'signal': '🎯'
                            }
        
        return None
    except Exception as e:
        return None

def build_combos_ai(legs, k, allow_sgp, optimizer, theover_data=None, min_probability=0.25, max_probability=0.70):
    """Build parlay combinations with AI scoring - deduplicates and keeps best odds
    Now filters parlays to realistic probability range (default: 25-70%)"""
    parlay_map = {}  # Maps parlay_key -> best parlay so far
    
    for combo in itertools.combinations(legs, k):
        if not allow_sgp and len({c["event_id"] for c in combo}) < k:
            continue
        
        # Skip combos with missing required fields
        try:
            # IMPROVED: Create a unique key that treats different spreads for same team as duplicates
            # This prevents parlays like "Bucs -2.5 + Bucs -3.0" which are redundant
            parlay_key_parts = []
            for c in combo:
                event_id = c.get('event_id', '')
                bet_type = c.get('type', '')
                team = c.get('team', '')
                side = c.get('side', '')
                
                # For spreads/totals, don't include exact point in key
                # This way Bucs -2.5 and Bucs -3.0 are treated as same bet
                if bet_type in ['Spread', 'Total']:
                    # Key without specific point value
                    key_part = f"{event_id}_{bet_type}_{team}_{side}"
                else:
                    # For ML, include everything
                    key_part = f"{event_id}_{bet_type}_{team}_{side}"
                
                parlay_key_parts.append(key_part)
            
            parlay_key = tuple(sorted(parlay_key_parts))
            
            # Check if this combination has duplicate legs (same team, same game, same bet type)
            # This catches cases like "Bucs -2.5 + Bucs -3.0" in the same parlay
            unique_bets = set(parlay_key_parts)
            if len(unique_bets) < len(combo):
                continue  # Skip parlays with duplicate bets
                
        except Exception:
            continue  # Skip this combo if we can't create a key
        
        d = 1.0
        p_market = 1.0
        p_ai = 1.0
        
        # Safety check: Skip if any leg has invalid decimal odds
        skip_combo = False
        for c in combo:
            leg_d = c.get("d")
            if leg_d is None or leg_d <= 0:
                skip_combo = True
                break
            d *= leg_d
            p_market *= c.get("p", 0.5)
            p_ai *= c.get("ai_prob", c.get("p", 0.5))
        
        if skip_combo or d <= 0:
            continue  # Skip this combo if odds are invalid
        
        # Get AI score for this parlay
        ai_metrics = optimizer.score_parlay(list(combo))
        
        # Calculate theover.ai validation bonus
        theover_bonus = 0.0
        theover_matches = 0
        theover_conflicts = 0
        
        if theover_data is not None:
            for leg in combo:
                result = match_theover_to_leg(leg, theover_data)
                if result and isinstance(result, dict):
                    matches = result.get('matches')
                    if matches == True:
                        theover_matches += 1
                        theover_bonus += 0.15  # 15% bonus per matching leg
                    elif matches == False:
                        theover_conflicts += 1
                        theover_bonus -= 0.10  # 10% penalty per conflicting leg
        
        profit = calculate_profit(d, 100)
        market_ev = ev_rate(p_market, d)
        
        # Apply theover.ai bonus to AI score
        base_ai_score = ai_metrics['score']
        boosted_ai_score = base_ai_score * (1.0 + theover_bonus)
        
        parlay_data = {
            "legs": combo,
            "d": d,
            "p": p_market,
            "p_ai": p_ai,
            "ev_market": market_ev,
            "ev_ai": ai_metrics['ai_ev'],
            "profit": profit,
            "ai_score": boosted_ai_score,  # Boosted score with theover.ai
            "base_ai_score": base_ai_score,  # Original score
            "theover_bonus": theover_bonus,
            "theover_matches": theover_matches,
            "theover_conflicts": theover_conflicts,
            "ai_confidence": ai_metrics['confidence'],
            "ai_edge": ai_metrics['edge'],
            "kalshi_factor": ai_metrics.get('kalshi_factor', 1.0),
            "kalshi_boost": ai_metrics.get('kalshi_boost', 0),
            "kalshi_legs": ai_metrics.get('kalshi_legs', 0),
            "apisports_factor": ai_metrics.get('apisports_factor', 1.0),
            "apisports_boost": ai_metrics.get('apisports_boost', 0),
            "apisports_legs": ai_metrics.get('apisports_legs', 0)
        }
        
        # Keep only the version with best combined odds (highest decimal odds = best payout)
        if parlay_key not in parlay_map or d > parlay_map[parlay_key]["d"]:
            parlay_map[parlay_key] = parlay_data
    
    # Convert back to list
    out = list(parlay_map.values())
    
    # FILTER: Keep only realistic parlays in the value zone
    # Remove longshots AND heavy favorites (chalk parlays)
    out = [p for p in out if min_probability <= p["p_ai"] <= max_probability]
    
    # SIMPLE & EFFECTIVE: Sort by AI Expected Value
    # This finds bets where AI thinks you have an edge
    # Positive EV = long-term profitable
    out.sort(key=lambda x: (x["ev_ai"], x["p_ai"]), reverse=True)
    
    # FINAL DEDUPLICATION: Remove parlays that are too similar
    # This catches edge cases where same game appears multiple times with slightly different bets
    final_parlays = []
    seen_game_combos = set()
    
    for parlay in out:
        # Create a signature of which games are in this parlay
        game_ids = tuple(sorted(set(leg['event_id'] for leg in parlay['legs'])))
        
        # For same-game combinations, also check the specific bets
        bet_signature = tuple(sorted([
            f"{leg['event_id']}_{leg['type']}_{leg.get('team', '')}_{leg.get('side', '')}"
            for leg in parlay['legs']
        ]))
        
        if bet_signature not in seen_game_combos:
            final_parlays.append(parlay)
            seen_game_combos.add(bet_signature)
    
    return final_parlays

def render_parlay_section_ai(title, rows, theover_data=None):
    """Render parlays with AI insights"""
    st.markdown(f"### {title}")
    if not rows:
        st.info("No combinations found with current filters")
        return
    
    for i, row in enumerate(rows, start=1):
        # AI confidence indicator
        conf = row['ai_confidence']
        if conf > 0.7:
            conf_icon = "🟢"
        elif conf > 0.5:
            conf_icon = "🟡"
        else:
            conf_icon = "🟠"
        
        # EV indicator
        ai_ev_pct = row['ev_ai'] * 100
        if ai_ev_pct > 5:
            ev_icon = "💰"
        elif ai_ev_pct > 0:
            ev_icon = "📈"
        else:
            ev_icon = "📉"
        
        # Probability warning
        prob = row['p_ai']
        if prob < 0.25:
            prob_warning = "⚠️"  # Warning for low probability
        elif prob < 0.35:
            prob_warning = "⚡"  # Caution
        else:
            prob_warning = ""  # Good probability
        
        # theover.ai boost indicator
        theover_boost = ""
        if row.get('theover_matches', 0) > 0:
            theover_boost = f" | 🎯 {row['theover_matches']} match"
            if row['theover_matches'] > 1:
                theover_boost += "es"
        elif row.get('theover_conflicts', 0) > 0:
            theover_boost = f" | ⚠️ {row['theover_conflicts']} conflict"
            if row['theover_conflicts'] > 1:
                theover_boost += "s"
        
        # Kalshi validation indicator with INFLUENCE
        kalshi_boost = ""
        kalshi_legs = sum(1 for leg in row.get('legs', []) if leg.get('kalshi_validation', {}).get('kalshi_available', False))
        kalshi_factor = row.get('kalshi_factor', 1.0)

        if kalshi_legs > 0:
            # Show if Kalshi boosted or reduced score
            if kalshi_factor > 1.05:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓ ↗️+{(kalshi_factor-1)*100:.0f}%"
            elif kalshi_factor < 0.95:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓ ↘️{(kalshi_factor-1)*100:.0f}%"
            else:
                kalshi_boost = f" | 📊 {kalshi_legs} Kalshi✓"

        apisports_info = ""
        apisports_legs = row.get('apisports_legs', 0)
        apisports_factor = row.get('apisports_factor', 1.0)
        if apisports_legs:
            if apisports_factor > 1.02:
                apisports_info = f" | 🏈 API-Sports {apisports_legs} ↗️"
            elif apisports_factor < 0.98:
                apisports_info = f" | 🏈 API-Sports {apisports_legs} ↘️"
            else:
                apisports_info = f" | 🏈 API-Sports {apisports_legs}"

        prob_pct = row['p_ai'] * 100
        with st.expander(
            f"{conf_icon}{ev_icon}{prob_warning} #{i} - AI Score: {row['ai_score']:.1f}{theover_boost}{kalshi_boost}{apisports_info} | Odds: {row['d']:.2f} | AI Prob: {prob_pct:.1f}% | Profit: ${row['profit']:.2f}"
        ):
            # Metrics
            col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
            with col_a:
                st.metric("Decimal Odds", f"{row['d']:.3f}")
            with col_b:
                st.metric("AI Probability", f"{prob_pct:.2f}%")
            with col_c:
                st.metric("AI Confidence", f"{conf*100:.1f}%")
            with col_d:
                st.metric("Profit on $100", f"${row['profit']:.2f}")
            with col_e:
                delta_color = "normal" if row['ev_ai'] > 0 else "inverse"
                st.metric("AI Expected Value", f"{ai_ev_pct:.2f}%")
            with col_f:
                if apisports_legs:
                    st.metric(
                        "API-Sports Legs",
                        f"{apisports_legs}/{len(row['legs'])}",
                        help="Number of legs with live API-Sports context",
                    )
                else:
                    st.metric("API-Sports Legs", "0", help="Provide an API-Sports key to enrich NFL legs")
            
            # KALSHI STATUS - ALWAYS SHOW (whether data exists or not)
            st.markdown("---")
            kalshi_legs_with_data = row.get('kalshi_legs', 0)
            total_legs = len(row.get('legs', []))
            
            if kalshi_legs_with_data > 0:
                # HAS KALSHI DATA - Show influence
                st.markdown("### 📊 Kalshi Prediction Market Influence:")

                synthetic_legs = sum(
                    1 for leg in row.get('legs', [])
                    if 'synthetic' in leg.get('kalshi_validation', {}).get('data_source', '')
                )

                if synthetic_legs:
                    st.info(
                        f"🧪 Using simulated Kalshi fallback for {synthetic_legs} leg(s) "
                        "because live market data was unavailable."
                    )


                col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                
                with col_k1:
                    st.metric(
                        "Kalshi Legs",
                        f"{row.get('kalshi_legs', 0)}/{len(row.get('legs', []))}",
                        help="How many legs have Kalshi market data"
                    )
                
                with col_k2:
                    kalshi_boost_val = row.get('kalshi_boost', 0)
                    delta_boost = float(kalshi_boost_val) if kalshi_boost_val else None
                    delta_color = "normal" if kalshi_boost_val >= 0 else "inverse"
                    st.metric(
                        "Kalshi Boost Points",
                        f"{kalshi_boost_val:+.0f}",
                        delta=delta_boost,
                        delta_color=delta_color,
                        help="Raw boost points from Kalshi validation (+15 = strong confirmation, -10 = contradiction)"
                    )
                
                with col_k3:
                    kalshi_factor_val = row.get('kalshi_factor', 1.0)
                    st.metric(
                        "Score Multiplier",
                        f"{kalshi_factor_val:.2f}x",
                        delta=f"{(kalshi_factor_val-1)*100:+.0f}%" if kalshi_factor_val != 1.0 else None,
                        help="How much Kalshi adjusted the AI score (1.0 = no change, >1.0 = boosted, <1.0 = reduced)"
                    )
                
                with col_k4:
                    # Calculate score change from Kalshi
                    base_score = row['ai_score'] / kalshi_factor_val if kalshi_factor_val != 0 else row['ai_score']
                    score_change = row['ai_score'] - base_score
                    st.metric(
                        "Score Impact",
                        f"{score_change:+.1f} pts",
                        help="How many points Kalshi added/subtracted from AI score"
                    )
                
                # Explanation of Kalshi influence
                if kalshi_factor_val > 1.05:
                    st.success(f"🟢 **Kalshi BOOSTED this parlay by {(kalshi_factor_val-1)*100:.0f}%** - Prediction markets confirm AI analysis!")
                elif kalshi_factor_val < 0.95:
                    st.warning(f"🟠 **Kalshi REDUCED this parlay by {(1-kalshi_factor_val)*100:.0f}%** - Prediction markets skeptical of AI picks.")
                else:
                    st.info("🟡 **Kalshi NEUTRAL** - Prediction markets neither strongly confirm nor contradict AI.")
            else:
                # NO KALSHI DATA - Explain why
                st.markdown("### 📊 Kalshi Prediction Market Status:")
                legs = row.get('legs', [])
                scopes = [leg.get('kalshi_validation', {}).get('market_scope') for leg in legs]
                unsupported_labels = [
                    leg.get('label')
                    for leg in legs
                    if leg.get('kalshi_validation', {}).get('market_scope') in {
                        'total_market', 'unsupported_market', 'totals_not_supported'
                    }
                ]
                error_labels = [
                    leg.get('label')
                    for leg in legs
                    if leg.get('kalshi_validation', {}).get('market_scope') == 'error'
                ]
                not_initialized = any(scope == 'not_initialized' for scope in scopes)
                disabled = (not st.session_state.get('kalshi_enabled', False)) or any(scope == 'disabled' for scope in scopes)

                if disabled:
                    st.info("Kalshi validation is turned off. Toggle the Kalshi checkbox above to blend prediction markets into the analysis.")
                elif unsupported_labels:
                    st.info("Kalshi does not publish totals/prop markets, so these leg(s) rely on AI + sentiment only:")
                    for label in unsupported_labels:
                        st.caption(f"• {label}")
                    st.caption("Moneyline and spread legs will include Kalshi coverage whenever a market is available.")
                elif not_initialized:
                    st.info("Kalshi markets have not loaded yet. Add your Kalshi API key or retry to use the live/synthetic market data.")
                elif error_labels:
                    st.warning("Kalshi validation encountered an error for these legs (falling back to AI + sentiment):")
                    for label in error_labels:
                        st.caption(f"• {label}")
                else:
                    st.warning(f"""
                    **⚠️ No Kalshi Data Available for this Parlay** ({kalshi_legs_with_data}/{total_legs} legs)

                    **This means:**
                    - ✅ Analysis still uses AI + Sentiment (2 of 3 sources)
                    - ⚠️ Missing prediction market validation
                    - 🔄 Kalshi Factor = 1.0x (neutral, no impact)
                    - 📊 AI Score unchanged by Kalshi

                    **Why no data?**
                    - Kalshi doesn't have markets for these specific games
                    - Kalshi focuses on season-long outcomes (playoffs, championships)
                    - Individual game spreads/totals rarely have Kalshi markets

                    **What this means:**
                    - Bet based on AI + Sentiment confidence
                    - Higher risk without 3rd source validation
                    - Consider checking Tab 4 for available Kalshi markets

                    💡 **Tip:** For Kalshi validation, focus on season futures, playoff odds, or major championships.
                    """)

            # theover.ai boost info if available
            if row.get('theover_bonus', 0) != 0:
                theover_bonus_pct = row['theover_bonus'] * 100
                if theover_bonus_pct > 0:
                    st.success(f"🎯 **theover.ai Boost:** +{theover_bonus_pct:.0f}% to AI score ({row.get('theover_matches', 0)} matching picks)")
                else:
                    st.warning(f"⚠️ **theover.ai Conflict:** {theover_bonus_pct:.0f}% penalty ({row.get('theover_conflicts', 0)} conflicting picks)")
            
            # Market vs AI comparison
            st.markdown("**📊 Market vs AI Analysis:**")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.write(f"Market EV: {row['ev_market']*100:.2f}%")
                st.write(f"Market Prob: {row['p']*100:.2f}%")
            with comp_col2:
                st.write(f"AI EV: {ai_ev_pct:.2f}%")
                st.write(f"AI Edge: {row['ai_edge']*100:.2f}%")
                
                # Show sentiment impact
                prob_diff = (row['p_ai'] - row['p']) * 100
                if abs(prob_diff) > 1:
                    if prob_diff > 0:
                        st.success(f"↗️ Sentiment boosted by {prob_diff:.1f}%")
                    else:
                        st.warning(f"↘️ Sentiment reduced by {abs(prob_diff):.1f}%")
            
            # Legs breakdown with theover.ai integration
            st.markdown("**🎯 Parlay Legs:**")
            legs_data = []
            has_theover = False
            theover_matches = 0
            theover_conflicts = 0
            
            for j, leg in enumerate(row["legs"], start=1):
                # Try to match with theover.ai data
                theover_result = match_theover_to_leg(leg, theover_data)
                
                if theover_result is not None:
                    has_theover = True
                    
                    # Handle dict result (new format with validation)
                    if isinstance(theover_result, dict):
                        pick = theover_result.get('pick', '')
                        matches = theover_result.get('matches')
                        signal = theover_result.get('signal', '🎯')
                        
                        # Count matches and conflicts for summary
                        if matches == True:
                            theover_matches += 1
                            theover_display = f"{signal} {pick}"
                        elif matches == False:
                            theover_conflicts += 1
                            theover_display = f"{signal} {pick}"
                        else:
                            theover_display = f"{signal} {pick}"
                    else:
                        # Handle numeric or simple string values (backward compatibility)
                        if isinstance(theover_result, (int, float)):
                            theover_display = f"🎯 {theover_result:.2f}"
                        else:
                            theover_display = f"🎯 {theover_result}"
                else:
                    theover_display = "—"
                
                # Kalshi validation display
                kalshi_display = "—"
                kalshi_influence_display = ""
                
                if 'kalshi_validation' in leg:
                    kv = leg['kalshi_validation']
                    if kv.get('kalshi_available'):
                        kalshi_prob = kv.get('kalshi_prob', 0) * 100
                        validation = kv.get('validation', 'unavailable')
                        data_source = kv.get('data_source', 'kalshi')

                        source_prefix = "🧪 " if data_source and 'synthetic' in data_source else ""

                        # Show Kalshi probability
                        if validation == 'confirms':
                            kalshi_display = f"{source_prefix}✅ {kalshi_prob:.1f}%"
                        elif validation == 'kalshi_higher':
                            kalshi_display = f"{source_prefix}📈 {kalshi_prob:.1f}%"
                        elif validation == 'strong_kalshi_higher':
                            kalshi_display = f"{source_prefix}🟢 {kalshi_prob:.1f}%"
                        elif validation == 'kalshi_lower':
                            kalshi_display = f"{source_prefix}📉 {kalshi_prob:.1f}%"
                        elif validation == 'strong_contradiction':
                            kalshi_display = f"{source_prefix}⚠️ {kalshi_prob:.1f}%"
                        else:
                            kalshi_display = f"{source_prefix}{kalshi_prob:.1f}%"

                        # Show how Kalshi influenced AI probability
                        if 'kalshi_influence' in leg:
                            influence = leg['kalshi_influence'] * 100
                            if abs(influence) > 0.1:
                                kalshi_influence_display = f"{influence:+.1f}%"
                            else:
                                kalshi_influence_display = "—"
                        else:
                            kalshi_influence_display = "—"
                    else:
                        kalshi_influence_display = "—"

                apisports_display = "—"
                apisports_details = leg.get('apisports')
                if isinstance(apisports_details, dict) and apisports_details:
                    parts = []
                    record = apisports_details.get('team_record')
                    if record:
                        parts.append(f"Record {record}")
                    trend = apisports_details.get('trend')
                    if trend:
                        icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(trend, '📊')
                        parts.append(f"{icon} {trend.capitalize()}")
                    avg_for = apisports_details.get('team_avg_points_for')
                    if isinstance(avg_for, (int, float)):
                        parts.append(f"{avg_for:.1f} pts for")
                    avg_against = apisports_details.get('team_avg_points_against')
                    if isinstance(avg_against, (int, float)):
                        parts.append(f"{avg_against:.1f} pts allowed")
                    status = apisports_details.get('status')
                    if status:
                        parts.append(status)
                    kickoff = apisports_details.get('kickoff')
                    if kickoff:
                        parts.append(kickoff)
                    apisports_display = " | ".join(parts) if parts else "Live data"

                leg_entry = {
                    "Leg": j,
                    "Type": leg["market"],
                    "Selection": leg["label"],
                    "Odds": f"{leg['d']:.3f}",
                    "Market %": f"{leg['p']*100:.1f}%",
                    "AI % (final)": f"{leg.get('ai_prob', leg['p'])*100:.1f}%",
                    "Kalshi": kalshi_display,
                    "K Impact": kalshi_influence_display,
                    "Sentiment": leg.get('sentiment_trend', 'N/A'),
                    "API-Sports": apisports_display,
                    "theover.ai": theover_display
                }
                legs_data.append(leg_entry)
            
            st.dataframe(pd.DataFrame(legs_data), use_container_width=True, hide_index=True)

            # Kalshi impact legend
            if any(leg.get('kalshi_validation', {}).get('kalshi_available') for leg in row.get("legs", [])):
                st.caption("**K Impact** = How much Kalshi adjusted AI probability (blended 50% AI + 30% Kalshi + 20% Market)")
            if any(leg.get('apisports') for leg in row.get("legs", [])):
                st.caption("**API-Sports** = Live NFL data (record, form, kickoff) from api-sports.io")

            # Show legend and summary
            if has_theover:
                col_legend1, col_legend2 = st.columns(2)
                with col_legend1:
                    st.caption("✅ = Matches theover.ai pick | ⚠️ = Conflicts with theover.ai | 🎯 = theover.ai data available")
                with col_legend2:
                    if theover_matches > 0:
                        st.success(f"✅ {theover_matches} leg(s) match theover.ai recommendations")
                    if theover_conflicts > 0:
                        st.warning(f"⚠️ {theover_conflicts} leg(s) conflict with theover.ai recommendations")
            
            # Kalshi validation legend and detailed influence
            kalshi_available = sum(1 for leg in row.get("legs", []) if leg.get('kalshi_validation', {}).get('kalshi_available', False))
            
            # Show info if Kalshi enabled but no data found
            if 'kalshi_validation' in row.get("legs", [{}])[0] and kalshi_available == 0:
                st.info("""
                **📊 Kalshi Validation Enabled** but no matching markets found for these games.
                
                **This is normal because:**
                - Kalshi may not have markets for all games
                - Markets might be for season-long outcomes (playoffs, championships) rather than individual games
                - Try checking Tab 4 to see what Kalshi markets are actually available
                
                **Note:** Parlay analysis still uses AI + Sentiment, just without Kalshi cross-validation.
                """)
            
            if kalshi_available > 0:
                kalshi_confirmed = sum(1 for leg in row.get("legs", []) if leg.get('kalshi_validation', {}).get('validation') == 'confirms')
                kalshi_higher = sum(1 for leg in row.get("legs", []) if 'higher' in leg.get('kalshi_validation', {}).get('validation', ''))
                kalshi_contradicts = sum(1 for leg in row.get("legs", []) if 'contradiction' in leg.get('kalshi_validation', {}).get('validation', ''))
                
                st.markdown("---")
                st.markdown("**📊 Kalshi Prediction Market Validation:**")
                
                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    st.caption("**Legend:** ✅ = Confirms | 📈 = Kalshi higher | 📉 = Kalshi lower | 🟢 = Strong value | ⚠️ = Contradiction")
                with col_k2:
                    if kalshi_confirmed > 0:
                        st.success(f"✅ {kalshi_confirmed} leg(s) confirmed by Kalshi")
                    if kalshi_higher > 0:
                        st.info(f"📈 {kalshi_higher} leg(s) show Kalshi value")
                    if kalshi_contradicts > 0:
                        st.warning(f"⚠️ {kalshi_contradicts} leg(s) contradicted by Kalshi")
                
                # Detailed Kalshi Influence Analysis
                st.markdown("**🔍 How Kalshi Influenced This Parlay:**")
                
                total_confidence_boost = 0
                total_kalshi_edge = 0
                kalshi_details = []
                
                for j, leg in enumerate(row.get("legs", []), start=1):
                    kv = leg.get('kalshi_validation', {})
                    if kv.get('kalshi_available'):
                        kalshi_prob = kv.get('kalshi_prob', 0)
                        validation = kv.get('validation', 'unavailable')
                        confidence_boost = kv.get('confidence_boost', 0)
                        edge = kv.get('edge', 0)
                        market_ticker = kv.get('market_ticker', 'N/A')
                        
                        total_confidence_boost += confidence_boost
                        total_kalshi_edge += edge
                        
                        # Create status icon
                        if validation == 'confirms':
                            status_icon = "✅"
                            status_text = "CONFIRMS"
                            status_color = "green"
                        elif validation == 'strong_kalshi_higher':
                            status_icon = "🟢"
                            status_text = "STRONG VALUE"
                            status_color = "green"
                        elif 'higher' in validation:
                            status_icon = "📈"
                            status_text = "KALSHI HIGHER"
                            status_color = "blue"
                        elif 'contradiction' in validation:
                            status_icon = "⚠️"
                            status_text = "CONTRADICTION"
                            status_color = "red"
                        else:
                            status_icon = "📉"
                            status_text = "KALSHI LOWER"
                            status_color = "orange"
                        
                        sportsbook_prob = leg.get('p', 0) * 100
                        kalshi_prob_pct = kalshi_prob * 100
                        discrepancy = abs(kalshi_prob - leg.get('p', 0)) * 100
                        
                        kalshi_details.append({
                            'Leg': j,
                            'Pick': leg.get('team', 'N/A'),
                            'Status': f"{status_icon} {status_text}",
                            'Sportsbook': f"{sportsbook_prob:.1f}%",
                            'Kalshi': f"{kalshi_prob_pct:.1f}%",
                            'Discrepancy': f"{discrepancy:.1f}%",
                            'Confidence Boost': f"{confidence_boost*100:+.0f}%",
                            'Edge': f"{edge*100:+.1f}%",
                            'Market': market_ticker[:20]
                        })
                
                if kalshi_details:
                    st.dataframe(pd.DataFrame(kalshi_details), use_container_width=True, hide_index=True)
                    
                    # Summary metrics
                    st.markdown("**📈 Kalshi Impact Summary:**")
                    col_impact1, col_impact2, col_impact3, col_impact4 = st.columns(4)
                    
                    with col_impact1:
                        st.metric(
                            "Legs Validated",
                            f"{kalshi_available}/{len(row.get('legs', []))}",
                            help="How many legs have Kalshi market data"
                        )
                    
                    with col_impact2:
                        st.metric(
                            "Total Confidence Boost",
                            f"{total_confidence_boost*100:+.0f}%",
                            help="How much Kalshi boosted overall confidence"
                        )
                    
                    with col_impact3:
                        st.metric(
                            "Additional Edge",
                            f"{total_kalshi_edge*100:+.1f}%",
                            help="Extra edge identified from Kalshi vs sportsbook"
                        )
                    
                    with col_impact4:
                        avg_discrepancy = sum(abs(leg.get('kalshi_validation', {}).get('discrepancy', 0)) 
                                            for leg in row.get("legs", []) 
                                            if leg.get('kalshi_validation', {}).get('kalshi_available')) / max(kalshi_available, 1)
                        st.metric(
                            "Avg Discrepancy",
                            f"{avg_discrepancy*100:.1f}%",
                            help="Average difference between Kalshi and sportsbook"
                        )
                    
                    # Interpretation
                    st.markdown("**💡 Interpretation:**")
                    
                    if total_confidence_boost >= 0.15:
                        st.success("🟢 **STRONG KALSHI CONFIRMATION** - All sources strongly agree. High confidence bet!")
                    elif total_confidence_boost >= 0.05:
                        st.info("🟡 **MODERATE CONFIRMATION** - Kalshi generally agrees. Good bet with decent validation.")
                    elif total_confidence_boost >= -0.05:
                        st.warning("🟠 **NEUTRAL VALIDATION** - Kalshi shows mixed signals. Proceed with caution.")
                    else:
                        st.error("🔴 **KALSHI DISAGREES** - Prediction market contradicts this parlay. Consider skipping or investigating further.")
                    
                    if total_kalshi_edge > 0.10:
                        st.success(f"💰 **VALUE DETECTED**: Kalshi shows {total_kalshi_edge*100:.1f}% additional edge! This parlay may be underpriced by sportsbooks.")
                    elif total_kalshi_edge < -0.10:
                        st.warning(f"⚠️ **OVERPRICED WARNING**: Kalshi thinks this parlay is overpriced. Sportsbooks may be offering poor value.")
                    
                    # Recommendation based on Kalshi
                    kalshi_score = (total_confidence_boost * 50) + (total_kalshi_edge * 30) + (kalshi_confirmed / max(kalshi_available, 1) * 20)
                    
                    st.markdown("**🎯 Kalshi-Based Recommendation:**")
                    if kalshi_score > 15:
                        st.success("✅ **KALSHI APPROVES** - Strong validation from prediction markets. Excellent bet!")
                    elif kalshi_score > 5:
                        st.info("🟡 **KALSHI CAUTIOUS** - Some validation but mixed signals. Decent bet if AI score is high.")
                    else:
                        st.warning("⚠️ **KALSHI SKEPTICAL** - Prediction market doesn't support this parlay. Bet with caution or skip.")
            
            # Betting scenarios
            st.markdown("**💵 Betting Scenarios:**")
            for stake in [50, 100, 250, 500]:
                profit_amt = calculate_profit(row['d'], stake)
                payout = stake + profit_amt
                exp_return = stake * (1 + row['ev_ai'])
                st.write(f"${stake} bet → ${payout:.2f} payout | Expected return: ${exp_return:.2f}")
            
            # CSV export
            csv_buf = io.StringIO()
            df_export = pd.DataFrame(legs_data)
            df_export.to_csv(csv_buf, index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"ai_parlay_{i}.csv",
                mime="text/csv",
                key=f"download_ai_{title}_{i}"
            )

# ============ STREAMLIT UI ============
st.set_page_config(page_title=APP_CFG["title"], layout="wide")
st.title("🤖 " + APP_CFG["title"])
st.caption("AI-powered parlay finder with sentiment analysis and machine learning predictions")

# Initialize AI components
if 'sentiment_analyzer' not in st.session_state:
    news_key = os.environ.get("NEWS_API_KEY", "")
    st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_key)
    st.session_state['news_api_key'] = news_key
if 'ml_predictor' not in st.session_state:
    st.session_state['ml_predictor'] = MLPredictor()
if 'ai_optimizer' not in st.session_state:
    st.session_state['ai_optimizer'] = AIOptimizer(
        st.session_state['sentiment_analyzer'],
        st.session_state['ml_predictor']
    )
if 'news_api_key' not in st.session_state:
    st.session_state['news_api_key'] = os.environ.get("NEWS_API_KEY", "")

# Initialize advanced analyzers
if 'sharp_detector' not in st.session_state:
    st.session_state['sharp_detector'] = SharpMoneyDetector()
if 'player_impact' not in st.session_state:
    st.session_state['player_impact'] = PlayerImpactAnalyzer()
if 'weather_analyzer' not in st.session_state:
    weather_key = os.environ.get("WEATHER_API_KEY", "")
    st.session_state['weather_analyzer'] = WeatherAnalyzer(weather_key)
if 'kelly_calculator' not in st.session_state:
    st.session_state['kelly_calculator'] = KellyCalculator()
if 'matchup_analyzer' not in st.session_state:
    st.session_state['matchup_analyzer'] = MatchupAnalyzer()
if 'advanced_stats' not in st.session_state:
    st.session_state['advanced_stats'] = AdvancedStatsIntegrator()
if 'social_analyzer' not in st.session_state:
    twitter_key = os.environ.get("TWITTER_API_KEY", "")
    st.session_state['social_analyzer'] = SocialMediaAnalyzer(twitter_key)
if 'kalshi_integrator' not in st.session_state:
    kalshi_key = os.environ.get("KALSHI_API_KEY", "")
    kalshi_secret = os.environ.get("KALSHI_API_SECRET", "")
    st.session_state['kalshi_integrator'] = KalshiIntegrator(kalshi_key, kalshi_secret)
if 'apisports_client' not in st.session_state:
    st.session_state['apisports_client'] = APISportsFootballClient()

# Main navigation tabs
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "🎯 Sports Betting Parlays",
    "🔍 Sentiment & AI Analysis",
    "🎨 Custom Parlay Builder",
    "📊 Kalshi Prediction Markets",
    "🏈 API-Sports NFL Live Data"
])

# ===== TAB 1: SPORTS BETTING PARLAYS =====
with main_tab1:
    apisports_client = st.session_state.get('apisports_client')
    if apisports_client is None:
        apisports_client = APISportsFootballClient()
        st.session_state['apisports_client'] = apisports_client

    # API Configuration
    stored_key = os.environ.get("ODDS_API_KEY", "")

    if 'api_key' in st.session_state:
        key = st.session_state['api_key']
    elif stored_key:
        key = stored_key
    else:
        key = ""

    if not key:
        key = st.text_input(
            "TheOddsAPI key (first time setup)", 
            value="",
            type="password",
            help="This will be remembered for future sessions"
        )
        if key:
            st.session_state['api_key'] = key
            st.success("✅ API key saved for this session!")
            st.rerun()
    else:
        if 'show_api_section' not in st.session_state:
            st.session_state['show_api_section'] = False
        
        if st.session_state['show_api_section']:
            new_key = st.text_input(
                "Update API key", 
                value="",
                type="password"
            )
            if new_key:
                st.session_state['api_key'] = new_key
                st.session_state['show_api_section'] = False
                st.success("✅ API key updated!")
                st.rerun()
        else:
            col_api1, col_api2 = st.columns([4, 1])
            with col_api1:
                st.success("🔑 API key is configured")
            with col_api2:
                if st.button("Change key"):
                    st.session_state['show_api_section'] = True
                    st.rerun()
    
    # News API Configuration (for real sentiment)
    st.markdown("---")
    st.markdown("### 📰 Real Sentiment Analysis (Optional)")
    
    col_news1, col_news2 = st.columns([3, 1])
    with col_news1:
        if not st.session_state.get('news_api_key'):
            with st.expander("ℹ️ Enable REAL Sentiment Analysis"):
                st.info("""
                **Upgrade to real sentiment analysis!**
                
                Currently using: Neutral placeholders (no real analysis)
                
                **With NewsAPI:**
                - ✅ Analyzes actual news articles
                - ✅ Real NLP sentiment scoring
                - ✅ Last 3 days of team news
                - ✅ Free tier: 100 requests/day
                
                **Get your free API key:**
                1. Visit [newsapi.org](https://newsapi.org/)
                2. Sign up (takes 1 minute)
                3. Copy your API key
                4. Paste below
                """)
                news_key_input = st.text_input(
                    "NewsAPI Key",
                    type="password",
                    help="Get free at https://newsapi.org/"
                )
                if news_key_input:
                    st.session_state['news_api_key'] = news_key_input
                    # Reinitialize sentiment analyzer with new key
                    st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_key_input)
                    st.success("✅ Real sentiment analysis enabled!")
                    st.rerun()
        else:
            st.success("📰 Real Sentiment Analysis: ENABLED")
            st.caption("Analyzing actual news articles for team sentiment")
            
    with col_news2:
        if st.session_state.get('news_api_key'):
            if st.button("Disable", key="disable_news_api"):
                st.session_state['news_api_key'] = ""
                st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(None)
                st.info("Switched to neutral sentiment (no API)")
                st.rerun()

    # theover.ai Integration Section
    st.markdown("---")
    st.markdown("### 📊 theover.ai Integration (Optional)")
    
    theover_method = st.radio(
        "Add theover.ai projections for enhanced analysis?",
        ["🚫 Skip", "📁 Upload CSV", "📋 Paste Data"],
        horizontal=True,
        help="theover.ai projections will be matched with parlay legs to show additional analysis"
    )
    
    theover_parlay_data = None
    
    if theover_method == "📁 Upload CSV":
        st.info("""
        **Expected CSV Format:**
        - Columns: Team/Player, Stat/Market, Projection
        - Example: "Lakers, Points Total, 112.5" or "Bucks ML, Win Probability, 0.65"
        """)
        
        uploaded_file = st.file_uploader(
            "Upload theover.ai CSV export",
            type=['csv'],
            help="Export your theover.ai projections as CSV",
            key="theover_parlay_upload"
        )
        
        if uploaded_file:
            try:
                theover_parlay_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(theover_parlay_data)} projections from theover.ai")
                
                with st.expander("📋 Preview theover.ai Data"):
                    st.dataframe(theover_parlay_data.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                
    elif theover_method == "📋 Paste Data":
        st.info("""
        **Paste Format (comma or tab-separated):**
        ```
        Team/Player,Stat/Market,Projection
        Lakers,Points Total,112.5
        Bucks,Win Probability,0.68
        ```
        """)
        
        pasted_data = st.text_area(
            "Paste theover.ai data here",
            height=150,
            placeholder="Lakers,Points Total,112.5\nBucks,Win Probability,0.68",
            key="theover_parlay_paste"
        )
        
        if pasted_data.strip():
            try:
                # Try comma-separated first
                if ',' in pasted_data:
                    from io import StringIO
                    theover_parlay_data = pd.read_csv(StringIO(pasted_data))
                # Try tab-separated (from Excel)
                elif '\t' in pasted_data:
                    from io import StringIO
                    theover_parlay_data = pd.read_csv(StringIO(pasted_data), sep='\t')
                else:
                    st.warning("Data format not recognized. Use comma or tab-separated values.")
                
                if theover_parlay_data is not None:
                    st.success(f"✅ Loaded {len(theover_parlay_data)} projections from theover.ai")
                    with st.expander("📋 Preview Pasted Data"):
                        st.dataframe(theover_parlay_data.head(10), use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error parsing data: {e}")
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        tz_name = st.text_input("Timezone (IANA)", value="America/New_York")
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone("UTC")
            st.warning("Invalid timezone; using UTC")
        st.session_state['user_timezone'] = getattr(tz, 'zone', tz_name)

    with col2:
        sel_date = st.date_input(
            "Only events on date (local to timezone)",
            value=pd.Timestamp.now(tz).date()
        )

        # Historical window slider
        _day_window = st.slider(
            "Include events within ±N days",
            0, 7, 0, 1,
            help="Leverage historical odds snapshots."
        )

    # Sport Selection
    sports = st.multiselect(
        "Sports keys", 
        options=APP_CFG["sports_common"], 
        default=APP_CFG["sports_common"][:6]
    )

    # AI Settings
    with st.expander("⚙️ AI Settings", expanded=False):
        st.markdown("### Machine Learning Configuration")
        
        st.info("""
        **✨ HIGH CONFIDENCE BETS MODE**
        
        - Minimum AI confidence: **60%** (high confidence only)
        - Probability range: **30-65%** (value zone, not chalk)
        - Ranked by: **Expected Value** (AI edge over market)
        - Sentiment weight: **40%** (significant factor)
        
        **Strategy: Only bet on high-confidence value picks** 🎯
        """)
        
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            use_sentiment = st.checkbox("Enable Sentiment Analysis", value=True, 
                                        help="Analyze news and social media sentiment")
            use_ml_predictions = st.checkbox("Enable ML Predictions", value=True,
                                            help="Use machine learning for probability adjustments")
        with col_ai2:
            min_ai_confidence = st.slider("Minimum AI Confidence", 0.0, 1.0, 0.60, 0.05,
                                          help="Filter out low-confidence predictions (0.60 = 60% confidence minimum)")
            min_parlay_probability = st.slider(
                "Minimum Parlay Probability", 
                0.20, 0.60, 0.30, 0.05,
                help="Filter out longshot parlays (0.30 = 30% min chance for high confidence)"
            )
            max_parlay_probability = st.slider(
                "Maximum Parlay Probability",
                0.45, 0.85, 0.65, 0.05,
                help="Exclude heavy favorites (0.65 = 65% max, keeps value plays only)"
            )
            
        st.caption("🔴 = Too risky (<30%) | 🟡 = Moderate (30-50%) | 🟢 = High confidence value (50-65%) | ❌ = Too safe (>65%)")

    col3, col4, col5 = st.columns(3)
    with col3:
        per_sport_events = st.slider("Max events per sport", 3, 50, 12, 1)
    with col4:
        show_top = st.slider("Show top N combos", 1, 50, 15, 1)
    with col5:
        allow_sgp = st.checkbox("Allow same-game legs", value=False)
    
    st.info("💡 **Duplicate Removal:** Identical bets from different bookmakers are automatically deduplicated, keeping only the best odds.")

    # Bet type filters
    st.subheader("Include Bet Types")
    col_bet1, col_bet2, col_bet3 = st.columns(3)
    with col_bet1:
        inc_ml = st.checkbox("Moneyline", value=True)
    with col_bet2:
        inc_spread = st.checkbox("Spreads", value=True)
    with col_bet3:
        inc_total = st.checkbox("Totals (O/U)", value=True)

    # Kalshi Validation Option
    st.markdown("---")
    st.subheader("📊 Kalshi Prediction Market Validation")
    use_kalshi = st.checkbox(
        "✅ Validate with Kalshi (Prediction Market Cross-Check)",
        value=False,
        help="Compare sportsbook odds with Kalshi prediction markets to find discrepancies and boost confidence"
    )

    st.session_state['kalshi_enabled'] = use_kalshi

    if use_kalshi:
        st.info("""
        **Kalshi Validation Benefits:**
        - 🎯 Cross-validates odds with prediction markets
        - 📈 Boosts confidence when markets agree (up to +15%)
        - ⚠️ Flags contradictions when markets disagree
        - 💎 Identifies arbitrage opportunities
        - 🔍 Shows additional edge from market discrepancies
        
        **Important Note:** Kalshi typically has markets for:
        - Season-long outcomes (playoffs, championships, win totals)
        - Major events and marquee matchups
        - NOT every individual regular season game
        
        If you see "—" in the Kalshi column, it means no matching market was found for that specific game.
        This is normal and expected. The parlay analysis will still work using AI + Sentiment.
        
        **Tip:** Check Tab 4 to see what Kalshi markets are actually available right now.
        """)
        
        with st.expander("ℹ️ Understanding Kalshi Market Coverage"):
            st.markdown("""
            **What Kalshi Markets Are Available:**
            
            ✅ **Common Kalshi Markets:**
            - "Will [Team] make the playoffs?"
            - "Will [Team] win the Super Bowl/Championship?"
            - "Will [Team] win more than X games?"
            - "Will [Player] win MVP?"
            - Major rivalry games or primetime matchups
            
            ❌ **NOT Usually Available:**
            - Individual regular season game outcomes
            - Week-to-week game spreads
            - Game totals (over/under)
            - Every team's games
            
            **Why This Matters:**
            - Your parlay might have 0 Kalshi matches → That's OK!
            - Kalshi is a bonus validation source, not required
            - AI + Sentiment still provide strong analysis
            
            **When Kalshi DOES Match:**
            - You get extra confidence boost
            - Can spot market inefficiencies
            - Valuable cross-validation
            
            **Recommendation:**
            - Leave Kalshi validation ON (no harm if no matches)
            - When it finds matches, that's a bonus!
            - Don't expect matches for every game
            """)

    st.markdown("---")
    st.subheader("🏈 API-Sports NFL Data Integration")
    current_api_sports_key = st.session_state.get('apisports_api_key', apisports_client.api_key if apisports_client else "")
    new_api_sports_key = st.text_input(
        "API-Sports Key (American football)",
        value=current_api_sports_key,
        type="password",
        help="Create a free account at https://api-sports.io/documentation/american-football/v1 to obtain a key"
    )
    if new_api_sports_key != current_api_sports_key:
        st.session_state['apisports_api_key'] = new_api_sports_key
        if apisports_client:
            apisports_client.update_api_key(new_api_sports_key)
        if new_api_sports_key:
            st.success("✅ API-Sports key saved for this session.")
        else:
            st.info("API-Sports integration disabled until a key is provided.")

    
    def is_within_date_window(iso_str) -> bool:
        """Return True when an event falls within the selected day ± window."""
        try:
            ts_local = pd.to_datetime(iso_str, utc=True).tz_convert(tz)
            event_date = ts_local.date()
            delta_days = (event_date - sel_date).days
            return -_day_window <= delta_days <= _day_window
        except Exception:
            return False

    if st.button("🤖 Find AI-Optimized Parlays", type="primary"):
        # Get API key from session state or environment only
        api_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        
        if not api_key:
            st.error("No API key provided. Please enter your API key above.")
            st.stop()
        
        if not (inc_ml or inc_spread or inc_total):
            st.error("Please select at least one bet type")
            st.stop()
        
        try:
            # Safely get AI components
            try:
                sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                ml_predictor = st.session_state.get('ml_predictor')
                ai_optimizer = st.session_state.get('ai_optimizer')
                
                if not sentiment_analyzer or not ml_predictor or not ai_optimizer:
                    st.error("AI components not initialized. Please refresh the page.")
                    st.stop()
            except Exception as e:
                st.error(f"Error accessing AI components: {str(e)}")
                st.stop()
        
            with st.spinner("🧠 Analyzing markets with AI..."):
                progress_bar = st.progress(0)
                all_legs = []
                apisports_games_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
                total_sports = len(sports or APP_CFG["sports_common"])
                
                for sport_idx, skey in enumerate(sports or APP_CFG["sports_common"]):
                    try:
                        progress_bar.progress((sport_idx) / total_sports)
                        snap = fetch_oddsapi_snapshot(api_key, skey)
                        
                        if not snap or not snap.get("events"):
                            continue  # Skip if no events
                        
                        for ev in (snap.get("events") or [])[:per_sport_events]:
                            try:
                                if not is_within_date_window(ev.get("commence_time")):
                                    continue
                                
                                eid = ev.get("id")
                                home = ev.get("home_team", "?")
                                away = ev.get("away_team", "?")
                                mkts = ev.get("markets") or {}
                                
                                if not eid or not home or not away:
                                    continue  # Skip invalid events

                                apisports_summary = None
                                apisports_payload_home = None
                                apisports_payload_away = None
                                apisports_payload_total = None

                                if skey == "americanfootball_nfl" and apisports_client and apisports_client.is_configured():
                                    try:
                                        event_ts = pd.to_datetime(ev.get("commence_time"), utc=True)
                                        tz_label = st.session_state.get('user_timezone') or getattr(tz, 'zone', 'UTC') or 'UTC'
                                        try:
                                            target_tz = pytz.timezone(tz_label)
                                        except Exception:
                                            target_tz = pytz.timezone('UTC')
                                        local_date = event_ts.tz_convert(target_tz).date()
                                        cache_key = (local_date.isoformat(), tz_label)

                                        if cache_key not in apisports_games_cache:
                                            apisports_games_cache[cache_key] = apisports_client.get_games_by_date(
                                                local_date,
                                                timezone=tz_label,
                                            )

                                        matched_game = apisports_client.match_game(
                                            apisports_games_cache.get(cache_key, []),
                                            home,
                                            away,
                                        )

                                        if matched_game:
                                            apisports_summary = apisports_client.build_game_summary(
                                                matched_game,
                                                tz_name=tz_label,
                                            )
                                            apisports_payload_home = build_leg_apisports_payload(apisports_summary, 'home')
                                            apisports_payload_away = build_leg_apisports_payload(apisports_summary, 'away')
                                            total_trend = apisports_payload_home.get('trend') or apisports_payload_away.get('trend')
                                            apisports_payload_total = {
                                                key: getattr(apisports_summary, attr)
                                                for key, attr in [
                                                    ('game_id', 'id'),
                                                    ('league', 'league'),
                                                    ('season', 'season'),
                                                    ('status', 'status'),
                                                    ('kickoff', 'kickoff_local'),
                                                    ('venue', 'venue'),
                                                ]
                                                if getattr(apisports_summary, attr, None)
                                            }
                                            if total_trend:
                                                apisports_payload_total['trend'] = total_trend
                                    except Exception:
                                        apisports_summary = None
                                        apisports_payload_home = None
                                        apisports_payload_away = None
                                        apisports_payload_total = None

                                # Get sentiment for both teams
                                try:
                                    home_sentiment = sentiment_analyzer.get_team_sentiment(home, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
                                    away_sentiment = sentiment_analyzer.get_team_sentiment(away, skey) if use_sentiment else {'score': 0, 'trend': 'neutral'}
                                except Exception:
                                    home_sentiment = {'score': 0, 'trend': 'neutral'}
                                    away_sentiment = {'score': 0, 'trend': 'neutral'}
                                
                                # Moneyline with AI
                                if inc_ml and "h2h" in mkts:
                                    hp = _dig(mkts["h2h"], "home.price")
                                    ap = _dig(mkts["h2h"], "away.price")
                                    
                                    if hp is not None and -750 <= hp <= 750:
                                        base_prob = implied_p_from_american(hp)
                                        ai_prob = base_prob
                                        
                                        if use_ml_predictions and ap is not None:
                                            ml_prediction = ml_predictor.predict_game_outcome(
                                                home, away, hp, ap,
                                                home_sentiment['score'], away_sentiment['score']
                                            )
                                            ai_prob = ml_prediction['home_prob']
                                            ai_confidence = ml_prediction['confidence']
                                            ai_edge = ml_prediction['edge']
                                        else:
                                            ai_confidence = 0.5
                                            ai_edge = 0
                                        
                                        if ai_confidence >= min_ai_confidence:
                                            decimal_odds = american_to_decimal_safe(hp)
                                            if decimal_odds is not None:  # Safety check
                                                leg_data = {
                                                    "event_id": eid,
                                                    "type": "Moneyline",
                                                    "team": home,
                                                    "side": "home",
                                                    "market": "ML",
                                                    "label": f"{away} @ {home} — {home} ML @{hp}",
                                                    "p": base_prob,
                                                    "ai_prob": ai_prob,
                                                    "ai_confidence": ai_confidence,
                                                    "ai_edge": ai_edge,
                                                    "d": decimal_odds,
                                                    "sentiment_trend": home_sentiment['trend']
                                                }

                                                if apisports_payload_home:
                                                    leg_data['apisports'] = apisports_payload_home

                                                integrate_kalshi_into_leg(
                                                    leg_data,
                                                    home,
                                                    away,
                                                    'home',
                                                    base_prob,
                                                    skey,
                                                    use_kalshi
                                                )

                                                all_legs.append(leg_data)
                            
                                    if ap is not None and -750 <= ap <= 750:
                                        base_prob = implied_p_from_american(ap)
                                        ai_prob = base_prob
                                        
                                        if use_ml_predictions and hp is not None:
                                            ml_prediction = ml_predictor.predict_game_outcome(
                                                home, away, hp, ap,
                                                home_sentiment['score'], away_sentiment['score']
                                            )
                                            ai_prob = ml_prediction['away_prob']
                                            ai_confidence = ml_prediction['confidence']
                                            ai_edge = ml_prediction['edge']
                                        else:
                                            ai_confidence = 0.5
                                            ai_edge = 0
                                        
                                        if ai_confidence >= min_ai_confidence:
                                            decimal_odds = american_to_decimal_safe(ap)
                                            if decimal_odds is not None:  # Safety check
                                                leg_data = {
                                                    "event_id": eid,
                                                    "type": "Moneyline",
                                                    "team": away,
                                                    "side": "away",
                                                    "market": "ML",
                                                    "label": f"{away} @ {home} — {away} ML @{ap}",
                                                    "p": base_prob,
                                                    "ai_prob": ai_prob,
                                                    "ai_confidence": ai_confidence,
                                                    "ai_edge": ai_edge,
                                                    "d": decimal_odds,
                                                    "sentiment_trend": away_sentiment['trend']
                                                }

                                                if apisports_payload_away:
                                                    leg_data['apisports'] = apisports_payload_away

                                                integrate_kalshi_into_leg(
                                                    leg_data,
                                                    home,
                                                    away,
                                                    'away',
                                                    base_prob,
                                                    skey,
                                                    use_kalshi
                                                )

                                                all_legs.append(leg_data)
                                
                                # Spreads
                                if inc_spread and "spreads" in mkts:
                                    for o in mkts["spreads"][:4]:
                                        nm, pt, pr = o.get("name"), o.get("point"), o.get("price")
                                        if nm is None or pt is None or pr is None: 
                                            continue
                                        
                                        base_prob = implied_p_from_american(pr)
                                        sentiment = home_sentiment if nm == home else away_sentiment
                                        
                                        # Simple sentiment adjustment for spreads (40% weight)
                                        ai_prob = base_prob * (1 + sentiment['score'] * 0.40)
                                        ai_prob = max(0.1, min(0.9, ai_prob))  # Clamp
                                        
                                        # Higher confidence for spread bets (adjusted from 0.6 to 0.65)
                                        ai_confidence = 0.65
                                        
                                        decimal_odds = american_to_decimal_safe(pr)
                                        if decimal_odds is not None and ai_confidence >= min_ai_confidence:  # Safety check
                                            leg_data = {
                                                "event_id": eid,
                                                "type": "Spread",
                                                "team": nm,
                                                "side": "home" if nm == home else "away",
                                                "point": pt,
                                                "market": "Spread",
                                                "label": f"{away} @ {home} — {nm} {pt:+.1f} @{pr}",
                                                "p": base_prob,
                                                "ai_prob": ai_prob,
                                                "ai_confidence": ai_confidence,
                                                "ai_edge": abs(ai_prob - base_prob),
                                                "d": decimal_odds,
                                                "sentiment_trend": sentiment['trend']
                                            }

                                            if nm == home and apisports_payload_home:
                                                leg_data['apisports'] = apisports_payload_home
                                            elif nm == away and apisports_payload_away:
                                                leg_data['apisports'] = apisports_payload_away

                                            integrate_kalshi_into_leg(
                                                leg_data,
                                                home,
                                                away,
                                                'home' if nm == home else 'away',
                                                base_prob,
                                                skey,
                                                use_kalshi
                                            )

                                            all_legs.append(leg_data)
                                
                                # Totals
                                if inc_total and "totals" in mkts:
                                    for o in mkts["totals"][:4]:
                                        nm, pt, pr = o.get("name"), o.get("point"), o.get("price")
                                        if nm is None or pt is None or pr is None: 
                                            continue
                                        
                                        base_prob = implied_p_from_american(pr)
                                        
                                        # For totals, combine both teams' offensive sentiment (40% weight)
                                        combined_sentiment = (home_sentiment['score'] + away_sentiment['score']) / 2
                                        ai_prob = base_prob * (1 + combined_sentiment * 0.40 * 0.5)
                                        ai_prob = max(0.1, min(0.9, ai_prob))
                                        
                                        # Higher confidence for totals (adjusted from 0.55 to 0.60)
                                        ai_confidence = 0.60
                                        
                                        decimal_odds = american_to_decimal_safe(pr)
                                        if decimal_odds is not None and ai_confidence >= min_ai_confidence:  # Safety check
                                            leg_data = {
                                                "event_id": eid,
                                                "type": "Total",
                                                "team": f"{home} vs {away}",
                                                "side": nm,  # Over or Under
                                                "point": pt,
                                                "market": "Total",
                                                "label": f"{away} @ {home} — {nm} {pt} @{pr}",
                                                "p": base_prob,
                                                "ai_prob": ai_prob,
                                                "ai_confidence": ai_confidence,
                                                "ai_edge": abs(ai_prob - base_prob),
                                                "d": decimal_odds,
                                                "sentiment_trend": "neutral"
                                            }

                                            if apisports_payload_total:
                                                leg_data['apisports'] = apisports_payload_total

                                            if use_kalshi:
                                                leg_data['kalshi_validation'] = {
                                                    'kalshi_available': False,
                                                    'validation': 'unsupported',
                                                    'edge': 0,
                                                    'confidence_boost': 0,
                                                    'market_scope': 'total_market',
                                                    'data_source': 'unsupported',
                                                    'reason': 'Kalshi does not list totals/over-under style markets'
                                                }
                                            else:
                                                leg_data['kalshi_validation'] = {
                                                    'kalshi_available': False,
                                                    'validation': 'disabled',
                                                    'edge': 0,
                                                    'confidence_boost': 0,
                                                    'market_scope': 'disabled',
                                                    'data_source': 'disabled'
                                                }

                                            all_legs.append(leg_data)
                            
                            except Exception as e:
                                # Skip this event if there's an error processing it
                                continue
                    
                    except Exception as e:
                        # Skip this sport if there's an error
                        st.warning(f"Error processing {skey}: {str(e)[:100]}")
                        continue
                
                progress_bar.progress(1.0)
                
                if not all_legs:
                    st.warning("No bets found for selected date/sports/filters")
                    st.stop()
                
                # AI Summary
                st.success(f"🤖 AI Analysis Complete: Found {len(all_legs)} betting opportunities")
                
                # Show AI insights
                with st.expander("📊 AI Market Analysis", expanded=True):
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    high_confidence = [leg for leg in all_legs if leg.get('ai_confidence', 0) > 0.7]
                    positive_ev = [leg for leg in all_legs if leg.get('ai_prob', 0) * leg['d'] > 1.05]
                    sentiment_edge = [leg for leg in all_legs if leg.get('sentiment_trend') == 'positive']
                    
                    with col_insight1:
                        st.metric("High Confidence Bets", len(high_confidence), 
                                 help="Bets with >70% AI confidence")
                    with col_insight2:
                        st.metric("Positive AI EV Bets", len(positive_ev),
                                 help="Bets with >5% expected value")
                    with col_insight3:
                        st.metric("Positive Sentiment", len(sentiment_edge),
                                 help="Teams with positive news sentiment")
                
                # Create tabs for different parlay sizes
                tab_2, tab_3, tab_4, tab_5 = st.tabs([
                    "🎯 2-Leg Parlays", 
                    "🎲 3-Leg Parlays", 
                    "🚀 4-Leg Parlays", 
                    "💎 5-Leg Parlays"
                ])
                
                with tab_2:
                    st.subheader("Best 2-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 2-leg combinations..."):
                            combos_2 = build_combos_ai(all_legs, 2, allow_sgp, ai_optimizer, theover_parlay_data, min_parlay_probability, max_parlay_probability)[:show_top]
                            render_parlay_section_ai("2-Leg AI Parlays", combos_2, theover_parlay_data)
                    except Exception as e:
                        st.error(f"Error building 2-leg parlays: {str(e)}")
                
                with tab_3:
                    st.subheader("Best 3-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 3-leg combinations..."):
                            combos_3 = build_combos_ai(all_legs, 3, allow_sgp, ai_optimizer, theover_parlay_data, min_parlay_probability, max_parlay_probability)[:show_top]
                            render_parlay_section_ai("3-Leg AI Parlays", combos_3, theover_parlay_data)
                    except Exception as e:
                        st.error(f"Error building 3-leg parlays: {str(e)}")
                
                with tab_4:
                    st.subheader("Best 4-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 4-leg combinations..."):
                            combos_4 = build_combos_ai(all_legs, 4, allow_sgp, ai_optimizer, theover_parlay_data, min_parlay_probability, max_parlay_probability)[:show_top]
                            render_parlay_section_ai("4-Leg AI Parlays", combos_4, theover_parlay_data)
                    except Exception as e:
                        st.error(f"Error building 4-leg parlays: {str(e)}")
                
                with tab_5:
                    st.subheader("Best 5-Leg AI-Optimized Parlays")
                    try:
                        with st.spinner("Calculating optimal 5-leg combinations..."):
                            combos_5 = build_combos_ai(all_legs, 5, allow_sgp, ai_optimizer, theover_parlay_data, min_parlay_probability, max_parlay_probability)[:show_top]
                            render_parlay_section_ai("5-Leg AI Parlays", combos_5, theover_parlay_data)
                    except Exception as e:
                        st.error(f"Error building 5-leg parlays: {str(e)}")
        
        except KeyError as e:
            st.error(f"Configuration error: Missing key {str(e)}. Please refresh the page.")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            with st.expander("Full Error Details (for debugging)"):
                st.code(traceback.format_exc())
            st.info("Please try:")
            st.markdown("""
            1. Refresh the page
            2. Check your API key is valid
            3. Try selecting fewer sports
            4. Try a different date
            5. Disable sentiment analysis if enabled
            """)
# ---- Global safe odds helper (always available) ------------------------------
def american_to_decimal_safe(odds) -> float | None:
    """
    Safe American→Decimal conversion.
    Returns None for None/0/invalid odds in (-100, 100) or on parsing errors.
    """
    try:
        if odds is None:
            return None
        o = float(odds)
        if abs(o) < 100:
            return None
        if o >= 100:
            return 1.0 + o/100.0
        else:
            return 1.0 + 100.0/abs(o)
    except Exception:
        return None
# -----------------------------------------------------------------------------
# ---- Safety shim to guarantee robust odds conversion ----
try:
    american_to_decimal_safe
except NameError:
    pass
# ---------------------------------------------------------
    
st.markdown("---")
st.markdown("""
    ### 🤖 AI Features Explained:

    **Sentiment Analysis** 🎭
    - Analyzes team news, social media, and expert opinions
    - Positive sentiment = team trending up, negative = trending down
    - Integrated into probability adjustments

    **ML Predictions** 🧠
    - Ensemble machine learning model analyzes odds patterns
    - Adjusts probabilities based on multiple factors
    - Higher confidence when ML and market agree

    **AI Confidence Score** 📊
    - Green (🟢): High confidence (>70%) - Strong AI signal
    - Yellow (🟡): Moderate confidence (50-70%) - Good opportunity
    - Orange (🟠): Lower confidence (<50%) - Higher risk

    **AI Expected Value** 💰
    - Compares AI probability vs market odds
    - Positive EV = mathematical edge over the market
    - Higher EV = better long-term profitability

    **AI Score** ⭐
    - Combined metric: EV × Confidence × Edge
    - Higher scores = best overall opportunities
    - Accounts for correlation in same-game parlays
    """)
    
st.caption("🟢 High Confidence | 💰 High +EV | 📈 Positive EV | 📉 Negative EV | Powered by AI & ML")

# ===== TAB 2: PRIZEPICKS =====

# ===== TAB 2: SENTIMENT & AI ANALYSIS =====
with main_tab2:
    st.header("🔍 Sentiment & AI Analysis Dashboard")
    st.markdown("**Advanced sentiment analysis using web scraping, news APIs, and AI-powered insights**")
    st.caption("Get deep insights into team performance, news sentiment, betting trends, and AI predictions")
    
    # API Configuration
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    
    col_api1, col_api2 = st.columns(2)
    with col_api1:
        odds_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
        if not odds_key:
            st.warning("⚠️ Odds API key not configured. Please set it in the Sports Betting tab.")
    
    with col_api2:
        news_key = st.session_state.get('news_api_key', "") or os.environ.get("NEWS_API_KEY", "")
        news_key_input = st.text_input(
            "NewsAPI Key (optional - enhances sentiment)",
            value=news_key,
            type="password",
            help="Get free key at https://newsapi.org"
        )
        if news_key_input != news_key:
            st.session_state['news_api_key'] = news_key_input
            st.session_state['sentiment_analyzer'] = RealSentimentAnalyzer(news_key_input)
    
    st.markdown("---")
    
    # Team Selection
    st.subheader("🎯 Select Teams to Analyze")
    
    col_sport, col_num = st.columns(2)
    with col_sport:
        analysis_sport = st.selectbox(
            "Sport",
            options=APP_CFG["sports_common"],
            key="analysis_sport"
        )
    
    with col_num:
        num_teams = st.slider("Number of teams to analyze", 2, 10, 5)
    
    # Fetch games and extract teams
    if st.button("🔍 Load Teams", type="primary"):
        if not odds_key:
            st.error("Please configure Odds API key first")
        else:
            with st.spinner(f"Loading {analysis_sport} teams..."):
                try:
                    snap = fetch_oddsapi_snapshot(odds_key, analysis_sport)
                    events = snap.get("events", [])
                    
                    if not events:
                        st.warning("No games found for this sport")
                    else:
                        # Extract unique teams
                        teams = set()
                        team_games = {}  # Map team to their games
                        
                        for ev in events:
                            home = ev.get("home_team")
                            away = ev.get("away_team")
                            
                            if home:
                                teams.add(home)
                                if home not in team_games:
                                    team_games[home] = []
                                team_games[home].append({
                                    'opponent': away,
                                    'location': 'home',
                                    'game_id': ev.get('id'),
                                    'commence_time': ev.get('commence_time')
                                })
                            
                            if away:
                                teams.add(away)
                                if away not in team_games:
                                    team_games[away] = []
                                team_games[away].append({
                                    'opponent': home,
                                    'location': 'away',
                                    'game_id': ev.get('id'),
                                    'commence_time': ev.get('commence_time')
                                })
                        
                        st.session_state['analysis_teams'] = sorted(list(teams))
                        st.session_state['team_games'] = team_games
                        st.success(f"✅ Found {len(teams)} teams with upcoming games")
                
                except Exception as e:
                    st.error(f"Error loading teams: {str(e)}")
    
    # Team Analysis
    if 'analysis_teams' in st.session_state and st.session_state['analysis_teams']:
        st.markdown("---")
        st.subheader("📊 Team Sentiment Analysis")
        
        teams = st.session_state['analysis_teams']
        selected_teams = st.multiselect(
            "Select teams to analyze",
            options=teams,
            default=teams[:min(num_teams, len(teams))],
            max_selections=10
        )
        
        if st.button("🤖 Run Deep Analysis", type="primary"):
            if not selected_teams:
                st.warning("Please select at least one team")
            else:
                sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                
                if not sentiment_analyzer:
                    st.error("Sentiment analyzer not initialized")
                else:
                    progress_bar = st.progress(0)
                    analysis_results = []
                    
                    for idx, team in enumerate(selected_teams):
                        progress_bar.progress((idx + 1) / len(selected_teams))
                        
                        with st.spinner(f"Analyzing {team}..."):
                            try:
                                # Get sentiment
                                sentiment = sentiment_analyzer.get_team_sentiment(team, analysis_sport)
                                
                                # Get game info
                                games = st.session_state['team_games'].get(team, [])
                                next_game = games[0] if games else None
                                
                                # Get odds if available
                                odds_data = None
                                if next_game:
                                    for ev in st.session_state.get('available_games', []):
                                        if ev.get('id') == next_game['game_id']:
                                            h2h = ev.get('markets', {}).get('h2h', {})
                                            if next_game['location'] == 'home':
                                                odds_data = _dig(h2h, 'home.price')
                                            else:
                                                odds_data = _dig(h2h, 'away.price')
                                            break
                                
                                analysis_results.append({
                                    'team': team,
                                    'sentiment': sentiment,
                                    'next_game': next_game,
                                    'odds': odds_data
                                })
                            
                            except Exception as e:
                                st.warning(f"Error analyzing {team}: {str(e)}")
                    
                    progress_bar.progress(1.0)
                    
                    # Display Results
                    if analysis_results:
                        st.markdown("---")
                        st.markdown("### 📈 Analysis Results")
                        
                        # Summary metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        positive_teams = sum(1 for r in analysis_results if r['sentiment']['trend'] == 'positive')
                        negative_teams = sum(1 for r in analysis_results if r['sentiment']['trend'] == 'negative')
                        avg_score = sum(r['sentiment']['score'] for r in analysis_results) / len(analysis_results)
                        avg_confidence = sum(r['sentiment']['confidence'] for r in analysis_results) / len(analysis_results)
                        
                        with col_m1:
                            st.metric("Positive Sentiment", positive_teams)
                        with col_m2:
                            st.metric("Negative Sentiment", negative_teams)
                        with col_m3:
                            st.metric("Avg Sentiment Score", f"{avg_score:+.2f}")
                        with col_m4:
                            st.metric("Avg Confidence", f"{avg_confidence*100:.0f}%")
                        
                        # Sort by sentiment score
                        analysis_results.sort(key=lambda x: x['sentiment']['score'], reverse=True)
                        
                        # Display each team
                        st.markdown("### 🏈 Team-by-Team Breakdown")
                        
                        for result in analysis_results:
                            team = result['team']
                            sentiment = result['sentiment']
                            next_game = result['next_game']
                            odds = result['odds']
                            
                            # Determine sentiment icon and color
                            if sentiment['trend'] == 'positive':
                                trend_icon = "🟢"
                                trend_color = "green"
                            elif sentiment['trend'] == 'negative':
                                trend_icon = "🔴"
                                trend_color = "red"
                            else:
                                trend_icon = "🟡"
                                trend_color = "orange"
                            
                            with st.expander(f"{trend_icon} {team} - {sentiment['trend'].upper()} ({sentiment['score']:+.2f})"):
                                col_info1, col_info2 = st.columns(2)
                                
                                with col_info1:
                                    st.markdown("**Sentiment Analysis:**")
                                    st.write(f"- **Score:** {sentiment['score']:+.2f}")
                                    st.write(f"- **Trend:** {sentiment['trend'].upper()}")
                                    st.write(f"- **Confidence:** {sentiment['confidence']*100:.0f}%")
                                    st.write(f"- **Sources:** {sentiment['sources']} articles")
                                    st.write(f"- **Method:** {sentiment['method']}")
                                
                                with col_info2:
                                    st.markdown("**Next Game:**")
                                    if next_game:
                                        location = "🏠 Home" if next_game['location'] == 'home' else "✈️ Away"
                                        st.write(f"- **vs {next_game['opponent']}** ({location})")
                                        if odds:
                                            prob = implied_p_from_american(odds)
                                            st.write(f"- **Odds:** {odds:+.0f}")
                                            st.write(f"- **Implied Prob:** {prob*100:.1f}%")
                                        
                                        # Try to parse time
                                        try:
                                            import datetime
                                            game_time = datetime.datetime.fromtimestamp(next_game['commence_time'])
                                            st.write(f"- **Time:** {game_time.strftime('%m/%d %I:%M %p')}")
                                        except:
                                            pass
                                    else:
                                        st.write("No upcoming game found")
                                
                                # Betting recommendation based on sentiment + odds
                                st.markdown("**💡 AI Betting Insight:**")
                                
                                if sentiment['score'] > 0.3 and sentiment['confidence'] > 0.6:
                                    if odds and odds > 0:  # Underdog with positive sentiment
                                        st.success("🟢 **STRONG VALUE** - Positive sentiment underdog. Market may be undervaluing this team.")
                                    elif odds and odds < -200:  # Heavy favorite with positive sentiment
                                        st.info("🟡 **GOOD SPOT** - Sentiment confirms favorite status, but odds may be steep.")
                                    else:
                                        st.success("🟢 **FAVORABLE** - Strong positive sentiment. Consider backing this team.")
                                
                                elif sentiment['score'] < -0.3 and sentiment['confidence'] > 0.6:
                                    if odds and odds < 0:  # Favorite with negative sentiment
                                        st.warning("🟠 **FADE CANDIDATE** - Negative sentiment on a favorite. Public may be overvaluing.")
                                    else:
                                        st.error("🔴 **AVOID** - Strong negative sentiment. Look elsewhere.")
                                
                                else:
                                    st.info("🟡 **NEUTRAL** - No strong sentiment signal. Rely on other factors.")
                        
                        # Export option
                        st.markdown("---")
                        export_data = []
                        for result in analysis_results:
                            export_data.append({
                                'Team': result['team'],
                                'Sentiment Score': f"{result['sentiment']['score']:+.2f}",
                                'Trend': result['sentiment']['trend'],
                                'Confidence': f"{result['sentiment']['confidence']*100:.0f}%",
                                'Sources': result['sentiment']['sources'],
                                'Next Opponent': result['next_game']['opponent'] if result['next_game'] else 'N/A',
                                'Location': result['next_game']['location'] if result['next_game'] else 'N/A',
                                'Odds': f"{result['odds']:+.0f}" if result['odds'] else 'N/A'
                            })
                        
                        df_export = pd.DataFrame(export_data)
                        csv_buf = io.StringIO()
                        df_export.to_csv(csv_buf, index=False)
                        
                        st.download_button(
                            "💾 Download Analysis CSV",
                            data=csv_buf.getvalue(),
                            file_name=f"sentiment_analysis_{analysis_sport}.csv",
                            mime="text/csv"
                        )
    
    else:
        st.info("👆 Click 'Load Teams' to start sentiment analysis")
    
    # Advanced Features Section
    st.markdown("---")
    st.markdown("### 🚀 Advanced Analysis Features")
    
    with st.expander("📰 News Sentiment Analysis"):
        st.markdown("""
        **How it works:**
        - Scrapes recent news articles about each team (last 3 days)
        - Uses NewsAPI.org for reliable news sources
        - Analyzes headlines and descriptions using NLP
        - Identifies positive words (win, dominant, stellar) vs negative words (lose, injury, struggle)
        - Calculates sentiment score from -1.0 (very negative) to +1.0 (very positive)
        - Provides confidence score based on number of sources and consistency
        
        **Positive Indicators:**
        - Winning streak, dominant performance, star player excelling
        - Record-breaking stats, momentum, clutch plays
        - Positive coaching changes, key player returns
        
        **Negative Indicators:**
        - Losing streak, injuries to key players, poor performance
        - Internal conflicts, coaching issues, suspensions
        - Defensive/offensive struggles, blown leads
        """)
    
    with st.expander("🤖 AI Prediction Model"):
        st.markdown("""
        **Machine Learning Components:**
        - **Input Features:** Home/Away odds, sentiment scores, historical patterns
        - **Model Type:** Gradient boosting with probability calibration
        - **Output:** Win probability for each team, confidence score, edge calculation
        
        **How AI Adjusts Probabilities:**
        1. Takes market-implied probability from odds
        2. Applies sentiment adjustment (±40% weight)
        3. Considers home/away advantage
        4. Calibrates based on historical accuracy
        5. Outputs adjusted probability + confidence
        
        **Confidence Scoring:**
        - High (70%+): Strong signal from multiple factors
        - Medium (50-70%): Moderate signals, some uncertainty
        - Low (<50%): Conflicting signals or limited data
        """)
    
    with st.expander("📊 Betting Trend Analysis"):
        st.markdown("""
        **Market Movements:**
        - Track line movements throughout the day
        - Identify sharp vs public money
        - Detect reverse line movement (line moves opposite to betting percentages)
        
        **Value Detection:**
        - Compare sentiment score to market odds
        - Find positive sentiment underdogs (best value)
        - Identify overvalued favorites with negative sentiment
        
        **Correlation Analysis:**
        - Sentiment vs actual outcomes (historical accuracy)
        - Best sports/leagues for sentiment analysis
        - Optimal bet types for sentiment-based betting
        """)
    
    with st.expander("🎯 How to Use This Analysis"):
        st.markdown("""
        **Step 1: Load & Analyze**
        1. Select sport and load current teams
        2. Choose 3-5 teams you're interested in
        3. Run deep analysis
        
        **Step 2: Interpret Results**
        - **🟢 Green (Positive)**: Team has favorable news/momentum
        - **🔴 Red (Negative)**: Team has concerning news/struggles  
        - **🟡 Yellow (Neutral)**: No strong sentiment signal
        
        **Step 3: Find Value**
        - Look for positive sentiment underdogs (market undervaluing)
        - Fade negative sentiment favorites (market overvaluing)
        - Combine with AI probability for best bets
        
        **Step 4: Validate**
        - Use Custom Parlay Builder to test specific picks
        - Compare sentiment analysis to AI expected value
        - Make informed decision based on multiple factors
        
        **Best Practices:**
        - Don't bet on sentiment alone - combine with AI analysis
        - Higher confidence scores are more reliable
        - More news sources = more reliable sentiment
        - Recent news (1-3 days) is most relevant
        """)
    
    # Tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Sentiment Analysis Tips:
    
    **What Makes Strong Sentiment:**
    - ✅ Multiple news sources (5+ articles)
    - ✅ High confidence score (70%+)
    - ✅ Recent news (last 24-48 hours)
    - ✅ Consistent trend across sources
    
    **Red Flags:**
    - ⚠️ Low confidence (<40%)
    - ⚠️ Few news sources (<3 articles)
    - ⚠️ Mixed signals (some positive, some negative)
    - ⚠️ Old news (4+ days ago)
    
    **Best Use Cases:**
    - 🎯 Finding undervalued underdogs with positive momentum
    - 🎯 Fading overvalued favorites with negative news
    - 🎯 Validating your own analysis with AI/sentiment
    - 🎯 Identifying injury/coaching impacts quickly
    
    **Combine With:**
    - Use sentiment for initial screening
    - Use AI analysis for probability adjustment
    - Use Custom Parlay Builder to test combinations
    - Compare multiple data points before betting
    """)

with main_tab3:
    st.header("🎨 Custom Parlay Builder")
    st.markdown("**Build your own parlay and get AI-powered analysis**")
    st.caption("Select 2-4 legs, then get comprehensive AI/ML analysis with sentiment, probability, and edge calculations")
    
    # API key check
    api_key = st.session_state.get('api_key', "") or os.environ.get("ODDS_API_KEY", "")
    
    if not api_key:
        st.warning("⚠️ Please enter your Odds API key in the 'Sports Betting Parlays' tab first")
        st.stop()
    
    # Initialize session state for custom parlay legs
    if 'custom_legs' not in st.session_state:
        st.session_state['custom_legs'] = []
    
    st.markdown("---")
    
    # Step 1: Fetch Available Games
    st.subheader("📋 Step 1: Select Sport & Date")
    
    col_sport, col_date = st.columns(2)
    with col_sport:
        custom_sport = st.selectbox(
            "Sport",
            options=APP_CFG["sports_common"],
            key="custom_sport"
        )
    with col_date:
        custom_date = st.date_input(
            "Game Date",
            value=pd.Timestamp.now().date(),
            key="custom_date"
        )
    
    if st.button("🔄 Load Games", type="primary"):
        with st.spinner(f"Loading {custom_sport} games..."):
            try:
                snap = fetch_oddsapi_snapshot(api_key, custom_sport)
                st.session_state['available_games'] = snap.get("events", [])
                st.success(f"✅ Loaded {len(st.session_state.get('available_games', []))} games")
            except Exception as e:
                st.error(f"Error loading games: {str(e)}")
    
    # Step 2: Add Legs to Custom Parlay
    if 'available_games' in st.session_state and st.session_state['available_games']:
        st.markdown("---")
        st.subheader("⚡ Step 2: Build Your Parlay")
        
        games = st.session_state['available_games']
        
        # Game selector
        game_options = [f"{g['away_team']} @ {g['home_team']}" for g in games]
        selected_game_idx = st.selectbox(
            "Select Game",
            options=range(len(game_options)),
            format_func=lambda x: game_options[x],
            key="game_selector"
        )
        
        if selected_game_idx is not None:
            selected_game = games[selected_game_idx]
            home_team = selected_game['home_team']
            away_team = selected_game['away_team']
            markets = selected_game.get('markets', {})
            
            # Bet type selector
            col_bet1, col_bet2 = st.columns(2)
            with col_bet1:
                bet_type = st.selectbox(
                    "Bet Type",
                    options=["Moneyline", "Spread", "Total"],
                    key="bet_type_selector"
                )
            
            with col_bet2:
                if bet_type == "Moneyline":
                    h2h = markets.get('h2h', {})
                    home_price = _dig(h2h, 'home.price')
                    away_price = _dig(h2h, 'away.price')
                    
                    if home_price and away_price:
                        selection = st.selectbox(
                            "Selection",
                            options=[
                                f"{home_team} ML @{home_price:+.0f}",
                                f"{away_team} ML @{away_price:+.0f}"
                            ],
                            key="ml_selector"
                        )
                        
                        # Parse selection
                        if home_team in selection:
                            pick_team = home_team
                            pick_price = home_price
                            pick_side = "home"
                        else:
                            pick_team = away_team
                            pick_price = away_price
                            pick_side = "away"
                    else:
                        st.warning("Moneyline odds not available for this game")
                        selection = None
                
                elif bet_type == "Spread":
                    spreads = markets.get('spreads', [])
                    if spreads:
                        spread_options = []
                        for s in spreads[:8]:  # Show up to 8 spread options
                            team = s.get('name')
                            point = s.get('point')
                            price = s.get('price')
                            if team and point is not None and price:
                                spread_options.append({
                                    'label': f"{team} {point:+.1f} @{price:+.0f}",
                                    'team': team,
                                    'point': point,
                                    'price': price,
                                    'side': 'home' if team == home_team else 'away'
                                })
                        
                        if spread_options:
                            selection = st.selectbox(
                                "Selection",
                                options=range(len(spread_options)),
                                format_func=lambda x: spread_options[x]['label'],
                                key="spread_selector"
                            )
                            
                            if selection is not None:
                                pick_data = spread_options[selection]
                                pick_team = pick_data['team']
                                pick_price = pick_data['price']
                                pick_side = pick_data['side']
                                pick_point = pick_data['point']
                        else:
                            st.warning("Spread odds not available")
                            selection = None
                    else:
                        st.warning("Spread odds not available")
                        selection = None
                
                elif bet_type == "Total":
                    totals = markets.get('totals', [])
                    if totals:
                        total_options = []
                        for t in totals[:8]:  # Show up to 8 total options
                            name = t.get('name', '')
                            point = t.get('point')
                            price = t.get('price')
                            if point is not None and price:
                                total_options.append({
                                    'label': f"{name} {point} @{price:+.0f}",
                                    'name': name,
                                    'point': point,
                                    'price': price
                                })
                        
                        if total_options:
                            selection = st.selectbox(
                                "Selection",
                                options=range(len(total_options)),
                                format_func=lambda x: total_options[x]['label'],
                                key="total_selector"
                            )
                            
                            if selection is not None:
                                pick_data = total_options[selection]
                                pick_team = f"{home_team} vs {away_team}"
                                pick_price = pick_data['price']
                                pick_side = pick_data['name']
                                pick_point = pick_data['point']
                        else:
                            st.warning("Total odds not available")
                            selection = None
                    else:
                        st.warning("Total odds not available")
                        selection = None
            
            # Add leg button
            if selection is not None and st.button("➕ Add to Parlay", type="secondary"):
                # Create leg data
                leg = {
                    'event_id': selected_game['id'],
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'type': bet_type,
                    'team': pick_team,
                    'side': pick_side,
                    'price': pick_price,
                    'd': american_to_decimal_safe(pick_price)
                }
                
                if bet_type in ['Spread', 'Total']:
                    leg['point'] = pick_point
                
                # Check if already at max legs
                if len(st.session_state['custom_legs']) >= 4:
                    st.warning("⚠️ Maximum 4 legs allowed. Remove a leg to add another.")
                else:
                    # Check for duplicates (same game + same bet type)
                    duplicate = False
                    for existing_leg in st.session_state['custom_legs']:
                        if (existing_leg['event_id'] == leg['event_id'] and 
                            existing_leg['type'] == leg['type'] and
                            existing_leg['team'] == leg['team']):
                            duplicate = True
                            break
                    
                    if duplicate:
                        st.warning("⚠️ This leg is already in your parlay")
                    else:
                        st.session_state['custom_legs'].append(leg)
                        st.success(f"✅ Added to parlay!")
                        st.rerun()
    
    # Step 3: Show Current Parlay
    if st.session_state['custom_legs']:
        st.markdown("---")
        st.subheader("🎯 Your Custom Parlay")
        
        for i, leg in enumerate(st.session_state['custom_legs'], 1):
            col_leg, col_remove = st.columns([5, 1])
            with col_leg:
                label = f"{leg['game']} — {leg['team']}"
                if leg['type'] == 'Spread':
                    label += f" {leg['point']:+.1f}"
                elif leg['type'] == 'Total':
                    label += f" {leg['side']} {leg['point']}"
                label += f" @{leg['price']:+.0f}"
                
                st.write(f"**Leg {i}:** {label}")
            
            with col_remove:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state['custom_legs'].pop(i-1)
                    st.rerun()
        
        # Step 4: Analyze Parlay
        if len(st.session_state['custom_legs']) >= 2:
            st.markdown("---")
            st.subheader("🤖 AI Analysis")
            
            if st.button("🔍 Analyze My Parlay", type="primary"):
                with st.spinner("🧠 Running AI/ML analysis..."):
                    try:
                        sentiment_analyzer = st.session_state.get('sentiment_analyzer')
                        ml_predictor = st.session_state.get('ml_predictor')
                        ai_optimizer = st.session_state.get('ai_optimizer')
                        
                        # Enhance legs with AI analysis
                        analyzed_legs = []
                        for leg in st.session_state['custom_legs']:
                            # Get sentiment for teams
                            try:
                                home_sentiment = sentiment_analyzer.get_team_sentiment(
                                    leg['home_team'], custom_sport
                                ) if sentiment_analyzer else {'score': 0, 'trend': 'neutral'}
                                away_sentiment = sentiment_analyzer.get_team_sentiment(
                                    leg['away_team'], custom_sport
                                ) if sentiment_analyzer else {'score': 0, 'trend': 'neutral'}
                            except Exception:
                                home_sentiment = {'score': 0, 'trend': 'neutral'}
                                away_sentiment = {'score': 0, 'trend': 'neutral'}
                            
                            # Calculate probabilities
                            base_prob = implied_p_from_american(leg['price'])
                            
                            # Get AI probability
                            if leg['type'] == 'Moneyline' and ml_predictor:
                                # Get opponent price for ML prediction
                                opp_price = None
                                for g in st.session_state['available_games']:
                                    if g['id'] == leg['event_id']:
                                        h2h = g.get('markets', {}).get('h2h', {})
                                        if leg['side'] == 'home':
                                            opp_price = _dig(h2h, 'away.price')
                                        else:
                                            opp_price = _dig(h2h, 'home.price')
                                        break
                                
                                if opp_price:
                                    ml_prediction = ml_predictor.predict_game_outcome(
                                        leg['home_team'], leg['away_team'],
                                        _dig(h2h, 'home.price'), _dig(h2h, 'away.price'),
                                        home_sentiment['score'], away_sentiment['score']
                                    )
                                    ai_prob = ml_prediction[f"{leg['side']}_prob"]
                                    ai_confidence = ml_prediction['confidence']
                                    ai_edge = ml_prediction['edge']
                                else:
                                    ai_prob = base_prob
                                    ai_confidence = 0.5
                                    ai_edge = 0
                            else:
                                # For spreads/totals, use sentiment adjustment
                                sentiment = home_sentiment if leg['team'] == leg['home_team'] else away_sentiment
                                if leg['type'] == 'Total':
                                    sentiment = {'score': (home_sentiment['score'] + away_sentiment['score']) / 2}
                                
                                ai_prob = base_prob * (1 + sentiment['score'] * 0.40)
                                ai_prob = max(0.1, min(0.9, ai_prob))
                                ai_confidence = 0.65 if leg['type'] == 'Spread' else 0.60
                                ai_edge = abs(ai_prob - base_prob)
                            
                            # Add analysis to leg
                            analyzed_leg = leg.copy()
                            analyzed_leg.update({
                                'p': base_prob,
                                'ai_prob': ai_prob,
                                'ai_confidence': ai_confidence,
                                'ai_edge': ai_edge,
                                'sentiment_trend': home_sentiment['trend'] if leg['side'] == 'home' else away_sentiment['trend'],
                                'home_sentiment': home_sentiment,
                                'away_sentiment': away_sentiment
                            })
                            analyzed_legs.append(analyzed_leg)
                        
                        # Calculate parlay metrics
                        combined_odds = 1.0
                        market_prob = 1.0
                        ai_prob = 1.0
                        
                        for leg in analyzed_legs:
                            combined_odds *= leg['d']
                            market_prob *= leg['p']
                            ai_prob *= leg['ai_prob']
                        
                        # Get AI optimizer score
                        if ai_optimizer:
                            ai_metrics = ai_optimizer.score_parlay(analyzed_legs)
                        else:
                            ai_metrics = {
                                'score': 50,
                                'ai_ev': (ai_prob * combined_odds) - 1.0,
                                'confidence': sum(l['ai_confidence'] for l in analyzed_legs) / len(analyzed_legs),
                                'edge': sum(l['ai_edge'] for l in analyzed_legs)
                            }
                        
                        # Calculate payouts
                        stake = 100
                        potential_payout = stake * combined_odds
                        potential_profit = potential_payout - stake
                        
                        # Market EV
                        market_ev = (market_prob * combined_odds) - 1.0
                        market_expected_return = stake * market_ev
                        
                        # AI EV
                        ai_expected_return = stake * ai_metrics['ai_ev']
                        
                        # Display Results
                        st.markdown("### 📊 Analysis Results")
                        
                        # Main metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Combined Odds", f"{combined_odds:.2f}x")
                        with col2:
                            st.metric("Market Probability", f"{market_prob*100:.1f}%")
                        with col3:
                            st.metric("AI Probability", f"{ai_prob*100:.1f}%")
                        with col4:
                            confidence_color = "🟢" if ai_metrics['confidence'] > 0.7 else ("🟡" if ai_metrics['confidence'] > 0.5 else "🟠")
                            st.metric("AI Confidence", f"{confidence_color} {ai_metrics['confidence']*100:.0f}%")
                        
                        # EV metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric(
                                "Market EV",
                                f"${market_expected_return:+.2f}",
                                help="Expected value based on market (no-vig) probabilities"
                            )
                        with col6:
                            ev_delta = ai_expected_return - market_expected_return
                            st.metric(
                                "AI EV",
                                f"${ai_expected_return:+.2f}",
                                delta=ev_delta,
                                help="Expected value based on AI-adjusted probabilities"
                            )
                        with col7:
                            st.metric("AI Score", f"{ai_metrics['score']:.1f}")
                        
                        # Sentiment Analysis
                        st.markdown("### 🎭 Sentiment Analysis")
                        for i, leg in enumerate(analyzed_legs, 1):
                            with st.expander(f"Leg {i}: {leg['game']}"):
                                col_sent1, col_sent2 = st.columns(2)
                                with col_sent1:
                                    st.write(f"**{leg['home_team']}**")
                                    st.write(f"Sentiment: {leg['home_sentiment']['trend'].upper()}")
                                    st.write(f"Score: {leg['home_sentiment']['score']:+.2f}")
                                with col_sent2:
                                    st.write(f"**{leg['away_team']}**")
                                    st.write(f"Sentiment: {leg['away_sentiment']['trend'].upper()}")
                                    st.write(f"Score: {leg['away_sentiment']['score']:+.2f}")
                        
                        # Detailed Leg Analysis
                        st.markdown("### 📋 Leg-by-Leg Breakdown")
                        leg_data = []
                        for i, leg in enumerate(analyzed_legs, 1):
                            leg_data.append({
                                "Leg": i,
                                "Game": leg['game'],
                                "Pick": f"{leg['team']} {leg['type']}",
                                "Odds": f"{leg['price']:+.0f}",
                                "Market %": f"{leg['p']*100:.1f}%",
                                "AI %": f"{leg['ai_prob']*100:.1f}%",
                                "AI Confidence": f"{leg['ai_confidence']*100:.0f}%",
                                "Edge": f"{leg['ai_edge']*100:+.1f}%",
                                "Sentiment": leg['sentiment_trend']
                            })
                        
                        st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)
                        
                        # Payout Scenarios
                        st.markdown("### 💰 Payout Scenarios")
                        st.write(f"**On a $100 bet:**")
                        st.write(f"- Total Payout: **${potential_payout:.2f}**")
                        st.write(f"- Profit: **${potential_profit:.2f}**")
                        st.write(f"- ROI: **{(potential_profit/stake)*100:.1f}%**")
                        
                        st.markdown("**Other Stakes:**")
                        for bet_amount in [25, 50, 100, 250, 500]:
                            payout = bet_amount * combined_odds
                            profit = payout - bet_amount
                            st.write(f"${bet_amount} → ${payout:.2f} (${profit:+.2f} profit)")
                        
                        # Recommendation
                        st.markdown("### 💡 AI Recommendation")
                        
                        # Decision logic
                        if ai_expected_return > 5 and ai_metrics['confidence'] > 0.65:
                            st.success("🟢 **STRONG PLAY** - Positive AI EV with high confidence")
                        elif ai_expected_return > 0 and ai_metrics['confidence'] > 0.55:
                            st.info("🟡 **CONSIDER** - Slight positive AI EV with moderate confidence")
                        elif market_expected_return > 0:
                            st.warning("🟠 **CAUTION** - Market EV positive but AI less confident")
                        else:
                            st.error("🔴 **AVOID** - Negative expected value")
                        
                        # Key insights
                        st.markdown("**Key Insights:**")
                        insights = []
                        
                        if ai_prob > market_prob * 1.1:
                            insights.append(f"✅ AI sees {((ai_prob/market_prob-1)*100):.0f}% better chance than market")
                        elif ai_prob < market_prob * 0.9:
                            insights.append(f"⚠️ AI sees {((1-ai_prob/market_prob)*100):.0f}% worse chance than market")
                        
                        if ai_metrics['confidence'] > 0.7:
                            insights.append("✅ High AI confidence across all legs")
                        elif ai_metrics['confidence'] < 0.5:
                            insights.append("⚠️ Low AI confidence - consider alternative picks")
                        
                        positive_sentiment = sum(1 for leg in analyzed_legs if leg['sentiment_trend'] == 'positive')
                        if positive_sentiment == len(analyzed_legs):
                            insights.append("✅ All picks have positive sentiment")
                        elif positive_sentiment == 0:
                            insights.append("⚠️ No picks have positive sentiment")
                        
                        for insight in insights:
                            st.write(insight)
                        
                        # Download option
                        st.markdown("---")
                        analysis_data = {
                            "Parlay": [f"Custom {len(analyzed_legs)}-Leg"],
                            "Combined Odds": [f"{combined_odds:.2f}"],
                            "Market Probability": [f"{market_prob*100:.1f}%"],
                            "AI Probability": [f"{ai_prob*100:.1f}%"],
                            "AI Confidence": [f"{ai_metrics['confidence']*100:.0f}%"],
                            "Market EV": [f"${market_expected_return:+.2f}"],
                            "AI EV": [f"${ai_expected_return:+.2f}"],
                            "AI Score": [f"{ai_metrics['score']:.1f}"]
                        }
                        
                        csv_buf = io.StringIO()
                        pd.DataFrame(analysis_data).to_csv(csv_buf, index=False)
                        st.download_button(
                            "💾 Download Analysis",
                            data=csv_buf.getvalue(),
                            file_name="custom_parlay_analysis.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error analyzing parlay: {str(e)}")
                        with st.expander("Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.info("ℹ️ Add at least 2 legs to analyze your parlay")
        
        # Clear parlay button
        if st.button("🗑️ Clear All Legs", type="secondary"):
            st.session_state['custom_legs'] = []
            st.rerun()
    else:
        st.info("👆 Load games and start building your parlay above")
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    ### 💡 Custom Parlay Tips:
    
    **Building Your Parlay:**
    - Start with 2-3 legs for better win probability
    - Mix different bet types (ML, Spread, Total) for variety
    - Avoid same-game parlays unless you have strong correlation thesis
    
    **Using AI Analysis:**
    - **Positive AI EV** = AI thinks you have an edge
    - **High Confidence (70%+)** = AI very sure about probabilities
    - **Positive Sentiment** = Recent news favors this pick
    - **Green AI Score** = Strong overall recommendation
    
    **Making Decisions:**
    - ✅ Target: Positive AI EV + High Confidence + Good Sentiment
    - ⚠️ Caution: Negative AI EV or Low Confidence
    - 🔴 Avoid: Multiple red flags (negative EV, low confidence, bad sentiment)
    """)

# ===== TAB 4: KALSHI PREDICTION MARKETS =====
with main_tab4:
    st.header("📊 Kalshi Prediction Markets")
    st.markdown("**Compare prediction market odds with traditional sportsbooks and AI analysis**")
    st.caption("Find arbitrage opportunities and value bets by comparing Kalshi's wisdom-of-crowds pricing with sportsbook odds")
    
    # API Configuration
    st.markdown("---")
    st.subheader("⚙️ Kalshi Configuration")
    
    col_kalshi1, col_kalshi2 = st.columns(2)
    with col_kalshi1:
        kalshi_key = st.text_input(
            "Kalshi API Key",
            value=st.session_state.get('kalshi_api_key', os.environ.get("KALSHI_API_KEY", "")),
            type="password",
            help="Get your API key from https://kalshi.com/account/api"
        )
        if kalshi_key != st.session_state.get('kalshi_api_key', ''):
            st.session_state['kalshi_api_key'] = kalshi_key
            st.session_state['kalshi_integrator'] = KalshiIntegrator(
                kalshi_key, 
                st.session_state.get('kalshi_api_secret', '')
            )
    
    with col_kalshi2:
        kalshi_secret = st.text_input(
            "Kalshi API Secret (optional)",
            value=st.session_state.get('kalshi_api_secret', os.environ.get("KALSHI_API_SECRET", "")),
            type="password",
            help="API secret for authenticated requests"
        )
        if kalshi_secret != st.session_state.get('kalshi_api_secret', ''):
            st.session_state['kalshi_api_secret'] = kalshi_secret
            st.session_state['kalshi_integrator'] = KalshiIntegrator(
                st.session_state.get('kalshi_api_key', ''),
                kalshi_secret
            )
    
    if not kalshi_key:
        st.info("💡 **Demo Mode:** You can explore Kalshi without API keys. For live trading, get your API key at [kalshi.com](https://kalshi.com)")
    
    st.markdown("---")
    
    # Main Analysis Sections
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["🔍 Browse Kalshi Sports Markets", "⚖️ Compare with Sportsbooks", "💎 Find Arbitrage Opportunities"],
        horizontal=True
    )
    
    kalshi = st.session_state.get('kalshi_integrator')
    
    if analysis_mode == "🔍 Browse Kalshi Sports Markets":
        st.subheader("🏈 Available Sports Betting Markets")
        
        if st.button("🔄 Load Kalshi Markets", type="primary"):
            with st.spinner("Fetching Kalshi markets..."):
                try:
                    markets = kalshi.get_sports_markets()
                    st.session_state['kalshi_markets'] = markets
                    st.success(f"✅ Loaded {len(markets)} sports markets")
                    if kalshi.using_synthetic_data():
                        st.warning("🧪 Live Kalshi API unavailable – showing synthetic demo markets instead.")
                        if kalshi.last_error:
                            st.caption(f"Last API error: {kalshi.last_error}")
                except Exception as e:
                    st.error(f"Error loading markets: {str(e)}")
                    st.info("💡 Try demo mode without API keys to explore sample markets")

        if 'kalshi_markets' in st.session_state and st.session_state['kalshi_markets']:
            markets = st.session_state['kalshi_markets']

            if kalshi and kalshi.using_synthetic_data():
                st.info("🧪 Displaying locally generated Kalshi fallback data for exploration.")

            st.markdown(f"### 📋 {len(markets)} Markets Available")
            
            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                sport_filter = st.selectbox(
                    "Filter by Sport",
                    options=["All"] + ["NFL", "NBA", "MLB", "NHL", "UFC", "Soccer"],
                    key="kalshi_sport_filter"
                )
            
            with col_filter2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Volume (High to Low)", "Close Date (Soonest)", "Title (A-Z)"],
                    key="kalshi_sort"
                )
            
            # Filter markets
            filtered_markets = markets
            if sport_filter != "All":
                filtered_markets = [m for m in markets if sport_filter.upper() in m.get('title', '').upper()]
            
            # Sort markets
            if sort_by == "Volume (High to Low)":
                filtered_markets.sort(key=lambda x: x.get('volume', 0), reverse=True)
            elif sort_by == "Close Date (Soonest)":
                filtered_markets.sort(key=lambda x: x.get('close_time', ''))
            else:
                filtered_markets.sort(key=lambda x: x.get('title', ''))
            
            # Display markets
            for i, market in enumerate(filtered_markets[:20]):  # Show top 20
                title = market.get('title', 'Unknown Market')
                ticker = market.get('ticker', '')
                volume = market.get('volume', 0)
                
                with st.expander(f"**{i+1}. {title}**"):
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.write(f"**Ticker:** {ticker}")
                        st.write(f"**Volume:** {volume:,} contracts")
                        st.write(f"**Status:** {market.get('status', 'unknown')}")
                    
                    with col_m2:
                        # Get orderbook for this market
                        button_key = f"analyze_{ticker}_{i}"
                        if st.button(f"📊 Analyze {ticker[:15]}...", key=button_key):
                            with st.spinner("Fetching market details..."):
                                try:
                                    orderbook = kalshi.get_orderbook(ticker)
                                    
                                    if orderbook:
                                        yes_bids = orderbook.get('yes', [])
                                        no_bids = orderbook.get('no', [])
                                        
                                        if yes_bids:
                                            best_yes_bid = yes_bids[0].get('price', 0) / 100
                                            st.success(f"**YES Price:** {best_yes_bid*100:.1f}¢ ({best_yes_bid*100:.1f}% probability)")
                                        
                                        if no_bids:
                                            best_no_bid = no_bids[0].get('price', 0) / 100
                                            st.info(f"**NO Price:** {best_no_bid*100:.1f}¢ ({best_no_bid*100:.1f}% probability)")
                                        
                                        # Kelly recommendation
                                        if yes_bids:
                                            st.markdown("**💰 Kelly Sizing:**")
                                            # Assume user has 55% confidence (can be adjusted)
                                            user_prob = 0.55
                                            kelly_result = kalshi.calculate_kelly_for_kalshi(
                                                best_yes_bid, user_prob, 1000
                                            )
                                            st.write(kelly_result['recommendation'])
                                    else:
                                        st.warning("No orderbook data available")
                                
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            
            if len(filtered_markets) > 20:
                st.info(f"Showing top 20 of {len(filtered_markets)} markets. Adjust filters to see others.")
        
        else:
            st.info("👆 Click 'Load Kalshi Markets' to see available betting opportunities")
    
    elif analysis_mode == "⚖️ Compare with Sportsbooks":
        st.subheader("⚖️ Kalshi vs Sportsbook Odds Comparison")
        st.markdown("Compare prediction market pricing with traditional sportsbook odds to find value")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### Kalshi Market")
            kalshi_market_title = st.text_input(
                "Market Title",
                placeholder="e.g., Will Chiefs win Super Bowl?",
                key="kalshi_market_compare"
            )
            kalshi_yes_price = st.number_input(
                "YES Price (cents)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.1,
                help="Price in cents (e.g., 65 = 65% probability)"
            ) / 100
        
        with col_comp2:
            st.markdown("#### Sportsbook")
            sb_selection = st.text_input(
                "Corresponding Bet",
                placeholder="e.g., Chiefs Super Bowl Winner",
                key="sb_compare"
            )
            sb_odds = st.number_input(
                "Sportsbook Odds (American)",
                min_value=-10000,
                max_value=10000,
                value=-150,
                step=5,
                help="American odds (e.g., -150, +200)"
            )
        
        if st.button("🔍 Compare Markets", type="primary"):
            # Calculate sportsbook implied probability
            sb_prob = implied_p_from_american(sb_odds)
            
            # Comparison
            discrepancy = abs(kalshi_yes_price - sb_prob)
            edge = kalshi_yes_price - sb_prob
            
            st.markdown("---")
            st.markdown("### 📊 Comparison Results")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Kalshi Probability", f"{kalshi_yes_price*100:.1f}%")
            with col_r2:
                st.metric("Sportsbook Probability", f"{sb_prob*100:.1f}%")
            with col_r3:
                st.metric("Discrepancy", f"{discrepancy*100:.1f}%", 
                         delta=f"{edge*100:+.1f}%" if edge != 0 else None)
            
            # Recommendation
            st.markdown("### 💡 Recommendation")
            
            if discrepancy > 0.10:  # 10%+ difference
                if kalshi_yes_price < sb_prob:
                    st.success(f"🟢 **STRONG VALUE on Kalshi YES**")
                    st.write(f"- Kalshi is pricing YES at {kalshi_yes_price*100:.1f}%")
                    st.write(f"- Sportsbook implies {sb_prob*100:.1f}%")
                    st.write(f"- **Edge: {(sb_prob - kalshi_yes_price)*100:.1f}% in your favor**")
                    st.write(f"- **Action:** Buy YES on Kalshi")
                else:
                    st.success(f"🟢 **STRONG VALUE on Sportsbook**")
                    st.write(f"- Kalshi is overpricing at {kalshi_yes_price*100:.1f}%")
                    st.write(f"- Sportsbook implies {sb_prob*100:.1f}%")
                    st.write(f"- **Edge: {(kalshi_yes_price - sb_prob)*100:.1f}%**")
                    st.write(f"- **Action:** Take sportsbook bet OR buy NO on Kalshi")
            
            elif discrepancy > 0.05:  # 5-10% difference
                if kalshi_yes_price < sb_prob:
                    st.info(f"🟡 **MODERATE VALUE on Kalshi YES**")
                    st.write(f"- Small edge of {(sb_prob - kalshi_yes_price)*100:.1f}%")
                    st.write(f"- Consider Kalshi YES if you agree with sportsbook assessment")
                else:
                    st.info(f"🟡 **MODERATE VALUE on Sportsbook**")
                    st.write(f"- Small edge of {(kalshi_yes_price - sb_prob)*100:.1f}%")
                    st.write(f"- Consider sportsbook if you agree with that probability")
            
            else:
                st.success("✅ **MARKETS IN AGREEMENT**")
                st.write(f"- Both markets pricing very similarly")
                st.write(f"- Difference of only {discrepancy*100:.1f}%")
                st.write(f"- No significant arbitrage or value opportunity")
                st.write(f"- Bet on either if you have additional information/analysis")
            
            # Kelly calculation for Kalshi
            st.markdown("### 💰 Optimal Bet Sizing (Kalshi)")
            
            user_confidence = st.slider(
                "Your Confidence Level",
                min_value=0.0,
                max_value=1.0,
                value=sb_prob,
                step=0.01,
                format="%.1f%%",
                help="Your personal assessment of the probability"
            )
            
            bankroll = st.number_input(
                "Your Bankroll",
                min_value=100,
                max_value=1000000,
                value=1000,
                step=100
            )
            
            kelly = kalshi.calculate_kelly_for_kalshi(kalshi_yes_price, user_confidence, bankroll)
            
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                st.metric("Kelly %", f"{kelly['kelly_percentage']:.2f}%")
            with col_k2:
                st.metric("Recommended Stake", f"${kelly['recommended_stake']:.0f}")
            with col_k3:
                st.metric("Expected Value", f"{kelly['expected_value']*100:+.1f}%")
            
            st.write(kelly['recommendation'])
    
    elif analysis_mode == "💎 Find Arbitrage Opportunities":
        st.subheader("💎 Arbitrage Opportunity Scanner")
        st.markdown("Automatically find discrepancies between Kalshi and traditional sportsbooks")
        
        st.info("🔧 **Coming Soon:** This feature will automatically scan all markets and identify arbitrage opportunities where you can profit regardless of outcome by betting both sides.")
        
        st.markdown("""
        **How Arbitrage Works:**
        
        1. **Find Discrepancy:** Kalshi prices YES at 40% but sportsbook implies 50%
        2. **Bet Both Sides:** 
           - Bet YES on Kalshi (40¢)
           - Bet NO equivalent on sportsbook
        3. **Lock Profit:** Guaranteed profit regardless of outcome
        
        **Requirements:**
        - Discrepancy must be > 10% (to cover fees)
        - Sufficient liquidity on both sides
        - Fast execution (prices move quickly)
        
        **Manual Search:**
        Use the "Compare with Sportsbooks" tab above to manually check for arbitrage opportunities.
        """)
    
    # Educational Section
    st.markdown("---")
    st.markdown("### 📚 Understanding Kalshi Prediction Markets")
    
    with st.expander("🤔 What is Kalshi?"):
        st.markdown("""
        **Kalshi** is a CFTC-regulated prediction market where you can trade on real-world events:
        
        - **Legal & Regulated:** First CFTC-regulated event contract exchange in the US
        - **Binary Outcomes:** Markets settle to either 0¢ or 100¢
        - **Pricing:** Prices represent probability (65¢ = 65% chance)
        - **Liquidity:** Limit orderbook like stocks
        
        **Example:**
        - Market: "Will Chiefs win their next game?"
        - YES trading at 70¢ = Market thinks 70% chance
        - If you buy YES at 70¢ and Chiefs win, you get 100¢ (30¢ profit)
        - If they lose, you get 0¢ (lose your 70¢)
        """)
    
    with st.expander("💡 Why Compare with Sportsbooks?"):
        st.markdown("""
        **Different Pricing Mechanisms:**
        
        **Kalshi (Prediction Market):**
        - Wisdom of crowds pricing
        - Real money at stake
        - Efficient market hypothesis
        - Can be slower to react to news
        
        **Sportsbooks:**
        - Built-in vig (~5-10%)
        - Designed to balance book
        - React quickly to sharp money
        - Public bias can skew lines
        
        **Opportunities:**
        - ✅ Kalshi often has better prices (lower vig)
        - ✅ Arbitrage when markets disagree significantly
        - ✅ Value bets when you trust one pricing over the other
        - ✅ Hedge existing positions
        """)
    
    with st.expander("🎯 How to Use This Tool"):
        st.markdown("""
        **Step 1: Browse Markets**
        - Load Kalshi sports markets
        - Find events you're interested in
        - Check current pricing
        
        **Step 2: Compare Prices**
        - Find the same event on a sportsbook
        - Use "Compare with Sportsbooks" tab
        - Look for 5%+ discrepancies
        
        **Step 3: Make Decision**
        - If Kalshi < Sportsbook: Buy YES on Kalshi
        - If Kalshi > Sportsbook: Take sportsbook OR buy NO on Kalshi
        - If similar: Use your own analysis to decide
        
        **Step 4: Size Optimally**
        - Use Kelly Calculator
        - Input your confidence level
        - Bet recommended amount
        
        **Step 5: Track & Learn**
        - Monitor your positions
        - See which market was more accurate
        - Refine your strategy
        """)
    
    with st.expander("⚠️ Important Considerations"):
        st.markdown("""
        **Advantages of Kalshi:**
        - ✅ Lower fees than sportsbooks
        - ✅ Can exit position early (sell before event)
        - ✅ Legal in most US states
        - ✅ No betting limits (unlike sportsbooks)
        
        **Disadvantages:**
        - ⚠️ Lower liquidity than sportsbooks
        - ⚠️ Spreads can be wide on low-volume markets
        - ⚠️ Fewer markets available
        - ⚠️ Funds take time to withdraw
        
        **Best Practices:**
        - Only trade on high-volume markets
        - Check the spread before entering
        - Start small to learn the platform
        - Compare fees with sportsbook vig
        - Consider exit liquidity
        """)
    
    # Tips section
    st.markdown("---")
    st.markdown("""
    ### 💡 Kalshi Trading Tips:
    
    **Finding Value:**
    - ✅ Look for 5%+ discrepancies with sportsbooks
    - ✅ Check multiple sportsbooks for best comparison
    - ✅ Use AI analysis from other tabs to inform your view
    - ✅ Focus on high-volume markets (easier exit)
    
    **Risk Management:**
    - ✅ Use Kelly Criterion for position sizing
    - ✅ Don't tie up too much capital (lower liquidity)
    - ✅ Set maximum position sizes
    - ✅ Consider exit strategy before entering
    
    **Advanced Strategies:**
    - 📈 Arbitrage between Kalshi and sportsbooks
    - 📈 Hedge existing sportsbook bets on Kalshi
    - 📈 Take early profits by selling before event
    - 📈 Buy when news breaks before market adjusts

    **Combining with AI:**
    - Use Tab 2 sentiment analysis to validate Kalshi prices
    - Use Tab 3 custom builder to calculate fair value
    - Compare AI probability with Kalshi pricing
    - Bet when AI and Kalshi agree on value
    """)

# ===== TAB 5: API-SPORTS LIVE DATA =====
with main_tab5:
    apisports_client = st.session_state.get('apisports_client')
    if apisports_client is None:
        apisports_client = APISportsFootballClient()
        st.session_state['apisports_client'] = apisports_client

    st.header("🏈 API-Sports NFL Live Data")
    st.markdown("**Pull live NFL context (records, form, scoring trends) directly from api-sports.io**")
    st.caption("Provide your API-Sports key in the Sports Betting tab to enable these insights.")

    if not apisports_client or not apisports_client.is_configured():
        st.warning("Add your API-Sports key in the Sports Betting tab to load live NFL data.")
    else:
        default_tz = st.session_state.get('user_timezone', 'America/New_York')
        tz_input = st.text_input("Timezone (IANA)", value=default_tz, key="apisports_live_tz")
        try:
            tz_obj = pytz.timezone(tz_input)
        except Exception:
            tz_obj = pytz.timezone('UTC')
            st.warning("Invalid timezone. Using UTC for display.")

        game_date = st.date_input(
            "Game date",
            value=pd.Timestamp.now(tz_obj).date(),
            key="apisports_live_date"
        )

        if st.button("Fetch NFL games", key="fetch_apisports_games"):
            with st.spinner("Loading NFL schedule from API-Sports..."):
                games = apisports_client.get_games_by_date(game_date, timezone=tz_input)

            if not games:
                if apisports_client.last_error:
                    st.error(f"API-Sports error: {apisports_client.last_error}")
                else:
                    st.info("No NFL games found for this date.")
            else:
                st.success(f"✅ Loaded {len(games)} games")
                for raw_game in games:
                    summary = apisports_client.build_game_summary(raw_game, tz_name=tz_input)
                    home = summary.home
                    away = summary.away

                    st.markdown("---")
                    st.subheader(f"{away.name} @ {home.name}")

                    col_meta, col_home, col_away = st.columns([2, 2, 2])
                    with col_meta:
                        st.write(f"Kickoff: {summary.kickoff_local or 'TBD'}")
                        st.write(f"Status: {summary.status or 'Scheduled'}")
                        st.write(f"Venue: {summary.venue or 'TBD'}")
                        st.write(f"Stage: {summary.stage or 'Regular Season'}")
                    with col_home:
                        st.write(f"**Home: {home.name}**")
                        st.write(f"Record: {home.record or '—'}")
                        st.write(f"Form: {home.form or '—'}")
                        if home.average_points_for is not None:
                            st.write(f"Pts For: {home.average_points_for:.1f}")
                        if home.average_points_against is not None:
                            st.write(f"Pts Allowed: {home.average_points_against:.1f}")
                        if home.trend:
                            icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(home.trend, '📊')
                            st.write(f"Trend: {icon} {home.trend.capitalize()}")
                    with col_away:
                        st.write(f"**Away: {away.name}**")
                        st.write(f"Record: {away.record or '—'}")
                        st.write(f"Form: {away.form or '—'}")
                        if away.average_points_for is not None:
                            st.write(f"Pts For: {away.average_points_for:.1f}")
                        if away.average_points_against is not None:
                            st.write(f"Pts Allowed: {away.average_points_against:.1f}")
                        if away.trend:
                            icon = {'hot': '🔥', 'cold': '🥶', 'neutral': '⚪️'}.get(away.trend, '📊')
                            st.write(f"Trend: {icon} {away.trend.capitalize()}")

                if apisports_client.last_error:
                    st.info(f"API-Sports notice: {apisports_client.last_error}")

        st.markdown("---")
        st.subheader("🌐 API-Sports League Widget")
        st.caption(
            "Embed the official API-Sports widget to explore leagues beyond the NFL using the same API key."
        )

        widget_key = (
            st.session_state.get('apisports_api_key')
            or (apisports_client.api_key if apisports_client else "")
        )
        if not widget_key:
            st.info("Provide an API-Sports key in the Sports Betting tab to load the widget.")
        else:
            sport_labels = {
                "NFL": "nfl",
                "NBA": "nba",
                "MLB": "mlb",
                "NHL": "nhl",
                "NCAAB": "ncaab",
                "NCAAF": "ncaaf",
                "WNBA": "wnba",
                "MLS": "mls",
            }
            selected_label = st.selectbox(
                "Widget sport",
                options=list(sport_labels.keys()),
                index=0,
                key="apisports_widget_sport",
                help="Choose which league the API-Sports widget should highlight.",
            )
            theme_label = st.selectbox(
                "Widget theme",
                options=["Light", "Dark"],
                index=0,
                key="apisports_widget_theme",
            )
            show_widget_errors = st.checkbox(
                "Show API error messages in widget",
                value=True,
                key="apisports_widget_errors",
            )
            widget_height = st.slider(
                "Widget height (px)",
                400,
                900,
                600,
                50,
                key="apisports_widget_height",
            )

            widget_theme = "white" if theme_label == "Light" else "dark"
            widget_sport = sport_labels.get(selected_label, "nfl")

            widget_html = f"""
            <div class=\"apisports-widget\">
              <script type=\"module\" src=\"https://widgets.api-sports.io/2.0.3/widgets.js\"></script>
              <api-sports-widget data-type=\"config\"
                data-key=\"{escape(widget_key)}\"
                data-sport=\"{escape(widget_sport)}\"
                data-lang=\"en\"
                data-theme=\"{escape(widget_theme)}\"
                data-show-errors=\"{str(show_widget_errors).lower()}\"
              ></api-sports-widget>
              <api-sports-widget data-type=\"leagues\"></api-sports-widget>
            </div>
            """
            components.html(widget_html, height=widget_height, scrolling=True)

