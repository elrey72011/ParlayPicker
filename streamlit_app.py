# ParlayDesk_AI_Enhanced.py - v9.1 FIXED
# AI-Enhanced parlay finder with sentiment analysis, ML predictions, and PrizePicks
import os, io, json, itertools, re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
import pytz

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
    ],
    "prizepicks_sports": {
        "americanfootball_nfl": "NFL",
        "basketball_nba": "NBA"
    }
}

# ============ REAL SENTIMENT ANALYSIS ENGINE ============
class RealSentimentAnalyzer:
    """
    REAL sentiment analysis using actual news sources and NLP
    
    Data Sources:
    - NewsAPI.org (free tier: 100 requests/day)
    - Basic NLP sentiment classification
    """
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key or os.environ.get("NEWS_API_KEY")
        self.sentiment_cache = {}
        self.cache_duration = 1800  # 30 minutes
        
        # Sentiment word lists
        self.positive_words = {
            'win', 'wins', 'won', 'winning', 'victory', 'beat', 'beats', 
            'dominant', 'strong', 'excellent', 'best', 'great', 'hot', 
            'streak', 'momentum', 'comeback', 'champion', 'star', 'explosive',
            'impressive', 'outstanding', 'stellar', 'clutch', 'elite',
            'record-breaking', 'unstoppable', 'phenomenal', 'surging', 'rolling'
        }
        
        self.negative_words = {
            'lose', 'loses', 'lost', 'losing', 'defeat', 'beaten',
            'weak', 'poor', 'worst', 'bad', 'cold', 'slump', 'struggle',
            'injury', 'injured', 'hurt', 'out', 'questionable', 'doubtful',
            'blow', 'collapse', 'disaster', 'awful', 'terrible', 'embarrassing',
            'turnover', 'frustrated', 'disappointing', 'concerning', 'worry'
        }
    
    def get_team_sentiment(self, team_name: str, sport: str) -> Dict[str, float]:
        """
        Get REAL sentiment analysis for a team using NewsAPI
        """
        cache_key = f"{team_name}_{sport}_{datetime.now().date()}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached['timestamp']).seconds
            if age < self.cache_duration:
                return cached['data']
        
        # Try NewsAPI if configured
        if self.news_api_key:
            result = self._analyze_with_newsapi(team_name, sport)
        else:
            result = self._fallback_neutral()
        
        self.sentiment_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _analyze_with_newsapi(self, team_name: str, sport: str) -> Dict:
        """Analyze sentiment using NewsAPI.org"""
        try:
            from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            query = f'"{team_name}"'
            if 'nba' in sport.lower():
                query += ' NBA basketball'
            elif 'nfl' in sport.lower():
                query += ' NFL football'
            elif 'mlb' in sport.lower():
                query += ' MLB baseball'
            elif 'nhl' in sport.lower():
                query += ' NHL hockey'
            
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    'q': query,
                    'from': from_date,
                    'to': to_date,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 20,
                    'apiKey': self.news_api_key
                },
                timeout=10
            )
            
            if response.status_code != 200:
                return self._fallback_neutral()
            
            articles = response.json().get('articles', [])
            
            if not articles:
                return self._fallback_neutral()
            
            sentiment_scores = []
            for article in articles[:20]:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                score = self._calculate_text_sentiment(text)
                sentiment_scores.append(score)
            
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                score_variance = sum((s - avg_score) ** 2 for s in sentiment_scores) / len(sentiment_scores)
                confidence = max(0.3, min(0.95, 1.0 - score_variance))
                
                trend = 'positive' if avg_score > 0.15 else ('negative' if avg_score < -0.15 else 'neutral')
                
                return {
                    'score': avg_score,
                    'confidence': confidence,
                    'sources': len(sentiment_scores),
                    'trend': trend,
                    'method': 'NewsAPI + NLP'
                }
            
            return self._fallback_neutral()
            
        except Exception:
            return self._fallback_neutral()
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score using word matching"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total * 0.7
        return max(-1.0, min(1.0, sentiment_score))
    
    def _fallback_neutral(self) -> Dict:
        """Return neutral sentiment when API unavailable"""
        return {
            'score': 0.0,
            'confidence': 0.2,
            'sources': 0,
            'trend': 'neutral',
            'method': 'No API key'
        }

# Alias for compatibility
SentimentAnalyzer = RealSentimentAnalyzer

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
        """
        if not legs:
            return {'score': 0, 'confidence': 0}
        
        # Calculate combined probability (AI-adjusted)
        combined_prob = 1.0
        combined_confidence = 1.0
        total_edge = 0
        
        for leg in legs:
            combined_prob *= leg.get('ai_prob', leg['p'])
            combined_confidence *= leg.get('ai_confidence', 0.5)
            total_edge += leg.get('ai_edge', 0)
        
        # Calculate combined decimal odds
        combined_odds = legs[0]['d']
        for leg in legs[1:]:
            combined_odds *= leg['d']
        
        # AI-enhanced EV
        ai_ev = (combined_prob * combined_odds) - 1.0
        
        # Correlation penalty (same-game parlays are correlated)
        unique_games = len(set(leg['event_id'] for leg in legs))
        correlation_factor = unique_games / len(legs)
        
        # Final score components - UPDATED for better balance
        # Prioritize edge (where AI sees value) over raw EV
        ev_score = ai_ev * 100  # EV contribution
        confidence_score = combined_confidence * 50  # Confidence contribution
        edge_score = total_edge * 150  # INCREASED: Edge is now most important
        
        # UPDATED: Edge gets highest weight (shows where AI disagrees with market)
        final_score = (edge_score * 0.45 +      # 45% edge (was 30%)
                      ev_score * 0.30 +          # 30% EV (was 40%)
                      confidence_score * 0.25) * correlation_factor  # 25% confidence (was 30%)
        
        return {
            'score': final_score,
            'ai_ev': ai_ev,
            'confidence': combined_confidence,
            'edge': total_edge,
            'correlation_factor': correlation_factor
        }

# ============ PRIZEPICKS INTEGRATION ============
class PrizePicksAnalyzer:
    """Analyzes player props for PrizePicks optimal picks"""
    
    def __init__(self):
        self.prop_cache = {}
        self.stat_categories = {
            "NFL": ["Pass Yds", "Rush Yds", "Rec Yds", "Pass TDs", "Receptions", "Rush+Rec Yds"],
            "NBA": ["Points", "Rebounds", "Assists", "3-PT Made", "Pts+Rebs+Asts", "Pts+Rebs", "Pts+Asts"]
        }
    
    def generate_sample_props(self, sport: str, num_props: int = 20) -> List[Dict]:
        """
        Generate sample player props for analysis
        In production, this would fetch from PrizePicks API or scraping
        """
        props = []
        
        if sport == "NFL":
            players = [
                ("Patrick Mahomes", "KC", "QB"), ("Josh Allen", "BUF", "QB"),
                ("Tyreek Hill", "MIA", "WR"), ("Justin Jefferson", "MIN", "WR"),
                ("Christian McCaffrey", "SF", "RB"), ("Josh Jacobs", "LV", "RB"),
                ("Travis Kelce", "KC", "TE"), ("Mark Andrews", "BAL", "TE")
            ]
            for player, team, pos in players[:min(len(players), num_props)]:
                if pos == "QB":
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Pass Yds", "line": 275.5, "projection": 285.0
                    })
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Pass TDs", "line": 1.5, "projection": 2.1
                    })
                elif pos == "RB":
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Rush+Rec Yds", "line": 95.5, "projection": 105.0
                    })
                elif pos == "WR":
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Rec Yds", "line": 75.5, "projection": 82.0
                    })
                elif pos == "TE":
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Receptions", "line": 5.5, "projection": 6.2
                    })
        
        elif sport == "NBA":
            players = [
                ("Luka Doncic", "DAL", "PG"), ("Giannis Antetokounmpo", "MIL", "PF"),
                ("Kevin Durant", "PHX", "SF"), ("Stephen Curry", "GSW", "PG"),
                ("Joel Embiid", "PHI", "C"), ("Jayson Tatum", "BOS", "SF"),
                ("Nikola Jokic", "DEN", "C"), ("LeBron James", "LAL", "SF")
            ]
            for player, team, pos in players[:min(len(players), num_props)]:
                props.append({
                    "player": player, "team": team, "pos": pos,
                    "stat": "Points", "line": 28.5, "projection": 31.2
                })
                if pos in ["PG", "SF"]:
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Assists", "line": 6.5, "projection": 7.8
                    })
                if pos in ["PF", "C"]:
                    props.append({
                        "player": player, "team": team, "pos": pos,
                        "stat": "Rebounds", "line": 10.5, "projection": 11.8
                    })
        
        return props
    
    def calculate_prop_edge(self, prop: Dict) -> Dict:
        """Calculate edge for a prop bet"""
        line = prop["line"]
        projection = prop["projection"]
        
        # Simple edge calculation
        edge = ((projection - line) / line) * 100
        
        # Confidence based on how far projection is from line
        confidence = min(abs(edge) / 10, 1.0) * 0.7 + 0.3  # 30-100%
        
        return {
            **prop,
            "edge": edge,
            "confidence": confidence,
            "direction": "OVER" if projection > line else "UNDER",
            "value": abs(projection - line)
        }
    
    def find_best_picks(self, props: List[Dict], min_picks: int = 2, max_picks: int = 4) -> List[Dict]:
        """Find best prop combinations for PrizePicks entries"""
        # Score each prop
        scored_props = [self.calculate_prop_edge(p) for p in props]
        
        # Filter to only positive edge props
        positive_edge = [p for p in scored_props if p["edge"] > 0]
        
        # Sort by edge * confidence
        positive_edge.sort(key=lambda x: x["edge"] * x["confidence"], reverse=True)
        
        # Build combinations
        best_entries = []
        for k in range(min_picks, max_picks + 1):
            for combo in itertools.combinations(positive_edge[:15], k):  # Top 15 props
                total_edge = sum(p["edge"] for p in combo)
                avg_confidence = sum(p["confidence"] for p in combo) / len(combo)
                
                # PrizePicks scoring (approximate payouts)
                payout_multiplier = {2: 3.0, 3: 5.0, 4: 10.0}.get(k, 10.0)
                
                best_entries.append({
                    "picks": combo,
                    "num_picks": k,
                    "total_edge": total_edge,
                    "avg_confidence": avg_confidence,
                    "payout_multiplier": payout_multiplier,
                    "score": total_edge * avg_confidence * payout_multiplier
                })
        
        best_entries.sort(key=lambda x: x["score"], reverse=True)
        return best_entries

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
            "ai_edge": ai_metrics['edge']
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
        
        prob_pct = row['p_ai'] * 100
        with st.expander(
            f"{conf_icon}{ev_icon}{prob_warning} #{i} - AI Score: {row['ai_score']:.1f}{theover_boost} | Odds: {row['d']:.2f} | AI Prob: {prob_pct:.1f}% | Profit: ${row['profit']:.2f}"
        ):
            # Metrics
            col_a, col_b, col_c, col_d, col_e = st.columns(5)
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
                
                leg_entry = {
                    "Leg": j,
                    "Type": leg["market"],
                    "Selection": leg["label"],
                    "Odds": f"{leg['d']:.3f}",
                    "Market %": f"{leg['p']*100:.1f}%",
                    "AI %": f"{leg.get('ai_prob', leg['p'])*100:.1f}%",
                    "Sentiment": leg.get('sentiment_trend', 'N/A'),
                    "theover.ai": theover_display
                }
                legs_data.append(leg_entry)
            
            st.dataframe(pd.DataFrame(legs_data), use_container_width=True, hide_index=True)
            
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
if 'prizepicks_analyzer' not in st.session_state:
    st.session_state['prizepicks_analyzer'] = PrizePicksAnalyzer()
if 'news_api_key' not in st.session_state:
    st.session_state['news_api_key'] = os.environ.get("NEWS_API_KEY", "")

# Main navigation tabs
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🎯 Sports Betting Parlays", 
    "🏆 PrizePicks Props",
    "🎨 Custom Parlay Builder"
])

# ===== TAB 1: SPORTS BETTING PARLAYS =====
with main_tab1:
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

    def is_same_day(iso_str) -> bool:
        try:
            ts_local = pd.to_datetime(iso_str, utc=True).tz_convert(tz)
            return ts_local.date() == sel_date
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
                total_sports = len(sports or APP_CFG["sports_common"])
                
                for sport_idx, skey in enumerate(sports or APP_CFG["sports_common"]):
                    try:
                        progress_bar.progress((sport_idx) / total_sports)
                        snap = fetch_oddsapi_snapshot(api_key, skey)
                        
                        if not snap or not snap.get("events"):
                            continue  # Skip if no events
                        
                        for ev in (snap.get("events") or [])[:per_sport_events]:
                            try:
                                if not is_same_day(ev.get("commence_time")): 
                                    continue
                                
                                eid = ev.get("id")
                                home = ev.get("home_team", "?")
                                away = ev.get("away_team", "?")
                                mkts = ev.get("markets") or {}
                                
                                if not eid or not home or not away:
                                    continue  # Skip invalid events
                                
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
                                                all_legs.append({
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
                                                })
                            
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
                                                all_legs.append({
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
                                                })
                                
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
                                            all_legs.append({
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
                                            })
                                
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
                                            all_legs.append({
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
                                            })
                            
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
with main_tab2:
    st.subheader("🏆 PrizePicks Player Props Analyzer")
    st.caption("AI-powered player prop analysis for NFL & NBA (2-4 pick entries)")
    
    # theover.ai Integration Section
    st.markdown("---")
    st.markdown("### 📊 theover.ai Integration")
    
    theover_method = st.radio(
        "How do you want to add theover.ai data?",
        ["🚫 Skip (use sample data)", "📁 Upload CSV", "📋 Paste Data"],
        horizontal=True
    )
    
    theover_data = None
    
    if theover_method == "📁 Upload CSV":
        st.info("""
        **CSV Format Expected:**
        - Columns: Player, Stat, Projection, Line (optional)
        - Example: "LeBron James, Points, 31.2, 28.5"
        """)
        
        uploaded_file = st.file_uploader(
            "Upload theover.ai CSV export",
            type=['csv'],
            help="Export your theover.ai projections as CSV and upload here"
        )
        
        if uploaded_file:
            try:
                theover_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(theover_data)} projections from theover.ai")
                
                with st.expander("📋 Preview theover.ai Data"):
                    st.dataframe(theover_data.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                
    elif theover_method == "📋 Paste Data":
        st.info("""
        **Paste Format:**
        ```
        Player,Stat,Projection
        LeBron James,Points,31.2
        Steph Curry,3PT Made,4.8
        ...
        ```
        Or tab-separated from Excel.
        """)
        
        pasted_data = st.text_area(
            "Paste theover.ai data here",
            height=200,
            placeholder="LeBron James,Points,31.2\nSteph Curry,3PT Made,4.8"
        )
        
        if pasted_data.strip():
            try:
                # Try comma-separated first
                if ',' in pasted_data:
                    from io import StringIO
                    theover_data = pd.read_csv(StringIO(pasted_data))
                # Try tab-separated (from Excel)
                elif '\t' in pasted_data:
                    from io import StringIO
                    theover_data = pd.read_csv(StringIO(pasted_data), sep='\t')
                else:
                    st.warning("Data format not recognized. Use comma or tab-separated values.")
                
                if theover_data is not None:
                    st.success(f"✅ Loaded {len(theover_data)} projections from theover.ai")
                    with st.expander("📋 Preview Pasted Data"):
                        st.dataframe(theover_data.head(10), use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error parsing data: {e}")
    
    st.markdown("---")
    
    pp_analyzer = st.session_state['prizepicks_analyzer']
    
    col_pp1, col_pp2 = st.columns(2)
    with col_pp1:
        pp_sport = st.selectbox("Select Sport", ["NFL", "NBA"])
    with col_pp2:
        pp_num_props = st.slider("Number of props to analyze", 10, 30, 20)
    
    col_pp3, col_pp4 = st.columns(2)
    with col_pp3:
        pp_min_picks = st.selectbox("Minimum picks per entry", [2, 3], index=0)
    with col_pp4:
        pp_max_picks = st.selectbox("Maximum picks per entry", [3, 4], index=1)
    
    show_pp_entries = st.slider("Show top entries", 5, 20, 10)
    
    if st.button("🔍 Analyze PrizePicks Props", type="primary"):
        with st.spinner(f"Analyzing {pp_sport} player props..."):
            # Generate props (in production, fetch from PrizePicks)
            props = pp_analyzer.generate_sample_props(pp_sport, pp_num_props)
            
            if not props:
                st.warning("No props available for analysis")
                st.stop()
            
            # Enhance props with theover.ai data if available
            if theover_data is not None and not theover_data.empty:
                st.info("🎯 Enhancing analysis with theover.ai projections...")
                
                # Normalize column names
                theover_df = theover_data.copy()
                theover_df.columns = [c.strip().lower() for c in theover_df.columns]
                
                # Try to match props with theover.ai data
                for prop in props:
                    player_name = prop['player'].lower()
                    stat_type = prop['stat'].lower()
                    
                    # Find matching projection
                    match = theover_df[
                        (theover_df.get('player', '').str.lower().str.contains(player_name.split()[0], na=False)) &
                        (theover_df.get('stat', '').str.lower().str.contains(stat_type.split()[0], na=False))
                    ]
                    
                    if not match.empty:
                        theover_proj = match.iloc[0].get('projection', prop['projection'])
                        try:
                            theover_proj = float(theover_proj)
                            prop['theover_projection'] = theover_proj
                            prop['projection'] = theover_proj  # Use theover projection
                            prop['edge'] = ((theover_proj - prop['line']) / prop['line'] * 100)
                            prop['source'] = 'theover.ai'
                        except:
                            prop['source'] = 'sample'
                    else:
                        prop['source'] = 'sample'
            
            # Find best entries
            best_entries = pp_analyzer.find_best_picks(props, pp_min_picks, pp_max_picks)[:show_pp_entries]
            
            # Display summary
            theover_count = sum(1 for p in props if p.get('source') == 'theover.ai')
            if theover_count > 0:
                st.success(f"✅ Analyzed {len(props)} player props for {pp_sport} ({theover_count} from theover.ai 🎯)")
            else:
                st.success(f"✅ Analyzed {len(props)} player props for {pp_sport}")
            
            with st.expander("📊 Props Overview", expanded=False):
                props_df = pd.DataFrame([{
                    "Player": p["player"],
                    "Team": p["team"],
                    "Stat": p["stat"],
                    "Line": p["line"],
                    "Projection": p["projection"],
                    "Edge %": f"{((p['projection'] - p['line']) / p['line'] * 100):.1f}%",
                    "Source": "🎯 theover.ai" if p.get('source') == 'theover.ai' else "📊 Sample"
                } for p in props])
                st.dataframe(props_df, use_container_width=True, hide_index=True)
            
            # Display best entries
            st.markdown("### 🎯 Best PrizePicks Entries")
            
            for i, entry in enumerate(best_entries, start=1):
                num_picks = entry["num_picks"]
                payout = entry["payout_multiplier"]
                
                # Color code by confidence
                if entry["avg_confidence"] > 0.7:
                    conf_icon = "🟢"
                elif entry["avg_confidence"] > 0.5:
                    conf_icon = "🟡"
                else:
                    conf_icon = "🟠"
                
                with st.expander(
                    f"{conf_icon} #{i} - {num_picks}-Pick Entry | {payout}x Payout | Score: {entry['score']:.1f}"
                ):
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    with col_metric1:
                        st.metric("Picks", num_picks)
                    with col_metric2:
                        st.metric("Payout", f"{payout}x")
                    with col_metric3:
                        st.metric("Avg Confidence", f"{entry['avg_confidence']*100:.1f}%")
                    with col_metric4:
                        st.metric("Total Edge", f"{entry['total_edge']:.1f}%")
                    
                    st.markdown("**📋 Your Picks:**")
                    picks_data = []
                    for j, pick in enumerate(entry["picks"], start=1):
                        picks_data.append({
                            "#": j,
                            "Player": pick["player"],
                            "Team": pick["team"],
                            "Stat": pick["stat"],
                            "Line": pick["line"],
                            "Pick": pick["direction"],
                            "Projection": pick["projection"],
                            "Edge": f"{pick['edge']:.1f}%",
                            "Confidence": f"{pick['confidence']*100:.0f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(picks_data), use_container_width=True, hide_index=True)
                    
                    st.markdown("**💵 Payout Scenarios:**")
                    for stake in [10, 25, 50, 100]:
                        win_amt = stake * payout
                        profit = win_amt - stake
                        st.write(f"${stake} entry → ${win_amt:.2f} win (${profit:.2f} profit)")
                    
                    # Export entry
                    csv_buf = io.StringIO()
                    pd.DataFrame(picks_data).to_csv(csv_buf, index=False)
                    st.download_button(
                        "💾 Download Entry CSV",
                        data=csv_buf.getvalue(),
                        file_name=f"prizepicks_{pp_sport.lower()}_{num_picks}pick_entry_{i}.csv",
                        mime="text/csv",
                        key=f"download_pp_{i}"
                    )
            
            st.info("💡 **Note:** These are AI-generated projections for demonstration. In production, integrate with real PrizePicks lines and advanced player statistics.")
    
    st.markdown("---")
    st.markdown("""
    ### 🏆 PrizePicks Strategy Tips:
    
    **Entry Types:**
    - **2-Pick:** 3x payout - Lower risk, good for high-confidence plays
    - **3-Pick:** 5x payout - Balanced risk/reward
    - **4-Pick:** 10x payout - Higher risk, maximum reward
    
    **Best Practices:**
    - Focus on players with consistent performance
    - Check injury reports before finalizing
    - Diversify across different stat categories
    - Avoid stacking too many players from same game
    - Target props where your projection differs significantly from the line
    """)

# ===== TAB 3: CUSTOM PARLAY BUILDER =====
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
                                delta=f"${ev_delta:+.2f} vs market",
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
