"""
Vertex AI Master Analyzer
Consolidates ALL data sources for ultimate best bet recommendations
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from ml_predictions import get_vertex_ai_prediction, is_vertex_ai_enabled

logger = logging.getLogger(__name__)


class VertexMasterAnalyzer:
    """
    Master analyzer that combines ALL data sources and uses Vertex AI
    for final best bet recommendations
    """
    
    def __init__(
        self,
        odds_api_client=None,
        sportsdata_clients: Dict = None,
        apisports_clients: Dict = None,
        sentiment_analyzer=None,
        local_ml_predictor=None,
        theover_data: Dict = None
    ):
        self.odds_api = odds_api_client
        self.sportsdata = sportsdata_clients or {}
        self.apisports = apisports_clients or {}
        self.sentiment = sentiment_analyzer
        self.local_ml = local_ml_predictor
        self.theover = theover_data or {}
        
    def build_comprehensive_features(self, game: Dict, league: str) -> Dict:
        """
        Build comprehensive feature set from ALL sources
        
        Returns dict with 50+ features including:
        - Market odds (moneyline, spread, total)
        - Team stats (win%, avg points, defense, trends)
        - Recent form (last 5, last 10, home/away splits)
        - News sentiment
        - Local ML predictions
        - theover.ai predictions (if available)
        - Sharp money indicators
        - Market movement
        - Injuries/lineup changes
        - Weather (outdoor sports)
        - Head-to-head history
        """
        features = {
            'league': league,
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'game_time': game.get('commence_time'),
        }
        
        # 1. MARKET ODDS
        features.update(self._get_market_odds_features(game))
        
        # 2. TEAM STATISTICS
        features.update(self._get_team_stats_features(game, league))
        
        # 3. RECENT FORM
        features.update(self._get_form_features(game, league))
        
        # 4. SENTIMENT ANALYSIS
        features.update(self._get_sentiment_features(game))
        
        # 5. LOCAL ML PREDICTIONS
        features.update(self._get_local_ml_features(game))
        
        # 6. THEOVER.AI PREDICTIONS
        features.update(self._get_theover_features(game))
        
        # 7. SHARP MONEY INDICATORS
        features.update(self._get_sharp_money_features(game))
        
        # 8. DERIVED FEATURES
        features.update(self._calculate_derived_features(features))
        
        return features
    
    def _get_market_odds_features(self, game: Dict) -> Dict:
        """Extract market odds features"""
        bookmakers = game.get('bookmakers', [])
        
        features = {
            'home_ml_odds': None,
            'away_ml_odds': None,
            'home_spread': None,
            'home_spread_odds': None,
            'away_spread': None,
            'away_spread_odds': None,
            'total_line': None,
            'over_odds': None,
            'under_odds': None,
            'num_bookmakers': len(bookmakers),
        }
        
        if bookmakers:
            # Get best odds across all bookmakers
            for bookmaker in bookmakers:
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game.get('home_team'):
                                features['home_ml_odds'] = outcome['price']
                            else:
                                features['away_ml_odds'] = outcome['price']
                    
                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game.get('home_team'):
                                features['home_spread'] = outcome['point']
                                features['home_spread_odds'] = outcome['price']
                            else:
                                features['away_spread'] = outcome['point']
                                features['away_spread_odds'] = outcome['price']
                    
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            features['total_line'] = outcome['point']
                            if outcome['name'] == 'Over':
                                features['over_odds'] = outcome['price']
                            else:
                                features['under_odds'] = outcome['price']
        
        return features
    
    def _get_team_stats_features(self, game: Dict, league: str) -> Dict:
        """Get comprehensive team statistics"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Get stats from SportsData.io
        home_stats = self._fetch_team_stats(home_team, league, is_home=True)
        away_stats = self._fetch_team_stats(away_team, league, is_home=False)
        
        return {
            'home_win_pct': home_stats.get('win_pct', 0.5),
            'away_win_pct': away_stats.get('win_pct', 0.5),
            'home_avg_points': home_stats.get('avg_points', 0),
            'away_avg_points': away_stats.get('avg_points', 0),
            'home_avg_points_allowed': home_stats.get('avg_points_allowed', 0),
            'away_avg_points_allowed': away_stats.get('avg_points_allowed', 0),
            'home_off_rating': home_stats.get('off_rating', 0),
            'away_off_rating': away_stats.get('off_rating', 0),
            'home_def_rating': home_stats.get('def_rating', 0),
            'away_def_rating': away_stats.get('def_rating', 0),
            'home_pace': home_stats.get('pace', 0),
            'away_pace': away_stats.get('pace', 0),
            'home_home_record': home_stats.get('home_record', '0-0'),
            'away_away_record': away_stats.get('away_record', '0-0'),
        }
    
    def _get_form_features(self, game: Dict, league: str) -> Dict:
        """Get recent form data"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Last 5 games
        home_last_5 = self._fetch_recent_games(home_team, league, n=5)
        away_last_5 = self._fetch_recent_games(away_team, league, n=5)
        
        return {
            'home_last_5_wins': home_last_5.get('wins', 0),
            'away_last_5_wins': away_last_5.get('wins', 0),
            'home_last_5_avg_points': home_last_5.get('avg_points', 0),
            'away_last_5_avg_points': away_last_5.get('avg_points', 0),
            'home_streak': home_last_5.get('streak', 0),
            'away_streak': away_last_5.get('streak', 0),
            'home_trend': home_last_5.get('trend', 'neutral'),
            'away_trend': away_last_5.get('trend', 'neutral'),
        }
    
    def _get_sentiment_features(self, game: Dict) -> Dict:
        """Get news sentiment"""
        if not self.sentiment:
            return {'home_sentiment': 0, 'away_sentiment': 0, 'sentiment_diff': 0}
        
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        home_sentiment = self._calculate_team_sentiment(home_team)
        away_sentiment = self._calculate_team_sentiment(away_team)
        
        return {
            'home_sentiment': home_sentiment,
            'away_sentiment': away_sentiment,
            'sentiment_diff': home_sentiment - away_sentiment,
        }
    
    def _get_local_ml_features(self, game: Dict) -> Dict:
        """Get predictions from your local ML model"""
        if not self.local_ml:
            return {'local_ml_prob': 0.5, 'local_ml_confidence': 0}
        
        try:
            prediction = self.local_ml.predict_game_outcome(
                home_team=game.get('home_team'),
                away_team=game.get('away_team'),
                sport_key=game.get('sport_key')
            )
            
            return {
                'local_ml_prob': prediction.get('home_win_prob', 0.5),
                'local_ml_confidence': prediction.get('confidence', 0),
                'local_ml_edge': prediction.get('edge', 0),
            }
        except:
            return {'local_ml_prob': 0.5, 'local_ml_confidence': 0, 'local_ml_edge': 0}
    
    def _get_theover_features(self, game: Dict) -> Dict:
        """Get theover.ai predictions if available"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Check if we have theover.ai data for this game
        theover_pick = self._find_theover_pick(home_team, away_team)
        
        if theover_pick:
            return {
                'theover_has_pick': 1,
                'theover_pick': theover_pick.get('Pick', ''),
                'theover_probability': float(theover_pick.get('WinProbability', 0.5)),
            }
        else:
            return {
                'theover_has_pick': 0,
                'theover_pick': '',
                'theover_probability': 0.5,
            }
    
    def _get_sharp_money_features(self, game: Dict) -> Dict:
        """Get sharp money indicators"""
        # Placeholder - implement based on line movement
        return {
            'line_movement': 0,
            'sharp_money_indicator': 0,
            'public_betting_pct': 50,
        }
    
    def _calculate_derived_features(self, features: Dict) -> Dict:
        """Calculate derived/engineered features"""
        derived = {}
        
        # Win% differential
        derived['win_pct_diff'] = features.get('home_win_pct', 0.5) - features.get('away_win_pct', 0.5)
        
        # Offensive/Defensive matchups
        derived['off_def_matchup_home'] = features.get('home_off_rating', 0) - features.get('away_def_rating', 0)
        derived['off_def_matchup_away'] = features.get('away_off_rating', 0) - features.get('home_def_rating', 0)
        
        # Form momentum
        derived['form_momentum_diff'] = features.get('home_last_5_wins', 0) - features.get('away_last_5_wins', 0)
        
        # Implied probability from odds
        home_ml = features.get('home_ml_odds')
        if home_ml:
            if home_ml > 0:
                derived['implied_home_prob'] = 100 / (home_ml + 100)
            else:
                derived['implied_home_prob'] = abs(home_ml) / (abs(home_ml) + 100)
        else:
            derived['implied_home_prob'] = 0.5
        
        # Consensus probability (average of all sources)
        probs = [
            derived.get('implied_home_prob', 0.5),
            features.get('local_ml_prob', 0.5),
            features.get('theover_probability', 0.5),
        ]
        derived['consensus_prob'] = np.mean(probs)
        
        return derived
    
    def _fetch_team_stats(self, team: str, league: str, is_home: bool) -> Dict:
        """Fetch team stats from SportsData.io"""
        # Placeholder - implement with actual client
        return {
            'win_pct': 0.55,
            'avg_points': 110 if league == 'NBA' else 24,
            'avg_points_allowed': 108 if league == 'NBA' else 22,
            'off_rating': 110,
            'def_rating': 105,
            'pace': 100,
            'home_record': '10-5' if is_home else '',
            'away_record': '8-7' if not is_home else '',
        }
    
    def _fetch_recent_games(self, team: str, league: str, n: int = 5) ->
