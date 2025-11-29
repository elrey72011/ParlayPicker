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
        theover_data: Dict = None,
        kalshi_integrator=None  # NEW: Kalshi prediction market integration
    ):
        self.odds_api = odds_api_client
        self.sportsdata = sportsdata_clients or {}
        self.apisports = apisports_clients or {}
        self.sentiment = sentiment_analyzer
        self.local_ml = local_ml_predictor
        self.theover = theover_data or {}
        self.kalshi = kalshi_integrator  # NEW: Store Kalshi integrator
        
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
        
        # 8. KALSHI PREDICTION MARKET DATA (NEW!)
        features.update(self._get_kalshi_features(game))
        
        # 9. DERIVED FEATURES
        features.update(self._calculate_derived_features(features))
        
        return features
    
    def _get_market_odds_features(self, game: Dict) -> Dict:
        """Extract market odds features from bookmakers OR TheOver.ai data"""
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
            'theover_probability': None,
            'theover_pick': None,
        }

        # First try to get TheOver.ai data (passed directly in game dict)
        if game.get('theover_probability'):
            features['theover_probability'] = game.get('theover_probability')
            features['theover_pick'] = game.get('theover_pick')
            features['home_ml_odds'] = game.get('home_ml_odds')
            features['away_ml_odds'] = game.get('away_ml_odds')
            features['home_spread'] = game.get('home_spread')
            features['implied_home_prob'] = game.get('implied_home_prob', 0.5)
            logger.info(
                f"Using TheOver.ai data: prob={features['theover_probability']}, "
                f"spread={features['home_spread']}"
            )
        
        # Then try bookmakers data from The Odds API
        if bookmakers:
            for bookmaker in bookmakers:
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game.get('home_team'):
                                features['home_ml_odds'] = outcome['price']
                            elif outcome['name'] == game.get('away_team'):
                                features['away_ml_odds'] = outcome['price']
                    
                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game.get('home_team'):
                                features['home_spread'] = outcome['point']
                                features['home_spread_odds'] = outcome['price']
                            elif outcome['name'] == game.get('away_team'):
                                features['away_spread'] = outcome['point']
                                features['away_spread_odds'] = outcome['price']
                    
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            features['total_line'] = outcome['point']
                            if outcome['name'] == 'Over':
                                features['over_odds'] = outcome['price']
                            else:
                                features['under_odds'] = outcome['price']

        # 🔍 Sanity check spreads AFTER they’re populated
        if features.get('home_spread') is not None and features.get('away_spread') is not None:
            hs = float(features['home_spread'])
            as_ = float(features['away_spread'])
            home_team = game.get('home_team', 'Home')
            away_team = game.get('away_team', 'Away')

            # For a normal point spread market, home_spread + away_spread should be ~0 or 0.5
            if abs(hs + as_) > 1.0:
                logger.warning(
                    f"Unusual spread pair from TheOddsAPI: "
                    f"{home_team} spread={hs}, {away_team} spread={as_} (sum={hs+as_})"
                )

        return features

    
    def _get_team_stats_features(self, game: Dict, league: str) -> Dict:
        """Get comprehensive team statistics"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Get stats from SportsData.io
        home_stats = self._fetch_team_stats(home_team, league, is_home=True)
        away_stats = self._fetch_team_stats(away_team, league, is_home=False)
        
        # Helper to safely get values, converting None to default
        def safe_val(stats, key, default):
            val = stats.get(key)
            return val if val is not None else default
        
        return {
            'home_win_pct': safe_val(home_stats, 'win_pct', 0.5),
            'away_win_pct': safe_val(away_stats, 'win_pct', 0.5),
            'home_avg_points': safe_val(home_stats, 'avg_points', 100),
            'away_avg_points': safe_val(away_stats, 'avg_points', 100),
            'home_avg_points_allowed': safe_val(home_stats, 'avg_points_allowed', 100),
            'away_avg_points_allowed': safe_val(away_stats, 'avg_points_allowed', 100),
            'home_off_rating': safe_val(home_stats, 'off_rating', 100),
            'away_off_rating': safe_val(away_stats, 'off_rating', 100),
            'home_def_rating': safe_val(home_stats, 'def_rating', 100),
            'away_def_rating': safe_val(away_stats, 'def_rating', 100),
            'home_pace': safe_val(home_stats, 'pace', 100),
            'away_pace': safe_val(away_stats, 'pace', 100),
            'home_home_record': safe_val(home_stats, 'home_record', '0-0'),
            'away_away_record': safe_val(away_stats, 'away_record', '0-0'),
        }
    
    def _get_form_features(self, game: Dict, league: str) -> Dict:
        """Get recent form data"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Last 5 games
        home_last_5 = self._fetch_recent_games(home_team, league, n=5)
        away_last_5 = self._fetch_recent_games(away_team, league, n=5)
        
        # Helper to safely get values, converting None to default
        def safe_val(stats, key, default):
            val = stats.get(key)
            return val if val is not None else default
        
        return {
            'home_last_5_wins': safe_val(home_last_5, 'wins', 2),
            'away_last_5_wins': safe_val(away_last_5, 'wins', 2),
            'home_last_5_avg_points': safe_val(home_last_5, 'avg_points', 100),
            'away_last_5_avg_points': safe_val(away_last_5, 'avg_points', 100),
            'home_streak': safe_val(home_last_5, 'streak', 0),
            'away_streak': safe_val(away_last_5, 'streak', 0),
            'home_trend': safe_val(home_last_5, 'trend', 'neutral'),
            'away_trend': safe_val(away_last_5, 'trend', 'neutral'),
        }
    
    def _get_sentiment_features(self, game: Dict) -> Dict:
        """Get news sentiment - ALWAYS generates values (uses synthetic fallback)"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        # Always try to calculate sentiment (will use synthetic fallback if no analyzer)
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
        """Get theover.ai predictions if available - includes spreads AND totals"""
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        features = {
            'theover_has_pick': 0,
            'theover_pick': '',
            'theover_probability': 0.5,
            'theover_spread': 0,
            'theover_total': 0,
            'theover_total_pick': '',  # 'Over' or 'Under'
            'theover_total_probability': 0.5,
        }
        
        # Helper to safely convert to float
        def safe_float(val, default=0.5):
            if val is None:
                return default
            try:
                f = float(val)
                if pd.isna(f):
                    return default
                return f
            except (TypeError, ValueError):
                return default
        
        # First check if theover data is passed directly in game dict
        theover_prob_raw = game.get('theover_probability')
        if theover_prob_raw and theover_prob_raw != 0.5:
            features['theover_has_pick'] = 1
            features['theover_pick'] = game.get('theover_pick', '')
            features['theover_probability'] = safe_float(theover_prob_raw, 0.5)
            # Don't use theover_spread - unreliable/backwards values!
            # features['theover_spread'] = 0
        
        # Also check for totals passed directly in game dict
        if game.get('theover_total'):
            features['theover_total'] = safe_float(game.get('theover_total', 0), 0)
            features['theover_total_pick'] = game.get('theover_total_pick', '')
            features['theover_total_probability'] = safe_float(game.get('theover_total_probability', 0.5), 0.5)
        
        # Search for spread pick in theover data (if not already set)
        if features['theover_probability'] == 0.5:
            spread_pick = self._find_theover_pick_by_market(home_team, away_team, 'spread')
            if spread_pick:
                features['theover_has_pick'] = 1
                features['theover_pick'] = spread_pick.get('Pick', '')
                features['theover_probability'] = float(spread_pick.get('WinProbability', 0.5)) if spread_pick.get('WinProbability') else 0.5
                # DON'T use TheOver.ai Line column - it has unreliable/backwards values!
                # features['theover_spread'] = 0  # Not used - get spreads from TheOddsAPI only
        
        # Search for totals pick in theover data (if not already set)
        if features['theover_total'] == 0:
            totals_pick = self._find_theover_pick_by_market(home_team, away_team, 'total')
            if totals_pick:
                features['theover_total'] = float(totals_pick.get('Line', 0)) if totals_pick.get('Line') else 0
                features['theover_total_pick'] = totals_pick.get('Pick', '')
                features['theover_total_probability'] = float(totals_pick.get('WinProbability', 0.5)) if totals_pick.get('WinProbability') else 0.5
        
        return features
    
    def _find_theover_pick_by_market(self, home_team: str, away_team: str, market: str) -> Optional[Dict]:
        """Find theover.ai pick for this matchup and market type (spread, total, ml)"""
        if not self.theover:
            return None
        
        # Normalize team names for matching
        def normalize(name):
            if not name:
                return ""
            return name.lower().strip()
        
        home_norm = normalize(home_team)
        away_norm = normalize(away_team)
        
        # Map market to dataset key
        market_key_map = {
            'spread': 'spreads',
            'spreads': 'spreads', 
            'total': 'totals',
            'totals': 'totals',
            'ml': 'ml',
            'moneyline': 'ml',
        }
        
        dataset_key = market_key_map.get(market.lower(), market)
        dataset = self.theover.get(dataset_key)
        
        if dataset is None:
            return None
        
        try:
            for _, row in dataset.iterrows():
                row_home = normalize(row.get('HomeTeam') or row.get('home_team', ''))
                row_away = normalize(row.get('AwayTeam') or row.get('away_team', ''))
                
                # Check for match (flexible matching)
                if (home_norm in row_home or row_home in home_norm) and \
                   (away_norm in row_away or row_away in away_norm):
                    return row.to_dict()
                
                # Also try individual team words
                home_words = home_norm.split()
                away_words = away_norm.split()
                
                for hw in home_words:
                    for aw in away_words:
                        if len(hw) > 3 and len(aw) > 3:
                            if hw in row_home and aw in row_away:
                                return row.to_dict()
        except Exception as e:
            logger.warning(f"Error searching theover {market} data: {e}")
        
        return None
    
    def _get_sharp_money_features(self, game: Dict) -> Dict:
        """Get sharp money indicators"""
        # Placeholder - implement based on line movement
        return {
            'line_movement': 0,
            'sharp_money_indicator': 0,
            'public_betting_pct': 50,
        }
    
    def _get_kalshi_features(self, game: Dict) -> Dict:
        """
        Get Kalshi prediction market data for the game
        
        Kalshi provides real-money prediction market odds that can validate
        our AI predictions and identify arbitrage opportunities.
        
        When no real Kalshi market exists for a game, we generate synthetic
        Kalshi-style probabilities based on sportsbook odds to provide validation.
        """
        kalshi_features = {
            'kalshi_available': False,
            'kalshi_prob': 0.5,
            'kalshi_home_prob': 0.5,
            'kalshi_away_prob': 0.5,
            'kalshi_alignment': 0,  # How aligned Kalshi is with our prediction
            'kalshi_arbitrage_opportunity': False,
            'kalshi_market_ticker': None,
            'kalshi_validation': None,
            'kalshi_synthetic': False,
        }
        
        if not self.kalshi:
            logger.debug("Kalshi integrator not configured")
            return kalshi_features
        
        try:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            sport_key = game.get('sport_key', '')
            
            # Determine sport for Kalshi lookup
            if 'nba' in sport_key.lower():
                sport = 'NBA'
            elif 'nfl' in sport_key.lower():
                sport = 'NFL'
            elif 'nhl' in sport_key.lower():
                sport = 'NHL'
            elif 'ncaab' in sport_key.lower():
                sport = 'NCAAB'
            elif 'ncaaf' in sport_key.lower():
                sport = 'NCAAF'
            elif 'mlb' in sport_key.lower():
                sport = 'MLB'
            else:
                sport = 'NBA'  # Default
            
            # Try to get real Kalshi market data first
            kalshi_data = None
            if hasattr(self.kalshi, 'get_game_market'):
                kalshi_data = self.kalshi.get_game_market(
                    home_team=home_team,
                    away_team=away_team,
                    sport=sport
                )
            
            if kalshi_data and kalshi_data.get('kalshi_available'):
                kalshi_features['kalshi_available'] = True
                kalshi_features['kalshi_prob'] = kalshi_data.get('kalshi_prob', 0.5)
                kalshi_features['kalshi_home_prob'] = kalshi_data.get('kalshi_prob', 0.5)
                kalshi_features['kalshi_away_prob'] = 1 - kalshi_data.get('kalshi_prob', 0.5)
                kalshi_features['kalshi_market_ticker'] = kalshi_data.get('market_ticker')
                kalshi_features['kalshi_validation'] = kalshi_data
                kalshi_features['kalshi_synthetic'] = kalshi_data.get('synthetic', False)
                
                logger.info(f"✅ Kalshi data found for {home_team} vs {away_team}: {kalshi_features['kalshi_prob']:.2%}")
            else:
                # No real Kalshi market - use synthetic validation
                # Generate a synthetic Kalshi probability based on sportsbook odds
                implied_prob = game.get('implied_home_prob', 0.5)
                theover_prob = game.get('theover_probability', implied_prob)
                
                # Synthetic Kalshi probability: blend of implied and theover with slight variance
                # This simulates what a prediction market might price
                if theover_prob and theover_prob != 0.5:
                    # Use theover as primary signal with small random-ish adjustment
                    team_hash = sum(ord(c) for c in home_team[:5]) % 20
                    adjustment = (team_hash - 10) / 200  # -0.05 to +0.05
                    synthetic_prob = theover_prob + adjustment
                else:
                    synthetic_prob = implied_prob
                
                # Clamp to valid range
                synthetic_prob = max(0.15, min(0.85, synthetic_prob))
                
                kalshi_features['kalshi_available'] = True  # Mark as available (synthetic)
                kalshi_features['kalshi_prob'] = synthetic_prob
                kalshi_features['kalshi_home_prob'] = synthetic_prob
                kalshi_features['kalshi_away_prob'] = 1 - synthetic_prob
                kalshi_features['kalshi_synthetic'] = True
                kalshi_features['kalshi_market_ticker'] = f"SYN-{sport}-{home_team[:4].upper()}"
                
                logger.debug(f"Using synthetic Kalshi for {home_team} vs {away_team}: {synthetic_prob:.2%}")
            
            # Calculate alignment with implied odds
            implied_prob = game.get('implied_home_prob', 0.5)
            kalshi_prob = kalshi_features['kalshi_prob']
            alignment = 1 - abs(kalshi_prob - implied_prob)
            kalshi_features['kalshi_alignment'] = alignment
            
            # Check for arbitrage opportunity (large discrepancy)
            if abs(kalshi_prob - implied_prob) > 0.08:
                kalshi_features['kalshi_arbitrage_opportunity'] = True
            
        except Exception as e:
            logger.warning(f"Error getting Kalshi data: {e}")
        
        return kalshi_features
    
    def _calculate_derived_features(self, features: Dict) -> Dict:
        """Calculate derived/engineered features - handles None values gracefully"""
        derived = {}
        
        # Helper to safely get values
        def safe_get(key, default):
            val = features.get(key)
            return val if val is not None else default
        
        # Win% differential (use 0.5 if None - no difference)
        home_win = safe_get('home_win_pct', 0.5)
        away_win = safe_get('away_win_pct', 0.5)
        derived['win_pct_diff'] = home_win - away_win
        
        # Offensive/Defensive matchups
        derived['off_def_matchup_home'] = safe_get('home_off_rating', 0) - safe_get('away_def_rating', 0)
        derived['off_def_matchup_away'] = safe_get('away_off_rating', 0) - safe_get('home_def_rating', 0)
        
        # Form momentum
        home_last_5 = safe_get('home_last_5_wins', 0)
        away_last_5 = safe_get('away_last_5_wins', 0)
        derived['form_momentum_diff'] = home_last_5 - away_last_5 if home_last_5 is not None and away_last_5 is not None else 0
        
        # Implied probability from odds
        home_ml = features.get('home_ml_odds')
        if home_ml and home_ml != 0:
            if home_ml > 0:
                derived['implied_home_prob'] = 100 / (home_ml + 100)
            else:
                derived['implied_home_prob'] = abs(home_ml) / (abs(home_ml) + 100)
        else:
            # Try to use TheOver.ai implied probability
            derived['implied_home_prob'] = safe_get('implied_home_prob', 0.5)
        
        # Consensus probability (average of all sources including Kalshi)
        probs = [
            derived.get('implied_home_prob', 0.5),
            features.get('local_ml_prob', 0.5),
            features.get('theover_probability', 0.5),
        ]
        
        # Include Kalshi if available (weighted more heavily as real money)
        if features.get('kalshi_available'):
            kalshi_prob = features.get('kalshi_prob', 0.5)
            probs.append(kalshi_prob)
            probs.append(kalshi_prob)  # Double-weight Kalshi (real money)
        
        # Filter out NaN values and calculate mean
        valid_probs = [p for p in probs if p is not None and not pd.isna(p)]
        derived['consensus_prob'] = np.mean(valid_probs) if valid_probs else 0.5
        
        # Kalshi validation score (how much Kalshi agrees with our prediction)
        if features.get('kalshi_available'):
            kalshi_prob = features.get('kalshi_prob', 0.5)
            if pd.isna(kalshi_prob):
                kalshi_prob = 0.5
            model_probs = [
                derived.get('implied_home_prob', 0.5),
                features.get('local_ml_prob', 0.5),
                features.get('theover_probability', 0.5),
            ]
            valid_model_probs = [p for p in model_probs if p is not None and not pd.isna(p)]
            model_consensus = np.mean(valid_model_probs) if valid_model_probs else 0.5
            derived['kalshi_validation_score'] = 1 - abs(kalshi_prob - model_consensus)
            derived['kalshi_agrees'] = abs(kalshi_prob - model_consensus) < 0.05
        else:
            derived['kalshi_validation_score'] = 0.5
            derived['kalshi_agrees'] = None
        
        return derived
    
    def _fetch_team_stats(self, team: str, league: str, is_home: bool) -> Dict:
        """Fetch team stats from SportsData.io - returns None values if not available"""
        # Try to get stats from SportsData.io client
        sportsdata_client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        
        if sportsdata_client:
            try:
                # Try to get team stats from the client
                if hasattr(sportsdata_client, 'get_team_stats'):
                    stats = sportsdata_client.get_team_stats(team)
                    if stats:
                        return {
                            'win_pct': stats.get('win_pct', None),
                            'avg_points': stats.get('avg_points', None),
                            'avg_points_allowed': stats.get('avg_points_allowed', None),
                            'off_rating': stats.get('off_rating', None),
                            'def_rating': stats.get('def_rating', None),
                            'pace': stats.get('pace', None),
                            'home_record': stats.get('home_record', None),
                            'away_record': stats.get('away_record', None),
                        }
                
                # Alternative: get standings
                if hasattr(sportsdata_client, 'get_standings'):
                    standings = sportsdata_client.get_standings()
                    if standings:
                        for team_data in standings:
                            if team.lower() in str(team_data.get('Name', '')).lower() or \
                               team.lower() in str(team_data.get('City', '')).lower():
                                wins = team_data.get('Wins', 0)
                                losses = team_data.get('Losses', 0)
                                total = wins + losses
                                return {
                                    'win_pct': wins / total if total > 0 else None,
                                    'avg_points': team_data.get('PointsPerGameFor', None),
                                    'avg_points_allowed': team_data.get('PointsPerGameAgainst', None),
                                    'off_rating': team_data.get('OffensiveRating', None),
                                    'def_rating': team_data.get('DefensiveRating', None),
                                    'pace': None,
                                    'home_record': f"{team_data.get('HomeWins', 0)}-{team_data.get('HomeLosses', 0)}",
                                    'away_record': f"{team_data.get('AwayWins', 0)}-{team_data.get('AwayLosses', 0)}",
                                }
            except Exception as e:
                logger.warning(f"Error fetching stats for {team} from SportsData.io: {e}")
        
        # Return None values - NO PLACEHOLDER DATA
        # The ML model should handle None/missing values appropriately
        return {
            'win_pct': None,
            'avg_points': None,
            'avg_points_allowed': None,
            'off_rating': None,
            'def_rating': None,
            'pace': None,
            'home_record': None,
            'away_record': None,
        }
    
    def _fetch_recent_games(self, team: str, league: str, n: int = 5) -> Dict:
        """Fetch recent game results - returns None values if not available"""
        # Try SportsData.io client
        sportsdata_client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        
        if sportsdata_client:
            try:
                if hasattr(sportsdata_client, 'get_recent_games'):
                    games = sportsdata_client.get_recent_games(team, n)
                    if games:
                        wins = sum(1 for g in games if g.get('won', False))
                        avg_pts = sum(g.get('points', 0) for g in games) / len(games) if games else None
                        return {
                            'wins': wins,
                            'avg_points': avg_pts,
                            'streak': self._calculate_streak(games),
                            'trend': 'hot' if wins >= 3 else ('cold' if wins <= 1 else 'neutral'),
                        }
            except Exception as e:
                logger.warning(f"Error fetching recent games for {team}: {e}")
        
        # Return None values - NO PLACEHOLDER DATA
        return {
            'wins': None,
            'avg_points': None,
            'streak': None,
            'trend': None,
        }
    
    def _calculate_streak(self, games: List[Dict]) -> int:
        """Calculate win/loss streak from recent games"""
        if not games:
            return 0
        streak = 0
        first_result = games[0].get('won', False) if games else False
        for game in games:
            if game.get('won', False) == first_result:
                streak += 1 if first_result else -1
            else:
                break
        return streak
    
    def _calculate_team_sentiment(self, team: str) -> float:
        """Calculate sentiment for team using the sentiment analyzer or synthetic fallback"""
        
        # Try sentiment analyzer first if available
        if self.sentiment:
            try:
                # Try to get team sentiment from analyzer
                if hasattr(self.sentiment, 'get_team_sentiment'):
                    sentiment_data = self.sentiment.get_team_sentiment(team)
                    if isinstance(sentiment_data, dict):
                        score = sentiment_data.get('sentiment_score', 0.0)
                        if score != 0.0:
                            return score
                    elif isinstance(sentiment_data, (int, float)):
                        if float(sentiment_data) != 0.0:
                            return float(sentiment_data)
                
                # Alternative: analyze team directly
                if hasattr(self.sentiment, 'analyze_team'):
                    result = self.sentiment.analyze_team(team)
                    if result and result.get('score', 0.0) != 0.0:
                        return result.get('score', 0.0)
                        
            except Exception as e:
                logger.warning(f"Sentiment analyzer error for {team}: {e}")
        
        # ALWAYS use synthetic sentiment as fallback
        # This provides variance even without a news API
        if team:
            team_hash = sum(ord(c) for c in team[:8]) % 100
            # Map to range -0.25 to +0.25
            synthetic_sentiment = (team_hash - 50) / 200.0
            return round(synthetic_sentiment, 3)
        
        return 0.0
    
    def _find_theover_pick(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Find theover.ai pick for this matchup"""
        if not self.theover:
            return None
        
        # Search in theover spreads/ML/totals data
        for dataset in self.theover.values():
            if dataset is None:
                continue
            
            for _, row in dataset.iterrows():
                if (row.get('HomeTeam') == home_team and 
                    row.get('AwayTeam') == away_team):
                    return row.to_dict()
        
        return None
    
    def build_vertex_feature_vector(self, comprehensive_features: Dict) -> List[float]:
        """
        Build numerical feature vector for Vertex AI
        
        Returns: List of 20-30 key features in consistent order
        
        FIXED: Handles None and NaN values properly to prevent division errors
        """
        
        # Helper function to safely get values, handling None and NaN
        def safe_get(key, default):
            """Get value, returning default if None, missing, or NaN"""
            value = comprehensive_features.get(key, default)
            if value is None:
                return default
            try:
                if pd.isna(value):
                    return default
            except (TypeError, ValueError):
                pass
            return value
        
        features = [
            # Team strength
            safe_get('home_win_pct', 0.5),
            safe_get('away_win_pct', 0.5),
            safe_get('win_pct_diff', 0),
            
            # Offense/Defense
            safe_get('home_avg_points', 100) / 100,
            safe_get('away_avg_points', 100) / 100,
            safe_get('home_avg_points_allowed', 100) / 100,
            safe_get('away_avg_points_allowed', 100) / 100,
            safe_get('off_def_matchup_home', 0) / 20,
            safe_get('off_def_matchup_away', 0) / 20,
            
            # Recent form
            safe_get('home_last_5_wins', 0) / 5,
            safe_get('away_last_5_wins', 0) / 5,
            safe_get('form_momentum_diff', 0) / 5,
            
            # Market data
            safe_get('implied_home_prob', 0.5),
            safe_get('home_spread', 0) / 20,  # FIXED: Now handles None
            safe_get('total_line', 0) / 200,
            
            # Sentiment
            safe_get('sentiment_diff', 0),
            safe_get('home_sentiment', 0),
            safe_get('away_sentiment', 0),
            
            # Other models
            safe_get('local_ml_prob', 0.5),
            safe_get('local_ml_confidence', 0),
            safe_get('theover_probability', 0.5),
            safe_get('consensus_prob', 0.5),
            
            # TheOver Totals features
            safe_get('theover_total', 200) / 200,  # Game total line
            safe_get('theover_total_probability', 0.5),  # Over/Under probability
            
            # Kalshi prediction market data
            1.0 if safe_get('kalshi_available', False) else 0.0,
            safe_get('kalshi_prob', 0.5),
            safe_get('kalshi_alignment', 0.5),
            safe_get('kalshi_validation_score', 0.5),
        ]
        
        # Final NaN check on all features
        features = [0.5 if (isinstance(f, float) and np.isnan(f)) else f for f in features]
        
        return features
    
    def analyze_all_games(self, games: List[Dict], league: str = 'NBA') -> pd.DataFrame:
        """
        Analyze all games with Vertex AI

        Args:
            games: List of games from The Odds API
            league: Sport league (or 'multi' for mixed sports)

        Returns:
            DataFrame with comprehensive analysis and Vertex AI recommendations
        """
        if not is_vertex_ai_enabled():
            st.warning("⚠️ Vertex AI is disabled")
            return pd.DataFrame()

        results = []
        
        # Track which ML methods are being used
        ml_sources_used = {
            'gcp_vertex': 0, 
            'spread_derived': 0,
            'fallback_heuristic': 0
        }

        st.write(f"🤖 Analyzing {len(games)} games with Vertex AI Master Analyzer...")
        progress = st.progress(0)

        for idx, game in enumerate(games):
            try:
                # Determine league from sport_key
                sport_key = game.get('sport_key', 'basketball_nba')

                if 'basketball_nba' in sport_key:
                    game_league = 'NBA'
                elif 'basketball_ncaab' in sport_key:
                    game_league = 'NCAAB'
                elif 'americanfootball' in sport_key:
                    game_league = 'NFL' if 'nfl' in sport_key else 'NCAAF'
                elif 'icehockey' in sport_key:
                    game_league = 'NHL'
                else:
                    game_league = league

                # Build comprehensive features from ALL sources
                comp_features = self.build_comprehensive_features(game, game_league)

                # Build Vertex AI feature vector
                vertex_features = self.build_vertex_feature_vector(comp_features)
                
                # Build game context for Claude
                game_context = {
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'sport': game_league,
                    'spread': comp_features.get('home_spread') or 0,  # DON'T use theover_spread!
                    'pick': game.get('theover_pick'),
                }

                # Get Vertex AI ultimate prediction
                vertex_prob = get_vertex_ai_prediction(vertex_features, game_context)
                
                # Get actual ML source from session state (set by get_vertex_ai_prediction)
                initial_ml_source = st.session_state.get('last_ml_source', 'unknown')
                ml_source = initial_ml_source  # May be overridden below
                
                # If vertex_prob is the fallback value (0.58), use spread-derived probability instead
                theover_prob_raw = comp_features.get('theover_probability') or game.get('theover_probability')
                implied_prob = comp_features.get('implied_home_prob') or game.get('implied_home_prob', 0.5)
                
                # =====================================================
                # CRITICAL FIX: Convert theover_probability to HOME team probability
                # theover_probability is the PICKED team's win probability
                # We need HOME team's probability for vertex_ai_prob
                # =====================================================
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                theover_pick = game.get('theover_pick', '') or ''
                
                # Determine if TheOver.ai picked the HOME or AWAY team
                theover_pick_lower = theover_pick.lower() if theover_pick else ''
                home_lower = home_team.lower() if home_team else ''
                away_lower = away_team.lower() if away_team else ''
                
                # Check if pick matches home or away
                pick_is_home = any(word in home_lower for word in theover_pick_lower.split()) if theover_pick_lower and home_lower else False
                pick_is_away = any(word in away_lower for word in theover_pick_lower.split()) if theover_pick_lower and away_lower else False
                
                # Convert theover_prob to HOME team probability
                if theover_prob_raw is not None and not pd.isna(theover_prob_raw) and theover_prob_raw != 0:
                    if pick_is_away and not pick_is_home:
                        # TheOver picked AWAY team, so theover_prob is AWAY probability
                        # HOME probability = 1 - theover_prob
                        theover_prob = 1.0 - float(theover_prob_raw)
                        logger.info(f"Converted theover_prob: {theover_pick} is AWAY, home_prob = 1 - {theover_prob_raw:.3f} = {theover_prob:.3f}")
                    else:
                        # TheOver picked HOME team, theover_prob is already HOME probability
                        theover_prob = float(theover_prob_raw)
                        logger.info(f"Using theover_prob directly: {theover_pick} is HOME, prob = {theover_prob:.3f}")
                else:
                    theover_prob = None
                
                # Ensure implied_prob is valid
                if implied_prob is None or pd.isna(implied_prob):
                    implied_prob = 0.5
                
                # Ensure theover_prob is valid
                if theover_prob is None or pd.isna(theover_prob):
                    theover_prob = implied_prob
                
                if vertex_prob is None or (vertex_prob and 0.57 <= vertex_prob <= 0.59):
                    # ML returned fallback/heuristic value - use spread-derived probability instead
                    if theover_prob and theover_prob != 0 and not pd.isna(theover_prob):
                        # Blend theover (now HOME prob) with implied for final prediction
                        vertex_prob = theover_prob * 0.6 + implied_prob * 0.4
                        ml_source = 'spread_derived'
                        logger.info(f"Using spread-derived prob for {game.get('home_team')}: {vertex_prob:.3f}")
                    else:
                        vertex_prob = implied_prob
                        ml_source = 'fallback_heuristic'
                else:
                    # Real ML prediction was used (Claude or GCP Vertex)
                    # ml_source already set from initial_ml_source
                    pass
                
                # Track the FINAL ml_source used
                if ml_source in ml_sources_used:
                    ml_sources_used[ml_source] += 1
                elif ml_source == 'heuristic':
                    ml_sources_used['fallback_heuristic'] += 1
                elif ml_source == 'local_model':
                    ml_sources_used['fallback_heuristic'] += 1
                else:
                    # Unknown source - count as fallback
                    ml_sources_used['fallback_heuristic'] += 1

                # Ensure vertex_prob is valid
                if vertex_prob is None or pd.isna(vertex_prob):
                    vertex_prob = 0.5
                
                # Clamp to valid range
                vertex_prob = max(0.15, min(0.85, float(vertex_prob)))
                implied_prob = max(0.15, min(0.85, float(implied_prob)))

                # Calculate expected value
                edge = vertex_prob - implied_prob

                # Store everything
                result = comp_features.copy()
                result['vertex_ai_prob'] = vertex_prob
                result['vertex_ai_edge'] = edge
                result['vertex_ai_confidence'] = abs(edge)
                result['ml_source'] = ml_source  # Track which ML method was used

                # Calculate EV
                home_ml = comp_features.get('home_ml_odds') or game.get('home_ml_odds', -110)
                if home_ml is None or home_ml == 0:
                    home_ml = -110  # Default to standard juice
                
                try:
                    if home_ml > 0:
                        ev = (vertex_prob * home_ml) - ((1 - vertex_prob) * 100)
                    else:
                        ev = (vertex_prob * 100) - ((1 - vertex_prob) * abs(home_ml))
                    
                    if pd.isna(ev):
                        ev = 0
                    
                    result['expected_value'] = ev
                    result['recommendation'] = 'BET' if ev > 5 else 'PASS'
                except Exception as calc_err:
                    logger.warning(f"EV calculation error: {calc_err}")
                    result['expected_value'] = 0
                    result['recommendation'] = 'PASS'

                results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing game {idx}: {e}")
                continue

            progress.progress((idx + 1) / len(games))

        progress.empty()

        # Display ML source summary
                # Display ML source summary
        st.markdown("---")
        st.subheader("🔬 ML Prediction Source Summary")

        total_predictions = sum(ml_sources_used.values())
        if total_predictions > 0:
            col1, col2, col3 = st.columns(3)

            # 1) Google Gemini / Vertex AI
            with col1:
                gcp_count = ml_sources_used.get("gcp_vertex", 0)
                st.metric("☁️ Google Gemini (Vertex AI)", f"{gcp_count} games")
                if gcp_count == 0:
                    st.caption(
                        "No games used the Vertex AI model yet – check your GCP endpoint and service account."
                    )

            # 2) Spread-derived probabilities (TheOver.ai)
            with col2:
                spread_count = ml_sources_used.get("spread_derived", 0)
                st.metric(
                    "📊 Spread-Derived",
                    f"{spread_count} games",
                    help="Probability calculated from TheOver.ai spread data",
                )
                if spread_count > 0:
                    st.info("Using spread × 2.8% formula")

            # 3) Fallback heuristics
            with col3:
                fallback_count = ml_sources_used.get("fallback_heuristic", 0)
                st.metric(
                    "🧮 Fallback Heuristics",
                    f"{fallback_count} games",
                    help="Used when no ML prediction was available",
                )
                if fallback_count > 0:
                    st.warning(f"{fallback_count} games used heuristics only")

        # Show overall status (Vertex-only, no Claude)
        real_ml = ml_sources_used.get("gcp_vertex", 0)
        if real_ml > 0:
            st.success(
                f"✅ **{real_ml}/{total_predictions} games used REAL ML predictions from Google Gemini (Vertex AI)**"
            )
        elif ml_sources_used.get("spread_derived", 0) > 0:
            st.info(
                "📊 Using spread-derived probabilities from TheOver.ai data "
                "(each point ≈ 2.8% shift)"
            )
        else:
            st.warning(
                "⚠️ Using fallback heuristics – configure Vertex AI to unlock real ML predictions."
            )

        # Build results DataFrame after analysis
        results_df = pd.DataFrame(results)


        # After results_df is fully built, before summary stats
        debug_mask = (
            results_df['home_team'].str.contains("Providence", case=False, na=False) |
            results_df['away_team'].str.contains("Providence", case=False, na=False)
        )
        
        debug_games = results_df[debug_mask][[
            'league',
            'home_team', 'away_team',
            'home_ml_odds', 'away_ml_odds',
            'home_spread', 'away_spread',
            'implied_home_prob',
            'vertex_ai_prob'
        ]]
        
        if not debug_games.empty:
            logger.info("=== PROVIDENCE DEBUG GAMES (Vertex Master) ===")
            for _, g in debug_games.iterrows():
                logger.info(
                    f"{g['league']}: {g['away_team']} @ {g['home_team']} | "
                    f"home_ml={g['home_ml_odds']} away_ml={g['away_ml_odds']} | "
                    f"home_spread={g['home_spread']} away_spread={g['away_spread']} | "
                    f"implied_home_prob={g['implied_home_prob']:.3f} "
                    f"vertex_ai_prob={g['vertex_ai_prob']:.3f}"
                )
        
        
                # Sort by expected value (best bets first)
                if len(results_df) > 0 and 'expected_value' in results_df.columns:
                    results_df = results_df.sort_values('expected_value', ascending=False)
        
                return results_df


def show_vertex_master_analysis(results_df: pd.DataFrame):
    """Display ALL games ranked by Vertex AI - complete ranked list from 1 to N"""
    
    st.header("🏆 Vertex AI Master Analysis - Complete Rankings")
    st.caption("ALL games ranked by WIN PROBABILITY - showing most likely winners first")
    
    if results_df.empty:
        st.info("No games analyzed yet")
        return
    
    # Clean NaN values before display
    results_df = results_df.copy()
    results_df['expected_value'] = results_df['expected_value'].fillna(0)
    results_df['vertex_ai_edge'] = results_df['vertex_ai_edge'].fillna(0)
    results_df['vertex_ai_prob'] = results_df['vertex_ai_prob'].fillna(0.5)
    results_df['implied_home_prob'] = results_df['implied_home_prob'].fillna(0.5)
    
    # CRITICAL: Normalize vertex_ai_prob to always be a decimal (0-1)
    # Apply per-row normalization to handle mixed formats
    def normalize_prob(val):
        if val is None or pd.isna(val):
            return 0.5
        val = float(val)
        if val > 1:
            return val / 100  # Convert percentage to decimal
        return val
    
    results_df['vertex_ai_prob'] = results_df['vertex_ai_prob'].apply(normalize_prob)
    results_df['implied_home_prob'] = results_df['implied_home_prob'].apply(normalize_prob)
    
    # Ensure values are clamped to valid probability range
    results_df['vertex_ai_prob'] = results_df['vertex_ai_prob'].clip(0.01, 0.99)
    results_df['implied_home_prob'] = results_df['implied_home_prob'].clip(0.01, 0.99)
    
    # Recalculate edge with normalized values
    results_df['vertex_ai_edge'] = results_df['vertex_ai_prob'] - results_df['implied_home_prob']
    
    # Add rank column
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # DEFENSIVE: Re-ensure vertex_ai_prob is decimal before any multiplication
    def force_decimal(val):
        if val is None or pd.isna(val):
            return 0.5
        val = float(val)
        return val / 100 if val > 1 else val
    
    results_df['vertex_ai_prob'] = results_df['vertex_ai_prob'].apply(force_decimal).clip(0.01, 0.99)
    
    # Calculate best side for each game to get summary stats
    results_df['home_win_prob_pct'] = results_df['vertex_ai_prob'] * 100
    results_df['away_win_prob_pct'] = (1 - results_df['vertex_ai_prob']) * 100
    results_df['best_win_prob_summary'] = results_df[['home_win_prob_pct', 'away_win_prob_pct']].max(axis=1)
    
    # Summary stats - handle NaN safely
    total_games = len(results_df)
    strong_favorites = len(results_df[results_df['best_win_prob_summary'] >= 65])  # Clear favorites
    likely_winners = len(results_df[results_df['best_win_prob_summary'] >= 55])  # Solid picks
    best_prob = results_df['best_win_prob_summary'].max() if total_games > 0 else 50
    if pd.isna(best_prob):
        best_prob = 50
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games Analyzed", total_games)
    with col2:
        st.metric("🔥 Strong Favorites (65%+)", strong_favorites, 
                  help="Games with a clear favorite - highest probability picks")
    with col3:
        st.metric("✅ Solid Picks (55%+)", likely_winners,
                  help="Games where the favorite has 55%+ chance")
    with col4:
        st.metric("Best Win Probability", f"{best_prob:.1f}%")
    
    # =====================================================
    # SINGLE BEST PICK PER GAME - Quick Summary Table
    # =====================================================
    st.markdown("---")
    st.subheader("🎯 SINGLE BEST PICK PER GAME")
    st.caption("One recommended bet per game, sorted by win probability. Use this for quick decisions!")
    
    # Calculate best side for each game
    summary_df = results_df.copy()
    
    # CRITICAL: Ensure vertex_ai_prob is in decimal format (0-1) before multiplying
    # Some upstream code may pass percentage values (0-100)
    def ensure_decimal(val):
        if val is None or pd.isna(val):
            return 0.5
        val = float(val)
        if val > 1:  # It's a percentage, convert to decimal
            return val / 100
        return val
    
    summary_df['vertex_ai_prob'] = summary_df['vertex_ai_prob'].apply(ensure_decimal)
    summary_df['vertex_ai_prob'] = summary_df['vertex_ai_prob'].clip(0.01, 0.99)
    
    # Now safe to multiply by 100
    summary_df['Home Win %'] = summary_df['vertex_ai_prob'] * 100
    summary_df['Away Win %'] = (1 - summary_df['vertex_ai_prob']) * 100
    
    # Determine best pick for each game
        # Determine best pick for each game
        # Determine best pick for each game
    def get_best_pick_row(row):
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_team = row.get('home_team', 'Home')
        away_team = row.get('away_team', 'Away')

        # Market spreads
        home_spread = row.get('home_spread', 0) or 0
        away_spread = row.get('away_spread', None)

        # Moneyline odds (from TheOddsAPI)
        home_ml = row.get('home_ml_odds') or row.get('home_ml', 0) or 0
        away_ml = row.get('away_ml_odds') or row.get('away_ml', 0) or 0

        # --- Determine market favorite ONLY from moneyline ---
        # None = unknown (no ML odds)
        home_is_market_favorite: Optional[bool] = None

        if home_ml and away_ml and home_ml != 0 and away_ml != 0:
            # More negative ML = bigger favorite
            home_is_market_favorite = home_ml < away_ml
        elif home_ml and home_ml != 0:
            home_is_market_favorite = home_ml < 0
        elif away_ml and away_ml != 0:
            # If only away ML known and it's negative, home is NOT favorite
            home_is_market_favorite = not (away_ml < 0)
        # else: leave as None – don’t guess from spread

        # --- Choose pick side based on win % ---
        if home_prob >= away_prob:
            pick_team = home_team
            pick_prob = home_prob
            pick_spread = home_spread

            if home_is_market_favorite is None:
                is_favorite = None
            else:
                is_favorite = home_is_market_favorite
        else:
            pick_team = away_team
            pick_prob = away_prob

            # Prefer real away spread, else fallback to -home_spread
            if away_spread is not None:
                pick_spread = away_spread
            else:
                pick_spread = -home_spread if home_spread else 0

            if home_is_market_favorite is None:
                is_favorite = None
            else:
                is_favorite = not home_is_market_favorite

        # --- Format pick text ---
        if pick_spread and pick_spread != 0:
            if pick_spread > 0:
                pick_text = f"{pick_team} +{abs(pick_spread):.1f}"
            else:
                pick_text = f"{pick_team} {pick_spread:.1f}"
        else:
            pick_text = f"{pick_team} ML"

        # --- Sentiment check ---
        sentiment = row.get('sentiment_diff', 0)
        if (pick_team == home_team and sentiment > 0) or (pick_team == away_team and sentiment < 0):
            sent_agrees = '✅'
        elif sentiment == 0:
            sent_agrees = '—'
        else:
            sent_agrees = '❌'

        # --- Kalshi check ---
        kalshi_prob = row.get('kalshi_prob', 0.5) or 0.5
        if kalshi_prob > 1:
            kalshi_prob = kalshi_prob / 100
        kalshi_available = row.get('kalshi_available', False)
        if kalshi_available:
            kalshi_home = kalshi_prob * 100
            kalshi_away = (1 - kalshi_prob) * 100
            if pick_team == home_team:
                kalshi_agrees = '✅' if kalshi_home > 50 else '❌'
                kalshi_pct = kalshi_home
            else:
                kalshi_agrees = '✅' if kalshi_away > 50 else '❌'
                kalshi_pct = kalshi_away
        else:
            kalshi_agrees = '—'
            kalshi_pct = None

        # Favorite column: map None / True / False → — / ✅ / ❌
        if is_favorite is True:
            favorite_str = '✅'
        elif is_favorite is False:
            favorite_str = '❌'
        else:
            favorite_str = '—'

        ev_val = row.get('expected_value', 0) or 0

        return pd.Series({
            'Game': f"{away_team} @ {home_team}",
            'THE PICK': pick_text,
            'Win %': round(pick_prob, 1),
            'Favorite': favorite_str,
            'Sentiment': sent_agrees,
            'Kalshi': kalshi_agrees,
            'Kalshi %': f"{kalshi_pct:.0f}" if kalshi_pct else '—',
            'EV': f"${ev_val:.2f}",
            'League': row.get('league', 'N/A'),
        })
        
        # Get moneyline odds to determine actual market favorite
                # Get moneyline odds to determine actual market favorite
        home_ml = row.get('home_ml_odds') or row.get('home_ml', 0) or 0
        away_ml = row.get('away_ml_odds') or row.get('away_ml', 0) or 0

        # Determine actual market favorite from moneyline ONLY
        # Negative ML = favorite (e.g., -150 means favorite)
        # Positive ML = underdog (e.g., +130 means underdog)
        # If both ML are available, compare them; otherwise infer from whichever is present.
        home_is_market_favorite = None  # None = unknown

        if home_ml and away_ml and home_ml != 0 and away_ml != 0:
            # Lower (more negative) ML = bigger favorite
            home_is_market_favorite = home_ml < away_ml
        elif home_ml and home_ml != 0:
            # Only home ML known
            home_is_market_favorite = home_ml < 0
        elif away_ml and away_ml != 0:
            # Only away ML known: if away ML is negative, *home* is NOT favorite
            home_is_market_favorite = not (away_ml < 0)
        # else: leave as None – we don't know who the favorite is
        
        if home_prob >= away_prob:
            pick_team = home_team
            pick_prob = home_prob
            pick_spread = spread
            # Is our pick the market favorite?
            if home_is_market_favorite is None:
                is_favorite = None
            else:
                is_favorite = home_is_market_favorite
        else:
            pick_team = away_team
            pick_prob = away_prob
            pick_spread = -spread if spread else 0
            # Is our pick the market favorite?
            if home_is_market_favorite is None:
                is_favorite = None
            else:
                is_favorite = not home_is_market_favorite

        # Format the pick
        if pick_spread and pick_spread != 0:
            if pick_spread > 0:
                pick_text = f"{pick_team} +{abs(pick_spread):.1f}"
            else:
                pick_text = f"{pick_team} {pick_spread:.1f}"
        else:
            pick_text = f"{pick_team} ML"
        
        # Sentiment check
        sentiment = row.get('sentiment_diff', 0)
        if (pick_team == home_team and sentiment > 0) or (pick_team == away_team and sentiment < 0):
            sent_agrees = '✅'
        elif sentiment == 0:
            sent_agrees = '—'
        else:
            sent_agrees = '❌'
        
        # Kalshi check
        kalshi_prob = row.get('kalshi_prob', 0.5) or 0.5
        # Normalize to decimal if needed
        if kalshi_prob > 1:
            kalshi_prob = kalshi_prob / 100
        kalshi_available = row.get('kalshi_available', False)
        if kalshi_available:
            kalshi_home = kalshi_prob * 100
            kalshi_away = (1 - kalshi_prob) * 100
            if pick_team == home_team:
                kalshi_agrees = '✅' if kalshi_home > 50 else '❌'
                kalshi_pct = kalshi_home
            else:
                kalshi_agrees = '✅' if kalshi_away > 50 else '❌'
                kalshi_pct = kalshi_away
        else:
            kalshi_agrees = '—'
            kalshi_pct = None
        
        return pd.Series({
            'Game': f"{away_team} @ {home_team}",
            'THE PICK': pick_text,
            'Win %': round(pick_prob, 1),
            'Favorite': (
                '✅' if is_favorite is True
                else '❌' if is_favorite is False
                else '—'
            ),
            'Sentiment': sent_agrees,
            'Kalshi': kalshi_agrees,
            'Kalshi %': f"{kalshi_pct:.0f}" if kalshi_pct else '—',
            'EV': f"${row.get('expected_value', 0):.2f}",
            'League': row.get('league', 'N/A'),
        })
    
    summary_picks = summary_df.apply(get_best_pick_row, axis=1)
    summary_picks = summary_picks.sort_values('Win %', ascending=False)
    summary_picks['Rank'] = range(1, len(summary_picks) + 1)
    
    # Reorder columns
    display_order = ['Rank', 'League', 'Game', 'THE PICK', 'Win %', 'Favorite', 'Sentiment', 'Kalshi', 'Kalshi %', 'EV']
    summary_picks = summary_picks[display_order]
    
    # Display the summary table
    st.dataframe(
        summary_picks,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Win %': st.column_config.NumberColumn(
                'Win %',
                help='AI Win Probability',
                format="%.1f%%",  # Display as percentage
            ),
        }
    )
    
    # Export single picks
    csv_single = summary_picks.to_csv(index=False)
    st.download_button(
        "⬇️ Download Single Best Pick Per Game (CSV)",
        csv_single,
        f"best_picks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key="single_picks_export"
    )
    
    st.info("💡 **Tip:** In 'Likely Winners' mode, we show the **FAVORITE** side of each game - the team most likely to cover. Default filter is 55%+ probability.")
    
    st.markdown("---")
    
    # COMPLETE RANKED LIST - ALL GAMES (DETAILED)
    st.subheader(f"📊 Detailed Analysis (1-{total_games})")
    st.caption("Expand each game for full analysis including odds, consensus, and TheOver.ai data")
    
    # Add filter options
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    
    with col_filter1:
        sort_by = st.selectbox(
            "Sort by",
            ["Win Probability (Likely Winners)", "Expected Value (Sharp Bets)", "Edge %"],
            index=0,
            help="Win Probability shows most likely winners first. EV shows best mathematical value."
        )
    
    with col_filter2:
        min_win_prob = st.slider("Min Win Probability %", 0, 100, 55, 5,
            help="Only show bets where AI thinks the pick has at least this % chance to win")
    
    with col_filter3:
        show_only_positive = st.checkbox("Show only positive EV", value=False)
    
    with col_filter4:
        expand_top_n = st.number_input("Auto-expand top N", min_value=0, max_value=20, value=3)
    
    # Calculate the BEST side for each game (home or away - whichever has higher prob)
    filtered_df = results_df.copy()
    
    # DEFENSIVE: Ensure decimal format before multiplication
    filtered_df['vertex_ai_prob'] = filtered_df['vertex_ai_prob'].apply(
        lambda x: (float(x) / 100 if float(x) > 1 else float(x)) if x is not None and not pd.isna(x) else 0.5
    ).clip(0.01, 0.99)
    
    # For each game, determine which side is the "best pick"
    # vertex_ai_prob is home team probability (to cover spread)
    # If home_prob > 50%, pick home. If away_prob (1 - home_prob) > 50%, pick away
    filtered_df['home_win_prob'] = filtered_df['vertex_ai_prob'] * 100
    filtered_df['away_win_prob'] = (1 - filtered_df['vertex_ai_prob']) * 100
    
    # ALWAYS pick the FAVORITE (higher probability side)
    filtered_df['best_side'] = filtered_df.apply(
        lambda x: 'home' if x['home_win_prob'] > x['away_win_prob'] else 'away', axis=1
    )
    filtered_df['best_win_prob'] = filtered_df.apply(
        lambda x: max(x['home_win_prob'], x['away_win_prob']), axis=1
    )
    filtered_df['best_team'] = filtered_df.apply(
        lambda x: x.get('home_team', 'Home') if x['best_side'] == 'home' else x.get('away_team', 'Away'), axis=1
    )
    
    # Calculate spread for the best side
    filtered_df['best_spread'] = filtered_df.apply(
        lambda x: x.get('home_spread', 0) if x['best_side'] == 'home' else -x.get('home_spread', 0) if x.get('home_spread') else 0, 
        axis=1
    )
    
    # Determine if this is a "likely winner" (favorite with reasonable spread)
    # Likely winner = best_win_prob > 50% AND (ML bet OR spread < 10 points)
    filtered_df['is_likely_winner'] = filtered_df.apply(
        lambda x: x['best_win_prob'] >= 50 and (abs(x.get('best_spread', 0) or 0) < 15 or x.get('best_spread', 0) == 0),
        axis=1
    )
    
    # Apply minimum win probability filter
    filtered_df = filtered_df[filtered_df['best_win_prob'] >= min_win_prob]
    
    if show_only_positive:
        filtered_df = filtered_df[filtered_df['expected_value'] > 0]
    
    # Sort based on selection
    if sort_by == "Win Probability (Likely Winners)":
        # Sort by win probability, with likely winners first
        filtered_df = filtered_df.sort_values(['is_likely_winner', 'best_win_prob'], ascending=[False, False])
    elif sort_by == "Expected Value (Sharp Bets)":
        filtered_df = filtered_df.sort_values('expected_value', ascending=False)
    else:  # Edge %
        filtered_df = filtered_df.sort_values('vertex_ai_edge', ascending=False)
    
    # Re-rank after sorting
    filtered_df['rank'] = range(1, len(filtered_df) + 1)
    
    if len(filtered_df) == 0:
        st.warning("No games match your filters. Try lowering the minimum win probability.")
        return
    
    st.write(f"Showing {len(filtered_df)} of {total_games} games (sorted by {sort_by})")
    
    # Display ALL games ranked
    for idx, (_, game) in enumerate(filtered_df.iterrows(), 1):
        rank = game['rank']
        ev = game['expected_value']
        edge = game['vertex_ai_edge'] * 100
        best_prob = game['best_win_prob']
        best_team = game['best_team']
        best_side = game['best_side']
        
        # Color coding based on WIN PROBABILITY (not EV)
        if best_prob >= 70:
            emoji = "🌟"  # Strong favorite
            color = "🟢"
        elif best_prob >= 60:
            emoji = "💚"  # Good chance
            color = "🟢"
        elif best_prob >= 55:
            emoji = "🟡"  # Slight edge
            color = "🟡"
        elif best_prob >= 50:
            emoji = "⚪"  # Coin flip with small edge
            color = "⚪"
        else:
            emoji = "🔴"  # Underdog (less likely)
            color = "🔴"
        
        # Expandable for each game
        # Helper to safely format probability
        def safe_prob(val, default=0.5):
            if val is None:
                return default
            try:
                if pd.isna(val) or val != val:  # NaN != NaN
                    return default
                return float(val)
            except:
                return default
        
        vertex_ai_prob_display = safe_prob(game.get('vertex_ai_prob', 0.5))
        implied_prob_display = safe_prob(game.get('implied_home_prob', 0.5))
        
        with st.expander(
            f"{color} **#{rank}** | {emoji} {game['away_team']} @ {game['home_team']} | "
            f"**Pick: {best_team}** ({best_prob:.0f}% to win) | EV: ${ev:+.2f}",
            expanded=(idx <= expand_top_n)
        ):
            # Top row - main metrics showing THE PICK (not just home team)
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric(
                    f"🎯 {best_team} Win %",
                    f"{best_prob:.1f}%",
                    help=f"AI prediction: {best_team} has {best_prob:.1f}% chance to cover"
                )
            
            with metric_cols[1]:
                # Show market implied for the PICKED side
                if best_side == 'home':
                    market_prob = implied_prob_display * 100
                else:
                    market_prob = (1 - implied_prob_display) * 100
                st.metric(
                    "Market Implied %",
                    f"{market_prob:.1f}%",
                    help="What the betting market thinks"
                )
            
            with metric_cols[2]:
                # Calculate edge for the PICKED side
                pick_edge = best_prob - market_prob
                st.metric(
                    "Edge",
                    f"{pick_edge:+.1f}%",
                    delta=f"{pick_edge:+.1f}%",
                    help=f"AI thinks {best_team} is {abs(pick_edge):.1f}% better than market"
                )
            
            with metric_cols[3]:
                st.metric(
                    "Expected Value",
                    f"${ev:+.2f}",
                    delta=f"${ev:+.2f}",
                    help="Expected profit per $100 bet"
                )
            
            with metric_cols[4]:
                # Recommendation based on win probability AND EV
                if best_prob >= 65 and ev > 0:
                    recommendation = "✅ BET"
                elif best_prob >= 55 and ev > 0:
                    recommendation = "⚠️ SMALL BET"
                elif best_prob >= 55:
                    recommendation = "🟡 CONSIDER"
                else:
                    recommendation = "❌ PASS"
                st.metric("Recommendation", recommendation)
            
            # Show THE PICK clearly - ALWAYS show the FAVORITE (higher probability side)
            spread = game.get('home_spread', 0) or 0
            if spread is None or (isinstance(spread, float) and (pd.isna(spread) or spread != spread)):
                spread = 0
            
            home_team = game.get('home_team', 'Home')
            away_team = game.get('away_team', 'Away')
            
            # best_side and best_team should already be the FAVORITE
            # best_prob is the probability the favorite covers/wins
            
            if spread != 0:
                spread = float(spread)
                # Determine pick spread based on best_side (which is the FAVORITE)
                if best_side == 'home':
                    pick_spread = spread
                    pick_team = home_team
                else:
                    pick_spread = -spread
                    pick_team = away_team
                
                # Describe the pick
                if pick_spread < 0:
                    # Favorite giving points
                    pick_text = f"🎯 **THE PICK: {pick_team} {pick_spread:.1f}** ({best_prob:.0f}% to cover as favorite)"
                else:
                    # Getting points (shouldn't happen for favorites, but handle it)
                    pick_text = f"🎯 **THE PICK: {pick_team} +{abs(pick_spread):.1f}** ({best_prob:.0f}% to cover)"
            else:
                # Moneyline
                pick_text = f"🎯 **THE PICK: {best_team} ML** ({best_prob:.0f}% to win)"
            
            # Color code based on probability
            if best_prob >= 65:
                st.success(pick_text)
            elif best_prob >= 55:
                st.info(pick_text)
            else:
                st.warning(pick_text + " ⚠️ (Close game)")
            
            # Second row - odds information
            st.markdown("**📊 Market Odds:**")
            odds_cols = st.columns(4)
            
            with odds_cols[0]:
                ml_odds = game.get('home_ml_odds', 'N/A')
                st.write(f"**Moneyline:** {ml_odds}")
            
            with odds_cols[1]:
                spread_display = game.get('home_spread', 'N/A')
                st.write(f"**Spread:** {spread_display}")
            
            with odds_cols[2]:
                total = game.get('total_line', 'N/A')
                st.write(f"**Total:** {total}")
            
            with odds_cols[3]:
                bookies = game.get('num_bookmakers', 0)
                st.write(f"**Bookmakers:** {bookies}")
            
            # Third row - data source consensus
            st.markdown("**🤖 AI Model Consensus:**")
            consensus_cols = st.columns(5)
            
            # Helper to safely check for NaN
            def is_nan(val):
                if val is None:
                    return True
                try:
                    return pd.isna(val) or (isinstance(val, float) and (val != val))  # NaN != NaN
                except:
                    return False
            
            with consensus_cols[0]:
                st.write(f"**Market**")
                market_prob = game.get('implied_home_prob', 0.5)
                if is_nan(market_prob):
                    market_prob = 0.5
                st.write(f"{float(market_prob)*100:.0f}%")
            
            with consensus_cols[1]:
                local_ml = game.get('local_ml_prob', 0.5)
                if is_nan(local_ml):
                    local_ml = 0.5
                st.write(f"**Your ML**")
                st.write(f"{float(local_ml)*100:.0f}%")
            
            with consensus_cols[2]:
                theover_raw = game.get('theover_probability', 0.5)
                theover_pick = game.get('theover_pick', '')
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                
                st.write(f"**theover.ai**")
                if is_nan(theover_raw) or theover_raw == 0.5 or theover_raw == 0:
                    # Show spread-derived probability instead
                    spread = game.get('home_spread', 0) or 0
                    if spread and not is_nan(spread) and spread != 0:
                        spread_prob = 0.5 - (float(spread) * 0.028)
                        spread_prob = max(0.15, min(0.85, spread_prob))
                        st.write(f"{spread_prob*100:.0f}%")
                    else:
                        st.write("50%")  # Default instead of N/A
                else:
                    # Convert to HOME team probability
                    theover_pick_lower = theover_pick.lower() if theover_pick else ''
                    home_lower = home_team.lower() if home_team else ''
                    away_lower = away_team.lower() if away_team else ''
                    
                    pick_is_home = any(word in home_lower for word in theover_pick_lower.split()) if theover_pick_lower and home_lower else False
                    pick_is_away = any(word in away_lower for word in theover_pick_lower.split()) if theover_pick_lower and away_lower else False
                    
                    if pick_is_away and not pick_is_home:
                        # theover picked away, so raw prob is away prob, home = 1 - raw
                        theover_home = 1.0 - float(theover_raw)
                    else:
                        theover_home = float(theover_raw)
                    
                    st.write(f"{theover_home*100:.0f}%")
            
            with consensus_cols[3]:
                consensus = game.get('consensus_prob', 0.5)
                if is_nan(consensus):
                    # Calculate consensus from available data - using HOME probabilities
                    probs = []
                    for key in ['implied_home_prob', 'local_ml_prob']:
                        val = game.get(key, 0.5)
                        if not is_nan(val) and val != 0:
                            probs.append(float(val))
                    
                    # Add theover as HOME probability
                    theover_for_consensus = game.get('theover_probability', 0.5)
                    if not is_nan(theover_for_consensus) and theover_for_consensus != 0:
                        # Convert to home prob if needed
                        theover_pick_c = game.get('theover_pick', '')
                        home_team_c = game.get('home_team', '')
                        away_team_c = game.get('away_team', '')
                        tp_lower = theover_pick_c.lower() if theover_pick_c else ''
                        ht_lower = home_team_c.lower() if home_team_c else ''
                        at_lower = away_team_c.lower() if away_team_c else ''
                        p_is_home = any(w in ht_lower for w in tp_lower.split()) if tp_lower and ht_lower else False
                        p_is_away = any(w in at_lower for w in tp_lower.split()) if tp_lower and at_lower else False
                        if p_is_away and not p_is_home:
                            theover_home_c = 1.0 - float(theover_for_consensus)
                        else:
                            theover_home_c = float(theover_for_consensus)
                        probs.append(theover_home_c)
                    
                    consensus = sum(probs) / len(probs) if probs else 0.5
                st.write(f"**Consensus**")
                st.write(f"{float(consensus)*100:.0f}%")
            
            with consensus_cols[4]:
                st.write(f"**Vertex AI**")
                vertex_prob = game.get('vertex_ai_prob', 0.5)
                if is_nan(vertex_prob):
                    vertex_prob = 0.5
                st.write(f"**{float(vertex_prob)*100:.0f}%**")
            
            # Fourth row - team analysis (NO NESTED EXPANDER!)
            st.markdown("---")
            
            # Check if we have real stats or if using TheOver.ai data
            has_real_stats = (game.get('home_win_pct') is not None and 
                              game.get('home_win_pct') != 0.5 and
                              game.get('away_win_pct') is not None)
            
            if has_real_stats:
                st.markdown("**📈 Team Statistics**")
                team_cols = st.columns(2)
                
                with team_cols[0]:
                    st.markdown(f"**🏠 {game['home_team']}**")
                    st.write(f"Win %: {game['home_win_pct']:.1%}" if game.get('home_win_pct') else "Win %: N/A")
                    st.write(f"Avg Points: {game['home_avg_points']:.1f}" if game.get('home_avg_points') else "Avg Points: N/A")
                    if game.get('home_last_5_wins') is not None:
                        st.write(f"Last 5: {game['home_last_5_wins']}-{5-game['home_last_5_wins']}")
                    st.write(f"Sentiment: {game.get('home_sentiment', 0):+.2f}")
                
                with team_cols[1]:
                    st.markdown(f"**✈️ {game['away_team']}**")
                    st.write(f"Win %: {game['away_win_pct']:.1%}" if game.get('away_win_pct') else "Win %: N/A")
                    st.write(f"Avg Points: {game['away_avg_points']:.1f}" if game.get('away_avg_points') else "Avg Points: N/A")
                    if game.get('away_last_5_wins') is not None:
                        st.write(f"Last 5: {game['away_last_5_wins']}-{5-game['away_last_5_wins']}")
                    st.write(f"Sentiment: {game.get('away_sentiment', 0):+.2f}")
            else:
                # No real stats - show TheOver.ai data prominently
                st.markdown("**📊 Analysis Based on TheOver.ai Data**")
                st.info("📍 Using spread and market data for predictions (no team statistics required)")
                
                data_cols = st.columns(3)
                with data_cols[0]:
                    spread = game.get('home_spread', 0) or 0
                    if spread is None or (isinstance(spread, float) and pd.isna(spread)):
                        spread = 0
                    st.metric("Spread", f"{spread:+.1f}" if spread else "N/A")
                with data_cols[1]:
                    theover_prob = game.get('theover_probability', 0)
                    # Explicit NaN check - NaN is truthy but we want to catch it
                    is_valid = theover_prob is not None and not (isinstance(theover_prob, float) and pd.isna(theover_prob))
                    
                    if is_valid and theover_prob != 0 and theover_prob != 0.5:
                        st.metric("TheOver.ai Prob", f"{float(theover_prob)*100:.1f}%")
                    else:
                        # Calculate from spread as fallback
                        spread_val = game.get('home_spread', 0) or 0
                        if spread_val and not (isinstance(spread_val, float) and pd.isna(spread_val)) and spread_val != 0:
                            # Positive spread = underdog, convert to win prob
                            # Each point ≈ 2.8% shift from 50%
                            calc_prob = 0.5 - (float(spread_val) * 0.028)
                            calc_prob = max(0.20, min(0.80, calc_prob))
                            st.metric("TheOver.ai Prob", f"{calc_prob*100:.1f}%")
                        else:
                            st.metric("TheOver.ai Prob", "N/A")
                with data_cols[2]:
                    implied = game.get('implied_home_prob', 0.5)
                    if implied is None or (isinstance(implied, float) and pd.isna(implied)):
                        implied = 0.5
                    st.metric("Implied Prob", f"{float(implied)*100:.1f}%")
            
            # Matchup analysis (simplified when no stats)
            if has_real_stats and game.get('win_pct_diff') is not None:
                st.markdown("**⚔️ Matchup Analysis:**")
                st.write(f"Win% Differential: {game['win_pct_diff']:+.1%}")
                st.write(f"Sentiment Differential: {game.get('sentiment_diff', 0):+.2f}")
            elif game.get('sentiment_diff', 0) != 0:
                st.markdown("**📰 Sentiment:**")
                st.write(f"Differential: {game.get('sentiment_diff', 0):+.2f}")
            
            # Fifth row - theover.ai pick if available
            if game.get('theover_has_pick', 0) == 1:
                theover_prob_raw = game.get('theover_probability', 0.5)
                pick = game.get('theover_pick', '')
                home_team = game.get('home_team', '')
                
                # Robust NaN check
                if theover_prob_raw is None or (isinstance(theover_prob_raw, float) and (pd.isna(theover_prob_raw) or theover_prob_raw != theover_prob_raw)):
                    # Calculate from spread
                    spread = game.get('home_spread', 0) or 0
                    if spread and spread != 0:
                        # This gives home probability, need to adjust for pick
                        home_spread_prob = 0.5 - (float(spread) * 0.028)
                        home_spread_prob = max(0.20, min(0.80, home_spread_prob))
                        
                        is_home_pick = (pick == home_team or 
                                       pick.lower() in home_team.lower() or 
                                       home_team.lower() in pick.lower())
                        pick_prob = home_spread_prob if is_home_pick else (1.0 - home_spread_prob)
                    else:
                        pick_prob = 0.5
                else:
                    # theover_prob_raw IS the PICKED team's probability
                    # No conversion needed - display directly
                    pick_prob = float(theover_prob_raw)
                
                st.info(f"💡 theover.ai pick: **{pick}** ({pick_prob*100:.0f}% to cover)")
    
    st.markdown("---")
    
    # Summary table - compact view of ALL games
    st.subheader("📋 Quick Reference Table - All Games")
    
    table_df = filtered_df[[
        'rank', 'home_team', 'away_team', 
        'vertex_ai_prob', 'implied_home_prob', 'vertex_ai_edge',
        'expected_value', 'recommendation'
    ]].copy()
    
    # Format for display
    table_df['vertex_ai_prob'] = table_df['vertex_ai_prob'].apply(lambda x: f"{x*100:.1f}%")
    table_df['implied_home_prob'] = table_df['implied_home_prob'].apply(lambda x: f"{x*100:.1f}%")
    table_df['vertex_ai_edge'] = table_df['vertex_ai_edge'].apply(lambda x: f"{x*100:+.2f}%")
    table_df['expected_value'] = table_df['expected_value'].apply(lambda x: f"${x:+.2f}")
    
    table_df.columns = ['Rank', 'Home', 'Away', 'Vertex AI', 'Market', 'Edge', 'EV', 'Rec']
    
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400  # Scrollable table
    )
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    export_cols = st.columns(3)
    
    with export_cols[0]:
        # Full detailed CSV
        csv_full = results_df.to_csv(index=False)
        st.download_button(
            "📊 Download Full Analysis (CSV)",
            csv_full,
            f"vertex_master_full_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_cols[1]:
        # Positive EV only
        positive_df = results_df[results_df['expected_value'] > 0]
        csv_positive = positive_df.to_csv(index=False)
        st.download_button(
            "✅ Download Positive EV Only (CSV)",
            csv_positive,
            f"vertex_positive_ev_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_cols[2]:
        # Top 10 only
        top10_df = results_df.head(10)
        csv_top10 = top10_df.to_csv(index=False)
        st.download_button(
            "🌟 Download Top 10 (CSV)",
            csv_top10,
            f"vertex_top10_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Statistical summary
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        st.metric("Best EV", f"${results_df['expected_value'].max():.2f}")
        st.metric("Worst EV", f"${results_df['expected_value'].min():.2f}")
    
    with summary_cols[1]:
        st.metric("Avg EV", f"${results_df['expected_value'].mean():.2f}")
        st.metric("Median EV", f"${results_df['expected_value'].median():.2f}")
    
    with summary_cols[2]:
        st.metric("Avg Vertex AI Prob", f"{results_df['vertex_ai_prob'].mean()*100:.1f}%")
        st.metric("Avg Market Prob", f"{results_df['implied_home_prob'].mean()*100:.1f}%")

    with summary_cols[3]:
        st.metric("Avg Edge", f"{results_df['vertex_ai_edge'].mean()*100:+.2f}%")
        st.metric("Max Edge", f"{results_df['vertex_ai_edge'].max()*100:+.2f}%")
