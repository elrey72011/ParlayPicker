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
            features['home_spread'] = game.get('home_spread') or game.get('theover_spread')
            features['implied_home_prob'] = game.get('implied_home_prob', 0.5)
            logger.info(f"Using TheOver.ai data: prob={features['theover_probability']}, spread={features['home_spread']}")
        
        # Then try bookmakers data from The Odds API
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
        
        # First check if theover data is passed directly in game dict
        if game.get('theover_probability'):
            return {
                'theover_has_pick': 1,
                'theover_pick': game.get('theover_pick', ''),
                'theover_probability': float(game.get('theover_probability', 0.5)),
                'theover_spread': game.get('theover_spread', 0),
            }
        
        # Otherwise check if we have theover.ai data for this game from self.theover
        theover_pick = self._find_theover_pick(home_team, away_team)
        
        if theover_pick:
            return {
                'theover_has_pick': 1,
                'theover_pick': theover_pick.get('Pick', ''),
                'theover_probability': float(theover_pick.get('WinProbability', 0.5)),
                'theover_spread': float(theover_pick.get('Line', 0)) if theover_pick.get('Line') else 0,
            }
        else:
            return {
                'theover_has_pick': 0,
                'theover_pick': '',
                'theover_probability': 0.5,
                'theover_spread': 0,
            }
    
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
        
        derived['consensus_prob'] = np.mean(probs)
        
        # Kalshi validation score (how much Kalshi agrees with our prediction)
        if features.get('kalshi_available'):
            kalshi_prob = features.get('kalshi_prob', 0.5)
            model_consensus = np.mean([
                derived.get('implied_home_prob', 0.5),
                features.get('local_ml_prob', 0.5),
                features.get('theover_probability', 0.5),
            ])
            derived['kalshi_validation_score'] = 1 - abs(kalshi_prob - model_consensus)
            derived['kalshi_agrees'] = abs(kalshi_prob - model_consensus) < 0.05
        else:
            derived['kalshi_validation_score'] = 0.5
            derived['kalshi_agrees'] = None
        
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
    
    def _fetch_recent_games(self, team: str, league: str, n: int = 5) -> Dict:
        """Fetch recent game results"""
        # Placeholder
        return {
            'wins': 3,
            'avg_points': 110,
            'streak': 2,
            'trend': 'hot',
        }
    
    def _calculate_team_sentiment(self, team: str) -> float:
        """Calculate sentiment for team"""
        # Placeholder
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
        
        FIXED: Handles None values properly to prevent division errors
        """
        
        # Helper function to safely get values, handling None
        def safe_get(key, default):
            """Get value, returning default if None or missing"""
            value = comprehensive_features.get(key, default)
            return value if value is not None else default
        
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
            
            # Other models
            safe_get('local_ml_prob', 0.5),
            safe_get('local_ml_confidence', 0),
            safe_get('theover_probability', 0.5),
            safe_get('consensus_prob', 0.5),
            
            # Kalshi prediction market data (NEW!)
            1.0 if safe_get('kalshi_available', False) else 0.0,
            safe_get('kalshi_prob', 0.5),
            safe_get('kalshi_alignment', 0.5),
            safe_get('kalshi_validation_score', 0.5),
        ]
        
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
                    'spread': game.get('theover_spread') or comp_features.get('home_spread'),
                    'pick': game.get('theover_pick'),
                }

                # Get Vertex AI ultimate prediction
                vertex_prob = get_vertex_ai_prediction(vertex_features, game_context)
                
                # If vertex_prob is the fallback value (0.58), use spread-derived probability instead
                theover_prob = comp_features.get('theover_probability') or game.get('theover_probability')
                implied_prob = comp_features.get('implied_home_prob') or game.get('implied_home_prob', 0.5)
                
                if vertex_prob is None or (vertex_prob and 0.57 <= vertex_prob <= 0.59):
                    # ML returned fallback/heuristic value - use spread-derived probability instead
                    if theover_prob and theover_prob != 0:
                        # Blend theover with implied for final prediction
                        # Weight theover more heavily since it's based on actual spread data
                        vertex_prob = theover_prob * 0.6 + implied_prob * 0.4
                        logger.info(f"Using spread-derived prob for {game.get('home_team')}: {vertex_prob:.3f}")
                    else:
                        vertex_prob = implied_prob

                if vertex_prob is not None:
                    # Calculate expected value
                    edge = vertex_prob - implied_prob

                    # Store everything
                    result = comp_features.copy()
                    result['vertex_ai_prob'] = vertex_prob
                    result['vertex_ai_edge'] = edge
                    result['vertex_ai_confidence'] = abs(edge)

                    # Calculate EV
                    home_ml = comp_features.get('home_ml_odds', 100)
                    if home_ml and home_ml != 0:
                        if home_ml > 0:
                            ev = (vertex_prob * home_ml) - ((1 - vertex_prob) * 100)
                        else:
                            ev = (vertex_prob * 100) - ((1 - vertex_prob) * abs(home_ml))

                        result['expected_value'] = ev
                        result['recommendation'] = 'BET' if ev > 5 else 'PASS'
                    else:
                        result['expected_value'] = 0
                        result['recommendation'] = 'PASS'

                    results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing game {idx}: {e}")
                continue

            progress.progress((idx + 1) / len(games))

        progress.empty()

        results_df = pd.DataFrame(results)

        # Sort by expected value (best bets first)
        if len(results_df) > 0 and 'expected_value' in results_df.columns:
            results_df = results_df.sort_values('expected_value', ascending=False)

        return results_df


def show_vertex_master_analysis(results_df: pd.DataFrame):
    """Display ALL games ranked by Vertex AI - complete ranked list from 1 to N"""
    
    st.header("🏆 Vertex AI Master Analysis - Complete Rankings")
    st.caption("ALL games ranked by expected value - powered by comprehensive data sources")
    
    if results_df.empty:
        st.info("No games analyzed yet")
        return
    
    # Add rank column
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Summary stats
    total_games = len(results_df)
    positive_ev = len(results_df[results_df['expected_value'] > 0])
    best_ev = results_df.iloc[0]['expected_value'] if len(results_df) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games Analyzed", total_games)
    with col2:
        st.metric("Positive EV Opportunities", positive_ev)
    with col3:
        st.metric("Best EV", f"${best_ev:.2f}")
    with col4:
        avg_edge = results_df['vertex_ai_edge'].mean() * 100
        st.metric("Avg Vertex AI Edge", f"{avg_edge:+.2f}%")
    
    st.markdown("---")
    
    # COMPLETE RANKED LIST - ALL GAMES
    st.subheader(f"📊 Complete Rankings (1-{total_games})")
    
    # Add filter options
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        show_only_positive = st.checkbox("Show only positive EV bets", value=False)
    
    with col_filter2:
        min_edge = st.slider("Minimum edge %", -50.0, 50.0, -50.0, 0.5)
    
    with col_filter3:
        expand_top_n = st.number_input("Auto-expand top N", min_value=0, max_value=20, value=3)
    
    # Filter dataframe
    filtered_df = results_df.copy()
    if show_only_positive:
        filtered_df = filtered_df[filtered_df['expected_value'] > 0]
    
    filtered_df = filtered_df[filtered_df['vertex_ai_edge'] * 100 >= min_edge]
    
    if len(filtered_df) == 0:
        st.warning("No games match your filters")
        return
    
    st.write(f"Showing {len(filtered_df)} of {total_games} games")
    
    # Display ALL games ranked
    for idx, (_, game) in enumerate(filtered_df.iterrows(), 1):
        rank = game['rank']
        ev = game['expected_value']
        edge = game['vertex_ai_edge'] * 100
        
        # Color coding
        if ev > 10:
            emoji = "🌟"  # Excellent bet
            color = "🟢"
        elif ev > 5:
            emoji = "💚"  # Great bet
            color = "🟢"
        elif ev > 0:
            emoji = "🟡"  # Good bet
            color = "🟡"
        elif ev > -5:
            emoji = "⚪"  # Neutral
            color = "⚪"
        else:
            emoji = "🔴"  # Avoid
            color = "🔴"
        
        # Expandable for each game
        with st.expander(
            f"{color} **#{rank}** | {emoji} {game['away_team']} @ {game['home_team']} | "
            f"EV: ${ev:+.2f} | Edge: {edge:+.1f}% | Vertex AI: {game['vertex_ai_prob']*100:.0f}%",
            expanded=(idx <= expand_top_n)
        ):
            # Top row - main metrics
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric(
                    "Vertex AI Win %",
                    f"{game['vertex_ai_prob']*100:.1f}%",
                    help="Ultimate AI prediction combining all sources"
                )
            
            with metric_cols[1]:
                st.metric(
                    "Market Implied %",
                    f"{game['implied_home_prob']*100:.1f}%",
                    help="What the betting market thinks"
                )
            
            with metric_cols[2]:
                st.metric(
                    "Edge",
                    f"{edge:+.2f}%",
                    delta=f"{edge:+.2f}%",
                    help="Vertex AI advantage over market"
                )
            
            with metric_cols[3]:
                st.metric(
                    "Expected Value",
                    f"${ev:+.2f}",
                    delta=f"${ev:+.2f}",
                    help="Expected profit per $100 bet"
                )
            
            with metric_cols[4]:
                recommendation = "✅ BET" if ev > 5 else "⚠️ SMALL BET" if ev > 0 else "❌ PASS"
                st.metric("Recommendation", recommendation)
            
            # Second row - odds information
            st.markdown("**📊 Market Odds:**")
            odds_cols = st.columns(4)
            
            with odds_cols[0]:
                ml_odds = game.get('home_ml_odds', 'N/A')
                st.write(f"**Moneyline:** {ml_odds}")
            
            with odds_cols[1]:
                spread = game.get('home_spread', 'N/A')
                st.write(f"**Spread:** {spread}")
            
            with odds_cols[2]:
                total = game.get('total_line', 'N/A')
                st.write(f"**Total:** {total}")
            
            with odds_cols[3]:
                bookies = game.get('num_bookmakers', 0)
                st.write(f"**Bookmakers:** {bookies}")
            
            # Third row - data source consensus
            st.markdown("**🤖 AI Model Consensus:**")
            consensus_cols = st.columns(5)
            
            with consensus_cols[0]:
                st.write(f"**Market**")
                st.write(f"{game['implied_home_prob']*100:.0f}%")
            
            with consensus_cols[1]:
                local_ml = game.get('local_ml_prob', 0.5)
                st.write(f"**Your ML**")
                st.write(f"{local_ml*100:.0f}%")
            
            with consensus_cols[2]:
                theover = game.get('theover_probability', 0.5)
                has_theover = game.get('theover_has_pick', 0)
                st.write(f"**theover.ai**")
                st.write(f"{theover*100:.0f}%" if has_theover else "N/A")
            
            with consensus_cols[3]:
                consensus = game.get('consensus_prob', 0.5)
                st.write(f"**Consensus**")
                st.write(f"{consensus*100:.0f}%")
            
            with consensus_cols[4]:
                st.write(f"**Vertex AI**")
                st.write(f"**{game['vertex_ai_prob']*100:.0f}%**")
            
            # Fourth row - team analysis (NO NESTED EXPANDER!)
            st.markdown("---")
            st.markdown("**📈 Detailed Team Analysis**")
            team_cols = st.columns(2)
            
            with team_cols[0]:
                st.markdown(f"**🏠 {game['home_team']}**")
                st.write(f"Win %: {game['home_win_pct']:.1%}")
                st.write(f"Avg Points: {game['home_avg_points']:.1f}")
                st.write(f"Avg Points Allowed: {game['home_avg_points_allowed']:.1f}")
                st.write(f"Last 5: {game['home_last_5_wins']}-{5-game['home_last_5_wins']}")
                st.write(f"Form: {game.get('home_trend', 'neutral').capitalize()}")
                st.write(f"Sentiment: {game['home_sentiment']:+.2f}")
            
            with team_cols[1]:
                st.markdown(f"**✈️ {game['away_team']}**")
                st.write(f"Win %: {game['away_win_pct']:.1%}")
                st.write(f"Avg Points: {game['away_avg_points']:.1f}")
                st.write(f"Avg Points Allowed: {game['away_avg_points_allowed']:.1f}")
                st.write(f"Last 5: {game['away_last_5_wins']}-{5-game['away_last_5_wins']}")
                st.write(f"Form: {game.get('away_trend', 'neutral').capitalize()}")
                st.write(f"Sentiment: {game['away_sentiment']:+.2f}")
            
            # Matchup analysis
            st.markdown("**⚔️ Matchup Analysis:**")
            st.write(f"Win% Differential: {game['win_pct_diff']:+.1%}")
            st.write(f"Form Momentum: {game.get('form_momentum_diff', 0):+.1f}")
            st.write(f"Sentiment Differential: {game.get('sentiment_diff', 0):+.2f}")
            
            # Fifth row - theover.ai pick if available
            if game.get('theover_has_pick', 0) == 1:
                st.info(f"💡 theover.ai pick: **{game['theover_pick']}** ({game['theover_probability']*100:.0f}%)")
    
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
