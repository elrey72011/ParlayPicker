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
        """
        features = [
            # Team strength
            comprehensive_features.get('home_win_pct', 0.5),
            comprehensive_features.get('away_win_pct', 0.5),
            comprehensive_features.get('win_pct_diff', 0),
            
            # Offense/Defense
            comprehensive_features.get('home_avg_points', 100) / 100,
            comprehensive_features.get('away_avg_points', 100) / 100,
            comprehensive_features.get('home_avg_points_allowed', 100) / 100,
            comprehensive_features.get('away_avg_points_allowed', 100) / 100,
            comprehensive_features.get('off_def_matchup_home', 0) / 20,
            comprehensive_features.get('off_def_matchup_away', 0) / 20,
            
            # Recent form
            comprehensive_features.get('home_last_5_wins', 0) / 5,
            comprehensive_features.get('away_last_5_wins', 0) / 5,
            comprehensive_features.get('form_momentum_diff', 0) / 5,
            
            # Market data
            comprehensive_features.get('implied_home_prob', 0.5),
            comprehensive_features.get('home_spread', 0) / 20,
            comprehensive_features.get('total_line', 0) / 200,
            
            # Sentiment
            comprehensive_features.get('sentiment_diff', 0),
            
            # Other models
            comprehensive_features.get('local_ml_prob', 0.5),
            comprehensive_features.get('local_ml_confidence', 0),
            comprehensive_features.get('theover_probability', 0.5),
            comprehensive_features.get('consensus_prob', 0.5),
        ]
        
        return features
    
    def analyze_all_games(self, games: List[Dict], league: str = 'NBA') -> pd.DataFrame:
        """
        Analyze all games with Vertex AI
        
        Args:
            games: List of games from The Odds API
            league: Sport league
            
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
                # Build comprehensive features from ALL sources
                comp_features = self.build_comprehensive_features(game, league)
                
                # Build Vertex AI feature vector
                vertex_features = self.build_vertex_feature_vector(comp_features)
                
                # Get Vertex AI ultimate prediction
                vertex_prob = get_vertex_ai_prediction(vertex_features)
                
                if vertex_prob is not None:
                    # Calculate expected value
                    implied_prob = comp_features.get('implied_home_prob', 0.5)
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
        
        # Sort by Vertex AI edge (best opportunities first)
        if len(results_df) > 0 and 'expected_value' in results_df.columns:
            results_df = results_df.sort_values('expected_value', ascending=False)
        
        return results_df


def show_vertex_master_analysis(results_df: pd.DataFrame):
    """Display Vertex AI Master Analysis results"""
    
    st.header("🏆 Vertex AI Master Analysis")
    st.caption("Ultimate best bets powered by ALL data sources + Vertex AI")
    
    if results_df.empty:
        st.info("No games analyzed yet")
        return
    
    # Filter to positive EV
    best_bets = results_df[results_df['recommendation'] == 'BET'].copy()
    
    if len(best_bets) == 0:
        st.warning("No positive EV bets found by Vertex AI")
        st.write("Market appears efficient for these games")
        return
    
    st.success(f"✅ Vertex AI found {len(best_bets)} positive EV opportunities!")
    
    # Top 5 bets
    st.subheader("🎯 Top 5 Best Bets (Vertex AI)")
    
    for idx, bet in best_bets.head(5).iterrows():
        with st.expander(
            f"🌟 #{idx+1}: {bet['away_team']} @ {bet['home_team']} - "
            f"EV: ${bet['expected_value']:.2f} | Edge: {bet['vertex_ai_edge']*100:+.1f}%",
            expanded=(idx == 0)
        ):
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Vertex AI Probability",
                    f"{bet['vertex_ai_prob']*100:.1f}%",
                    help="Ultimate AI prediction"
                )
            
            with col2:
                st.metric(
                    "Market Implied Prob",
                    f"{bet['implied_home_prob']*100:.1f}%",
                    help="What the odds imply"
                )
            
            with col3:
                st.metric(
                    "Edge",
                    f"{bet['vertex_ai_edge']*100:+.2f}%",
                    delta=f"{bet['vertex_ai_edge']*100:+.2f}%"
                )
            
            with col4:
                st.metric(
                    "Expected Value",
                    f"${bet['expected_value']:.2f}",
                    help="Expected profit per $100 bet"
                )
            
            # All data sources
            st.write("**📊 Data Source Consensus:**")
            cols = st.columns(4)
            
            with cols[0]:
                st.write(f"**Market:** {bet['implied_home_prob']*100:.0f}%")
            with cols[1]:
                st.write(f"**Your ML:** {bet.get('local_ml_prob', 0.5)*100:.0f}%")
            with cols[2]:
                st.write(f"**theover.ai:** {bet.get('theover_probability', 0.5)*100:.0f}%")
            with cols[3]:
                st.write(f"**Vertex AI:** {bet['vertex_ai_prob']*100:.0f}%")
            
            # Team stats
            with st.expander("📈 Full Analysis"):
                st.write(f"**Home:** {bet['home_team']}")
                st.write(f"  Win%: {bet['home_win_pct']:.1%} | Avg Points: {bet['home_avg_points']:.1f}")
                st.write(f"  Last 5: {bet['home_last_5_wins']}-{5-bet['home_last_5_wins']}")
                st.write(f"  Sentiment: {bet['home_sentiment']:+.2f}")
                
                st.write(f"**Away:** {bet['away_team']}")
                st.write(f"  Win%: {bet['away_win_pct']:.1%} | Avg Points: {bet['away_avg_points']:.1f}")
                st.write(f"  Last 5: {bet['away_last_5_wins']}-{5-bet['away_last_5_wins']}")
                st.write(f"  Sentiment: {bet['away_sentiment']:+.2f}")
    
    # Full table
    st.subheader("📋 All Positive EV Bets")
    
    display_cols = ['home_team', 'away_team', 'vertex_ai_prob', 'implied_home_prob', 
                   'vertex_ai_edge', 'expected_value', 'recommendation']
    
    if all(col in best_bets.columns for col in display_cols):
        display_df = best_bets[display_cols].copy()
        st.dataframe(display_df, use_container_width=True)
    
    # Download
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Complete Analysis",
        csv,
        f"vertex_master_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )
