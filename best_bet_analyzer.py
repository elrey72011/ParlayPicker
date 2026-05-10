"""
Best Bet Analyzer - Combines CSV odds with API data for Vertex AI predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BestBetAnalyzer:
    """
    Analyzes betting opportunities by combining:
    - CSV odds data
    - The Odds API
    - SportsData.io stats
    - API-Sports data
    - NewsAPI sentiment
    """
    
    def __init__(
        self,
        odds_api_client=None,
        sportsdata_client=None,
        apisports_client=None,
        sentiment_analyzer=None,
        ml_predictor=None
    ):
        self.odds_api = odds_api_client
        self.sportsdata = sportsdata_client
        self.apisports = apisports_client
        self.sentiment = sentiment_analyzer
        self.ml_predictor = ml_predictor
        
    def parse_odds_csv(self, csv_file) -> pd.DataFrame:
        """
        Parse uploaded odds CSV file
        
        Expected columns: game_id, date, home_team, away_team, 
                         home_odds, away_odds, total, over_odds, under_odds
        """
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} games from CSV")
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return pd.DataFrame()
    
    def enrich_game_with_apis(self, game: Dict) -> Dict:
        """
        Enrich a single game with API data
        
        Returns comprehensive feature set:
        - Team stats (win %, avg points, etc.)
        - Recent form (last 5 games)
        - Head-to-head history
        - Injuries
        - News sentiment
        - Weather (for outdoor sports)
        - Sharp money indicators
        """
        enriched = game.copy()
        home_team = game['home_team']
        away_team = game['away_team']
        
        # 1. Get team stats from SportsData.io
        # Always call so missing-client warnings are surfaced; None values are
        # handled in build_features_for_prediction rather than silently using 0.5.
        home_stats = self._get_team_stats(home_team, 'home')
        away_stats = self._get_team_stats(away_team, 'away')
        enriched.update({k: v for k, v in home_stats.items() if v is not None})
        enriched.update({k: v for k, v in away_stats.items() if v is not None})

        # 2. Get recent form from API-Sports
        home_form = self._get_recent_form(home_team)
        away_form = self._get_recent_form(away_team)
        if home_form.get('wins') is not None:
            enriched['home_last_5_wins'] = home_form['wins']
        if away_form.get('wins') is not None:
            enriched['away_last_5_wins'] = away_form['wins']

        # 3. Get news sentiment (neutral 0.0 when unavailable — correct prior)
        enriched['home_sentiment'] = self._get_team_sentiment(home_team)
        enriched['away_sentiment'] = self._get_team_sentiment(away_team)
        
        # 4. Calculate derived features
        enriched['odds_diff'] = abs(game.get('home_odds', 0) - game.get('away_odds', 0))
        enriched['total_line'] = game.get('total', 0)
        
        return enriched
    
    def _get_team_stats(self, team_name: str, prefix: str) -> Dict:
        """Get team statistics from SportsData client.

        Returns None values when no real client is wired up so that callers can
        detect missing data rather than treating identical 0.5 placeholders as
        real signal.  Wire self.sportsdata to a live SportsDataIO client to
        populate these fields.
        """
        if self.sportsdata is None:
            logger.warning(
                f"No SportsData client configured — returning None stats for {team_name}. "
                "Team stats will be omitted from the feature vector."
            )
            return {
                f'{prefix}_win_pct': None,
                f'{prefix}_ppg': None,
                f'{prefix}_oppg': None,
                f'{prefix}_home_win_pct': None,
            }
        # TODO: call self.sportsdata and return real values
        return {
            f'{prefix}_win_pct': None,
            f'{prefix}_ppg': None,
            f'{prefix}_oppg': None,
            f'{prefix}_home_win_pct': None,
        }

    def _get_recent_form(self, team_name: str) -> Dict:
        """Get recent form (last 5 games) from API-Sports client.

        Returns None values when no real client is wired up.
        """
        if self.apisports is None:
            logger.warning(
                f"No API-Sports client configured — returning None form for {team_name}."
            )
            return {'wins': None, 'losses': None}
        # TODO: call self.apisports and return real values
        return {'wins': None, 'losses': None}

    def _get_team_sentiment(self, team_name: str) -> float:
        """Get news sentiment score (-1 to 1).

        Returns 0.0 (neutral) when no sentiment client is configured, which is
        the correct uninformative prior rather than injecting a fake signal.
        """
        if self.sentiment is None:
            return 0.0
        # TODO: call self.sentiment and return real value
        return 0.0
    
    def build_features_for_prediction(self, enriched_game: Dict) -> List[float]:
        """
        Build feature vector for Vertex AI prediction.

        Uses None-aware helper so that stats not populated by a real API client
        fall back to a neutral prior (0.5 for rates, 0.0 for deltas) rather than
        silently using a hardcoded placeholder that looks like real data.
        """
        def _get(key, neutral):
            val = enriched_game.get(key)
            return neutral if val is None else float(val)

        last5_home = enriched_game.get('home_last_5_wins')
        last5_away = enriched_game.get('away_last_5_wins')

        features = [
            _get('home_win_pct', 0.5),
            _get('away_win_pct', 0.5),
            _get('home_ppg', 0.5),
            _get('away_ppg', 0.5),
            _get('home_oppg', 0.5),
            _get('away_oppg', 0.5),
            (float(last5_home) / 5.0) if last5_home is not None else 0.5,
            (float(last5_away) / 5.0) if last5_away is not None else 0.5,
            _get('home_sentiment', 0.0),
            _get('away_sentiment', 0.0),
            _get('odds_diff', 0.0) / 1000,
            _get('total_line', 0.0) / 200,
        ]

        return features
    
    def analyze_all_games(self, csv_file) -> pd.DataFrame:
        """
        Analyze all games from CSV
        
        Returns DataFrame with:
        - Original odds data
        - Enriched API data
        - Vertex AI predictions
        - Expected value calculations
        - Betting recommendations
        """
        # Parse CSV
        games_df = self.parse_odds_csv(csv_file)
        
        if games_df.empty:
            return pd.DataFrame()
        
        results = []
        
        for idx, game in games_df.iterrows():
            try:
                # Enrich with API data
                enriched = self.enrich_game_with_apis(game.to_dict())
                
                # Build features
                features = self.build_features_for_prediction(enriched)
                
                # Get Vertex AI prediction
                from ml_predictions import get_vertex_ai_prediction, is_vertex_ai_enabled
                
                if is_vertex_ai_enabled():
                    ai_win_prob = get_vertex_ai_prediction(features)
                else:
                    ai_win_prob = None
                
                # Calculate expected value
                if ai_win_prob is not None:
                    home_odds = enriched.get('home_odds', 0)
                    
                    # Convert American odds to probability
                    if home_odds > 0:
                        market_prob = 100 / (home_odds + 100)
                    else:
                        market_prob = abs(home_odds) / (abs(home_odds) + 100)
                    
                    # Calculate edge
                    edge = ai_win_prob - market_prob
                    
                    # Calculate expected value (EV)
                    if home_odds > 0:
                        ev = (ai_win_prob * home_odds) - ((1 - ai_win_prob) * 100)
                    else:
                        ev = (ai_win_prob * 100) - ((1 - ai_win_prob) * abs(home_odds))
                    
                    enriched['ai_win_prob'] = ai_win_prob
                    enriched['market_prob'] = market_prob
                    enriched['edge'] = edge
                    enriched['expected_value'] = ev
                    enriched['bet_recommendation'] = 'BET' if edge > 0.05 else 'PASS'
                    enriched['confidence'] = abs(edge)
                
                results.append(enriched)
                
            except Exception as e:
                logger.error(f"Error analyzing game {idx}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Sort by expected value (best bets first)
        if 'expected_value' in results_df.columns:
            results_df = results_df.sort_values('expected_value', ascending=False)
        
        return results_df


def show_best_bets_analysis(csv_file, analyzer: BestBetAnalyzer):
    """
    Display best bets analysis in Streamlit
    """
    st.header("🎯 Best Bet Finder - AI Analysis")
    st.caption("Combines your CSV odds with live API data and Vertex AI predictions")
    
    with st.spinner("Analyzing all games... This may take a minute..."):
        results_df = analyzer.analyze_all_games(csv_file)
    
    if results_df.empty:
        st.error("No games to analyze")
        return
    
    st.success(f"✅ Analyzed {len(results_df)} games")
    
    # Show top bets
    st.subheader("🏆 Top Betting Opportunities")
    
    top_bets = results_df.head(10)
    
    for idx, bet in top_bets.iterrows():
        with st.expander(
            f"{'🟢' if bet.get('bet_recommendation') == 'BET' else '🟡'} "
            f"{bet['away_team']} @ {bet['home_team']} - "
            f"EV: {bet.get('expected_value', 0):.2f}%"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "AI Win Probability",
                    f"{bet.get('ai_win_prob', 0) * 100:.1f}%"
                )
                st.metric(
                    "Market Probability",
                    f"{bet.get('market_prob', 0) * 100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Edge",
                    f"{bet.get('edge', 0) * 100:.2f}%",
                    delta=f"{bet.get('edge', 0) * 100:.2f}%"
                )
                st.metric(
                    "Expected Value",
                    f"{bet.get('expected_value', 0):.2f}%"
                )
            
            with col3:
                st.metric("Home Odds", bet.get('home_odds', 'N/A'))
                st.metric("Recommendation", bet.get('bet_recommendation', 'PASS'))
            
            # Show key stats
            st.write("**Key Stats:**")
            st.write(f"Home Win%: {bet.get('home_win_pct', 0):.1%}")
            st.write(f"Away Win%: {bet.get('away_win_pct', 0):.1%}")
            st.write(f"Home Sentiment: {bet.get('home_sentiment', 0):.2f}")
            st.write(f"Away Sentiment: {bet.get('away_sentiment', 0):.2f}")
    
    # Show full table
    st.subheader("📊 All Games")
    
    display_cols = [
        'home_team', 'away_team', 'home_odds', 'away_odds',
        'ai_win_prob', 'edge', 'expected_value', 'bet_recommendation'
    ]
    
    available_cols = [col for col in display_cols if col in results_df.columns]
    st.dataframe(results_df[available_cols])
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Analysis",
        csv,
        "best_bets_analysis.csv",
        "text/csv"
    )

# Ensured avg_points is ppg and def_rating is oppg
