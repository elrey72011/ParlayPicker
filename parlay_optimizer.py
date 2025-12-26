"""
Parlay Optimizer
Uses ML model predictions to construct optimal parlays
Maximizes expected value while managing correlation and risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParlayOptimizer:
    """Optimize parlay selection based on ML predictions and betting strategy"""
    
    def __init__(self, model_dir: str, min_edge: float = 0.01):
        """
        Initialize optimizer
        
        Args:
            model_dir: Directory containing trained models
            min_edge: Minimum edge required to consider a bet
        """
        self.model_dir = Path(model_dir)
        self.min_edge = min_edge
        self.models = {}
        self.scalers = {}
        
    def load_models(self, sport: str):
        """
        Load trained models for a sport
        
        Args:
            sport: Sport name (e.g., 'NFL', 'NBA')
        """
        logger.info(f"Loading models for {sport}")
        
        model_types = ['moneyline', 'spread', 'totals']
        
        for model_type in model_types:
            model_name = f'{sport}_{model_type}'
            model_file = self.model_dir / f'{model_name}.pkl'
            scaler_file = self.model_dir / f'{model_name}_scaler.pkl'
            
            if model_file.exists() and scaler_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                with open(scaler_file, 'rb') as f:
                    self.scalers[model_type] = pickle.load(f)
                logger.info(f"Loaded {model_type} model")
            else:
                logger.warning(f"Model files not found for {model_type}")
    
    def calculate_implied_probability(self, odds: float) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            odds: American odds (e.g., -110, +150)
            
        Returns:
            Implied probability (0-1)
        """
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def american_to_decimal(self, odds: float) -> float:
        """
        Convert American odds to decimal odds
        
        Args:
            odds: American odds
            
        Returns:
            Decimal odds
        """
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    def calculate_expected_value(
        self,
        model_prob: float,
        odds: float,
        stake: float = 100
    ) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            model_prob: Model's predicted probability
            odds: Betting odds (American)
            stake: Bet amount
            
        Returns:
            Expected value
        """
        decimal_odds = self.american_to_decimal(odds)
        potential_profit = stake * (decimal_odds - 1)
        expected_return = (model_prob * potential_profit) - ((1 - model_prob) * stake)
        
        return expected_return / stake  # Return as percentage
    
    def predict_game(
        self,
        game_features: pd.DataFrame,
        bet_type: str
    ) -> float:
        """
        Get model prediction for a game
        
        Args:
            game_features: Feature row for the game
            bet_type: Type of bet ('moneyline', 'spread', 'totals')
            
        Returns:
            Predicted probability
        """
        if bet_type not in self.models:
            raise ValueError(f"Model not loaded for {bet_type}")
        
        model = self.models[bet_type]
        scaler = self.scalers[bet_type]
        
        # Scale features
        X_scaled = scaler.transform(game_features)
        
        # Get prediction
        prob = model.predict_proba(X_scaled)[0, 1]
        
        return prob
    
    def analyze_single_bets(
        self,
        games_df: pd.DataFrame,
        odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze all individual bets for expected value
        
        Args:
            games_df: DataFrame with game features
            odds_df: DataFrame with betting odds from theover.ai
            
        Returns:
            DataFrame with bet analysis
        """
        logger.info("Analyzing single bets for edge")
        
        bets = []
        
        for idx, game in games_df.iterrows():
            game_id = game.get('GameID')
            
            # Get corresponding odds
            game_odds = odds_df[odds_df['GameID'] == game_id]
            
            if game_odds.empty:
                continue
            
            # Prepare features (remove target columns if present)
            feature_cols = [col for col in game.index 
                          if not col.startswith('Target_') 
                          and col not in ['GameID', 'DateTime', 'HomeTeam', 'AwayTeam']]
            game_features = game[feature_cols].to_frame().T
            
            # Analyze moneyline
            if 'moneyline' in self.models:
                try:
                    ml_prob = self.predict_game(game_features, 'moneyline')
                    home_odds = game_odds['HomeMoneyLine'].values[0]
                    away_odds = game_odds['AwayMoneyLine'].values[0]
                    
                    home_ev = self.calculate_expected_value(ml_prob, home_odds)
                    away_ev = self.calculate_expected_value(1 - ml_prob, away_odds)
                    
                    if home_ev > self.min_edge:
                        bets.append({
                            'GameID': game_id,
                            'HomeTeam': game['HomeTeam'],
                            'AwayTeam': game['AwayTeam'],
                            'BetType': 'Moneyline',
                            'Selection': game['HomeTeam'],
                            'ModelProb': ml_prob,
                            'Odds': home_odds,
                            'ImpliedProb': self.calculate_implied_probability(home_odds),
                            'Edge': ml_prob - self.calculate_implied_probability(home_odds),
                            'ExpectedValue': home_ev,
                            'Confidence': 'High' if home_ev > 0.1 else 'Medium'
                        })
                    
                    if away_ev > self.min_edge:
                        bets.append({
                            'GameID': game_id,
                            'HomeTeam': game['HomeTeam'],
                            'AwayTeam': game['AwayTeam'],
                            'BetType': 'Moneyline',
                            'Selection': game['AwayTeam'],
                            'ModelProb': 1 - ml_prob,
                            'Odds': away_odds,
                            'ImpliedProb': self.calculate_implied_probability(away_odds),
                            'Edge': (1 - ml_prob) - self.calculate_implied_probability(away_odds),
                            'ExpectedValue': away_ev,
                            'Confidence': 'High' if away_ev > 0.1 else 'Medium'
                        })
                except Exception as e:
                    logger.error(f"Error analyzing moneyline for game {game_id}: {e}")
            
            # Analyze spread
            if 'spread' in self.models and 'PointSpread' in game_odds.columns:
                try:
                    spread_prob = self.predict_game(game_features, 'spread')
                    spread_odds = game_odds['HomeSpreadOdds'].values[0]
                    
                    spread_ev = self.calculate_expected_value(spread_prob, spread_odds)
                    
                    if spread_ev > self.min_edge:
                        bets.append({
                            'GameID': game_id,
                            'HomeTeam': game['HomeTeam'],
                            'AwayTeam': game['AwayTeam'],
                            'BetType': 'Spread',
                            'Selection': f"{game['HomeTeam']} {game_odds['PointSpread'].values[0]}",
                            'ModelProb': spread_prob,
                            'Odds': spread_odds,
                            'ImpliedProb': self.calculate_implied_probability(spread_odds),
                            'Edge': spread_prob - self.calculate_implied_probability(spread_odds),
                            'ExpectedValue': spread_ev,
                            'Confidence': 'High' if spread_ev > 0.1 else 'Medium'
                        })
                except Exception as e:
                    logger.error(f"Error analyzing spread for game {game_id}: {e}")
            
            # Analyze totals
            if 'totals' in self.models and 'OverUnder' in game_odds.columns:
                try:
                    over_prob = self.predict_game(game_features, 'totals')
                    over_odds = game_odds['OverOdds'].values[0]
                    under_odds = game_odds['UnderOdds'].values[0]
                    
                    over_ev = self.calculate_expected_value(over_prob, over_odds)
                    under_ev = self.calculate_expected_value(1 - over_prob, under_odds)
                    
                    if over_ev > self.min_edge:
                        bets.append({
                            'GameID': game_id,
                            'HomeTeam': game['HomeTeam'],
                            'AwayTeam': game['AwayTeam'],
                            'BetType': 'Total',
                            'Selection': f"Over {game_odds['OverUnder'].values[0]}",
                            'ModelProb': over_prob,
                            'Odds': over_odds,
                            'ImpliedProb': self.calculate_implied_probability(over_odds),
                            'Edge': over_prob - self.calculate_implied_probability(over_odds),
                            'ExpectedValue': over_ev,
                            'Confidence': 'High' if over_ev > 0.1 else 'Medium'
                        })
                    
                    if under_ev > self.min_edge:
                        bets.append({
                            'GameID': game_id,
                            'HomeTeam': game['HomeTeam'],
                            'AwayTeam': game['AwayTeam'],
                            'BetType': 'Total',
                            'Selection': f"Under {game_odds['OverUnder'].values[0]}",
                            'ModelProb': 1 - over_prob,
                            'Odds': under_odds,
                            'ImpliedProb': self.calculate_implied_probability(under_odds),
                            'Edge': (1 - over_prob) - self.calculate_implied_probability(under_odds),
                            'ExpectedValue': under_ev,
                            'Confidence': 'High' if under_ev > 0.1 else 'Medium'
                        })
                except Exception as e:
                    logger.error(f"Error analyzing totals for game {game_id}: {e}")
        
        bets_df = pd.DataFrame(bets)
        
        if not bets_df.empty:
            bets_df = bets_df.sort_values('ExpectedValue', ascending=False)
            logger.info(f"Found {len(bets_df)} positive EV bets")
        
        return bets_df
    
    def calculate_parlay_probability(self, legs: List[Dict]) -> float:
        """
        Calculate parlay win probability (assuming independence)
        
        Args:
            legs: List of bet dictionaries
            
        Returns:
            Combined probability
        """
        prob = 1.0
        for leg in legs:
            prob *= leg['ModelProb']
        return prob
    
    def calculate_parlay_odds(self, legs: List[Dict]) -> float:
        """
        Calculate parlay payout odds
        
        Args:
            legs: List of bet dictionaries
            
        Returns:
            Combined decimal odds
        """
        odds = 1.0
        for leg in legs:
            odds *= self.american_to_decimal(leg['Odds'])
        return odds
    
    def check_correlation(self, bet1: Dict, bet2: Dict) -> str:
        """
        Check if two bets are correlated
        
        Args:
            bet1, bet2: Bet dictionaries
            
        Returns:
            Correlation level: 'none', 'low', 'high'
        """
        # Same game = highly correlated
        if bet1['GameID'] == bet2['GameID']:
            # Same game different bet types = medium correlation
            if bet1['BetType'] != bet2['BetType']:
                return 'high'
            else:
                return 'high'  # Same game same bet type
        
        # Different games = check for divisional matchups, etc.
        # This is simplified - you'd want sport-specific logic
        return 'none'
    
    def generate_parlays(
        self,
        bets_df: pd.DataFrame,
        parlay_size: int = 3,
        max_parlays: int = 10,
        allow_correlation: bool = False
    ) -> List[Dict]:
        """
        Generate optimal parlays
        
        Args:
            bets_df: DataFrame with analyzed bets
            parlay_size: Number of legs per parlay
            max_parlays: Maximum number of parlays to return
            allow_correlation: Whether to allow correlated bets
            
        Returns:
            List of optimal parlays
        """
        logger.info(f"Generating {parlay_size}-leg parlays")
        
        # Filter to high-confidence bets
        high_value_bets = bets_df[bets_df['ExpectedValue'] > self.min_edge].to_dict('records')
        
        if len(high_value_bets) < parlay_size:
            logger.warning(f"Not enough high-value bets ({len(high_value_bets)}) for {parlay_size}-leg parlays")
            return []
        
        parlays = []
        
        # Generate all possible combinations
        for combo in combinations(high_value_bets, parlay_size):
            legs = list(combo)
            
            # Check for correlation if not allowed
            if not allow_correlation:
                has_correlation = False
                for i in range(len(legs)):
                    for j in range(i + 1, len(legs)):
                        if self.check_correlation(legs[i], legs[j]) in ['high']:
                            has_correlation = True
                            break
                    if has_correlation:
                        break
                
                if has_correlation:
                    continue
            
            # Calculate parlay metrics
            parlay_prob = self.calculate_parlay_probability(legs)
            parlay_odds = self.calculate_parlay_odds(legs)
            
            # Expected value
            ev = (parlay_prob * (parlay_odds - 1)) - (1 - parlay_prob)
            
            # Only include positive EV parlays
            if ev > 0:
                parlays.append({
                    'legs': legs,
                    'num_legs': len(legs),
                    'win_probability': parlay_prob,
                    'decimal_odds': parlay_odds,
                    'american_odds': self.decimal_to_american(parlay_odds),
                    'expected_value': ev,
                    'kelly_fraction': self.calculate_kelly(parlay_prob, parlay_odds)
                })
        
        # Sort by expected value
        parlays.sort(key=lambda x: x['expected_value'], reverse=True)
        
        logger.info(f"Generated {len(parlays)} positive EV parlays")
        
        return parlays[:max_parlays]
    
    def decimal_to_american(self, decimal_odds: float) -> float:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)
    
    def calculate_kelly(self, win_prob: float, decimal_odds: float) -> float:
        """
        Calculate Kelly Criterion bet size
        
        Args:
            win_prob: Probability of winning
            decimal_odds: Decimal odds
            
        Returns:
            Fraction of bankroll to bet (0-1)
        """
        q = 1 - win_prob  # Probability of losing
        b = decimal_odds - 1  # Net odds
        
        kelly = (win_prob * b - q) / b
        
        # Use fractional Kelly (1/4 Kelly) for safety
        return max(0, min(kelly * 0.25, 0.1))  # Cap at 10% of bankroll
    
    def generate_betting_card(
        self,
        single_bets: pd.DataFrame,
        parlays: List[Dict],
        output_file: str = 'betting_card.csv'
    ):
        """
        Generate final betting recommendations
        
        Args:
            single_bets: Top single bets
            parlays: Top parlays
            output_file: Output CSV filename
        """
        logger.info("Generating betting card")
        
        # Top single bets
        top_singles = single_bets.head(10)
        
        # Format parlays for display
        parlay_rows = []
        for i, parlay in enumerate(parlays, 1):
            leg_descriptions = [
                f"{leg['Selection']} ({leg['Odds']:+.0f})"
                for leg in parlay['legs']
            ]
            
            parlay_rows.append({
                'Parlay': f"Parlay {i}",
                'Legs': ' | '.join(leg_descriptions),
                'Probability': f"{parlay['win_probability']:.2%}",
                'Odds': f"{parlay['american_odds']:+.0f}",
                'ExpectedValue': f"{parlay['expected_value']:.2%}",
                'Kelly': f"{parlay['kelly_fraction']:.2%}"
            })
        
        parlay_df = pd.DataFrame(parlay_rows)
        
        # Save to CSV
        with open(output_file, 'w') as f:
            f.write("=== TOP SINGLE BETS ===\n")
            top_singles.to_csv(f, index=False)
            f.write("\n\n=== TOP PARLAYS ===\n")
            parlay_df.to_csv(f, index=False)
        
        logger.info(f"Betting card saved to {output_file}")
        
        return top_singles, parlay_df


def main():
    """Example usage"""
    
    # Initialize optimizer
    optimizer = ParlayOptimizer(
        model_dir='./models/nfl',
        min_edge=0.01  # 1% minimum edge
    )
    
    # Load models
    optimizer.load_models('NFL')
    
    # Load today's games with features
    games_df = pd.read_csv('./data/todays_games_features.csv')
    
    # Load odds from theover.ai
    odds_df = pd.read_csv('./data/theover_odds.csv')
    
    # Analyze single bets
    single_bets = optimizer.analyze_single_bets(games_df, odds_df)
    
    # Generate parlays
    parlays_3leg = optimizer.generate_parlays(single_bets, parlay_size=3, max_parlays=10)
    parlays_4leg = optimizer.generate_parlays(single_bets, parlay_size=4, max_parlays=5)
    
    # Generate betting card
    optimizer.generate_betting_card(
        single_bets,
        parlays_3leg + parlays_4leg,
        'todays_picks.csv'
    )
    
    print(f"\nFound {len(single_bets)} +EV bets")
    print(f"Generated {len(parlays_3leg)} 3-leg parlays")
    print(f"Generated {len(parlays_4leg)} 4-leg parlays")


if __name__ == "__main__":
    main()
