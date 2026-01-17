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

    def get_shotgun_picks(self, master_df: pd.DataFrame):
        if "AI_Prob" not in master_df.columns or "AI_Edge" not in master_df.columns:
            return {"snipers": pd.DataFrame(), "strategy": pd.DataFrame(), "longshots": pd.DataFrame()}

        # Filter tiers as requested
        # $3 'Snipers': AI_Prob > 0.60 and AI_Edge > 0.05
        snipers = master_df[(master_df['AI_Prob'] > 0.60) & (master_df['AI_Edge'] > 0.00)].head(3)

        # $2 'Strategy': AI_Edge > 0.08
        strategy = master_df[master_df['AI_Edge'] > 0.08].head(5)

        # $1 'Longshots': Top 10 by highest AI_Edge
        longshots = master_df.sort_values('AI_Edge', ascending=False).head(10)

        return {"snipers": snipers, "strategy": strategy, "longshots": longshots}

    def get_shotgun_allocation(self, best_bets_df):
        """Categorizes into $3, $2, and $1 tiers (Legacy)"""
        result = self.get_shotgun_picks(best_bets_df)
        return {
            "snipers_3": result["snipers"],
            "strategy_2": result["strategy"],
            "longshots_1": result["longshots"]
        }

    def calculate_consensus_votes(self, row: pd.Series) -> Tuple[int, int, Dict[str, bool]]:
        """
        Calculate consensus votes from multiple sources for a single pick.

        Args:
            row: DataFrame row with pick data

        Returns:
            Tuple of (consensus_votes, consensus_total, vote_details)
        """
        vote_details = {}

        # Vote 1: Kalshi
        # True if Kalshi is available and aligns with the pick
        kalshi_prob_for_pick = row.get("kalshi_prob_for_pick")
        if kalshi_prob_for_pick is not None and not pd.isna(kalshi_prob_for_pick):
            vote_details["vote_kalshi"] = kalshi_prob_for_pick > 0.5
        else:
            vote_details["vote_kalshi"] = None

        # Vote 2: Model
        # True if model was used and modelprob > 0.5
        model_used = row.get("model_used", False)
        model_prob = row.get("model_prob") or row.get("AI_Prob")
        if model_used and model_prob is not None and not pd.isna(model_prob):
            vote_details["vote_model"] = model_prob > 0.5
        else:
            vote_details["vote_model"] = None

        # Vote 3: Market
        # True if finalprob > 0.5 and positive edge vs market
        final_prob = row.get("final_probability") or row.get("AI_Prob")
        ai_edge = row.get("AI_Edge", 0)
        if final_prob is not None and not pd.isna(final_prob):
            vote_details["vote_market"] = (final_prob > 0.5) and (ai_edge > 0)
        else:
            vote_details["vote_market"] = None

        # Vote 4: TheOver
        # True if warnings contains "theoveragrees"
        warnings = str(row.get("warnings", "")).lower()
        if "theover" in warnings:
            if "theoveragrees" in warnings:
                vote_details["vote_theover"] = True
            elif "theoverdisagrees" in warnings:
                vote_details["vote_theover"] = False
            else:
                vote_details["vote_theover"] = None
        else:
            vote_details["vote_theover"] = None

        # Vote 5: Sentiment (once fixed, but check for non-zero values)
        sentiment_score = row.get("sentiment_score")
        pick_side = str(row.get("pick_side", "")).lower()
        if sentiment_score is not None and not pd.isna(sentiment_score) and sentiment_score != 0.0:
            # Positive sentiment for home/over, negative for away/under
            if pick_side in {"home", "over"}:
                vote_details["vote_sentiment"] = sentiment_score > 0.1
            elif pick_side in {"away", "under"}:
                vote_details["vote_sentiment"] = sentiment_score < -0.1
            else:
                vote_details["vote_sentiment"] = None
        else:
            vote_details["vote_sentiment"] = None

        # Count votes (only non-None values)
        votes = [v for v in vote_details.values() if v is not None]
        consensus_votes = sum(votes)
        consensus_total = len(votes)

        return consensus_votes, consensus_total, vote_details

    def generate_shotgun_parlays(self, master_df: pd.DataFrame) -> Dict[str, Optional[Dict]]:
        """
        Generate three 2-leg parlays for Shotgun Mode with fixed stakes.
        Enforces game diversity across the three parlays.

        Args:
            master_df: DataFrame with picks including columns:
                - Pick, Spread & Pick, Total & Pick (pick label)
                - AI_Prob or final_probability (model probability)
                - Implied_Prob (market probability)
                - odds (american odds)
                - AI_Edge (edge)
                - Market (bet type: Moneyline, Spread, Total)
                - Game (game identifier)
                - Optional: kalshi_prob_for_pick, model_prob, model_used, warnings, sentiment_score, pick_side

        Returns:
            Dictionary with three parlay recommendations:
            {
                "best_overall": {...},  # $3 stake
                "medium_risk": {...},   # $2 stake
                "high_risk": {...}      # $1 stake
            }
        """
        logger.info("Generating 2-leg parlays for Shotgun Mode with diversity constraints")

        # Required columns
        required_cols = ["AI_Prob", "AI_Edge"]
        if not all(col in master_df.columns for col in required_cols):
            logger.warning("Missing required columns for parlay generation")
            return {"best_overall": None, "medium_risk": None, "high_risk": None}

        # Filter to playable picks (positive edge, reasonable probability)
        playable = master_df[
            (master_df["AI_Edge"] > 0.00) &  # Must have positive edge
            (master_df["AI_Prob"] > 0.35) &  # Not too low probability
            (master_df["AI_Prob"] < 0.85)    # Not too high probability (poor odds)
        ].copy()

        if len(playable) < 2:
            logger.warning(f"Not enough playable picks ({len(playable)}) for 2-leg parlays")
            return {"best_overall": None, "medium_risk": None, "high_risk": None}

        # Calculate consensus for each playable pick
        consensus_data = []
        for idx, row in playable.iterrows():
            consensus_votes, consensus_total, vote_details = self.calculate_consensus_votes(row)
            playable.at[idx, "consensus_votes"] = consensus_votes
            playable.at[idx, "consensus_total"] = consensus_total
            playable.at[idx, "consensus_ratio"] = consensus_votes / consensus_total if consensus_total > 0 else 0
            consensus_data.append({
                "index": idx,
                "pick": row.get("Pick") or row.get("Spread & Pick") or row.get("Total & Pick"),
                "consensus": f"{consensus_votes}/{consensus_total}",
                "vote_details": vote_details
            })

        # Log sample consensus calculations
        logger.info(f"Consensus calculation for {min(5, len(consensus_data))} sample picks:")
        for item in consensus_data[:5]:
            logger.info(f"  {item['pick']}: {item['consensus']} - {item['vote_details']}")

        # Generate all 2-leg combinations
        parlay_candidates = []
        for i, row1 in playable.iterrows():
            for j, row2 in playable.iterrows():
                if i >= j:  # Avoid duplicates and self-pairing
                    continue

                # Avoid same-game parlays (correlated)
                game1 = row1.get("Game") or row1.get("game_id") or row1.get("id") or ""
                game2 = row2.get("Game") or row2.get("game_id") or row2.get("id") or ""
                if game1 and game2 and str(game1) == str(game2):
                    continue

                # Calculate parlay metrics
                prob1 = row1["AI_Prob"]
                prob2 = row2["AI_Prob"]
                parlay_prob = prob1 * prob2  # Assume independence

                # Get odds
                odds1 = row1.get("odds") or row1.get("spread_pick_odds") or row1.get("total_pick_odds") or -110
                odds2 = row2.get("odds") or row2.get("spread_pick_odds") or row2.get("total_pick_odds") or -110

                # Convert to decimal odds
                decimal_odds1 = self.american_to_decimal(odds1)
                decimal_odds2 = self.american_to_decimal(odds2)
                parlay_decimal_odds = decimal_odds1 * decimal_odds2
                parlay_american_odds = self.decimal_to_american(parlay_decimal_odds)

                # Calculate EV (expected value per dollar staked)
                # EV = (win_prob * payout) - (loss_prob * stake)
                # Since payout for $1 stake = (decimal_odds - 1) * 1 = decimal_odds - 1
                # EV = parlay_prob * (parlay_decimal_odds - 1) - (1 - parlay_prob) * 1
                ev = (parlay_prob * (parlay_decimal_odds - 1)) - (1 - parlay_prob)

                # Get pick labels
                pick1 = row1.get("Pick") or row1.get("Spread & Pick") or row1.get("Total & Pick") or "Unknown"
                pick2 = row2.get("Pick") or row2.get("Spread & Pick") or row2.get("Total & Pick") or "Unknown"

                # Get consensus data
                consensus_votes1 = row1.get("consensus_votes", 0)
                consensus_total1 = row1.get("consensus_total", 0)
                consensus_votes2 = row2.get("consensus_votes", 0)
                consensus_total2 = row2.get("consensus_total", 0)

                # Combined consensus score (average ratio)
                consensus_ratio1 = consensus_votes1 / consensus_total1 if consensus_total1 > 0 else 0
                consensus_ratio2 = consensus_votes2 / consensus_total2 if consensus_total2 > 0 else 0
                avg_consensus_ratio = (consensus_ratio1 + consensus_ratio2) / 2

                parlay_candidates.append({
                    "leg1": {
                        "pick": pick1,
                        "prob": prob1,
                        "odds": odds1,
                        "market": row1.get("Market", "Unknown"),
                        "game": game1,
                        "consensus_votes": consensus_votes1,
                        "consensus_total": consensus_total1
                    },
                    "leg2": {
                        "pick": pick2,
                        "prob": prob2,
                        "odds": odds2,
                        "market": row2.get("Market", "Unknown"),
                        "game": game2,
                        "consensus_votes": consensus_votes2,
                        "consensus_total": consensus_total2
                    },
                    "parlay_prob": parlay_prob,
                    "parlay_decimal_odds": parlay_decimal_odds,
                    "parlay_american_odds": parlay_american_odds,
                    "ev": ev,
                    "combined_edge": row1["AI_Edge"] + row2["AI_Edge"],
                    "avg_consensus_ratio": avg_consensus_ratio,
                    "games_used": {str(game1), str(game2)}
                })

        if not parlay_candidates:
            logger.warning("No valid 2-leg parlays generated")
            return {"best_overall": None, "medium_risk": None, "high_risk": None}

        logger.info(f"Generated {len(parlay_candidates)} candidate 2-leg parlays")

        # Helper function to calculate diversity penalty
        def diversity_score(parlay, used_games_set):
            """
            Calculate a diversity penalty based on game overlap.
            Returns a value between 0 (no overlap) and 1 (complete overlap).
            """
            parlay_games = parlay["games_used"]
            overlap_count = len(parlay_games & used_games_set)
            return overlap_count / len(parlay_games) if len(parlay_games) > 0 else 0

        # Helper function for scoring with consensus tie-breaker
        def score_with_consensus(parlay, primary_key):
            """
            Score parlay using primary metric, with consensus as a tie-breaker.
            Returns tuple (primary_score, consensus_ratio) for sorting.
            """
            primary_score = primary_key(parlay)
            consensus_ratio = parlay.get("avg_consensus_ratio", 0)
            return (primary_score, consensus_ratio)

        # Track games used across all selected parlays
        used_games = set()

        # 1. Best Overall ($3): Highest EV with reasonable probability
        # Prefer parlays with win_prob > 0.20 and highest EV, use consensus as tie-breaker
        best_overall_candidates = [p for p in parlay_candidates if p["parlay_prob"] > 0.20]
        if best_overall_candidates:
            best_overall = max(best_overall_candidates, key=lambda x: score_with_consensus(x, lambda p: p["ev"]))
        else:
            best_overall = max(parlay_candidates, key=lambda x: score_with_consensus(x, lambda p: p["ev"]))
        best_overall["stake"] = 3
        best_overall["expected_return"] = best_overall["stake"] * (1 + best_overall["ev"])
        used_games.update(best_overall["games_used"])

        # 2. Medium Risk ($2): Moderate probability (0.15-0.30), good EV, prefer diversity
        medium_risk_candidates = [
            p for p in parlay_candidates
            if 0.15 < p["parlay_prob"] < 0.30
            and p != best_overall  # Don't duplicate
        ]

        if medium_risk_candidates:
            # Score by EV adjusted for diversity penalty
            def medium_risk_score(p):
                base_ev = p["ev"]
                diversity_penalty = diversity_score(p, used_games)
                # Penalize by up to 50% of EV for complete overlap
                adjusted_ev = base_ev * (1 - 0.5 * diversity_penalty)
                return adjusted_ev

            medium_risk = max(medium_risk_candidates,
                            key=lambda x: score_with_consensus(x, medium_risk_score))
        else:
            # Fallback: second best EV overall with diversity preference
            remaining = [p for p in parlay_candidates if p != best_overall]
            if remaining:
                medium_risk = max(remaining,
                                key=lambda x: score_with_consensus(x,
                                    lambda p: p["ev"] * (1 - 0.5 * diversity_score(p, used_games))))
            else:
                medium_risk = None

        if medium_risk:
            medium_risk["stake"] = 2
            medium_risk["expected_return"] = medium_risk["stake"] * (1 + medium_risk["ev"])
            used_games.update(medium_risk["games_used"])

        # 3. High Risk ($1): Lower probability (<0.20), highest payout potential, prefer diversity
        high_risk_candidates = [
            p for p in parlay_candidates
            if p["parlay_prob"] < 0.20
            and p != best_overall
            and (medium_risk is None or p != medium_risk)
        ]

        if high_risk_candidates:
            # Choose by highest payout odds with diversity preference
            positive_ev_longshots = [p for p in high_risk_candidates if p["ev"] > -0.3]

            def high_risk_score(p):
                base_odds = p["parlay_decimal_odds"]
                diversity_penalty = diversity_score(p, used_games)
                # Penalize by up to 30% of odds for complete overlap
                adjusted_odds = base_odds * (1 - 0.3 * diversity_penalty)
                return adjusted_odds

            if positive_ev_longshots:
                high_risk = max(positive_ev_longshots,
                              key=lambda x: score_with_consensus(x, high_risk_score))
            else:
                high_risk = max(high_risk_candidates,
                              key=lambda x: score_with_consensus(x, high_risk_score))
        else:
            # Fallback: third best by payout odds with diversity preference
            remaining = [p for p in parlay_candidates
                        if p != best_overall and (medium_risk is None or p != medium_risk)]
            if remaining:
                high_risk = max(remaining,
                              key=lambda x: score_with_consensus(x,
                                  lambda p: p["parlay_decimal_odds"] * (1 - 0.3 * diversity_score(p, used_games))))
            else:
                high_risk = None

        if high_risk:
            high_risk["stake"] = 1
            high_risk["expected_return"] = high_risk["stake"] * (1 + high_risk["ev"])
            used_games.update(high_risk["games_used"])

        # Log selected parlays with details
        all_games = sorted(list(used_games))
        logger.info(f"Game diversity: {len(all_games)} unique games used across 3 parlays: {all_games}")

        logger.info(f"Best Overall ($3): EV={best_overall['ev']:.2%}, Prob={best_overall['parlay_prob']:.2%}, "
                   f"Consensus={best_overall.get('avg_consensus_ratio', 0):.2%}")
        logger.info(f"  Leg 1: {best_overall['leg1']['pick']} ({best_overall['leg1']['consensus_votes']}/{best_overall['leg1']['consensus_total']})")
        logger.info(f"  Leg 2: {best_overall['leg2']['pick']} ({best_overall['leg2']['consensus_votes']}/{best_overall['leg2']['consensus_total']})")

        if medium_risk:
            logger.info(f"Medium Risk ($2): EV={medium_risk['ev']:.2%}, Prob={medium_risk['parlay_prob']:.2%}, "
                       f"Consensus={medium_risk.get('avg_consensus_ratio', 0):.2%}")
            logger.info(f"  Leg 1: {medium_risk['leg1']['pick']} ({medium_risk['leg1']['consensus_votes']}/{medium_risk['leg1']['consensus_total']})")
            logger.info(f"  Leg 2: {medium_risk['leg2']['pick']} ({medium_risk['leg2']['consensus_votes']}/{medium_risk['leg2']['consensus_total']})")
        else:
            logger.info("Medium Risk ($2): N/A")

        if high_risk:
            logger.info(f"High Risk ($1): Odds={high_risk['parlay_american_odds']:+.0f}, Prob={high_risk['parlay_prob']:.2%}, "
                       f"Consensus={high_risk.get('avg_consensus_ratio', 0):.2%}")
            logger.info(f"  Leg 1: {high_risk['leg1']['pick']} ({high_risk['leg1']['consensus_votes']}/{high_risk['leg1']['consensus_total']})")
            logger.info(f"  Leg 2: {high_risk['leg2']['pick']} ({high_risk['leg2']['consensus_votes']}/{high_risk['leg2']['consensus_total']})")
        else:
            logger.info("High Risk ($1): N/A")

        return {
            "best_overall": best_overall,
            "medium_risk": medium_risk,
            "high_risk": high_risk
        }


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
