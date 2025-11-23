"""
Advanced Feature Engineering for Sports Betting ML Models
Creates features optimized for spread, moneyline, and totals predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BettingFeatureEngineer:
    """Engineer betting-specific features for ML models"""
    
    def __init__(self, sport: str):
        """
        Initialize feature engineer
        
        Args:
            sport: Sport name (NFL, NBA, etc.)
        """
        self.sport = sport
        
    def calculate_elo_ratings(self, df: pd.DataFrame, k_factor: float = 20) -> pd.DataFrame:
        """
        Calculate ELO ratings for teams
        
        Args:
            df: DataFrame with game results
            k_factor: ELO K-factor (higher = more volatile)
            
        Returns:
            DataFrame with ELO ratings
        """
        logger.info("Calculating ELO ratings")
        
        # Initialize ELO ratings (1500 is standard starting point)
        team_elos = {}
        df = df.sort_values('DateTime')
        
        home_elo = []
        away_elo = []
        
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get current ELO or initialize
            home_rating = team_elos.get(home_team, 1500)
            away_rating = team_elos.get(away_team, 1500)
            
            home_elo.append(home_rating)
            away_elo.append(away_rating)
            
            # Calculate expected scores
            home_expected = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
            away_expected = 1 - home_expected
            
            # Actual result (1 for win, 0 for loss, 0.5 for tie)
            if pd.notna(row.get('HomeScore')) and pd.notna(row.get('AwayScore')):
                if row['HomeScore'] > row['AwayScore']:
                    home_actual, away_actual = 1, 0
                elif row['HomeScore'] < row['AwayScore']:
                    home_actual, away_actual = 0, 1
                else:
                    home_actual, away_actual = 0.5, 0.5
                
                # Update ratings
                team_elos[home_team] = home_rating + k_factor * (home_actual - home_expected)
                team_elos[away_team] = away_rating + k_factor * (away_actual - away_expected)
        
        df['HomeELO'] = home_elo
        df['AwayELO'] = away_elo
        df['ELODiff'] = df['HomeELO'] - df['AwayELO']
        
        return df
    
    def add_recent_form(self, df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Add recent form features (wins, points, etc.)
        
        Args:
            df: DataFrame with game results
            windows: List of window sizes for rolling stats
            
        Returns:
            DataFrame with form features
        """
        logger.info(f"Adding recent form features with windows: {windows}")
        
        df = df.sort_values('DateTime')
        
        for window in windows:
            # Home team recent performance
            df[f'Home_Wins_L{window}'] = df.groupby('HomeTeam')['HomeWin'].transform(
                lambda x: x.rolling(window, min_periods=1).sum().shift(1)
            )
            
            df[f'Home_AvgScore_L{window}'] = df.groupby('HomeTeam')['HomeScore'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            
            df[f'Home_AvgAllowed_L{window}'] = df.groupby('HomeTeam')['AwayScore'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            
            # Away team recent performance (need to flip home/away)
            # Create temporary away records
            away_games = df[['DateTime', 'AwayTeam', 'AwayScore', 'HomeScore']].copy()
            away_games.columns = ['DateTime', 'Team', 'Score', 'Allowed']
            away_games['Win'] = (away_games['Score'] > away_games['Allowed']).astype(int)
            
            away_games = away_games.sort_values(['Team', 'DateTime'])
            
            df[f'Away_Wins_L{window}'] = df.groupby('AwayTeam').apply(
                lambda x: away_games[away_games['Team'] == x.name]['Win'].rolling(window, min_periods=1).sum().shift(1)
            ).values
            
            df[f'Away_AvgScore_L{window}'] = df.groupby('AwayTeam')['AwayScore'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            
            df[f'Away_AvgAllowed_L{window}'] = df.groupby('AwayTeam')['HomeScore'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
        
        return df
    
    def add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days of rest between games
        
        Args:
            df: DataFrame with game dates
            
        Returns:
            DataFrame with rest day features
        """
        logger.info("Calculating rest days")
        
        df = df.sort_values('DateTime')
        
        # Home team rest
        df['HomeRestDays'] = df.groupby('HomeTeam')['DateTime'].diff().dt.days
        
        # Away team rest
        df['AwayRestDays'] = df.groupby('AwayTeam')['DateTime'].diff().dt.days
        
        # Fill first games with median rest
        df['HomeRestDays'].fillna(df['HomeRestDays'].median(), inplace=True)
        df['AwayRestDays'].fillna(df['AwayRestDays'].median(), inplace=True)
        
        # Back-to-back indicator
        df['HomeBackToBack'] = (df['HomeRestDays'] <= 1).astype(int)
        df['AwayBackToBack'] = (df['AwayRestDays'] <= 1).astype(int)
        
        return df
    
    def add_head_to_head(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Add head-to-head record features
        
        Args:
            df: DataFrame with games
            window: Number of recent H2H games to consider
            
        Returns:
            DataFrame with H2H features
        """
        logger.info("Adding head-to-head features")
        
        df = df.sort_values('DateTime')
        
        # Create matchup identifier
        df['Matchup'] = df.apply(
            lambda x: tuple(sorted([x['HomeTeam'], x['AwayTeam']])), 
            axis=1
        )
        
        # Calculate H2H record
        df['HomeWins_H2H'] = 0
        df['AwayWins_H2H'] = 0
        df['TotalGames_H2H'] = 0
        
        for idx, row in df.iterrows():
            matchup = row['Matchup']
            current_date = row['DateTime']
            
            # Get previous games in this matchup
            prev_games = df[
                (df['Matchup'] == matchup) & 
                (df['DateTime'] < current_date)
            ].tail(window)
            
            if len(prev_games) > 0:
                # Count wins for each team
                home_wins = len(prev_games[
                    (prev_games['HomeTeam'] == row['HomeTeam']) & 
                    (prev_games['HomeScore'] > prev_games['AwayScore'])
                ])
                
                away_wins_as_away = len(prev_games[
                    (prev_games['AwayTeam'] == row['AwayTeam']) & 
                    (prev_games['AwayScore'] > prev_games['HomeScore'])
                ])
                
                # When teams switched venues
                away_wins_as_home = len(prev_games[
                    (prev_games['HomeTeam'] == row['AwayTeam']) & 
                    (prev_games['HomeScore'] > prev_games['AwayScore'])
                ])
                
                df.at[idx, 'HomeWins_H2H'] = home_wins
                df.at[idx, 'AwayWins_H2H'] = away_wins_as_away + away_wins_as_home
                df.at[idx, 'TotalGames_H2H'] = len(prev_games)
        
        return df
    
    def add_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pace/tempo features (particularly important for NBA/NCAAB)
        
        Args:
            df: DataFrame with game data
            
        Returns:
            DataFrame with pace features
        """
        logger.info("Adding pace features")
        
        if self.sport not in ['NBA', 'NCAAB']:
            return df
        
        # Calculate possessions (approximate formula)
        # Possessions ≈ FGA + 0.44*FTA - ORB + TOV
        
        if all(col in df.columns for col in ['HomeScore', 'AwayScore']):
            df['TotalPossessions'] = df['HomeScore'] + df['AwayScore']  # Simplified
            
            # Team pace (rolling average)
            df['HomePace_L5'] = df.groupby('HomeTeam')['TotalPossessions'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            df['AwayPace_L5'] = df.groupby('AwayTeam')['TotalPossessions'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            df['CombinedPace'] = (df['HomePace_L5'] + df['AwayPace_L5']) / 2
        
        return df
    
    def add_strength_of_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strength of schedule metrics
        
        Args:
            df: DataFrame with game results
            
        Returns:
            DataFrame with SOS features
        """
        logger.info("Calculating strength of schedule")
        
        df = df.sort_values('DateTime')
        
        # Calculate team win percentages
        team_records = {}
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]
            
            home_wins = (home_games['HomeScore'] > home_games['AwayScore']).sum()
            away_wins = (away_games['AwayScore'] > away_games['HomeScore']).sum()
            total_games = len(home_games) + len(away_games)
            
            team_records[team] = (home_wins + away_wins) / total_games if total_games > 0 else 0.5
        
        # Calculate opponent win percentage
        df['OppWinPct_Home'] = df['AwayTeam'].map(team_records)
        df['OppWinPct_Away'] = df['HomeTeam'].map(team_records)
        
        # Rolling average of opponent strength
        df['SOS_Home_L5'] = df.groupby('HomeTeam')['OppWinPct_Home'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        df['SOS_Away_L5'] = df.groupby('AwayTeam')['OppWinPct_Away'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        return df
    
    def add_betting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features from betting lines (implied probabilities, CLV, etc.)
        
        Args:
            df: DataFrame with odds data
            
        Returns:
            DataFrame with betting features
        """
        logger.info("Adding betting-specific features")
        
        # Convert American odds to implied probability
        def american_to_prob(odds):
            if pd.isna(odds):
                return None
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        
        # Moneyline implied probabilities
        if 'HomeMoneyLine' in df.columns:
            df['HomeML_ImpliedProb'] = df['HomeMoneyLine'].apply(american_to_prob)
            df['AwayML_ImpliedProb'] = df['AwayMoneyLine'].apply(american_to_prob) if 'AwayMoneyLine' in df.columns else None
        
        # Point spread movement
        if 'PointSpread' in df.columns and 'PointSpreadOpen' in df.columns:
            df['SpreadMovement'] = df['PointSpread'] - df['PointSpreadOpen']
        
        # Over/Under movement
        if 'OverUnder' in df.columns and 'OverUnderOpen' in df.columns:
            df['TotalMovement'] = df['OverUnder'] - df['OverUnderOpen']
        
        # Public betting percentage (if available)
        if 'HomePublicBettingPercent' in df.columns:
            df['PublicFavorite'] = (df['HomePublicBettingPercent'] > 50).astype(int)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for different bet types
        
        Args:
            df: DataFrame with game results
            
        Returns:
            DataFrame with target variables
        """
        logger.info("Creating target variables")
        
        # Moneyline targets
        df['Target_HomeML'] = (df['HomeScore'] > df['AwayScore']).astype(int)
        df['Target_AwayML'] = (df['AwayScore'] > df['HomeScore']).astype(int)
        
        # Spread targets
        if 'PointSpread' in df.columns:
            df['Target_HomeSpread'] = (df['HomeScore'] + df['PointSpread'] > df['AwayScore']).astype(int)
            df['Target_AwaySpread'] = (df['AwayScore'] - df['PointSpread'] > df['HomeScore']).astype(int)
        
        # Over/Under targets
        if 'OverUnder' in df.columns:
            df['Target_Over'] = (df['HomeScore'] + df['AwayScore'] > df['OverUnder']).astype(int)
            df['Target_Under'] = (df['HomeScore'] + df['AwayScore'] < df['OverUnder']).astype(int)
        
        # Regression targets (for margin/total prediction)
        df['Target_Margin'] = df['HomeScore'] - df['AwayScore']
        df['Target_Total'] = df['HomeScore'] + df['AwayScore']
        
        return df
    
    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            df: Raw game data
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Running full feature engineering for {self.sport}")
        
        # Run all feature engineering steps
        df = self.calculate_elo_ratings(df)
        df = self.add_recent_form(df)
        df = self.add_rest_days(df)
        df = self.add_head_to_head(df)
        df = self.add_pace_features(df)
        df = self.add_strength_of_schedule(df)
        df = self.add_betting_features(df)
        df = self.create_target_variables(df)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} total features")
        
        return df


def main():
    """Example usage"""
    # Load your processed data
    df = pd.read_csv('./training_data/nfl_processed.csv')
    
    # Initialize feature engineer
    engineer = BettingFeatureEngineer('NFL')
    
    # Create all features
    df_features = engineer.process_all_features(df)
    
    # Save
    df_features.to_csv('./training_data/nfl_ml_ready.csv', index=False)
    print(f"Created {len(df_features.columns)} features for {len(df_features)} games")


if __name__ == "__main__":
    main()
