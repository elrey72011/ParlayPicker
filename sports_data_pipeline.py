"""
Sports Data Pipeline for ML Training
Pulls historical data from sportsdata.io API for NFL, NBA, NCAAB, NCAAF, NHL
Prepares features for betting prediction models
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SportsDataPipeline:
    """Main pipeline class for fetching and processing sports data"""
    
    BASE_URLS = {
        'NFL': 'https://api.sportsdata.io/v3/nfl',
        'NBA': 'https://api.sportsdata.io/v3/nba',
        'NCAAB': 'https://api.sportsdata.io/v3/cbb',
        'NCAAF': 'https://api.sportsdata.io/v3/cfb',
        'NHL': 'https://api.sportsdata.io/v3/nhl'
    }
    
    def __init__(self, api_keys: Dict[str, str], output_dir: str = './data'):
        """
        Initialize the pipeline
        
        Args:
            api_keys: Dictionary mapping sport names to API keys
            output_dir: Directory to save processed data
        """
        self.api_keys = api_keys
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _make_request(self, sport: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with error handling and rate limiting
        
        Args:
            sport: Sport name (NFL, NBA, etc.)
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response data
        """
        url = f"{self.BASE_URLS[sport]}/scores/json/{endpoint}"
        headers = {'Ocp-Apim-Subscription-Key': self.api_keys[sport]}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.5)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {sport} {endpoint}: {e}")
            return None
    
    def fetch_historical_games(self, sport: str, seasons: List[str]) -> pd.DataFrame:
        """
        Fetch historical game data for multiple seasons
        
        Args:
            sport: Sport name
            seasons: List of season strings (e.g., ['2023', '2024'])
            
        Returns:
            DataFrame with historical games
        """
        all_games = []
        
        for season in seasons:
            logger.info(f"Fetching {sport} games for {season} season")
            
            if sport in ['NFL', 'NCAAF']:
                # These use season type (REG, POST)
                for season_type in ['REG', 'POST']:
                    games = self._make_request(sport, f'Scores/{season}{season_type}')
                    if games:
                        all_games.extend(games)
            else:
                # NBA, NCAAB, NHL use full season
                games = self._make_request(sport, f'Games/{season}')
                if games:
                    all_games.extend(games)
        
        df = pd.DataFrame(all_games)
        logger.info(f"Fetched {len(df)} games for {sport}")
        return df
    
    def fetch_team_stats(self, sport: str, season: str) -> pd.DataFrame:
        """
        Fetch team statistics for a season
        
        Args:
            sport: Sport name
            season: Season string
            
        Returns:
            DataFrame with team stats
        """
        logger.info(f"Fetching {sport} team stats for {season}")
        
        endpoint = f'TeamSeasonStats/{season}'
        if sport in ['NFL', 'NCAAF']:
            endpoint = f'TeamSeasonStats/{season}REG'
            
        stats = self._make_request(sport, endpoint)
        
        if stats:
            df = pd.DataFrame(stats)
            logger.info(f"Fetched stats for {len(df)} teams")
            return df
        return pd.DataFrame()
    
    def fetch_odds_history(self, sport: str, season: str) -> pd.DataFrame:
        """
        Fetch historical odds/lines data
        
        Args:
            sport: Sport name
            season: Season string
            
        Returns:
            DataFrame with odds history
        """
        logger.info(f"Fetching {sport} odds for {season}")
        
        # Adjust endpoint based on sport
        if sport in ['NFL', 'NCAAF']:
            endpoint = f'GameOddsByWeek/{season}/1'  # Start with week 1
            all_odds = []
            for week in range(1, 18):  # Adjust range as needed
                odds = self._make_request(sport, f'GameOddsByWeek/{season}/{week}')
                if odds:
                    all_odds.extend(odds)
        else:
            endpoint = f'GameOdds/{season}'
            all_odds = self._make_request(sport, endpoint)
        
        if all_odds:
            df = pd.DataFrame(all_odds)
            logger.info(f"Fetched odds for {len(df)} games")
            return df
        return pd.DataFrame()
    
    def engineer_features(self, games_df: pd.DataFrame, sport: str) -> pd.DataFrame:
        """
        Engineer features for ML model training
        
        Args:
            games_df: Raw games data
            sport: Sport name
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {sport}")
        
        df = games_df.copy()
        
        # Convert date columns
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['Month'] = df['DateTime'].dt.month
        
        # Home/Away indicators
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            df['IsHome'] = 1  # Can add away rows if needed
        
        # Score-based features
        if 'HomeScore' in df.columns and 'AwayScore' in df.columns:
            df['TotalPoints'] = df['HomeScore'] + df['AwayScore']
            df['ScoreDifferential'] = df['HomeScore'] - df['AwayScore']
            df['HomeWin'] = (df['HomeScore'] > df['AwayScore']).astype(int)
        
        # Point spread features
        if 'PointSpread' in df.columns:
            df['SpreadCovered'] = (df['ScoreDifferential'] > df['PointSpread']).astype(int)
        
        # Over/Under features
        if 'OverUnder' in df.columns:
            df['WentOver'] = (df['TotalPoints'] > df['OverUnder']).astype(int)
        
        # Rolling averages (need to sort by team and date first)
        # This is a simplified version - you'll want to expand this
        df = df.sort_values('DateTime')
        
        logger.info(f"Engineered {len(df.columns)} features")
        return df
    
    def add_rolling_team_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Add rolling statistics for each team
        
        Args:
            df: DataFrame with game data
            window: Number of games for rolling window
            
        Returns:
            DataFrame with rolling stats
        """
        logger.info(f"Calculating {window}-game rolling stats")
        
        # Sort by team and date
        df = df.sort_values(['HomeTeam', 'DateTime'])
        
        # Calculate rolling stats for home team
        for col in ['HomeScore', 'AwayScore']:
            if col in df.columns:
                new_col = f'{col}_Rolling{window}'
                df[new_col] = df.groupby('HomeTeam')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

                # Backfill initial NaNs with the team's historical average
                df[new_col] = df[new_col].fillna(
                    df.groupby('HomeTeam')[col].transform('mean')
                )

                # If still NaN (e.g. team has no data at all), use overall mean
                overall_mean = df[col].mean()
                if pd.notna(overall_mean):
                    df[new_col] = df[new_col].fillna(overall_mean)
                else:
                    df[new_col] = df[new_col].fillna(0.0)
        
        return df
    
    def process_sport(self, sport: str, seasons: List[str], include_odds: bool = True) -> pd.DataFrame:
        """
        Complete pipeline for a single sport
        
        Args:
            sport: Sport name
            seasons: List of seasons to process
            include_odds: Whether to include odds data
            
        Returns:
            Processed DataFrame ready for ML training
        """
        logger.info(f"Processing {sport} data")
        
        # Fetch games
        games_df = self.fetch_historical_games(sport, seasons)
        
        if games_df.empty:
            logger.warning(f"No games data for {sport}")
            return pd.DataFrame()
        
        # Engineer features
        df = self.engineer_features(games_df, sport)
        
        # Add rolling stats
        df = self.add_rolling_team_stats(df)
        
        # Fetch and merge odds if requested
        if include_odds:
            odds_df = self.fetch_odds_history(sport, seasons[-1])  # Most recent season
            if not odds_df.empty:
                # Merge odds data (adjust keys based on actual column names)
                if 'GameID' in df.columns and 'GameID' in odds_df.columns:
                    df = df.merge(odds_df, on='GameID', how='left', suffixes=('', '_odds'))
        
        # Validation: Diagnostic check for non-null features before passing to Predict Batch phase
        feature_cols = [col for col in df.columns if 'Rolling' in col or col.startswith('feature_') or col in ['TotalPoints', 'ScoreDifferential', 'HomeWin']]
        if feature_cols:
            non_null_counts = df[feature_cols].notna().sum()
            avg_non_null = non_null_counts.mean()
            logger.info(f"Validation: Engineered {len(feature_cols)} feature columns. Average non-null count per feature: {avg_non_null:.1f} out of {len(df)} rows.")

        # Save processed data
        output_file = self.output_dir / f'{sport.lower()}_processed.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {sport} data to {output_file}")
        
        return df
    
    def run_all_sports(self, seasons: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Run pipeline for all sports
        
        Args:
            seasons: List of seasons to process
            
        Returns:
            Dictionary mapping sport names to processed DataFrames
        """
        results = {}
        
        for sport in self.BASE_URLS.keys():
            if sport in self.api_keys:
                try:
                    df = self.process_sport(sport, seasons)
                    results[sport] = df
                except Exception as e:
                    logger.error(f"Error processing {sport}: {e}")
                    continue
            else:
                logger.warning(f"No API key provided for {sport}")
        
        return results


def main():
    """Example usage"""
    
    # Configure your API keys
    api_keys = {
        'NFL': 'YOUR_NFL_API_KEY',
        'NBA': 'YOUR_NBA_API_KEY',
        'NCAAB': 'YOUR_NCAAB_API_KEY',
        'NCAAF': 'YOUR_NCAAF_API_KEY',
        'NHL': 'YOUR_NHL_API_KEY'
    }
    
    # Initialize pipeline
    pipeline = SportsDataPipeline(api_keys, output_dir='./training_data')
    
    # Process recent seasons
    seasons = ['2022', '2023', '2024', '2025', '2026']
    
    # Run for all sports
    results = pipeline.run_all_sports(seasons)
    
    # Print summary
    for sport, df in results.items():
        print(f"\n{sport}: {len(df)} games processed")
        if not df.empty:
            print(f"Columns: {list(df.columns)[:10]}...")  # First 10 columns


if __name__ == "__main__":
    main()
