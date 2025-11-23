"""
Collect historical game data with outcomes
This builds the training dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from google.cloud import storage
import json

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/Robert/Downloads/elite-hangar-479017-m8-401b051a3b72.json'

def fetch_nba_historical_games(start_date, end_date):
    """
    Fetch NBA games with outcomes
    
    Returns DataFrame with:
    - game_date, home_team, away_team
    - home_score, away_score
    - home_spread, total_points
    - home_win, cover, over
    """
    
    # TODO: Use your SportsData.io or API-Sports client
    # For now, creating sample structure
    
    games = []
    
    # Example game structure
    game = {
        'game_date': '2024-11-15',
        'league': 'NBA',
        'home_team': 'Lakers',
        'away_team': 'Celtics',
        'home_score': 115,
        'away_score': 110,
        'home_spread': -5.5,  # Lakers were 5.5 point favorites
        'total_line': 225,
        # Outcomes
        'home_win': 1,  # Lakers won
        'cover': 0,  # Lakers won by 5, didn't cover -5.5
        'over': 0,  # Total was 225, exactly the line
        # Stats (for features)
        'home_win_pct': 0.65,
        'away_win_pct': 0.58,
        'home_avg_points': 112,
        'away_avg_points': 108,
        'home_def_rating': 105,
        'away_def_rating': 107,
    }
    
    games.append(game)
    
    return pd.DataFrame(games)


def build_training_dataset():
    """
    Build complete training dataset from historical games
    """
    print("🏀 Collecting Historical NBA Data...\n")
    
    # Collect last 3 months of games
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    nba_games = fetch_nba_historical_games(start_date, end_date)
    
    print(f"Collected {len(nba_games)} NBA games")
    
    # Add features
    nba_games['spread_abs'] = abs(nba_games['home_spread'])
    nba_games['total_line'] = nba_games['total_line']
    nba_games['home_away_diff'] = nba_games['home_win_pct'] - nba_games['away_win_pct']
    
    # Save
    output_file = f"training_data_nba_{datetime.now().strftime('%Y%m%d')}.csv"
    nba_games.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"Shape: {nba_games.shape}")
    print(f"\nOutcome distribution:")
    print(f"Home wins: {nba_games['home_win'].sum()} ({nba_games['home_win'].mean():.1%})")
    print(f"Covers: {nba_games['cover'].sum()} ({nba_games['cover'].mean():.1%})")
    print(f"Overs: {nba_games['over'].sum()} ({nba_games['over'].mean():.1%})")
    
    return nba_games


if __name__ == "__main__":
    df = build_training_dataset()
