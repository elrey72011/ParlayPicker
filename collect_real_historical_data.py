"""
Collect REAL historical data from your APIs for training
Uses your existing SportsData.io and API-Sports clients
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# Add your app to path
sys.path.append('C:/Users/Robert/PycharmProjects/PythonProject11')

# Import your clients
from app_core import (
    SportsDataNBAClient,
    SportsDataNFLClient,
    SportsDataNHLClient,
    APISportsBasketballClient
)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/Robert/Downloads/elite-hangar-479017-m8-401b051a3b72.json'

def collect_nba_games():
    """Collect NBA games with real data"""
    
    print("🏀 Collecting NBA Historical Data...\n")
    
    # You'll need your API keys
    api_key = input("Enter your SportsData.io NBA key: ")
    
    client = SportsDataNBAClient(api_key=api_key)
    
    games = []
    
    # Get last 30 days of scores
    for days_ago in range(30):
        game_date = datetime.now() - timedelta(days=days_ago)
        date_str = game_date.strftime('%Y-%m-%d')
        
        print(f"Fetching {date_str}...")
        
        try:
            daily_games = client.get_scores_by_date(date_str)
            
            for game in daily_games:
                if game.get('Status') != 'Final':
                    continue
                
                home_score = game.get('HomeScore', 0)
                away_score = game.get('AwayScore', 0)
                
                if home_score == 0 and away_score == 0:
                    continue
                
                # Extract data
                game_data = {
                    'game_date': date_str,
                    'league': 'NBA',
                    'home_team': game.get('HomeTeam'),
                    'away_team': game.get('AwayTeam'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_win': 1 if home_score > away_score else 0,
                    # TODO: Add more stats from game object
                }
                
                games.append(game_data)
                
        except Exception as e:
            print(f"Error on {date_str}: {e}")
            continue
    
    df = pd.DataFrame(games)
    print(f"\n✅ Collected {len(df)} completed games")
    
    return df


if __name__ == "__main__":
    nba_df = collect_nba_games()
    
    output_file = f"real_training_data_{datetime.now().strftime('%Y%m%d')}.csv"
    nba_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved to: {output_file}")
    print(nba_df.head())
