import pandas as pd
import requests
import numpy as np
from google.cloud import aiplatform
import sys
import os

# --- 1. SETUP IMPORTS ---
# Add project root to path so we can import from app_core and config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import THE_ODDS_API_KEY, VERTEX_CONFIG, MASTER_DATA_FILE
# This now correctly looks inside the 'app_core' folder
from app_core.feature_engine import prepare_features_for_inference

def fetch_todays_odds():
    """Fetches LIVE upcoming odds from The Odds API"""
    print("📡 Fetching live odds from API...")
    all_games = []
    sports = ['basketball_nba', 'icehockey_nhl', 'basketball_ncaab']
    
    for sport in sports:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            params = {
                "apiKey": THE_ODDS_API_KEY, 
                "regions": "us", 
                "markets": "h2h,spreads", 
                "oddsFormat": "american"
            }
            res = requests.get(url, params=params)
            data = res.json()
            
            if isinstance(data, list):
                for game in data:
                    row = {
                        'game_id': game['id'], 
                        'sport': sport, 
                        'commence_time': game['commence_time'],
                        'home_team': game['home_team'], 
                        'away_team': game['away_team'],
                        'home_score': np.nan, 
                        'away_score': np.nan,
                        # Defaults (will be filled by rolling stats logic)
                        'home_win_pct': 0.5, 
                        'away_win_pct': 0.5, 
                        'spread_normalized': 0.0 
                    }
                    all_games.append(row)
        except Exception as e:
            print(f"❌ Error fetching {sport}: {e}")
            
    return pd.DataFrame(all_games)

def get_vertex_predictions(df):
    """Sends clean data to Vertex AI for prediction"""
    print(f"🔮 Calling Vertex AI for {len(df)} games...")
    
    aiplatform.init(
        project=VERTEX_CONFIG['project_id'], 
        location=VERTEX_CONFIG['location']
    )
    
    endpoint_name = f"projects/{VERTEX_CONFIG['project_id']}/locations/{VERTEX_CONFIG['location']}/endpoints/{VERTEX_CONFIG['endpoint_id']}"
    endpoint = aiplatform.Endpoint(endpoint_name)
    
    # 1. Select ONLY feature columns
    features = VERTEX_CONFIG['feature_cols']
    clean_df = df.copy()
    
    # Ensure all columns exist
    for col in features:
        if col not in clean_df.columns:
            clean_df[col] = 0.0
            
    # 2. Strict Type Conversion (Fixes 'Unicode-3' Error)
    clean_df = clean_df[features].fillna(0.0)
    X_pred = []
    for row in clean_df.values:
        clean_row = []
        for val in row:
            try:
                clean_row.append(float(val))
            except:
                clean_row.append(0.0)
        X_pred.append(clean_row)
    
    # 3. Predict
    try:
        preds = endpoint.predict(instances=X_pred).predictions
        df['ml_win_prob'] = preds
    except Exception as e:
        print(f"❌ Vertex Prediction Error: {e}")
        df['ml_win_prob'] = 0.5 
        
    return df

def main():
    # 1. Load History
    try:
        history = pd.read_csv(MASTER_DATA_FILE)
        print(f"✅ Loaded history: {len(history)} games")
    except FileNotFoundError:
        print(f"⚠️ History file not found at {MASTER_DATA_FILE}. Creating empty.")
        history = pd.DataFrame(columns=['game_id', 'sport', 'commence_time'])

    # 2. Get Today's Games
    today = fetch_todays_odds()
    if today.empty:
        print("❌ No games found today.")
        return

    # 3. Engineer Features
    ready_df = prepare_features_for_inference(history, today)
    
    # 4. Predict
    results = get_vertex_predictions(ready_df)
    
    # 5. Calculate Edge & Save
    results['edge'] = results['ml_win_prob'] - 0.5  
    results = results.sort_values('edge', ascending=False)
    
    print("\n🚀 TOP PICKS:")
    print(results[['sport', 'home_team', 'away_team', 'ml_win_prob', 'edge']].head(5))
    
    results.to_csv("todays_picks.csv", index=False)
    print("\n✅ Saved to todays_picks.csv")

if __name__ == "__main__":
    main()
