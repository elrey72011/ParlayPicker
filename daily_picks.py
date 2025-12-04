# Save this as: parlaypicker/daily_picks.py
import pandas as pd
import requests
import numpy as np
from google.cloud import aiplatform
import os

# --- IMPORT YOUR MODULES ---
from src.feature_engine import prepare_features_for_inference
# from src.kalshi_api import get_kalshi_markets  <-- Uncomment when you add your Kalshi file
# from src.theover_api import get_ai_scores      <-- Uncomment when you add your TheOver file

# --- CONFIGURATION ---
PROJECT_ID = "elite-hangar-479017-m8"
REGION = "us-central1"
ENDPOINT_ID = "155434660283809792"  # Your WORKING endpoint ID
ODDS_API_KEY = "b722c798f7bca605da45a09dba155152" 

FEATURE_COLS = [
    "home_win_pct", "away_win_pct", "home_avg_points", "away_avg_points",
    "home_form_last5", "away_form_last5", "home_pd_last5", "away_pd_last5",
    "home_sos_last5", "away_sos_last5", "win_pct_diff", "avg_points_diff",
    "form_diff", "pd_last5_diff", "sos_diff", "def_rating_diff", "streak_diff",
    "spread_normalized", "public_betting_centered", "sharp_vs_public",
    "rest_advantage", "back_to_back", "primetime_game", "division_game",
    "injuries_impact", "weather_factor", "line_movement", "total_movement",
    "model_consensus", "theover_probability"
]

def fetch_todays_odds():
    """Fetches LIVE upcoming odds"""
    print("📡 Fetching live odds from API...")
    # [Paste your fetch logic here, or import it if you have it in another file]
    # For brevity, reusing the logic from your notebook:
    all_games = []
    sports = ['basketball_nba', 'icehockey_nhl', 'basketball_ncaab']
    for sport in sports:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h,spreads", "oddsFormat": "american"}
            res = requests.get(url, params=params)
            data = res.json()
            if isinstance(data, list):
                for game in data:
                    row = {
                        'game_id': game['id'], 'sport': sport, 'commence_time': game['commence_time'],
                        'home_team': game['home_team'], 'away_team': game['away_team'],
                        'home_score': np.nan, 'away_score': np.nan,
                        'home_win_pct': 0.5, 'away_win_pct': 0.5, 'spread_normalized': 0.0 # Placeholder
                    }
                    all_games.append(row)
        except Exception as e:
            print(f"❌ Error {sport}: {e}")
    return pd.DataFrame(all_games)

def get_vertex_predictions(df):
    print(f"🔮 calling Vertex AI for {len(df)} games...")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")
    
    # Ensure float conversion to avoid "Unicode-3" error
    clean_df = df[FEATURE_COLS].copy()
    
    # Fill defaults for any missing columns
    for col in FEATURE_COLS:
        if col not in clean_df.columns:
            clean_df[col] = 0.0
            
    # Force float type
    X_pred = clean_df.astype(float).values.tolist()
    
    preds = endpoint.predict(instances=X_pred).predictions
    df['ml_win_prob'] = preds
    return df

def main():
    # 1. Load History
    try:
        history = pd.read_csv("master_all_sports.csv")
    except:
        print("⚠️ No history found, creating empty DataFrame")
        history = pd.DataFrame(columns=['game_id', 'sport', 'commence_time'])

    # 2. Get Today's Games
    today = fetch_todays_odds()
    if today.empty:
        print("❌ No games found today.")
        return

    # 3. Merge External Alpha (Kalshi, TheOver, etc.)
    # today = today.merge(get_kalshi_markets(), ...) 
    
    # 4. Engineer Features
    ready_df = prepare_features_for_inference(history, today)
    
    # 5. Predict
    results = get_vertex_predictions(ready_df)
    
    # 6. Calculate Edge & Save
    results['edge'] = results['ml_win_prob'] - 0.5  # Or implied prob if you have it
    results = results.sort_values('edge', ascending=False)
    
    print("\n🚀 TOP PICKS:")
    print(results[['sport', 'home_team', 'away_team', 'ml_win_prob', 'edge']].head(5))
    
    results.to_csv("todays_picks.csv", index=False)
    print("\n✅ Saved to todays_picks.csv")

if __name__ == "__main__":
    main()
