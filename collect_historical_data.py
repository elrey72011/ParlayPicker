"""
Collect Historical Game Data for ML Training
Uses The Odds API (Paid Tier) to fetch high-density historical data for model training.

Endpoints used:
- Scores & Results: /v4/sports/{sport}/scores/?daysFrom=45&apiKey={key}
- Historical Odds: /v4/historical/sports/{sport}/odds/ (1 hour prior to commence_time)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time
import argparse
from typing import Dict, List, Optional, Any

# ============================================================
# ODDS API CONFIGURATION
# ============================================================

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Map internal sport names to Odds API keys
SPORT_KEYS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NHL": "icehockey_nhl",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf"
}

# Feature names for ML training
FEATURE_NAMES = [
    "home_win_pct", "away_win_pct", "home_ppg", "away_ppg",
    "home_oppg", "away_oppg", "spread_normalized",
    "home_last_5", "away_last_5", "home_home_record", "away_away_record",
    "head_to_head", "rest_advantage", "injuries_impact", "weather_factor",
    "public_betting_pct", "sharp_money_indicator", "line_movement",
    "total_movement", "model_consensus", "theover_probability",
    "implied_home_prob", "home_streak", "away_streak", "division_game",
    "back_to_back", "primetime_game",
]

# ============================================================
# NORMALIZATION UTILS
# ============================================================

def normalize_spread(spread: float, sport: str) -> float:
    max_spreads = {"NBA": 25, "NHL": 3.5, "NCAAB": 40, "NFL": 20, "NCAAF": 40}
    max_spread = max_spreads.get(sport, 20)
    if spread is None or pd.isna(spread):
        return 0.5
    try:
        normalized = (float(spread) + max_spread) / (2 * max_spread)
        return float(np.clip(normalized, 0, 1))
    except (ValueError, TypeError):
        return 0.5

def normalize_points(points: float, sport: str) -> float:
    ranges = {
        "NBA": (90, 130), "NCAAB": (50, 90), "NHL": (1.5, 4.5),
        "NFL": (14, 35), "NCAAF": (14, 45),
    }
    min_pts, max_pts = ranges.get(sport, (0, 100))
    if points is None or pd.isna(points):
        return 0.5
    try:
        normalized = (float(points) - min_pts) / (max_pts - min_pts)
        return float(np.clip(normalized, 0, 1))
    except (ValueError, TypeError):
        return 0.5

def safe_american_to_prob(odds: float) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

# ============================================================
# API CLIENT
# ============================================================

class OddsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_scores(self, sport_key: str, days_from: int = 45) -> List[Dict]:
        """Fetch historical scores up to 45 days back"""
        url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores"
        params = {
            "apiKey": self.api_key,
            "daysFrom": days_from
        }

        print(f"  Fetching scores for {sport_key} (daysFrom={days_from})...")
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("  Rate limited, waiting 5 seconds...")
                time.sleep(5)
                return self.get_scores(sport_key, days_from)
            else:
                print(f"  Error fetching scores: {resp.status_code} - {resp.text}")
                return []
        except Exception as e:
            print(f"  Exception fetching scores: {e}")
            return []

    def get_historical_odds(self, sport_key: str, snapshot_time: str) -> List[Dict]:
        """Fetch odds snapshot at a specific point in time"""
        url = f"{ODDS_API_BASE_URL}/historical/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads",
            "date": snapshot_time
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
            elif resp.status_code == 429:
                print("  Rate limited, waiting 5 seconds...")
                time.sleep(5)
                return self.get_historical_odds(sport_key, snapshot_time)
            else:
                print(f"  Error fetching historical odds: {resp.status_code}")
                return []
        except Exception as e:
            print(f"  Exception fetching historical odds: {e}")
            return []

# ============================================================
# MAIN COLLECTION LOGIC
# ============================================================

def collect_sport_data(sport: str, sport_key: str, client: OddsAPIClient) -> List[Dict]:
    games = client.get_scores(sport_key, days_from=45)
    
    # Filter for completed games
    completed_games = [g for g in games if g.get("completed") == True]
    print(f"  Found {len(completed_games)} completed games for {sport}")
    
    # Sort chronologically by commence_time so our rolling stats calculate properly
    completed_games.sort(key=lambda x: x.get("commence_time", ""))
    
    # Team stats tracker for the rolling window
    # team_name -> list of (points_scored, points_allowed, win/loss (1/0))
    team_history = {}
    
    all_game_records = []
    
    for game in completed_games:
        commence_time_str = game.get("commence_time")
        if not commence_time_str:
            continue
            
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        scores = game.get("scores")
        if not scores:
            continue
            
        home_score = 0
        away_score = 0
        for s in scores:
            if s.get("name") == home_team:
                home_score = float(s.get("score") or 0)
            elif s.get("name") == away_team:
                away_score = float(s.get("score") or 0)

        if home_score == 0 and away_score == 0:
            continue
            
        # Determine winner
        home_won = 1 if home_score > away_score else 0
        away_won = 1 if away_score > home_score else 0

        # Calculate rolling stats (before adding current game)
        home_hist = team_history.get(home_team, [])
        away_hist = team_history.get(away_team, [])

        # We want the last 5 games
        home_last_5 = home_hist[-5:] if len(home_hist) > 0 else []
        away_last_5 = away_hist[-5:] if len(away_hist) > 0 else []

        # Calculate PPG / OPPG / Win Pct
        home_ppg = sum([h[0] for h in home_last_5]) / len(home_last_5) if home_last_5 else None
        home_oppg = sum([h[1] for h in home_last_5]) / len(home_last_5) if home_last_5 else None
        home_win_pct = sum([h[2] for h in home_hist]) / len(home_hist) if home_hist else 0.5

        away_ppg = sum([h[0] for h in away_last_5]) / len(away_last_5) if away_last_5 else None
        away_oppg = sum([h[1] for h in away_last_5]) / len(away_last_5) if away_last_5 else None
        away_win_pct = sum([h[2] for h in away_hist]) / len(away_hist) if away_hist else 0.5

        # "Gold Standard" Snapshot: 1 hour prior to commence_time
        commence_dt = datetime.strptime(commence_time_str.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
        snapshot_dt = commence_dt - timedelta(hours=1)
        snapshot_str = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        implied_home_prob = 0.5
        spread_normalized = 0.5

        # Fetch the historical odds snapshot, use a local cache to save credits
        if not hasattr(client, "_snapshot_cache"):
            client._snapshot_cache = {}
            
        cache_key = f"{sport_key}_{snapshot_str}"
        if cache_key in client._snapshot_cache:
            snapshot_odds = client._snapshot_cache[cache_key]
        else:
            snapshot_odds = client.get_historical_odds(sport_key, snapshot_str)
            client._snapshot_cache[cache_key] = snapshot_odds
            # Brief pause to respect rate limits if we fetched a new snapshot
            time.sleep(0.5)
        
        # Find this specific game in the snapshot
        target_game = None
        for g in snapshot_odds:
            if g.get("home_team") == home_team and g.get("away_team") == away_team:
                target_game = g
                break

        if target_game:
            bookmakers = target_game.get("bookmakers", [])
            if bookmakers:
                # Use Pinnacle or consensus average
                pinnacle = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
                bookie_to_use = pinnacle if pinnacle else bookmakers[0]

                for market in bookie_to_use.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_team:
                                price = outcome.get("price")
                                if price:
                                    if price > 1.0 and price < 100.0:
                                        # decimal odds
                                        implied_home_prob = 1.0 / price
                                    elif price >= 100.0 or price <= -100.0:
                                        # american odds
                                        implied_home_prob = safe_american_to_prob(price)
                    elif market.get("key") == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_team:
                                point = outcome.get("point")
                                if point is not None:
                                    spread_normalized = normalize_spread(float(point), sport)

        # Baseline features
        features = {
            "home_win_pct": np.clip(home_win_pct, 0, 1),
            "away_win_pct": np.clip(away_win_pct, 0, 1),
            "home_ppg": normalize_points(home_ppg, sport),
            "away_ppg": normalize_points(away_ppg, sport),
            "home_oppg": normalize_points(home_oppg, sport),
            "away_oppg": normalize_points(away_oppg, sport),
            "spread_normalized": spread_normalized,
            "implied_home_prob": implied_home_prob,
        }

        # Add default neutral values for the rest of the features to satisfy matrix
        for f in FEATURE_NAMES:
            if f not in features:
                features[f] = 0.5 if "pct" in f or "prob" in f else 0.0

        game_record = {
            "game_id": game.get("id"),
            "date": commence_dt.strftime("%Y-%m-%d"),
            "commence_time": commence_time_str,
            "sport": sport,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_won,
            **features
        }

        all_game_records.append(game_record)

        # Update history for next game
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        team_history[home_team].append((home_score, away_score, home_won))
        team_history[away_team].append((away_score, home_score, away_won))

    return all_game_records


def collect_historical_data(sports: List[str], output_file: str = "training_data.csv",
                            api_key: str = None) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"🏈 THE ODDS API HISTORICAL DATA COLLECTOR (PAID TIER)")
    print(f"{'='*60}")
    print(f"🏀 Sports: {', '.join(sports)}")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")
    
    if not api_key:
        print("❌ Error: ODDS_API_KEY is required.")
        return pd.DataFrame()

    client = OddsAPIClient(api_key=api_key)
    
    all_games = []
    for sport in sports:
        sport_key = SPORT_KEYS.get(sport)
        if not sport_key:
            print(f"  ⚠️ Skipping {sport}: unknown sport key")
            continue

        print(f"\n📊 Collecting {sport}...")
        games = collect_sport_data(sport, sport_key, client)
        all_games.extend(games)
        print(f"  ✅ {len(games)} {sport} games processed")

    if not all_games:
        print("\n❌ No games collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_games)
    
    df = df.sort_values("commence_time")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE: {len(df)} games")
    print(f"📅 {df['date'].min()} to {df['date'].max()}")
    for sport in sports:
        sg = df[df["sport"] == sport]
        if len(sg) > 0:
            print(f"  {sport}: {len(sg)} (Home win: {sg['home_won'].mean():.1%})")
    print(f"Overall home win: {df['home_won'].mean():.1%}")
    print(f"💾 Saved: {output_file}")
    print(f"{'='*60}\n")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Collect historical data using The Odds API (Paid Tier)")
    parser.add_argument("--sports", nargs="+", default=["NBA", "NHL", "NCAAB"],
                       choices=["NBA", "NHL", "NCAAB", "NFL", "NCAAF"])
    parser.add_argument("--output", type=str, default="data/master_all_sports.csv")
    parser.add_argument("--api-key", type=str, default=ODDS_API_KEY)
    
    args = parser.parse_args()
    
    df = collect_historical_data(args.sports, args.output, args.api_key)
    
if __name__ == "__main__":
    main()
