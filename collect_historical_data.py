"""
Collect Historical Game Data for ML Training using The Odds API
Provides true closing lines and calculated rolling averages.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time
import argparse
from typing import Dict, List, Optional, Any
from config import DATA_DIR, THE_ODDS_API_KEY

# ============================================================
# UTILS
# ============================================================
def get_historical_snapshot_timestamp(commence_time_str: str, lookback_hours: int = 1) -> str:
    """
    Calculates the ISO8601 timestamp for a historical odds snapshot.
    """
    try:
        # 1. Parse the commence time into a UTC-aware Timestamp
        dt = pd.to_datetime(commence_time_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')

        # 2. Subtract the lookback window (the 'Closing Line' window)
        snapshot_dt = dt - timedelta(hours=lookback_hours)

        # 3. Return formatted string with 'Z' suffix required by The Odds API
        return snapshot_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Error formatting snapshot timestamp: {e}")
        return commence_time_str


def normalize_points(points: float, sport: str) -> float:
    ranges = {
        "NBA": (90, 130), "NCAAB": (50, 90), "NHL": (1.5, 4.5),
        "NFL": (14, 35), "NCAAF": (14, 45),
    }
    min_pts, max_pts = ranges.get(sport.upper(), (0, 100))
    if pd.isna(points) or points is None:
        return 0.5
    normalized = (float(points) - min_pts) / (max_pts - min_pts)
    return float(np.clip(normalized, 0.0, 1.0))

# Feature names for ML training
FEATURE_NAMES = [
    "home_win_pct", "away_win_pct", "home_ppg", "away_ppg",
    "home_oppg", "away_oppg", "spread_normalized",
    "home_last_5", "away_last_5", "home_home_record", "away_away_record",
    "head_to_head", "rest_advantage", "injuries_impact", "weather_factor",
    "public_betting_pct", "sharp_money_indicator", "line_movement",
    "total_movement", "model_consensus", "theover_probability",
    "implied_home_prob", "kalshi_prob", "home_streak", "away_streak", "division_game",
    "back_to_back", "primetime_game",
]

# The Odds API sports keys
ODDS_API_SPORTS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NHL": "icehockey_nhl",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf"
}


# ============================================================
# THE ODDS API CLIENT
# ============================================================
class TheOddsAPIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or THE_ODDS_API_KEY
        if not self.api_key:
            print("⚠️ WARNING: THE_ODDS_API_KEY is not set.")
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self._historical_cache = {}

    def get_scores(self, sport: str, days_from: int) -> List[Dict]:
        """Fetch scores from the last X days"""
        odds_sport = ODDS_API_SPORTS.get(sport)
        if not odds_sport:
            print(f"  ⚠️ Unknown sport: {sport}")
            return []
        
        url = f"{self.base_url}/{odds_sport}/scores"
        params = {
            "apiKey": self.api_key,
            "daysFrom": days_from
        }
        try:
            print(f"    Fetching scores for {sport} (last {days_from} days)...")
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                 print("    ⚠️ Rate limited, waiting 60s...")
                 time.sleep(60)
                 return self.get_scores(sport, days_from)
            else:
                print(f"    ⚠️ API error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"    ⚠️ Exception fetching scores: {e}")
            return []

    def get_historical_odds(self, sport: str, snapshot_date: str) -> List[Dict]:
        """Fetch historical odds for a specific snapshot date."""
        # Cache based on timestamp to conserve quota
        cache_key = f"{sport}_{snapshot_date}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        odds_sport = ODDS_API_SPORTS.get(sport)
        if not odds_sport:
            return []
            
        url = f"https://api.the-odds-api.com/v4/historical/sports/{odds_sport}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads",
            "date": snapshot_date
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json().get('data', [])
                self._historical_cache[cache_key] = data
                return data
            elif response.status_code == 429:
                print("    ⚠️ Rate limited on historical, waiting 60s...")
                time.sleep(60)
                return self.get_historical_odds(sport, snapshot_date)
            else:
                print(f"    ⚠️ Historical API error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"    ⚠️ Exception fetching historical odds: {e}")
            return []


# ============================================================
# DATA COLLECTION
# ============================================================

def calculate_implied_prob(price: float) -> float:
    """Calculate implied probability from American or Decimal odds."""
    if price is None:
         return 0.5
    if price == 0:
        return 0.5
    if abs(price) >= 100:
        # American
        if price > 0:
            return 100.0 / (price + 100.0)
        else:
            return abs(price) / (abs(price) + 100.0)
    elif 1.0 < price < 100.0:
        # Decimal
        return 1.0 / price
    elif 0.0 < price <= 1.0:
        return price
    return 0.5


def collect_historical_data(days: int, sports: List[str], output_file: str = "master_all_sports.csv", api_key: str = None) -> pd.DataFrame:
    client = TheOddsAPIClient(api_key=api_key)
    all_games = []

    print(f"\n{'='*60}")
    print(f"🏈 THE ODDS API HISTORICAL DATA COLLECTOR (PAID TIER)")
    print(f"{'='*60}")
    print(f"📅 Lookback: Last {days} days")
    print(f"🏀 Sports: {', '.join(sports)}")

    # Use config data directory if simple filename provided
    if "/" not in output_file and "\\" not in output_file:
        output_file = str(DATA_DIR / output_file)
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")

    for sport in sports:
        print(f"\n📊 Collecting {sport}...")
        
        # 1. Fetch scores
        scores_data = client.get_scores(sport, days_from=days)
        if not scores_data:
             continue

        # Filter for completed games only
        completed_games = [g for g in scores_data if g.get('completed') == True]
        # Sort chronologically to correctly calculate rolling averages
        completed_games.sort(key=lambda x: x.get('commence_time', ''))
        print(f"    Found {len(completed_games)} completed games.")
        
        # Tracking for rolling averages
        # team_name -> list of recent dicts {'pts': scored, 'opp_pts': allowed, 'win': 1/0}
        team_history = {}
        
        valid_games = 0
        
        for game in completed_games:
            game_id = game.get('id')
            commence_time = game.get('commence_time')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            scores = game.get('scores')
            if not scores:
                continue

            home_score = next((s['score'] for s in scores if s['name'] == home_team), None)
            away_score = next((s['score'] for s in scores if s['name'] == away_team), None)
            
            if home_score is None or away_score is None:
                 continue

            try:
                home_score = float(home_score)
                away_score = float(away_score)
            except ValueError:
                continue

            home_won = 1 if home_score > away_score else 0

            # --- Calculate 5-game rolling averages BEFORE updating with this game's result ---
            h_hist = team_history.get(home_team, [])[-5:]
            a_hist = team_history.get(away_team, [])[-5:]
            
            h_win_pct = sum(x['win'] for x in h_hist) / len(h_hist) if h_hist else 0.5
            a_win_pct = sum(x['win'] for x in a_hist) / len(a_hist) if a_hist else 0.5
            
            # Use appropriate default PPG if no history
            default_pts = {"NBA": 110, "NCAAB": 70, "NHL": 3.0, "NFL": 24, "NCAAF": 24}.get(sport, 50)
            
            h_ppg_raw = sum(x['pts'] for x in h_hist) / len(h_hist) if h_hist else default_pts
            h_oppg_raw = sum(x['opp_pts'] for x in h_hist) / len(h_hist) if h_hist else default_pts
            a_ppg_raw = sum(x['pts'] for x in a_hist) / len(a_hist) if a_hist else default_pts
            a_oppg_raw = sum(x['opp_pts'] for x in a_hist) / len(a_hist) if a_hist else default_pts
            
            # --- Fetch 1-hour prior Closing Odds ---
            snapshot_time = get_historical_snapshot_timestamp(commence_time, lookback_hours=1)
            hist_odds_data = client.get_historical_odds(sport, snapshot_time)
            
            implied_home_prob = None
            market_price = None
            
            # Find this game in the snapshot
            game_odds = next((g for g in hist_odds_data if g.get('id') == game_id or (g.get('home_team') == home_team and g.get('away_team') == away_team)), None)
            
            if game_odds and game_odds.get('bookmakers'):
                # Try to find a primary bookmaker for H2H
                bookmakers = game_odds['bookmakers']
                # Prefer draftkings or fanduel if available
                preferred = next((b for b in bookmakers if b['key'] in ['draftkings', 'fanduel', 'betmgm', 'pinnacle']), bookmakers[0])

                h2h_market = next((m for m in preferred.get('markets', []) if m['key'] == 'h2h'), None)
                if h2h_market and h2h_market.get('outcomes'):
                    outcomes = h2h_market['outcomes']
                    home_outcome = next((o for o in outcomes if o['name'] == home_team), None)
                    if home_outcome:
                        market_price = home_outcome.get('price')
                        implied_home_prob = calculate_implied_prob(market_price)

            # Fallbacks for implied prob
            if implied_home_prob is None:
                implied_home_prob = max(0.35, min(0.65, h_win_pct))

            # --- Build Record ---
            game_record = {
                "game_id": game_id,
                "sport": sport,
                "league": sport,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_won": home_won,

                # Features
                "home_win_pct": h_win_pct,
                "away_win_pct": a_win_pct,
                "home_ppg": normalize_points(h_ppg_raw, sport),
                "away_ppg": normalize_points(a_ppg_raw, sport),
                "home_oppg": normalize_points(h_oppg_raw, sport),
                "away_oppg": normalize_points(a_oppg_raw, sport),
                "implied_home_prob": implied_home_prob,
                "kalshi_prob": implied_home_prob, # Proxy if actual Kalshi not available
                "home_price": market_price,
            }
            
            # Fill defaults for remaining features
            for feature in FEATURE_NAMES:
                 if feature not in game_record:
                     game_record[feature] = 0.5 if "pct" in feature or "prob" in feature else 0.0

            all_games.append(game_record)
            valid_games += 1

            # --- Update Team History for NEXT games ---
            if home_team not in team_history:
                team_history[home_team] = []
            if away_team not in team_history:
                team_history[away_team] = []

            team_history[home_team].append({'pts': home_score, 'opp_pts': away_score, 'win': home_won})
            team_history[away_team].append({'pts': away_score, 'opp_pts': home_score, 'win': 1 - home_won})

        print(f"    ✅ {valid_games} valid historical records built for {sport}")

    if not all_games:
        print("\n❌ No games collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_games)
    
    # Sort and save
    df = df.sort_values("commence_time")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE: {len(df)} games")
    print(f"📅 {df['commence_time'].min()} to {df['commence_time'].max()}")
    print(f"💾 Saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect historical data using The Odds API")
    parser.add_argument("--days", type=int, default=45, help="Number of days to look back (default 45)")
    parser.add_argument("--sports", nargs="+", default=["NBA", "NHL", "NCAAB"],
                       choices=["NBA", "NHL", "NCAAB", "NFL", "NCAAF"])
    parser.add_argument("--output", type=str, default="master_all_sports.csv")
    parser.add_argument("--api-key", type=str)
    
    args = parser.parse_args()
    
    collect_historical_data(days=args.days, sports=args.sports, output_file=args.output, api_key=args.api_key)


def run_backfill(sports: List[str], days: int = 2):
    from config import DATA_DIR
    output_path = str(DATA_DIR / "master_all_sports.csv")
    return collect_historical_data(days=days, sports=sports, output_file=output_path)


if __name__ == "__main__":
    main()
