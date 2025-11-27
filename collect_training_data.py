#!/usr/bin/env python3
"""
Historical Data Collector for Sports Betting ML Training

Collects completed games with outcomes from multiple APIs to build training dataset.
Supports: NBA, NHL, NCAAB, NCAAF, NFL

Usage:
    python collect_training_data.py --days 30 --sports NBA NHL NCAAB
    python collect_training_data.py --start-date 2024-01-01 --end-date 2024-03-31
    
Requirements:
    pip install requests pandas python-dateutil
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dateutil import parser as date_parser

# ============================================================
# CONFIGURATION
# ============================================================

# API Keys - set via environment variables or directly here
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORTSDATA_API_KEY = os.environ.get("SPORTSDATA_API_KEY", "")
API_SPORTS_KEY = os.environ.get("API_SPORTS_KEY", "")

# Sport configurations
SPORT_CONFIGS = {
    "NBA": {
        "odds_api_key": "basketball_nba",
        "sportsdata_base": "https://api.sportsdata.io/v3/nba",
        "api_sports_league": 12,  # NBA league ID
    },
    "NHL": {
        "odds_api_key": "icehockey_nhl",
        "sportsdata_base": "https://api.sportsdata.io/v3/nhl",
        "api_sports_league": 57,  # NHL league ID
    },
    "NCAAB": {
        "odds_api_key": "basketball_ncaab",
        "sportsdata_base": "https://api.sportsdata.io/v3/cbb",
        "api_sports_league": 116,  # NCAA Basketball
    },
    "NCAAF": {
        "odds_api_key": "americanfootball_ncaaf",
        "sportsdata_base": "https://api.sportsdata.io/v3/cfb",
        "api_sports_league": 1,  # NCAA Football
    },
    "NFL": {
        "odds_api_key": "americanfootball_nfl",
        "sportsdata_base": "https://api.sportsdata.io/v3/nfl",
        "api_sports_league": 1,  # NFL
    },
}

# Feature names for training (27 features)
FEATURE_NAMES = [
    "home_win_pct",
    "away_win_pct",
    "home_avg_points",
    "away_avg_points",
    "home_def_rating",
    "away_def_rating",
    "spread_normalized",
    "home_last_5",
    "away_last_5",
    "home_home_record",
    "away_away_record",
    "head_to_head",
    "rest_advantage",
    "injuries_impact",
    "weather_factor",
    "public_betting_pct",
    "sharp_money_indicator",
    "line_movement",
    "total_movement",
    "model_consensus",
    "theover_probability",
    "implied_home_prob",
    "home_streak",
    "away_streak",
    "division_game",
    "back_to_back",
    "primetime_game",
]


# ============================================================
# API CLIENTS
# ============================================================

class OddsAPIClient:
    """Client for The Odds API"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_historical_odds(self, sport: str, date: str) -> List[Dict]:
        """Get historical odds for a specific date"""
        # Note: Historical odds requires paid plan
        url = f"{self.BASE_URL}/historical/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "date": date,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get("data", [])
            elif response.status_code == 422:
                # Historical endpoint not available, try scores
                return []
            else:
                print(f"Odds API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"Odds API exception: {e}")
            return []
    
    def get_scores(self, sport: str, days_from: int = 3) -> List[Dict]:
        """Get recent scores (completed games)"""
        url = f"{self.BASE_URL}/sports/{sport}/scores"
        params = {
            "apiKey": self.api_key,
            "daysFrom": days_from,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Scores API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"Scores API exception: {e}")
            return []


class SportsDataIOClient:
    """Client for SportsData.io API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_games_by_date(self, sport: str, date: str) -> List[Dict]:
        """Get games for a specific date"""
        config = SPORT_CONFIGS.get(sport, {})
        base_url = config.get("sportsdata_base", "")
        
        if not base_url:
            return []
        
        url = f"{base_url}/scores/json/GamesByDate/{date}"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SportsData.io error for {sport}: {response.status_code}")
                return []
        except Exception as e:
            print(f"SportsData.io exception: {e}")
            return []
    
    def get_team_stats(self, sport: str, season: str) -> Dict[str, Dict]:
        """Get team season stats"""
        config = SPORT_CONFIGS.get(sport, {})
        base_url = config.get("sportsdata_base", "")
        
        if not base_url:
            return {}
        
        url = f"{base_url}/scores/json/TeamSeasonStats/{season}"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                stats = response.json()
                # Index by team name/key
                return {s.get("Team", s.get("Name", "")): s for s in stats}
            else:
                return {}
        except Exception as e:
            print(f"Team stats exception: {e}")
            return {}
    
    def get_standings(self, sport: str, season: str) -> Dict[str, Dict]:
        """Get team standings"""
        config = SPORT_CONFIGS.get(sport, {})
        base_url = config.get("sportsdata_base", "")
        
        if not base_url:
            return {}
        
        url = f"{base_url}/scores/json/Standings/{season}"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                standings = response.json()
                return {s.get("Team", s.get("Name", "")): s for s in standings}
            else:
                return {}
        except Exception as e:
            print(f"Standings exception: {e}")
            return {}


class APISportsClient:
    """Client for API-Sports"""
    
    BASE_URLS = {
        "NBA": "https://v1.basketball.api-sports.io",
        "NHL": "https://v1.hockey.api-sports.io",
        "NCAAB": "https://v1.basketball.api-sports.io",
        "NFL": "https://v1.american-football.api-sports.io",
        "NCAAF": "https://v1.american-football.api-sports.io",
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_games(self, sport: str, date: str) -> List[Dict]:
        """Get games for a specific date"""
        base_url = self.BASE_URLS.get(sport, "")
        config = SPORT_CONFIGS.get(sport, {})
        league_id = config.get("api_sports_league", 0)
        
        if not base_url or not league_id:
            return []
        
        url = f"{base_url}/games"
        headers = {"x-apisports-key": self.api_key}
        params = {"date": date, "league": league_id}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", [])
            else:
                return []
        except Exception as e:
            print(f"API-Sports exception: {e}")
            return []


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_win_percentage(wins: int, losses: int) -> float:
    """Calculate win percentage"""
    total = wins + losses
    if total == 0:
        return 0.5
    return wins / total


def calculate_implied_probability(odds: int) -> float:
    """Convert American odds to implied probability"""
    if odds is None or odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def normalize_spread(spread: float, sport: str) -> float:
    """Normalize spread to 0-1 range based on sport"""
    max_spreads = {
        "NBA": 20,
        "NHL": 3,
        "NCAAB": 35,
        "NCAAF": 40,
        "NFL": 20,
    }
    max_spread = max_spreads.get(sport, 20)
    normalized = (spread + max_spread) / (2 * max_spread)
    return np.clip(normalized, 0, 1)


def extract_features(
    game: Dict,
    home_stats: Dict,
    away_stats: Dict,
    odds_data: Dict,
    sport: str
) -> Dict:
    """Extract all 27 features for a game"""
    
    features = {}
    
    # Basic win percentages
    home_wins = home_stats.get("Wins", home_stats.get("wins", 0)) or 0
    home_losses = home_stats.get("Losses", home_stats.get("losses", 0)) or 0
    away_wins = away_stats.get("Wins", away_stats.get("wins", 0)) or 0
    away_losses = away_stats.get("Losses", away_stats.get("losses", 0)) or 0
    
    features["home_win_pct"] = calculate_win_percentage(home_wins, home_losses)
    features["away_win_pct"] = calculate_win_percentage(away_wins, away_losses)
    
    # Points/Goals
    if sport in ["NBA", "NCAAB"]:
        features["home_avg_points"] = home_stats.get("PointsPerGame", home_stats.get("points", {}).get("for", {}).get("average", {}).get("all", 100)) or 100
        features["away_avg_points"] = away_stats.get("PointsPerGame", away_stats.get("points", {}).get("for", {}).get("average", {}).get("all", 100)) or 100
        features["home_def_rating"] = home_stats.get("OpponentPointsPerGame", 100) or 100
        features["away_def_rating"] = away_stats.get("OpponentPointsPerGame", 100) or 100
    elif sport == "NHL":
        features["home_avg_points"] = home_stats.get("GoalsPerGame", 3.0) or 3.0
        features["away_avg_points"] = away_stats.get("GoalsPerGame", 3.0) or 3.0
        features["home_def_rating"] = home_stats.get("GoalsAgainstPerGame", 3.0) or 3.0
        features["away_def_rating"] = away_stats.get("GoalsAgainstPerGame", 3.0) or 3.0
    else:  # Football
        features["home_avg_points"] = home_stats.get("PointsPerGame", 24) or 24
        features["away_avg_points"] = away_stats.get("PointsPerGame", 24) or 24
        features["home_def_rating"] = home_stats.get("OpponentPointsPerGame", 24) or 24
        features["away_def_rating"] = away_stats.get("OpponentPointsPerGame", 24) or 24
    
    # Spread
    spread = odds_data.get("spread", 0) or 0
    features["spread_normalized"] = normalize_spread(spread, sport)
    
    # Last 5 games (default to win pct if not available)
    features["home_last_5"] = home_stats.get("LastFiveWinPct", features["home_win_pct"])
    features["away_last_5"] = away_stats.get("LastFiveWinPct", features["away_win_pct"])
    
    # Home/Away records
    home_home_wins = home_stats.get("HomeWins", home_stats.get("home", {}).get("win", home_wins // 2)) or 0
    home_home_losses = home_stats.get("HomeLosses", home_stats.get("home", {}).get("lose", home_losses // 2)) or 0
    away_away_wins = away_stats.get("AwayWins", away_stats.get("away", {}).get("win", away_wins // 2)) or 0
    away_away_losses = away_stats.get("AwayLosses", away_stats.get("away", {}).get("lose", away_losses // 2)) or 0
    
    features["home_home_record"] = calculate_win_percentage(home_home_wins, home_home_losses)
    features["away_away_record"] = calculate_win_percentage(away_away_wins, away_away_losses)
    
    # Head to head (default to 0.5 if not available)
    features["head_to_head"] = 0.5
    
    # Rest advantage (default to 0)
    features["rest_advantage"] = game.get("rest_advantage", 0) or 0
    
    # Injuries (default to 0)
    features["injuries_impact"] = 0
    
    # Weather (only for outdoor sports)
    features["weather_factor"] = 0 if sport in ["NBA", "NHL", "NCAAB"] else game.get("weather_impact", 0)
    
    # Betting metrics
    features["public_betting_pct"] = odds_data.get("public_pct", 0.5) or 0.5
    features["sharp_money_indicator"] = odds_data.get("sharp_indicator", 0) or 0
    features["line_movement"] = odds_data.get("line_movement", 0) or 0
    features["total_movement"] = odds_data.get("total_movement", 0) or 0
    features["model_consensus"] = 0.5
    
    # TheOver probability (if available)
    features["theover_probability"] = odds_data.get("theover_prob", 0.5) or 0.5
    
    # Implied probability from moneyline
    home_ml = odds_data.get("home_ml", -110) or -110
    features["implied_home_prob"] = calculate_implied_probability(home_ml)
    
    # Streaks (default to 0)
    features["home_streak"] = home_stats.get("Streak", 0) or 0
    features["away_streak"] = away_stats.get("Streak", 0) or 0
    
    # Game context
    features["division_game"] = 1 if game.get("division_game", False) else 0
    features["back_to_back"] = 1 if game.get("back_to_back", False) else 0
    features["primetime_game"] = 1 if game.get("primetime", False) else 0
    
    return features


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_games_for_date(
    date: str,
    sports: List[str],
    odds_client: OddsAPIClient,
    sportsdata_client: SportsDataIOClient,
    api_sports_client: APISportsClient,
) -> List[Dict]:
    """Collect all completed games for a specific date"""
    
    all_games = []
    
    for sport in sports:
        print(f"  Collecting {sport} games for {date}...")
        
        config = SPORT_CONFIGS.get(sport, {})
        games = []
        team_stats = {}
        
        # Try SportsData.io first (most comprehensive)
        if SPORTSDATA_API_KEY:
            sportsdata_games = sportsdata_client.get_games_by_date(sport, date)
            
            # Get team stats for the season
            season = datetime.strptime(date, "%Y-%m-%d").year
            if sport in ["NBA", "NHL", "NCAAB"]:
                # NBA/NHL seasons span two years
                month = datetime.strptime(date, "%Y-%m-%d").month
                if month < 7:  # Before July, use previous year's season
                    season = season - 1
            team_stats = sportsdata_client.get_team_stats(sport, str(season))
            
            for game in sportsdata_games:
                # Only include completed games
                status = game.get("Status", "")
                if status not in ["Final", "F", "F/OT", "Closed"]:
                    continue
                
                home_team = game.get("HomeTeam", "")
                away_team = game.get("AwayTeam", "")
                home_score = game.get("HomeTeamScore", game.get("HomeScore", 0)) or 0
                away_score = game.get("AwayTeamScore", game.get("AwayScore", 0)) or 0
                
                if not home_team or not away_team:
                    continue
                
                games.append({
                    "game_id": game.get("GameID", game.get("GameId", f"{date}_{home_team}_{away_team}")),
                    "date": date,
                    "sport": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_won": 1 if home_score > away_score else 0,
                    "spread": game.get("PointSpread", 0),
                    "total": game.get("OverUnder", 0),
                    "status": status,
                })
        
        # Fallback to API-Sports
        elif API_SPORTS_KEY:
            api_sports_games = api_sports_client.get_games(sport, date)
            
            for game in api_sports_games:
                status = game.get("status", {}).get("long", "")
                if status not in ["Finished", "After Over Time", "Game Finished"]:
                    continue
                
                home_team = game.get("teams", {}).get("home", {}).get("name", "")
                away_team = game.get("teams", {}).get("away", {}).get("name", "")
                home_score = game.get("scores", {}).get("home", {}).get("total", 0) or 0
                away_score = game.get("scores", {}).get("away", {}).get("total", 0) or 0
                
                if not home_team or not away_team:
                    continue
                
                games.append({
                    "game_id": game.get("id", f"{date}_{home_team}_{away_team}"),
                    "date": date,
                    "sport": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_won": 1 if home_score > away_score else 0,
                    "spread": 0,
                    "total": 0,
                    "status": status,
                })
        
        # Try to get odds data for each game
        odds_key = config.get("odds_api_key", "")
        if ODDS_API_KEY and odds_key:
            scores = odds_client.get_scores(odds_key, days_from=7)
            odds_lookup = {}
            for score in scores:
                key = f"{score.get('away_team', '')}_{score.get('home_team', '')}"
                odds_lookup[key] = score
        else:
            odds_lookup = {}
        
        # Extract features for each game
        for game in games:
            home_team = game["home_team"]
            away_team = game["away_team"]
            
            # Get team stats
            home_stats = team_stats.get(home_team, {})
            away_stats = team_stats.get(away_team, {})
            
            # Get odds data
            odds_key = f"{away_team}_{home_team}"
            odds_data = odds_lookup.get(odds_key, {"spread": game.get("spread", 0)})
            
            # Extract features
            features = extract_features(game, home_stats, away_stats, odds_data, sport)
            
            # Combine game info with features
            game_record = {
                "game_id": game["game_id"],
                "date": game["date"],
                "sport": game["sport"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "home_won": game["home_won"],
                **features,
            }
            
            all_games.append(game_record)
        
        print(f"    Found {len(games)} completed {sport} games")
        time.sleep(0.5)  # Rate limiting
    
    return all_games


def collect_historical_data(
    start_date: datetime,
    end_date: datetime,
    sports: List[str],
    output_file: str = "training_data.csv",
) -> pd.DataFrame:
    """Collect historical data for a date range"""
    
    print(f"\n{'='*60}")
    print(f"COLLECTING HISTORICAL DATA")
    print(f"{'='*60}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Sports: {', '.join(sports)}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Initialize clients
    odds_client = OddsAPIClient(ODDS_API_KEY) if ODDS_API_KEY else None
    sportsdata_client = SportsDataIOClient(SPORTSDATA_API_KEY) if SPORTSDATA_API_KEY else None
    api_sports_client = APISportsClient(API_SPORTS_KEY) if API_SPORTS_KEY else None
    
    if not any([ODDS_API_KEY, SPORTSDATA_API_KEY, API_SPORTS_KEY]):
        print("ERROR: No API keys configured!")
        print("Set at least one of:")
        print("  - ODDS_API_KEY")
        print("  - SPORTSDATA_API_KEY")
        print("  - API_SPORTS_KEY")
        return pd.DataFrame()
    
    all_games = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Processing {date_str}...")
        
        games = collect_games_for_date(
            date_str,
            sports,
            odds_client,
            sportsdata_client,
            api_sports_client,
        )
        
        all_games.extend(games)
        current_date += timedelta(days=1)
        
        # Progress update
        if len(all_games) % 100 == 0 and len(all_games) > 0:
            print(f"  Total games collected: {len(all_games)}")
    
    # Create DataFrame
    if all_games:
        df = pd.DataFrame(all_games)
        
        # Ensure all feature columns exist
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                df[feature] = 0.5 if "pct" in feature or "prob" in feature else 0
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total games: {len(df)}")
        print(f"By sport:")
        for sport in sports:
            count = len(df[df["sport"] == sport])
            print(f"  {sport}: {count}")
        print(f"Home win rate: {df['home_won'].mean():.1%}")
        print(f"Saved to: {output_file}")
        print(f"{'='*60}\n")
        
        return df
    else:
        print("No games collected!")
        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collect historical sports data for ML training")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back (default: 30)")
    parser.add_argument("--sports", nargs="+", default=["NBA", "NHL", "NCAAB"], 
                       help="Sports to collect (default: NBA NHL NCAAB)")
    parser.add_argument("--output", type=str, default="training_data.csv", help="Output file")
    parser.add_argument("--odds-key", type=str, help="The Odds API key")
    parser.add_argument("--sportsdata-key", type=str, help="SportsData.io API key")
    parser.add_argument("--api-sports-key", type=str, help="API-Sports key")
    
    args = parser.parse_args()
    
    # Set API keys from arguments
    global ODDS_API_KEY, SPORTSDATA_API_KEY, API_SPORTS_KEY
    if args.odds_key:
        ODDS_API_KEY = args.odds_key
    if args.sportsdata_key:
        SPORTSDATA_API_KEY = args.sportsdata_key
    if args.api_sports_key:
        API_SPORTS_KEY = args.api_sports_key
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=args.days)
    
    # Validate sports
    valid_sports = list(SPORT_CONFIGS.keys())
    sports = [s.upper() for s in args.sports]
    for sport in sports:
        if sport not in valid_sports:
            print(f"Warning: {sport} not supported. Valid options: {valid_sports}")
            sports.remove(sport)
    
    if not sports:
        print("No valid sports specified!")
        return
    
    # Collect data
    df = collect_historical_data(
        start_date=start_date,
        end_date=end_date,
        sports=sports,
        output_file=args.output,
    )
    
    if len(df) > 0:
        print("\nNext steps:")
        print("1. Review the data in the CSV")
        print("2. Train a model:")
        print(f"   python train_vertex_model.py --data {args.output} --project-id YOUR_PROJECT_ID --deploy")


if __name__ == "__main__":
    main()
