"""
Collect Historical Game Data for ML Training
Uses API-Sports for NBA, NHL, NCAAB, NCAAF, NFL

API-Sports endpoints:
- Basketball (NBA/NCAAB): https://v1.basketball.api-sports.io
- Hockey (NHL): https://v1.hockey.api-sports.io
- American Football (NFL/NCAAF): https://v1.american-football.api-sports.io

Usage:
    python collect_historical_data.py --days 30 --sports NBA NHL NCAAB
    python collect_historical_data.py --season 2024 --sports NBA NHL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time
import argparse

import asyncio
from typing import Dict, List, Optional, Any

# ============================================================
# API-SPORTS CONFIGURATION
# ============================================================

# Your API-Sports key (same key works for all sports)
API_SPORTS_KEY = os.environ.get("API_SPORTS_KEY", "07972c891e3d56fbc6298b5c2a07b152")

# API-Sports Base URLs
API_SPORTS_URLS = {
    "NBA": "https://v1.basketball.api-sports.io",
    "NCAAB": "https://v1.basketball.api-sports.io",
    "NHL": "https://v1.hockey.api-sports.io",
    "NFL": "https://v1.american-football.api-sports.io",
    "NCAAF": "https://v1.american-football.api-sports.io",
}

# League IDs for API-Sports
LEAGUE_IDS = {
    "NBA": 12,      # NBA
    "NCAAB": 116,   # NCAA Basketball
    "NHL": 57,      # NHL
    "NFL": 1,       # NFL
    "NCAAF": 2,     # NCAA Football
}

# Feature names for ML training (27 features)
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
# API-SPORTS CLIENT
# ============================================================

class APISportsClient:
    """Client for API-Sports - Works for all sports with same key"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_SPORTS_KEY
        self.headers = {
            "x-apisports-key": self.api_key,
        }
        self._team_stats_cache = {}
        self._standings_cache = {}
    
    def _get_base_url(self, sport: str) -> str:
        return API_SPORTS_URLS.get(sport, "")
    
    def _make_request(self, sport: str, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Make API request with error handling"""
        base_url = self._get_base_url(sport)
        if not base_url:
            print(f"  ⚠️ Unknown sport: {sport}")
            return None
        
        url = f"{base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API errors
                errors = data.get("errors", [])
                if errors and len(errors) > 0:
                    if isinstance(errors, dict) and errors:
                        print(f"  ⚠️ API error: {errors}")
                        return None
                
                return data.get("response", [])
            
            elif response.status_code == 429:
                print(f"  ⚠️ Rate limited, waiting 60s...")
                time.sleep(60)
                return self._make_request(sport, endpoint, params)
            
            else:
                print(f"  ⚠️ API error {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout for {endpoint}")
            return None
        except Exception as e:
            print(f"  ⚠️ Exception: {e}")
            return None
    
    def get_games_by_season(self, sport: str, season: str, league_id: int) -> List[Dict]:
        """Get all games for a season"""
        params = {"season": season, "league": league_id}
        return self._make_request(sport, "games", params) or []
    
    def get_team_statistics(self, sport: str, team_id: int, season: str, league_id: int) -> Dict:
        """Get team statistics for a season"""
        cache_key = f"{sport}_{team_id}_{season}"
        if cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]
        
        params = {"team": team_id, "season": season, "league": league_id}
        result = self._make_request(sport, "teams/statistics", params)
        
        if result:
            stats = result[0] if isinstance(result, list) and len(result) > 0 else result
            self._team_stats_cache[cache_key] = stats
            return stats
        return {}
    
    def get_standings(self, sport: str, season: str, league_id: int) -> Dict[int, Dict]:
        """Get standings for a season, indexed by team ID"""
        cache_key = f"{sport}_{season}_{league_id}"
        if cache_key in self._standings_cache:
            return self._standings_cache[cache_key]
        
        params = {"season": season, "league": league_id}
        result = self._make_request(sport, "standings", params)
        
        if not result:
            return {}
        
        indexed = {}
        if isinstance(result, list):
            for item in result:
                if isinstance(item, list):
                    for team_standing in item:
                        team_id = team_standing.get("team", {}).get("id")
                        if team_id:
                            indexed[team_id] = team_standing
                elif isinstance(item, dict):
                    team_id = item.get("team", {}).get("id")
                    if team_id:
                        indexed[team_id] = item
        
        self._standings_cache[cache_key] = indexed
        return indexed


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_win_pct(wins: int, losses: int, default: float = 0.5) -> float:
    total = (wins or 0) + (losses or 0)
    if total == 0:
        return default
    return (wins or 0) / total


def normalize_spread(spread: float, sport: str) -> float:
    max_spreads = {"NBA": 25, "NHL": 3.5, "NCAAB": 40, "NFL": 20, "NCAAF": 40}
    max_spread = max_spreads.get(sport, 20)
    if spread is None:
        return 0.5
    normalized = (float(spread) + max_spread) / (2 * max_spread)
    return np.clip(normalized, 0, 1)


def normalize_points(points: float, sport: str) -> float:
    ranges = {
        "NBA": (90, 130), "NCAAB": (50, 90), "NHL": (1.5, 4.5),
        "NFL": (14, 35), "NCAAF": (14, 45),
    }
    min_pts, max_pts = ranges.get(sport, (0, 100))
    if points is None:
        return 0.5
    normalized = (float(points) - min_pts) / (max_pts - min_pts)
    return np.clip(normalized, 0, 1)


def extract_game_features(game: Dict, home_stats: Dict, away_stats: Dict,
                          home_standing: Dict, away_standing: Dict, sport: str) -> Dict:
    """Extract all 27 features for a game"""
    features = {}
    
    # Win Percentages
    home_games = home_standing.get("games", {})
    away_games = away_standing.get("games", {})
    
    if sport in ["NBA", "NCAAB"]:
        home_wins = home_games.get("win", {}).get("total", 0) or 0
        home_losses = home_games.get("lose", {}).get("total", 0) or 0
        away_wins = away_games.get("win", {}).get("total", 0) or 0
        away_losses = away_games.get("lose", {}).get("total", 0) or 0
    else:
        home_wins = home_standing.get("won", 0) or 0
        home_losses = home_standing.get("lost", 0) or 0
        away_wins = away_standing.get("won", 0) or 0
        away_losses = away_standing.get("lost", 0) or 0
    
    features["home_win_pct"] = calculate_win_pct(home_wins, home_losses)
    features["away_win_pct"] = calculate_win_pct(away_wins, away_losses)
    
    # Scoring Stats
    if sport in ["NBA", "NCAAB"]:
        home_pts = home_stats.get("points", {}).get("for", {}).get("average", {}).get("all", 100) or 100
        away_pts = away_stats.get("points", {}).get("for", {}).get("average", {}).get("all", 100) or 100
        home_def = home_stats.get("points", {}).get("against", {}).get("average", {}).get("all", 100) or 100
        away_def = away_stats.get("points", {}).get("against", {}).get("average", {}).get("all", 100) or 100
    elif sport == "NHL":
        home_pts = home_stats.get("goals", {}).get("for", 3.0) or 3.0
        away_pts = away_stats.get("goals", {}).get("for", 3.0) or 3.0
        home_def = home_stats.get("goals", {}).get("against", 3.0) or 3.0
        away_def = away_stats.get("goals", {}).get("against", 3.0) or 3.0
    else:
        home_pts = 24
        away_pts = 24
        home_def = 24
        away_def = 24
    
    features["home_ppg"] = normalize_points(home_pts, sport)
    features["away_ppg"] = normalize_points(away_pts, sport)
    features["home_oppg"] = normalize_points(home_def, sport)
    features["away_oppg"] = normalize_points(away_def, sport)
    
    # Spread (default)
    features["spread_normalized"] = 0.5
    
    # Last 5 / Form
    if sport in ["NBA", "NCAAB"]:
        home_form = home_standing.get("form", "") or ""
        away_form = away_standing.get("form", "") or ""
        features["home_last_5"] = home_form[-5:].count("W") / 5 if home_form else features["home_win_pct"]
        features["away_last_5"] = away_form[-5:].count("W") / 5 if away_form else features["away_win_pct"]
    else:
        features["home_last_5"] = features["home_win_pct"]
        features["away_last_5"] = features["away_win_pct"]
    
    # Home/Away Records
    if sport in ["NBA", "NCAAB"]:
        h_h_w = home_games.get("win", {}).get("home", 0) or 0
        h_h_l = home_games.get("lose", {}).get("home", 0) or 0
        a_a_w = away_games.get("win", {}).get("away", 0) or 0
        a_a_l = away_games.get("lose", {}).get("away", 0) or 0
    else:
        h_h_w = home_wins // 2
        h_h_l = home_losses // 2
        a_a_w = away_wins // 2
        a_a_l = away_losses // 2
    
    features["home_home_record"] = calculate_win_pct(h_h_w, h_h_l)
    features["away_away_record"] = calculate_win_pct(a_a_w, a_a_l)
    
    # Other features (defaults)
    features["head_to_head"] = 0.5
    features["rest_advantage"] = 0
    features["injuries_impact"] = 0
    features["weather_factor"] = 0
    features["public_betting_pct"] = 0.5
    features["sharp_money_indicator"] = 0
    features["line_movement"] = 0
    features["total_movement"] = 0
    features["model_consensus"] = 0.5
    features["theover_probability"] = features["home_win_pct"]

    # Implied Home Prob Extraction
    implied_prob = None

    # Helper function for inline odds parsing
    def _parse_odds_inline(odds_dict, keys):
        for key in keys:
            if key in odds_dict and odds_dict[key] is not None:
                try:
                    val = float(odds_dict[key])
                    if val != 0.5 and val != 0.0:
                        if abs(val) >= 100:
                            # American Odds
                            return 100.0 / (val + 100.0) if val > 0 else abs(val) / (abs(val) + 100.0)
                        elif 1.0 < val < 100.0:
                            # Decimal Odds
                            return 1.0 / val
                        elif 0.0 < val <= 1.0:
                            # Already implied probability
                            return val
                except (ValueError, TypeError):
                    continue
        return None

    # 1. Try Bookmaker Odds (Implied Prob / Odds) prioritizing market_probability
    implied_prob = _parse_odds_inline(game, ["market_probability", "implied_home_prob", "odds_american", "odds_home", "home_price", "home_odds"])

    # 2. Extract Kalshi Prob
    kalshi_prob = _parse_odds_inline(game, ["kalshi_prob", "kalshi_probability"])

    # Hierarchy: Market Price -> Kalshi Price -> Clamped Win Pct
    base_clamped_pct = max(0.35, min(0.65, features["home_win_pct"]))

    if implied_prob is not None:
        features["implied_home_prob"] = float(implied_prob)
    elif kalshi_prob is not None:
        features["implied_home_prob"] = float(kalshi_prob)
    else:
        features["implied_home_prob"] = base_clamped_pct

    if kalshi_prob is not None:
        features["kalshi_prob"] = float(kalshi_prob)
    elif implied_prob is not None:
        features["kalshi_prob"] = float(implied_prob)
    else:
        features["kalshi_prob"] = base_clamped_pct
    
    # Streaks
    def parse_streak(s):
        if not s:
            return 0
        if s.startswith("W"):
            return int(s[1:]) if len(s) > 1 and s[1:].isdigit() else 1
        elif s.startswith("L"):
            return -int(s[1:]) if len(s) > 1 and s[1:].isdigit() else -1
        return 0
    
    features["home_streak"] = np.clip(parse_streak(home_standing.get("streak", "")) / 10, -1, 1)
    features["away_streak"] = np.clip(parse_streak(away_standing.get("streak", "")) / 10, -1, 1)
    
    # Game Context
    home_group = home_standing.get("group", {}).get("name", "") if isinstance(home_standing.get("group"), dict) else ""
    away_group = away_standing.get("group", {}).get("name", "") if isinstance(away_standing.get("group"), dict) else ""
    features["division_game"] = 1 if home_group and home_group == away_group else 0
    features["back_to_back"] = 0
    features["primetime_game"] = 0
    
    return features


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_sport_data(sport: str, start_date: datetime, end_date: datetime,
                       client: APISportsClient) -> List[Dict]:
    """Collect historical data for a single sport"""
    
    league_id = LEAGUE_IDS.get(sport)
    if not league_id:
        print(f"  ⚠️ Unknown league for {sport}")
        return []
    
    all_games = []
    
    # Determine seasons
    seasons = set()
    current = start_date
    while current <= end_date:
        if sport in ["NBA", "NCAAB", "NHL"]:
            if current.month >= 10:
                seasons.add(f"{current.year}-{current.year + 1}")
            else:
                seasons.add(f"{current.year - 1}-{current.year}")
        else:
            seasons.add(str(current.year))
        current += timedelta(days=30)
    
    print(f"  📅 Seasons: {sorted(seasons)}")
    
    async def fetch_season_data(season: str, sem: asyncio.Semaphore):
        async with sem:
            standings = await asyncio.to_thread(client.get_standings, sport, season, league_id)
            games = await asyncio.to_thread(client.get_games_by_season, sport, season, league_id)
            return season, standings, games

    async def fetch_all_seasons():
        sem = asyncio.Semaphore(5)
        tasks = [fetch_season_data(s, sem) for s in sorted(seasons)]
        return await asyncio.gather(*tasks)

    season_data_results = asyncio.run(fetch_all_seasons())
    season_data_map = {season: (standings, games) for season, standings, games in season_data_results}

    for season in sorted(seasons):
        print(f"  🏀 Loading {sport} {season}...")
        
        standings, games = season_data_map[season]
        print(f"    {len(standings)} teams in standings")
        
        if not games:
            print(f"    No games found")
            continue
        
        print(f"    {len(games)} total games")
        
        completed = 0
        for game in games:
            status = game.get("status", {})
            status_short = status.get("short", "")
            
            if status_short not in ["FT", "AOT", "AET", "AP"]:
                continue
            
            game_date_str = game.get("date", "")
            if not game_date_str:
                continue
            
            try:
                if "T" in game_date_str:
                    game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                else:
                    game_date = datetime.strptime(game_date_str[:10], "%Y-%m-%d")
            except:
                continue
            
            if game_date.replace(tzinfo=None) < start_date or game_date.replace(tzinfo=None) > end_date:
                continue
            
            teams = game.get("teams", {})
            home_data = teams.get("home", {})
            away_data = teams.get("away", {})
            
            home_id = home_data.get("id")
            away_id = away_data.get("id")
            home_name = home_data.get("name", "")
            away_name = away_data.get("name", "")
            
            if not home_name or not away_name:
                continue
            
            scores = game.get("scores", {})
            if sport in ["NBA", "NCAAB"]:
                home_score = scores.get("home", {}).get("total", 0) or 0
                away_score = scores.get("away", {}).get("total", 0) or 0
            elif sport == "NHL":
                home_score = scores.get("home", 0) or 0
                away_score = scores.get("away", 0) or 0
            else:
                home_score = scores.get("home", {}).get("total", 0) or 0
                away_score = scores.get("away", {}).get("total", 0) or 0
            
            if home_score == 0 and away_score == 0:
                continue
            
            home_standing = standings.get(home_id, {})
            away_standing = standings.get(away_id, {})
            
            home_stats = client.get_team_statistics(sport, home_id, season, league_id) if home_id else {}
            away_stats = client.get_team_statistics(sport, away_id, season, league_id) if away_id else {}
            
            features = extract_game_features(game, home_stats, away_stats,
                                             home_standing, away_standing, sport)
            
            game_record = {
                "game_id": game.get("id", f"{game_date_str[:10]}_{home_name}_{away_name}"),
                "date": game_date_str[:10],
                "sport": sport,
                "season": season,
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "home_won": 1 if home_score > away_score else 0,
                **features,
            }
            
            all_games.append(game_record)
            completed += 1
        
        print(f"    ✅ {completed} completed games in date range")
        time.sleep(1)
    
    return all_games


def collect_historical_data(start_date: datetime, end_date: datetime,
                            sports: List[str], output_file: str = "training_data.csv",
                            api_key: str = None) -> pd.DataFrame:
    """Main collection function"""
    
    print(f"\n{'='*60}")
    print(f"🏈 API-SPORTS HISTORICAL DATA COLLECTOR")
    print(f"{'='*60}")
    print(f"📅 {start_date.date()} to {end_date.date()}")
    print(f"🏀 Sports: {', '.join(sports)}")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")
    
    client = APISportsClient(api_key=api_key)
    
    all_games = []
    for sport in sports:
        print(f"\n📊 Collecting {sport}...")
        games = collect_sport_data(sport, start_date, end_date, client)
        all_games.extend(games)
        print(f"  ✅ {len(games)} {sport} games")
    
    if not all_games:
        print("\n❌ No games collected!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_games)
    
    for feature in FEATURE_NAMES:
        if feature not in df.columns:
            df[feature] = 0.5 if "pct" in feature or "prob" in feature else 0
    
    df = df.sort_values("date")
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
    parser = argparse.ArgumentParser(description="Collect historical data using API-Sports")
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--sports", nargs="+", default=["NBA", "NHL", "NCAAB"],
                       choices=["NBA", "NHL", "NCAAB", "NFL", "NCAAF"])
    parser.add_argument("--output", type=str, default="training_data.csv")
    parser.add_argument("--api-key", type=str)
    
    args = parser.parse_args()
    
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)
    
    df = collect_historical_data(start_date, end_date, args.sports, args.output, args.api_key)
    
    if len(df) > 0:
        print("\n📋 Next: python train_vertex_model.py --data", args.output,
              "--project-id elite-hangar-479017-m8 --deploy")


if __name__ == "__main__":
    main()
