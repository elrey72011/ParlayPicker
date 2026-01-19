import requests
import logging
from datetime import datetime, time
import pytz
from typing import List, Dict

logger = logging.getLogger(__name__)

class TheOddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions="us", markets="h2h,spreads,totals"):
        if not api_key:
            raise ValueError("TheOddsAPI API key is required")

        self.api_key = api_key
        self.regions = regions
        self.markets = markets

    def get_odds(self, sport_key: str):
        url = f"{self.BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "american",
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        return resp.json()

def filter_games_today_only(games: List[Dict]) -> List[Dict]:
    """
    Filter games to only include those on current day (00:00 - 23:59 EST).

    Args:
        games: List of game dicts with 'commence_time' in UTC

    Returns:
        Filtered list containing only today's games in EST
    """
    if not games:
        return []

    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)

    # Define today's boundaries in EST
    today_start = est.localize(datetime.combine(now_est.date(), time(0, 0, 0)))  # 00:00:00 EST
    today_end = est.localize(datetime.combine(now_est.date(), time(23, 59, 59)))  # 23:59:59 EST

    filtered_games = []

    for game in games:
        # Parse UTC commence time
        commence_time_str = game.get('commence_time', '')
        if not commence_time_str:
            continue

        try:
            # Parse UTC commence time (handle Z)
            commence_utc = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))

            # Convert to EST
            commence_est = commence_utc.astimezone(est)

            # Check if game is today (EST)
            if today_start <= commence_est <= today_end:
                filtered_games.append(game)
            else:
                logger.debug(f"Filtered out {game.get('home_team')} vs {game.get('away_team')} - "
                            f"starts {commence_est.strftime('%Y-%m-%d %H:%M')} EST (not today)")
        except Exception as e:
            logger.warning(f"Date parsing failed for game {game.get('home_team', 'Unknown')}: {e}")

    # Summary logging
    filtered_count = len(games) - len(filtered_games)
    if filtered_count > 0:
        logger.info(f"DATE FILTER: Removed {filtered_count} games not on today's date")
        logger.info(f"DATE FILTER: Today is {now_est.strftime('%Y-%m-%d')} EST")
        logger.info(f"DATE FILTER: Keeping games from {today_start.strftime('%H:%M')} to {today_end.strftime('%H:%M')} EST")

    return filtered_games
