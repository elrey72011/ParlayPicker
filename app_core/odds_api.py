import requests
import logging
from datetime import datetime, time
import pytz
from typing import List, Dict

logger = logging.getLogger(__name__)

class OddsAPIAuthError(Exception):
    """Exception raised for authentication errors with The Odds API (e.g., 401, 403 or missing key)."""
    pass

class TheOddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions="us_ex,us", markets="h2h,spreads,totals", bookmakers="novig,draftkings,fanduel", oddsFormat="american"):
        if not api_key:
            raise ValueError("TheOddsAPI API key is required")

        self.api_key = api_key
        self.regions = regions
        self.markets = markets
        self.bookmakers = bookmakers
        self.oddsFormat = oddsFormat

    def get_odds(self, sport_key: str, date: str = None):
        if date:
            url = f"{self.BASE_URL}/historical/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": self.markets,
                "bookmakers": self.bookmakers,
                "oddsFormat": self.oddsFormat,
                "date": date
            }
        else:
            url = f"{self.BASE_URL}/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": self.markets,
                "bookmakers": self.bookmakers,
                "oddsFormat": self.oddsFormat,
            }

        all_data = []

        import time
        max_retries = 3

        while True:
            backoff = 2.0
            for attempt in range(max_retries + 1):
                resp = requests.get(url, params=params, timeout=15)

                if resp.status_code != 200:
                    logger.error(f"Odds API Failed [{resp.status_code}]: {resp.text}")
                    return []

                try:
                    resp.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    if resp.status_code == 429:
                        if attempt < max_retries:
                            logger.warning(f"The Odds API 429 Too Many Requests. Retrying in {backoff} seconds...")
                            time.sleep(backoff)
                            backoff *= 2.0
                            continue
                        else:
                            raise
                    elif resp.status_code in (401, 403):
                        raise OddsAPIAuthError(f"Invalid or missing API Key")
                    raise

            data = resp.json()
            import json
            import os
            os.makedirs('data', exist_ok=True)
            with open('data/live_odds_debug.json', 'w') as f:
                json.dump(data, f, indent=4)

            # Unwrap historical data wrapper if present
            if isinstance(data, dict) and "data" in data:
                games_data = data.get("data", [])
            else:
                games_data = data

            if isinstance(games_data, list):
                all_data.extend(games_data)
            else:
                return games_data

            # Check for pagination (cursor-based or page-based)
            total_results = int(resp.headers.get("x-total-results", len(all_data)))
            next_page_token = resp.headers.get("x-next-page") or (data.get('next_page') if isinstance(data, dict) else None)

            if next_page_token:
                params["cursor"] = next_page_token
                continue

            # Alternative: explicit limit-and-offset or header logic if they specify
            if len(all_data) >= total_results:
                break

            if isinstance(games_data, list) and len(games_data) == 0:
                break

            break

        # Filter out games that don't have any bookmakers
        filtered_data = [game for game in all_data if game.get("bookmakers")]
        return filtered_data

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

import pandas as pd

def export_raw_odds_api(odds_response: Dict, filename: str = None) -> str:
    """Export raw odds_api.com response for debugging, properly flattening bookmaker markets."""
    import pandas as pd
    rows = []

    games = odds_response if isinstance(odds_response, list) else odds_response.get('data', [])
    for game in games:
        for book in game.get('bookmakers', []):
            book_key = book.get('key', '')

            # Start building base row for this bookmaker in this game
            row = {
                'game_id': game.get('id'),
                'home_team': game.get('home_team'),
                'away_team': game.get('away_team'),
                'book': book_key,
                'commence_time': game.get('commence_time')
            }

            # Extract spreads
            for market in book.get('markets', []):
                if market.get('key') == 'spreads':
                    for o in market.get('outcomes', []):
                        prefix = f"{book_key}_" if book_key == "novig" else ""
                        if o.get('name') == game.get('home_team'):
                            row[f'{prefix}home_point'] = o.get('point')
                            row[f'{prefix}home_price'] = o.get('price')
                        elif o.get('name') == game.get('away_team'):
                            row[f'{prefix}away_point'] = o.get('point')
                            row[f'{prefix}away_price'] = o.get('price')

                # Extract totals
                elif market.get('key') == 'totals':
                    for o in market.get('outcomes', []):
                        prefix = f"{book_key}_" if book_key == "novig" else ""
                        if str(o.get('name')).lower() == 'over':
                            row[f'{prefix}over_point'] = o.get('point')
                            row[f'{prefix}over_price'] = o.get('price')
                        elif str(o.get('name')).lower() == 'under':
                            row[f'{prefix}under_point'] = o.get('point')
                            row[f'{prefix}under_price'] = o.get('price')

            rows.append(row)

    df = pd.DataFrame(rows)
    filename = filename or f"odds_api_raw_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Raw odds export: {filename}")
    return filename
