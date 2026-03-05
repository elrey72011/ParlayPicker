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

import pandas as pd

def get_novig_reference(row: Dict) -> Dict:
    """Extract Novig spread and total as canonical reference."""
    novig_ref = {}
    for book in row.get('bookmakers', []):
        if 'novig' in book.get('key', '').lower():
            for market in book.get('markets', []):
                if market.get('key') == 'h2h_spreads':
                    outcomes = market.get('outcomes', [])
                    home_o = next((o for o in outcomes if o.get('name') == row.get('home_team')), None)
                    away_o = next((o for o in outcomes if o.get('name') == row.get('away_team')), None)
                    if home_o and away_o:
                        novig_ref.update({
                            'home_spread_point': home_o.get('point'),
                            'away_spread_point': away_o.get('point'),
                            'home_spread_price': home_o.get('price'),
                            'away_spread_price': away_o.get('price'),
                        })
                elif market.get('key') == 'totals':
                    outcomes = market.get('outcomes', [])
                    over_o = next((o for o in outcomes if str(o.get('name')).lower() == 'over'), None)
                    under_o = next((o for o in outcomes if str(o.get('name')).lower() == 'under'), None)
                    if over_o and under_o:
                        novig_ref.update({
                            'over_point': over_o.get('point'),
                            'under_point': under_o.get('point'),
                            'over_price': over_o.get('price'),
                            'under_price': under_o.get('price'),
                        })
            break
    return novig_ref if novig_ref else None

def process_odds_with_novig_priority(row: Dict) -> Dict:
    """
    Process row odds using Novig as the canonical reference point.
    To be used within the main parse/process odds loop in Streamlit.
    """
    novig = get_novig_reference(row)
    if novig:
        if 'home_spread_point' in novig:
            row['reference_spread_point'] = novig['home_spread_point']
            row['best_spread_price'] = min(row.get('best_spread_price', 999), novig['home_spread_price'])
            row['novig_spread_used'] = True
        if 'over_point' in novig:
            row['reference_total_point'] = novig['over_point']
            row['novig_total_used'] = True
    else:
        row['novig_spread_used'] = False
        row['novig_total_used'] = False

    return row


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
                if market.get('key') == 'h2h_spreads':
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
