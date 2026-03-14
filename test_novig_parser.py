import os
import json
import logging

from core.team_mapper import normalize_team_name
from app_core.odds_api import TheOddsAPIClient

# Basic setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test key for mock
api_key = "test_key_for_sandbox"

# Instantiate client explicitly requesting spreads and totals
client = TheOddsAPIClient(
    api_key=api_key,
    regions="us_ex",
    markets="spreads,totals",
    bookmakers="novig",
)

# Mock the requests.get call
import unittest.mock as mock

def mock_get(*args, **kwargs):
    print("\n--- MOCK HTTP REQUEST ---")
    print(f"URL: {args[0]}")
    print(f"Params: {kwargs.get('params')}")
    print("-------------------------\n")

    # Return a dummy response that looks like Novig JSON
    class MockResponse:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {
                    "id": "mock_game_1",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2023-10-24T23:30:00Z",
                    "home_team": "Denver Nuggets",
                    "away_team": "Los Angeles Lakers",
                    "bookmakers": [
                        {
                            "key": "novig",
                            "title": "Novig",
                            "last_update": "2023-10-24T23:00:00Z",
                            "markets": [
                                {
                                    "key": "spreads",
                                    "last_update": "2023-10-24T23:00:00Z",
                                    "outcomes": [
                                        {"name": "Denver Nuggets", "price": -110, "point": -5.5},
                                        {"name": "Los Angeles Lakers", "price": -110, "point": 5.5}
                                    ]
                                },
                                {
                                    "key": "totals",
                                    "last_update": "2023-10-24T23:00:00Z",
                                    "outcomes": [
                                        {"name": "Over", "price": -105, "point": 227.5},
                                        {"name": "Under", "price": -115, "point": 227.5}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]

    return MockResponse()

@mock.patch('requests.get', side_effect=mock_get)
def fetch_and_parse(mock_request):
    sport_key = "basketball_nba"
    logger.info(f"Fetching live odds for {sport_key}...")

    try:
        games = client.get_odds(sport_key)

        if not games:
            logger.info("No games found.")
            return

        first_game = games[0]

        print("\n--- RAW JSON RESPONSE FOR FIRST GAME (Bookmakers) ---")
        print(json.dumps(first_game.get('bookmakers', []), indent=2))

        game_id = first_game.get('id')
        commence_time_str = first_game.get('commence_time', '')

        games_dict = {}
        games_dict[game_id] = {
            'game_id': game_id,
            'home_team': normalize_team_name(first_game.get('home_team')),
            'away_team': normalize_team_name(first_game.get('away_team')),
            'raw_home_team': first_game.get('home_team'),
            'raw_away_team': first_game.get('away_team'),
            'commence_time': commence_time_str,
            'uuid': game_id,
        }

        for book in first_game.get('bookmakers', []):
            book_key = book.get('key', '')
            if book_key != 'novig':
                continue

            for market in book.get('markets', []):
                if market.get('key') == 'spreads':
                    for o in market.get('outcomes', []):
                        if normalize_team_name(o.get('name')) == games_dict[game_id]['home_team']:
                            games_dict[game_id]['novig_home_point'] = o.get('point')
                            games_dict[game_id]['novig_home_price'] = o.get('price')
                            games_dict[game_id]['odds_source_spread'] = book_key
                        elif normalize_team_name(o.get('name')) == games_dict[game_id]['away_team']:
                            games_dict[game_id]['novig_away_point'] = o.get('point')
                            games_dict[game_id]['novig_away_price'] = o.get('price')
                            games_dict[game_id]['odds_source_spread'] = book_key
                elif market.get('key') == 'totals':
                    for o in market.get('outcomes', []):
                        if o.get('name') == 'Over':
                            games_dict[game_id]['novig_over_point'] = o.get('point')
                            games_dict[game_id]['novig_over_price'] = o.get('price')
                            games_dict[game_id]['odds_source_total'] = book_key
                        elif o.get('name') == 'Under':
                            games_dict[game_id]['novig_under_point'] = o.get('point')
                            games_dict[game_id]['novig_under_price'] = o.get('price')
                            games_dict[game_id]['odds_source_total'] = book_key

        print("\n--- PARSED DICTIONARY RESULT ---")
        print(json.dumps(games_dict[game_id], indent=2))

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_parse()
