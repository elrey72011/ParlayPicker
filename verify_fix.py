
import sys
import os
from datetime import date

# Add repo root to path
sys.path.append(os.getcwd())

from app_core.apisports import APISportsBasketballClient
from app_core.sportsdata import SportsDataNBAClient

def test_apisports_returns_list():
    client = APISportsBasketballClient(api_key="fake_key")
    # Force _request to return None to test the fix
    original_request = client._request
    client._request = lambda *args, **kwargs: None

    games = client.get_games_by_date(date.today())
    print(f"APISports get_games_by_date returned type: {type(games)}")
    if games == []:
        print("PASS: APISports returned [] on None payload.")
    else:
        print(f"FAIL: APISports returned {games}")
        sys.exit(1)

def test_sportsdata_returns_list():
    client = SportsDataNBAClient(api_key="fake_key")
    # Force _request to return None
    client._request = lambda *args, **kwargs: None

    games = client.get_scores_by_date(date.today())
    print(f"SportsData get_scores_by_date returned type: {type(games)}")
    if games == []:
        print("PASS: SportsData returned [] on None payload.")
    else:
        print(f"FAIL: SportsData returned {games}")
        sys.exit(1)

if __name__ == "__main__":
    test_apisports_returns_list()
    test_sportsdata_returns_list()
