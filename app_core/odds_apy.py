import requests
import logging

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
