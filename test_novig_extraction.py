import os
import json
import logging
from app_core.odds_api import TheOddsAPIClient

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_novig_extraction():
    # Attempt to get API key from environment, fallback to a placeholder (which will fail if not set, but expected)
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("⚠️ ODDS_API_KEY environment variable not set. Please set it before running this test.")
        return

    print("🚀 Initializing TheOddsAPIClient for 'novig' bookmaker in 'us_ex' region...")
    client = TheOddsAPIClient(
        api_key=api_key,
        regions="us_ex",
        markets="h2h,spreads,totals",
        bookmakers="novig"
    )

    sport = "basketball_nba"
    print(f"📡 Fetching live odds for {sport}...")

    try:
        games = client.get_odds(sport_key=sport)
    except Exception as e:
        print(f"❌ Error fetching odds: {e}")
        return

    if not games:
        print(f"⚠️ No games found for {sport} or no novig odds available.")
        return

    print(f"✅ Successfully fetched data for {len(games)} games.")
    print("-" * 50)

    # Print raw JSON for the first game
    first_game = games[0]
    print("📝 Raw JSON for the first game:")
    print(json.dumps(first_game, indent=2))
    print("-" * 50)

    print("🔍 Extracting Novig Odds:")
    found_novig = False

    for game in games:
        game_id = game.get("id")
        home = game.get("home_team")
        away = game.get("away_team")
        print(f"\n🏀 Game: {away} @ {home} (ID: {game_id})")

        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") == "novig":
                found_novig = True
                print("  ✅ Found 'novig' bookmaker")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    print(f"    📊 Market: {market_key}")

                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        point = outcome.get("point")

                        if point is not None:
                            print(f"      ➡️ {name}: Line {point}, Odds {price}")
                        else:
                            print(f"      ➡️ {name}: Odds {price}")

    if not found_novig:
        print("❌ WARNING: The 'novig' bookmaker key was not found in any of the returned games.")

if __name__ == "__main__":
    test_novig_extraction()
