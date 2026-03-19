import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser
import pytz

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_core.team_name_matcher import TeamNameMatcher

# Define target sports
TARGET_SPORTS = ['basketball_nba', 'icehockey_nhl', 'basketball_ncaab']
DAYS_TO_BACKFILL = 7

# Bookmakers to check for closing lines
TARGET_BOOKMAKERS = ['novig', 'draftkings', 'fanduel', 'pinnacle']

def iso8601_to_est(iso_string: str):
    """Converts ISO 8601 string to Eastern Time aware datetime."""
    dt_object_utc = parser.parse(str(iso_string))
    if dt_object_utc.tzinfo is None:
        dt_object_utc = dt_object_utc.replace(tzinfo=pytz.UTC)
    else:
        dt_object_utc = dt_object_utc.astimezone(pytz.UTC)
    eastern_tz = pytz.timezone("America/New_York")
    return dt_object_utc.astimezone(eastern_tz)

def get_api_key():
    """Retrieve the API key from environment or prompt."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        api_key = input("Please enter your The Odds API key: ").strip()
    return api_key

def fetch_historical_odds(api_key, sport, date_iso):
    """Fetch historical odds for a specific sport and date."""
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "date": date_iso
    }

    print(f"Fetching odds for {sport} on {date_iso}...")
    response = requests.get(url, params=params)
    time.sleep(1)  # Rate limiting safety

    if response.status_code != 200:
        print(f"Failed to fetch odds for {sport} on {date_iso}. Status: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

def fetch_scores(api_key, sport, days_from):
    """Fetch scores for a specific sport for the last X days."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
        "dateFormat": "iso"
    }

    print(f"Fetching scores for {sport} (last {days_from} days)...")
    response = requests.get(url, params=params)
    time.sleep(1)  # Rate limiting safety

    if response.status_code != 200:
        print(f"Failed to fetch scores for {sport}. Status: {response.status_code}")
        print(response.text)
        return []

    return response.json()

def process_game(game_odds, scores_map):
    """Process a single game's odds and attach scores if available."""
    game_id = game_odds.get("id")
    sport_key = game_odds.get("sport_key", "unknown")
    home_team = str(game_odds.get("home_team", ""))
    away_team = str(game_odds.get("away_team", ""))
    commence_time = game_odds.get("commence_time")

    if not home_team or not away_team or not commence_time:
        return None

    # Normalize team names
    norm_home = TeamNameMatcher.normalize(home_team)
    norm_away = TeamNameMatcher.normalize(away_team)

    # Generate match ID logic matching core/streamlit_pipeline.py _matchup_id
    teams_sorted = sorted([norm_home, norm_away])

    try:
        est_ts = pd.Timestamp(iso8601_to_est(commence_time))
        game_date = est_ts.floor("D").strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Failed to parse commence_time {commence_time}: {e}")
        return None

    matchup_id = "_".join(teams_sorted) + "_" + game_date

    # Check if we have scores for this game
    score_data = scores_map.get(game_id)
    if not score_data or not score_data.get("completed"):
        # We only want completed games for historical backfill
        return None

    home_score = None
    away_score = None

    scores = score_data.get("scores")
    if scores and isinstance(scores, list):
        for score_entry in scores:
            if score_entry.get("name") == home_team:
                home_score = score_entry.get("score")
            elif score_entry.get("name") == away_team:
                away_score = score_entry.get("score")

    if home_score is None or away_score is None:
        return None

    try:
        home_score = float(home_score)
        away_score = float(away_score)
    except (ValueError, TypeError):
        return None

    # Extract closing odds
    closing_spread = None
    closing_total = None
    implied_home_prob = 0.5

    # Get the best available lines from requested bookmakers
    bookmakers = game_odds.get("bookmakers", [])
    if not bookmakers:
        return None

    # Prefer novig, then pinnacle, then draftkings, then fanduel
    bookie_priority = {b: i for i, b in enumerate(TARGET_BOOKMAKERS)}
    sorted_bookmakers = sorted(
        bookmakers,
        key=lambda x: bookie_priority.get(x.get("key", ""), 999)
    )

    for bookmaker in sorted_bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") == "spreads" and closing_spread is None:
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        closing_spread = outcome.get("point")
            elif market.get("key") == "totals" and closing_total is None:
                for outcome in market.get("outcomes", []):
                    if str(outcome.get("name", "")).lower() == "over":
                        closing_total = outcome.get("point")
            elif market.get("key") == "h2h" and implied_home_prob == 0.5:
                # Calculate implied probability from American odds
                home_odds = None
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        home_odds = outcome.get("price")

                if home_odds is not None:
                    # American odds conversion
                    if home_odds < 0:
                        implied_home_prob = abs(home_odds) / (abs(home_odds) + 100)
                    elif home_odds > 0:
                        implied_home_prob = 100 / (home_odds + 100)

    # Fallback to defaults if markets missing but we still want the row
    closing_spread = closing_spread if closing_spread is not None else 0.0
    closing_total = closing_total if closing_total is not None else 100.0

    # Determine winner
    home_won = 1 if home_score > away_score else 0

    return {
        "matchup_id": matchup_id,
        "game_id": game_id,
        "sport": sport_key.split("_")[-1].upper(),
        "home_team": norm_home,
        "away_team": norm_away,
        "commence_time": commence_time,
        "game_date": game_date,
        "home_score": home_score,
        "away_score": away_score,
        "home_won": home_won,
        "closing_spread": closing_spread,
        "closing_total": closing_total,
        "implied_home_prob": implied_home_prob
    }

def main():
    print("Starting historical data backfill...")
    api_key = get_api_key()
    if not api_key:
        print("API key is required. Exiting.")
        return

    # Master CSV path
    master_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "master_all_sports.csv")

    # Load existing data to check duplicates
    if os.path.exists(master_csv_path):
        existing_df = pd.read_csv(master_csv_path)
        print(f"Loaded existing master CSV with {len(existing_df)} rows.")

        # We need a standardized matchup_id for deduplication
        existing_df["_norm_home"] = existing_df["home_team"].apply(lambda x: TeamNameMatcher.normalize(str(x)))
        existing_df["_norm_away"] = existing_df["away_team"].apply(lambda x: TeamNameMatcher.normalize(str(x)))

        def make_id(row):
            teams = sorted([str(row["_norm_home"]), str(row["_norm_away"])])
            try:
                dt = pd.Timestamp(iso8601_to_est(row["commence_time"]))
                date_str = dt.floor("D").strftime("%Y-%m-%d")
                return "_".join(teams) + "_" + date_str
            except:
                return "_".join(teams) + "_unknown_date"

        existing_df["_temp_matchup_id"] = existing_df.apply(make_id, axis=1)
        existing_ids = set(existing_df["_temp_matchup_id"].dropna().unique())
        print(f"Found {len(existing_ids)} unique games in existing data.")
    else:
        print("Master CSV not found. A new one will be created.")
        existing_df = pd.DataFrame()
        existing_ids = set()

    all_new_games = []

    for sport in TARGET_SPORTS:
        # 1. Fetch all scores for the past X days
        scores_data = fetch_scores(api_key, sport, DAYS_TO_BACKFILL)
        scores_map = {game["id"]: game for game in scores_data}
        print(f"Found {len(scores_map)} total games in the last {DAYS_TO_BACKFILL} days for {sport}.")

        # 2. Iterate through each day to fetch historical odds
        # Loop forward from today backwards so we get the latest line first
        for days_ago in range(DAYS_TO_BACKFILL + 1):
            # Target date format: 2024-03-15T12:00:00Z
            target_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
            date_iso = target_date.strftime("%Y-%m-%dT12:00:00Z")

            odds_data = fetch_historical_odds(api_key, sport, date_iso)
            print(f"Found {len(odds_data)} games with odds on {date_iso} for {sport}.")

            for game_odds in odds_data:
                processed = process_game(game_odds, scores_map)
                if processed and processed["matchup_id"] not in existing_ids:
                    all_new_games.append(processed)
                    existing_ids.add(processed["matchup_id"]) # Prevent duplicates in the same run

    if not all_new_games:
        print("No new completed games with odds found to append.")
        return

    new_df = pd.DataFrame(all_new_games)
    print(f"\nSuccessfully processed {len(new_df)} new games.")

    # Match the master CSV column structure exactly
    if not existing_df.empty:
        # Keep original columns
        original_columns = [c for c in existing_df.columns if c not in ["_norm_home", "_norm_away", "_temp_matchup_id"]]

        # Add new rows, filling missing columns with NaN
        for col in original_columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA

        new_df = new_df[original_columns] # Reorder to match

        # Concatenate and save
        combined_df = pd.concat([existing_df[original_columns], new_df], ignore_index=True)
    else:
        combined_df = new_df.drop(columns=["matchup_id", "game_date"], errors='ignore')

    combined_df.to_csv(master_csv_path, index=False)
    print(f"✅ Successfully appended {len(new_df)} games to {master_csv_path}")

if __name__ == "__main__":
    main()
