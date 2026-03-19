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

# Import normalize_team_name from the core mapping logic
from core.team_mapper import normalize_team_name

# We also import streamlit to attempt reading st.secrets safely
try:
    import streamlit as st
except ImportError:
    st = None

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
    """Retrieve the API key from st.secrets or environment."""
    api_key = None

    # Try Streamlit Secrets first
    if st is not None:
        try:
            api_key = st.secrets.get("ODDS_API_KEY")
        except Exception:
            pass

    # Fallback to Environment variables
    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY")

    # User prompt if running locally and missing key
    if not api_key:
        print("API Key not found in st.secrets or environment.")
        # PLACEHOLDER: Paste your odds API key here manually if needed:
        # api_key = "YOUR_KEY_HERE"
        if not api_key:
            api_key = input("Please enter your The Odds API key: ").strip()

    return api_key

def fetch_historical_odds(api_key, sport, date_iso):
    """Fetch historical odds for a specific sport and date using the paid endpoint."""
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "date": date_iso
    }

    print(f"Fetching historical odds for {sport} on {date_iso}...")
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

def clean_team_name_for_id(name: str) -> str:
    """Helper to strip non-alphanumeric and lowercase, exactly like the pipeline."""
    import re
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def process_game(game_odds, scores_map):
    """Process a single game's odds and attach scores if available."""
    game_id = game_odds.get("id")
    sport_key = game_odds.get("sport_key", "unknown")
    home_team = str(game_odds.get("home_team", ""))
    away_team = str(game_odds.get("away_team", ""))
    commence_time = game_odds.get("commence_time")

    if not home_team or not away_team or not commence_time:
        return None

    # Normalization: use core.team_mapper.normalize_team_name (Source of Truth)
    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)

    # Generate Matchup ID Logic matching pipeline (League + Sorted Teams + Game Date)
    sport_mapped = sport_key.split("_")[-1].upper()
    if sport_mapped == "NCAAB":
        sport_mapped = "NCAAB"  # ensure consistency

    home_cleaned = clean_team_name_for_id(norm_home).upper()
    away_cleaned = clean_team_name_for_id(norm_away).upper()

    # Sort strictly alphabetically
    team_a = min(home_cleaned, away_cleaned)
    team_b = max(home_cleaned, away_cleaned)

    try:
        est_ts = pd.Timestamp(iso8601_to_est(commence_time))
        game_date = est_ts.floor("D").strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Failed to parse commence_time {commence_time}: {e}")
        return None

    matchup_id = f"{sport_mapped}|{team_a}|{team_b}|{game_date}"

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

    # Add missing default columns to match master_all_sports.csv exactly
    return {
        "matchup_id": matchup_id,
        "home_win_pct": pd.NA,
        "away_win_pct": pd.NA,
        "home_avg_points": pd.NA,
        "away_avg_points": pd.NA,
        "home_def_rating": pd.NA,
        "away_def_rating": pd.NA,
        "spread_normalized": closing_spread / 100.0 if closing_spread != 0 else 0.0,
        "implied_home_prob": implied_home_prob,
        "home_last_5": pd.NA,
        "away_last_5": pd.NA,
        "home_home_record": pd.NA,
        "away_away_record": pd.NA,
        "head_to_head": pd.NA,
        "home_streak": pd.NA,
        "away_streak": pd.NA,
        "rest_advantage": pd.NA,
        "injuries_impact": pd.NA,
        "weather_factor": pd.NA,
        "back_to_back": pd.NA,
        "primetime_game": pd.NA,
        "division_game": pd.NA,
        "public_betting_pct": pd.NA,
        "sharp_money_indicator": pd.NA,
        "line_movement": pd.NA,
        "total_movement": pd.NA,
        "model_consensus": pd.NA,
        "theover_probability": implied_home_prob,  # Base fallback
        "home_won": home_won,
        "home_score": home_score,
        "away_score": away_score,
        "game_id": game_id,
        "sport": sport_mapped,
        "home_team": norm_home,
        "away_team": norm_away,
        "commence_time": commence_time
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
        existing_df["_norm_home"] = existing_df["home_team"].apply(lambda x: normalize_team_name(str(x)))
        existing_df["_norm_away"] = existing_df["away_team"].apply(lambda x: normalize_team_name(str(x)))

        def make_id(row):
            sport = str(row.get("sport", "UNKNOWN")).upper()
            home_cl = clean_team_name_for_id(str(row.get("_norm_home", ""))).upper()
            away_cl = clean_team_name_for_id(str(row.get("_norm_away", ""))).upper()
            team_a = min(home_cl, away_cl)
            team_b = max(home_cl, away_cl)
            try:
                dt = pd.Timestamp(iso8601_to_est(row["commence_time"]))
                date_str = dt.floor("D").strftime("%Y-%m-%d")
                return f"{sport}|{team_a}|{team_b}|{date_str}"
            except:
                return f"{sport}|{team_a}|{team_b}|unknown_date"

        existing_df["_temp_matchup_id"] = existing_df.apply(make_id, axis=1)
        existing_ids = set(existing_df["_temp_matchup_id"].dropna().unique())
        print(f"Found {len(existing_ids)} unique games in existing data.")
    else:
        print("Master CSV not found. A new one will be created.")
        existing_df = pd.DataFrame()
        existing_ids = set()

    all_new_games = []
    league_counts = {"NBA": 0, "NHL": 0, "NCAAB": 0}

    for sport in TARGET_SPORTS:
        # Fetch all scores for the past X days
        scores_data = fetch_scores(api_key, sport, DAYS_TO_BACKFILL)
        scores_map = {game["id"]: game for game in scores_data}
        print(f"Found {len(scores_map)} total games in the last {DAYS_TO_BACKFILL} days for {sport}.")

        # Iterate through each day to fetch historical odds
        # Loop backwards so we get the latest line first
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

                    lg = processed["sport"]
                    if lg in league_counts:
                        league_counts[lg] += 1
                    else:
                        league_counts[lg] = 1

    if not all_new_games:
        print("No new completed games with odds found to append.")
        return

    new_df = pd.DataFrame(all_new_games)
    print(f"\nSuccessfully processed {len(new_df)} new games.")

    # Match the master CSV column structure exactly
    if not existing_df.empty:
        original_columns = [c for c in existing_df.columns if c not in ["_norm_home", "_norm_away", "_temp_matchup_id"]]

        # Fill any missing columns to avoid dropping column definitions
        for col in original_columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA

        new_df = new_df[original_columns] # Reorder to match

        # Concatenate and save
        combined_df = pd.concat([existing_df[original_columns], new_df], ignore_index=True)
    else:
        combined_df = new_df.drop(columns=["matchup_id"], errors='ignore')

    combined_df.to_csv(master_csv_path, index=False)
    print(f"✅ Successfully appended {len(new_df)} games to {master_csv_path}")
    print("\n--- Backfill Summary ---")
    for lg, count in league_counts.items():
        print(f"{lg}: {count} new games added")

if __name__ == "__main__":
    main()
