#!/usr/bin/env python3
"""Debug script to test Kalshi matching for a known game."""

import logging
import os
import sys
from datetime import datetime, timedelta
import pytz

# Add repo root to path
sys.path.append(os.getcwd())

from app_core.kalshi_integrator import KalshiIntegrator, match_game_to_kalshi, parse_event_ticker_codes, resolve_team_code

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_kalshi_match():
    """Test matching for a known NBA game with active Kalshi markets."""

    # Initialize integrator
    kalshi_key = None
    kalshi_secret = None

    # 1. Try Streamlit Secrets
    try:
        import streamlit as st
        try:
            kalshi_key = st.secrets.get("kalshi_api_key") or st.secrets.get("KALSHI_API_KEY")
            kalshi_secret = st.secrets.get("kalshi_api_secret") or st.secrets.get("KALSHI_API_SECRET")
        except Exception:
            pass # Secrets not found
    except ImportError:
        pass

    # 2. Try Env Vars
    if not kalshi_key:
        kalshi_key = os.getenv("KALSHI_API_KEY")
    if not kalshi_secret:
        kalshi_secret = os.getenv("KALSHI_API_SECRET")

    # 3. Try Manual TOML Read
    if not kalshi_key or not kalshi_secret:
        try:
            import toml
            secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                logger.info(f"Reading secrets from {secrets_path}")
                with open(secrets_path, "r") as f:
                    secrets = toml.load(f)
                    # Handle both top-level and [general] section
                    kalshi_key = secrets.get("KALSHI_API_KEY") or secrets.get("kalshi_api_key")
                    kalshi_secret = secrets.get("KALSHI_API_SECRET") or secrets.get("kalshi_api_secret")

                    if not kalshi_key and "general" in secrets:
                        gen = secrets["general"]
                        kalshi_key = gen.get("KALSHI_API_KEY") or gen.get("kalshi_api_key")
                        kalshi_secret = gen.get("KALSHI_API_SECRET") or gen.get("kalshi_api_secret")
        except Exception as e:
            logger.warning(f"Failed to read secrets.toml: {e}")

    if not kalshi_key or not kalshi_secret:
        logger.error("❌ Kalshi API credentials not found in env, streamlit secrets, or .streamlit/secrets.toml")
        return

    # Mask keys for logging
    masked_key = f"{kalshi_key[:4]}...{kalshi_key[-4:]}" if kalshi_key else "None"
    logger.info(f"🔑 Keys found: API_KEY={masked_key}")

    integrator = KalshiIntegrator(kalshi_key, kalshi_secret)
    logger.info(f"✅ KalshiIntegrator initialized: {integrator is not None}")

    # Verify API Connectivity
    try:
        balance = integrator.get_markets(status="active", limit=1)
        logger.info("✅ API Connectivity Check Passed (fetched 1 market)")
    except Exception as e:
        logger.error(f"❌ API Connectivity Check Failed: {e}")
        return

    # Fetch active events to find a real game candidate
    target_series = "KXNCAAMBGAME" # Try NCAAB as that was the user's issue
    logger.info(f"Fetching active {target_series} events to find a candidate...")

    events_resp = integrator.get_events(target_series, status="active", limit=10)
    events = events_resp.get("events", [])

    if not events:
        logger.warning(f"No active {target_series} events found. Trying NBA...")
        target_series = "KXNBAGAME"
        events_resp = integrator.get_events(target_series, status="active", limit=10)
        events = events_resp.get("events", [])

    if not events:
        logger.warning("No active events found via get_events. Dumping active markets...")
        markets = integrator.get_markets(status="active")
        logger.info(f"Found {len(markets)} active markets total.")
        if markets:
            logger.info(f"Sample market 1: {markets[0].get('ticker')}")
        return

    # Pick the first event
    candidate_event = events[0]
    ticker = candidate_event.get("ticker")
    logger.info(f"Picked candidate event ticker: {ticker}")

    # Parse teams from ticker
    parsed = parse_event_ticker_codes(ticker)
    home_code = parsed.get("home")
    away_code = parsed.get("away")
    date_token = parsed.get("date_token")

    logger.info(f"Parsed codes: Home={home_code}, Away={away_code}, Date={date_token}")

    # Map codes to names
    from app_core.kalshi_integrator import NBA_TEAM_CODE_MAP, NCAAB_TEAM_CODE_MAP

    code_to_name = {}
    if "NBA" in ticker:
        code_to_name = {v: k for k, v in NBA_TEAM_CODE_MAP.items()}
    elif "NCAA" in ticker:
        code_to_name = {v: k for k, v in NCAAB_TEAM_CODE_MAP.items()}

    home_team = code_to_name.get(home_code, home_code) # Fallback to code if not found
    away_team = code_to_name.get(away_code, away_code)

    logger.info(f"Mapped to names: Home='{home_team}', Away='{away_team}'")

    # Use current time or event time if available
    game_time = datetime.now(pytz.UTC) + timedelta(hours=1)

    # Try to parse date token if possible
    # 26FEB15 -> 2026-02-15
    try:
        dt = datetime.strptime(date_token, "%y%b%d").replace(tzinfo=pytz.UTC)
        # Set to noon on that day
        game_time = dt + timedelta(hours=12)
        logger.info(f"Inferred game time from ticker: {game_time}")
    except:
        pass

    test_game = {
        "home_team": home_team,
        "away_team": away_team,
        "league": "NCAAB" if "NCAA" in ticker else "NBA",
        "commence_time": game_time,
        "id": "test_game_001"
    }

    logger.info(f"\n🔍 Testing match for: {test_game['home_team']} vs {test_game['away_team']}")

    # Test SPREAD matching
    # Note: match_game_to_kalshi arguments: league, home, away, game_time, integrator...
    result = match_game_to_kalshi(
        league=test_game["league"],
        home_team=test_game["home_team"],
        away_team=test_game["away_team"],
        game_time=test_game["commence_time"],
        integrator=integrator,
        requested_market_type="SPREAD"
    )

    logger.info(f"\n📊 SPREAD Match Result:")
    logger.info(f"   Matched: {result.matched}")
    logger.info(f"   Reason: {result.reason}")
    logger.info(f"   Market Ticker: {result.market_ticker}")
    logger.info(f"   Prob: {result.probability}")
    logger.info(f"   Label: {result.label}")
    logger.info(f"   Kalshi Available: {result.kalshi_available}")

    if result.debug:
        logger.info(f"   Debug Info: {result.debug}")

if __name__ == "__main__":
    test_kalshi_match()
