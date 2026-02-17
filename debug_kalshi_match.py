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

    try:
        import streamlit as st
        try:
            kalshi_key = st.secrets.get("kalshi_api_key") or st.secrets.get("KALSHI_API_KEY")
            kalshi_secret = st.secrets.get("kalshi_api_secret") or st.secrets.get("KALSHI_API_SECRET")
        except Exception:
            pass # Secrets not found
    except ImportError:
        pass

    if not kalshi_key:
        kalshi_key = os.getenv("KALSHI_API_KEY")
    if not kalshi_secret:
        kalshi_secret = os.getenv("KALSHI_API_SECRET")

    if not kalshi_key or not kalshi_secret:
        logger.error("❌ Kalshi API credentials not found in env or secrets")
        return

    integrator = KalshiIntegrator(kalshi_key, kalshi_secret)
    logger.info(f"✅ KalshiIntegrator initialized: {integrator is not None}")

    # Fetch active NBA events to find a real game candidate
    logger.info("Fetching active NBA events to find a candidate...")
    events_resp = integrator.get_events("KXNBAGAME", status="active", limit=5)
    events = events_resp.get("events", [])

    if not events:
        logger.warning("No active NBA events found. Trying active markets from get_markets...")
        # Fallback to get_markets if get_events fails or returns empty
        markets = integrator.get_markets(status="active")
        # Filter for NBA game markets
        events = [] # We can't easily reconstruct full event from market without more logic, but let's try to extract info
        for m in markets:
            t = m.get("ticker", "")
            if t.startswith("KXNBAGAME"):
                 # Create a mock event object
                 events.append({"ticker": m.get("event_ticker"), "start_time": m.get("open_time")}) # Approximation
                 if len(events) >= 5: break

    if not events:
        logger.error("❌ No active NBA events or markets found to test with.")
        return

    # Pick the first event
    candidate_event = events[0]
    ticker = candidate_event.get("ticker")
    logger.info(f"Picked candidate event ticker: {ticker}")

    # Parse teams from ticker
    parsed = parse_event_ticker_codes(ticker)
    home_code = parsed.get("home")
    away_code = parsed.get("away")

    logger.info(f"Parsed codes: Home={home_code}, Away={away_code}")

    # We need full names to test the matcher properly, as match_game_to_kalshi starts with full names
    # and converts them to codes.
    # Let's try to reverse map codes to names using NBA_TEAM_CODE_MAP values if possible,
    # or just use the codes as names and hope the cleaner handles it (it might not).
    # Better: check app_core/kalshi_integrator.py maps.

    from app_core.kalshi_integrator import NBA_TEAM_CODE_MAP

    # Reverse map
    code_to_name = {v: k for k, v in NBA_TEAM_CODE_MAP.items()}

    home_team = code_to_name.get(home_code, home_code) # Fallback to code if not found
    away_team = code_to_name.get(away_code, away_code)

    logger.info(f"Mapped to names: Home='{home_team}', Away='{away_team}'")

    # Use current time or event time if available
    # match_game_to_kalshi uses game_time to check against close_time
    # Let's assume the game is 'today' or recently in future
    game_time = datetime.now(pytz.UTC) + timedelta(hours=1)

    test_game = {
        "home_team": home_team,
        "away_team": away_team,
        "league": "NBA",
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

    if result.debug:
        logger.info(f"   Debug Info: {result.debug}")

    # Test TOTAL matching
    result_total = match_game_to_kalshi(
        league=test_game["league"],
        home_team=test_game["home_team"],
        away_team=test_game["away_team"],
        game_time=test_game["commence_time"],
        integrator=integrator,
        requested_market_type="TOTAL"
    )

    logger.info(f"\n📊 TOTAL Match Result:")
    logger.info(f"   Matched: {result_total.matched}")
    logger.info(f"   Reason: {result_total.reason}")
    logger.info(f"   Market Ticker: {result_total.market_ticker}")

if __name__ == "__main__":
    test_kalshi_match()
