import sys
import unittest
from unittest.mock import MagicMock
import logging
import re
import traceback
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Mock external dependencies
sys.modules['streamlit'] = MagicMock()
import streamlit as st
st.session_state = {}

# Mocks for globals expected by fetch_kalshi_markets
logger = logging.getLogger("test")
logging.basicConfig(level=logging.INFO)

kalshi_integrator = MagicMock()
# Setup basic mock behavior
kalshi_integrator.last_fetch_meta = {}
kalshi_integrator.last_request_params = {}
kalshi_integrator.split_market_kinds.return_value = {}

LEAGUE_SERIES_MAP = {"NCAAB": ["KXNCAAB"]}

def league_game_prefix(league):
    return "KXNCAABGAME" if league == "NCAAB" else "KXGAME"

def get_local_tz():
    return "UTC"

def parse_commence_to_utc(date_str):
    # Simple parser for test
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except:
        return datetime.now()

try:
    from zoneinfo import ZoneInfo
except ImportError:
    class ZoneInfo:
        def __init__(self, key): pass

# Read the function code from streamlit_app.py
# We extract lines 5669 to 5910 approximately
with open("streamlit_app.py", "r") as f:
    lines = f.readlines()
    # Find start and end of fetch_kalshi_markets
    start_line = 0
    end_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def fetch_kalshi_markets("):
            start_line = i
        if line.strip().startswith("@st.cache_data") and i > start_line and start_line > 0:
            end_line = i
            break

    if end_line == 0:
        # Fallback if next function decorator not found immediately
        # Look for next def
        for i in range(start_line + 1, len(lines)):
            if line.strip().startswith("def "):
                end_line = i
                break

    if end_line == 0:
        end_line = len(lines)

    code = "".join(lines[start_line:end_line])

# Execute the code to define the function in current namespace
exec(code)

class TestKalshiNCAAB(unittest.TestCase):
    def setUp(self):
        kalshi_integrator.reset_mock()
        st.session_state = {}

    def test_pagination_ncaab(self):
        print("\n--- Test 1: NCAAB Pagination ---")
        # Setup mock return
        kalshi_integrator.get_league_markets.return_value = []
        kalshi_integrator.last_fetch_meta = {}
        kalshi_integrator.split_market_kinds.return_value = {}

        # Call function
        fetch_kalshi_markets("NCAAB")

        # Verify max_pages=20
        kalshi_integrator.get_league_markets.assert_called_with(
            "NCAAB",
            min_prefix_hits=20,
            max_pages=20
        )
        print("✅ Pagination check passed: max_pages=20 for NCAAB")

    def test_pagination_nba(self):
        print("\n--- Test 2: NBA Pagination (Regression Check) ---")
        # Setup mock return
        kalshi_integrator.get_league_markets.return_value = []

        # Call function
        fetch_kalshi_markets("NBA")

        # Verify max_pages=5 (default)
        kalshi_integrator.get_league_markets.assert_called_with(
            "NBA",
            min_prefix_hits=20,
            max_pages=5
        )
        print("✅ Pagination check passed: max_pages=5 for NBA")

    def test_filter_safety_guard(self):
        print("\n--- Test 3: Filter Safety Guard ---")
        # Setup mock to return a list of markets
        mock_market = {
            "ticker": "KXNCAABGAME-25DEC17-UCONN-PURDUE",
            "event_ticker": "KXNCAABGAME-25DEC17-UCONN-PURDUE"
        }
        kalshi_integrator.get_league_markets.return_value = [mock_market]
        kalshi_integrator.last_fetch_meta = {"game_prefix_used": "KXNCAABGAME"}

        # Split returns candidates
        kalshi_integrator.split_market_kinds.return_value = {
            "single_game_candidates": [mock_market]
        }

        # Call function with a date that won't match
        res = fetch_kalshi_markets("NCAAB", commence_times_utc=["2025-01-01T00:00:00Z"])

        # Expectation: Pool should not be empty
        pool = st.session_state.get("kalshi_markets_game_pool", {}).get("NCAAB")

        if pool and len(pool) > 0:
            print(f"✅ Safety guard passed: Pool size {len(pool)} despite date mismatch")
        else:
            self.fail("❌ Safety guard failed: Pool is empty")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
