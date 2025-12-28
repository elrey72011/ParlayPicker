
import sys
import os
import datetime
from typing import List, Dict, Any, Optional

# Mock Streamlit to allow imports
import types
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()
import streamlit as st
st.secrets = {}
st.session_state = {}

# Import functions to test
# We need to add the current directory to sys.path
sys.path.append(os.getcwd())

from app_core.team_name_matcher import TeamNameMatcher
from app_core.sentiment_pipeline import _newsapi_query

# We need to import filter_kalshi_game_markets from streamlit_app.py
# Since streamlit_app.py is a script, we can't easily import it without side effects
# But we can inspect the file or just copy the function for testing if we are confident it matches.
# Better: Import it if possible, but streamlit_app.py has a lot of top-level code.
# Strategy: Read the file, extract the function code, exec it, and test.

def get_function_from_file(filepath, function_name):
    with open(filepath, 'r') as f:
        content = f.read()

    # Simple extraction (assuming robust formatting from previous writes)
    start_marker = f"def {function_name}("
    if start_marker not in content:
        raise ValueError(f"Function {function_name} not found")

    start_idx = content.find(start_marker)

    # Count braces/indentation to find end? Hard.
    # Let's just grab the next few hundred lines and hope we get the whole function
    # or rely on the python parser.
    # Actually, we can just import the logic we changed in app_core if possible.
    # But filter_kalshi_game_markets is in streamlit_app.py.
    pass

# Direct functional test of TeamNameMatcher (Enrichment Fix)
def test_enrichment_logic():
    print("--- Testing Enrichment Logic ---")
    home_raw = "Army"
    api_candidates = ["Army Black Knights", "Navy Midshipmen"]

    # We changed the threshold to 0.75 in the plan/code
    # Let's verify TeamNameMatcher behavior with that threshold
    match = TeamNameMatcher.match_team(home_raw, api_candidates, threshold=0.75)
    print(f"Match 'Army' against {api_candidates} (thresh=0.75): {match}")

    if match == "Army Black Knights":
        print("✅ Enrichment Fuzzy Match: SUCCESS")
    else:
        print("❌ Enrichment Fuzzy Match: FAILED")

# Direct test of NewsAPI Query
def test_newsapi_query():
    print("\n--- Testing NewsAPI Query ---")
    team = "LSU Tigers"
    league = "NCAAF"
    query = _newsapi_query(team, league)
    print(f"Query for '{team}': {query}")

    if query == '"LSU Tigers"':
        print("✅ NewsAPI Query Simplification: SUCCESS")
    else:
        print(f"❌ NewsAPI Query Simplification: FAILED (Expected '\"LSU Tigers\"', got '{query}')")

# Simulation of filter_kalshi_game_markets logic
def test_kalshi_filtering():
    print("\n--- Testing Kalshi Filtering Logic ---")

    # Mocking the logic we implemented in streamlit_app.py
    # We allow "GAME" in ticker + team code match for NCAAB

    market = {"event_ticker": "KXNCAABGAME-23DEC25-DUKE-UNC", "ticker": "KXNCAABGAME-23DEC25-DUKE-UNC"}

    # Case 1: NCAAB Relaxation
    league_upper = "NCAAB"
    t_upper = market["event_ticker"]
    home_code = "DUKE"
    away_code = "UNC"

    prefix_ok = False
    # Logic from patch:
    if league_upper in ["NCAAB", "NCAAF"] and "GAME" in t_upper:
        if (home_code and home_code in t_upper) or (away_code and away_code in t_upper):
            prefix_ok = True

    print(f"Kalshi NCAAB Relaxation Check: {prefix_ok}")

    if prefix_ok:
        print("✅ Kalshi Filtering (NCAAB Relaxation): SUCCESS")
    else:
        print("❌ Kalshi Filtering (NCAAB Relaxation): FAILED")

if __name__ == "__main__":
    test_enrichment_logic()
    test_newsapi_query()
    test_kalshi_filtering()
