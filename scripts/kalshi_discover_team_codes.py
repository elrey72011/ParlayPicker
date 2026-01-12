import sys
import os
import re
import json
from datetime import datetime
import pytz

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app_core.kalshi_integrator import KalshiIntegrator, parse_event_ticker_codes, resolve_team_code
    # Import map constants
    from app_core.kalshi_integrator import (
        NBA_TEAM_CODE_MAP, NFL_TEAM_CODE_MAP, NHL_TEAM_CODE_MAP,
        NCAAF_TEAM_CODE_MAP, NCAAB_TEAM_CODE_MAP, LEAGUE_SERIES_MAP
    )
except ImportError:
    print("Failed to import KalshiIntegrator modules. Run from repo root.")
    sys.exit(1)

# Mock Streamlit secrets if not present
import streamlit as st
if not hasattr(st, "secrets"):
    st.secrets = {}
    if "KALSHI_API_KEY" in os.environ:
        st.secrets["KALSHI_API_KEY"] = os.environ["KALSHI_API_KEY"]
    if "KALSHI_API_SECRET" in os.environ:
        st.secrets["KALSHI_API_SECRET"] = os.environ["KALSHI_API_SECRET"]

def discover_codes():
    print("--- Kalshi Team Code Discovery (Script Mode) ---")

    try:
        ki = KalshiIntegrator()
        if not ki.api_key:
             print("Error: KALSHI_API_KEY not found. Skipping live scan.")
             return
    except Exception as e:
        print(f"Failed to init KalshiIntegrator: {e}")
        return

    leagues = ["NBA", "NFL", "NHL", "NCAAB"]

    for league in leagues:
        print(f"\nScanning {league}...")

        # 1. Determine series ticker
        series_ticker = LEAGUE_SERIES_MAP.get(league)
        if isinstance(series_ticker, list):
            series_ticker = series_ticker[0] # Prefer first (e.g., KXNBAGAME)

        if league == "NCAAB": series_ticker = "KXNCAAMBGAME"

        if not series_ticker:
            print(f"Skipping {league}: No series ticker found.")
            continue

        try:
            # 2. Call get_events
            # Using status='active' to find live games is WRONG for /events
            # Removing status param as per fix
            events_resp = ki.get_events(series_ticker, limit=100)
            events = events_resp.get("events", [])

            if not events:
                print(f"  No active events found for {series_ticker}.")
                continue

            unknown_codes = set()
            known_codes = set()

            # Select map
            map_to_use = None
            if league == "NBA": map_to_use = NBA_TEAM_CODE_MAP
            elif league == "NFL": map_to_use = NFL_TEAM_CODE_MAP
            elif league == "NHL": map_to_use = NHL_TEAM_CODE_MAP
            elif league == "NCAAB": map_to_use = NCAAB_TEAM_CODE_MAP

            valid_codes = set(map_to_use.values()) if map_to_use else set()
            sample_unknowns = {}

            # 3. Parse Tickers
            for evt in events:
                ticker = evt.get("ticker")
                parsed = parse_event_ticker_codes(ticker)
                if not parsed:
                    continue

                for side in ["home", "away"]:
                    raw_code = parsed.get(side)
                    if not raw_code:
                        continue

                    # Use resolved code for verification (handles aliases)
                    code = resolve_team_code(raw_code, league)

                    if code in valid_codes:
                        known_codes.add(code)
                    else:
                        unknown_codes.add(raw_code)
                        if raw_code not in sample_unknowns:
                            sample_unknowns[raw_code] = ticker

            print(f"  Events Scanned: {len(events)}")
            print(f"  Known Codes Found: {len(known_codes)}")

            if unknown_codes:
                sorted_unknowns = sorted(list(unknown_codes))
                print(f"  UNKNOWN CODES ({len(sorted_unknowns)}): {sorted_unknowns}")
                print("  Sample tickers for unknowns:")
                for code in sorted_unknowns[:5]:
                     print(f"    {code}: {sample_unknowns.get(code)}")
            else:
                print("  All codes matched known aliases/maps!")

        except Exception as e:
            print(f"Exception scanning {league}: {e}")

if __name__ == "__main__":
    discover_codes()
