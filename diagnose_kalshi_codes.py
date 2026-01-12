"""
Diagnostic script to auto-validate Kalshi team codes against internal maps.
Scans active events for NBA, NFL, NHL, MLB, NCAAF, NCAAB and checks if parsed codes are known.
"""
import os
import sys
import logging
from typing import Dict, List, Any

# Ensure app_core is in path
sys.path.append(os.getcwd())

try:
    from app_core.kalshi_integrator import KalshiIntegrator, parse_event_ticker_codes, resolve_team_code
    from app_core.kalshi_integrator import (
        NBA_TEAM_CODE_MAP, NFL_TEAM_CODE_MAP, NHL_TEAM_CODE_MAP,
        MLB_TEAM_CODE_MAP, NCAAF_TEAM_CODE_MAP, NCAAB_TEAM_CODE_MAP
    )
except ImportError:
    print("Error: Could not import KalshiIntegrator or maps. Run from repo root.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose_kalshi")

LEAGUES_TO_CHECK = ["NBA", "NFL", "NHL", "MLB", "NCAAF", "NCAAB"]

def get_code_map(league: str) -> Dict[str, str]:
    if league == "NBA": return NBA_TEAM_CODE_MAP
    if league == "NFL": return NFL_TEAM_CODE_MAP
    if league == "NHL": return NHL_TEAM_CODE_MAP
    if league == "MLB": return MLB_TEAM_CODE_MAP
    if league == "NCAAF": return NCAAF_TEAM_CODE_MAP
    if league == "NCAAB": return NCAAB_TEAM_CODE_MAP
    return {}

def run_diagnosis():
    # Check keys
    key = os.getenv("KALSHI_API_KEY")
    secret = os.getenv("KALSHI_API_SECRET")

    if not key or not secret:
        # Try to load from streamlit secrets if available (local dev)
        try:
            import streamlit as st
            key = st.secrets.get("KALSHI_API_KEY")
            secret = st.secrets.get("KALSHI_API_SECRET")
        except:
            pass

    if not key or not secret:
        print("❌ SKIPPING: KALSHI_API_KEY/SECRET not found in env or secrets.")
        return

    kalshi = KalshiIntegrator(api_key=key, api_secret=secret)

    print("\n--- Kalshi Team Code Diagnosis ---")

    total_unknowns = 0

    for league in LEAGUES_TO_CHECK:
        print(f"\nScanning {league}...")
        try:
            result = kalshi.scan_and_verify_team_codes(league)
            if "error" in result:
                print(f"  ⚠️ Error: {result['error']}")
                continue

            print(f"  Events Scanned: {result['events_scanned']}")
            print(f"  Known Codes in Map: {result['known_codes_count']}")

            unknowns = result.get("unknown_codes", [])
            if unknowns:
                print(f"  ❌ UNKNOWN CODES FOUND ({len(unknowns)}):")
                print(f"     {', '.join(unknowns)}")
                total_unknowns += len(unknowns)

                # Show samples
                samples = result.get("sample_unknowns", {})
                for code in unknowns[:5]:
                    if code in samples:
                        print(f"     - {code} found in {samples[code]}")
            else:
                print("  ✅ All extracted codes matched known map.")

        except Exception as e:
            print(f"  ⚠️ Exception scanning {league}: {e}")

    print("\n----------------------------------")
    if total_unknowns > 0:
        print(f"❌ FAIL: Found {total_unknowns} unknown codes total. Update mappings in kalshi_integrator.py.")
    else:
        print("✅ PASS: All active event codes are known.")

if __name__ == "__main__":
    run_diagnosis()
