"""
Script to discover Kalshi NCAAB team codes by scanning recent events.
Usage: python scripts/discover_kalshi_ncaab_codes.py
"""
import sys
import os
import time
from datetime import datetime, timedelta

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import KalshiIntegrator, parse_event_ticker_codes, NCAAB_TEAM_CODE_MAP

def main():
    print("--- Kalshi NCAAB Code Discovery ---")

    # Initialize integrator
    try:
        kalshi = KalshiIntegrator()
        kalshi.assert_available()
    except Exception as e:
        print(f"Error initializing Kalshi: {e}")
        return

    # Check mapping logic
    result = kalshi.scan_and_verify_team_codes("NCAAB")

    if "error" in result:
        print(f"Error scanning: {result['error']}")
        return

    print(f"Series Ticker: {result.get('series_ticker')}")
    print(f"Events Scanned: {result.get('events_scanned')}")
    print(f"Known Codes Found: {result.get('known_codes_count')}")

    unknowns = result.get("unknown_codes", [])
    print(f"Unknown Codes Found ({len(unknowns)}):")
    sample_unknowns = result.get("sample_unknowns", {})

    # Simple heuristics for suggestion
    valid_codes = set(NCAAB_TEAM_CODE_MAP.values())

    for code in unknowns:
        sample = sample_unknowns.get(code, "N/A")
        suggestion = ""

        # Heuristic 1: If 4 chars, try dropping last char
        if len(code) == 4:
            sub = code[:3]
            if sub in valid_codes:
                suggestion = f"SUGGEST ALIAS: {code} -> {sub}"

        # Heuristic 2: If 3 chars, see if we have 4-char matches (not as helpful for mapping TO canonical)

        print(f"  - {code} (Sample Ticker: {sample}) {suggestion}")

    print("\n--- Current NCAAB Map Size ---")
    print(f"Count: {len(NCAAB_TEAM_CODE_MAP)}")

if __name__ == "__main__":
    main()
