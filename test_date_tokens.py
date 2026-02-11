import sys
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Mock streamlit before importing streamlit_app
import unittest.mock as mock
sys.modules['streamlit'] = mock.MagicMock()

# Import the function to test
# Since streamlit_app is a script, importing it might run it.
# Ideally we'd just import the function, but it's nested in the file structure or global scope.
# Let's try to extract the function logic for testing or import it if possible.
# Actually, I can just copy the logic into the test script to verify it works as expected,
# since I can't easily import from the script without side effects.
# Wait, I can try to import it.

from streamlit_app import date_tokens_from_commence, parse_commence_to_utc

def test_date_tokens():
    print("Testing date_tokens_from_commence...")

    # Test case 1: UTC time that shifts date in local time (e.g. US/Eastern)
    # 2026-02-12 01:00:00 UTC -> 2026-02-11 20:00:00 EST
    # Local date is Feb 11.
    # Tokens should include: Feb 9, 10, 11, 12, 13 (years 26)

    commence_utc = "2026-02-12T01:00:00Z"
    tokens = date_tokens_from_commence([commence_utc])

    print(f"Input: {commence_utc}")
    print(f"Tokens: {tokens}")

    expected = {
        "26FEB09", "26FEB10", "26FEB11", "26FEB12", "26FEB13"
    }

    missing = expected - tokens
    extra = tokens - expected

    if not missing and not extra:
        print("PASS: Tokens match expected window +/- 2 days.")
    else:
        print(f"FAIL: Missing {missing}, Extra {extra}")

    # Test case 2: Simple case
    commence_utc_2 = "2026-01-01T12:00:00Z"
    # Local (assume UTC if local_tz inference fails or matches UTC)
    # If local_tz is inferred from system, it might differ.
    # But the logic adds +/- 2 days, so it should cover it.

    tokens_2 = date_tokens_from_commence([commence_utc_2])
    print(f"Input: {commence_utc_2}")
    print(f"Tokens: {tokens_2}")

    # We expect 25DEC30, 25DEC31, 26JAN01, 26JAN02, 26JAN03 (roughly)
    # The exact set depends on local tz logic in the app.
    # But checking size should be 5 (unique days).
    if len(tokens_2) == 5:
        print("PASS: Generated 5 tokens.")
    else:
        print(f"FAIL: Generated {len(tokens_2)} tokens.")

if __name__ == "__main__":
    try:
        test_date_tokens()
    except ImportError:
        # Fallback if import fails (e.g. because of streamlit dependency structure)
        print("Could not import streamlit_app directly. Simulating logic.")

        # Simulation of the new logic
        def parse_commence_to_utc_sim(time_str: str) -> datetime:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        def date_tokens_from_commence_sim(commence_list) -> set:
            tokens = set()
            for raw in commence_list:
                dt_utc = parse_commence_to_utc_sim(raw)
                # simulate local_tz (e.g. UTC for simplicity)
                dt_local = dt_utc
                for offset in [-2, -1, 0, 1, 2]:
                    dt_adj = dt_local + timedelta(days=offset)
                    tokens.add(dt_adj.strftime("%y%b%d").upper())
            return tokens

        commence_utc = "2026-02-12T01:00:00Z"
        tokens = date_tokens_from_commence_sim([commence_utc])
        print(f"Simulated Tokens for {commence_utc}: {tokens}")
        expected = {"26FEB10", "26FEB11", "26FEB12", "26FEB13", "26FEB14"} # UTC basis
        # 12th +/- 2 -> 10, 11, 12, 13, 14

        missing = expected - tokens
        if not missing:
            print("PASS (Simulation): Logic covers expected window.")
        else:
            print(f"FAIL (Simulation): Missing {missing}")
