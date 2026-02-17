import os
import sys
from unittest.mock import MagicMock

# Mock streamlit before importing app_core
import streamlit
streamlit.secrets = MagicMock()
streamlit.secrets.get = MagicMock(return_value=None)

# Set dummy env vars for keys to pass intialization if they check existence (though we mocked secrets)
os.environ["KALSHI_API_KEY"] = "dummy_key"
os.environ["KALSHI_API_SECRET"] = "dummy_secret"

from app_core.kalshi_integrator import KalshiIntegrator, NCAAB_TEAM_CODE_MAP

def verify_fixes():
    print("--- Verifying Fixes ---")

    # 1. Check Aliases
    aliases = ["HOLY CROSS", "LOYOLA MD", "UTSA", "CHARLOTTE", "OMAHA", "DENVER"]
    print("Checking Aliases:")
    for a in aliases:
        if a in NCAAB_TEAM_CODE_MAP:
            print(f"  ✅ {a} -> {NCAAB_TEAM_CODE_MAP[a]}")
        else:
            print(f"  ❌ {a} MISSING")

    # 2. Check KalshiIntegrator instantiation
    try:
        kalshi = KalshiIntegrator()
        print("\n✅ KalshiIntegrator initialized successfully.")
    except Exception as e:
        print(f"\n❌ KalshiIntegrator initialization failed: {e}")

    # 3. We cannot easily check _match_via_events logic without mocking a full game and API response,
    # but successful import suggests syntax is correct.

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_fixes()
