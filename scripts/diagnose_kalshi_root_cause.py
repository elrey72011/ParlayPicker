import os
import sys
import logging
import json
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_core.kalshi_integrator import KalshiIntegrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_secrets():
    """Try to load secrets from .streamlit/secrets.toml if st.secrets is empty."""
    # Monkeypatch st.secrets to avoid StreamlitSecretNotFoundError if running standalone
    try:
        if not hasattr(st, "secrets"):
            st.secrets = {}
        # Try accessing to see if it raises
        _ = st.secrets.get("KALSHI_API_KEY")
    except Exception:
        # If it raises, replace it with a plain dict
        st.secrets = {}

    try:
        import toml
        secrets_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            print(f"Loading secrets from {secrets_path}")
            with open(secrets_path, 'r') as f:
                secrets = toml.load(f)
                # Populate env vars
                if "KALSHI_API_KEY" in secrets:
                    os.environ["KALSHI_API_KEY"] = secrets["KALSHI_API_KEY"]
                    st.secrets["KALSHI_API_KEY"] = secrets["KALSHI_API_KEY"]
                if "KALSHI_API_SECRET" in secrets:
                    os.environ["KALSHI_API_SECRET"] = secrets["KALSHI_API_SECRET"]
                    st.secrets["KALSHI_API_SECRET"] = secrets["KALSHI_API_SECRET"]
    except ImportError:
        print("toml module not found, skipping manual secrets load")
    except Exception as e:
        print(f"Error loading secrets: {e}")

def run_diagnostic():
    print("--- Kalshi Diagnostic Script ---")

    # Load secrets
    load_secrets()

    # Check Env Vars
    api_key = os.getenv("KALSHI_API_KEY")
    api_secret = os.getenv("KALSHI_API_SECRET")

    print(f"Environment Variable 'KALSHI_API_KEY': {'SET' if api_key else 'MISSING'}")
    if api_key:
        print(f"  Value: {api_key[:8]}... (Length: {len(api_key)})")
        if api_key == "your_key_here":
            print("  ❌ ERROR: Key is default placeholder 'your_key_here'")

    print(f"Environment Variable 'KALSHI_API_SECRET': {'SET' if api_secret else 'MISSING'}")

    # Initialize Integrator
    print("\nInitializing KalshiIntegrator...")
    try:
        ki = KalshiIntegrator()

        print(f"Integrator API Key: {ki.api_key[:8] if ki.api_key else 'NONE'}...")
        print(f"Credentials Valid Flag: {getattr(ki, '_credentials_valid', 'ATTR MISSING')}")

        if not getattr(ki, '_credentials_valid', False):
            print("❌ Integrator reports invalid credentials.")
            return

        # Health Check
        print("\nRunning Health Check (NBA)...")
        health = ki.health_check("NBA")
        print(json.dumps(health, indent=2))

        # Force Fresh Data & Check Events
        print("\n--- Fix #2: Force Fresh Data & Check Events ---")
        print("Clearing events cache...")
        ki.clear_events_cache()

        series_ticker = "KXNBAGAME"
        print(f"Fetching events for {series_ticker} (no status filter)...")
        try:
            events_resp = ki.get_events(series_ticker, status=None, use_cache=False)
            events = events_resp.get('events', [])
            print(f"✅ Kalshi Events Fetched: {len(events)}")

            if events:
                print("Sample Event:")
                print(json.dumps(events[0], indent=2))
            else:
                print("⚠️ No events returned.")

        except Exception as e:
            print(f"❌ Kalshi Fetch Failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Failed to initialize integrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
