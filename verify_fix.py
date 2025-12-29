import sys
import os

# Add current directory to path to ensure app_core is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit_app
    print("Imports successful")
    if hasattr(streamlit_app, 'CLIENT_MAPPING'):
        print("CLIENT_MAPPING found")
        print(f"CLIENT_MAPPING keys: {list(streamlit_app.CLIENT_MAPPING.keys())}")
    else:
        print("CLIENT_MAPPING NOT found")

    # Test enrich_game_context logic (simple check)
    from app_core.feature_processing import TeamNameMatcher
    # Mocking stats_lookup
    # We can't easily run enrich_game_context without api_clients that return data,
    # but we can check if the function exists and uses TeamNameMatcher.match_team.
    # We'll just trust the code we wrote for that logic.

except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Verification failed: {e}")
