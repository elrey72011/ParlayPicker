import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

def verify_alias_map():
    print("Verifying Alias Map...")
    try:
        from app_core.theover_ingest import TEAM_ALIAS_MAP
        expected = {
            "Ottawa": "Ottawa Senators",
            "Winnipeg": "Winnipeg Jets",
            "LA": "Los Angeles Kings",
            "SJ": "San Jose Sharks",
            "NY Islanders": "New York Islanders"
        }
        for k, v in expected.items():
            if TEAM_ALIAS_MAP.get(k) != v:
                print(f"FAIL: {k} maps to {TEAM_ALIAS_MAP.get(k)}, expected {v}")
                return False
        print("PASS: Alias Map verification successful.")
        return True
    except Exception as e:
        print(f"FAIL: Could not import or verify map: {e}")
        return False

def verify_weights():
    print("\nVerifying Weights in streamlit_app.py...")
    try:
        with open("streamlit_app.py", "r") as f:
            content = f.read()

        # Check hardcoded block
        checks = [
            '"ml_weight": 0.20',
            '"sentiment_weight": 0.05',
            '"kalshi_weight": 0.55',
            '"odds_weight": 0.15',
            '"theover_weight": 0.10'
        ]

        all_passed = True
        for c in checks:
            if c not in content:
                print(f"FAIL: Could not find string '{c}'")
                all_passed = False
            else:
                print(f"PASS: Found '{c}'")

        return all_passed
    except Exception as e:
        print(f"FAIL: {e}")
        return False

def verify_sharp_money():
    print("\nVerifying Sharp Money UI...")
    try:
        with open("streamlit_app.py", "r") as f:
            content = f.read()

        if 'top_df_display["Sharp Money"]' in content and '"💰"' in content and '"🔥"' in content:
             print("PASS: Sharp Money column logic found.")
             return True
        else:
             print("FAIL: Sharp Money logic missing.")
             return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

def verify_time_buffer():
    print("\nVerifying Time Buffer in theover_ingest.py...")
    try:
        with open("app_core/theover_ingest.py", "r") as f:
            content = f.read()

        if 'diff_hours <= 12' in content:
            print("PASS: 12-hour buffer check found.")
            return True
        else:
            print("FAIL: 12-hour buffer logic missing.")
            return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    r1 = verify_alias_map()
    r2 = verify_weights()
    r3 = verify_sharp_money()
    r4 = verify_time_buffer()

    if r1 and r2 and r3 and r4:
        print("\nALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print("\nSOME VERIFICATIONS FAILED")
        sys.exit(1)
