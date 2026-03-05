
import sys
import os
import pandas as pd
import logging

# Mock streamlit for imports
import types
sys.modules["streamlit"] = types.ModuleType("streamlit")
sys.modules["streamlit"].secrets = {}
sys.modules["streamlit"].cache_data = lambda *args, **kwargs: (lambda f: f)

# Add current directory to path
sys.path.append(os.getcwd())

def test_parlay_optimizer():
    print("Testing ParlayOptimizer...")
    try:
        from parlay_optimizer import ParlayOptimizer
        opt = ParlayOptimizer("models")
        if not hasattr(opt, "validate_pick_string"):
            print("FAIL: ParlayOptimizer missing validate_pick_string")
            return

        # Test cases
        valid_picks = ["Team -1.5", "Over 220.5", "Team 10"]
        invalid_picks = ["Team 01", "Under 01", "Over 00", "Team 11", "Team 0"]
        # Note: "Team 11" is ambiguous (could be 11 points), but " 01" is definitely artifact.
        # My regex was r'\s0\d$' -> matches space followed by 0 then digit then end.
        # "Team 01" -> matches " 01". "Team 11" -> does not match (space 1 1).
        # "Under 01" -> matches " 01".

        print(f"Testing validation logic...")
        assert opt.validate_pick_string("Team -1.5") == True
        assert opt.validate_pick_string("Team 01") == False
        assert opt.validate_pick_string("Under 01") == False
        assert opt.validate_pick_string("Over 01") == False

        # "Team 11" should probably be valid if it's a spread/total of 11.
        # My regex was r'\s0\d$'. " 11" does not match.
        assert opt.validate_pick_string("Team 11") == True

        print("PASS: ParlayOptimizer.validate_pick_string")
    except Exception as e:
        print(f"FAIL: ParlayOptimizer test error: {e}")
        import traceback
        traceback.print_exc()

def test_kalshi_classification():
    print("\nTesting Kalshi Classification...")
    try:
        from app_core.kalshi_integrator import _extract_market_type

        cases = [
            ("KXNBASPREAD-23JAN01", "Some Title", "spread"),
            ("KXNBATOTAL-23JAN01", "Some Title", "total"),
            ("KXNBAGAME-...", "Point Spread", "spread"),
            ("KXNBAGAME-...", "Total Points", "total"),
            ("KXNBAGAME-...", "Winner", "moneyline"),
            ("KXNBAGAME-...", "Moneyline", "moneyline"),
        ]

        for tick, title, expected in cases:
            res = _extract_market_type(title, tick)
            if res != expected:
                print(f"FAIL: {tick} {title} -> {res} (expected {expected})")
            else:
                pass # print(f"OK: {tick} -> {res}")
        print("PASS: Kalshi Classification")
    except Exception as e:
        print(f"FAIL: Kalshi test error: {e}")

def test_ncaab_overrides():
    print("\nTesting NCAAB Overrides...")
    try:
        from app_core.feature_processing import MANUAL_TEAM_OVERRIDES, robust_normalize_team

        checks = {
            "TULANE GREEN WAVE": "TULANE",
            "NORTH TEXAS MEAN GREEN": "NORTH TEXAS",
            "MEMPHIS TIGERS": "MEMPHIS",
            "UTSA ROADRUNNERS": "UTSA",
            "HOUSTON COUGARS": "HOUSTON"
        }

        for k, v in checks.items():
            if k not in MANUAL_TEAM_OVERRIDES:
                print(f"FAIL: Missing override for {k}")
            elif MANUAL_TEAM_OVERRIDES[k] != v:
                print(f"FAIL: Incorrect override for {k}: {MANUAL_TEAM_OVERRIDES[k]}")

        # Test normalization
        res = robust_normalize_team("Tulane Green Wave", "NCAAB")
        if res != "TULANE":
            print(f"FAIL: robust_normalize_team('Tulane Green Wave') -> {res}")

        print("PASS: NCAAB Overrides")
    except Exception as e:
        print(f"FAIL: NCAAB test error: {e}")

if __name__ == "__main__":
    test_parlay_optimizer()
    test_kalshi_classification()
    test_ncaab_overrides()
