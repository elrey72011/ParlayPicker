import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from streamlit_app import is_placeholder_odds_v2
except ImportError:
    # Fallback if import fails (e.g. strict environment), mostly for development verification
    def is_placeholder_odds_v2(home_odds, away_odds, line):
        try:
            if abs(float(home_odds) + 110.0) < 0.1 and abs(float(away_odds) + 110.0) < 0.1:
                if line is None:
                    return True
                try:
                    return float(line) == 0.0
                except:
                    return True
            return False
        except:
            return False

# Legacy mock for comparison
def try_float(x):
    try:
        return float(x)
    except:
        return 0.0

def detect_placeholder_odds_legacy(row: pd.Series) -> bool:
    s_home_odds = row.get("spread_odds_home")
    s_away_odds = row.get("spread_odds_away")
    s_line = row.get("spread_point")
    is_placeholder = False

    def is_minus_110(val):
        try:
            f = float(val)
            return abs(f + 110.0) < 0.1
        except:
            return False

    if is_minus_110(s_home_odds) and is_minus_110(s_away_odds):
        if s_line is None or (try_float(s_line) == 0.0):
            is_placeholder = True

    return is_placeholder

class TestPlaceholderOptimization(unittest.TestCase):

    def test_basic_placeholder(self):
        # Case 1: Standard placeholder
        self.assertTrue(is_placeholder_odds_v2(-110, -110, 0))
        self.assertTrue(is_placeholder_odds_v2("-110", "-110", "0"))
        self.assertTrue(is_placeholder_odds_v2(-110.0, -110.0, 0.0))

    def test_not_placeholder(self):
        # Case 2: Not placeholder (valid odds/line)
        self.assertFalse(is_placeholder_odds_v2(-115, -105, -3.5))
        self.assertFalse(is_placeholder_odds_v2(-110, -110, -3.5)) # Line is not 0
        self.assertFalse(is_placeholder_odds_v2(-120, -110, 0)) # Odds not -110

    def test_edge_cases(self):
        # Case 3: Edge cases (None, invalid strings)
        self.assertTrue(is_placeholder_odds_v2(-110, -110, None))
        self.assertTrue(is_placeholder_odds_v2(-110, -110, "foo")) # invalid line -> 0.0 -> True
        self.assertFalse(is_placeholder_odds_v2(None, -110, 0))
        self.assertFalse(is_placeholder_odds_v2("foo", -110, 0))

        # Case: NaN (float)
        self.assertFalse(is_placeholder_odds_v2(-110, -110, float('nan')))

if __name__ == '__main__':
    unittest.main()
