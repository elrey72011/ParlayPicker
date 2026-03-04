import unittest
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_core.feature_processing import _parse_streak, _parse_form

class TestFeatureProcessing(unittest.TestCase):

    def test_parse_streak_happy_paths(self):
        # Normal win streaks
        self.assertEqual(_parse_streak("W1"), 1.0)
        self.assertEqual(_parse_streak("W3"), 3.0)
        self.assertEqual(_parse_streak("W10"), 10.0)

        # Normal loss streaks
        self.assertEqual(_parse_streak("L1"), -1.0)
        self.assertEqual(_parse_streak("L4"), -4.0)
        self.assertEqual(_parse_streak("L12"), -12.0)

    def test_parse_streak_edge_cases(self):
        # Zeros
        self.assertEqual(_parse_streak("W0"), 0.0)
        self.assertEqual(_parse_streak("L0"), 0.0)

        # Lowercase casing behavior
        # Note: Current implementation does `val if streak_str.startswith('W') else -val`.
        # So "w3" -> -3.0. We test this explicit behavior for now.
        self.assertEqual(_parse_streak("w3"), -3.0)
        self.assertEqual(_parse_streak("l2"), -2.0)

        # Missing input
        self.assertEqual(_parse_streak(None), 0.0)
        self.assertEqual(_parse_streak(""), 0.0)

    def test_parse_streak_error_paths(self):
        # Letters with no numbers
        self.assertEqual(_parse_streak("W"), 0.0)
        self.assertEqual(_parse_streak("L"), 0.0)

        # Strings without letters
        # Note: current implementation does `int(streak_str[1:])`.
        # So "123" gives int("23") = 23, and since it doesn't start with 'W', it returns -23.0.
        self.assertEqual(_parse_streak("123"), -23.0)

        # "3" gives int("") -> throws exception -> returns 0.0
        self.assertEqual(_parse_streak("3"), 0.0)

        # Invalid characters or malformed
        # "W-1" -> int("-1") -> -1. Since it starts with 'W', it returns -1.0.
        self.assertEqual(_parse_streak("W-1"), -1.0)

        # "W1.5" -> int("1.5") -> raises ValueError -> returns 0.0
        self.assertEqual(_parse_streak("W1.5"), 0.0)
        self.assertEqual(_parse_streak("XYZ"), 0.0)
        self.assertEqual(_parse_streak("WXYZ"), 0.0)
        self.assertEqual(_parse_streak("   "), 0.0)

    def test_parse_form_happy_paths(self):
        # All wins
        self.assertEqual(_parse_form("WWWWW"), 1.0)
        self.assertEqual(_parse_form("W"), 1.0)

        # All losses
        self.assertEqual(_parse_form("LLLLL"), 0.0)
        self.assertEqual(_parse_form("L"), 0.0)

        # Mixed
        self.assertEqual(_parse_form("WWLWL"), 0.6)
        self.assertEqual(_parse_form("LWLWL"), 0.4)

        # Mixed casing
        self.assertEqual(_parse_form("wWlWl"), 0.6)

    def test_parse_form_edge_and_error_paths(self):
        # Missing or empty
        self.assertEqual(_parse_form(None), 0.5)
        self.assertEqual(_parse_form(""), 0.5)

        # Non-standard characters (ignored as losses if they aren't 'W')
        self.assertEqual(_parse_form("WWXYZ"), 0.4) # 2 wins out of 5 chars
        self.assertEqual(_parse_form("   "), 0.0) # 0 wins out of 3 chars

if __name__ == '__main__':
    unittest.main()
