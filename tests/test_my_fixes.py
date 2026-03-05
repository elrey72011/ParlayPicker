import unittest
from app_core.kalshi_integrator import KALSHI_NCAAB_TEAM_CODES

class TestMyFixes(unittest.TestCase):
    def test_team_codes(self):
        self.assertEqual(KALSHI_NCAAB_TEAM_CODES.get("Merrimack"), "MRMK")

if __name__ == '__main__':
    unittest.main()
