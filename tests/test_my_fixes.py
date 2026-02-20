import unittest
from app_core.kalshi_integrator import KALSHI_NCAAB_TEAM_CODES, normalize_team_for_kalshi, generate_comprehensive_team_variants

class TestMyFixes(unittest.TestCase):
    def test_team_codes(self):
        # Verify codes I added
        self.assertEqual(KALSHI_NCAAB_TEAM_CODES.get("Merrimack"), "MER")
        self.assertEqual(KALSHI_NCAAB_TEAM_CODES.get("Purdue"), "PUR")
        self.assertEqual(KALSHI_NCAAB_TEAM_CODES.get("Indiana"), "IND")

        # Verify normalization
        self.assertEqual(normalize_team_for_kalshi("Merrimack Warriors"), "MER")
        self.assertEqual(normalize_team_for_kalshi("Purdue Boilermakers"), "PUR")
        self.assertEqual(normalize_team_for_kalshi("Indiana Hoosiers"), "IND")

    def test_variants(self):
        # Verify variants include the code
        mer_vars = generate_comprehensive_team_variants("Merrimack Warriors", "NCAAB")
        self.assertIn("MER", mer_vars)

        pur_vars = generate_comprehensive_team_variants("Purdue Boilermakers", "NCAAB")
        self.assertIn("PUR", pur_vars)

        ind_vars = generate_comprehensive_team_variants("Indiana Hoosiers", "NCAAB")
        self.assertIn("IND", ind_vars)

if __name__ == '__main__':
    unittest.main()
