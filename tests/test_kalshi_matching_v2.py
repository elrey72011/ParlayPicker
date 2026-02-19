import unittest
from datetime import datetime, timezone
import pytz
from unittest.mock import MagicMock, patch

# Import the class to test (assuming app_core is in python path)
from app_core.kalshi_integrator import (
    KalshiIntegrator,
    generate_comprehensive_team_variants,
    canonical_team_name,
    calculate_game_match_score,
    calculate_team_match_score
)

class TestKalshiMatchingV2(unittest.TestCase):
    def setUp(self):
        # Mock API credentials to bypass __init__ checks
        self.integrator = KalshiIntegrator(api_key="mock", api_secret="mock", required=False)

    def test_canonical_normalization(self):
        """Verify normalization logic handles special cases"""
        cases = [
            ("North Texas Mean Green", "NORTH TEXAS"), # Mascot stripping
            ("Hawai'i Rainbow Warriors", "HAWAII"), # Punctuation + Mascot
            ("Saint Mary's", "ST MARYS"), # Saint -> St, apostrophe
            ("Florida Int'l", "FLORIDA INTERNATIONAL"), # Int'l -> International
            ("Arkansas-Little Rock", "ARKANSAS LITTLE ROCK"), # Hyphen -> Space
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(canonical_team_name(inp), expected)

    def test_exact_code_bonus(self):
        """Verify exact code matches get 100 points regardless of length"""
        # Cal Poly -> CP
        score, match = calculate_team_match_score("CP", ["CAL POLY", "CP", "CPM"])
        self.assertEqual(score, 100.0)
        self.assertEqual(match, "CP")

        # North Texas -> UNT
        score, match = calculate_team_match_score("UNT", ["NORTH TEXAS", "UNT"])
        self.assertEqual(score, 100.0)
        self.assertEqual(match, "UNT")

    def test_target_failures(self):
        """
        Replicate the specific failure cases identified in diagnostics.
        These failed because scores were < 80.0 or mappings were missing.
        """
        # Case 1: Arkansas St (Red Wolves) vs Louisiana (Ragin Cajuns)
        # Ticker: KXNCAAMBSPREAD-26FEB19ARSTULL-ULL7
        # Codes: ARST vs ULL
        # Variants should include: ARST, ULL
        home_team = "Louisiana Ragin' Cajuns"
        away_team = "Arkansas St Red Wolves"

        # Verify variants generation
        home_vars = generate_comprehensive_team_variants(home_team, "NCAAB")
        away_vars = generate_comprehensive_team_variants(away_team, "NCAAB")

        self.assertIn("ULL", home_vars)
        self.assertIn("ARST", away_vars)

        # Verify Matching Score
        ticker = "KXNCAAMBSPREAD-26FEB19ARSTULL-ULL7"
        score, details = calculate_game_match_score(ticker, away_vars, home_vars)

        # Should be high (>= 90.0). ARST (4-char) gets 100, ULL (3-char boundary) gets 90.
        self.assertGreaterEqual(score, 90.0)

        # Case 2: Cal Poly vs Hawai'i
        # Ticker: ...CPHAW...
        # Codes: CP vs HAW
        home_team = "Hawai'i Rainbow Warriors"
        away_team = "Cal Poly Mustangs"

        home_vars = generate_comprehensive_team_variants(home_team, "NCAAB")
        away_vars = generate_comprehensive_team_variants(away_team, "NCAAB")

        # Check normalization/mapping
        self.assertIn("HAW", home_vars) # From map
        self.assertIn("CP", away_vars)  # From map

        ticker = "KXNCAAMBSPREAD-26FEB19CPHAW-HAW10.5"
        score, details = calculate_game_match_score(ticker, away_vars, home_vars)
        self.assertGreaterEqual(score, 90.0)

    def test_golden_set_stability(self):
        """Ensure we didn't break existing good matches"""
        # NBA: Boston vs GSW
        home = "Golden State Warriors"
        away = "Boston Celtics"
        ticker = "KXNBASPREAD-26FEB19BOSGSW-BOS5.5"

        home_vars = generate_comprehensive_team_variants(home, "NBA")
        away_vars = generate_comprehensive_team_variants(away, "NBA")

        score, details = calculate_game_match_score(ticker, away_vars, home_vars)
        self.assertGreaterEqual(score, 90.0)

    def test_date_proximity_scoring(self):
        """Verify date proximity logic in match_game_to_kalshi_markets"""
        # Mocking the API response using real codes DUK/UNC so parser works
        mock_markets = [{
            "ticker": "KXNCAAMBGAME-26FEB19DUKUNC",
            "event_ticker": "KXNCAAMBGAME-26FEB19DUKUNC",
            "title": "UNC vs Duke",
            "expiration_time": "2026-02-19T12:00:00Z" # Noon UTC
        }]

        # Target Game: 1 hour difference (Should get +5 bonus)
        game_time_close = datetime(2026, 2, 19, 13, 0, tzinfo=timezone.utc)

        # Target Game: 25 hours difference (Should get -5 penalty)
        game_time_far = datetime(2026, 2, 20, 13, 0, tzinfo=timezone.utc)

        # 1. Close match
        res_close = self.integrator.match_game_to_kalshi_markets(
            "Duke", "North Carolina", "26FEB19", mock_markets, "NCAAB", game_time_close
        )
        # Assuming it matched
        if not res_close["GAME"]:
            self.fail("Failed to match Duke vs UNC in close test")
        score_close = res_close["GAME"][0]["_match_score"]

        # 2. Far match
        res_far = self.integrator.match_game_to_kalshi_markets(
            "Duke", "North Carolina", "26FEB19", mock_markets, "NCAAB", game_time_far
        )
        if not res_far["GAME"]:
            self.fail("Failed to match Duke vs UNC in far test")
        score_far = res_far["GAME"][0]["_match_score"]

        # Close score should be higher
        self.assertGreater(score_close, score_far, f"Close score {score_close} should be > Far score {score_far}")

if __name__ == '__main__':
    unittest.main()
