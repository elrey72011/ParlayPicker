"""
Test Kalshi NBA Matching Debug
Tests the enhanced matching logic with specific NBA games from Jan 23, 2026
"""

import sys
import os
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app_core.kalshi_integrator import KalshiIntegrator, match_game_to_kalshi

def test_nba_game(home_team, away_team, game_time_str):
    """Test matching for a specific NBA game"""
    print(f"\n{'='*80}")
    print(f"Testing: {away_team} @ {home_team}")
    print(f"Time: {game_time_str}")
    print(f"{'='*80}\n")

    # Parse game time
    game_time = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
    if game_time.tzinfo is None:
        game_time = pytz.utc.localize(game_time)

    # Initialize integrator
    kalshi = KalshiIntegrator()

    if not kalshi.api_key:
        print("❌ No Kalshi API key found. Set KALSHI_API_KEY environment variable.")
        return

    # Attempt to match
    result = match_game_to_kalshi(
        league="NBA",
        home_team=home_team,
        away_team=away_team,
        game_time=game_time,
        integrator=kalshi,
        status=None  # Don't filter by status
    )

    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Matched: {result.matched}")
    print(f"Kalshi Available: {result.kalshi_available}")
    print(f"Reason: {result.reason}")
    if result.matched:
        print(f"✅ SUCCESS - Event ID: {result.raw_event_id}")
        print(f"   Label: {result.label}")
        print(f"   Probability: {result.probability}")
        # Show spread/total market availability
        has_spread = bool(getattr(result, 'spread_markets', None))
        has_total = bool(getattr(result, 'total_markets', None))
        spread_count = len(result.spread_markets) if hasattr(result, 'spread_markets') and result.spread_markets else 0
        total_count = len(result.total_markets) if hasattr(result, 'total_markets') and result.total_markets else 0
        print(f"   📊 SPREAD/TOTAL STATUS:")
        print(f"      Winner: ✅ (matched)")
        print(f"      Spread: {'✅' if has_spread else '❌'} ({spread_count} markets)")
        print(f"      Total:  {'✅' if has_total else '❌'} ({total_count} markets)")
    else:
        print(f"❌ FAILED TO MATCH")
        if hasattr(result, 'debug') and result.debug:
            print(f"   Debug Info: {result.debug}")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Test games from the issue description
    # All times are in UTC

    print("\n" + "="*80)
    print("KALSHI NBA MATCHING DEBUG TEST")
    print("Testing games from January 23, 2026")
    print("="*80)

    # Test case 1: Brooklyn @ Boston (expected to fail currently)
    test_nba_game(
        home_team="Boston Celtics",
        away_team="Brooklyn Nets",
        game_time_str="2026-01-24T00:40:00"  # 7:40 PM ET = 00:40 UTC next day
    )

    # Test case 2: OKC @ Indiana (expected to fail currently)
    test_nba_game(
        home_team="Indiana Pacers",
        away_team="Oklahoma City Thunder",
        game_time_str="2026-01-24T01:10:00"  # 8:10 PM ET = 01:10 UTC next day
    )

    # Test case 3: Milwaukee @ Denver (expected to fail currently)
    test_nba_game(
        home_team="Denver Nuggets",
        away_team="Milwaukee Bucks",
        game_time_str="2026-01-24T02:10:00"  # 9:10 PM ET / 7:10 PM MT = 02:10 UTC next day
    )

    # Test case 4: Cleveland @ Sacramento (expected to fail currently)
    test_nba_game(
        home_team="Sacramento Kings",
        away_team="Cleveland Cavaliers",
        game_time_str="2026-01-24T03:00:00"  # 10:00 PM ET / 7:00 PM PT = 03:00 UTC next day
    )

    # Test case 5: Memphis @ New Orleans (expected to work)
    test_nba_game(
        home_team="New Orleans Pelicans",
        away_team="Memphis Grizzlies",
        game_time_str="2026-01-24T01:00:00"  # Approximate time
    )

    # ============================================
    # NHL TESTS - Verify spread/total fix works for NHL too
    # ============================================
    print("\n" + "="*80)
    print("KALSHI NHL MATCHING DEBUG TEST")
    print("Testing NHL games for spread/total fix verification")
    print("="*80)

    def test_nhl_game(home_team, away_team, game_time_str):
        """Test matching for a specific NHL game"""
        print(f"\n{'='*80}")
        print(f"Testing: {away_team} @ {home_team}")
        print(f"Time: {game_time_str}")
        print(f"{'='*80}\n")

        game_time = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
        if game_time.tzinfo is None:
            game_time = pytz.utc.localize(game_time)

        kalshi = KalshiIntegrator()
        if not kalshi.api_key:
            print("❌ No Kalshi API key found. Set KALSHI_API_KEY environment variable.")
            return

        result = match_game_to_kalshi(
            league="NHL",
            home_team=home_team,
            away_team=away_team,
            game_time=game_time,
            integrator=kalshi,
            status=None
        )

        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(f"Matched: {result.matched}")
        print(f"Kalshi Available: {result.kalshi_available}")
        print(f"Reason: {result.reason}")
        if result.matched:
            print(f"✅ SUCCESS - Event ID: {result.raw_event_id}")
            print(f"   Label: {result.label}")
            has_spread = bool(getattr(result, 'spread_markets', None))
            has_total = bool(getattr(result, 'total_markets', None))
            spread_count = len(result.spread_markets) if hasattr(result, 'spread_markets') and result.spread_markets else 0
            total_count = len(result.total_markets) if hasattr(result, 'total_markets') and result.total_markets else 0
            print(f"   📊 SPREAD/TOTAL STATUS:")
            print(f"      Winner: ✅ (matched)")
            print(f"      Spread: {'✅' if has_spread else '❌'} ({spread_count} markets)")
            print(f"      Total:  {'✅' if has_total else '❌'} ({total_count} markets)")
        else:
            print(f"❌ FAILED TO MATCH")
        print("="*80 + "\n")

    # NHL Test: Anaheim @ Vancouver (from issue description)
    test_nhl_game(
        home_team="Vancouver Canucks",
        away_team="Anaheim Ducks",
        game_time_str="2026-01-29T03:00:00"  # Approximate time
    )

    # NHL Test: Boston @ Nashville
    test_nhl_game(
        home_team="Nashville Predators",
        away_team="Boston Bruins",
        game_time_str="2026-01-28T01:00:00"  # Approximate time
    )

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
