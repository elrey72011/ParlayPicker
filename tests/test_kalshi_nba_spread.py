
import sys
import os
import logging
from typing import Dict, Any, List

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_core.kalshi_integrator import match_nba_spread, canonical_team_name

# Mock logging
logging.basicConfig(level=logging.INFO)

def test_match_nba_spread():
    print("Testing match_nba_spread...")

    # Scenario 1: Favorite (Toronto -6.5)
    # Target: Toronto wins by > 6.5
    # Expect: Match to "Toronto wins by over 6.5 Points?" (YES side)
    row_fav = {
        'spread_pick_team': 'Toronto Raptors',
        'spread_pick_line': -6.5,
        'Home': 'Toronto Raptors',
        'Away': 'Chicago Bulls'
    }

    markets_fav = [
        {
            'yes_side': 'Toronto wins by over 6.5 Points?',
            'ticker': 'KXNBASPREAD-TOR6.5',
            'probability': 0.52
        },
        {
            'yes_side': 'Chicago wins by over 6.5 Points?', # Wrong team
            'ticker': 'KXNBASPREAD-CHI6.5',
            'probability': 0.48
        }
    ]

    match_fav = match_nba_spread(row_fav, markets_fav)
    assert match_fav is not None, "Should match favorite"
    assert match_fav['market']['ticker'] == 'KXNBASPREAD-TOR6.5'
    assert match_fav['kalshi_prob_for_pick'] == 0.52
    print("✅ Scenario 1 (Favorite) Passed")

    # Scenario 2: Underdog (Brooklyn +16)
    # Target: Cleveland wins by > 16 (approx) -> NO side
    # Expect: Match to "Cleveland wins by over 15.5 Points?" (NO side)
    row_dog = {
        'spread_pick_team': 'Brooklyn Nets',
        'spread_pick_line': 16.0,
        'Home': 'Cleveland Cavaliers',
        'Away': 'Brooklyn Nets'
    }

    markets_dog = [
        {
            'yes_side': 'Cleveland wins by over 15.5 Points?', # Correct market
            'ticker': 'KXNBASPREAD-CLE15.5',
            'probability': 0.55
        },
        {
            'yes_side': 'Brooklyn wins by over 7.5 Points?', # Trap market (wrong spread logic)
            'ticker': 'KXNBASPREAD-BKN7.5',
            'probability': 0.10
        }
    ]

    match_dog = match_nba_spread(row_dog, markets_dog)
    assert match_dog is not None, "Should match underdog"
    assert match_dog['market']['ticker'] == 'KXNBASPREAD-CLE15.5', "Should match Opponent market"
    # Probability should be 1.0 - 0.55 = 0.45
    assert abs(match_dog['kalshi_prob_for_pick'] - 0.45) < 0.001, f"Prob should be inverted. Got {match_dog['kalshi_prob_for_pick']}"
    print("✅ Scenario 2 (Underdog) Passed")

    # Scenario 3: Underdog (Atlanta +1)
    # Target: Philadelphia wins by > 1 -> NO side
    # Expect: Match to "Philadelphia wins by over 0.5 Points?" (NO side)
    row_dog2 = {
        'spread_pick_team': 'Atlanta Hawks',
        'spread_pick_line': 1.0,
        'Home': 'Philadelphia 76ers',
        'Away': 'Atlanta Hawks'
    }

    markets_dog2 = [
        {
            'yes_side': 'Philadelphia wins by over 0.5 Points?', # Correct market (close enough)
            'ticker': 'KXNBASPREAD-PHI0.5',
            'probability': 0.60
        }
    ]

    match_dog2 = match_nba_spread(row_dog2, markets_dog2)
    assert match_dog2 is not None
    assert match_dog2['market']['ticker'] == 'KXNBASPREAD-PHI0.5'
    assert abs(match_dog2['kalshi_prob_for_pick'] - 0.40) < 0.001
    print("✅ Scenario 3 (Underdog Close) Passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_match_nba_spread()
