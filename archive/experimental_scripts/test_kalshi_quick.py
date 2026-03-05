import os
import sys
sys.path.append('.')

from app_core.kalshi_integrator import KalshiIntegrator, match_game_to_kalshi
from datetime import datetime
import pytz

# Initialize
ki = KalshiIntegrator()

print(f"Credentials Valid: {ki._credentials_valid}")
print(f"API Key: {ki.api_key[:10] if ki.api_key else 'NONE'}...")

# Test the exact game
result = match_game_to_kalshi(
    league="NCAAB",
    home_team="Rhode Island Rams",
    away_team="Saint Louis Billikens",
    game_time=datetime(2026, 2, 18, 0, 2, 48, tzinfo=pytz.UTC),
    integrator=ki,
    status=None,
    requested_market_type="SPREAD"
)

print(f"\nMatch Result:")
print(f"  Matched: {result.matched}")
print(f"  Reason: {result.reason}")
print(f"  Kalshi Available: {result.kalshi_available}")
if result.debug:
    print(f"  Debug: {result.debug}")
