#!/usr/bin/env python3
"""
Test script to verify HasKalshiMarket flag computation
"""
import pandas as pd
import sys
sys.path.insert(0, '/home/user/ParlayPicker')

from app_core.new_summary_logic import build_game_summary_v2

# Create sample test data mimicking master_df structure
test_data = {
    "league": ["NBA", "NBA", "NBA", "NBA", "NBA", "NBA"],
    "Home": ["Lakers", "Lakers", "Lakers", "Warriors", "Warriors", "Warriors"],
    "Away": ["Celtics", "Celtics", "Celtics", "Nets", "Nets", "Nets"],
    "Commence (UTC)": ["2026-01-20T00:00:00", "2026-01-20T00:00:00", "2026-01-20T00:00:00",
                        "2026-01-20T01:00:00", "2026-01-20T01:00:00", "2026-01-20T01:00:00"],
    "Commence (Local)": ["2026-01-19T19:00:00", "2026-01-19T19:00:00", "2026-01-19T19:00:00",
                          "2026-01-19T20:00:00", "2026-01-19T20:00:00", "2026-01-19T20:00:00"],
    "Local Date": ["2026-01-19", "2026-01-19", "2026-01-19", "2026-01-19", "2026-01-19", "2026-01-19"],
    "Market": ["Moneyline", "Spread", "Total", "Moneyline", "Spread", "Total"],
    "Pick": ["Lakers", "Lakers", "Over", "Warriors", "Warriors", "Over"],
    "Line": [None, -5.5, 220.5, None, -3.0, 215.0],
    "final_probability": [0.65, 0.58, 0.54, 0.70, 0.62, 0.52],

    # Kalshi data - Game 1 has Kalshi markets, Game 2 does not
    "kalshi_matched": [True, True, True, True, False, False],
    "kalshi_prob_spread": [None, 0.55, None, None, None, None],
    "kalshi_prob_total": [None, None, 0.52, None, None, None],
    "spread_prob_pick_kalshi": [None, 0.55, None, None, None, None],
    "total_prob_pick_kalshi": [None, None, 0.52, None, None, None],
}

df = pd.DataFrame(test_data)

print("=" * 80)
print("TEST: HasKalshiMarket Flag Computation")
print("=" * 80)

print("\nInput Data:")
print(df[["Home", "Away", "Market", "kalshi_matched", "kalshi_prob_spread", "kalshi_prob_total"]].to_string())

print("\n" + "=" * 80)
print("Running build_game_summary_v2...")
print("=" * 80)

result = build_game_summary_v2(df)

print("\nOutput Summary:")
print(result[["Home", "Away", "HasKalshiMarket", "Spread Pick", "Kalshi Spread Prob", "Total Pick", "Kalshi Total Prob"]].to_string())

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Verify Game 1 (Lakers vs Celtics) should have HasKalshiMarket = True
game1 = result[(result["Home"] == "Lakers") & (result["Away"] == "Celtics")]
if not game1.empty:
    has_kalshi = game1.iloc[0]["HasKalshiMarket"]
    print(f"✓ Game 1 (Lakers vs Celtics): HasKalshiMarket = {has_kalshi}")
    if has_kalshi == True:
        print("  ✅ PASS: Game 1 has Kalshi markets (kalshi_matched=True, has spread/total probs)")
    else:
        print("  ❌ FAIL: Game 1 should have HasKalshiMarket=True")
else:
    print("  ❌ FAIL: Game 1 not found in summary")

# Verify Game 2 (Warriors vs Nets) should have HasKalshiMarket = False
game2 = result[(result["Home"] == "Warriors") & (result["Away"] == "Nets")]
if not game2.empty:
    has_kalshi = game2.iloc[0]["HasKalshiMarket"]
    print(f"✓ Game 2 (Warriors vs Nets): HasKalshiMarket = {has_kalshi}")
    if has_kalshi == False:
        print("  ✅ PASS: Game 2 does not have Kalshi markets (kalshi_matched=False)")
    else:
        print("  ❌ FAIL: Game 2 should have HasKalshiMarket=False")
else:
    print("  ❌ FAIL: Game 2 not found in summary")

# Count games with Kalshi markets
total_games = len(result)
games_with_kalshi = int(result["HasKalshiMarket"].sum())

print(f"\n📊 Summary: {total_games} total games, {games_with_kalshi} with Kalshi markets")

if games_with_kalshi == 1:
    print("✅ PASS: Correct count of games with Kalshi markets")
else:
    print(f"❌ FAIL: Expected 1 game with Kalshi markets, got {games_with_kalshi}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
