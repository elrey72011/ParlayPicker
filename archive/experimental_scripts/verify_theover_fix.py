#!/usr/bin/env python3
"""
Verification Script for TheOver Cross-League Contamination Fix
Tests PR #909: Fix cross-league team validation to only match full names
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import just what we need by loading the module directly
import importlib.util
spec = importlib.util.spec_from_file_location("theover_ingest", "app_core/theover_ingest.py")
theover_module = importlib.util.module_from_spec(spec)

# Set up logger first
logger = logging.getLogger("app_core.theover_ingest")
logging.basicConfig(level=logging.WARNING)

# Now load the module
try:
    spec.loader.exec_module(theover_module)
    _validate_team_for_league = theover_module._validate_team_for_league
    _resolve_team_alias = theover_module._resolve_team_alias
except Exception as e:
    print(f"Error loading module: {e}")
    sys.exit(1)

def test_validation_function():
    """Test the _validate_team_for_league function with known cases."""

    print("=" * 80)
    print("VERIFICATION: TheOver Cross-League Contamination Fix (PR #909)")
    print("=" * 80)
    print()

    # Test cases that should PASS (not trigger false positives)
    should_pass = [
        ("Boston Celtics", "NBA"),
        ("Toronto Raptors", "NBA"),
        ("Indiana Pacers", "NBA"),
        ("New Orleans Pelicans", "NBA"),
        ("Indiana Hoosiers", "NCAAB"),
        ("Buffalo Bulls", "NCAAB"),
        ("Charlotte 49ers", "NCAAB"),
        ("Boston Bruins", "NHL"),
        ("Toronto Maple Leafs", "NHL"),
        ("Buffalo Sabres", "NHL"),
        ("New Orleans Saints", "NFL"),
        ("Indiana Colts", "NFL"),
    ]

    # Test cases that should FAIL (actual cross-league contamination)
    should_fail = [
        ("Boston Celtics", "NHL"),  # NBA team in NHL
        ("Boston Bruins", "NBA"),   # NHL team in NBA
        ("Indiana Pacers", "NCAAF"),  # NBA team in NCAAF
        ("Indiana Hoosiers", "NFL"),  # NCAAB team in NFL
    ]

    print("TEST 1: False Positive Prevention (Should All PASS)")
    print("-" * 80)
    passed = 0
    failed = 0

    for team, league in should_pass:
        result = _validate_team_for_league(team, league)
        status = "✅ PASS" if result else "❌ FAIL (False Positive!)"
        print(f"{status}: {team:30s} in {league:10s} → {result}")
        if result:
            passed += 1
        else:
            failed += 1

    print()
    print(f"False Positive Tests: {passed}/{len(should_pass)} passed")
    if failed > 0:
        print(f"⚠️  WARNING: {failed} false positives detected!")
    else:
        print("✅ All false positive tests passed!")

    print()
    print("TEST 2: True Contamination Detection (Should All FAIL)")
    print("-" * 80)

    detected = 0
    missed = 0

    for team, league in should_fail:
        result = _validate_team_for_league(team, league)
        status = "✅ DETECTED" if not result else "❌ MISSED"
        print(f"{status}: {team:30s} in {league:10s} → {result}")
        if not result:
            detected += 1
        else:
            missed += 1

    print()
    print(f"Contamination Detection: {detected}/{len(should_fail)} detected")
    if missed > 0:
        print(f"⚠️  WARNING: {missed} contaminations missed!")
    else:
        print("✅ All contaminations properly detected!")

    print()
    print("=" * 80)
    print("TEST 3: League-Locked Alias Resolution")
    print("=" * 80)
    print()

    # Test league-locked aliasing
    alias_tests = [
        ("Boston", "NBA", "Boston Celtics"),
        ("Boston", "NHL", "Boston Bruins"),
        ("Indiana", "NBA", "Indiana Pacers"),
        ("Indiana", "NCAAF", "Indiana"),  # No alias in map -> returns original
        ("Buffalo", "NHL", "Buffalo Sabres"),
        ("Buffalo", "NCAAB", "Buffalo"),  # No alias in map -> returns original
        ("Toronto", "NBA", "Toronto Raptors"),
        ("Toronto", "NHL", "Toronto Maple Leafs"),
        ("New Orleans", "NBA", "New Orleans Pelicans"),
        ("New Orleans", "NCAAB", "New Orleans Privateers"),
    ]

    alias_passed = 0
    alias_failed = 0

    for alias, league, expected in alias_tests:
        result = _resolve_team_alias(alias, league)
        match = result == expected
        status = "✅ PASS" if match else "❌ FAIL"
        print(f"{status}: '{alias}' in {league:10s} → '{result}' (expected: '{expected}')")
        if match:
            alias_passed += 1
        else:
            alias_failed += 1

    print()
    print(f"Alias Resolution: {alias_passed}/{len(alias_tests)} passed")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tests = len(should_pass) + len(should_fail) + len(alias_tests)
    total_passed = passed + detected + alias_passed
    total_failed = failed + missed + alias_failed

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print()

    if total_failed == 0:
        print("🎉 ALL TESTS PASSED! TheOver fix is working correctly!")
        return 0
    else:
        print(f"⚠️  {total_failed} tests failed. Review the fix.")
        return 1

if __name__ == "__main__":
    exit_code = test_validation_function()
    sys.exit(exit_code)
