"""
Test script to verify team name normalization fixes.
"""
from core.team_mapper import normalize_team_name

test_cases = [
    ("L.A. Lakers", "Los Angeles Lakers"),
    ("LA Lakers", "Los Angeles Lakers"),
    ("Los Angeles Lakers", "Los Angeles Lakers"),
    ("Northwestern St.", "Northwestern State"),
    ("Northwestern State", "Northwestern State"),
    ("Michigan St.", "Michigan State"),
    ("Michigan State", "Michigan State"),
    ("St. Bonaventure", "Saint Bonaventure"),
    ("Penn St.", "Penn State"),
    ("Penn State", "Penn State"),
    ("North Carolina", "North Carolina"),
    ("UNC", "North Carolina"),
    ("Nicholls State", "Nicholls State"),
    ("Nicholls St.", "Nicholls State"),
    ("USC", "Usc"), # Should remain Usc due to removal of USC mapping and title casing
]

print("Testing normalize_team_name():\n")
all_passed = True

for input_name, expected in test_cases:
    result = normalize_team_name(input_name)
    passed = result == expected
    all_passed = all_passed and passed

    status = "✅" if passed else "❌"
    print(f"{status} {input_name:30s} -> {result:30s} (expected: {expected})")

print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")