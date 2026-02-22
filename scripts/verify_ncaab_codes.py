import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_core.kalshi_integrator import generate_comprehensive_team_variants

# List of teams and their expected Kalshi ticker codes
TEST_CASES = [
    # Regression check for Duke (from failed unit test)
    ("Duke", "DUK"),
    ("North Carolina", "UNC"),

    # User specific requests
    ("Western Michigan", "WMU"),
    ("WESTERN MICHIGAN", "WMU"),
    ("Western Michigan Broncos", "WMU"),

    ("Eastern Michigan", "EMU"),
    ("EASTERN MICHIGAN", "EMU"),
    ("Eastern Michigan Eagles", "EMU"),

    ("Bellarmine", "BELL"),
    ("BELLARMINE", "BELL"),
    ("Bellarmine Knights", "BELL"),

    ("CSU Bakersfield", "CSUB"),
    ("CSU BAKERSFIELD", "CSUB"),
    ("CSU Bakersfield Roadrunners", "CSUB"),

    ("CSU Fullerton", "CSUF"),
    ("CSU FULLERTON", "CSUF"),
    ("CSU Fullerton Titans", "CSUF"),

    ("Portland State", "PSU"),
    ("PORTLAND STATE", "PSU"),
    ("Portland State Vikings", "PSU"),

    # From Log Failures - adding variants
    ("Arkansas-Pine Bluff", "UAPB"),
    ("Arkansas-Pine Bluff Golden Lions", "UAPB"),

    ("Prairie View", "PVAM"),
    ("Prairie View Panthers", "PVAM"),

    ("Eastern Kentucky", "EKU"),
    ("Eastern Kentucky Colonels", "EKU"),

    ("NJIT", "NJIT"),
    ("NJIT Highlanders", "NJIT"),

    ("Vermont", "VT"),
    ("Vermont Catamounts", "VT"),

    ("Grand Canyon", "GCU"),
    ("Grand Canyon Antelopes", "GCU"),

    ("Wyoming", "WYO"),
    ("Wyoming Cowboys", "WYO"),

    ("Auburn", "AUB"),
    ("Auburn Tigers", "AUB"),

    ("Kentucky", "UK"),
    ("Kentucky Wildcats", "UK"),

    ("Long Beach St", "LBSU"),
    ("Long Beach St 49ers", "LBSU"),

    ("CSU Northridge", "CSUN"),
    ("CSU Northridge Matadors", "CSUN"),

    ("Sacramento St", "SAC"),
    ("Sacramento St Hornets", "SAC"),

    ("Idaho", "IDA"),
    ("Idaho Vandals", "IDA"),

    ("Murray State", "MURS"),
    ("Murray State Racers", "MURS"),
]

passed = 0
failed = 0

print(f"Verifying NCAAB Team Code Generation for {len(TEST_CASES)} cases...")
print("-" * 60)

for team_name, expected_code in TEST_CASES:
    variants = generate_comprehensive_team_variants(team_name, league="NCAAB")

    # We want the expected code to be one of the variants
    if expected_code in variants:
        print(f"✅ PASS: '{team_name}' -> contains '{expected_code}'")
        passed += 1
    else:
        print(f"❌ FAIL: '{team_name}' -> Expecting '{expected_code}'")
        print(f"     Generated: {variants}")
        failed += 1

print("-" * 60)
print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
