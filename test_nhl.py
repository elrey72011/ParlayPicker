import logging
from app_core.feature_processing import robust_normalize_team

logging.basicConfig(level=logging.INFO, format="%(message)s")

tests = ["Colorado", "Florida", "NY Rangers", "Carolina", "Tampa Bay", "Vegas", "San Jose", "New Jersey", "Dallas"]
for t in tests:
    print(f"Testing {t}:")
    result = robust_normalize_team(t, league="NHL")
    print(f"Result: {result}\n")
