import logging
import sys
from app_core.feature_processing import robust_normalize_team

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

print(robust_normalize_team("Colorado", "NHL"))
print(robust_normalize_team("Florida", "NHL"))
print(robust_normalize_team("Chicago Blackhawks", "NHL"))
