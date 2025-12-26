import sys
import os
sys.path.append(os.getcwd())
from app_core.apisports import APISportsFootballClient

def check_method():
    if hasattr(APISportsFootballClient, 'get_team_stats'):
        print("APISportsFootballClient has get_team_stats")
    else:
        print("APISportsFootballClient MISSING get_team_stats")

if __name__ == "__main__":
    check_method()
