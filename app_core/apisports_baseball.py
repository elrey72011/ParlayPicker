import logging
from app_core.apisports import _APISportsBaseClient

logger = logging.getLogger(__name__)

class APISportsBaseballClient(_APISportsBaseClient):
    """Wrapper around the API-Sports baseball endpoints (MLB focus)."""

    BASE_URL = "https://v1.baseball.api-sports.io"
    DEFAULT_LEAGUE_ID = 1  # MLB
    SECRET_ENV_PRIORITY = ("MLB_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY")
    SPORT_KEY = "baseball_mlb"
    SPORT_NAME = "MLB"
    STAT_CATEGORY_OFFENSE = "runs"
    STAT_CATEGORY_DEFENSE = "runs"
    SCORING_METRIC_LABEL = "runs"
    SEASON_CUTOFF_MONTH = 2
    SEASON_FORMAT = "single"
