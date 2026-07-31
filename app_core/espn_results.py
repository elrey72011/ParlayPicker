import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ESPN's free public scoreboard endpoints
ESPN_ENDPOINTS = {
    'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
    'WNBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard',
    'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
    'MLB': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard',
    'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'NCAAB': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
}


def _requested_leagues(leagues: list) -> list[str]:
    """Return normalized, de-duplicated league keys in request order."""

    requested: list[str] = []
    for league in leagues or []:
        normalized = str(league or "").upper().strip()
        if normalized and normalized not in requested:
            requested.append(normalized)
    return requested


def fetch_espn_results(
    leagues: list,
    target_date: datetime.date = None,
    attempts: int = 2,
) -> pd.DataFrame:
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).date()

    date_str = target_date.strftime("%Y%m%d")
    results = []
    requested_leagues = _requested_leagues(leagues)
    unsupported_leagues = [
        league for league in requested_leagues if league not in ESPN_ENDPOINTS
    ]
    if unsupported_leagues:
        logger.warning(
            "No ESPN scoreboard provider configured for leagues: %s",
            unsupported_leagues,
        )

    for league_upper in requested_leagues:
        if league_upper not in ESPN_ENDPOINTS:
            continue

        url = f"{ESPN_ENDPOINTS[league_upper]}?dates={date_str}"
        league_results = []
        max_attempts = max(1, int(attempts or 1))

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch %s from ESPN (attempt %s/%s). Status: %s",
                        league_upper,
                        attempt,
                        max_attempts,
                        response.status_code,
                    )
                    continue

                data = response.json()
                events = data.get('events', [])

                for event in events:
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue

                    match = competitions[0]
                    competitors = match.get('competitors', [])

                    if len(competitors) != 2:
                        continue

                    home_team_data = next((team for team in competitors if team.get('homeAway') == 'home'), None)
                    away_team_data = next((team for team in competitors if team.get('homeAway') == 'away'), None)

                    if not home_team_data or not away_team_data:
                        continue

                    status = match.get('status', {}).get('type', {}).get('state')
                    if status != 'post':
                        continue

                    home_name = home_team_data.get('team', {}).get('displayName')
                    away_name = away_team_data.get('team', {}).get('displayName')
                    home_score = home_team_data.get('score')
                    away_score = away_team_data.get('score')

                    # A zero is a valid final score (notably MLB shutouts). Only
                    # absent scores should keep a final from entering grading.
                    if home_score is not None and away_score is not None:
                        league_results.append({
                            'league': league_upper,
                            'home_team': home_name,
                            'away_team': away_name,
                            'home_score': int(home_score),
                            'away_score': int(away_score),
                            'date': target_date.strftime("%Y-%m-%d")
                        })

                # A successful response should not be retried just because every
                # event is still in progress. The dashboard's explicit backfill
                # control handles games that become final later.
                break

            except Exception as e:
                logger.warning(
                    "Error fetching/parsing ESPN data for %s (attempt %s/%s): %s",
                    league_upper,
                    attempt,
                    max_attempts,
                    e,
                )

        results.extend(league_results)

    df = pd.DataFrame(results)

    if not df.empty:
        from app_core.result_team_names import normalize_result_team

        df['home_team'] = df['home_team'].apply(lambda x: normalize_result_team(x) if pd.notnull(x) else x)
        df['away_team'] = df['away_team'].apply(lambda x: normalize_result_team(x) if pd.notnull(x) else x)

    df.attrs["requested_leagues"] = requested_leagues
    df.attrs["unsupported_leagues"] = unsupported_leagues
    df.attrs["supported_leagues"] = [
        league for league in requested_leagues if league in ESPN_ENDPOINTS
    ]
    return df
