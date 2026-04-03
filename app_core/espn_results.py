import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ESPN's free public scoreboard endpoints
ESPN_ENDPOINTS = {
    'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
    'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
    'MLB': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard',
    'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'NCAAB': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
}

def fetch_espn_results(leagues: list, target_date: datetime.date = None) -> pd.DataFrame:
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).date()

    date_str = target_date.strftime("%Y%m%d")
    results = []

    for league in leagues:
        league_upper = league.upper()
        if league_upper not in ESPN_ENDPOINTS:
            continue

        url = f"{ESPN_ENDPOINTS[league_upper]}?dates={date_str}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {league_upper} from ESPN. Status: {response.status_code}")
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

                if home_score and away_score:
                    results.append({
                        'league': league_upper,
                        'home_team': home_name,
                        'away_team': away_name,
                        'home_score': int(home_score),
                        'away_score': int(away_score),
                        'date': target_date.strftime("%Y-%m-%d")
                    })

        except Exception as e:
            logger.error(f"Error parsing ESPN data for {league_upper}: {e}")

    df = pd.DataFrame(results)

    if not df.empty:
        from app_core.feature_processing import robust_normalize_team
        df['home_team'] = df['home_team'].apply(lambda x: robust_normalize_team(str(x)) if pd.notnull(x) else x)
        df['away_team'] = df['away_team'].apply(lambda x: robust_normalize_team(str(x)) if pd.notnull(x) else x)

    return df
