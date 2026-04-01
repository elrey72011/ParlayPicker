import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict

from app_core.apisports import APISportsBasketballClient, APISportsHockeyClient, APISportsFootballClient, get_key
from app_core.apisports_baseball import APISportsBaseballClient

logger = logging.getLogger(__name__)

def fetch_yesterdays_results(leagues: List[str]) -> pd.DataFrame:
    """
    Fetches the final scores for yesterday's games across specified leagues
    using API-Sports clients.

    Returns a unified DataFrame suitable for attach_results:
    ['league', 'home_team', 'away_team', 'home_score', 'away_score', 'date']
    """
    target_date = (datetime.now() - timedelta(days=1)).date()
    logger.info(f"Fetching results for {target_date} for leagues: {leagues}")

    results = []

    # Basketball Client covers NBA and NCAAB
    bball_client = None
    if 'NBA' in leagues or 'NCAAB' in leagues:
        bball_key = get_key('NBA') or get_key()
        bball_client = APISportsBasketballClient(api_key=bball_key)

    if 'NBA' in leagues and bball_client:
        games = bball_client.get_games_by_date(target_date, league_id=12) # 12 = NBA
        for g in games:
            scores = g.get('scores', {})
            home_score = (scores.get('home') or {}).get('total')
            away_score = (scores.get('away') or {}).get('total')
            if home_score is not None and away_score is not None:
                results.append({
                    'league': 'NBA',
                    'home_team': (g.get('teams', {}).get('home') or {}).get('name'),
                    'away_team': (g.get('teams', {}).get('away') or {}).get('name'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': target_date.strftime("%Y-%m-%d")
                })

    if 'NCAAB' in leagues and bball_client:
        games = bball_client.get_games_by_date(target_date, league_id=116) # 116 = NCAAB
        for g in games:
            scores = g.get('scores', {})
            home_score = (scores.get('home') or {}).get('total')
            away_score = (scores.get('away') or {}).get('total')
            if home_score is not None and away_score is not None:
                results.append({
                    'league': 'NCAAB',
                    'home_team': (g.get('teams', {}).get('home') or {}).get('name'),
                    'away_team': (g.get('teams', {}).get('away') or {}).get('name'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': target_date.strftime("%Y-%m-%d")
                })

    if 'NHL' in leagues:
        nhl_key = get_key('NHL') or get_key()
        nhl_client = APISportsHockeyClient(api_key=nhl_key)
        games = nhl_client.get_games_by_date(target_date, league_id=57) # 57 = NHL
        for g in games:
            scores = g.get('scores', {})
            home_score = scores.get('home')
            away_score = scores.get('away')
            if home_score is not None and away_score is not None:
                results.append({
                    'league': 'NHL',
                    'home_team': (g.get('teams', {}).get('home') or {}).get('name'),
                    'away_team': (g.get('teams', {}).get('away') or {}).get('name'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': target_date.strftime("%Y-%m-%d")
                })

    if 'MLB' in leagues:
        mlb_key = get_key('MLB') or get_key()
        mlb_client = APISportsBaseballClient(api_key=mlb_key)
        games = mlb_client.get_games_by_date(target_date, league_id=1) # 1 = MLB
        for g in games:
            scores = g.get('scores', {})
            home_score = scores.get('home', {}).get('total')
            away_score = scores.get('away', {}).get('total')
            if home_score is not None and away_score is not None:
                results.append({
                    'league': 'MLB',
                    'home_team': (g.get('teams', {}).get('home') or {}).get('name'),
                    'away_team': (g.get('teams', {}).get('away') or {}).get('name'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': target_date.strftime("%Y-%m-%d")
                })

    if 'NFL' in leagues:
        nfl_key = get_key('NFL') or get_key()
        nfl_client = APISportsFootballClient(api_key=nfl_key)
        games = nfl_client.get_games_by_date(target_date, league_id=1) # 1 = NFL
        for g in games:
            scores = g.get('scores', {})
            home_score = scores.get('home', {}).get('total')
            away_score = scores.get('away', {}).get('total')
            if home_score is not None and away_score is not None:
                results.append({
                    'league': 'NFL',
                    'home_team': (g.get('teams', {}).get('home') or {}).get('name'),
                    'away_team': (g.get('teams', {}).get('away') or {}).get('name'),
                    'home_score': home_score,
                    'away_score': away_score,
                    'date': target_date.strftime("%Y-%m-%d")
                })

    df = pd.DataFrame(results)

    if not df.empty:
        # Pre-clean the names so attach_results joins easily
        from app_core.feature_processing import robust_normalize_team
        from app_core.prediction_engine import clean_team_name

        df['home_team'] = df['home_team'].apply(lambda x: robust_normalize_team(str(x)) if pd.notnull(x) else x)
        df['away_team'] = df['away_team'].apply(lambda x: robust_normalize_team(str(x)) if pd.notnull(x) else x)

    return df
