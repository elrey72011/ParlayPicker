import logging
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Mock imports for app_core.feature_processing
sys.modules['nba_api.stats.endpoints'] = MagicMock()
sys.modules['nfl_data_py'] = MagicMock()
sys.modules['cfbd'] = MagicMock()
sys.modules['nhlpy'] = MagicMock()
sys.modules['cbbpy.mens_scraper'] = MagicMock()
sys.modules['streamlit'] = MagicMock()

# Now import the module to test
from app_core import feature_processing
# Also mock TeamNameMatcher since it might not have data in sandbox
feature_processing.TeamNameMatcher.normalize = lambda x: x

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

class TestStatsFetching(unittest.TestCase):

    @patch('app_core.feature_processing.leaguedashteamstats.LeagueDashTeamStats')
    def test_fetch_nba_stats(self, mock_dash):
        # Mock NBA response
        mock_df = pd.DataFrame({
            'TEAM_NAME': ['Celtics', 'Lakers'],
            'GP': [10, 10],
            'PTS': [120.0, 110.0], # Per game
            'PLUS_MINUS': [10.0, -5.0],
            'W_PCT': [0.8, 0.4],
            'TOV': [12.0, 14.0],
            'AST': [25.0, 22.0],
            'REB': [45.0, 40.0]
        })
        mock_dash.return_value.get_data_frames.return_value = [mock_df]

        stats = feature_processing.fetch_nba_stats(2024)

        self.assertEqual(len(stats), 2)
        celtics = stats[0]
        self.assertEqual(celtics['team_norm'], 'Celtics') # Using lambda identity mock
        self.assertEqual(celtics['points_per_game'], 120.0) # Direct map
        self.assertEqual(celtics['points_allowed_per_game'], 110.0) # 120 - 10

        # Verify call arguments
        mock_dash.assert_called_with(
            season='2024-25',
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame'
        )

    @patch('app_core.feature_processing.cfbd')
    @patch('app_core.feature_processing._get_secret')
    def test_fetch_ncaaf_stats(self, mock_secret, mock_cfbd):
        mock_secret.return_value = 'fake_key'

        # Mock StatsApi
        mock_stats_api = MagicMock()
        mock_cfbd.StatsApi.return_value = mock_stats_api

        # Mock Season Stats Response
        class MockStat:
            def __init__(self, team, games, points, opp_points):
                self.team = team
                self.games = games
                self.offense = MagicMock()
                self.offense.points = points
                self.offense.games = games
                self.offense.turnovers = 10
                self.defense = MagicMock()
                self.defense.points = opp_points

        # Make sure this returns an iterable
        mock_stats_api.get_team_season_stats.return_value = [
            MockStat('Army', 10, 300, 100), # 30 ppg, 10 oppg
            MockStat('Navy', 10, 200, 200)
        ]

        # Mock GamesApi for Win Pct
        mock_games_api = MagicMock()
        mock_cfbd.GamesApi.return_value = mock_games_api

        # Mock Games for win calc
        class MockGame:
            def __init__(self, h, a, hp, ap):
                self.home_team = h
                self.away_team = a
                self.home_points = hp
                self.away_points = ap

        mock_games_api.get_games.return_value = [
            MockGame('Army', 'Navy', 30, 20), # Army win
            MockGame('Army', 'Air Force', 40, 10) # Army win
        ]

        stats = feature_processing.fetch_ncaaf_stats(2024)

        army = next(s for s in stats if s['team_norm'] == 'Army')
        self.assertEqual(army['points_per_game'], 30.0) # 300 / 10
        self.assertEqual(army['points_allowed_per_game'], 10.0) # 100 / 10
        self.assertEqual(army['win_pct'], 1.0) # 2 wins / 2 games in mock schedule

        # Verify calls
        mock_stats_api.get_team_season_stats.assert_called_with(year=2024)

    @patch('app_core.feature_processing.NHLClient')
    def test_fetch_nhl_stats(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock Standings
        mock_client.standings.league_standings.return_value = {
            'standings': [
                {
                    'teamName': {'default': 'Bruins'},
                    'gamesPlayed': 10,
                    'wins': 6,
                    'goalsPerGame': 3.5, # New field
                    'goalFor': 35,
                    'goalAgainst': 25
                }
            ]
        }

        stats = feature_processing.fetch_nhl_stats(2024)
        bruins = stats[0]

        self.assertEqual(bruins['points_per_game'], 3.5) # Used goalsPerGame
        self.assertEqual(bruins['points_allowed_per_game'], 2.5) # Calculated 25/10
        self.assertEqual(bruins['win_pct'], 0.6)

if __name__ == '__main__':
    unittest.main()
