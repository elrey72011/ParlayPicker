"""Unranked audited games remain visible without inheriting a betting quote."""
import unittest

import pandas as pd

from app_core.game_coverage import publication_games
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, validate_package


RUN = "20260926T170259.058481Z"


class GameCoverageTests(unittest.TestCase):
    def test_unranked_games_reach_all_public_views_without_a_pick(self):
        ranked = pd.DataFrame([{
            "export_run_id": RUN, "matchup_id": "ranked", "league": "MLB",
            "Home": "Home A", "Away": "Away A", "Local Date": "2026-09-26",
            "Commence (Local)": "2026-09-26 7:00 PM ET", "Play_Stake": 0.0,
            "Bettable": False,
        }])
        audit = pd.DataFrame([
            {"export_run_id": RUN, "matchup_id": "ranked", "league": "MLB",
             "home_team": "Home A", "away_team": "Away A",
             "game_time_est": "2026-09-26 7:00 PM ET",
             "quote_chronology_status": "VERIFIED_PREGAME",
             "candidate_context": "CURRENT_PREGAME"},
            {"export_run_id": RUN, "matchup_id": "untimed", "league": "NCAAF",
             "home_team": "Home B", "away_team": "Away B",
             "game_time_est": "2026-09-26 8:00 PM ET",
             "quote_chronology_status": "QUOTE_TIME_MISSING",
             "candidate_context": "HISTORICAL_BACKTEST",
             "best_available_rank": 1, "best_pick": "Home B -3.5",
             "market_type": "spread_home", "odds_american": -110},
            {"export_run_id": RUN, "matchup_id": "started", "league": "NCAAF",
             "home_team": "Home C", "away_team": "Away C",
             "game_time_est": "2026-09-26 12:00 PM ET",
             "quote_chronology_status": "POST_START_QUOTE",
             "candidate_context": "POST_START_DIAGNOSTIC"},
        ])
        games, coverage = publication_games(ranked, audit)
        self.assertEqual(len(games), 3)
        self.assertEqual(set(coverage.matchup_id), {"untimed", "started"})
        frames = [per_game_board(games, audit, family, novig_only=True,
                                 college_fallback=True)
                  for family in ("overall", "sides", "totals")]
        for frame in frames:
            self.assertEqual(len(frame), 3)
            missing = frame[frame.matchup_id.isin(coverage.matchup_id)]
            self.assertTrue(missing.status.eq("PASS").all())
            self.assertTrue(missing.Play_Stake.eq(0).all())
            self.assertTrue(missing.odds.isna().all())
            self.assertTrue(missing.win_probability.isna().all())
            self.assertTrue(missing.selection_label.eq("Unavailable").all())
        package = build_package(*frames)
        validate_package(package)
        self.assertEqual(len(package["games"]["overall"]), 3)

    def test_mixed_analysis_runs_are_rejected(self):
        best = pd.DataFrame([{"export_run_id": RUN, "matchup_id": "ranked"}])
        audit = pd.DataFrame([{
            "export_run_id": "20260926T180000.000000Z", "matchup_id": "other",
            "league": "MLB", "home_team": "Home", "away_team": "Away",
            "game_time_est": "2026-09-26 7:00 PM ET",
        }])
        with self.assertRaisesRegex(ValueError, "different runs"):
            publication_games(best, audit)

    def test_all_unranked_slate_still_has_a_public_board(self):
        audit = pd.DataFrame([{
            "export_run_id": RUN, "matchup_id": "only-game", "league": "NCAAF",
            "home_team": "Home", "away_team": "Away",
            "game_time_est": "2026-09-26 8:00 PM ET",
            "quote_chronology_status": "QUOTE_TIME_MISSING",
            "candidate_context": "HISTORICAL_BACKTEST",
        }])
        games, coverage = publication_games(pd.DataFrame(), audit)
        self.assertEqual(len(coverage), 1)
        frames = [per_game_board(games, audit, family, novig_only=True, college_fallback=True)
                  for family in ("overall", "sides", "totals")]
        package = build_package(*frames)
        validate_package(package)
        self.assertEqual(package["games"]["overall"][0]["status"], "PASS")
        self.assertIsNone(package["games"]["overall"][0]["odds"])


if __name__ == "__main__":
    unittest.main()
