import pandas as pd
import logging
import os
from unittest.mock import patch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

def main():
    from core.streamlit_pipeline import run_analysis_pipeline, build_best_picks_df
    logger.info("Starting Headless Production Pipeline Run with mock dataframe...")

    # We will upload a mock "TheOver" export to give it a slate since odds are blocked/faked
    mock_theover = pd.DataFrame([
        {
            "league": "NBA",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "game_date": "2026-03-31T00:00:00Z",
            "market_type": "spread_home",
            "spread_line": -5.5,
            "odds_american": -110,
            "ml_probability": 0.58,
        },
        {
            "league": "NBA",
            "home_team": "Los Angeles Lakers",
            "away_team": "Sacramento Kings",
            "game_date": "2026-03-31T00:00:00Z",
            "market_type": "spread_away",
            "spread_line": 2.5,
            "odds_american": -110,
            "ml_probability": 0.53,
        }
    ])

    with patch('app_core.odds_api.TheOddsAPIClient.get_odds') as mock_odds:
        mock_odds.return_value = [] # Test empty expansion safely or we can mock data

        # Run the exact same production path
        analysis_df, _, diagnostics = run_analysis_pipeline(
            sports=['NCAAB', 'NBA', 'NHL'],
            max_rows=100,
            spreads_df=mock_theover
        )

        if not analysis_df.empty:
            from app_core.kalshi_integrator import KalshiIntegrator
            k_integrator = KalshiIntegrator()
            enriched_df = k_integrator.enrich(analysis_df)
            best_picks_df = build_best_picks_df(enriched_df)
            logger.info(f"Generated {len(best_picks_df)} best picks out of {len(analysis_df)} raw predictions.")
        else:
            logger.info("analysis_df was empty, check logic.")

    logger.info("Pipeline Complete.")

if __name__ == "__main__":
    main()
