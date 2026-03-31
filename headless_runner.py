import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging to go to console to ensure we capture it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('headless_runner')

# Initialize Streamlit secrets environment early to pass api key checks
os.environ["ODDS_API_KEY"] = "dummy_key_if_needed" # Ensure this won't crash if it tries to fetch live odds, though ideally we use saved ones

def run_headless():
    logger.info("Starting headless runner for pipeline analysis...")

    # Use existing components safely
    from core.streamlit_pipeline import run_analysis_pipeline, load_base_data
    from core.theover_loader import load_theover_csv

    # Check if we have TheOver CSVs locally to test with
    # Look for known test files or use recent ones if available
    data_dir = Path("data")

    spreads_path = data_dir / "theover_spreads.csv"
    totals_path = data_dir / "theover_totals.csv"

    spreads_df = None
    totals_df = None

    class DummyUpload:
        def __init__(self, path):
            self.path = path
        def seek(self, pos):
            pass
        def read(self):
            with open(self.path, 'rb') as f:
                return f.read()

    if spreads_path.exists():
        with open(spreads_path, 'rb') as f:
            spreads_df, _ = load_theover_csv(f)

    if totals_path.exists():
        with open(totals_path, 'rb') as f:
            totals_df, _ = load_theover_csv(f)

    logger.info("Running full pipeline...")

    try:
        # Run with MLB, NBA, NHL, NCAAB to cover the bases, or just let it default to all
        sports = ["NBA", "NHL", "NCAAB", "NFL", "MLB"]
        analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
            sports=sports,
            max_rows=1000,
            use_ml=True,
            spreads_df=spreads_df,
            totals_df=totals_df
        )

        logger.info(f"Pipeline completed successfully. Generated {len(analysis_df)} analysis rows and {len(best_picks_df)} best picks.")

        # Output summary of the diagnostics, specially ones related to flatness
        if diagnostics:
            logger.info("--- Diagnostics Summary ---")
            for k, v in diagnostics.items():
                # Filter to some key ones to avoid blowing up the console
                if k in ["total_games", "bet_rows", "positive_ev_rows", "ml_predictions"]:
                    logger.info(f"  {k}: {v}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_headless()
