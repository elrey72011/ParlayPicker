import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Configure robust standard logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from core.streamlit_pipeline import run_analysis_pipeline, build_best_picks_df

# Read the uploaded data simulating what the streamlit app uses
try:
    spreads_df = pd.read_csv("data/theover_spreads.csv")
    totals_df = pd.read_csv("data/theover_totals.csv")
except Exception as e:
    logging.error(f"Could not load data: {e}")
    sys.exit(1)

logging.info("Starting Headless Pipeline Run...")

# This matches the call made in streamlit_app.py
analysis_df, _, diagnostics = run_analysis_pipeline(
    sports=["NBA", "NHL", "NCAAB", "MLB", "NFL"],
    max_rows=1000,
    use_ml=True,
    spreads_df=spreads_df,
    totals_df=totals_df
)

# Now we need to pass this enriched analysis_df to Kalshi
try:
    from app_core.kalshi_integrator import enrich_with_kalshi_markets
    analysis_df = enrich_with_kalshi_markets(analysis_df)
except Exception as e:
    logging.error(f"Failed Kalshi enrichment: {e}")

# Build best picks explicitly as done in streamlit_app.py
best_picks_df = build_best_picks_df(analysis_df)

logging.info("Headless Pipeline Run Complete.")
logging.info(f"Generated {len(best_picks_df)} final picks out of {len(analysis_df)} analysis rows.")

# Create an export timestamp matching user logs
export_ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
export_path = f"best_picks_export_HEADLESS_{export_ts}.csv"
best_picks_df.to_csv(export_path, index=False)
logging.info(f"Exported to {export_path}")
