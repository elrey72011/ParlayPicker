import pandas as pd
import logging
from app_core.export_utils import find_yesterdays_export
from app_core.results_fetcher import fetch_yesterdays_results
from app_core.results_ingestion import attach_results

logger = logging.getLogger(__name__)

def run_performance_pipeline() -> pd.DataFrame | None:
    """
    Executes the pipeline to fetch yesterday's export, fetch final scores
    for the leagues in that export, and attach the scores to calculate outcomes.
    Returns the enriched DataFrame or None if no export was found.
    """
    export_path = find_yesterdays_export()
    if not export_path:
         logger.warning("No best picks export found for yesterday.")
         return None

    try:
         picks_df = pd.read_csv(export_path)
    except Exception as e:
         logger.error(f"Failed to read {export_path}: {e}")
         return None

    if picks_df.empty:
         return picks_df

    leagues = picks_df['league'].dropna().unique().tolist()

    results_df = fetch_yesterdays_results(leagues)

    if results_df.empty:
         logger.warning(f"No results found for yesterday's games across leagues: {leagues}")

    enriched_picks_df = attach_results(picks_df, results_df)

    return enriched_picks_df
