import os
import sys
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, str(Path(os.getcwd())))

os.environ["ODDS_API_KEY"] = "DUMMY_KEY_TO_TRIGGER_FALLBACK"

from core.streamlit_pipeline import run_analysis_pipeline

def main():
    print("======================================")
    print("RUNNING PIPELINE DIAGNOSTICS HARNESS (LOCAL)")
    print("======================================")

    # Use real test upload data
    csv_file = "theover_log_dump.csv"
    print(f"Loading {csv_file} as fallback")
    df = pd.read_csv(csv_file)

    # Theover raw CSV usually has different headers. Try to map or rename
    # but the pipeline handles most standard aliases.

    analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
        sports=["NBA", "NHL", "NCAAB"],
        use_ml=True,
        spreads_df=df,
        totals_df=None
    )

    print("\n--- RESULTS ---")
    print(f"Analysis DF rows: {len(analysis_df)}")
    if not analysis_df.empty:
         print(f"Unique ML Probs generated: {analysis_df['ml_probability'].nunique()}")
         print(f"Unique Calibrated Probs: {analysis_df['calibrated_probability'].nunique()}")

    print(f"Best Picks DF rows: {len(best_picks_df)}")
    if not best_picks_df.empty:
         print("\nPick Status Breakdown:")
         print(best_picks_df['Pick_Status'].value_counts())
         print("\nOdds Source Breakdown:")
         print(best_picks_df['odds_source'].value_counts())
         print("\nModel Fallback Statuses:")
         if "model_status" in analysis_df.columns:
             print(analysis_df["model_status"].value_counts())

if __name__ == "__main__":
    main()
