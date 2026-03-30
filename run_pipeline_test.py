import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from core.streamlit_pipeline import run_analysis_pipeline

print("Running pipeline analysis with ML enabled...")
# Using a small subset of the master slate logic
# to trigger the real ML processing logic on a live sample.
df, best_picks, diagnostics = run_analysis_pipeline(sports=["NBA", "NHL"], max_rows=100, use_ml=True)

print("Finished analysis pipeline.")
