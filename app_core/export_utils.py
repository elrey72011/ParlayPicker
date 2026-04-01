import os
import glob
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def find_yesterdays_export() -> str | None:
    """
    Finds the most recently created 'best_picks_export' file from the previous day.
    Specifically targets pattern 'best_picks_export - YYYY-MM-DD*.csv' in app_exports/
    or 'best_picks_export*.csv' fallback, parsing dates from filenames or checking modification time.
    """
    export_dir = "app_exports"
    if not os.path.exists(export_dir):
        return None

    # Target date is yesterday (relative to system current time)
    yesterday = (datetime.now() - timedelta(days=1)).date()
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # 1. Look for explicit date string in filename
    # e.g., best_picks_export - 2026-03-07.csv
    # e.g., best_picks_export_HEADLESS_2026-03-07T...
    # e.g., 2026-03-07T21-56_export.csv

    # We will gather all csvs
    all_csvs = glob.glob(os.path.join(export_dir, "*.csv"))

    candidate_files = []
    for filepath in all_csvs:
        filename = os.path.basename(filepath)

        # Check if yesterday's date string is in the filename
        if yesterday_str in filename:
            candidate_files.append((filepath, os.path.getmtime(filepath)))
            continue

        # Or parse standard 'best_picks_export' files without explicit dates, using mtime
        if "best_picks_export" in filename.lower():
            mtime = os.path.getmtime(filepath)
            file_date = datetime.fromtimestamp(mtime).date()
            if file_date == yesterday:
                 candidate_files.append((filepath, mtime))

    if not candidate_files:
         return None

    # Sort by mtime descending (newest first)
    candidate_files.sort(key=lambda x: x[1], reverse=True)
    return candidate_files[0][0]
