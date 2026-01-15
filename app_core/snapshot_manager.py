import os
import logging
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo
# Avoid top-level import of market_tracker to prevent circular dependency
# import app_core.market_tracker as market_tracker

logger = logging.getLogger("snapshot_manager")

# Snapshot Directory - Explicitly enforced as per task instructions
# Hardcoded loading path to resolve "No noon baseline found" error
SNAPSHOT_DIR = "data/snapshots"

# Ensure directory exists (if we have permissions, otherwise this might fail but path is forced)
try:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
except Exception:
    pass # Ignore permission errors if on read-only system, path is what matters for loading

# Time Windows (ET) - Duplicated from market_tracker to avoid circular import
NOON_BASELINE_START = time(9, 0)
NOON_BASELINE_END = time(13, 0)

def get_et_now():
    """Returns current time in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York"))

def get_snapshot_filename(date_str, suffix):
    """Returns the expected filename for a given date and suffix."""
    return os.path.join(SNAPSHOT_DIR, f"{date_str}_{suffix}.csv")

def save_noon_baseline(df: pd.DataFrame, force: bool = False) -> bool:
    """
    Saves the dataframe to /snapshots/[YYYY-MM-DD]_noon_baseline.csv.
    Checks time window (9 AM - 1 PM ET) unless force=True.
    Returns True if saved or already exists, False if outside window (and not forced) or error.
    """
    if df is None or df.empty:
        return False

    now_et = get_et_now()
    current_time = now_et.time()
    date_str = now_et.date().isoformat()

    # Logic: If forced OR (current_time is between 9 AM and 1 PM)
    if force or (NOON_BASELINE_START <= current_time <= NOON_BASELINE_END):
        # Use _noon_baseline.csv as requested for consistency
        filename = get_snapshot_filename(date_str, "noon_baseline")

        # Check for existence to avoid overwriting baseline unless forced
        if os.path.exists(filename) and not force:
            logger.info(f"Noon baseline {filename} already exists. Skipping save.")
            return True

        try:
            # Ensure dir exists before saving
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            # Save Full DataFrame as baseline
            df.to_csv(filename, index=False)
            logger.info(f"Saved Noon Baseline to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Noon Baseline: {e}")
            return False

    return False

def check_noon_baseline_status() -> str:
    """
    Checks if the Noon Baseline exists for today.
    Returns status string for UI.
    """
    now_et = get_et_now()
    date_str = now_et.date().isoformat()
    filename = get_snapshot_filename(date_str, "noon_baseline")

    if os.path.exists(filename):
        return "✅ Noon Baseline Cached"
    else:
        return "⚠️ Noon Baseline Missing"
