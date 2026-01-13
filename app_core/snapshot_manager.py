import os
import logging
import pandas as pd
from datetime import datetime, time
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger("snapshot_manager")

# Persistent Directory
# Check if /mount/src/parlaypicker exists (Cloud Run), otherwise use local data/
PERSISTENT_MOUNT_ROOT = "/mount/src/parlaypicker"
if os.path.exists(PERSISTENT_MOUNT_ROOT):
    SNAPSHOT_DIR = os.path.join(PERSISTENT_MOUNT_ROOT, "data", "snapshots")
else:
    SNAPSHOT_DIR = os.path.join("data", "snapshots")

# Time Windows (US/Eastern)
NOON_BASELINE_START = time(9, 0)
NOON_BASELINE_END = time(13, 0)

EVENING_UPDATE_START = time(17, 30)
EVENING_UPDATE_END = time(19, 30)

LATE_UPDATE_START = time(21, 0)
LATE_UPDATE_END = time(23, 0)

def get_et_now():
    """Returns current time in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York"))

def get_snapshot_filename(date_str, suffix):
    """Returns the expected filename for a given date and suffix."""
    return os.path.join(SNAPSHOT_DIR, f"{date_str}_{suffix}.csv")

def get_active_snapshot_status():
    """
    Returns a string describing the current active snapshot window/status.
    Used for Sidebar display.
    """
    now_et = get_et_now()
    current_time = now_et.time()
    date_str = now_et.date().isoformat()

    status = "Standard Mode"

    # Check what exists
    noon_exists = os.path.exists(get_snapshot_filename(date_str, "noon_baseline"))
    eve_exists = os.path.exists(get_snapshot_filename(date_str, "evening_update"))
    late_exists = os.path.exists(get_snapshot_filename(date_str, "late_update"))

    if NOON_BASELINE_START <= current_time <= NOON_BASELINE_END:
        status = "Noon Baseline Window (Recording)" if not noon_exists else "Noon Baseline Loaded"
    elif EVENING_UPDATE_START <= current_time <= EVENING_UPDATE_END:
        status = "Evening Update Window"
    elif LATE_UPDATE_START <= current_time <= LATE_UPDATE_END:
        status = "Late/Closing Window"

    loaded_info = []
    if noon_exists: loaded_info.append("Noon")
    if eve_exists: loaded_info.append("Eve")
    if late_exists: loaded_info.append("Late")

    if loaded_info:
        return f"{status} | Saved: {', '.join(loaded_info)}"
    return status

def save_snapshot(df: pd.DataFrame) -> str:
    """
    Saves the dataframe snapshot based on current time window.
    Returns a status string indicating what happened.
    """
    if df is None or df.empty:
        return "No Data to Save"

    now_et = get_et_now()
    current_time = now_et.time()
    date_str = now_et.date().isoformat()

    suffix = None
    check_exists = False # Only true for Noon Baseline to prevent overwrite

    if NOON_BASELINE_START <= current_time <= NOON_BASELINE_END:
        suffix = "noon_baseline"
        check_exists = True
    elif EVENING_UPDATE_START <= current_time <= EVENING_UPDATE_END:
        suffix = "evening_update"
    elif LATE_UPDATE_START <= current_time <= LATE_UPDATE_END:
        suffix = "late_update"

    if not suffix:
        return "Outside Snapshot Window"

    filename = get_snapshot_filename(date_str, suffix)

    # Task 2: "Check for this on startup; do not re-scrape if present" (Noon Baseline)
    # Here we implement "do not overwrite if present" for noon baseline logic
    if check_exists and os.path.exists(filename):
        logger.info(f"Snapshot {filename} already exists. Skipping save.")
        return f"Skipped (Exists): {suffix}"

    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # Columns to persist for CLV and movement tracking
        # Ensure we capture enough to calculate deltas later
        cols_to_save = [
            'league', 'Home', 'Away', 'Commence (UTC)',
            'Implied_Prob', 'Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff',
            'Spread & Pick', 'Total & Pick', 'final_probability', 'Pick',
            'spread_pick_line', 'total_pick_line', 'spread_edge', 'total_edge',
            'ticket_pct', 'money_pct', 'sharpness'
        ]

        # Save intersection of available columns
        cols = [c for c in cols_to_save if c in df.columns]

        df[cols].to_csv(filename, index=False)
        logger.info(f"Saved snapshot to {filename}")
        return f"Saved: {suffix}"

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return f"Error: {e}"

def load_and_compare(current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads baseline (noon) snapshot and computes deltas if we are in Evening/Late windows.
    Also handles CLV (Closing Line Value) logic for Late Update.
    Returns enriched dataframe with 'delta_...' columns.
    """
    if current_df is None or current_df.empty:
        return current_df

    now_et = get_et_now()
    current_time = now_et.time()
    date_str = now_et.date().isoformat()

    # Only compare if we are PAST the noon window (i.e., Evening or Late)
    if current_time < EVENING_UPDATE_START:
        return current_df

    baseline_file = get_snapshot_filename(date_str, "noon_baseline")

    if not os.path.exists(baseline_file):
        return current_df

    try:
        baseline_df = pd.read_csv(baseline_file)

        # Rename baseline columns
        suffix_map = {
            'Implied_Prob': 'Implied_Prob_noon',
            'Home_Sentiment': 'Home_Sentiment_noon',
            'Away_Sentiment': 'Away_Sentiment_noon',
            'Sentiment_Diff': 'Sentiment_Diff_noon',
            'spread_pick_line': 'spread_pick_line_noon',
            'total_pick_line': 'total_pick_line_noon',
            'final_probability': 'final_probability_noon',
            'Pick': 'Pick_noon',
            'spread_edge': 'spread_edge_noon',
            'total_edge': 'total_edge_noon'
        }

        baseline_subset = baseline_df[['league', 'Home', 'Away'] + [c for c in suffix_map.keys() if c in baseline_df.columns]].copy()
        baseline_subset.rename(columns=suffix_map, inplace=True)

        # Merge
        merged = pd.merge(current_df, baseline_subset, on=['league', 'Home', 'Away'], how='left')

        # Delta Calculations
        def calc_delta(curr_col, noon_col):
            if curr_col in merged.columns and noon_col in merged.columns:
                return pd.to_numeric(merged[curr_col], errors='coerce') - pd.to_numeric(merged[noon_col], errors='coerce')
            return 0.0

        merged['delta_implied_prob'] = calc_delta('Implied_Prob', 'Implied_Prob_noon')
        merged['delta_sentiment'] = calc_delta('Sentiment_Diff', 'Sentiment_Diff_noon')
        merged['line_move_spread'] = calc_delta('spread_pick_line', 'spread_pick_line_noon')
        merged['line_move_total'] = calc_delta('total_pick_line', 'total_pick_line_noon')

        # CLV Deltas
        merged['clv_spread_edge_diff'] = calc_delta('spread_edge', 'spread_edge_noon')
        merged['clv_total_edge_diff'] = calc_delta('total_edge', 'total_edge_noon')

        # Line Movement Flag (>= 0.5)
        def _get_movement_alert(row):
            alerts = []

            # Market Steam (> 3%)
            delta_prob = row.get('delta_implied_prob', 0)
            if pd.notnull(delta_prob) and abs(delta_prob) > 0.03:
                direction = "Steam" if delta_prob > 0 else "Drift"
                alerts.append(f"Market {direction} ({delta_prob:+.1%})")

            # Spread Move
            spread_move = row.get('line_move_spread', 0)
            if pd.notnull(spread_move) and abs(spread_move) >= 0.5:
                alerts.append(f"Spread Move: {spread_move:+.1f}")

            # Total Move
            total_move = row.get('line_move_total', 0)
            if pd.notnull(total_move) and abs(total_move) >= 0.5:
                alerts.append(f"Total Move: {total_move:+.1f}")

            return "; ".join(alerts)

        merged['movement_alerts'] = merged.apply(_get_movement_alert, axis=1)

        return merged

    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return current_df
