import os
import logging
import pandas as pd
from datetime import datetime, time, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo # Fallback if needed

# Import snapshot manager to share constants
import app_core.snapshot_manager as snapshot_manager

logger = logging.getLogger("market_tracker")

# Define Snapshot Directory
# Use the centralized definition from snapshot_manager
SNAPSHOT_DIR = snapshot_manager.SNAPSHOT_DIR
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Time Windows (ET)
NOON_BASELINE_START = time(9, 0)
NOON_BASELINE_END = time(13, 0)

EVENING_UPDATE_START = time(17, 30)
EVENING_UPDATE_END = time(19, 30)

LATE_UPDATE_START = time(21, 0)
LATE_UPDATE_END = time(23, 0)

def get_et_now():
    """Returns current time in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York"))

def get_baseline_filename(date_str):
    """Helper to get noon baseline filename from snapshot manager."""
    return snapshot_manager.get_snapshot_filename(date_str, "noon_baseline")

def get_snapshot_filename(date_str, suffix):
    """Returns the expected filename for a given date and suffix."""
    return os.path.join(SNAPSHOT_DIR, f"{date_str}_{suffix}.csv")

def save_snapshot(df: pd.DataFrame):
    """
    Saves the dataframe snapshot based on current time window.
    - Noon Baseline: [YYYY-MM-DD]_noon_baseline.csv (9a-1p)
      * Checks existence; does not overwrite if present (as per requirements).
    - Evening Update: [YYYY-MM-DD]_evening_update.csv (5:30p-7:30p)
    - Late Update: [YYYY-MM-DD]_late_update.csv (9p-11p)
    """
    if df is None or df.empty:
        return

    now_et = get_et_now()
    current_time = now_et.time()
    date_str = now_et.date().isoformat()

    suffix = None
    check_exists = False

    if NOON_BASELINE_START <= current_time <= NOON_BASELINE_END:
        suffix = "noon_baseline"
        check_exists = True
    elif EVENING_UPDATE_START <= current_time <= EVENING_UPDATE_END:
        suffix = "evening_update"
    elif LATE_UPDATE_START <= current_time <= LATE_UPDATE_END:
        suffix = "late_update"

    if not suffix:
        return # Outside windows

    filename = get_snapshot_filename(date_str, suffix)

    # Task 1: "Check for existence on startup; do not re-scrape if present" (for Noon)
    if check_exists and os.path.exists(filename):
        logger.info(f"Snapshot {filename} already exists. Skipping save.")
        return

    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # Columns to persist for CLV and movement tracking
        cols_to_save = [
            'league', 'Home', 'Away', 'Commence (UTC)',
            'Implied_Prob', 'Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff',
            'Spread & Pick', 'Total & Pick', 'final_probability', 'Pick',
            'spread_pick_line', 'total_pick_line', 'spread_edge', 'total_edge'
        ]

        # Intersect with actual columns
        cols = [c for c in cols_to_save if c in df.columns]

        df[cols].to_csv(filename, index=False)
        logger.info(f"Saved snapshot to {filename}")

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")

def load_and_compare(current_df: pd.DataFrame):
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

    # Trigger: Run initiated after 5:30 PM
    if current_time < EVENING_UPDATE_START:
        return current_df

    baseline_file = get_snapshot_filename(date_str, "noon_baseline")

    if not os.path.exists(baseline_file):
        logger.info("No noon baseline found for comparison.")
        return current_df

    try:
        baseline_df = pd.read_csv(baseline_file)

        # Rename baseline columns with suffix
        metric_cols = {
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

        # Select subset
        baseline_subset = baseline_df[['league', 'Home', 'Away'] + [c for c in metric_cols.keys() if c in baseline_df.columns]].copy()
        baseline_subset.rename(columns=metric_cols, inplace=True)

        # Merge on composite key
        merged = pd.merge(current_df, baseline_subset, on=['league', 'Home', 'Away'], how='left')

        # Safety Check: Ensure we didn't lose rows (Left Join should preserve them)
        if len(merged) < len(current_df):
            logger.warning(f"CRITICAL: Market Tracker merge truncated data! Input: {len(current_df)}, Output: {len(merged)}")
            return current_df

        # --- CALCULATE DELTAS ---

        # Helper for safe numeric subtraction
        def calc_delta(curr_col, noon_col):
            if curr_col in merged.columns and noon_col in merged.columns:
                return pd.to_numeric(merged[curr_col], errors='coerce') - pd.to_numeric(merged[noon_col], errors='coerce')
            return 0.0

        merged['delta_implied_prob'] = calc_delta('Implied_Prob', 'Implied_Prob_noon')
        merged['delta_sentiment'] = calc_delta('Sentiment_Diff', 'Sentiment_Diff_noon')
        merged['line_move_spread'] = calc_delta('spread_pick_line', 'spread_pick_line_noon')
        merged['line_move_total'] = calc_delta('total_pick_line', 'total_pick_line_noon')

        # CLV Deltas (Edge movement)
        # Difference between early-day edge and final closing line (current edge)
        # Note: If edge decreased (e.g. 5% -> 1%), it means market moved towards our pick (price got worse/efficient).
        merged['clv_spread_edge_diff'] = calc_delta('spread_edge', 'spread_edge_noon')
        merged['clv_total_edge_diff'] = calc_delta('total_edge', 'total_edge_noon')

        # --- ALERTS & FLAGS ---
        def _get_movement_alert(row):
            alerts = []

            # Market Steam: > 3% prob move
            delta_prob = row.get('delta_implied_prob', 0)
            if pd.notnull(delta_prob) and abs(delta_prob) > 0.03:
                direction = "Steam" if delta_prob > 0 else "Drift"
                alerts.append(f"Market {direction} ({delta_prob:+.1%})")

            # Sentiment Shift
            delta_sent = row.get('delta_sentiment', 0)
            if pd.notnull(delta_sent) and abs(delta_sent) > 0.1: # Threshold for significant sentiment shift?
                 direction = "Positive" if delta_sent > 0 else "Negative"
                 alerts.append(f"Sentiment Shift {direction} ({delta_sent:+.2f})")

            # Line Movement >= 0.5 (Task 1 Req)
            spread_move = row.get('line_move_spread', 0)
            if pd.notnull(spread_move) and abs(spread_move) >= 0.5:
                alerts.append(f"Spread Move: {spread_move:+.1f}")
                # Interpretation depends on side picked, but raw movement is useful enough.

            total_move = row.get('line_move_total', 0)
            if pd.notnull(total_move) and abs(total_move) >= 0.5:
                alerts.append(f"Total Move: {total_move:+.1f}")

            return "; ".join(alerts)

        merged['movement_alerts'] = merged.apply(_get_movement_alert, axis=1)

        # Task 3: Late Game CLV Logic
        # "For games starting after 9 PM, ensure the system captures the 'Late Update' as the final Closing Line."
        # This function runs whenever called. If called during Late window, it captures the current state vs noon.
        # We can flag CLV specifically if in Late Window.
        if current_time >= LATE_UPDATE_START:
            merged['is_closing_line'] = True
        else:
            merged['is_closing_line'] = False

        return merged

    except Exception as e:
        logger.error(f"Failed to load/compare baseline: {e}")
        return current_df

# Maintain backward compatibility alias if needed, but updated calls should use save_snapshot
save_baseline_if_appropriate = save_snapshot
load_baseline_for_comparison = load_and_compare
