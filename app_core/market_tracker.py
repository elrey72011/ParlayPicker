import os
import logging
import pandas as pd
from datetime import datetime, time, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo # Fallback if needed

logger = logging.getLogger("market_tracker")

# Define Snapshot Directory
# Prefer absolute path for persistent mount if it exists, otherwise fallback to local relative path
PERSISTENT_MOUNT_ROOT = "/mount/src/parlaypicker"
if os.path.exists(PERSISTENT_MOUNT_ROOT):
    SNAPSHOT_DIR = os.path.join(PERSISTENT_MOUNT_ROOT, "data", "snapshots")
else:
    SNAPSHOT_DIR = os.path.join("data", "snapshots")

NOON_BASELINE_START = time(9, 0)
NOON_BASELINE_END = time(13, 0)
EVENING_UPDATE_START = time(17, 30)

def get_et_now():
    """Returns current time in US/Eastern."""
    return datetime.now(ZoneInfo("America/New_York"))

def get_baseline_filename(date_str):
    """Returns the expected filename for a given date."""
    return os.path.join(SNAPSHOT_DIR, f"{date_str}_noon_baseline.csv")

def save_baseline_if_appropriate(df: pd.DataFrame):
    """
    Saves the dataframe as a noon baseline if within the time window.
    Checks if file exists to avoid overwriting (or maybe we want to overwrite to get the latest 'noon' snapshot?
    The prompt says "check... to see if a baseline... exists before prompting", implying we should probably respect existing unless forced.
    For auto-save, we'll save if it doesn't exist, or maybe update it.
    Let's stick to: Save if inside window.
    """
    if df is None or df.empty:
        return

    now_et = get_et_now()
    current_time = now_et.time()

    # Check window: 9:00 AM - 1:00 PM ET
    if NOON_BASELINE_START <= current_time <= NOON_BASELINE_END:
        date_str = now_et.date().isoformat()
        filename = get_baseline_filename(date_str)

        # Save relevant columns for comparison
        # We need identifying cols + metric cols
        cols_to_save = [
            'league', 'Home', 'Away', 'Commence (UTC)',
            'Implied_Prob', 'Home_Sentiment', 'Away_Sentiment', 'Sentiment_Diff',
            'Spread & Pick', 'Total & Pick', 'final_probability', 'Pick',
            'spread_pick_line', 'total_pick_line'
        ]

        # Filter existing cols
        cols = [c for c in cols_to_save if c in df.columns]

        try:
            # Ensure directory exists before saving
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)

            # We overwrite if in window to capture the "latest" noon state?
            # Or preserve the "first" noon state?
            # "Any data run completed... should automatically be saved".
            # If I run at 9:05 and 12:55, 12:55 is probably better "noon" data?
            # Or maybe 9:05 is better baseline?
            # Let's save, but log it.

            # Actually, "Consistency: If the app restarts, it must check ... to see if a baseline ... exists before prompting for a new one."
            # This implies manual workflow? But also "Auto-Save".
            # I will overwrite if in window, assuming latest in window is most accurate "noon" closing view before games.

            df[cols].to_csv(filename, index=False)
            logger.info(f"Saved noon baseline to {filename}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")

def load_baseline_for_comparison(current_df: pd.DataFrame):
    """
    Loads baseline for today and computes deltas if current time > 5:30 PM ET.
    Returns enriched dataframe with delta columns.
    """
    if current_df is None or current_df.empty:
        return current_df

    now_et = get_et_now()

    # Trigger: Run initiated after 5:30 PM
    if now_et.time() < EVENING_UPDATE_START:
        return current_df

    date_str = now_et.date().isoformat()
    filename = get_baseline_filename(date_str)

    if not os.path.exists(filename):
        logger.info("No baseline file found for comparison.")
        return current_df

    try:
        baseline_df = pd.read_csv(filename)

        # Merge on keys
        # We use a composite key of League + Home + Away
        # Commence might vary slightly if data provider updates it, so avoid strictly joining on it if possible,
        # but usually it's safe. Let's use League/Home/Away.

        # Prepare baseline for merge
        # Rename columns to _noon
        metric_cols = {
            'Implied_Prob': 'Implied_Prob_noon',
            'Home_Sentiment': 'Home_Sentiment_noon',
            'Away_Sentiment': 'Away_Sentiment_noon',
            'Sentiment_Diff': 'Sentiment_Diff_noon',
            'spread_pick_line': 'spread_pick_line_noon',
            'total_pick_line': 'total_pick_line_noon',
            'final_probability': 'final_probability_noon',
            'Pick': 'Pick_noon'
        }

        baseline_subset = baseline_df[['league', 'Home', 'Away'] + [c for c in metric_cols.keys() if c in baseline_df.columns]].copy()
        baseline_subset.rename(columns=metric_cols, inplace=True)

        # Merge
        merged = pd.merge(current_df, baseline_subset, on=['league', 'Home', 'Away'], how='left')

        # Calculate Deltas
        # 1. Delta Implied Prob
        if 'Implied_Prob_noon' in merged.columns and 'Implied_Prob' in merged.columns:
            merged['delta_implied_prob'] = pd.to_numeric(merged['Implied_Prob'], errors='coerce') - pd.to_numeric(merged['Implied_Prob_noon'], errors='coerce')

        # 2. Delta Sentiment
        if 'Sentiment_Diff_noon' in merged.columns and 'Sentiment_Diff' in merged.columns:
             merged['delta_sentiment'] = pd.to_numeric(merged['Sentiment_Diff'], errors='coerce') - pd.to_numeric(merged['Sentiment_Diff_noon'], errors='coerce')

        # 3. Line Movement
        # Spread
        if 'spread_pick_line_noon' in merged.columns and 'spread_pick_line' in merged.columns:
            # Check for movement
            # We store the raw movement value (current - noon)
            # Need to handle strings/nulls
            curr_line = pd.to_numeric(merged['spread_pick_line'], errors='coerce')
            noon_line = pd.to_numeric(merged['spread_pick_line_noon'], errors='coerce')
            merged['line_move_spread'] = curr_line - noon_line

        # Total
        if 'total_pick_line_noon' in merged.columns and 'total_pick_line' in merged.columns:
            curr_total = pd.to_numeric(merged['total_pick_line'], errors='coerce')
            noon_total = pd.to_numeric(merged['total_pick_line_noon'], errors='coerce')
            merged['line_move_total'] = curr_total - noon_total

        # 4. Warnings / Flags
        # "If the line moved against the model's original pick"
        # Original Pick is Pick_noon. Current Pick is Pick.
        # Actually, if the model pick changed, that's interesting too.
        # But specifically: "Line moving against original pick"

        def _get_movement_alert(row):
            alerts = []

            # Market Steam: > 3% prob move
            delta_prob = row.get('delta_implied_prob', 0)
            if pd.notnull(delta_prob) and abs(delta_prob) > 0.03:
                direction = "Steam" if delta_prob > 0 else "Drift" # Assuming positive is good for the pick?
                # Wait, implied prob is usually associated with a side. Which side?
                # The 'Implied_Prob' column in master_df is typically for the "Pick".
                # If Pick changed, comparing probs is apples to oranges.
                # Assuming Pick stayed same or we just track magnitude.
                alerts.append(f"Market {direction} ({delta_prob:+.1%})")

            # Line Move against pick
            # If we picked Home -5.5 and line is now -6.0, line moved AGAINST us (harder to cover).
            # If we picked Home -5.5 and line is now -5.0, line moved WITH us (easier).
            # If we picked Over 50.5 and line is now 51.5, line moved AGAINST us (harder).
            # If we picked Under 50.5 and line is now 49.5, line moved AGAINST us.

            pick_noon = str(row.get('Pick_noon') or "")

            # Spread Logic
            # General Rule: Delta > 0 implies market moving AGAINST the team's strength (e.g. -5 -> -4 or +5 -> +6).
            # If we picked the team, Delta > 0 is bad (Market Dislikes).
            # If we picked the team, Delta < 0 is good (Market Likes / Steam).
            spread_move = row.get('line_move_spread', 0)
            if pd.notnull(spread_move) and abs(spread_move) >= 0.5:
                alerts.append(f"Spread Move: {spread_move:+.1f}")
                if spread_move > 0:
                    alerts.append("⚠️ Market Fading Pick")
                elif spread_move < 0:
                    alerts.append("🔥 Market Steaming Pick")

            # Total Logic
            # Over: Delta > 0 (Steam/Good), Delta < 0 (Fade/Bad)
            # Under: Delta < 0 (Steam/Good), Delta > 0 (Fade/Bad)
            total_move = row.get('line_move_total', 0)
            if pd.notnull(total_move) and abs(total_move) >= 0.5:
                alerts.append(f"Total Move: {total_move:+.1f}")
                pick_side = str(row.get('Pick_noon') or "").lower()

                if "over" in pick_side:
                    if total_move < 0:
                        alerts.append("⚠️ Total Dropping (Against Over)")
                    else:
                        alerts.append("🔥 Total Rising (With Over)")
                elif "under" in pick_side:
                    if total_move > 0:
                        alerts.append("⚠️ Total Rising (Against Under)")
                    else:
                        alerts.append("🔥 Total Dropping (With Under)")

            return "; ".join(alerts)

        merged['movement_alerts'] = merged.apply(_get_movement_alert, axis=1)

        return merged

    except Exception as e:
        logger.error(f"Failed to load/compare baseline: {e}")
        return current_df
