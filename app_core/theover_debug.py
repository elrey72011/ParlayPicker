import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger("theover_debug")

def generate_debug_export(debug_log: List[Dict[str, Any]]) -> str:
    """
    Converts the TheOver debug log (list of dicts) into a CSV string.
    Includes columns: csv_home, csv_away, matched_home, matched_away, confidence, status, closest_matches
    """
    if not debug_log:
        return ""

    try:
        df = pd.DataFrame(debug_log)

        # Ensure we have the desired columns (fill missing with empty strings)
        desired_cols = [
            "csv_home", "csv_away",
            "matched_home", "matched_away",
            "confidence", "status",
            "closest_matches", "source_date", "target_date"
        ]

        # Filter/Reorder if columns exist
        existing_cols = [c for c in desired_cols if c in df.columns]
        df = df[existing_cols]

        return df.to_csv(index=False)
    except Exception as e:
        logger.error(f"Failed to generate TheOver debug export: {e}")
        return ""
