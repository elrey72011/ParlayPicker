import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _is_placeholder(val):
    if pd.isna(val) or val is None or val == "":
        return True
    if isinstance(val, (int, float)):
        # Treat typical defaults as placeholders for critical cols
        if not np.isfinite(val):
            return True
        if val in [0.0, 0.5, 1.0]:
            return True
    return False

def collapse_duplicate_columns(df: pd.DataFrame, critical_cols: list = None) -> pd.DataFrame:
    """
    Detects and collapses duplicated column names in a DataFrame row-wise.
    For each duplicated column name:
    - Iterates left-to-right.
    - Picks the first non-null, non-empty, finite value.
    - If it's a critical column, prefers the most informative populated value over placeholders if possible.
    - Assigns the canonical collapsed column back to the name.
    - Leaves the DataFrame with unique column names.

    Args:
        df: The DataFrame to clean.
        critical_cols: List of column names to prioritize valid data over placeholders (like 0.5, 0.0, '').

    Returns:
        DataFrame with unique column names.
    """
    if not df.columns.duplicated().any():
        return df

    if critical_cols is None:
        critical_cols = []

    dup_cols = df.columns[df.columns.duplicated()].unique()
    logger.info(f"Collapsing {len(dup_cols)} duplicate column names: {list(dup_cols)}")

    # Operate on a copy to avoid SettingWithCopy
    out_df = df.copy()

    # We will build a new set of dataframes to concatenate to avoid fragmentation and insertion issues
    cols_to_drop = list(dup_cols)
    canonical_series_dict = {}

    for col_name in dup_cols:
        dup_df = out_df.loc[:, out_df.columns == col_name]
        logger.info(f"Column '{col_name}' appears {dup_df.shape[1]} times. Coalescing...")

        is_critical = col_name in critical_cols

        # We will coalesce row by row
        collapsed_vals = []
        for idx in range(len(dup_df)):
            row = dup_df.iloc[idx]

            best_val = np.nan
            found_valid = False

            for i in range(len(row)):
                val = row.iloc[i]

                # Check empty
                if pd.isna(val) or val == "" or val is None:
                    continue

                # Check finite
                if isinstance(val, (int, float)) and not np.isfinite(val):
                    continue

                if is_critical:
                    if not found_valid:
                        best_val = val
                        found_valid = True
                    else:
                        # If we already have a valid value, only replace if the NEW one is more informative
                        if _is_placeholder(best_val) and not _is_placeholder(val):
                            best_val = val
                else:
                    # Non-critical: Left-to-right wins immediately
                    best_val = val
                    found_valid = True
                    break

            collapsed_vals.append(best_val)

        canonical_series_dict[col_name] = pd.Series(collapsed_vals, index=out_df.index)

    # Drop all duplicated columns
    out_df = out_df.loc[:, ~out_df.columns.isin(cols_to_drop)]

    # Add back the coalesced canonical columns
    for col_name, series in canonical_series_dict.items():
        out_df[col_name] = series

    if out_df.columns.duplicated().any():
        remaining_dups = out_df.columns[out_df.columns.duplicated()].unique()
        raise RuntimeError(f"Duplicate columns STILL exist after collapse: {list(remaining_dups)}")

    return out_df
