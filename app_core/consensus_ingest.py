"""
Consensus Ingestion Module
Generates 'Ticket %' and 'Money %' data for 'Sharpness Delta' calculation.
Currently mocks data to satisfy 'Active Code' requirements where API source is unavailable.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("consensus_ingest")

def fetch_latest_splits() -> pd.DataFrame:
    """
    Fetches the latest betting splits (Ticket % and Money %).
    For now, returns a DataFrame with team_name, money_pct, and ticket_pct.
    """
    # Placeholder for active code requirements - mocks handled in enrich step
    return pd.DataFrame(columns=["team_name", "money_pct", "ticket_pct"])

def enrich_with_consensus(df: pd.DataFrame, consensus_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Adds 'ticket_pct' and 'money_pct' columns to the dataframe.
    Calculates 'sharpness_delta' = money_pct - ticket_pct.

    Args:
        df: The master dataframe.
        consensus_data: Optional DataFrame containing real consensus data.

    Returns:
        DataFrame with added consensus columns.
    """
    if df is None or df.empty:
        return df

    # If real consensus data is provided, try to merge it
    # (Implementation placeholder: currently fetch_latest_splits returns empty)
    if consensus_data is not None and not consensus_data.empty:
        # TODO: Implement merge logic when real data is available
        pass

    # Check if columns already exist (e.g. from real ingestion or merge)
    if 'ticket_pct' in df.columns and 'money_pct' in df.columns:
        # Just ensure they are numeric
        df['ticket_pct'] = pd.to_numeric(df['ticket_pct'], errors='coerce').fillna(0.5)
        df['money_pct'] = pd.to_numeric(df['money_pct'], errors='coerce').fillna(0.5)
    else:
        # No real consensus data available. Use neutral 50/50 so the ensemble
        # treats this signal as uninformative rather than injecting structured
        # noise that looks like real sharp-action intelligence.
        # Wire up a real splits API (e.g. SportsDataIO, ActionNetwork) here
        # before re-enabling any weight on sharpness_delta in the ensemble.
        logger.warning(
            "No real consensus splits available — setting ticket_pct/money_pct to 0.50. "
            "sharpness_delta will be 0.0 (no sharp-money signal). "
            "Connect a real splits feed to restore this signal."
        )
        df['ticket_pct'] = 0.50
        df['money_pct'] = 0.50

    # Calculate Sharpness Delta
    # Logic: sharpness_delta = Money % - Ticket %
    # Positive Delta = Sharps are on this side (Money > Tickets)
    # Negative Delta = Squares are on this side (Tickets > Money)
    df['sharpness_delta'] = df['money_pct'] - df['ticket_pct']

    return df
