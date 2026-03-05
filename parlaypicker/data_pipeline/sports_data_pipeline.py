"""Simplified end-to-end sports data pipeline."""
from __future__ import annotations

import pandas as pd
from parlaypicker.data_pipeline.odds_collector import fetch_odds
from parlaypicker.data_pipeline.feature_engineering import engineer_features


def run_pipeline(sport: str, date_key: str, workers: int = 1) -> pd.DataFrame:
    payload = fetch_odds(sport, date_key)
    games = payload.get("games", [])
    df = pd.DataFrame(games)
    if df.empty:
        return df
    return engineer_features(df, workers=workers)
