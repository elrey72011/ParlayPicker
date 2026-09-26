"""Date and quote evidence for older ranking tests unrelated to chronology.

Production never infers these timestamps.  The tests supply them explicitly
here so their ranking and status assertions run inside a synthetic pregame
slate; chronology-specific tests exercise missing and late times separately.
"""
from unittest.mock import patch

import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def build_pregame_best_picks_df(analysis: pd.DataFrame, *args, **kwargs):
    frame = analysis.copy()
    dates = pd.to_datetime(frame.get("game_date"), utc=True, errors="coerce")
    dates = dates.fillna(pd.Timestamp("2099-01-01T00:00:00Z"))
    starts = dates.dt.normalize() + pd.Timedelta(hours=20)
    if "game_start_utc" not in frame:
        frame["game_start_utc"] = starts.map(lambda value: value.isoformat())
    else:
        frame["game_start_utc"] = frame["game_start_utc"].fillna(starts.map(lambda value: value.isoformat()))
    if "odds_recorded_at" not in frame:
        frame["odds_recorded_at"] = (starts - pd.Timedelta(hours=1)).map(lambda value: value.isoformat())
    else:
        frame["odds_recorded_at"] = frame["odds_recorded_at"].fillna((starts - pd.Timedelta(hours=1)).map(lambda value: value.isoformat()))
    as_of = starts.min() - pd.Timedelta(hours=2)
    with patch("app_core.candidate_chronology.now_utc", return_value=as_of):
        return build_best_picks_df(frame, *args, **kwargs)
