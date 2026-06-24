"""Bucket-conditional calibration: global isotonic curve + per-bucket realized tilt."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.probability_calibration import apply_bucket_calibration, apply_calibration

# Steep global curve (mirrors the live shape: 0.55->.49, 0.58->.53, 0.62->.57).
_CAL = [[0.50, 0.44], [0.55, 0.49], [0.58, 0.53], [0.62, 0.57], [0.69, 0.625]]
_BSTATS = {
    "overall": {"n": 313, "win_rate": 0.52},
    "buckets": {
        "MLB:under:Agrees": {"n": 62, "wins": 38, "win_rate": 0.613},   # proven good
        "MLB:over:Neutral": {"n": 92, "wins": 45, "win_rate": 0.489},   # proven bad
        "MLB:under:Disagrees": {"n": 28, "wins": 14, "win_rate": 0.50},  # coin flip
    },
}


def test_proven_good_bucket_lifts_above_global():
    probs = pd.Series([0.55])
    glob = float(apply_calibration(probs, _CAL).iloc[0])
    bucket = float(apply_bucket_calibration(probs, ["MLB:under:Agrees"], _CAL, _BSTATS).iloc[0])
    assert bucket > glob               # under:Agrees is credited
    assert bucket > 0.524              # ...enough to clear a -110 break-even


def test_proven_bad_bucket_drops_below_global():
    probs = pd.Series([0.60])
    glob = float(apply_calibration(probs, _CAL).iloc[0])
    bucket = float(apply_bucket_calibration(probs, ["MLB:over:Neutral"], _CAL, _BSTATS).iloc[0])
    assert bucket < glob               # over:Neutral is penalized


def test_unknown_bucket_and_missing_stats_fall_back_to_global():
    probs = pd.Series([0.58, 0.58])
    glob = apply_calibration(probs, _CAL)
    # unknown bucket -> no tilt
    same = apply_bucket_calibration(probs, ["MLB:side:Whatever", "NHL:over:Agrees"], _CAL, _BSTATS)
    assert abs(float(same.iloc[0]) - float(glob.iloc[0])) < 1e-9
    # no stats at all -> plain global
    none_stats = apply_bucket_calibration(probs, ["MLB:under:Agrees", "x"], _CAL, None)
    assert abs(float(none_stats.iloc[0]) - float(glob.iloc[0])) < 1e-9


def test_thin_bucket_barely_moves():
    # A 28-pick coin-flip bucket shrinks hard toward overall -> small move.
    probs = pd.Series([0.58])
    glob = float(apply_calibration(probs, _CAL).iloc[0])
    bucket = float(apply_bucket_calibration(probs, ["MLB:under:Disagrees"], _CAL, _BSTATS).iloc[0])
    assert abs(bucket - glob) < 0.02


def test_nan_passthrough():
    out = apply_bucket_calibration(pd.Series([float("nan")]), ["MLB:under:Agrees"], _CAL, _BSTATS)
    assert pd.isna(out.iloc[0])
