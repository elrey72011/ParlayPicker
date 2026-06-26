"""Public-betting-% is contrarian-faded on MLB totals (heavy public -> less likely)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _fade_contrarian_public, _scoped_theover_blend_fade
from app_core.weights_config import (
    MLB_PUBLIC_BETTING_FADE_SOURCES,
    MLB_PUBLIC_BETTING_FADE_STRENGTH,
)


def test_contrarian_inverts_deviation_from_half():
    # P(Over)_new = 0.5 - strength*(P-0.5). Heavy-public Over -> below 0.5.
    out = _fade_contrarian_public(
        [0.88, 0.50, 0.44], pd.Series(["public_betting_pct"] * 3),
        MLB_PUBLIC_BETTING_FADE_SOURCES, MLB_PUBLIC_BETTING_FADE_STRENGTH,
    )
    s = MLB_PUBLIC_BETTING_FADE_STRENGTH
    assert abs(out[0] - (0.5 - s * (0.88 - 0.5))) < 1e-9
    assert out[0] < 0.5                 # 88% public over -> treated as < 50% likely
    assert abs(out[1] - 0.5) < 1e-9     # pick'em public -> unchanged
    assert out[2] > 0.5                 # public on the under -> over treated as > 0.5 (fade)


def test_only_public_source_is_faded():
    # A non-public source passes through this transform untouched.
    out = _fade_contrarian_public(
        [0.84, 0.84], pd.Series(["public_betting_pct", "model_hit_rate"]),
        MLB_PUBLIC_BETTING_FADE_SOURCES, MLB_PUBLIC_BETTING_FADE_STRENGTH,
    )
    assert out[0] < 0.5            # public faded
    assert abs(out[1] - 0.84) < 1e-9  # model_hit_rate untouched here (it's neutralized elsewhere)


def test_scoped_blend_fades_public_on_mlb_totals_only():
    idx = pd.RangeIndex(2)
    # Same public 0.84 Over: MLB total -> contrarian-faded; NBA total -> followed (untouched).
    out = _scoped_theover_blend_fade(
        [0.84, 0.84], pd.Series(["public_betting_pct", "public_betting_pct"]),
        ["MLB", "NBA"], ["total_over", "total_over"], idx,
    )
    assert out.iloc[0] < 0.5            # MLB total -> faded
    assert abs(out.iloc[1] - 0.84) < 1e-9  # NBA total -> not MLB-scoped, left alone


def test_scoped_blend_still_neutralizes_model_hit_rate():
    idx = pd.RangeIndex(2)
    out = _scoped_theover_blend_fade(
        [0.875, 0.30], pd.Series(["model_hit_rate", "model_hit_rate_flipped"]),
        ["MLB", "MLB"], ["total_over", "total_under"], idx,
    )
    assert abs(out.iloc[0] - 0.5) < 1e-9
    assert abs(out.iloc[1] - 0.5) < 1e-9
