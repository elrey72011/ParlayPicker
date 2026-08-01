"""The MLB-tuned TheOver fade shrink must stay scoped to MLB totals.

`MLB_THEOVER_FADE_SHRINK` was eased 0.75 -> 0.25 on MLB-only flipped-game evidence,
but `_fade_theover` runs over the whole blend frame and a faded WinProbSource
(`model_hit_rate_flipped`) can ride on NBA/NHL totals too. `_scoped_theover_blend_fade`
must apply the MLB shrink only to MLB totals and keep `THEOVER_FADE_SHRINK_DEFAULT`
elsewhere, so the MLB tuning never silently moves non-MLB calibrated probabilities.
"""
import numpy as np
import pandas as pd

from core.streamlit_pipeline import _scoped_theover_blend_fade
from app_core.weights_config import MLB_THEOVER_FADE_SHRINK, THEOVER_FADE_SHRINK_DEFAULT


def _faded(p, shrink):
    return 0.5 + (p - 0.5) * (1.0 - shrink)


def test_mlb_total_gets_reduced_shrink_non_mlb_keeps_default():
    idx = [0, 1, 2, 3]
    theover = np.array([0.20, 0.20, 0.20, 0.20])
    src = pd.Series(["model_hit_rate_flipped"] * 4)
    league = pd.Series(["MLB", "NHL", "NBA", "MLB"])
    market_type = pd.Series(["total_over", "total_under", "total_over", "spread_home"])

    out = _scoped_theover_blend_fade(theover, src, league, market_type, idx)

    # MLB total -> reduced MLB shrink (0.25)
    assert abs(out.iloc[0] - _faded(0.20, MLB_THEOVER_FADE_SHRINK)) < 1e-9
    # NHL / NBA totals -> legacy default shrink, unchanged by the MLB retune
    assert abs(out.iloc[1] - _faded(0.20, THEOVER_FADE_SHRINK_DEFAULT)) < 1e-9
    assert abs(out.iloc[2] - _faded(0.20, THEOVER_FADE_SHRINK_DEFAULT)) < 1e-9
    # MLB non-total (spread) -> default, not the totals-only MLB shrink
    assert abs(out.iloc[3] - _faded(0.20, THEOVER_FADE_SHRINK_DEFAULT)) < 1e-9


def test_mlb_and_non_mlb_diverge_only_because_shrinks_differ():
    # Guards the regression: if someone collapses the two shrinks back to one constant,
    # MLB and non-MLB totals would produce identical fades and this separation breaks.
    assert MLB_THEOVER_FADE_SHRINK != THEOVER_FADE_SHRINK_DEFAULT
    theover = np.array([0.20, 0.20])
    src = pd.Series(["model_hit_rate_flipped", "model_hit_rate_flipped"])
    out = _scoped_theover_blend_fade(
        theover, src, pd.Series(["MLB", "NHL"]), pd.Series(["total_over", "total_over"]), [0, 1]
    )
    assert out.iloc[0] != out.iloc[1]


def test_unfaded_sources_pass_through_regardless_of_league():
    # A non-faded source (e.g. model_hit_rate) is untouched in every league/market.
    theover = np.array([0.83, 0.83])
    src = pd.Series(["independent_projection", "independent_projection"])
    out = _scoped_theover_blend_fade(
        theover, src, pd.Series(["MLB", "NBA"]), pd.Series(["total_over", "total_over"]), [0, 1]
    )
    assert abs(out.iloc[0] - 0.83) < 1e-9
    assert abs(out.iloc[1] - 0.83) < 1e-9
