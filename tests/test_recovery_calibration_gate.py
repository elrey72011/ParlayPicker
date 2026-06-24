"""Empty-card recovery must not STAKE calibrated-negative picks.

The 6-24 card promoted two unders to Actionable (Toronto Under 8.5, TB Under 7.5) on raw
EV, but their calibrated win probability sat below break-even — the overconfidence the
calibration is meant to remove. This pins the gate that now blocks that.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _calibrated_beats_breakeven

# Calibration that shrinks the 0.50-0.55 band to ~.43-.45 (matches the live fitted table).
_CAL = [[0.45, 0.40], [0.52, 0.435], [0.55, 0.46], [0.60, 0.55], [0.70, 0.66]]


def test_overconfident_unders_fail_the_gate():
    # The two picks the 6-24 card actually staked.
    eff_win = pd.Series([0.5357, 0.5192])      # Toronto Under 8.5, TB Under 7.5
    odds = pd.Series([100.0, -102.0])          # break-even .500 / .505
    gate = _calibrated_beats_breakeven(eff_win, odds, _CAL)
    assert not gate.iloc[0]   # calibrated ~.45 < .500 -> blocked
    assert not gate.iloc[1]   # calibrated ~.43 < .505 -> blocked


def test_genuinely_strong_pick_passes_the_gate():
    eff_win = pd.Series([0.62])                # calibrates to ~.585
    odds = pd.Series([-110.0])                 # break-even ~.524
    assert _calibrated_beats_breakeven(eff_win, odds, _CAL).iloc[0]


def test_no_calibration_falls_back_to_raw_win():
    # Without a table, gate on the raw win vs break-even (no shrink).
    eff_win = pd.Series([0.5357])
    odds = pd.Series([100.0])                  # break-even .500
    assert _calibrated_beats_breakeven(eff_win, odds, None).iloc[0]   # .536 > .500 raw


def test_missing_odds_fails_closed():
    eff_win = pd.Series([0.70])
    odds = pd.Series([float("nan")])
    assert not _calibrated_beats_breakeven(eff_win, odds, _CAL).iloc[0]  # no price -> don't stake
