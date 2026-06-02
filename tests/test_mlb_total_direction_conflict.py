"""Regression tests for MLB total Over/Under direction resolution.

Covers the "most-confident source wins" rule in
``core.streamlit_pipeline._mlb_total_direction_conflict``. The motivating defect
(1 Jun 2026 slate): TheOver and Kalshi disagreed on direction, their independent
fixed penalties cancelled across the total family, and selection collapsed to raw
EV/edge. The helper now penalizes only the losing direction, decided by whichever
source is further from 0.50.

Probabilities are oriented to each row's pick direction (>0.5 supports the row,
<0.5 favors the opposite side). For one game we pass the Over row and the Under
row together; exactly one should be flagged as the losing direction.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _mlb_total_direction_conflict


def _resolve(kalshi_over, theover_over):
    """Run the helper for one MLB game's Over+Under rows.

    Inputs are the Over-oriented probabilities; the Under row gets ``1 - p``.
    Returns ``(over_penalized, under_penalized)`` booleans.
    """
    kalshi = np.array([kalshi_over, 1.0 - kalshi_over], dtype=float)
    theover = np.array([theover_over, 1.0 - theover_over], dtype=float)
    is_mlb = np.array([True, True])
    _k_opp, _t_opp, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    return bool(conflict[0]), bool(conflict[1])


def test_no_cancellation_when_sources_disagree_more_confident_wins():
    # 1 Jun Cincinnati: Kalshi 0.585 Over (conf .085), TheOver 0.333 Over => .667
    # Under (conf .167). TheOver is more confident => Under is kept, Over penalized.
    over_pen, under_pen = _resolve(kalshi_over=0.585, theover_over=0.333)
    assert over_pen is True
    assert under_pen is False
    # The previous bug penalized BOTH rows (cancellation); ensure that cannot happen.
    assert not (over_pen and under_pen)


def test_kalshi_wins_direction_when_more_confident():
    # Kalshi strongly Over (0.66, conf .16) vs TheOver mildly Under-leaning
    # (0.45 Over, conf .05). Kalshi dominates => Under row penalized.
    over_pen, under_pen = _resolve(kalshi_over=0.66, theover_over=0.45)
    assert under_pen is True
    assert over_pen is False


def test_sources_agree_penalizes_the_losing_direction_only():
    # Both favor Over => the Under row is the loser; Over kept.
    over_pen, under_pen = _resolve(kalshi_over=0.62, theover_over=0.70)
    assert over_pen is False
    assert under_pen is True


def test_lone_source_decides_when_other_is_neutral_default():
    # TheOver at exactly 0.5 (default_0.5 "no read") => only Kalshi has an opinion.
    over_pen, under_pen = _resolve(kalshi_over=0.40, theover_over=0.50)
    # Kalshi favors Under => the Over row is penalized.
    assert over_pen is True
    assert under_pen is False


def test_confidence_tie_penalizes_neither():
    # Exactly-equal opposing confidence (0.125, an exact float): Kalshi supports Over
    # by .125, TheOver opposes Over by .125. Tie => defer to EV/edge, penalize neither.
    kalshi = np.array([0.625, 0.375], dtype=float)   # Over row: Kalshi favors Over
    theover = np.array([0.375, 0.625], dtype=float)  # Over row: TheOver favors Under
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    assert not conflict.any()


def test_both_neutral_or_missing_penalizes_neither():
    # TheOver default 0.5 and Kalshi NaN => no directional opinion at all.
    kalshi = np.array([np.nan, np.nan], dtype=float)
    theover = np.array([0.5, 0.5], dtype=float)
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    assert not conflict.any()


def test_non_mlb_total_rows_never_flagged():
    # Even a clear directional conflict is ignored for non-MLB-total rows.
    kalshi = np.array([0.30, 0.70], dtype=float)
    theover = np.array([0.80, 0.20], dtype=float)
    is_mlb = np.array([False, False])
    _k, _t, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    assert not conflict.any()


def test_untrusted_source_neutralizes_theover_kalshi_decides():
    # 2 Jun pattern: TheOver leans Under via model_hit_rate_flipped (P(Over)=0.30 =>
    # P(Under)=0.70), Kalshi leans Over (P(Under)=0.42). Untrusted source blanks
    # TheOver, so Kalshi decides => the Under row is the loser.
    kalshi = np.array([0.58, 0.42], dtype=float)   # Over row supports Over
    theover = np.array([0.30, 0.70], dtype=float)  # Under row "supports" Under via flip
    src = ["model_hit_rate_flipped", "model_hit_rate_flipped"]
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, untrusted_sources={"model_hit_rate_flipped"},
    )
    assert not conflict[0]  # Over kept
    assert conflict[1]      # Under penalized (flips pick to Over, aligned with Kalshi)


def test_trusted_source_keeps_theover_in_play():
    # Identical numbers, but a genuine model_hit_rate read is trusted: TheOver
    # (conf .20) beats Kalshi (conf .08) => the Over row is the loser, Under kept.
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([0.30, 0.70], dtype=float)
    src = ["model_hit_rate", "model_hit_rate"]
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, untrusted_sources={"model_hit_rate_flipped"},
    )
    assert conflict[0]       # Over penalized
    assert not conflict[1]   # Under kept


def test_nan_theover_defers_to_kalshi():
    # A blanked/absent TheOver (NaN) is no-opinion: a lone Kalshi favoring Over
    # penalizes the Under row.
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([np.nan, np.nan], dtype=float)
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    assert not conflict[0] and conflict[1]


def test_source_attribution_arrays():
    # When Kalshi opposes the losing (Over) row, the kalshi-opposes flag is set there.
    kalshi = np.array([0.40, 0.60], dtype=float)   # Over row: Kalshi favors Under
    theover = np.array([0.42, 0.58], dtype=float)  # Over row: TheOver favors Under too
    is_mlb = np.array([True, True])
    k_opp, t_opp, conflict = _mlb_total_direction_conflict(is_mlb, kalshi, theover)
    # Over row (index 0) is the loser; both sources opposed it.
    assert conflict[0] and not conflict[1]
    assert k_opp[0] and t_opp[0]
    assert not k_opp[1] and not t_opp[1]
