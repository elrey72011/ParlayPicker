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


def test_equal_raw_confidence_breaks_toward_kalshi_when_weighted():
    # Equal RAW confidence (0.125 each): Kalshi supports Over, TheOver supports Under.
    # With the default Kalshi direction weight (>1), Kalshi's weighted confidence wins
    # the otherwise-tied call, so the Under row is penalized and the Kalshi-aligned Over
    # stands. This encodes the n=363 finding that Kalshi is the more reliable direction
    # signal on MLB totals.
    kalshi = np.array([0.625, 0.375], dtype=float)   # Over row: Kalshi favors Over
    theover = np.array([0.375, 0.625], dtype=float)  # Over row: TheOver favors Under
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover, kalshi_confidence_weight=1.5
    )
    assert not conflict[0]  # Over (Kalshi's side) kept
    assert conflict[1]      # Under penalized


def test_confidence_tie_penalizes_neither_at_unit_weight():
    # With weight 1.0 (symmetric legacy behavior) an exact confidence tie defers to
    # EV/edge and penalizes neither row.
    kalshi = np.array([0.625, 0.375], dtype=float)
    theover = np.array([0.375, 0.625], dtype=float)
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover, kalshi_confidence_weight=1.0
    )
    assert not conflict.any()


def test_kalshi_weight_reclaims_marginally_stronger_theover():
    # Kalshi mildly Over (0.60, raw conf .10) vs TheOver mildly Under (0.37, raw conf
    # .13). Symmetric (weight 1.0): TheOver wins, Under stands (the losing Disagrees
    # pick). Weighted 1.5: Kalshi's .15 > TheOver's .13, so the Under is penalized and
    # the Kalshi-aligned Over stands — exactly the marginal case the weight reclaims.
    kalshi = np.array([0.60, 0.40], dtype=float)
    theover = np.array([0.37, 0.63], dtype=float)
    is_mlb = np.array([True, True])
    _k, _t, sym = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover, kalshi_confidence_weight=1.0
    )
    assert sym[0] and not sym[1]  # legacy: Over penalized, Under stands
    _k, _t, weighted = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover, kalshi_confidence_weight=1.5
    )
    assert not weighted[0] and weighted[1]  # weighted: Under penalized, Kalshi's Over stands


def test_kalshi_weight_does_not_override_decisively_stronger_theover():
    # A genuinely strong TheOver pitcher read (0.30, conf .20) still beats a mildly
    # confident Kalshi (0.58, conf .08) even weighted 1.5 (.12 < .20): TheOver's Under
    # stands. The weight reclaims marginal calls, not decisive ones.
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([0.30, 0.70], dtype=float)
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover, kalshi_confidence_weight=1.5
    )
    assert conflict[0]       # Over penalized -> TheOver's Under stands
    assert not conflict[1]


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


def test_faded_source_lets_kalshi_decide():
    # 2 Jun pattern: TheOver leans Under via model_hit_rate_flipped (P(Over)=0.30 =>
    # conf .20), Kalshi leans Over (0.58, conf .08). With a strong fade (shrink 0.75),
    # TheOver's 0.30 shrinks to 0.425 (conf .075) < Kalshi's .08, so Kalshi decides =>
    # the Under row is the loser (pick flips to the Kalshi-aligned Over).
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([0.30, 0.70], dtype=float)
    src = ["model_hit_rate_flipped", "model_hit_rate_flipped"]
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, fade_sources={"model_hit_rate_flipped"}, fade_shrink=0.75,
    )
    assert not conflict[0]  # Over kept
    assert conflict[1]      # Under penalized

    # Full fade (shrink 1.0) reproduces the old "drop it" behavior identically.
    _k, _t, conflict_full = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, fade_sources={"model_hit_rate_flipped"}, fade_shrink=1.0,
    )
    assert not conflict_full[0] and conflict_full[1]


def test_light_fade_still_lets_strong_theover_win():
    # A light fade (shrink 0.25) keeps most of TheOver's conviction: 0.30 -> 0.35
    # (conf .15) still beats Kalshi's .08, so TheOver's Under stands. This is the
    # "respect the signal" end of the knob.
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([0.30, 0.70], dtype=float)
    src = ["model_hit_rate_flipped", "model_hit_rate_flipped"]
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, fade_sources={"model_hit_rate_flipped"}, fade_shrink=0.25,
    )
    assert conflict[0]       # Over penalized -> Under stands (TheOver respected)
    assert not conflict[1]


def test_trusted_source_keeps_theover_in_play():
    # A genuine model_hit_rate read is never faded (not in fade_sources): TheOver
    # (conf .20) beats Kalshi (conf .08) => the Over row is the loser, Under kept.
    kalshi = np.array([0.58, 0.42], dtype=float)
    theover = np.array([0.30, 0.70], dtype=float)
    src = ["model_hit_rate", "model_hit_rate"]
    is_mlb = np.array([True, True])
    _k, _t, conflict = _mlb_total_direction_conflict(
        is_mlb, kalshi, theover,
        theover_source=src, fade_sources={"model_hit_rate_flipped"}, fade_shrink=0.75,
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
