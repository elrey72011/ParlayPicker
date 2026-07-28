"""Absolute value gate for production sports-betting recommendations.

Candidate ranking answers "which available pick is best for this game?"  This
module answers the separate question "is that pick good enough to bet at the
offered price?"  Keeping the questions separate lets the UI show a directional
read for every game without turning a relative winner into a funded wager.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# A production pick must beat its exact price-implied break-even probability by
# at least two percentage points.  This is intentionally absolute rather than a
# within-slate rank: a bad slate is allowed to produce zero bets.
MIN_PRODUCTION_CALIBRATED_EDGE = 0.02
MIN_PRODUCTION_MODEL_EV = 0.0


def _numeric_series(value: object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(value, pd.Series):
        out = pd.to_numeric(value, errors="coerce")
        return out.reindex(index) if index is not None else out
    if index is None:
        if isinstance(value, (list, tuple, np.ndarray)):
            return pd.to_numeric(pd.Series(value), errors="coerce")
        return pd.to_numeric(pd.Series([value]), errors="coerce")
    if np.isscalar(value) or value is None:
        return pd.to_numeric(pd.Series(value, index=index), errors="coerce")
    return pd.to_numeric(pd.Series(value, index=index), errors="coerce")


def evaluate_absolute_production_gate(
    calibrated_probability: object,
    break_even_probability: object,
    model_expected_value: object,
    *,
    min_edge: float = MIN_PRODUCTION_CALIBRATED_EDGE,
    min_model_ev: float = MIN_PRODUCTION_MODEL_EV,
) -> pd.DataFrame:
    """Return an index-aligned production decision and its priced-edge metrics.

    A row passes only when all three inputs are usable, model EV is strictly
    positive, and calibrated probability clears the sportsbook break-even
    probability by ``min_edge``.  Missing calibration or price fails closed.
    """
    if isinstance(calibrated_probability, pd.Series):
        index = calibrated_probability.index
    elif isinstance(break_even_probability, pd.Series):
        index = break_even_probability.index
    elif isinstance(model_expected_value, pd.Series):
        index = model_expected_value.index
    else:
        index = None

    probability = _numeric_series(calibrated_probability, index)
    if index is None:
        index = probability.index
    break_even = _numeric_series(break_even_probability, index)
    model_ev = _numeric_series(model_expected_value, index)

    absolute_edge = probability - break_even
    calibrated_ev = (probability / break_even) - 1.0
    valid = (
        probability.between(0.0, 1.0, inclusive="both")
        & break_even.gt(0.0)
        & break_even.lt(1.0)
        & model_ev.notna()
    )
    passed = (
        valid
        & model_ev.gt(float(min_model_ev))
        & absolute_edge.ge(float(min_edge))
    )

    reason = pd.Series("qualified", index=index, dtype="object")
    reason.loc[probability.isna()] = "missing calibrated probability"
    reason.loc[probability.notna() & ~probability.between(0.0, 1.0, inclusive="both")] = (
        "invalid calibrated probability"
    )
    reason.loc[break_even.isna()] = "missing sportsbook break-even price"
    reason.loc[break_even.notna() & ~(break_even.gt(0.0) & break_even.lt(1.0))] = (
        "invalid sportsbook break-even price"
    )
    reason.loc[valid & ~model_ev.gt(float(min_model_ev))] = "model EV is not positive"
    thin_edge = valid & model_ev.gt(float(min_model_ev)) & ~absolute_edge.ge(float(min_edge))
    reason.loc[thin_edge] = absolute_edge.loc[thin_edge].map(
        lambda edge: (
            f"calibrated edge below {float(min_edge):.1%} safety margin "
            f"({edge:+.1%})"
        )
    )

    return pd.DataFrame(
        {
            "production_gate_pass": passed.fillna(False).astype(bool),
            "absolute_production_edge": absolute_edge,
            "calibrated_expected_value": calibrated_ev,
            "production_gate_reason": reason,
        },
        index=index,
    )

