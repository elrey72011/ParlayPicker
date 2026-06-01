"""Conditioned carve-out for the sub-8.0 MLB total_over guardrail.

``core/streamlit_pipeline.py`` demotes every MLB ``total_over`` whose line is
below ``MLB_OVER_MIN_TOTAL_LINE`` (8.0) to "High Variance/Speculative". That
blanket rule also suppresses strong, market-aligned low-line overs. This module
holds the single, pure predicate that decides when such a pick may KEEP its
already-earned ``Actionable`` status instead of being force-demoted.

It is intentionally dependency-free and side-effect-free so the decision can be
unit tested in isolation (see ``tests/test_low_line_over_override.py``) and so
the betting behaviour change is trivially reversible via the
``MLB_LOW_LINE_OVER_OVERRIDE_ENABLED`` flag in ``app_core/weights_config.py``.

Supporting backtest: ``scripts/backtest_low_line_over.py``.
"""
from __future__ import annotations

from app_core.weights_config import (
    MLB_OVER_MIN_TOTAL_LINE,
    MLB_LOW_LINE_OVER_OVERRIDE_ENABLED,
    MLB_LOW_LINE_OVER_OVERRIDE_MIN_WIN_PROB,
    MLB_LOW_LINE_OVER_OVERRIDE_MIN_EDGE,
)


def low_line_over_override_applies(
    *,
    league: str | None,
    market_type: str | None,
    total_line: float | None,
    status: str | None,
    consensus: str | None,
    effective_win_probability: float | None,
    effective_edge: float | None,
) -> bool:
    """Return True if a sub-8.0 MLB over should keep its ``Actionable`` status.

    All conditions must hold:
      * the feature flag is enabled;
      * it is an MLB ``total_over`` with a line below ``MLB_OVER_MIN_TOTAL_LINE``;
      * the pick is *already* ``Actionable`` (the carve-out never promotes a
        Below Threshold pick — it only prevents a demotion);
      * consensus is ``Agrees`` (Neutral/Disagrees low-line overs stay vetoed,
        per the backtest); and
      * the shrinkage-adjusted win probability and edge clear their floors.
    """
    if not MLB_LOW_LINE_OVER_OVERRIDE_ENABLED:
        return False
    if (league or "").strip().upper() != "MLB":
        return False
    if (market_type or "").strip().lower() != "total_over":
        return False
    if total_line is None:
        return False
    try:
        if float(total_line) >= MLB_OVER_MIN_TOTAL_LINE:
            return False
    except (TypeError, ValueError):
        return False
    if status != "Actionable":
        return False
    if (consensus or "").strip() != "Agrees":
        return False
    if effective_win_probability is None or effective_edge is None:
        return False
    return (
        effective_win_probability >= MLB_LOW_LINE_OVER_OVERRIDE_MIN_WIN_PROB
        and effective_edge >= MLB_LOW_LINE_OVER_OVERRIDE_MIN_EDGE
    )
