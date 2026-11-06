"""Pinnacle's no-vig probability for a pick, as a shadow market anchor (D4; owner 2026-11-06).

Pinnacle is the sharp reference. Computed only when Pinnacle priced both sides at **exactly**
the pick's line and its quote is fresh (``MAX_AGE_MINUTES``); otherwise None, never a
guess. Descriptive: the live blend still anchors on the book price until the owner
switches, after the blend-weight study and a shadow comparison.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

from core.clv import no_vig_prob

MAX_AGE_MINUTES = 20


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _line(row):
    for key in ("market_line_used", "total_line" if str(row.get("market_type", "")).startswith("total") else "spread_line"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def fair(row, now=None):
    """P(pick side) from Pinnacle's two sides at the pick's exact line, or None."""
    now = now or datetime.now(timezone.utc)
    price, opposing, point = (_number(row.get(k)) for k in
                              ("pinnacle_odds_american", "pinnacle_opposing_odds_american", "pinnacle_point"))
    line = _line(row)
    if None in (price, opposing, point, line) or abs(point - line) > 1e-9 or abs(price) < 100 or abs(opposing) < 100:
        return None
    try:
        updated = datetime.fromisoformat(str(row.get("pinnacle_updated_at")).replace("Z", "+00:00"))
    except ValueError:
        return None
    if updated.tzinfo is None or not 0 <= (now - updated).total_seconds() <= MAX_AGE_MINUTES * 60:
        return None
    return no_vig_prob(price, opposing)


def fair_series(frame, now=None):
    import pandas as pd
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")
    return pd.Series([fair(r, now) for r in frame.to_dict("records")], index=frame.index, dtype="float64")
