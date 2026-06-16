"""Closing-line value (CLV) utilities — pure, dependency-light, unit-tested.

CLV is the single best available proxy for genuine betting edge: a bettor who
consistently beats the closing line (gets a better number/price than the market
settles at) is, over a large sample, a long-term winner regardless of short-run
results. The pipeline currently does NOT capture closing lines, so scripts/
capture_closing_lines.py snapshots them near game start and this module scores
the open-vs-close move for each pick.

All functions are pure and side-effect free so they can be unit tested without
the odds API. Probabilities are de-vigged two-way where both prices are known.
"""
from __future__ import annotations

from typing import Optional


def american_to_implied(odds: Optional[float]) -> Optional[float]:
    """Implied (vig-inclusive) win probability from American odds. None if unknown."""
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def no_vig_prob(pick_odds: Optional[float], opp_odds: Optional[float]) -> Optional[float]:
    """De-vigged probability of the pick side from both two-way American prices.

    Falls back to the raw implied probability when the opposing price is missing
    (still useful, just vig-inclusive).
    """
    p = american_to_implied(pick_odds)
    q = american_to_implied(opp_odds)
    if p is None:
        return None
    if q is None or (p + q) <= 0:
        return p
    return p / (p + q)


def line_clv(side: str, open_line: Optional[float], close_line: Optional[float]) -> Optional[float]:
    """Favorable line movement (in points) for a totals pick.

    Over wants the LOWER number, so a close above the entry line is favorable
    (you hold the cheaper number). Under wants the HIGHER number. Returns a
    signed value: positive = the number moved in the bettor's favor.
    """
    if open_line is None or close_line is None:
        return None
    try:
        o = float(open_line); c = float(close_line)
    except (TypeError, ValueError):
        return None
    s = str(side).strip().lower()
    if s.startswith("over"):
        return c - o
    if s.startswith("under"):
        return o - c
    return None


def price_clv(pick_open_odds: Optional[float], pick_close_odds: Optional[float],
              opp_open_odds: Optional[float] = None, opp_close_odds: Optional[float] = None) -> Optional[float]:
    """Favorable price movement, expressed as no-vig probability points.

    You beat the close on price when your entry no-vig prob is BELOW the closing
    no-vig prob (you bought the side cheaper than where it settled). Returns
    ``close_novig - open_novig``: positive = you got a better price than the close.
    """
    open_p = no_vig_prob(pick_open_odds, opp_open_odds)
    close_p = no_vig_prob(pick_close_odds, opp_close_odds)
    if open_p is None or close_p is None:
        return None
    return close_p - open_p


def closing_line_value(
    side: str,
    open_line: Optional[float] = None,
    close_line: Optional[float] = None,
    pick_open_odds: Optional[float] = None,
    pick_close_odds: Optional[float] = None,
    opp_open_odds: Optional[float] = None,
    opp_close_odds: Optional[float] = None,
) -> dict:
    """Score a pick's open-vs-close move.

    Returns ``{line_clv, price_clv, beat_close}`` where ``beat_close`` is True
    when the net move (line points + price prob, with line weighted ~0.5 pt per
    1% as a rough totals convention) is in the bettor's favor. ``beat_close`` is
    None when neither line nor price is computable.
    """
    lc = line_clv(side, open_line, close_line)
    pc = price_clv(pick_open_odds, pick_close_odds, opp_open_odds, opp_close_odds)
    # Combine: ~0.5 total points ≈ 1% no-vig prob for MLB totals (rough but standard).
    net = 0.0
    have = False
    if lc is not None:
        net += lc * 0.02  # 0.5 pt -> ~0.01 prob
        have = True
    if pc is not None:
        net += pc
        have = True
    return {
        "line_clv": lc,
        "price_clv": pc,
        "beat_close": (net > 0) if have else None,
    }
