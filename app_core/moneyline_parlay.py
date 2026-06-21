"""Moneyline parlay-leg eligibility (v1, gated by ENABLE_MONEYLINE_PARLAY_LEGS).

Moneyline is the model's native output: ``ml_probability`` already IS P(team wins), so a
moneyline leg needs no margin/cover conversion (unlike the run line). These helpers decide
whether a moneyline pick is a usable PARLAY leg: the price must sit in a sane odds range
(heavy favorites carry little value and poor parlay equity; longshots add variance) and the
model must show a real edge over the price implied probability. Pure + unit-tested; the
pipeline wiring that generates the candidate rows is gated behind the config flag.
"""
from __future__ import annotations

import pandas as pd


def american_to_implied(odds: float) -> float:
    """De-vig-free single-side implied probability of an American price."""
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / (-odds))


def moneyline_leg_eligible(
    model_win_prob: float,
    american_odds: float,
    min_edge: float | None = None,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> dict:
    """Return {eligible, edge, ev, implied, reason} for a moneyline parlay leg.

    edge = model_win_prob - implied(price). A leg is eligible when the price is inside
    [min_odds, max_odds] AND edge >= min_edge AND EV at the offered price is positive.
    Thresholds default to the weights_config values.
    """
    from app_core.weights_config import (
        MONEYLINE_PARLAY_MIN_ODDS,
        MONEYLINE_PARLAY_MAX_ODDS,
        MONEYLINE_PARLAY_MIN_EDGE,
    )

    min_edge = MONEYLINE_PARLAY_MIN_EDGE if min_edge is None else min_edge
    min_odds = MONEYLINE_PARLAY_MIN_ODDS if min_odds is None else min_odds
    max_odds = MONEYLINE_PARLAY_MAX_ODDS if max_odds is None else max_odds

    if model_win_prob is None or pd.isna(model_win_prob) or american_odds is None or pd.isna(american_odds):
        return {"eligible": False, "edge": None, "ev": None, "implied": None, "reason": "missing prob/odds"}

    p = float(model_win_prob)
    odds = float(american_odds)
    implied = american_to_implied(odds)
    edge = p - implied
    ev = p * (american_to_decimal(odds) - 1.0) - (1.0 - p)

    if not (min_odds <= odds <= max_odds):
        return {"eligible": False, "edge": round(edge, 4), "ev": round(ev, 4),
                "implied": round(implied, 4), "reason": f"odds {odds:g} outside [{min_odds:g},{max_odds:g}]"}
    if edge < min_edge:
        return {"eligible": False, "edge": round(edge, 4), "ev": round(ev, 4),
                "implied": round(implied, 4), "reason": f"edge {edge:+.1%} below {min_edge:.0%} bar"}
    if ev <= 0:
        return {"eligible": False, "edge": round(edge, 4), "ev": round(ev, 4),
                "implied": round(implied, 4), "reason": "non-positive EV"}
    return {"eligible": True, "edge": round(edge, 4), "ev": round(ev, 4),
            "implied": round(implied, 4), "reason": f"edge {edge:+.1%}, EV {ev:+.1%}"}
