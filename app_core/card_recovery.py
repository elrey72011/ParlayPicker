"""Empty-card-recovery emptiness check (status-based).

The pipeline-level empty-card recovery in ``core/streamlit_pipeline.py`` runs
inside ``build_best_picks_df`` — which executes *before* Kelly is sized (Kelly is
attached downstream in ``streamlit_app._attach_kelly_to_best_picks``). The
original gate keyed on ``production_eligible`` (``Actionable AND Kelly_Bet_Size > 0``),
which is structurally always-False at that stage, so the "is the card empty?"
test never actually gated on emptiness: recovery could backfill a redundant pick
even when a legitimate ``Actionable`` pick (e.g. a sub-8.0 over carve-out) was
already on the card.

This module centralises the correct, status-based emptiness definition — the
same one the downstream ``streamlit_app`` recovery already uses — so both code
paths agree and the decision is unit-testable in isolation.
"""
from __future__ import annotations

import pandas as pd


def actionable_card_is_empty(pick_status) -> bool:
    """Return True when no pick is ``Actionable`` by status.

    Emptiness is status-based, NOT Kelly-based, because Kelly is sized after
    ``build_best_picks_df`` returns. Accepts either a DataFrame (uses its
    ``Pick_Status`` column) or a Series/iterable of status strings.
    """
    if isinstance(pick_status, pd.DataFrame):
        if "Pick_Status" not in pick_status.columns:
            return True
        statuses = pick_status["Pick_Status"]
    else:
        statuses = pick_status

    if not isinstance(statuses, pd.Series):
        statuses = pd.Series(list(statuses), dtype="object")

    if statuses.empty:
        return True

    return int(statuses.astype(str).str.strip().eq("Actionable").sum()) == 0


def controlled_value_price_gate(
    probability,
    american_odds,
    consensus,
    *,
    min_absolute_edge: float,
    disagrees_min_absolute_edge: float,
    min_american_odds: float,
    max_american_odds: float,
) -> pd.DataFrame:
    """Evaluate the exact price-aware gate for a controlled value card.

    Unlike a fixed win-rate floor, this permits a plus-money outcome below 50%
    only when its calibrated probability clears that exact price's break-even
    probability. Contrarian rows require the larger ``Disagrees`` margin.
    """
    if isinstance(probability, pd.Series):
        index = probability.index
    elif isinstance(american_odds, pd.Series):
        index = american_odds.index
    elif isinstance(consensus, pd.Series):
        index = consensus.index
    else:
        index = pd.RangeIndex(1)

    prob = pd.to_numeric(pd.Series(probability, index=index), errors="coerce")
    odds = pd.to_numeric(pd.Series(american_odds, index=index), errors="coerce")
    consensus_text = pd.Series(consensus, index=index).astype("string").fillna("").str.strip()

    break_even = pd.Series(float("nan"), index=index, dtype=float)
    negative = odds.lt(0)
    positive = odds.gt(0)
    break_even.loc[negative] = odds.loc[negative].abs() / (
        odds.loc[negative].abs() + 100.0
    )
    break_even.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)

    absolute_edge = prob - break_even
    required_edge = pd.Series(float(min_absolute_edge), index=index, dtype=float)
    required_edge.loc[consensus_text.eq("Disagrees")] = float(
        disagrees_min_absolute_edge
    )
    price_allowed = odds.between(
        float(min_american_odds), float(max_american_odds), inclusive="both"
    )
    passed = (
        prob.between(0.0, 1.0, inclusive="both")
        & break_even.notna()
        & price_allowed
        & absolute_edge.ge(required_edge)
    )
    return pd.DataFrame(
        {
            "controlled_value_price_gate_pass": passed.fillna(False).astype(bool),
            "controlled_value_break_even_probability": break_even,
            "controlled_value_absolute_edge": absolute_edge,
            "controlled_value_required_edge": required_edge,
            "controlled_value_price_allowed": price_allowed.fillna(False).astype(bool),
        },
        index=index,
    )
