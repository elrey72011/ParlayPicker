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
