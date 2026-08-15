"""Accuracy-first shortlist for the daily game card.

The full-board export still carries one best-available pick per game, while the
precision shortlist intentionally trades coverage for a small set of the most
likely winners.  It never grants wagering authority or creates a stake; the
existing production gate remains the only source of ``Bettable=True``.
"""
from __future__ import annotations

import pandas as pd


PRECISION_CARD_MAX_PICKS = 2
PRECISION_CARD_MIN_WIN_PROBABILITY = 0.60
PRECISION_CARD_MIN_AMERICAN_ODDS = -220
PRECISION_CARD_TARGET_HIT_RATE = 0.75


def _strict_bool(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def _probability(frame: pd.DataFrame) -> pd.Series:
    probability = pd.to_numeric(
        frame.get("WinProbability", pd.Series(float("nan"), index=frame.index)),
        errors="coerce",
    )
    for fallback in (
        "effective_win_probability",
        "empirical_win_probability",
        "selection_probability_used",
    ):
        if fallback in frame.columns:
            probability = probability.fillna(
                pd.to_numeric(frame[fallback], errors="coerce")
            )
    return probability


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def attach_precision_card(
    frame: pd.DataFrame | None,
    *,
    max_picks: int = PRECISION_CARD_MAX_PICKS,
    min_win_probability: float = PRECISION_CARD_MIN_WIN_PROBABILITY,
    min_american_odds: int = PRECISION_CARD_MIN_AMERICAN_ODDS,
) -> pd.DataFrame | None:
    """Annotate a best-picks slate with a fail-closed top-confidence shortlist.

    Ranking is global across the slate, not tier-first.  A row must have a
    verified final selection, verified live event/line identity, at least the
    minimum calibrated win probability, and a price no shorter than the policy
    floor.  Selection is informational unless the ordinary production gate has
    independently approved a positive stake.
    """

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    probability = _probability(out)
    odds = pd.to_numeric(
        out.get("odds_american", pd.Series(float("nan"), index=out.index)),
        errors="coerce",
    )
    line_source = out.get(
        "market_line_source", pd.Series("", index=out.index, dtype="object")
    ).astype("string").fillna("").str.strip().str.casefold()

    verified = (
        _strict_bool(out, "final_pick_valid")
        & _strict_bool(out, "best_available_selection_verified")
        & _strict_bool(out, "best_available_ranking_verified")
        & _strict_bool(out, "line_consistency_flag")
        & _strict_bool(out, "line_event_identity_match_flag")
        & line_source.eq("live")
    )
    started = pd.Series(False, index=out.index, dtype=bool)
    for column in ("Started", "started", "is_started"):
        if column in out.columns:
            started |= _strict_bool(out, column)
    # The public best-picks export intentionally omits the internal ``Started``
    # boolean, but preserves the closure as ``Bet_Decision=STARTED`` and/or
    # ``Play_Tier=STARTED``.  Read both representations so a shortlist built
    # from the export cannot resurrect an already-started game.
    for column in ("Bet_Decision", "Play_Tier", "Tier"):
        if column in out.columns:
            status = (
                out[column]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.casefold()
            )
            started |= status.eq("started")

    probability_ok = probability.ge(float(min_win_probability))
    price_ok = odds.notna() & odds.ne(0) & odds.ge(float(min_american_odds))
    eligible = verified & ~started & probability_ok & price_ok

    rank = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if eligible.any():
        ranking = pd.DataFrame(
            {
                "probability": probability.loc[eligible],
                "best_available_score": _numeric_series(
                    out, "best_available_score"
                ).loc[eligible],
                "expected_value": _numeric_series(
                    out, "expected_value"
                ).loc[eligible],
                "original_order": range(int(eligible.sum())),
            },
            index=out.index[eligible],
        ).sort_values(
            ["probability", "best_available_score", "expected_value", "original_order"],
            ascending=[False, False, False, True],
            na_position="last",
            kind="mergesort",
        )
        rank.loc[ranking.index] = pd.Series(
            range(1, len(ranking) + 1), index=ranking.index, dtype="Int64"
        )

    selected = eligible & rank.le(max(0, int(max_picks))).fillna(False)
    bettable = _strict_bool(out, "Bettable")
    stake = pd.to_numeric(
        out.get("Play_Stake", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    approved = selected & bettable & stake.gt(0.0)

    out["Precision_Card"] = selected
    out["Precision_Rank"] = rank
    out["Precision_Probability"] = probability
    out["Precision_Target_Hit_Rate"] = float(PRECISION_CARD_TARGET_HIT_RATE)
    out["Precision_Wager_Approved"] = approved
    out["Precision_Card_Instruction"] = "NOT ON PRECISION SHORTLIST"
    out.loc[selected, "Precision_Card_Instruction"] = (
        "PRECISION SHORTLIST - NO APP-APPROVED STAKE"
    )
    out.loc[approved, "Precision_Card_Instruction"] = "BET - APP APPROVED"

    reason = pd.Series("Outside the top-confidence slots.", index=out.index, dtype="object")
    reason.loc[~verified] = "Excluded: final selection or live line identity is not verified."
    reason.loc[started] = "Excluded: game has started."
    reason.loc[verified & ~started & ~probability_ok] = (
        f"Excluded: calibrated win probability is below {float(min_win_probability):.0%}."
    )
    reason.loc[verified & ~started & probability_ok & ~price_ok] = (
        f"Excluded: offered price is shorter than {int(min_american_odds):+d}."
    )
    reason.loc[selected] = (
        "Selected by global calibrated win probability; 75% is a monitoring target, not a guarantee."
    )
    out["Precision_Card_Reason"] = reason
    return out


def precision_shortlist(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return only precision-selected rows, ordered by precision rank."""

    annotated = attach_precision_card(frame)
    if annotated is None:
        return pd.DataFrame()
    if annotated.empty:
        return annotated.copy()
    selected = _strict_bool(annotated, "Precision_Card")
    return annotated.loc[selected].sort_values("Precision_Rank").reset_index(drop=True)
