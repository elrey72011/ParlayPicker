"""Absolute value gate for production sports-betting recommendations.

Candidate ranking answers "which available pick is best for this game?"  This
module answers the separate question "is that pick good enough to bet at the
offered price?"  Keeping the questions separate lets the UI show a directional
read for every game without turning a relative winner into a funded wager.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.price_value import price_value


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


def _boolean_series(
    value: object, index: pd.Index, *, default: bool
) -> pd.Series:
    if value is None:
        return pd.Series(default, index=index, dtype=bool)
    if isinstance(value, pd.Series):
        values = value.reindex(index)
    elif np.isscalar(value):
        values = pd.Series(value, index=index)
    else:
        values = pd.Series(value, index=index)
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "1.0", "yes", "y"})


def evaluate_absolute_production_gate(
    calibrated_probability: object,
    break_even_probability: object = None,
    model_expected_value: object = None,
    *,
    push_probability: object = None,
    decimal_odds: object = None,
    conservative_probability: object = None,
    explicit_contract_mask: object = None,
    conservative_bound_mask: object = None,
    min_edge: float = MIN_PRODUCTION_CALIBRATED_EDGE,
    min_model_ev: float = MIN_PRODUCTION_MODEL_EV,
) -> pd.DataFrame:
    """Return one index-aligned, push-aware production price decision.

    ``calibrated_probability`` is unconditional win mass whenever
    ``push_probability`` and ``decimal_odds`` are supplied.  In that mode all
    displayed values come from :func:`core.price_value.price_value`; a supplied
    break-even must agree with the same quote.  The legacy three-positional-
    argument form remains available for no-push callers and is explicitly
    labelled in the result.

    ``model_expected_value`` remains a separate upstream diagnostic gate.  It
    cannot replace the final mean EV calculated from the final probability
    mass and exact quote.
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
    model_ev = _numeric_series(model_expected_value, index)
    explicit_price_contract = _boolean_series(
        explicit_contract_mask,
        index,
        default=push_probability is not None or decimal_odds is not None,
    )
    supplied_break_even = _numeric_series(break_even_probability, index)
    supplied_push = _numeric_series(push_probability, index)
    supplied_decimal = _numeric_series(decimal_odds, index)
    legacy_decimal = (1.0 / supplied_break_even).replace(
        [np.inf, -np.inf], np.nan
    )
    push = supplied_push.where(explicit_price_contract, 0.0)
    decimal = supplied_decimal.where(explicit_price_contract, legacy_decimal)

    rows: list[dict[str, float] | None] = [
        price_value(win, push_mass, price, minimum_edge=float(min_edge))
        for win, push_mass, price in zip(probability, push, decimal)
    ]
    priced = pd.DataFrame(
        [
            item
            if item is not None
            else {
                "p_win": np.nan,
                "p_push": np.nan,
                "p_loss": np.nan,
                "break_even": np.nan,
                "expected_value": np.nan,
                "edge": np.nan,
                "minimum_decimal_price": np.nan,
                "full_kelly": np.nan,
            }
            for item in rows
        ],
        index=index,
    )
    break_even = pd.to_numeric(priced["break_even"], errors="coerce")
    absolute_edge = pd.to_numeric(priced["edge"], errors="coerce")
    calibrated_ev = pd.to_numeric(priced["expected_value"], errors="coerce")
    price_valid = pd.Series([item is not None for item in rows], index=index, dtype=bool)
    break_even_matches = pd.Series(True, index=index, dtype=bool)
    if break_even_probability is not None:
        explicit_comparison = explicit_price_contract & supplied_break_even.notna()
        break_even_matches.loc[explicit_comparison] = np.isclose(
            supplied_break_even.loc[explicit_comparison],
            break_even.loc[explicit_comparison],
            atol=1e-10,
            rtol=1e-10,
        )

    conservative = _numeric_series(conservative_probability, index)
    conservative_supplied = _boolean_series(
        conservative_bound_mask,
        index,
        default=conservative_probability is not None,
    )
    conservative_rows = [
        price_value(win, push_mass, price, minimum_edge=float(min_edge))
        if bound_supplied
        else None
        for win, push_mass, price, bound_supplied in zip(
            conservative, push, decimal, conservative_supplied
        )
    ]
    conservative_ev = pd.Series(
        [
            item["expected_value"] if item is not None else np.nan
            for item in conservative_rows
        ],
        index=index,
        dtype=float,
    )
    conservative_valid = pd.Series(True, index=index, dtype=bool)
    supplied_bound_valid = pd.Series(
        [item is not None for item in conservative_rows], index=index, dtype=bool
    ) & conservative.le(probability)
    conservative_valid.loc[conservative_supplied] = supplied_bound_valid.loc[
        conservative_supplied
    ]

    model_valid = model_ev.notna() & np.isfinite(model_ev)
    valid = price_valid & break_even_matches & conservative_valid & model_valid
    passed = (
        valid
        & model_ev.gt(float(min_model_ev))
        & calibrated_ev.gt(0.0)
        & absolute_edge.ge(float(min_edge))
    )

    reason = pd.Series("qualified", index=index, dtype="object")
    reason.loc[probability.isna()] = "missing calibrated probability"
    reason.loc[probability.notna() & ~probability.between(0.0, 1.0, inclusive="both")] = (
        "invalid calibrated probability"
    )
    legacy_price_contract = ~explicit_price_contract
    reason.loc[explicit_price_contract & push.isna()] = "missing push probability"
    reason.loc[
        explicit_price_contract
        & push.notna()
        & ~(push.ge(0.0) & push.lt(1.0))
    ] = "invalid push probability"
    reason.loc[explicit_price_contract & decimal.isna()] = "missing decimal odds"
    reason.loc[
        explicit_price_contract & decimal.notna() & ~decimal.gt(1.0)
    ] = "invalid decimal odds"
    reason.loc[
        explicit_price_contract
        & probability.notna()
        & push.notna()
        & probability.add(push).gt(1.0 + 1e-10)
    ] = "invalid unconditional probability mass"
    reason.loc[
        explicit_price_contract & price_valid & ~break_even_matches
    ] = "supplied break-even does not match exact quote and push mass"
    reason.loc[legacy_price_contract & supplied_break_even.isna()] = (
        "missing sportsbook break-even price"
    )
    reason.loc[
        legacy_price_contract
        & supplied_break_even.notna()
        & ~(supplied_break_even.gt(0.0) & supplied_break_even.lt(1.0))
    ] = "invalid sportsbook break-even price"
    reason.loc[
        conservative_supplied
        & conservative.notna()
        & conservative.gt(probability)
    ] = "conservative win probability exceeds mean win probability"
    reason.loc[
        conservative_supplied
        & ~conservative_valid
        & conservative.le(probability)
    ] = "invalid conservative probability mass"
    reason.loc[~np.isfinite(model_ev)] = "missing or invalid model EV"
    reason.loc[valid & ~model_ev.gt(float(min_model_ev))] = "model EV is not positive"
    reason.loc[
        valid & model_ev.gt(float(min_model_ev)) & ~calibrated_ev.gt(0.0)
    ] = "final priced EV is not positive"
    thin_edge = (
        valid
        & model_ev.gt(float(min_model_ev))
        & calibrated_ev.gt(0.0)
        & ~absolute_edge.ge(float(min_edge))
    )
    reason.loc[thin_edge] = absolute_edge.loc[thin_edge].map(
        lambda edge: (
            f"calibrated edge below {float(min_edge):.1%} safety margin "
            f"({edge:+.1%})"
        )
    )

    return pd.DataFrame(
        {
            "production_gate_pass": passed.fillna(False).astype(bool),
            "final_p_win": priced["p_win"],
            "final_p_push": priced["p_push"],
            "final_p_loss": priced["p_loss"],
            "sportsbook_break_even_probability": break_even,
            "absolute_production_edge": absolute_edge,
            "calibrated_expected_value": calibrated_ev,
            "mean_expected_value_per_unit": calibrated_ev,
            "conservative_expected_value_per_unit": conservative_ev,
            "minimum_acceptable_decimal_odds": priced["minimum_decimal_price"],
            "pricing_contract_status": pd.Series(
                np.where(
                    explicit_price_contract,
                    "PUSH_AWARE_VERIFIED",
                    "LEGACY_NO_PUSH_COMPATIBILITY",
                ),
                index=index,
            ).where(valid, "INVALID"),
            "production_gate_reason": reason,
        },
        index=index,
    )

