"""Tests for the empty-card-recovery emptiness check.

Locks the status-based definition that fixes the structural bug where the
pipeline recovery gated on `production_eligible` (Actionable AND Kelly>0) — which
is always-False inside build_best_picks_df because Kelly is sized downstream — and
so backfilled a redundant pick even when an Actionable pick was already present.
"""
import pandas as pd

from app_core.card_recovery import (
    actionable_card_is_empty,
    controlled_value_price_gate,
)


def _df(statuses):
    return pd.DataFrame({"Pick_Status": statuses})


def test_empty_when_no_actionable():
    assert actionable_card_is_empty(_df(["High Variance/Speculative", "Below Threshold"])) is True


def test_not_empty_with_one_actionable():
    assert actionable_card_is_empty(_df(["Actionable", "High Variance/Speculative"])) is False


def test_carveout_actionable_suppresses_recovery():
    # The regression case: a single Actionable pick (e.g. a sub-8.0 over carve-out)
    # surrounded by Below Threshold picks must NOT look like an empty card, so the
    # redundant Dodgers-style backfill is suppressed.
    assert actionable_card_is_empty(_df(["Actionable", "Below Threshold", "Below Threshold"])) is False


def test_no_play_and_below_threshold_are_empty():
    assert actionable_card_is_empty(_df(["No Play", "Below Threshold", "High Variance/Speculative"])) is True


def test_accepts_series_and_strips_whitespace():
    assert actionable_card_is_empty(pd.Series([" Actionable ", "No Play"])) is False
    assert actionable_card_is_empty(pd.Series(["No Play", "Below Threshold"])) is True


def test_accepts_plain_iterable():
    assert actionable_card_is_empty(["Actionable"]) is False
    assert actionable_card_is_empty(["High Variance/Speculative"]) is True


def test_missing_column_is_treated_as_empty():
    assert actionable_card_is_empty(pd.DataFrame({"other": [1, 2]})) is True


def test_empty_frame_is_empty():
    assert actionable_card_is_empty(_df([])) is True


def test_aug5_plus_money_value_is_price_aware_and_contrarian_bar_is_strict():
    # Exact empirical probabilities/prices from the 5 Aug slate. Cincinnati and
    # New York clear the 3-point Disagrees margin; Milwaukee and the WNBA under
    # clear 2 points but not the stricter contrarian bar. Arizona is below price.
    gate = controlled_value_price_gate(
        pd.Series([0.4622642416, 0.4485899256, 0.3839439630, 0.5092094336, 0.5408813205]),
        pd.Series([138, 141, 178, 106, -141]),
        pd.Series(["Disagrees", "Disagrees", "Disagrees", "Disagrees", "Agrees"]),
        min_absolute_edge=0.02,
        disagrees_min_absolute_edge=0.03,
        min_american_odds=-200,
        max_american_odds=200,
    )

    assert gate["controlled_value_price_gate_pass"].tolist() == [
        True, True, False, False, False
    ]
    assert gate.loc[0, "controlled_value_break_even_probability"] < 0.43
    assert gate.loc[0, "controlled_value_absolute_edge"] > 0.04


def test_controlled_value_price_gate_rejects_unsafe_juice():
    gate = controlled_value_price_gate(
        pd.Series([0.75, 0.35]),
        pd.Series([-250, 250]),
        pd.Series(["Agrees", "Neutral"]),
        min_absolute_edge=0.02,
        disagrees_min_absolute_edge=0.03,
        min_american_odds=-200,
        max_american_odds=200,
    )

    assert not gate["controlled_value_price_allowed"].any()
    assert not gate["controlled_value_price_gate_pass"].any()
