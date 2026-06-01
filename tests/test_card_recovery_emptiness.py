"""Tests for the empty-card-recovery emptiness check.

Locks the status-based definition that fixes the structural bug where the
pipeline recovery gated on `production_eligible` (Actionable AND Kelly>0) — which
is always-False inside build_best_picks_df because Kelly is sized downstream — and
so backfilled a redundant pick even when an Actionable pick was already present.
"""
import pandas as pd

from app_core.card_recovery import actionable_card_is_empty


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
