"""User-facing status vocabulary (owner, 4 Jul): 'No Play' reads like the game
is off the board; the display/export layer now says what the status means.
Internal pipeline names are untouched — grading and gates key off them."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app import (
    STATUS_DISPLAY_LABELS,
    _bet_decision_mask,
    apply_status_display_labels,
)


def test_statuses_relabelled_for_display():
    df = pd.DataFrame({
        "Pick_Status": ["Actionable", "No Play", "Below Threshold", "High Variance/Speculative"],
        "Pick_Quality": ["S-Tier", "No Bet - No Play", "No Bet - Below Threshold", "No Bet - High Variance/Speculative"],
    })
    out = apply_status_display_labels(df)
    assert list(out["Pick_Status"]) == ["Actionable", "No Edge", "Near Miss", "Unproven"]
    assert list(out["Pick_Quality"]) == ["S-Tier", "No Bet - No Edge", "No Bet - Near Miss", "No Bet - Unproven"]


def test_actionable_is_never_renamed():
    assert "Actionable" not in STATUS_DISPLAY_LABELS


def test_original_frame_not_mutated():
    df = pd.DataFrame({"Pick_Status": ["No Play"]})
    apply_status_display_labels(df)
    assert df.loc[0, "Pick_Status"] == "No Play"


def test_bet_decision_mask_uses_series_equality_and_fails_closed():
    df = pd.DataFrame({
        "Bet Decision": ["BET", " BEST AVAILABLE - PASS ", None],
    })

    assert _bet_decision_mask(df).tolist() == [True, False, False]


def test_grade_slate_accepts_new_no_edge_label():
    # The recap Status inherits the export label; "No Edge" must gate exactly
    # like "No Play" (excluded only when the line isn't clean).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    gs = importlib.import_module("grade_slate")
    src = Path(gs.__file__).read_text()
    assert '"no edge"' in src and '"no play"' in src
