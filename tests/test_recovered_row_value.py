"""Recovered-line value neutralization (21 Jun).

A row whose live total was rejected and recovered to the uploaded reference keeps the
REJECTED live odds, so its EV/edge are a line/odds mismatch -- a fake number. Those rows
must show zero EV and sort to the bottom of their tier, never headline the card.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import _neutralize_recovered_row_value

_REC = "upload_total_fallback_after_rejected_live"


def _df():
    return pd.DataFrame([
        # A recovered HV row with a fake +56% EV and the best rank -> must drop + zero out.
        {"Pick_Status": "High Variance/Speculative", "market_line_source_detail": _REC,
         "Triple_Filter_Rank": 1, "parlay_rank": 1, "best_pick": "Houston Over 8.5",
         "expected_value": 0.56, "edge": 0.12, "effective_expected_value": 0.56,
         "effective_edge": 0.12, "production_expected_value": 0.0, "production_edge": 0.0},
        # A clean HV row with a real, smaller edge -> should rank ahead of the recovered one.
        {"Pick_Status": "High Variance/Speculative", "market_line_source_detail": "",
         "Triple_Filter_Rank": 2, "parlay_rank": 2, "best_pick": "Clean HV",
         "expected_value": 0.06, "edge": 0.04, "effective_expected_value": 0.06,
         "effective_edge": 0.04, "production_expected_value": 0.0, "production_edge": 0.0},
    ])


def test_recovered_row_ev_is_zeroed():
    df = _df()
    df["best_available_score"] = [1.75, 0.50]
    df["best_available_runner_up_score"] = [0.50, 0.25]
    df["best_available_score_gap"] = [1.25, 0.25]
    df["best_available_selection_reason"] = ["old score", "clean score"]
    out = _neutralize_recovered_row_value(df)
    rec = out[out["best_pick"] == "Houston Over 8.5"].iloc[0]
    assert float(rec["expected_value"]) == 0.0
    assert float(rec["edge"]) == 0.0
    assert float(rec["effective_expected_value"]) == 0.0
    assert float(rec["best_available_score"]) == 0.0
    assert pd.isna(rec["best_available_runner_up_score"])
    assert pd.isna(rec["best_available_score_gap"])
    assert "Research-only upload line fallback" in rec["best_available_selection_reason"]


def test_recovered_row_sorts_below_clean_row_in_same_tier():
    out = _neutralize_recovered_row_value(_df())
    # Clean HV row first, recovered row last; parlay_rank renumbered 1..N.
    assert out.iloc[0]["best_pick"] == "Clean HV"
    assert out.iloc[1]["best_pick"] == "Houston Over 8.5"
    assert list(out["parlay_rank"]) == [1, 2]


def test_clean_row_value_untouched():
    out = _neutralize_recovered_row_value(_df())
    clean = out[out["best_pick"] == "Clean HV"].iloc[0]
    assert float(clean["expected_value"]) == 0.06


def test_no_recovered_rows_is_noop():
    df = _df().copy()
    df["market_line_source_detail"] = ""
    out = _neutralize_recovered_row_value(df)
    # Untouched EV when nothing is recovered.
    assert float(out[out["best_pick"] == "Houston Over 8.5"].iloc[0]["expected_value"]) == 0.56


def test_recovered_row_status_unchanged():
    out = _neutralize_recovered_row_value(_df())
    rec = out[out["best_pick"] == "Houston Over 8.5"].iloc[0]
    assert rec["Pick_Status"] == "High Variance/Speculative"
