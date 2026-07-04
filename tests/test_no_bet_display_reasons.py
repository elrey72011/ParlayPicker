"""Display-side cleanup (owner request, 4 Jul): non-Actionable picks collapse
into one "No Edge" group with a plain-English reason. This tests the reason
mapper — the raw statuses in exports/grading are untouched by the cleanup."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit_app import _friendly_no_bet_reason


def test_known_stages_map_to_plain_english():
    cases = {
        "baseline_guardrail": "negative EV",
        "game_already_started": "already started",
        "min_win_probability_floor": "55%",
        "empirical_proven_losing_bucket": "Proven-losing",
        "divergence_viability_floor": "viability floor",
    }
    for stage, fragment in cases.items():
        reason = _friendly_no_bet_reason({"status_blocker_stage": stage})
        assert fragment.lower() in reason.lower(), (stage, reason)


def test_unknown_stage_falls_back_to_status_reason():
    row = pd.Series({
        "status_blocker_stage": "some_new_stage",
        "Status_Reason": "Fails stricter total_under cold-market penalty",
    })
    assert _friendly_no_bet_reason(row) == "Fails stricter total_under cold-market penalty"


def test_empty_row_gets_generic_reason():
    assert _friendly_no_bet_reason({}) == "Did not qualify"
    assert _friendly_no_bet_reason({"status_blocker_stage": "", "Status_Reason": "nan"}) == "Did not qualify"
