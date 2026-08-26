import os
import sys
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from streamlit_app import (
    _analysis_input_signature,
    _analysis_inputs_stale,
    _should_run_pipeline,
)


class _Upload(BytesIO):
    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name
        self.size = len(content)


def test_should_run_pipeline_triggers_once_per_counter_increment():
    state = {}

    assert _should_run_pipeline(state, 0) is False
    assert _should_run_pipeline(state, 1) is True
    assert _should_run_pipeline(state, 1) is False
    assert _should_run_pipeline(state, 1) is False

    assert _should_run_pipeline(state, 2) is True
    assert _should_run_pipeline(state, 2) is False


def test_should_run_pipeline_blocks_duplicate_signature_replays():
    state = {}
    controls = {
        "sports": ["MLB", "NHL"],
        "use_ml": True,
        "use_gemini": False,
        "bankroll": 1000,
        "theover_spreads": object(),
        "theover_totals": None,
    }
    assert _should_run_pipeline(state, 1, controls) is True
    state["last_processed_run_counter"] = 0
    assert _should_run_pipeline(state, 1, controls) is False


def test_analysis_signature_detects_replaced_theover_file_contents():
    controls = {
        "sports": ["MLB"],
        "use_ml": True,
        "use_gemini": False,
        "bankroll": 1000,
        "theover_spreads": _Upload("sides.csv", b"League,HomeTeam\nMLB,Atlanta\n"),
        "theover_totals": None,
    }
    before = _analysis_input_signature(controls)
    same_content = dict(controls)
    same_content["theover_spreads"] = _Upload(
        "sides.csv", b"League,HomeTeam\nMLB,Atlanta\n"
    )
    assert _analysis_input_signature(same_content) == before

    controls["theover_spreads"] = _Upload(
        "sides.csv", b"League,HomeTeam\nMLB,Arizona\n"
    )

    assert _analysis_input_signature(controls) != before


def test_changed_theover_upload_marks_existing_results_stale():
    controls = {
        "sports": ["MLB"],
        "use_ml": True,
        "use_gemini": False,
        "bankroll": 1000,
        "theover_spreads": _Upload("sides.csv", b"old"),
        "theover_totals": None,
    }
    state = {
        "analysis_df": pd.DataFrame([{"league": "MLB"}]),
        "last_successful_pipeline_signature": _analysis_input_signature(controls),
    }
    assert _analysis_inputs_stale(state, controls) is False

    controls["theover_spreads"] = _Upload("sides.csv", b"new")
    assert _analysis_inputs_stale(state, controls) is True
