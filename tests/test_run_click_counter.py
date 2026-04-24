import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import _should_run_pipeline


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
