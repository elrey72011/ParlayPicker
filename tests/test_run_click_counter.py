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
