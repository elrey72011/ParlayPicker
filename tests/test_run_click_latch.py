import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import _consume_run_click


def test_consume_run_click_edge_trigger_prevents_repeat_when_sticky_true():
    state = {}

    assert _consume_run_click(state, False) is False
    assert _consume_run_click(state, True) is True
    assert _consume_run_click(state, True) is False
    assert _consume_run_click(state, True) is False

    assert _consume_run_click(state, False) is False
    assert _consume_run_click(state, True) is True
