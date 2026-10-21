"""SYNTHETIC harness tolerance; functional deferred-history checks remain strict."""
from threading import Event, Thread
import time
import pytest
from streamlit.testing.v1 import AppTest
from app.ui import public_results, publish_panel
import test_drive_history_loading as original
from test_drive_history_loading import offline


def test_only_completed_analysis_test_has_explicit_ten_second_budget(monkeypatch):
    native = AppTest.from_string
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs.get('default_timeout', 3))
        return native(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(AppTest, 'from_string', capture)
        original.test_actual_completed_analysis_publish_caller_defers_history(patch)
    assert seen == [10]
    assert AppTest.from_string == native
    assert AppTest.from_string('pass').default_timeout == 3


def test_bounded_non_history_token_render_delay_preserves_actual_assertions(monkeypatch):
    native = publish_panel.render_publish_panel
    entered, release, cancel = Event(), Event(), Event()
    calls, elapsed = [], []
    def release_after_delay():
        if entered.wait(10) and not cancel.wait(3.2):
            release.set()
    worker = Thread(target=release_after_delay, daemon=True)
    def render(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            start = time.perf_counter(); entered.set()
            assert release.wait(6), 'SYNTHETIC render gate was not released'
            elapsed.append(time.perf_counter()-start)
        return native(*args, **kwargs)
    worker.start()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(publish_panel, 'render_publish_panel', render)
            original.test_actual_completed_analysis_publish_caller_defers_history(patch)
        assert len(elapsed) == 1 and 3 <= elapsed[0] < 6
    finally:
        cancel.set(); entered.set(); release.set(); worker.join(10)
    assert not worker.is_alive()
    assert publish_panel.render_publish_panel is native


def test_eager_history_is_rejected_for_behavior_not_timeout(monkeypatch):
    seen = []
    native = public_results.render_history
    def eager(setting, **kwargs):
        # The original test installs its actual forbidden restore function.
        try:
            return public_results.restore_history(setting)
        except AssertionError as exc:
            seen.append(str(exc))
            raise
    with monkeypatch.context() as patch:
        patch.setattr(public_results, 'render_history', eager)
        with pytest.raises((AssertionError, KeyError)) as caught:
            original.test_actual_completed_analysis_publish_caller_defers_history(patch)
    assert seen and set(seen) == {'Display-only rerun must not restore'}
    assert 'timed out' not in str(caught.value).lower()
    assert public_results.render_history is native
