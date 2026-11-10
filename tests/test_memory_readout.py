"""Memory readout (owner 2026-11-10): where the app's memory grew. No data is read; numbers are SYNTHETIC."""
from datetime import datetime, timezone
import logging

import pytest
from streamlit.testing.v1 import AppTest

from app_core import memory_readout as mr


@pytest.fixture(autouse=True)
def _clean():
    mr.clear()
    yield
    mr.clear()


def test_each_stage_notes_memory_its_rise_and_the_peak_and_logs_one_line(caplog):
    values = iter([900.4, 1500.2, 1400.0])
    clock = lambda: datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="app_core.memory_readout"):
        first = mr.record("Load inputs", read=lambda: next(values), peak=lambda: 950.0, clock=clock)
        second = mr.record("Fetch odds", read=lambda: next(values), peak=lambda: 1600.0, clock=clock)
        third = mr.record("Save evidence", read=lambda: next(values), peak=lambda: 1600.0, clock=clock)
    assert first == {"at": "2026-11-10T15:00:00+00:00", "stage": "Load inputs", "rss_mb": 900, "rose_mb": None,
                     "peak_mb": 950}
    assert (second["rose_mb"], third["rose_mb"]) == (600, -100)
    assert "MEMORY stage=Fetch odds rss_mb=1500 rose_mb=600 peak_mb=1600" in caplog.text
    found = mr.readout(read=lambda: 2300.0, peak=lambda: 2400.0)
    assert found["rss_mb"] == 2300 and found["peak_mb"] == 2400 and found["limit_mb"] == 2700
    assert found["near_limit"] is True and found["share_of_limit"] == pytest.approx(2300 / 2700)
    assert found["largest_rise"]["stage"] == "Fetch odds" and [n["stage"] for n in found["notes"]] == [
        "Load inputs", "Fetch odds", "Save evidence"]
    assert mr.readout(read=lambda: 800.0, peak=lambda: None)["near_limit"] is False


def test_a_readout_never_breaks_its_caller_and_the_history_is_bounded():
    def broken():
        raise RuntimeError("no /proc")
    assert mr.record("stage", read=broken) is None and mr.record("stage", read=lambda: None) is None
    assert mr.readout(read=lambda: None, peak=lambda: None)["rss_mb"] is None
    for i in range(mr.HISTORY + 15):
        mr.record(f"stage {i}", read=lambda: 100.0, peak=lambda: 100.0)
    notes = mr.readout(read=lambda: 100.0, peak=lambda: 100.0)["notes"]
    assert len(notes) == mr.HISTORY and notes[-1]["stage"] == f"stage {mr.HISTORY + 14}"
    assert mr.rss_mb() is None or mr.rss_mb() > 0                       # the real reading, where available


def test_the_stage_timer_notes_memory_at_the_end_of_every_stage():
    from app_core.stage_timing import StageTimer
    timer = StageTimer()
    timer.start("Load inputs and initialize evidence")
    timer.start("Fetch odds, team statistics and model predictions")
    timer.finish()
    notes = mr.readout()["notes"]
    if mr.rss_mb() is None:
        assert notes == []
    else:
        assert [n["stage"] for n in notes] == ["Load inputs and initialize evidence",
                                               "Fetch odds, team statistics and model predictions"]
    assert set(timer.timings) == {"Load inputs and initialize evidence",
                                  "Fetch odds, team statistics and model predictions"}


def test_the_memory_panel_shows_use_the_limit_and_the_largest_rise():
    script = ("from app_core import memory_readout as mr\n"
              "from app.ui.memory_readout import render_memory_readout\n"
              "mr.clear()\n"
              "values = iter([900.0, 2300.0])\n"
              "mr.record('Load inputs', read=lambda: next(values), peak=lambda: 950.0)\n"
              "mr.record('Fetch odds', read=lambda: next(values), peak=lambda: 2350.0)\n"
              "mr.rss_mb = lambda: 2300.0\nmr.peak_mb = lambda: 2350.0\n"
              "mr.readout.__kwdefaults__.update(read=mr.rss_mb, peak=mr.peak_mb)\n"
              "render_memory_readout()\n")
    saved = (mr.rss_mb, mr.peak_mb, dict(mr.readout.__kwdefaults__))
    try:
        at = AppTest.from_string(script)
        at.run(timeout=30)
    finally:
        mr.rss_mb, mr.peak_mb = saved[0], saved[1]
        mr.readout.__kwdefaults__.update(saved[2])
    assert not at.exception
    assert any("Using 2,300 MB now, peak 2,350 MB" in m.value and "2,700 MB (85% used)" in m.value for m in at.markdown)
    assert len(at.warning) == 1 and "within 20% of the limit" in at.warning[0].value
    assert any('Largest rise so far: 1,400 MB during "Fetch odds"' in c.value for c in at.caption)
    assert len(at.dataframe) == 1


def test_the_panel_is_rendered_by_the_app_and_not_inside_the_readiness_dashboard():
    # CI 2026-11-10: memory notes are process-wide, so a panel inside the dashboard added a table whose
    # presence depended on what had run before, and the dashboard's own table count failed.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "render_memory_readout" not in (root / "app/ui/readiness_dashboard.py").read_text(encoding="utf-8")
    assert "render_memory_readout()" in (root / "streamlit_app.py").read_text(encoding="utf-8")
