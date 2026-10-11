"""Clock isolation for four named synthetic pilot tests, never production.

Their original inference/export clocks were frozen, but ranking consulted the
wall clock. Only these original successful synthetic cases use their declared
clock; other cases, including stale/start rejections, are untouched.
"""
import pytest

TARGET = 'tests/test_ncaaf_pilot.py::test_accepted_synthetic_caller_then_existing_capture_export_display'


def applies(nodeid):
    return nodeid.split('[', 1)[0] == TARGET


@pytest.fixture(autouse=True)
def ncaaf_pilot_synthetic_ranking_clock(request, monkeypatch):
    if applies(request.node.nodeid):
        declared = request.module.previous.NOW
        monkeypatch.setattr('app_core.candidate_chronology.now_utc', lambda: declared.isoformat())
