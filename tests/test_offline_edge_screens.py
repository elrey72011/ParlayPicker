"""Offline edge screens (roadmap M5): the segment arithmetic and that the report runs on the stored data."""
import importlib

import pandas as pd
import pytest

screens = importlib.import_module("scripts.offline_edge_screens")


def test_decimal_odds_and_break_even():
    assert screens.decimal_odds(-110) == pytest.approx(1.909090909)
    assert screens.decimal_odds(150) == pytest.approx(2.5)
    assert 1 / screens.decimal_odds(-141) == pytest.approx(0.585, abs=0.001)


def test_segment_excess_roi_and_day_clustered_z(monkeypatch):
    monkeypatch.setattr(screens, "MIN_PICKS", 4)
    frame = pd.DataFrame({"won": [1, 1, 0, 1], "breakeven": [0.5] * 4, "decimal": [2.0] * 4,
                          "day": ["a", "a", "b", "b"]})
    row = screens.segment(frame, "x")
    assert row["picks"] == 4 and row["days"] == 2 and row["win_rate"] == 0.75 and row["excess"] == 0.25
    assert row["roi"] == pytest.approx(0.5)
    # Residuals sum to +1 on day a and 0 on day b: clustered standard error 1, z = 1.
    assert row["z"] == pytest.approx(1.0)
    assert screens.segment(frame.head(3), "too few") is None


def test_report_runs_on_the_stored_data_and_props_are_below_break_even():
    frame, rows = screens.props()
    overall = rows[0]
    assert overall["picks"] == len(frame) >= 741 and overall["excess"] < 0 and overall["roi"] < 0
    text = screens.report()
    assert "## MLB props against their logged prices" in text and "## Graded best picks against Kalshi" in text
