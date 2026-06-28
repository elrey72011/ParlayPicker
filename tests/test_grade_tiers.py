"""Tier-results tracking: pure recap grading and per-tier summary."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.grade_tiers import grade_recap, summarize_by_tier, tier_of


def test_tier_grouping():
    assert tier_of("No Play") == "No Play"
    assert tier_of("Actionable") == "Actionable"
    assert tier_of("Below Threshold") == "Below/HV"
    assert tier_of("High Variance/Speculative") == "Below/HV"


def test_grade_recap_skips_non_wl_and_derives_date():
    recap = pd.DataFrame([
        {"Home": "Tampa Bay", "Away": "Arizona", "Pick Taken": "Under 8.5", "Outcome": "WIN",
         "Status": "Below Threshold", "export_run_id": "20260627T142621Z"},
        {"Home": "X", "Away": "Y", "Pick Taken": "Over 8.5", "Outcome": "N/A",     # skipped
         "Status": "No Play", "export_run_id": "20260627T142621Z"},
        {"Home": "Boston", "Away": "New York Yankees", "Pick Taken": "Under 8.5", "Outcome": "LOSS",
         "Status": "No Play", "export_run_id": "20260627T142621Z"},
    ])
    rows = grade_recap(recap)
    assert len(rows) == 2                       # N/A dropped
    assert rows[0]["date"] == "2026-06-27"      # derived from run id
    assert rows[0]["matchup"] == "Arizona @ Tampa Bay"
    assert rows[1]["tier"] == "No Play"


def test_summary_record_winrate_and_breakeven_pnl():
    df = pd.DataFrame([
        {"tier": "No Play", "outcome": "WIN"},
        {"tier": "No Play", "outcome": "WIN"},
        {"tier": "No Play", "outcome": "LOSS"},
        {"tier": "Below/HV", "outcome": "LOSS"},
    ])
    s = summarize_by_tier(df).set_index("tier")
    assert s.loc["No Play", "W"] == 2 and s.loc["No Play", "L"] == 1
    assert abs(s.loc["No Play", "win_rate"] - 2 / 3) < 1e-9
    # flat 1u at -110: 2 wins * 0.909 - 1 loss = +0.818 units
    assert abs(s.loc["No Play", "units"] - (2 * (100 / 110) - 1)) < 1e-2
    assert s.loc["Below/HV", "units"] == -1.0   # a single loss = -1 unit
