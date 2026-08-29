"""grade_slate: clean below-stake "No Play" picks must grade into the calibration; only
corrupt-line No Play rows stay excluded (22 Jun, user-directed — learn from the full slate).

Before this, every "No Play" recap row was dropped, so a clean overs-bleed slate (overs 0-5,
unders 5-1, all No Play / Below Threshold) contributed only its 2 Below-Threshold rows to the
calibration — the directional signal the model most needed was thrown away. These tests pin
that a No Play on a CLEAN export line grades, while a No Play whose export line carries a
provenance warning (the 6-19/20/21 corrupt-feed shape) stays out.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.grade_slate import grade


def _export(rows):
    return pd.DataFrame(rows)


def _recap(rows):
    return pd.DataFrame(rows)


def _grade(tmp_path, export_rows, recap_rows):
    exp_p = tmp_path / "export.csv"
    rec_p = tmp_path / "recap.csv"
    out_p = tmp_path / "graded.csv"
    _export(export_rows).to_csv(exp_p, index=False)
    _recap(recap_rows).to_csv(rec_p, index=False)
    grade(exp_p, rec_p, out_p)
    return pd.read_csv(out_p)


def test_clean_no_play_below_stake_is_graded(tmp_path):
    export_rows = [{
        "league": "MLB", "Home": "Tampa Bay", "Away": "Kansas City",
        "best_pick": "Under 7.5", "market_type": "total_under",
        "consensus_agreement": "Agrees", "Kelly_Bet_Size": 0.0, "odds_american": -110,
        "line_provenance_warning": "", "line_consistency_flag": True,
    }]
    recap_rows = [{
        "Home": "Tampa Bay", "Away": "Kansas City", "Pick Taken": "Under 7.5",
        "actual_home_score": 1, "actual_away_score": 2, "Outcome": "WIN", "Status": "No Play",
    }]
    out = _grade(tmp_path, export_rows, recap_rows)
    assert len(out) == 1
    assert out.iloc[0]["W/L"] == "WIN"
    assert out.iloc[0]["consensus_agreement"] == "Agrees"


def test_corrupt_line_no_play_is_excluded(tmp_path):
    export_rows = [{
        "league": "MLB", "Home": "Miami", "Away": "Texas",
        "best_pick": "Over 8.5", "market_type": "total_over",
        "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0, "odds_american": 265,
        "line_provenance_warning": "upload_total_fallback_after_rejected_live",
        "line_consistency_flag": True,
    }]
    recap_rows = [{
        "Home": "Miami", "Away": "Texas", "Pick Taken": "Over 8.5",
        "actual_home_score": 3, "actual_away_score": 4, "Outcome": "LOSS", "Status": "No Play",
    }]
    out = _grade(tmp_path, export_rows, recap_rows)
    assert len(out) == 0


def test_inconsistent_line_no_play_is_excluded(tmp_path):
    export_rows = [{
        "league": "MLB", "Home": "Miami", "Away": "Texas",
        "best_pick": "Over 8.5", "market_type": "total_over",
        "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0, "odds_american": -110,
        "line_provenance_warning": "", "line_consistency_flag": False,
    }]
    recap_rows = [{
        "Home": "Miami", "Away": "Texas", "Pick Taken": "Over 8.5",
        "actual_home_score": 3, "actual_away_score": 4, "Outcome": "LOSS", "Status": "No Play",
    }]
    out = _grade(tmp_path, export_rows, recap_rows)
    assert len(out) == 0


def test_zero_zero_no_play_voided_even_when_clean(tmp_path):
    # Postponed game: clean line but a 0-0 final is not a real result -> voided.
    export_rows = [{
        "league": "MLB", "Home": "New York Mets", "Away": "Chicago Cubs",
        "best_pick": "Under 8.5", "market_type": "total_under",
        "consensus_agreement": "Disagrees", "Kelly_Bet_Size": 0.0, "odds_american": -110,
        "line_provenance_warning": "", "line_consistency_flag": True,
    }]
    recap_rows = [{
        "Home": "New York Mets", "Away": "Chicago Cubs", "Pick Taken": "Under 8.5",
        "actual_home_score": 0, "actual_away_score": 0, "Outcome": "N/A", "Status": "No Play",
    }]
    out = _grade(tmp_path, export_rows, recap_rows)
    assert len(out) == 0


def test_blank_outcomes_recover_from_exact_final_scores(tmp_path):
    export_rows = [
        {
            "league": "NCAAF", "Home": "West Florida", "Away": "Southern Illinois",
            "best_pick": "Under 55.5", "market_type": "total_under",
            "consensus_agreement": "Agrees", "Kelly_Bet_Size": 0.0,
            "odds_american": -110, "line_provenance_warning": "",
            "line_consistency_flag": True,
        },
        {
            "league": "NCAAF", "Home": "Murray State", "Away": "Eastern Illinois",
            "best_pick": "Eastern Illinois +3.0", "market_type": "spread_away",
            "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0,
            "odds_american": -110, "line_provenance_warning": "",
            "line_consistency_flag": True,
        },
    ]
    recap_rows = [
        {
            "Home": "West Florida", "Away": "Southern Illinois",
            "Pick Taken": "Under 55.5", "actual_home_score": 17,
            "actual_away_score": 20, "Outcome": "N/A", "Status": "Best Available / Pass",
        },
        {
            "Home": "Murray State", "Away": "Eastern Illinois",
            "Pick Taken": "Eastern Illinois +3.0", "actual_home_score": 21,
            "actual_away_score": 20, "Outcome": "", "Status": "Best Available / Pass",
        },
    ]

    out = _grade(tmp_path, export_rows, recap_rows)

    assert out["W/L"].tolist() == ["WIN", "WIN"]


def test_blank_outcome_with_unmatched_spread_team_stays_ungraded(tmp_path):
    export_rows = [{
        "league": "NCAAF", "Home": "West Florida", "Away": "Southern Illinois",
        "best_pick": "Unknown Team +3.0", "market_type": "spread_away",
        "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0,
        "odds_american": -110, "line_provenance_warning": "",
        "line_consistency_flag": True,
    }]
    recap_rows = [{
        "Home": "West Florida", "Away": "Southern Illinois",
        "Pick Taken": "Unknown Team +3.0", "actual_home_score": 17,
        "actual_away_score": 20, "Outcome": "N/A", "Status": "Best Available / Pass",
    }]

    out = _grade(tmp_path, export_rows, recap_rows)

    assert out.empty
