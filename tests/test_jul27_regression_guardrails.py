"""Regression guards for the 27 Jul game/prop corrections."""
from __future__ import annotations

import pandas as pd

from app_core.prop_grading import grade_prop_export, grading_summary
from app_core.prop_runner import apply_production_prop_gate


def test_total_bases_is_research_only_in_both_directions():
    card = pd.DataFrame(
        [
            {
                "player": "Over Batter",
                "matchup": "A @ B",
                "market_type": "batter_total_bases_over",
                "best_pick": "Over Batter Over 1.5 Total Bases",
                "line": 1.5,
                "odds_american": -110,
                "WinProbability": 0.70,
                "expected_value": 0.15,
                "expected_count": 2.2,
                "Pick_Status": "Actionable",
                "Market_Probation": False,
                "CalibrationSource": "directional",
                "DirectionalCalibrationSampleSize": 50,
                "Kelly_Bet_Size": 5.0,
            },
            {
                "player": "Under Batter",
                "matchup": "C @ D",
                "market_type": "batter_total_bases_under",
                "best_pick": "Under Batter Under 1.5 Total Bases",
                "line": 1.5,
                "odds_american": -110,
                "WinProbability": 0.70,
                "expected_value": 0.15,
                "expected_count": 0.8,
                "Pick_Status": "Actionable",
                "Market_Probation": False,
                "CalibrationSource": "directional",
                "DirectionalCalibrationSampleSize": 50,
                "Kelly_Bet_Size": 5.0,
            },
        ]
    )

    gated = apply_production_prop_gate(card)

    assert not gated["production_market_allowed"].any()
    assert not gated["production_eligible"].any()
    assert gated["Kelly_Bet_Size"].eq(0.0).all()
    assert gated["production_gate_reason"].str.contains(
        "total bases are production-disabled", case=False
    ).all()


def test_unresolved_prop_is_explicitly_pending_and_not_in_win_rate():
    card = pd.DataFrame(
        [
            {
                "player": "Test Batter",
                "market_type": "batter_hits_over",
                "best_pick": "Test Batter Over 0.5 Hits",
                "line": 0.5,
                "odds_american": -110,
                "Kelly_Bet_Size": 1.0,
            }
        ]
    )

    graded = grade_prop_export(
        card,
        "2026-07-26",
        name_resolver=lambda _card, _date: {"test batter": 123},
        actual_fetcher=lambda _player_id, _participant_type: None,
    )
    summary = grading_summary(graded)

    assert graded.iloc[0]["result"] == "PENDING"
    assert pd.isna(graded.iloc[0]["profit"])
    assert summary["graded"] == 0
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["unresolved"] == 1
