from __future__ import annotations

import pandas as pd

from app_core.export_scope import label_wager_export, production_wagers


def test_game_export_separates_production_bets_from_coverage():
    frame = pd.DataFrame(
        {
            "best_pick": ["Over 8.5", "Boston +1.5", "Under 7.5"],
            "Bet_Decision": ["BET", "PASS", "BET"],
            "production_eligible": [True, False, False],
            "Play_Stake": [5.0, 0.0, 2.0],
        }
    )

    labeled = label_wager_export(frame)

    assert labeled["Bettable"].tolist() == [True, False, False]
    assert labeled["Wager_Instruction"].tolist() == [
        "BET - APP APPROVED",
        "DO NOT BET - $0 PASS / RESEARCH",
        "DO NOT BET - $0 PASS / RESEARCH",
    ]
    assert production_wagers(frame)["best_pick"].tolist() == ["Over 8.5"]


def test_prop_grading_export_keeps_research_but_marks_it_do_not_bet():
    frame = pd.DataFrame(
        {
            "player_name": ["Funded Player", "Research Player"],
            "Stake_Status": ["Funded", "Research / No Stake"],
            "production_eligible": [True, False],
            "Kelly_Bet_Size": [1.0, 0.0],
        }
    )

    labeled = label_wager_export(frame)

    assert labeled["Export_Scope"].tolist() == ["PRODUCTION BET", "COVERAGE / RESEARCH"]
    assert production_wagers(frame)["player_name"].tolist() == ["Funded Player"]
