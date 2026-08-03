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


def test_unqualified_game_row_gets_explicit_best_available_scope():
    frame = pd.DataFrame({
        "best_pick": ["Under 8.5"],
        "display_pick": ["Under 8.5"],
        "qualified_pick": [False],
        "Bet_Decision": ["BEST AVAILABLE - PASS"],
        "production_eligible": [False],
        "Play_Stake": [0.0],
        "Pick_Status": ["Below Threshold"],
        "Pick_Quality": ["No Edge"],
    })

    labeled = label_wager_export(frame)

    assert labeled.loc[0, "Export_Scope"] == "BEST AVAILABLE PICK / RESEARCH"
    assert labeled.loc[0, "Wager_Instruction"] == (
        "DO NOT BET - BEST AVAILABLE PICK DOES NOT CLEAR THE WAGER GATE"
    )
    assert labeled.loc[0, "Pick_Status"] == "Best Available / Pass"
    assert labeled.loc[0, "Pick_Quality"] == "No Bet - Best Available"


def test_qualified_unfunded_game_row_stays_a_zero_dollar_lean():
    frame = pd.DataFrame({
        "best_pick": ["Chicago +1.5"],
        "qualified_pick": [True],
        "Bet_Decision": ["QUALIFIED LEAN - PASS"],
        "production_eligible": [False],
        "Play_Stake": [0.0],
        "Pick_Status": ["Near Miss"],
        "Pick_Quality": ["Lean"],
    })

    labeled = label_wager_export(frame)

    assert not bool(labeled.loc[0, "Bettable"])
    assert labeled.loc[0, "Export_Scope"] == "QUALIFIED LEAN / RESEARCH"
    assert labeled.loc[0, "Wager_Instruction"] == (
        "DO NOT BET - QUALIFIED LEAN HAS NO APPROVED STAKE"
    )
    assert labeled.loc[0, "Pick_Status"] == "Qualified Lean / Pass"
    assert labeled.loc[0, "Pick_Quality"] == "No Bet - Qualified Lean"


def test_qualified_unfunded_export_cannot_retain_stale_bet_labels_or_stake():
    frame = pd.DataFrame({
        "best_pick": ["Under 11.5"],
        "qualified_pick": [True],
        "Bet_Decision": ["BET"],
        "Play_Tier": ["BET"],
        "production_eligible": [False],
        "Play_Units": [2.0],
        "Play_Stake": [2.0],
        "Kelly_Bet_Size": [0.0],
        "Wager_Approved": [False],
        "Pick_Status": ["Actionable"],
        "Pick_Quality": ["Value"],
    })

    labeled = label_wager_export(frame)

    assert not bool(labeled.loc[0, "Bettable"])
    assert not bool(labeled.loc[0, "Wager_Approved"])
    assert labeled.loc[0, "Bet_Decision"] == "QUALIFIED LEAN - PASS"
    assert labeled.loc[0, "Play_Tier"] == "LEAN"
    assert float(labeled.loc[0, "Play_Units"]) == 0.0
    assert float(labeled.loc[0, "Play_Stake"]) == 0.0
    assert labeled.loc[0, "Wager_Instruction"].startswith("DO NOT BET")


def test_public_export_scope_columns_are_case_insensitively_unique():
    frame = pd.DataFrame(
        {
            "best_pick": ["Over 8.5"],
            "wager_instruction": [
                "DO NOT TREAT AS AN APPROVED BET; diagnostic coverage only."
            ],
            "Bet_Decision": ["PASS"],
            "production_eligible": [False],
            "Play_Stake": [0.0],
        }
    )

    labeled = label_wager_export(frame)

    folded = [column.casefold() for column in labeled.columns]
    assert len(folded) == len(set(folded))
    assert "wager_instruction" not in labeled.columns
    assert labeled.loc[0, "Wager_Instruction"] == "DO NOT BET - $0 PASS / RESEARCH"
