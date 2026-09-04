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


def test_controlled_value_wager_has_distinct_sellable_scope():
    frame = pd.DataFrame({
        "best_pick": ["Cincinnati -1.5"],
        "Bet_Decision": ["BET"],
        "production_eligible": [True],
        "wager_approved": [True],
        "controlled_card_recovery": [True],
        "Play_Stake": [5.0],
        "Kelly_Bet_Size": [5.0],
        "sellable_as_value_card": [False],
        "commercial_tier": ["Best Available / Pass"],
        "export_role": ["BEST AVAILABLE PICK - PASS / RESEARCH"],
        "empty_card_recovery_triggered": [False],
        "empty_card_recovery_promoted_count": [0],
        "empty_card_recovery_kelly_total": [0.0],
        "final_actionable_count": [0],
        "production_card_empty_flag": [True],
        "production_card_empty_after_recovery_flag": [True],
    })

    labeled = label_wager_export(frame)

    assert bool(labeled.loc[0, "Bettable"])
    assert labeled.loc[0, "Export_Scope"] == "CONTROLLED VALUE BET"
    assert labeled.loc[0, "Wager_Instruction"] == (
        "BET - CONTROLLED VALUE CARD / SMALL STAKE / NOT PREMIUM"
    )
    assert bool(labeled.loc[0, "controlled_card_recovery"])
    assert bool(labeled.loc[0, "sellable_as_value_card"])
    assert labeled.loc[0, "commercial_tier"] == "Controlled Value Pick"
    assert labeled.loc[0, "export_role"] == "CONTROLLED VALUE WAGER"
    assert bool(labeled.loc[0, "empty_card_recovery_triggered"])
    assert labeled.loc[0, "empty_card_recovery_promoted_count"] == 1
    assert labeled.loc[0, "empty_card_recovery_kelly_total"] == 5.0
    assert labeled.loc[0, "final_actionable_count"] == 1
    assert not bool(labeled.loc[0, "production_card_empty_flag"])
    assert not bool(labeled.loc[0, "production_card_empty_after_recovery_flag"])
    assert production_wagers(frame)["best_pick"].tolist() == ["Cincinnati -1.5"]


def test_unfunded_recovery_attempt_cannot_remain_sellable_or_claim_promotion():
    frame = pd.DataFrame({
        "best_pick": ["San Francisco +1.5"],
        "qualified_pick": [True],
        "Bet_Decision": ["QUALIFIED LEAN - PASS"],
        "Play_Tier": ["LEAN"],
        "production_eligible": [True],
        "wager_approved": [True],
        "controlled_card_recovery": [True],
        "Play_Stake": [5.0],
        "Kelly_Bet_Size": [5.0],
        "sellable_as_value_card": [True],
        "commercial_tier": ["Controlled Value Pick"],
        "commercial_reason": ["stale recovery promotion"],
        "export_role": ["CONTROLLED VALUE WAGER"],
        "empty_card_recovery_triggered": [True],
        "empty_card_recovery_promoted_count": [1],
        "empty_card_recovery_kelly_total": [5.0],
        "final_actionable_count": [1],
        "final_positive_kelly_count": [1],
        "production_card_empty_flag": [False],
        "production_card_empty_after_recovery_flag": [False],
        "production_card_recovery_reason": ["stale recovery success"],
        "production_card_empty_reason": [""],
    })

    labeled = label_wager_export(frame)

    assert not bool(labeled.loc[0, "Bettable"])
    assert not bool(labeled.loc[0, "controlled_card_recovery"])
    assert not bool(labeled.loc[0, "sellable_as_value_card"])
    assert labeled.loc[0, "commercial_tier"] == "Qualified Lean / Pass"
    assert labeled.loc[0, "export_role"] == "QUALIFIED LEAN - PASS"
    assert labeled.loc[0, "Play_Stake"] == 0.0
    assert labeled.loc[0, "Kelly_Bet_Size"] == 0.0
    assert not bool(labeled.loc[0, "empty_card_recovery_triggered"])
    assert labeled.loc[0, "empty_card_recovery_promoted_count"] == 0
    assert labeled.loc[0, "empty_card_recovery_kelly_total"] == 0.0
    assert labeled.loc[0, "final_actionable_count"] == 0
    assert labeled.loc[0, "final_positive_kelly_count"] == 0
    assert bool(labeled.loc[0, "production_card_empty_flag"])
    assert bool(labeled.loc[0, "production_card_empty_after_recovery_flag"])
    assert labeled.loc[0, "production_card_recovery_reason"] == ""
    assert labeled.loc[0, "production_card_empty_reason"]


def test_prop_grading_export_keeps_research_but_marks_it_do_not_bet():
    frame = pd.DataFrame(
        {
            "player_name": ["Funded Player", "Research Player"],
            "Stake_Status": ["Funded", "Research / No Stake"],
            "production_eligible": [True, False],
            "Kelly_Bet_Size": [1.0, 0.0],
            "extended_flat_stake": [1.0, 1.0],
        }
    )

    labeled = label_wager_export(frame)

    assert labeled["Export_Scope"].tolist() == ["PRODUCTION BET", "COVERAGE / RESEARCH"]
    assert labeled["extended_flat_stake"].tolist() == [1.0, 0.0]
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


def test_unqualified_pass_replaces_stale_actionable_and_qualified_reasons():
    frame = pd.DataFrame({
        "best_pick": ["Athletics +1.5"],
        "qualified_pick": [False],
        "qualification_reason": [
            "PASS: final calibrated win probability is below 60%."
        ],
        "Bet_Decision": ["BEST AVAILABLE - PASS"],
        "Play_Tier": ["AVOID"],
        "production_eligible": [False],
        "Play_Stake": [0.0],
        "Pick_Status": ["Actionable"],
        "Status_Reason": [
            "Actionable (empirical): proven-bucket realized edge +4.3%"
        ],
        "Production_Gate_Reason": ["qualified"],
    })

    labeled = label_wager_export(frame)

    assert not bool(labeled.loc[0, "Bettable"])
    assert labeled.loc[0, "Pick_Status"] == "Best Available / Pass"
    assert labeled.loc[0, "Status_Reason"] == frame.loc[0, "qualification_reason"]
    assert (
        labeled.loc[0, "Production_Gate_Reason"]
        == frame.loc[0, "qualification_reason"]
    )


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
