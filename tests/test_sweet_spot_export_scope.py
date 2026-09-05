import pandas as pd
import pytest

from streamlit_app import _prepare_sweet_spot_card


def _row(**overrides):
    return {
        "league": "NCAAF", "home_team": "Middle Tennessee", "away_team": "Murray State",
        "game_date": "2026-09-05", "market_type": "total_under", "best_pick": "Under 55.5",
        "Pick_Status": "Actionable", "Status_Reason": "Actionable intermediate model score",
        "calibrated_probability": 0.67, "effective_win_probability": 0.67,
        "expected_value": 0.30, "effective_expected_value": 0.30, "edge": 0.17,
        "odds_american": -110, "consensus_agreement": "Agrees", "Kelly_Bet_Size": 8.0,
        "production_eligible": False, "wager_approved": False, "qualified_pick": False,
        "qualification_reason": "PASS: final production checks rejected the row.",
        "gemini_gate_enabled": True, "gemini_review_status": "APPROVE",
        "export_run_id": "20260905T164027Z", **overrides,
    }


@pytest.mark.parametrize("categorical", [False, True])
def test_sweet_spot_reconciles_intermediate_actionable_to_unfunded_pass(categorical):
    source = pd.DataFrame([_row()])
    if categorical:
        source["Pick_Status"] = source["Pick_Status"].astype("category")
    result = _prepare_sweet_spot_card(source)
    assert result.loc[0, "Model_Pick_Status"] == "Actionable"
    assert result.loc[0, "Pick_Status"] == "Best Available / Pass"
    assert not result.loc[0, "Bettable"]
    assert result.loc[0, "Play_Stake"] == result.loc[0, "Kelly_Bet_Size"] == 0.0
    assert result.loc[0, "Wager_Instruction"].startswith("DO NOT BET")
    assert result.loc[0, "export_run_id"] == "20260905T164027Z"
    assert result.loc[0, "gemini_review_status"] == "APPROVE"
    assert source.loc[0, "Pick_Status"] == "Actionable"
    assert source.loc[0, "Kelly_Bet_Size"] == 8.0


def test_sweet_spot_keeps_explicitly_approved_stake(monkeypatch):
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)
    result = _prepare_sweet_spot_card(pd.DataFrame([_row(
        production_eligible=True, wager_approved=True, qualified_pick=True,
    )]))
    assert result.loc[0, "Bettable"]
    assert result.loc[0, "Play_Stake"] == 8.0
    assert result.loc[0, "Wager_Instruction"] == "BET - APP APPROVED"
