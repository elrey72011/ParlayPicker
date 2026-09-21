from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app_core.controlled_trial import (
    MAX_BANKROLL_FRACTION_PER_PICK,
    MAX_DOLLARS_PER_PICK,
    apply_trials,
    select_review_candidates,
    validate_contract,
)


NOW = datetime(2026, 9, 21, 19, 50, tzinfo=timezone.utc)


def candidate(**changes):
    row = {
        "candidate_id": "detroit-run-line",
        "game_id": "mlb-det-was",
        "matchup_id": "mlb-det-was",
        "league": "MLB",
        "sport": "MLB",
        "home_team": "Detroit",
        "away_team": "Washington",
        "market_type": "spread_home",
        "best_pick": "Detroit -1.5",
        "selection": "Detroit -1.5",
        "spread_line": -1.5,
        "odds_american": 167,
        "odds_source": "novig",
        "quote_bookmaker": "novig",
        "calibrated_probability": 0.3981907740602143,
        "expected_value": 0.06316936674077211,
        "game_start_utc": (NOW + timedelta(hours=2)).isoformat(),
        "odds_recorded_at": (NOW - timedelta(minutes=5)).isoformat(),
        "quote_binding_verified": True,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "identity_verified": True,
        "degraded_feature_subset_flag": False,
        "critical_feature_error": False,
    }
    row.update(changes)
    return row


def final_row(**changes):
    row = {
        "candidate_id": "research-winner",
        "game_id": "mlb-det-was",
        "matchup_id": "mlb-det-was",
        "league": "MLB",
        "home_team": "Detroit",
        "away_team": "Washington",
        "best_pick": "Washington +1.5",
        "market_type": "spread_away",
        "odds_american": -174,
        "production_eligible": False,
        "wager_approved": False,
        "Bettable": False,
        "Play_Stake": 0.0,
        "Kelly_Bet_Size": 0.0,
        "wager_contract": {"production_eligible": False},
    }
    row.update(changes)
    return row


def reviewed(**changes):
    row = candidate(
        controlled_trial_deterministic_eligible=True,
        controlled_trial_candidate=True,
        controlled_trial_estimated_probability=0.3981907740602143,
        controlled_trial_break_even_probability=1 / 2.67,
        controlled_trial_estimated_price_edge=0.3981907740602143 - 1 / 2.67,
        controlled_trial_estimated_expected_value=0.3981907740602143 * 2.67 - 1,
        gemini_review_status="APPROVE",
        gemini_stake_multiplier=1.0,
        gemini_reviewed_at=NOW.isoformat(),
    )
    row.update(changes)
    return row


def test_value_candidate_is_selected_instead_of_negative_ev_probability_winner():
    rows = pd.DataFrame([
        candidate(),
        candidate(
            candidate_id="washington-run-line",
            market_type="spread_away",
            best_pick="Washington +1.5",
            selection="Washington +1.5",
            spread_line=1.5,
            odds_american=-174,
            calibrated_probability=0.6031519446910828,
            expected_value=-0.050209006635881126,
        ),
    ])
    chosen = select_review_candidates(rows, now=NOW)
    assert chosen["best_pick"].tolist() == ["Detroit -1.5"]
    assert chosen.iloc[0]["controlled_trial_estimated_price_edge"] == pytest.approx(0.02365893885)


@pytest.mark.parametrize(
    "change, blocker",
    [
        ({"calibrated_probability": 0.38}, "price edge below 2.0%"),
        ({"expected_value": -0.01}, "producer expected value is not positive"),
        ({"quote_binding_verified": False}, "exact quote not verified"),
        ({"line_consistency_flag": False}, "line identity not verified"),
        ({"odds_recorded_at": (NOW - timedelta(minutes=31)).isoformat()}, "quote stale"),
        ({"game_start_utc": (NOW - timedelta(minutes=1)).isoformat()}, "game started"),
        ({"critical_feature_error": True}, "degraded or critical input state"),
    ],
)
def test_deterministic_trial_gate_fails_closed(change, blocker):
    row = pd.DataFrame([candidate(**change)])
    evaluated = select_review_candidates(row, now=NOW)
    assert evaluated.empty
    from app_core.controlled_trial import evaluate_candidates

    assert blocker in evaluate_candidates(row, now=NOW).iloc[0]["controlled_trial_gate_reason"]


def test_degraded_total_is_not_a_trial_candidate():
    row = candidate(
        market_type="total_over",
        best_pick="Over 8.5",
        selection="Over 8.5",
        total_line=8.5,
        total_input_status="DEGRADED",
    )
    assert select_review_candidates(pd.DataFrame([row]), now=NOW).empty


def test_approved_review_creates_capped_trial_without_production_approval():
    out = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 1000, now=NOW)
    row = out.iloc[0]
    contract = row["controlled_trial_contract"]
    assert row["best_pick"] == "Detroit -1.5"
    assert row["production_eligible"] is False or not bool(row["production_eligible"])
    assert row["wager_approved"] is False or not bool(row["wager_approved"])
    assert row["Pick_Status"] == "Controlled Trial"
    assert contract["trial_eligible"] is True
    assert contract["recommended_bet_amount"] == 2.50
    assert contract["bankroll_fraction"] == pytest.approx(MAX_BANKROLL_FRACTION_PER_PICK)
    validate_contract(contract)


def test_absolute_dollar_cap_and_medium_multiplier():
    high = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 10000, now=NOW)
    assert high.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == MAX_DOLLARS_PER_PICK
    medium = apply_trials(
        pd.DataFrame([final_row()]),
        pd.DataFrame([reviewed(gemini_stake_multiplier=0.75)]),
        1000,
        now=NOW,
    )
    assert medium.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == 1.88


def test_gemini_hold_and_validated_wager_are_never_overridden():
    held = apply_trials(
        pd.DataFrame([final_row()]),
        pd.DataFrame([reviewed(gemini_review_status="HOLD", gemini_stake_multiplier=0.0)]),
        1000,
        now=NOW,
    )
    assert "controlled_trial_contract" not in held or pd.isna(held.iloc[0].get("controlled_trial_contract"))

    validated = apply_trials(
        pd.DataFrame([final_row(wager_contract={"production_eligible": True})]),
        pd.DataFrame([reviewed()]),
        1000,
        now=NOW,
    )
    assert validated.iloc[0]["best_pick"] == "Washington +1.5"


def test_contract_validator_rejects_inflated_stake():
    out = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 1000, now=NOW)
    contract = dict(out.iloc[0]["controlled_trial_contract"])
    contract["recommended_bet_amount"] = MAX_DOLLARS_PER_PICK + 0.01
    with pytest.raises(ValueError, match="Invalid eligible controlled trial"):
        validate_contract(contract)
