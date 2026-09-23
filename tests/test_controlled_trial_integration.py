import json
from datetime import datetime

import pandas as pd
import pytest

from app_core.controlled_trial import (
    AUTHORITY, LEGACY_PUBLIC_FIELDS, LEGACY_VERSION, PUBLIC_FIELDS, VERSION,
    public_fields_for_version, validate_contract,
)
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, pick_record, validate_package
from app_core.public_history import selections
from integrations.gemini_client import _deterministic_review_mask


RUN = "20260911T200000.000000Z"
QUOTE = "2026-09-11T19:59:00+00:00"
START = "2026-09-11T23:00:00+00:00"


def contract(**updates):
    value = {
        "version": VERSION,
        "authority": AUTHORITY,
        "candidate_id": "detroit-run-line",
        "game_id": "g1",
        "sport": "MLB",
        "market_type": "spread_home",
        "selection": "Detroit -1.5",
        "line": -1.5,
        "odds": 167.0,
        "sportsbook": "Novig",
        "quote_timestamp": QUOTE,
        "start": START,
        "probability_source": "calibrated_probability",
        "source_probability_semantics": "win_conditional_on_decision",
        "probability_semantics": "win_unconditional_with_push",
        "estimated_probability": 0.3981907740602143,
        "push_probability": 0.0,
        "loss_probability": 1 - 0.3981907740602143,
        "break_even_probability": 1 / 2.67,
        "estimated_price_edge": 0.3981907740602143 - 1 / 2.67,
        "estimated_expected_value": 0.3981907740602143 * 2.67 - 1,
        "gemini_review_status": "APPROVE",
        "gemini_stake_multiplier": 1.0,
        "identity_verified": True,
        "quote_verified": True,
        "quote_fresh": True,
        "data_quality_status": "NO_DEGRADED_FLAGS",
        "trial_eligible": True,
        "bankroll_fraction": 0.0025,
        "recommended_bet_amount": 2.5,
        "reason": "Owner-authorized controlled trial",
    }
    value.update(updates)
    assert set(value) == set(PUBLIC_FIELDS)
    return value


def test_archived_v1_contract_is_read_only():
    historical = {key: value for key, value in contract().items()
                  if key in LEGACY_PUBLIC_FIELDS}
    historical["version"] = LEGACY_VERSION
    assert public_fields_for_version(historical) == LEGACY_PUBLIC_FIELDS
    with pytest.raises(ValueError):
        validate_contract(historical)
    validate_contract(historical, read_only_legacy=True)


def candidate():
    return {
        "candidate_id": "detroit-run-line",
        "matchup_id": "g1",
        "game_id": "g1",
        "export_run_id": RUN,
        "league": "MLB",
        "home_team": "Detroit",
        "away_team": "Washington",
        "game_date": "2026-09-11",
        "best_pick": "Detroit -1.5",
        "market_type": "spread_home",
        "spread_line": -1.5,
        "odds_american": 167,
        "calibrated_probability": 0.3981907740602143,
        "expected_value": 0.06316936674077211,
        "best_available_rank": 2,
        "best_available_family_rank": 1,
        "provider_quotes": json.dumps([
            {
                "book": "novig",
                "market_type": "spread_home",
                "point": -1.5,
                "price": 167,
                "recorded_at": QUOTE,
            }
        ]),
    }


def final():
    return {
        "candidate_id": "detroit-run-line",
        "matchup_id": "g1",
        "game_id": "g1",
        "export_run_id": RUN,
        "league": "MLB",
        "Home": "Detroit",
        "Away": "Washington",
        "Local Date": "2026-09-11",
        "game_time_est": "2026-09-11 7:00 PM ET",
        "best_pick": "Detroit -1.5",
        "market_type": "spread_home",
        "odds_american": 167,
        "Bettable": False,
        "Play_Stake": 0.0,
        "production_eligible": False,
        "wager_approved": False,
        "gemini_review_status": "APPROVE",
        "controlled_trial_contract": contract(),
    }


def trial_board_row():
    return per_game_board(
        pd.DataFrame([final()]),
        pd.DataFrame([candidate()]),
        novig_only=True,
    ).iloc[0]


def test_trial_candidate_consumes_gemini_review_without_strict_eligibility():
    frame = pd.DataFrame([
        {"production_eligible": False, "controlled_trial_candidate": True},
        {"production_eligible": False, "controlled_trial_candidate": False},
    ])
    assert _deterministic_review_mask(frame).tolist() == [True, False]


def test_exact_trial_ticket_round_trips_to_public_board_and_history():
    row = trial_board_row()
    assert row["status"] == "TRIAL"
    assert row["Trial_Stake"] == 2.5
    assert not row["Bettable"] and row["Play_Stake"] == 0

    public = pick_record(row)
    assert public["status"] == "TRIAL"
    assert public["pick"] == "Detroit -1.5"
    assert public["controlled_trial_contract"]["recommended_bet_amount"] == 2.5

    frame = pd.DataFrame([row])
    package = build_package(frame, frame.copy(), frame.copy())
    validate_package(package)
    history = selections([
        {
            "confirmed_at": "2026-09-11T20:01:00+00:00",
            "package_hash": "trial-package",
            "package": package,
        }
    ])
    assert history
    assert {item["group"] for item in history} == {"Controlled trial"}


def test_public_trial_rejects_contract_row_mismatch():
    row = trial_board_row().copy()
    row["controlled_trial_contract"] = contract(recommended_bet_amount=3.0)
    with pytest.raises(ValueError, match="authorization"):
        pick_record(row)
