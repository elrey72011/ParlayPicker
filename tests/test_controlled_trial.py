from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app_core.controlled_trial import (
    MAX_BANKROLL_FRACTION_PER_PICK,
    MAX_DOLLARS_PER_PICK,
    MAX_PICKS_PER_SLATE,
    MAX_REVIEW_CANDIDATES_PER_SLATE,
    apply_trials,
    attest_candidate_integrity,
    enabled,
    evaluate_candidates,
    select_review_candidates,
    validate_contract,
)
from app_core.trial_authority import record_consent, release_reservation
from core.exposure_ledger import append


NOW = datetime(2026, 9, 21, 19, 50, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def trial_runtime(tmp_path, monkeypatch):
    """Synthetic consent and ledger remain isolated from real owner state."""
    consent = tmp_path / "consent.sqlite3"
    exposure = tmp_path / "exposure.sqlite3"
    reservations = tmp_path / "reservations.sqlite3"
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIALS_ENABLED", "1")
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_CONSENT_LEDGER", str(consent))
    monkeypatch.setenv("PARLAYPICKER_EXPOSURE_LEDGER", str(exposure))
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER", str(reservations))
    record_consent(consent, "GRANTED", "synthetic-test-owner", confirmed=True,
                   now=NOW - timedelta(minutes=1), expires_at=NOW + timedelta(days=1))
    configuration_index = 0
    def configure(bankroll=1000, **limits):
        nonlocal configuration_index
        caps = dict(total_cap=.05, daily_cap=.05, weekly_cap=.05,
                    game_cap=.01, team_cap=.01)
        caps.update(limits)
        append(exposure, dict(status="CONFIGURED", bankroll=bankroll,
                              unit_value=10, currency="USD", **caps),
               confirmed=True, now=NOW - timedelta(minutes=1) + timedelta(seconds=configuration_index))
        configuration_index += 1
        return consent, exposure, reservations
    configure()
    return configure


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
        "probability_semantics": "win_conditional_on_decision",
        "home_team_id": "DET",
        "away_team_id": "WSH",
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
        controlled_trial_probability_source="calibrated_probability",
        controlled_trial_source_probability_semantics="win_conditional_on_decision",
        controlled_trial_estimated_probability=0.3981907740602143,
        controlled_trial_push_probability=0.0,
        controlled_trial_loss_probability=1 - 0.3981907740602143,
        controlled_trial_break_even_probability=1 / 2.67,
        controlled_trial_estimated_price_edge=0.3981907740602143 - 1 / 2.67,
        controlled_trial_estimated_expected_value=0.3981907740602143 * 2.67 - 1,
        gemini_review_status="APPROVE",
        gemini_stake_multiplier=1.0,
        gemini_reviewed_at=NOW.isoformat(),
    )
    row.update(changes)
    from app_core.controlled_trial import _trial_input_hash
    row["controlled_trial_input_hash"] = _trial_input_hash(row)
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


def test_expanded_alternate_is_attested_from_its_exact_provider_quote():
    from app_core.prediction_evidence import bind_authoritative_candidates

    row = candidate(
        identity_verified=None,
        line_consistency_flag=None,
        line_event_identity_match_flag=None,
        quote_binding_verified=None,
        provider_event_id=None,
        provider_namespace=None,
        provider_quotes='[{"book":"novig","market_type":"spread_home","point":-1.5,"price":167,"recorded_at":"2026-09-21T19:45:00+00:00","provider_event_id":"odds-event-1","provider_namespace":"odds_api"}]',
    )
    bound = bind_authoritative_candidates(pd.DataFrame([row]))
    certified = attest_candidate_integrity(bound)
    assert bool(certified.iloc[0]["quote_binding_verified"])
    assert certified.iloc[0]["provider_event_id"] == "odds-event-1"
    assert bool(certified.iloc[0]["line_consistency_flag"])
    assert bool(certified.iloc[0]["line_event_identity_match_flag"])
    assert bool(certified.iloc[0]["identity_verified"])
    assert select_review_candidates(certified, now=NOW)["best_pick"].tolist() == ["Detroit -1.5"]


def test_candidate_attestation_preserves_explicit_identity_veto():
    row = candidate(
        identity_verified=False,
        line_event_identity_match_flag=None,
        provider_event_id="odds-event-1",
        provider_namespace="odds_api",
        provider_quotes='[{"book":"novig","market_type":"spread_home","point":-1.5,"price":167,"recorded_at":"2026-09-21T19:45:00+00:00","provider_event_id":"odds-event-1","provider_namespace":"odds_api"}]',
    )
    certified = attest_candidate_integrity(pd.DataFrame([row]))
    assert not bool(certified.iloc[0]["identity_verified"])
    assert not bool(certified.iloc[0]["line_event_identity_match_flag"])
    assert select_review_candidates(certified, now=NOW).empty


def test_candidate_attestation_rejects_cross_event_quote_payload():
    row = candidate(
        identity_verified=None,
        line_consistency_flag=None,
        line_event_identity_match_flag=None,
        provider_event_id="odds-event-1",
        provider_namespace="odds_api",
        provider_quotes='[{"book":"novig","market_type":"spread_home","point":-1.5,"price":167,"recorded_at":"2026-09-21T19:45:00+00:00","provider_event_id":"odds-event-1","provider_namespace":"odds_api"},{"book":"fanduel","market_type":"spread_home","point":-1.5,"price":160,"recorded_at":"2026-09-21T19:45:00+00:00","provider_event_id":"different-event","provider_namespace":"odds_api"}]',
    )
    certified = attest_candidate_integrity(pd.DataFrame([row]))
    assert not bool(certified.iloc[0]["line_event_identity_match_flag"])
    assert select_review_candidates(certified, now=NOW).empty


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


def test_nfl_controlled_trial_requires_complete_recent_result_and_injury_context():
    complete = candidate(
        league="NFL",
        sport="NFL",
        ml_probability_source="score-distribution-v1:nfl",
        nfl_context_model_used=True,
        nfl_context_status="complete",
    )
    assert not select_review_candidates(pd.DataFrame([complete]), now=NOW).empty

    from app_core.controlled_trial import evaluate_candidates

    for change in (
        {"nfl_context_status": "recent_form_only"},
        {"nfl_context_model_used": False},
        {"ml_probability_source": ""},
    ):
        row = dict(complete, **change)
        evaluated = evaluate_candidates(pd.DataFrame([row]), now=NOW).iloc[0]
        assert not evaluated.controlled_trial_deterministic_eligible
        assert "NFL recent-result and injury context incomplete" in evaluated.controlled_trial_gate_reason


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


def test_absolute_dollar_cap_and_medium_multiplier(trial_runtime, tmp_path, monkeypatch):
    trial_runtime(10000)
    high = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 10000, now=NOW)
    assert high.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == MAX_DOLLARS_PER_PICK
    trial_runtime(1000)
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER", str(tmp_path / "medium-reservations.sqlite3"))
    medium = apply_trials(
        pd.DataFrame([final_row()]),
        pd.DataFrame([reviewed(gemini_stake_multiplier=0.75)]),
        1000,
        now=NOW,
    )
    assert medium.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == 1.87


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


def test_trial_default_off_without_switch_or_persisted_consent(monkeypatch):
    monkeypatch.delenv("PARLAYPICKER_CONTROLLED_TRIALS_ENABLED", raising=False)
    assert not enabled(now=NOW)
    diagnostics = {}
    out = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]),
                       1000, now=NOW, diagnostics=diagnostics)
    assert "controlled_trial_contract" not in out
    assert diagnostics["controlled_trial_authorization_status"] == "TRIALS_DISABLED"
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIALS_ENABLED", "1")
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_CONSENT_LEDGER", "/no/such/trial-consent.sqlite3")
    assert not enabled(now=NOW)


def test_revoked_consent_and_missing_exposure_fail_closed(trial_runtime, tmp_path, monkeypatch):
    consent, _, _ = trial_runtime()
    record_consent(consent, "REVOKED", "synthetic-test-owner", confirmed=True,
                   now=NOW - timedelta(seconds=30))
    assert not enabled(now=NOW)
    held = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 1000, now=NOW)
    assert "controlled_trial_contract" not in held
    record_consent(consent, "GRANTED", "synthetic-test-owner", confirmed=True,
                   now=NOW - timedelta(seconds=20), expires_at=NOW + timedelta(days=1))
    monkeypatch.setenv("PARLAYPICKER_EXPOSURE_LEDGER", str(tmp_path / "missing.sqlite3"))
    diagnostics = {}
    held = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]), 1000,
                        now=NOW, diagnostics=diagnostics)
    assert "controlled_trial_contract" not in held
    assert diagnostics["controlled_trial_allocation_status"] == "EXPOSURE_LEDGER_UNAVAILABLE"


def test_existing_committed_exposure_blocks_trial(trial_runtime):
    _, ledger, _ = trial_runtime()
    append(ledger, dict(status="COMMITTED", bet_id="synthetic-straight",
                        source_snapshot_id="synthetic-receipt", sportsbook="Novig",
                        stake_dollars=10.0, legs=[dict(sport="MLB", game_id="mlb-det-was",
                        team_ids=["DET", "WSH"], market="spread_home",
                        selection="Detroit -1.5", odds=167, line=-1.5)]),
           confirmed=True, now=NOW - timedelta(seconds=30))
    diagnostics = {}
    held = apply_trials(pd.DataFrame([final_row()]), pd.DataFrame([reviewed()]),
                        1000, now=NOW, diagnostics=diagnostics)
    assert held.iloc[0].get("controlled_trial_contract") is None
    assert diagnostics["controlled_trial_allocation_reasons"]["EXPOSURE_CAP_EXHAUSTED"] == 1


def test_repeat_refresh_reuses_reservation_and_release_prevents_replay(trial_runtime):
    import sqlite3
    _, _, reservations = trial_runtime()
    frame = pd.DataFrame([final_row()])
    reviews = pd.DataFrame([reviewed()])
    first = apply_trials(frame, reviews, 1000, now=NOW)
    second = apply_trials(frame, reviews, 1000, now=NOW + timedelta(minutes=1))
    assert first.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == 2.5
    assert second.iloc[0]["controlled_trial_contract"]["recommended_bet_amount"] == 2.5
    with sqlite3.connect(reservations) as db:
        ids = [row[0] for row in db.execute("SELECT reservation_id FROM reservations")]
    assert len(ids) == 1
    release_reservation(reservations, ids[0], "synthetic-test-owner", confirmed=True, now=NOW)
    third = apply_trials(frame, reviews, 1000, now=NOW + timedelta(minutes=1))
    assert third.iloc[0].get("controlled_trial_contract") is None


def test_review_budget_and_final_cap_are_distinct_and_backfill_after_holds():
    assert MAX_REVIEW_CANDIDATES_PER_SLATE > MAX_PICKS_PER_SLATE
    rows = []
    review_rows = []
    finals = []
    for index in range(4):
        game = f"synthetic-game-{index}"
        changes = dict(candidate_id=f"synthetic-candidate-{index}", game_id=game,
                       matchup_id=game, home_team_id=f"HOME{index}",
                       away_team_id=f"AWAY{index}")
        rows.append(candidate(**changes))
        review_rows.append(reviewed(**changes, gemini_review_status="HOLD" if index < 2 else "APPROVE"))
        finals.append(final_row(game_id=game, matchup_id=game))
    selected = select_review_candidates(pd.DataFrame(rows), now=NOW)
    assert len(selected) == 4
    out = apply_trials(pd.DataFrame(finals), pd.DataFrame(review_rows), 1000, now=NOW)
    assert out.get("controlled_trial_eligible", pd.Series(False, index=out.index)).fillna(False).sum() == 2
    assert out.iloc[0].get("controlled_trial_contract") is None
    assert out.iloc[1].get("controlled_trial_contract") is None


def test_unreviewed_budget_candidates_get_explicit_status(monkeypatch):
    from app_core.controlled_trial_pipeline import prepare_review_candidates
    import app_core.prediction_evidence as evidence
    import app_core.controlled_trial as trial
    import core.prospective_uncertainty as uncertainty

    monkeypatch.setattr(evidence, "bind_authoritative_candidates", lambda frame: frame)
    monkeypatch.setattr(trial, "attest_candidate_integrity", lambda frame: frame)
    monkeypatch.setattr(uncertainty, "prepare_live", lambda frame: frame)
    rows = [candidate(candidate_id=f"synthetic-{index}", game_id=f"game-{index}",
                      matchup_id=f"game-{index}")
            for index in range(MAX_REVIEW_CANDIDATES_PER_SLATE + 1)]
    diagnostics = {"candidate_authority_df": pd.DataFrame(rows)}
    pool, selected = prepare_review_candidates(diagnostics, now=NOW)
    assert len(selected) == MAX_REVIEW_CANDIDATES_PER_SLATE
    assert diagnostics["controlled_trial_deterministic_eligible_count"] == len(rows)
    assert diagnostics["controlled_trial_not_reviewed_budget_count"] == 1
    assert pool.controlled_trial_review_status.eq("NOT_REVIEWED_BUDGET").sum() == 1


def test_push_semantics_and_price_change_are_fail_closed():
    whole = candidate(spread_line=-2.0, best_pick="Detroit -2.0", selection="Detroit -2.0",
                      calibrated_probability=.55, push_probability=.05,
                      expected_value=.05)
    scored = evaluate_candidates(pd.DataFrame([whole]), now=NOW).iloc[0]
    assert scored.controlled_trial_deterministic_eligible
    assert scored.controlled_trial_estimated_probability == pytest.approx(.55 * .95)
    assert scored.controlled_trial_estimated_expected_value == pytest.approx(.55 * .95 * 2.67 + .05 - 1)
    missing_push = evaluate_candidates(pd.DataFrame([dict(whole, push_probability=None)]), now=NOW).iloc[0]
    assert not missing_push.controlled_trial_deterministic_eligible
    assert "push semantics unavailable" in missing_push.controlled_trial_gate_reason
    inconsistent = evaluate_candidates(pd.DataFrame([dict(whole, loss_probability_unconditional=.1)]), now=NOW).iloc[0]
    assert not inconsistent.controlled_trial_deterministic_eligible
    moved = apply_trials(pd.DataFrame([final_row()]),
                         pd.DataFrame([reviewed(odds_american=180)]), 1000, now=NOW)
    assert moved.iloc[0].get("controlled_trial_contract") is None
    changed_line = reviewed()
    changed_line["spread_line"] = -2.0
    changed_line_result = apply_trials(pd.DataFrame([final_row()]),
                                       pd.DataFrame([changed_line]), 1000, now=NOW)
    assert changed_line_result.iloc[0].get("controlled_trial_contract") is None
    fallback_only = evaluate_candidates(pd.DataFrame([candidate(
        calibrated_probability=None, best_available_probability=.9)]), now=NOW).iloc[0]
    assert not fallback_only.controlled_trial_deterministic_eligible
